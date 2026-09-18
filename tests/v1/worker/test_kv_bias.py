# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the paged per-key attention bias (``vllm.v1.worker.kv_bias``).

Capability gating, allocation geometry, KV-pool accounting, slot zeroing and
kernel entry points; the kernels themselves run on the CPU through the Triton
interpreter in a subprocess (the GPU tests live in
``tests/kernels/attention/test_triton_unified_attention.py``).
"""

import inspect
import os
import subprocess
import sys
import textwrap
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from vllm.v1.attention.backend import AttentionBackend, AttentionType
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend
from vllm.v1.attention.backends.triton_attn_diffkv import (
    TritonAttentionDiffKVBackend,
)
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.core.kv_cache_utils import (
    kv_bias_bytes_per_block,
    kv_memory_after_bias_reservation,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker import kv_bias

REPO_ROOT = Path(__file__).resolve().parents[3]


def fake_layer(
    backend=TritonAttentionBackend,
    dtype=torch.bfloat16,
    kv_cache_dtype="auto",
    attn_type=AttentionType.DECODER,
    shape=(6, 2, 16, 8),
):
    return NS(
        get_attn_backend=lambda: backend,
        kv_cache_torch_dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        attn_type=attn_type,
        kv_cache=torch.zeros(shape, dtype=dtype),
        bias_cache=None,
    )


def test_only_the_triton_backend_with_an_unquantized_float_cache_gets_a_buffer():
    assert TritonAttentionBackend.get_name() == "TRITON_ATTN"
    assert TritonAttentionBackend.supports_kv_bias() is True
    assert AttentionBackend.supports_kv_bias() is False
    assert FlashAttentionBackend.supports_kv_bias() is False
    assert TritonAttentionDiffKVBackend.supports_kv_bias() is False
    assert kv_bias.layer_supports_kv_bias(fake_layer())
    assert kv_bias.layer_supports_kv_bias(fake_layer(dtype=torch.float16))
    assert not kv_bias.layer_supports_kv_bias(fake_layer(backend=FlashAttentionBackend))
    assert not kv_bias.layer_supports_kv_bias(
        fake_layer(dtype=torch.uint8, kv_cache_dtype="fp8")
    )
    assert not kv_bias.layer_supports_kv_bias(
        fake_layer(dtype=torch.int8, kv_cache_dtype="int8_per_token_head")
    )
    assert not kv_bias.layer_supports_kv_bias(
        fake_layer(attn_type=AttentionType.ENCODER_ONLY)
    )
    assert not kv_bias.layer_supports_kv_bias(NS(kv_cache=torch.zeros(2)))


def test_allocation_follows_the_cache_geometry_and_kv_sharing():
    layers = {
        "fa0": fake_layer(),
        "fa1": fake_layer(shape=(6, 2, 16, 8)),
        "flash": fake_layer(backend=FlashAttentionBackend),
        "shared": fake_layer(),
        "mamba": NS(kv_cache=(torch.zeros(6, 3),)),
    }
    kv_caches = {name: layer.kv_cache for name, layer in layers.items()}
    kv_caches["mamba"] = torch.zeros(6, 1, 1, 64, dtype=torch.int8)
    caches = kv_bias.allocate_kv_bias_caches(
        kv_caches, layers, {"shared": "fa0"}, torch.device("cpu")
    )
    assert set(caches) == {"fa0", "fa1", "shared"}
    for name in ("fa0", "fa1"):
        assert caches[name].shape == (6, 2, 16)  # blocks, kv heads, slots
        assert caches[name].dtype == torch.float32 and not caches[name].any()
        assert layers[name].bias_cache is caches[name]
        assert kv_bias.bias_cache_of(layers[name]) is caches[name]
    assert caches["shared"] is caches["fa0"]
    assert layers["shared"].bias_cache is caches["fa0"]
    assert layers["flash"].bias_cache is None
    assert kv_bias.bias_cache_of(layers["flash"]) is None
    total = sum(
        t.numel() * t.element_size() for n, t in caches.items() if n != "shared"
    )
    assert total == 2 * 6 * 2 * 16 * 4
    layers["fa0"].bias_cache = torch.zeros(6, 2, 15)
    with pytest.raises(ValueError, match="bias_cache must be float32"):
        kv_bias.bias_cache_of(layers["fa0"])
    layers["fa0"].bias_cache = torch.zeros(6, 2, 16, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="bias_cache must be float32"):
        kv_bias.bias_cache_of(layers["fa0"])


def test_kv_pool_sizing_budgets_the_bias_beside_every_page():
    spec = FullAttentionSpec(
        block_size=528, num_kv_heads=4, head_size=256, dtype=torch.bfloat16
    )
    biased = replace(spec, kv_bias=True)
    assert spec.kv_bias_bytes_per_block == 0
    assert biased.kv_bias_bytes_per_block == 528 * 4 * 4
    assert biased.page_size_bytes == spec.page_size_bytes  # page layout unchanged
    assert FullAttentionSpec.merge([biased, biased]).kv_bias is True
    assert FullAttentionSpec.merge([spec, spec]).kv_bias is False
    groups = [
        KVCacheGroupSpec(layer_names=[f"fa{i}" for i in range(8)], kv_cache_spec=biased)
    ]
    assert kv_bias_bytes_per_block(groups) == 8 * 528 * 4 * 4
    assert (
        kv_bias_bytes_per_block(
            [KVCacheGroupSpec(layer_names=["fa0"], kv_cache_spec=spec)]
        )
        == 0
    )
    available = 10 * 2**30
    kv_per_block = spec.page_size_bytes * 8  # general case: page * group size
    bias = kv_bias_bytes_per_block(groups)
    reserved = kv_memory_after_bias_reservation(available, kv_per_block, bias)
    num_blocks = reserved // kv_per_block
    assert num_blocks == available // (kv_per_block + bias)
    assert num_blocks * (kv_per_block + bias) <= available
    assert (num_blocks + 1) * (kv_per_block + bias) > available
    assert kv_memory_after_bias_reservation(available, kv_per_block, 0) == available
    uniform = UniformTypeKVCacheSpecs(
        block_size=528, kv_cache_specs={"fa0": biased, "fa1": spec}
    )
    assert (
        kv_bias_bytes_per_block(
            [KVCacheGroupSpec(layer_names=["fa0", "fa1"], kv_cache_spec=uniform)]
        )
        == 528 * 4 * 4
    )


def test_zeroing_fallback_touches_only_written_slots_and_skips_padding():
    bias = torch.full((4, 3, 8), 2.5)
    kv_bias.zero_kv_bias_slots(bias, torch.tensor([0, 9, -1, 31]))
    assert not bias[0, :, 0].any() and not bias[1, :, 1].any()
    assert not bias[3, :, 7].any()
    assert int(bias.eq(2.5).sum()) == 4 * 3 * 8 - 3 * 3
    kv_bias.zero_kv_bias_slots(bias, torch.tensor([-1, -1]))
    kv_bias.zero_kv_bias_slots(bias, torch.tensor([], dtype=torch.long))
    assert int(bias.eq(2.5).sum()) == 4 * 3 * 8 - 3 * 3


def test_kernel_entry_points_default_to_the_upstream_no_bias_path():
    assert inspect.signature(unified_attention).parameters["bias_cache"].default is None
    assert (
        inspect.signature(triton_reshape_and_cache_flash)
        .parameters["bias_cache"]
        .default
        is None
    )


INTERPRETER_SCRIPT = textwrap.dedent(
    """
    import json, torch
    # The GPU is hidden: keep the CUDA-platform wrappers off device queries.
    torch.cuda.get_device_capability = lambda *a, **k: (9, 0)
    from vllm.platforms import current_platform
    current_platform.__class__.is_device_capability_family = classmethod(
        lambda cls, *a, **k: False
    )
    from tests.kernels.attention.kv_bias_reference import run_all_checks
    errors = run_all_checks(torch.device("cpu"), torch.float32, atol=1e-4, rtol=1e-4)
    print("KV_BIAS_INTERPRETER_OK", json.dumps(errors))
    """
)


def test_kernels_under_the_triton_interpreter_match_the_torch_reference():
    """The real Triton kernels executed on the CPU (``TRITON_INTERPRET=1``, GPU
    hidden): bias on/off vs the reference for 2D prefill+decode and 3D decode,
    zero bias bit-identical to no bias, fused/standalone slot zeroing, and a
    block reused after an import attending with bias 0."""
    env = {
        **os.environ,
        "TRITON_INTERPRET": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONPATH": os.pathsep.join(
            [str(REPO_ROOT), *filter(None, [os.environ.get("PYTHONPATH")])]
        ),
    }
    result = subprocess.run(
        [sys.executable, "-c", INTERPRETER_SCRIPT],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=1500,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-6000:]
    assert "KV_BIAS_INTERPRETER_OK" in result.stdout, result.stdout[-2000:]
