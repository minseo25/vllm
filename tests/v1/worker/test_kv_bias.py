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

from vllm.config import ModelConfig, VllmConfig
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
from vllm.v1.core import kv_cache_utils
from vllm.v1.core.kv_cache_utils import (
    _max_memory_usage_bytes_from_groups,
    _promote_local_kv_cache_specs,
    get_kv_cache_config_from_groups,
    kv_bias_bytes_per_block,
    kv_memory_after_bias_reservation,
    max_memory_usage_bytes,
)
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
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


def bias_config(flagged_layers, unflagged_layers=(), mamba_layers=()):
    """A KV cache config whose FA group spec budgets the bias (or not)."""
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=4, dtype=torch.bfloat16
    )
    groups = []
    if flagged_layers:
        groups.append(
            KVCacheGroupSpec(list(flagged_layers), replace(spec, kv_bias=True))
        )
    if unflagged_layers:
        groups.append(KVCacheGroupSpec(list(unflagged_layers), spec))
    if mamba_layers:
        groups.append(
            KVCacheGroupSpec(
                list(mamba_layers),
                MambaSpec(block_size=16, shapes=((2, 8),), dtypes=(torch.int8,)),
            )
        )
    return NS(kv_cache_groups=groups)


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


def test_allocation_follows_the_config_specs_cache_geometry_and_kv_sharing():
    layers = {
        "fa0": fake_layer(),
        "fa1": fake_layer(shape=(6, 2, 16, 8)),
        "flash": fake_layer(backend=FlashAttentionBackend),
        "shared": fake_layer(),
        "mamba": NS(kv_cache=(torch.zeros(6, 3),)),
    }
    kv_caches = {name: layer.kv_cache for name, layer in layers.items()}
    kv_caches["mamba"] = torch.zeros(6, 1, 1, 64, dtype=torch.int8)
    # KV-sharing layers join their target's group (as the runner does).
    config = bias_config(["fa0", "fa1", "shared"], ["flash"], ["mamba"])
    assert kv_bias.kv_bias_layers(config) == {"fa0", "fa1", "shared"}
    caches = kv_bias.allocate_kv_bias_caches(
        kv_caches, layers, {"shared": "fa0"}, torch.device("cpu"), config
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


def test_config_specs_are_the_single_source_of_truth_for_allocation():
    """Budget (spec flag) and allocation (backend) must not diverge: a flagged
    layer on a bias-less backend is an error, an unflagged bias-capable layer
    gets no buffer, and a Triton layer is allocated only through its spec."""
    layers = {"fa0": fake_layer(), "flash": fake_layer(backend=FlashAttentionBackend)}
    kv_caches = {name: layer.kv_cache for name, layer in layers.items()}
    # Nothing flagged: no buffer even though fa0's backend supports the bias.
    caches = kv_bias.allocate_kv_bias_caches(
        kv_caches, layers, {}, torch.device("cpu"), bias_config([], ["fa0", "flash"])
    )
    assert caches == {} and layers["fa0"].bias_cache is None
    # Flagged FLASH layer: budgeted but not applicable -> refused.
    with pytest.raises(RuntimeError, match="cannot apply one"):
        kv_bias.allocate_kv_bias_caches(
            kv_caches, layers, {}, torch.device("cpu"), bias_config(["flash"])
        )
    assert layers["flash"].bias_cache is None
    # UniformTypeKVCacheSpecs groups flag per layer.
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=4, dtype=torch.bfloat16
    )
    uniform = NS(
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["fa0", "flash"],
                UniformTypeKVCacheSpecs(
                    block_size=16,
                    kv_cache_specs={"fa0": replace(spec, kv_bias=True), "flash": spec},
                ),
            )
        ]
    )
    assert kv_bias.kv_bias_layers(uniform) == {"fa0"}
    caches = kv_bias.allocate_kv_bias_caches(
        kv_caches, layers, {}, torch.device("cpu"), uniform
    )
    assert set(caches) == {"fa0"} and layers["flash"].bias_cache is None


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
    assert kv_memory_after_bias_reservation(0, kv_per_block, bias) == 0
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
    zero bias bit-identical to no bias, fused/standalone slot zeroing, a block
    reused after an import attending with bias 0, and the ``kv_bias_mask``
    control (beta = -20 on a token set attends like its eviction, a head-0 mask
    differs from both)."""
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


# ------------------------------------------------- pool sizing per layout (M3)


def hybrid_groups(bias=True):
    """Mamba + full-attention groups as the hybrid allocator builds them: equal
    (padded) page sizes, manager block 528 (33 kernel blocks of 16)."""
    fa = FullAttentionSpec(
        block_size=528, num_kv_heads=2, head_size=16, dtype=torch.bfloat16
    )
    mamba = MambaSpec(block_size=528, shapes=((2, 3000),), dtypes=(torch.float32,))
    page = max(fa.page_size_bytes, mamba.page_size_bytes)
    fa = replace(fa, page_size_padded=page, kv_bias=bias)
    mamba = replace(mamba, page_size_padded=page)
    return [
        KVCacheGroupSpec(["fa0", "fa1", "fa2"], fa),
        KVCacheGroupSpec(["m0", "m1"], mamba),
    ]


def uniform_type_group(bias=True):
    """One UniformTypeKVCacheSpecs group over full-attention layers of two
    hidden sizes plus a Mamba layer (the layout code treats per-layer specs
    generically; a Mamba layer contributes no bias)."""
    small = FullAttentionSpec(
        block_size=528,
        num_kv_heads=2,
        head_size=16,
        dtype=torch.bfloat16,
        kv_bias=bias,
    )
    wide = replace(small, num_kv_heads=4)
    mamba = MambaSpec(block_size=528, shapes=((2, 3000),), dtypes=(torch.float32,))
    specs = {"fa0": small, "fa1": wide, "m0": mamba}
    return [
        KVCacheGroupSpec(
            list(specs), UniformTypeKVCacheSpecs(block_size=528, kv_cache_specs=specs)
        )
    ]


def caches_like_the_runner(config, kernel_block_size=16):
    """Per-layer cache views shaped as ``initialize_kv_cache_tensors`` produces
    for the Triton backend: ``(num_blocks * (528 // 16), Hkv, 16, 2 * hs)``."""
    layers, kv_caches = {}, {}
    for group in config.kv_cache_groups:
        for name in group.layer_names:
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = spec.kv_cache_specs[name]
            if not isinstance(spec, FullAttentionSpec):
                layers[name] = NS(kv_cache=(torch.zeros(config.num_blocks, 1),))
                kv_caches[name] = torch.zeros(
                    config.num_blocks, 1, 1, 8, dtype=torch.int8
                )
                continue
            kernel_blocks = config.num_blocks * (spec.block_size // kernel_block_size)
            shape = (
                kernel_blocks,
                spec.num_kv_heads,
                kernel_block_size,
                2 * spec.head_size,
            )
            layers[name] = fake_layer(shape=shape)
            kv_caches[name] = layers[name].kv_cache
    return layers, kv_caches


@pytest.mark.parametrize("layout", ["general", "uniform_type", "packed"])
def test_pool_sizing_and_allocation_agree_per_layout(layout, monkeypatch):
    """``get_kv_cache_config_from_groups`` leaves room for the bias buffers
    (pages + num_blocks * bias bytes <= available, and one more block would not
    fit), the buffers the runner allocates from that config total exactly
    ``num_blocks * kv_bias_bytes_per_block`` although they are shaped in kernel
    blocks (16) rather than manager blocks (528), and the max-length memory
    check counts them."""
    vllm_config = VllmConfig(model_config=ModelConfig(max_model_len=16))
    groups = uniform_type_group() if layout == "uniform_type" else hybrid_groups()
    if layout == "packed":
        monkeypatch.setattr(
            kv_cache_utils, "_use_packed_kv_cache_config", lambda *a, **k: True
        )
    bias_per_block = kv_bias_bytes_per_block(groups)
    assert bias_per_block > 0
    kv_per_block = {
        "general": groups[0].kv_cache_spec.page_size_bytes * 3,  # group size 3
        "uniform_type": groups[0].kv_cache_spec.page_size_bytes,
        "packed": 3 * groups[0].kv_cache_spec.page_size_bytes,  # widest group
    }[layout]
    available = 7 * (kv_per_block + bias_per_block) + kv_per_block // 2
    config = get_kv_cache_config_from_groups(vllm_config, groups, available)
    assert config.num_blocks == 7
    if layout == "packed":
        # Packed tensors alias one backing allocation of block_stride * blocks.
        kv_bytes = max(t.size for t in config.kv_cache_tensors)
        assert kv_bytes == kv_per_block * config.num_blocks
    else:
        kv_bytes = sum(t.size for t in config.kv_cache_tensors)
    assert kv_bytes + config.num_blocks * bias_per_block <= available
    assert (config.num_blocks + 1) * (kv_per_block + bias_per_block) > available
    # Without the bias flag the whole budget goes to pages.
    plain = (
        uniform_type_group(False) if layout == "uniform_type" else hybrid_groups(False)
    )
    plain_blocks = get_kv_cache_config_from_groups(
        vllm_config, plain, available
    ).num_blocks
    assert plain_blocks == available // kv_per_block >= config.num_blocks
    # The runner's allocation from this config equals the budgeted bytes.
    layers, kv_caches = caches_like_the_runner(config)
    caches = kv_bias.allocate_kv_bias_caches(
        kv_caches, layers, {}, torch.device("cpu"), config
    )
    assert set(caches) == {n for n in layers if n.startswith("fa")}
    assert (
        sum(t.numel() * t.element_size() for t in caches.values())
        == config.num_blocks * bias_per_block
    )
    for name, bias in caches.items():
        assert bias.shape == layers[name].kv_cache.shape[:3]
        assert bias.dtype == torch.float32
    # Max-length memory checks charge one bias block per page of the request.
    pages = 1  # max_model_len 16 fits one 528-token block
    if layout == "uniform_type":
        expected = sum(
            spec.max_memory_usage_bytes(vllm_config)
            + pages * getattr(spec, "kv_bias_bytes_per_block", 0)
            for spec in groups[0].kv_cache_spec.kv_cache_specs.values()
        )
        assert _max_memory_usage_bytes_from_groups(vllm_config, groups) == expected
        assert max_memory_usage_bytes(vllm_config, [groups[0].kv_cache_spec]) == (
            expected
        )
    elif layout == "general":
        blocks_needed = 2  # one page per group for a 16-token request
        assert _max_memory_usage_bytes_from_groups(vllm_config, groups) == (
            blocks_needed * (kv_per_block + bias_per_block)
        )
        specs = [g.kv_cache_spec for g in groups]
        assert (
            max_memory_usage_bytes(vllm_config, specs)
            == sum(s.max_memory_usage_bytes(vllm_config) for s in specs)
            + pages * groups[0].kv_cache_spec.kv_bias_bytes_per_block
        )


def test_promoted_local_attention_specs_keep_the_bias_flag():
    """Hybrid-manager-off promotion rebuilds sliding-window / chunked specs as
    full attention; the bias budget flag (and block-stride indexing) survive."""
    full = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=16,
        dtype=torch.bfloat16,
        kv_bias=True,
    )
    swa = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=16,
        dtype=torch.bfloat16,
        sliding_window=8,
        kv_bias=True,
        indexes_kv_by_block_stride=True,
    )
    chunked = ChunkedLocalAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=16,
        dtype=torch.bfloat16,
        attention_chunk_size=4,
        kv_bias=True,
    )
    promoted = _promote_local_kv_cache_specs({"f": full, "s": swa, "c": chunked})
    assert all(isinstance(spec, FullAttentionSpec) for spec in promoted.values())
    assert all(spec.kv_bias for spec in promoted.values())
    assert promoted["s"].indexes_kv_by_block_stride is True
    assert promoted["s"].sliding_window == 8
    assert promoted["c"].attention_chunk_size == 4
    assert sum(spec.kv_bias_bytes_per_block for spec in promoted.values()) == (
        3 * 16 * 2 * 4
    )
    # Equal promoted specs merge into one group that keeps the flag.
    merged = FullAttentionSpec.merge([promoted["f"], promoted["c"]])
    assert merged.kv_bias is True
    assert kv_bias_bytes_per_block([KVCacheGroupSpec(["f", "c"], merged)]) == (
        2 * 16 * 2 * 4
    )
    unflagged = _promote_local_kv_cache_specs(
        {"f": replace(full, kv_bias=False), "s": replace(swa, kv_bias=False)}
    )
    assert not any(spec.kv_bias for spec in unflagged.values())
