# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged per-key attention-logit bias buffers for native compaction.

Faithful attention matching (arXiv 2602.16284) stores
``softmax(q C_k^T + beta) C_v``: beside the pages of every bias-capable
full-attention layer lives a float32 ``bias_cache`` of shape
``[num_blocks, num_kv_heads, block_size]`` that the Triton unified attention
kernel adds to the scaled logits of each key, in prefill and decode alike.

Invariants:

* Every slot written by the normal KV write (prefill and decode) has its bias
  reset to zero in the same kernel launch, so a page reused by a later request
  never exposes a bias imported for an earlier one. Only the compaction store
  writes non-zero rows, at import, into exactly the slots it writes K/V to.
* The buffers are allocated right after the KV cache, sized from each layer's
  cache shape, and budgeted per block when the KV pool is sized
  (``AttentionSpec.kv_bias`` -> ``kv_cache_utils.kv_bias_bytes_per_block``),
  so pages plus bias buffers fit the ``gpu_memory_utilization`` budget.
* Whether a layer gets a buffer follows only the resolved attention backend
  (``AttentionBackend.supports_kv_bias``, today ``TRITON_ATTN``) and its KV
  cache dtype; there is no environment switch. Running without buffers means
  selecting another backend (``attention_backend="FLASH_ATTN"``).
"""

from __future__ import annotations

from typing import Any

import torch

from vllm.v1.attention.backend import AttentionType
from vllm.v1.kv_cache_interface import KVQuantMode, get_kv_quant_mode

KV_BIAS_DTYPE = torch.float32
SUPPORTED_KV_CACHE_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
UNSUPPORTED_MESSAGE = (
    "attention backend cannot apply per-key bias: the layer has no paged bias "
    "buffer (needs the Triton attention backend, attention_backend='TRITON_ATTN', "
    "and an unquantized fp16/bf16/fp32 KV cache)"
)


def backend_supports_kv_bias(backend: Any) -> bool:
    supports = getattr(backend, "supports_kv_bias", None)
    return callable(supports) and bool(supports())


def layer_supports_kv_bias(layer: Any) -> bool:
    """Decoder attention on a bias-capable backend with an unquantized float cache."""
    get_backend = getattr(layer, "get_attn_backend", None)
    if not callable(get_backend) or not backend_supports_kv_bias(get_backend()):
        return False
    if getattr(layer, "attn_type", AttentionType.DECODER) != AttentionType.DECODER:
        return False
    cache_dtype = getattr(layer, "kv_cache_torch_dtype", None)
    quant_mode = get_kv_quant_mode(getattr(layer, "kv_cache_dtype", "auto"))
    return cache_dtype in SUPPORTED_KV_CACHE_DTYPES and quant_mode == KVQuantMode.NONE


def kv_bias_shape(kv_cache: torch.Tensor) -> tuple[int, int, int]:
    """``(num_blocks, num_kv_heads, block_size)`` of a logical ``(B, H, N, C)`` view.

    The cache view bound to an attention layer is logically blocks-first,
    head-major, whatever its physical strides (NHD or HND).
    """
    if not isinstance(kv_cache, torch.Tensor) or kv_cache.ndim != 4:
        raise ValueError("KV bias needs a logical [blocks, heads, slots, 2*hs] cache")
    return int(kv_cache.shape[0]), int(kv_cache.shape[1]), int(kv_cache.shape[2])


def bias_cache_of(layer: Any) -> torch.Tensor | None:
    """The layer's bias buffer, validated against its cache; None when absent."""
    bias = getattr(layer, "bias_cache", None)
    if bias is None:
        return None
    cache = layer.kv_cache
    if (
        not isinstance(bias, torch.Tensor)
        or bias.dtype != KV_BIAS_DTYPE
        or tuple(bias.shape) != kv_bias_shape(cache)
        or bias.device != cache.device
    ):
        raise ValueError(
            "bias_cache must be float32 [num_blocks, num_kv_heads, block_size] on "
            "the cache device"
        )
    return bias


def allocate_kv_bias_caches(
    kv_caches: dict[str, torch.Tensor],
    forward_context: dict[str, Any],
    shared_kv_cache_layers: dict[str, str],
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    """Allocate zeroed bias buffers for every bias-capable attention layer.

    Sized from each layer's bound KV cache view (``shape[:3]`` of the logical
    ``(B, H, N, C)`` cache), so the total is ``num_blocks *
    kv_bias_bytes_per_block`` exactly as budgeted. Layers that share another
    layer's KV cache share its bias buffer. Returns ``{layer_name: buffer}``.
    """
    caches: dict[str, torch.Tensor] = {}
    for name, kv_cache in kv_caches.items():
        if name in shared_kv_cache_layers:
            continue
        layer = forward_context.get(name)
        if layer is None or not layer_supports_kv_bias(layer):
            continue
        if not isinstance(kv_cache, torch.Tensor) or kv_cache.ndim != 4:
            continue
        bias = torch.zeros(kv_bias_shape(kv_cache), dtype=KV_BIAS_DTYPE, device=device)
        layer.bias_cache = bias
        caches[name] = bias
    for name, target in shared_kv_cache_layers.items():
        if target in caches and name in forward_context:
            forward_context[name].bias_cache = caches[target]
            caches[name] = caches[target]
    return caches


def zero_kv_bias_slots(bias_cache: torch.Tensor, slot_mapping: torch.Tensor) -> None:
    """Zero the bias of every slot in ``slot_mapping`` (negative = padding).

    On CUDA this is the graph-capturable Triton kernel; elsewhere (CPU tests)
    the same semantics in torch.
    """
    if bias_cache.device.type == "cuda":
        from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
            triton_zero_kv_bias_slots,
        )

        triton_zero_kv_bias_slots(bias_cache, slot_mapping)
        return
    slots = slot_mapping.to(dtype=torch.long)
    slots = slots[slots >= 0]
    if slots.numel() == 0:
        return
    block_size = bias_cache.shape[2]
    bias_cache[slots // block_size, :, slots % block_size] = 0.0
