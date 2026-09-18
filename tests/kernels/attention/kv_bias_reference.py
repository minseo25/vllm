# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch reference and checks for the paged per-key bias of the Triton kernels.

Shared by the GPU tests in ``test_triton_unified_attention.py`` and by the CPU
run of the same kernels under the Triton interpreter
(``tests/v1/worker/test_kv_bias.py``), so both execute identical checks.
"""

import torch

from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
    triton_zero_kv_bias_slots,
)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import KVQuantMode

NUM_SEGMENTS = 16


def ref_paged_attn_with_bias(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
    bias_cache: torch.Tensor | None = None,
) -> torch.Tensor:
    """Causal paged attention in float32 with ``softmax(scale q k^T + beta)``.

    ``key_cache``/``value_cache`` are ``[num_blocks, block_size, kv_heads,
    head_size]``; ``bias_cache`` is ``[num_blocks, kv_heads, block_size]`` with
    one additive logit per (key slot, kv head), broadcast over the GQA group.
    """
    block_tables = block_tables.cpu()
    _, block_size, num_kv_heads, head_size = key_cache.shape
    outputs = []
    start = 0
    for i, (query_len, kv_len) in enumerate(zip(query_lens, kv_lens)):
        q = query[start : start + query_len].float() * scale
        num_blocks = (kv_len + block_size - 1) // block_size
        blocks = block_tables[i, :num_blocks].long().to(key_cache.device)
        k = key_cache[blocks].reshape(-1, num_kv_heads, head_size)[:kv_len].float()
        v = value_cache[blocks].reshape(-1, num_kv_heads, head_size)[:kv_len].float()
        group = q.shape[1] // num_kv_heads
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k)
        if bias_cache is not None:
            # [blocks, kv_heads, slots] -> [keys, kv_heads] -> [heads, 1, keys]
            beta = bias_cache[blocks].permute(0, 2, 1).reshape(-1, num_kv_heads)
            beta = beta[:kv_len].float().T.repeat_interleave(group, dim=0)
            attn = attn + beta[:, None, :]
        mask = torch.triu(
            torch.ones(query_len, kv_len, device=attn.device),
            diagonal=kv_len - query_len + 1,
        ).bool()
        attn.masked_fill_(mask, float("-inf"))
        out = torch.einsum("hqk,khd->qhd", torch.softmax(attn, dim=-1), v)
        outputs.append(out.to(query.dtype))
        start += query_len
    return torch.cat(outputs, dim=0)


def run_unified_attention(
    query,
    key_cache,
    value_cache,
    query_lens,
    kv_lens,
    block_tables,
    scale,
    seq_threshold_3D,
    bias_cache,
):
    num_query_heads, head_size = query.shape[1], query.shape[2]
    device = query.device
    cu_query_lens = torch.tensor(
        [0] + list(query_lens), dtype=torch.int32, device=device
    ).cumsum(dim=0, dtype=torch.int32)
    seqused_k = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    head_size_padded = 1 << (head_size - 1).bit_length()
    segm_output = torch.empty(
        (seq_threshold_3D, num_query_heads, NUM_SEGMENTS, head_size_padded),
        dtype=torch.float32,
        device=device,
    )
    segm_max = torch.empty(
        (seq_threshold_3D, num_query_heads, NUM_SEGMENTS),
        dtype=torch.float32,
        device=device,
    )
    segm_expsum = torch.empty_like(segm_max)
    output = torch.empty_like(query)
    unified_attention(
        q=query,
        k=key_cache,
        v=value_cache,
        out=output,
        cu_seqlens_q=cu_query_lens,
        max_seqlen_q=max(query_lens),
        seqused_k=seqused_k,
        max_seqlen_k=max(kv_lens),
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=block_tables,
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        seq_threshold_3D=seq_threshold_3D,
        num_par_softmax_segments=NUM_SEGMENTS,
        softmax_segm_output=segm_output,
        softmax_segm_max=segm_max,
        softmax_segm_expsum=segm_expsum,
        kv_quant_mode=KVQuantMode.NONE,
        bias_cache=bias_cache,
    )
    return output


def check_attention_bias(
    device,
    dtype,
    seq_lens,
    num_heads,
    head_size,
    seq_threshold_3D,
    *,
    block_size=16,
    num_blocks=64,
    atol,
    rtol,
    seed=0,
) -> dict:
    """Kernel vs reference with and without bias; zero bias equals no bias."""
    torch.manual_seed(seed)
    query_lens = [s[0] for s in seq_lens]
    kv_lens = [s[1] for s in seq_lens]
    num_query_heads, num_kv_heads = num_heads
    query = torch.randn(
        sum(query_lens), num_query_heads, head_size, dtype=dtype, device=device
    )
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device=device
    )
    value_cache = torch.randn_like(key_cache)
    bias_cache = (
        torch.randn(
            num_blocks, num_kv_heads, block_size, dtype=torch.float32, device=device
        )
        * 1.5
    )
    max_blocks = (max(kv_lens) + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (len(seq_lens), max_blocks), dtype=torch.int32, device=device
    )
    scale = head_size**-0.5
    common = (query, key_cache, value_cache, query_lens, kv_lens, block_tables, scale)
    errors = {}
    for label, bias in (("nobias", None), ("bias", bias_cache)):
        out = run_unified_attention(*common, seq_threshold_3D, bias)
        ref = ref_paged_attn_with_bias(*common, bias)
        torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)
        errors[label] = float((out.float() - ref.float()).abs().max())
    # A biased kernel must change the result (the bias is not silently dropped).
    ref_nobias = ref_paged_attn_with_bias(*common, None)
    ref_bias = ref_paged_attn_with_bias(*common, bias_cache)
    assert not torch.allclose(ref_nobias, ref_bias, atol=atol, rtol=rtol)
    # An all-zero bias buffer reproduces the bias-less kernel bit for bit.
    out_none = run_unified_attention(*common, seq_threshold_3D, None)
    out_zero = run_unified_attention(
        *common, seq_threshold_3D, torch.zeros_like(bias_cache)
    )
    assert torch.equal(out_none, out_zero)
    return errors


def check_cache_write_zeroes_bias(device, dtype, *, block_size=16, seed=0) -> None:
    """Fused and standalone zeroing: only written slots, padding ignored."""
    torch.manual_seed(seed)
    num_blocks, num_heads, head_size = 6, 3, 32
    key = torch.randn(4, num_heads, head_size, dtype=dtype, device=device)
    value = torch.randn_like(key)
    key_cache = torch.zeros(
        num_blocks, block_size, num_heads, head_size, dtype=dtype, device=device
    )
    value_cache = torch.zeros_like(key_cache)
    # An import left non-zero biases on block 2 (slots 0..3) and block 4.
    bias_cache = torch.zeros(
        num_blocks, num_heads, block_size, dtype=torch.float32, device=device
    )
    bias_cache[2, :, :4] = 1.25
    bias_cache[4] = -0.5
    slot_mapping = torch.tensor(
        [2 * block_size + 0, 2 * block_size + 1, -1, 4 * block_size + 7],
        dtype=torch.long,
        device=device,
    )
    one = torch.tensor(1.0, dtype=torch.float32, device=device)
    triton_reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        "auto",
        one,
        one,
        bias_cache=bias_cache,
    )
    assert torch.equal(key_cache[2, 0], key[0]) and torch.equal(
        value_cache[2, 1], value[1]
    )
    assert torch.equal(key_cache[4, 7], key[3])
    assert not bias_cache[2, :, :2].any()  # written slots are reset
    assert torch.all(bias_cache[2, :, 2:4] == 1.25)  # unwritten slots keep the bias
    assert not bias_cache[4, :, 7].any()
    assert torch.all(bias_cache[4, :, :7] == -0.5) and torch.all(
        bias_cache[4, :, 8:] == -0.5
    )
    assert not bias_cache[[0, 1, 3, 5]].any()
    # Without a bias buffer the write is the upstream one and touches no bias.
    stale = bias_cache.clone()
    triton_reshape_and_cache_flash(
        key, value, key_cache, value_cache, slot_mapping, "auto", one, one
    )
    assert torch.equal(bias_cache, stale)
    # The standalone kernel (fused-rope / per-token-head paths) does the same.
    triton_zero_kv_bias_slots(
        bias_cache,
        torch.tensor(
            [2 * block_size + 2, 2 * block_size + 3, -1],
            dtype=torch.long,
            device=device,
        ),
    )
    assert not bias_cache[2].any()
    assert torch.all(bias_cache[4, :, :7] == -0.5)


def check_block_reuse_reads_zero_bias(device, dtype, *, block_size=16, seed=0) -> None:
    """Import a bias into a block, then let a new request's write reuse it: the
    reused slots attend with bias 0 (kernel output equals the no-bias output)."""
    torch.manual_seed(seed)
    num_blocks, num_kv_heads, num_heads, head_size = 8, 2, 4, 32
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device=device
    )
    value_cache = torch.randn_like(key_cache)
    bias_cache = torch.zeros(
        num_blocks, num_kv_heads, block_size, dtype=torch.float32, device=device
    )
    block_tables = torch.tensor([[3, 5]], dtype=torch.int32, device=device)
    kv_len = block_size + 5
    # "Import": non-zero bias on every slot the request will use.
    bias_cache[3] = 2.0
    bias_cache[5, :, :5] = -3.0
    query = torch.randn(1, num_heads, head_size, dtype=dtype, device=device)
    scale = head_size**-0.5
    common = (query, key_cache, value_cache, [1], [kv_len], block_tables, scale)
    biased = run_unified_attention(*common, 0, bias_cache)
    assert torch.allclose(
        biased, ref_paged_attn_with_bias(*common, bias_cache), atol=5e-2, rtol=5e-2
    )
    # A later request writes all of those slots through the normal KV path.
    slots = torch.cat(
        [
            torch.arange(3 * block_size, 4 * block_size),
            torch.arange(5 * block_size, 5 * block_size + 5),
        ]
    ).to(device)
    key = torch.randn(kv_len, num_kv_heads, head_size, dtype=dtype, device=device)
    value = torch.randn_like(key)
    one = torch.tensor(1.0, dtype=torch.float32, device=device)
    triton_reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slots,
        "auto",
        one,
        one,
        bias_cache=bias_cache,
    )
    assert not bias_cache.any()
    reused = run_unified_attention(*common, 0, bias_cache)
    plain = run_unified_attention(*common, 0, None)
    assert torch.equal(reused, plain)


def run_all_checks(device, dtype, *, atol, rtol) -> dict:
    errors = {
        "prefill_decode_2d": check_attention_bias(
            device,
            dtype,
            [(1, 40), (5, 18), (33, 70)],
            (4, 2),
            32,
            0,
            atol=atol,
            rtol=rtol,
        ),
        "decode_3d": check_attention_bias(
            device,
            dtype,
            [(1, 40), (1, 17), (1, 70)],
            (4, 2),
            32,
            8,
            atol=atol,
            rtol=rtol,
        ),
    }
    check_cache_write_zeroes_bias(device, dtype)
    check_block_reuse_reads_zero_bias(device, dtype)
    return errors
