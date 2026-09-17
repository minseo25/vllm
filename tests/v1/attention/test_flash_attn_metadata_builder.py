# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU metadata checks for FA3 graph workspace eligibility, without kernels."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends import flash_attn
from vllm.v1.kv_cache_interface import FullAttentionSpec


def _build(monkeypatch, mode, query_lens, num_spec_tokens=0, capture_limit=16384):
    """Exercise the real builder while replacing only the FA3 scheduler kernel."""
    calls = []

    def schedule(**kwargs):
        calls.append(kwargs)
        return torch.zeros(
            1 + ((kwargs["batch_size"] + 3) // 4) * 16, dtype=torch.int32
        )

    monkeypatch.setattr(flash_attn, "get_flash_attn_version", lambda: 3)
    monkeypatch.setattr(flash_attn, "get_scheduler_metadata", schedule, raising=False)
    monkeypatch.setattr(flash_attn, "_get_sliding_window_configs", lambda _: {None})
    model_config = SimpleNamespace(
        get_num_attention_heads=lambda _: 16,
        get_num_kv_heads=lambda _: 4,
        get_head_size=lambda: 256,
        rswa_window=None,
    )
    config = SimpleNamespace(
        model_config=model_config,
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=1),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        compilation_config=SimpleNamespace(
            cudagraph_mode=mode, max_cudagraph_capture_size=capture_limit
        ),
        attention_config=SimpleNamespace(flash_attn_max_num_splits_for_cuda_graph=32),
        scheduler_config=SimpleNamespace(max_num_seqs=32),
        num_speculative_tokens=num_spec_tokens,
    )
    builder = flash_attn.FlashAttentionMetadataBuilder(
        FullAttentionSpec(
            block_size=16, num_kv_heads=4, head_size=256, dtype=torch.bfloat16
        ),
        ["layer.0"],
        config,
        torch.device("cpu"),
    )
    starts = torch.tensor([0, *torch.tensor(query_lens).cumsum(0)], dtype=torch.int32)
    common = CommonAttentionMetadata(
        query_start_loc=starts,
        query_start_loc_cpu=starts,
        seq_lens=torch.full((len(query_lens),), 32768, dtype=torch.int32),
        num_reqs=len(query_lens),
        num_actual_tokens=sum(query_lens),
        max_query_len=max(query_lens),
        max_seq_len=32768,
        block_table_tensor=torch.zeros((len(query_lens), 1), dtype=torch.int32),
        slot_mapping=torch.zeros(sum(query_lens), dtype=torch.int64),
    )
    return builder.build(0, common), calls


@pytest.mark.parametrize(
    "mode", [CUDAGraphMode.FULL_AND_PIECEWISE, CUDAGraphMode.FULL_DECODE_ONLY]
)
@pytest.mark.parametrize("query_lens", [[16384], [303] * 32, [1, 1, 1024]])
def test_separate_prefill_uses_heuristic_splits(monkeypatch, mode, query_lens):
    """A large piecewise/eager prefill must not allocate 32-way accumulators."""
    monkeypatch.setattr(flash_attn.envs, "VLLM_BATCH_INVARIANT", False)
    metadata, calls = _build(monkeypatch, mode, query_lens)
    assert metadata.max_num_splits == 0
    assert calls[0]["num_splits"] == 0


@pytest.mark.parametrize(
    "mode", [CUDAGraphMode.FULL_AND_PIECEWISE, CUDAGraphMode.FULL_DECODE_ONLY]
)
@pytest.mark.parametrize(
    "num_spec_tokens,query_lens",
    [(0, [1] * 32), (0, [1, 1, 1, 0]), (3, [4] * 32), (3, [4, 4, 4, 0])],
)
def test_separate_decode_keeps_graph_split_bound(
    monkeypatch, mode, num_spec_tokens, query_lens
):
    """Ordinary/speculative decode, including padded rows, keeps capture shapes."""
    monkeypatch.setattr(flash_attn.envs, "VLLM_BATCH_INVARIANT", False)
    metadata, calls = _build(monkeypatch, mode, query_lens, num_spec_tokens)
    assert metadata.max_num_splits == 32
    assert calls[0]["num_splits"] == 32


@pytest.mark.parametrize(
    "mode,query_lens,capture_limit,expected",
    [
        (CUDAGraphMode.FULL, [16384], 16384, 32),
        (CUDAGraphMode.NONE, [1], 16384, 0),
        (CUDAGraphMode.PIECEWISE, [16384], 16384, 0),
        (CUDAGraphMode.FULL_AND_PIECEWISE, [1] * 32, 16, 0),
        (CUDAGraphMode.FULL_AND_PIECEWISE, [1], None, 0),
    ],
)
def test_other_graph_modes_and_capture_limit_are_unchanged(
    monkeypatch, mode, query_lens, capture_limit, expected
):
    monkeypatch.setattr(flash_attn.envs, "VLLM_BATCH_INVARIANT", False)
    metadata, calls = _build(monkeypatch, mode, query_lens, capture_limit=capture_limit)
    assert metadata.max_num_splits == expected
    assert calls[0]["num_splits"] == expected


def test_query_longer_than_speculative_decode_uses_heuristic(monkeypatch):
    monkeypatch.setattr(flash_attn.envs, "VLLM_BATCH_INVARIANT", False)
    metadata, calls = _build(
        monkeypatch, CUDAGraphMode.FULL_AND_PIECEWISE, [5], num_spec_tokens=3
    )
    assert metadata.max_num_splits == 0
    assert calls[0]["num_splits"] == 0


@pytest.mark.parametrize("query_lens", [[1] * 32, [16384]])
def test_batch_invariance_still_forces_one_split(monkeypatch, query_lens):
    monkeypatch.setattr(flash_attn.envs, "VLLM_BATCH_INVARIANT", True)
    metadata, calls = _build(monkeypatch, CUDAGraphMode.FULL_AND_PIECEWISE, query_lens)
    assert metadata.max_num_splits == 1
    assert calls == []
