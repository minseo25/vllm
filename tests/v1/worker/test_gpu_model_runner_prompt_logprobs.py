# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regressions for bounded V1 prompt-logprob vocabulary projection.

Exercise the real runner method and sampler with an identity output head.
Compare complete results against a monolithic sampler reference, including
next-token alignment, packed request offsets and delayed prefill completion.
Only pinned allocation and the rank reduction's compiler wrapper are disabled;
no model, accelerator, or large-vocabulary allocation is needed.
"""

import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import vllm.utils.torch_utils as torch_utils
import vllm.v1.sample.sampler as sampler_module
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.sample.sampler import Sampler
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

pytestmark = pytest.mark.cpu_test
VOCAB = 19
MODES = ("raw_logprobs", "processed_logprobs", "raw_logits", "processed_logits")


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch_utils, "PIN_MEMORY", False)
    monkeypatch.setattr(
        sampler_module,
        "batched_count_greater_than",
        sampler_module.batched_count_greater_than.__wrapped__,
    )


def hidden_rows(rows):
    values = torch.arange(rows * VOCAB).reshape(rows, VOCAB)
    return ((values * 13 + 7) % 97 / 10 - 5).to(torch.bfloat16)


def request(length):
    return SimpleNamespace(
        prompt_token_ids=[(i * 7 + 3) % VOCAB for i in range(length)],
        num_computed_tokens=0,
        in_progress_prompt_logprobs_cpu=None,
    )


def runner(requests, counts, *, mode="raw_logprobs", order=None):
    order = list(requests) if order is None else order
    calls = []
    previous_logits = []

    def project(hidden):
        assert 0 < len(hidden) <= 1024, "Unbounded vocabulary projection"
        assert all(ref() is None for ref in previous_logits), (
            "Previous projection retained across chunks/requests"
        )
        calls.append(len(hidden))
        logits = hidden.clone()
        previous_logits.append(weakref.ref(logits))
        return logits

    return SimpleNamespace(
        num_prompt_logprobs=dict(counts),
        requests=requests,
        device=torch.device("cpu"),
        model=SimpleNamespace(compute_logits=project),
        model_config=SimpleNamespace(logprobs_mode=mode),
        sampler=SimpleNamespace(
            compute_logprobs=Sampler.compute_logprobs,
            gather_logprobs=Sampler.gather_logprobs,
        ),
        input_batch=SimpleNamespace(
            req_id_to_index={req_id: i for i, req_id in enumerate(order)}
        ),
        query_start_loc=SimpleNamespace(np=np.array([0])),
        _sync_device=Mock(),
        projection_rows=calls,
    )


def reference(hidden, req, count, mode):
    scores = (
        hidden.float()
        if mode in ("raw_logits", "processed_logits")
        else Sampler.compute_logprobs(hidden)
    )
    return Sampler.gather_logprobs(
        scores, count, torch.tensor(req.prompt_token_ids[1:])
    )


def assert_same(actual, expected):
    assert isinstance(actual, LogprobsTensors)
    assert actual.logprob_token_ids.dtype == torch.int32
    assert actual.logprobs.dtype == torch.float32
    assert actual.selected_token_ranks.dtype == torch.int32
    assert actual.logprobs.device.type == "cpu"
    assert torch.equal(actual.logprob_token_ids, expected.logprob_token_ids)
    torch.testing.assert_close(actual.logprobs, expected.logprobs, rtol=0, atol=0)
    assert torch.equal(actual.selected_token_ranks, expected.selected_token_ranks)
    assert actual.cu_num_generated_tokens is None


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("count", [0, 3, VOCAB])
def test_projection_chunks_preserve_complete_prompt_api(mode, count):
    # V1 resolves prompt_logprobs=-1 to vocab_size before this method.
    req = request(2054)
    r = runner({"r": req}, {"r": count}, mode=mode)
    hidden = hidden_rows(2054)
    expected = reference(hidden[:-1], req, count, mode)
    result = GPUModelRunner._get_prompt_logprobs_dict(r, hidden, {"r": 2054})
    assert_same(result["r"], expected)
    assert r.projection_rows == [1024, 1024, 5]
    assert not r.num_prompt_logprobs
    assert req.in_progress_prompt_logprobs_cpu is None
    r._sync_device.assert_called_once()


@pytest.mark.parametrize("mode", MODES)
def test_partial_prefill_accumulates_next_token_scores_at_packed_offsets(mode):
    req = request(2300)
    r = runner({"r": req}, {"r": 3}, mode=mode)
    full = hidden_rows(2300)
    expected = reference(full[:-1], req, 3, mode)
    r.query_start_loc.np = np.array([7])
    first_hidden = torch.cat([hidden_rows(7), full[:1300]])
    assert GPUModelRunner._get_prompt_logprobs_dict(r, first_hidden, {"r": 1300}) == {}
    pending = req.in_progress_prompt_logprobs_cpu
    assert pending is not None
    torch.testing.assert_close(pending.logprobs[:1300], expected.logprobs[:1300])
    r._sync_device.assert_not_called()

    req.num_computed_tokens = 1300
    r.query_start_loc.np = np.array([11])
    last_hidden = torch.cat([hidden_rows(11), full[1300:]])
    result = GPUModelRunner._get_prompt_logprobs_dict(r, last_hidden, {"r": 1000})
    assert result["r"] is pending
    assert_same(result["r"], expected)
    assert r.projection_rows == [1024, 276, 999]
    assert not r.num_prompt_logprobs
    r._sync_device.assert_called_once()


def test_exact_penultimate_prefill_defers_delivery_without_reprojecting():
    req = request(1025)
    r = runner({"r": req}, {"r": 2})
    full = hidden_rows(1025)
    expected = reference(full[:-1], req, 2, "raw_logprobs")
    assert GPUModelRunner._get_prompt_logprobs_dict(r, full[:-1], {"r": 1024}) == {}
    assert "r" in r.num_prompt_logprobs
    req.num_computed_tokens = 1024
    result = GPUModelRunner._get_prompt_logprobs_dict(r, full[-1:], {"r": 1})
    assert_same(result["r"], expected)
    assert r.projection_rows == [1024]
    assert req.in_progress_prompt_logprobs_cpu is None
    r._sync_device.assert_called_once()


def test_single_token_prompt_delivers_empty_result_without_projection():
    req = request(1)
    r = runner({"r": req}, {"r": 0})
    result = GPUModelRunner._get_prompt_logprobs_dict(r, hidden_rows(1), {"r": 1})
    assert result["r"].logprobs.shape == (0, 1)
    assert result["r"].selected_token_ranks.shape == (0,)
    assert not r.projection_rows and not r.num_prompt_logprobs
    assert req.in_progress_prompt_logprobs_cpu is None
    r._sync_device.assert_called_once()


def test_request_order_offsets_and_unscheduled_requests_remain_independent():
    early, late, paused = request(4), request(1100), request(8)
    embeddings = request(1)
    embeddings.prompt_token_ids = None
    requests = {"late": late, "early": early, "paused": paused, "embed": embeddings}
    r = runner(
        requests,
        {"late": 2, "early": 0, "paused": 3, "embed": 1},
        order=["early", "late", "embed"],
    )
    r.query_start_loc.np = np.array([0, 4, 1104])
    hidden = hidden_rows(1105)
    result = GPUModelRunner._get_prompt_logprobs_dict(
        r, hidden, {"early": 4, "late": 1100, "embed": 1}
    )
    assert set(result) == {"early", "late"}
    assert_same(result["early"], reference(hidden[:3], early, 0, "raw_logprobs"))
    assert_same(result["late"], reference(hidden[4:1103], late, 2, "raw_logprobs"))
    assert r.num_prompt_logprobs == {"paused": 3, "embed": 1}
    assert paused.in_progress_prompt_logprobs_cpu is None
    assert embeddings.in_progress_prompt_logprobs_cpu is None
    assert r.projection_rows == [1024, 75, 3]
    r._sync_device.assert_called_once()
