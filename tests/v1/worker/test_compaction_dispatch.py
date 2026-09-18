# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real cudagraph dispatch with native compaction query export (Codex M2).

Uses the real ``CudagraphDispatcher`` and ``GPUModelRunner`` dispatch helper on
a fake runner so the uniform-decode heuristic is exercised as in production: a
one-token scheduled slice inside an active export range used to dispatch FULL,
which the controller then refused. No model runs; CPU only.
"""

from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from tests.v1.worker.test_compaction_q import (
    FA,
    advance,
    forward_inputs,
    make_controller,
    query_batch,
    run_attention,
)
from vllm.config import CUDAGraphMode
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.worker import gpu_model_runner
from vllm.v1.worker.compaction import CompactionContractError
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

CAPTURE_SIZES = [1, 2, 4, 8]


def make_dispatcher(max_num_seqs=4):
    compilation = NS(
        cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        is_attention_compiled_piecewise=lambda: True,
        max_cudagraph_capture_size=CAPTURE_SIZES[-1],
        cudagraph_capture_sizes=list(CAPTURE_SIZES),
        compile_sizes=None,
        cudagraph_specialize_lora=False,
    )
    config = NS(
        compilation_config=compilation,
        num_speculative_tokens=0,
        lora_config=None,
        scheduler_config=NS(max_num_seqs=max_num_seqs),
    )
    dispatcher = CudagraphDispatcher(config)
    dispatcher.initialize_cudagraph_keys(
        CUDAGraphMode.FULL_AND_PIECEWISE, uniform_decode_query_len=1
    )
    return dispatcher


def make_runner(controller, *, dp_size=1):
    """A runner double carrying exactly the state the dispatch helper reads."""
    batch = controller.runner.input_batch
    batch.num_computed_tokens_cpu = np.array(
        batch.num_computed_tokens_cpu, dtype=np.int32
    )
    batch.lora_id_to_lora_request = {}
    parallel = NS(
        data_parallel_size=dp_size, data_parallel_rank=0, tensor_parallel_size=1
    )
    return NS(
        uniform_decode_query_len=1,
        _is_uniform_decode=GPUModelRunner._is_uniform_decode,
        model_config=NS(is_encoder_decoder=False),
        input_batch=batch,
        _attn_group_iterator=lambda: iter(()),
        _pad_for_sequence_parallelism=lambda n: n,
        cudagraph_dispatcher=make_dispatcher(),
        compilation_config=NS(pass_config=NS(enable_sp=False)),
        vllm_config=NS(
            parallel_config=parallel,
            observability_config=NS(cudagraph_metrics=False),
        ),
        parallel_config=parallel,
        compaction=controller,
        requests=controller.runner.requests,
    )


def dispatch(runner, counts, **overrides):
    counts = np.array(counts, dtype=np.int32)
    return GPUModelRunner._determine_batch_execution_and_padding(
        runner,
        num_tokens=int(counts.sum()),
        num_reqs=len(counts),
        num_scheduled_tokens_np=counts,
        max_num_scheduled_tokens=int(counts.max()),
        use_cascade_attn=False,
        **overrides,
    )


def export_descriptor(start=4, end=5, prompt=5, operation_id="e"):
    return dict(
        operation_id=operation_id,
        expected_prompt_tokens=prompt,
        start_cursor=start,
        export_q={"name": "Q", "token_range": [start, end]},
    )


def test_one_token_exported_first_chunk_dispatches_piecewise_not_full():
    requests = [("r", 4, 1, export_descriptor(), 2)]
    controller, layers, _ = make_controller(requests)
    runner = make_runner(controller)
    # Codex's reproduction: without the inspection the heuristic picks FULL and
    # the forward-time guard fails the input.
    runner.compaction = None
    mode, desc, *_ = dispatch(runner, [1])
    assert (mode, desc.num_tokens, desc.uniform) == (CUDAGraphMode.FULL, 1, True)
    metadata, positions, counts = forward_inputs(requests, controller)
    with pytest.raises(CompactionContractError, match="FULL"):
        controller.before_forward(metadata, positions, counts, mode.name, desc.num_reqs)
    # With the inspection FULL is excluded; nothing was bound or allocated by it.
    controller, layers, _ = make_controller(requests)
    runner = make_runner(controller)
    mode, desc, should_ubatch, across_dp, stats = dispatch(runner, [1])
    assert (mode, desc.num_tokens, desc.uniform) == (CUDAGraphMode.PIECEWISE, 1, False)
    assert (should_ubatch, across_dp, stats) == (False, None, None)
    assert controller.operations == {} and controller.q_exports()["exports"] == {}
    metadata, positions, counts = forward_inputs(requests, controller)
    boundary = controller.before_forward(
        metadata, positions, counts, mode.name, desc.num_reqs
    )
    run_attention(layers, query_batch(1), metadata)
    controller.after_forward(boundary)
    assert controller.q_exports()["exports"]["Q"]["complete"]
    assert controller.results(["e"])["e"]["q_export"]["rows"] == 1


def test_legacy_armed_continuation_one_token_dispatches_piecewise():
    requests = [("r", 4, 1, None, 2)]
    controller, layers, _ = make_controller(requests)
    controller.arm(
        expected_prompt_tokens=5,
        start_cursor=4,
        export_q={"name": "L", "token_range": [4, 5]},
    )
    runner = make_runner(controller)
    mode, desc, *_ = dispatch(runner, [1])
    assert mode == CUDAGraphMode.PIECEWISE and desc.num_tokens == 1
    metadata, positions, counts = forward_inputs(requests, controller)
    boundary = controller.before_forward(metadata, positions, counts, mode.name, 1)
    run_attention(layers, query_batch(1), metadata)
    controller.after_forward(boundary)
    assert controller.result()["q_export"]["complete"]


def test_one_token_remainder_of_a_longer_export_dispatches_piecewise():
    desc = export_descriptor(start=0, end=5, prompt=5)
    requests = [("r", 0, 4, desc, 2)]
    controller, layers, _ = make_controller(requests)
    runner = make_runner(controller)
    mode, batch_desc, *_ = dispatch(runner, [4])
    assert mode == CUDAGraphMode.PIECEWISE and batch_desc.num_tokens == 4
    metadata, positions, counts = forward_inputs(requests, controller)
    boundary = controller.before_forward(metadata, positions, counts, mode.name, 1)
    run_attention(layers, query_batch(4), metadata)
    controller.after_forward(boundary)
    remainder = [("r", 4, 1, desc, 2)]
    advance(controller, remainder)
    runner = make_runner(controller)
    mode, batch_desc, *_ = dispatch(runner, [1])
    assert mode == CUDAGraphMode.PIECEWISE and batch_desc.num_tokens == 1
    metadata, positions, counts = forward_inputs(remainder, controller)
    boundary = controller.before_forward(metadata, positions, counts, mode.name, 1)
    run_attention(layers, query_batch(1) + 50, metadata)
    controller.after_forward(boundary)
    receipt = controller.results(["e"])["e"]
    assert receipt["q_export"]["complete"] and [
        c["rows"] for c in receipt["q_export_chunks"]
    ] == [4, 1]


def test_mixed_padded_batch_with_an_exported_row_is_not_dispatched_full():
    requests = [("dec", 10, 1, None, 3), ("flag", 4, 1, export_descriptor(), 2)]
    controller, layers, _ = make_controller(requests)
    runner = make_runner(controller)
    runner.compaction = None
    mode, desc, *_ = dispatch(runner, [1, 1])
    assert (mode, desc.uniform, desc.num_tokens) == (CUDAGraphMode.FULL, True, 2)
    runner.compaction = controller
    mode, desc, *_ = dispatch(runner, [1, 1])
    assert (mode, desc.uniform, desc.num_tokens) == (CUDAGraphMode.PIECEWISE, False, 2)
    metadata, positions, counts = forward_inputs(requests, controller)
    boundary = controller.before_forward(
        metadata, positions, counts, mode.name, desc.num_reqs
    )
    query = query_batch(2)
    run_attention(layers, query, metadata)
    controller.after_forward(boundary)
    for name in FA:
        assert torch.equal(controller.q_store.tensors("Q")[name], query[1:2])


def test_ordinary_full_decode_outside_the_export_range_is_unaffected():
    # The flagged request has passed its range; a plain decode row rides along.
    requests = [("dec", 10, 1, None, 3), ("flag", 5, 1, export_descriptor(), 2)]
    controller, layers, _ = make_controller(requests)
    runner = make_runner(controller)
    mode, desc, *_ = dispatch(runner, [1, 1])
    assert (mode, desc.uniform, desc.num_tokens) == (CUDAGraphMode.FULL, True, 2)
    # A batch with no instrumented request at all keeps FULL as well.
    plain = [("a", 7, 1, None, 3), ("b", 9, 1, None, 2)]
    controller, layers, _ = make_controller(plain)
    runner = make_runner(controller)
    mode, desc, *_ = dispatch(runner, [1, 1])
    assert (
        mode == CUDAGraphMode.FULL
        and controller.requires_query_export(2, [1, 1]) is False
    )


def test_capture_override_skips_the_inspection_and_keeps_full():
    requests = [("r", 4, 1, export_descriptor(), 2)]
    controller, layers, _ = make_controller(requests)
    runner = make_runner(controller)
    calls = []
    original = controller.requires_query_export
    controller.requires_query_export = lambda *a: calls.append(a) or original(*a)
    mode, desc, *_ = dispatch(runner, [1], force_uniform_decode=True)
    assert mode == CUDAGraphMode.FULL and desc.uniform and calls == []
    mode, *_ = dispatch(runner, [1], force_uniform_decode=False)
    assert mode == CUDAGraphMode.PIECEWISE and calls == []
    mode, *_ = dispatch(runner, [1])
    assert mode == CUDAGraphMode.PIECEWISE and len(calls) == 1


def test_dp_redispatch_keeps_full_excluded(monkeypatch):
    requests = [("r", 4, 1, export_descriptor(), 2)]
    controller, layers, _ = make_controller(requests)
    runner = make_runner(controller, dp_size=2)
    seen = []

    def fake_coordinate(**kwargs):
        seen.append(kwargs)
        return (
            False,
            torch.tensor([kwargs["num_tokens_padded"]] * 2),
            kwargs["cudagraph_mode"],
        )

    monkeypatch.setattr(gpu_model_runner, "coordinate_batch_across_dp", fake_coordinate)
    mode, desc, should_ubatch, across_dp, _ = dispatch(runner, [1])
    assert mode == CUDAGraphMode.PIECEWISE and desc.num_tokens == 1
    assert across_dp.tolist() == [1, 1] and should_ubatch is False
    assert seen[0]["cudagraph_mode"] == CUDAGraphMode.PIECEWISE.value
    assert seen[0]["uniform_decode"] is True  # the heuristic still saw a decode shape
