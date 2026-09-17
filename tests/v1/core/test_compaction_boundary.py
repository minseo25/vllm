# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""An intervention must occur between forwards, even with a large token budget."""

from types import SimpleNamespace as NS

import pytest

from vllm.v1.core.sched.compaction import limit_compaction_tokens


def request(boundary=73, expected=128):
    return NS(
        num_prompt_tokens=128,
        sampling_params=NS(
            extra_args={
                "native_compaction_v1": {
                    "restore_at": boundary,
                    "expected_prompt_tokens": expected,
                },
            }
        ),
    )


def test_large_prefill_is_split_once_at_intervention_then_continues():
    req = request()
    assert limit_compaction_tokens(req, 0, 128) == 73
    assert limit_compaction_tokens(req, 73, 55) == 55
    assert limit_compaction_tokens(req, 128, 1) == 1


def test_small_chunks_do_not_overshoot_boundary_or_schedule_zero_work():
    req = request()
    assert limit_compaction_tokens(req, 0, 64) == 64
    assert limit_compaction_tokens(req, 64, 64) == 9
    assert limit_compaction_tokens(req, 73, 32) == 32


def test_ordinary_and_zero_cursor_operations_do_not_change_scheduling():
    assert limit_compaction_tokens(NS(sampling_params=None), 10, 90) == 90
    assert limit_compaction_tokens(request(0), 0, 128) == 128


def test_common_prefill_barrier_does_not_delay_initial_state_carry():
    req = request(0)
    req.sampling_params.extra_args["native_compaction_v1"]["prefill_boundary"] = 73
    assert limit_compaction_tokens(req, 0, 128) == 73
    assert limit_compaction_tokens(req, 73, 55) == 55


def test_distinct_intervention_and_prefill_barriers_are_both_observed():
    req = request(40)
    req.sampling_params.extra_args["native_compaction_v1"]["prefill_boundary"] = 73
    assert limit_compaction_tokens(req, 0, 128) == 40
    assert limit_compaction_tokens(req, 40, 88) == 33
    assert limit_compaction_tokens(req, 73, 55) == 55


@pytest.mark.parametrize(
    "boundary,expected", [(True, 128), (-1, 128), (128, 128), (73, 129)]
)
def test_invalid_boundary_fails_before_allocation(boundary, expected):
    with pytest.raises(ValueError, match="boundary"):
        limit_compaction_tokens(request(boundary, expected), 0, 128)


def test_preempting_an_instrumented_request_is_refused_with_a_scheduler_message():
    from vllm.v1.core.sched.compaction import refuse_compaction_preemption

    plain = NS(request_id="p", sampling_params=None)
    refuse_compaction_preemption(plain)
    refuse_compaction_preemption(NS(request_id="q", sampling_params=NS(extra_args=None)))
    with pytest.raises(RuntimeError, match="Refusing to preempt native compaction request r"):
        refuse_compaction_preemption(NS(request_id="r", **vars(request())))
