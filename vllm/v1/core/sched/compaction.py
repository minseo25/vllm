# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler boundaries for explicitly instrumented compaction requests."""

from typing import Any

DESCRIPTOR_KEY = "native_compaction_v1"


def compaction_descriptor(request: Any) -> Any:
    params = getattr(request, "sampling_params", None)
    extras = params.extra_args if params is not None else None
    return (extras or {}).get(DESCRIPTOR_KEY)


def limit_compaction_tokens(request: Any, computed: int, scheduled: int) -> int:
    """Prevent a forward from consuming both sides of a state/KV intervention."""
    descriptor = compaction_descriptor(request)
    if descriptor is None:
        return scheduled
    if not isinstance(descriptor, dict):
        raise ValueError("native_compaction_v1 must be an operation descriptor")
    boundary = descriptor.get("restore_at", 0)
    expected = descriptor.get("expected_prompt_tokens")
    if (
        type(boundary) is not int
        or type(expected) is not int
        or not 0 <= boundary < expected
        or expected != request.num_prompt_tokens
    ):
        raise ValueError("Invalid native compaction intervention boundary")
    prefill_boundary = descriptor.get("prefill_boundary")
    if prefill_boundary is not None and (
        type(prefill_boundary) is not int or not 0 < prefill_boundary < expected
    ):
        raise ValueError("Invalid native compaction prefill boundary")
    for limit in (boundary, prefill_boundary):
        if limit is not None and computed < limit:
            scheduled = min(scheduled, limit - computed)
    return scheduled


def refuse_compaction_preemption(request: Any) -> None:
    """Fail closed with a scheduler-level message instead of a worker contract error.

    A preempted request restarts its prefill from a different cursor, which the
    native boundary contract cannot honour; the worker would otherwise fail the
    whole forward with "Unexpected request, preemption, or noncontiguous cursor".
    The cure is a smaller probe cohort or a larger KV pool, decided by the
    caller, not a silent restart.
    """
    if compaction_descriptor(request) is None:
        return
    raise RuntimeError(
        "Refusing to preempt native compaction request "
        f"{getattr(request, 'request_id', '?')}: the KV pool cannot hold the "
        "current cohort; shrink the probe batch or raise the KV budget"
    )
