# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU ownership/routing tests; these do not validate CUDA state consumption."""

from types import SimpleNamespace as NS

import pytest
import torch

from vllm.v1.worker.compaction import (
    DESCRIPTOR_KEY,
    BoundaryOperation,
    CompactionContractError,
    NativeCompactionController,
    SnapshotStore,
    resolve_slot,
    validate_configuration,
    verify_fa_context,
)


def slot(name="layer0", family="gdn", index=1, prefill=True, value=0.0):
    states = (torch.full((3, 2, 3), value), torch.full((3, 2, 4, 4), value))
    metadata = NS(
        num_prefills=int(prefill), num_decodes=int(not prefill), num_spec_decodes=0
    )
    indices = torch.tensor([index])
    if family == "gdn":
        metadata.prefill_state_indices = indices
        metadata.non_spec_state_indices_tensor = indices
        metadata.has_initial_state = torch.tensor([False]) if prefill else None
        metadata.prefill_has_initial_state = torch.tensor([False]) if prefill else None
    else:
        metadata.state_indices_tensor_p = indices if prefill else None
        metadata.state_indices_tensor_d = indices[:, None] if not prefill else None
        metadata.has_initial_states_p = torch.tensor([False]) if prefill else None
        metadata.prep_initial_states = False
    return resolve_slot(name, family, states, metadata)


@pytest.mark.parametrize("family", ["gdn", "mamba2"])
@pytest.mark.parametrize("prefill", [False, True])
def test_restore_routes_to_current_slot_and_snapshot_survives_probe(family, prefill):
    source = slot(family=family, value=3.0)
    store = SnapshotStore()
    original = store.capture("H", [source], {"processed_prompt_tokens": 4})
    for state in source.states:
        state.fill_(99)
    target = slot(family=family, index=2, prefill=prefill, value=-1.0)
    receipt = store.restore("H", [target])
    assert receipt["snapshot_unchanged_after_restore"] is None
    assert store.audit("H")["immutable_digest_verified"]
    for state in target.states:
        assert torch.all(state[2] == 3)
        assert torch.all(state[:2] == -1)
        state.fill_(57)  # A continued probe can mutate its entire live cache.
    assert store.describe("H")["digest"] == original["digest"]
    store.restore("H", [target])
    assert torch.all(target.states[1][2] == 3)
    if prefill and family == "mamba2":
        assert target.metadata.has_initial_states_p.item()
        assert target.metadata.prep_initial_states
    elif prefill:
        assert target.metadata.has_initial_state.item()
        assert target.metadata.prefill_has_initial_state.item()


def test_restore_validates_all_layers_before_mutating_any_destination():
    store = SnapshotStore()
    store.capture("H", [slot("a", value=3.0), slot("b", value=3.0)], {})
    a, b = slot("a", value=-1.0), slot("b", value=-1.0)
    b.states[1].resize_(3, 2, 5, 5)
    with pytest.raises(CompactionContractError, match="shape/dtype"):
        store.restore("H", [a, b])
    assert torch.all(a.states[0] == -1)
    assert not a.metadata.has_initial_state.item()


def test_prompt_end_is_captured_once_and_excludes_later_decode_state():
    store = SnapshotStore()
    current = slot(value=1.0)
    operation = BoundaryOperation(
        store, capture_name="H", restore_name=None, expected_prompt_tokens=5
    )
    operation.before("request", 5, 0, 3, [current])
    operation.after(3, [current])
    assert store.list()["snapshots"] == {}
    operation.before("request", 5, 3, 2, [current])
    operation.after(2, [current])
    digest = store.describe("H")["digest"]
    current.states[1].fill_(12)
    operation.before("request", 5, 5, 1, [current])
    operation.after(1, [current])
    receipt = operation.result()
    assert receipt["prompt_tokens_seen"] == 5
    assert receipt["processed_cache_cursor"] == 6
    assert receipt["consumed_generation_tokens"] == 1
    assert receipt["snapshot"]["boundary"]["sampled_output_tokens_consumed"] == 0
    assert store.describe("H")["digest"] == digest
    assert operation.closed and receipt["forward_calls"] == 3


def test_restore_happens_once_then_later_chunks_keep_their_own_trajectory():
    store = SnapshotStore()
    store.capture("H", [slot(value=7.0)], {})
    current = slot(value=0.0)
    operation = BoundaryOperation(
        store, capture_name=None, restore_name="H", expected_prompt_tokens=4
    )
    operation.before("new", 4, 0, 2, [current])
    assert current.states[1][1, 0, 0, 0] == 7
    current.states[1][1].fill_(13)
    operation.after(2, [current])
    operation.before("new", 4, 2, 2, [current])
    assert current.states[1][1, 0, 0, 0] == 13
    operation.after(2, [current])
    assert operation.result()["restored_from"] == "H"


@pytest.mark.parametrize("computed,count,prompt", [(1, 2, 4), (0, 5, 4), (0, 2, 3)])
def test_refuses_existing_context_boundary_crossing_and_prompt_mismatch(
    computed, count, prompt
):
    operation = BoundaryOperation(
        SnapshotStore(), capture_name=None, restore_name=None, expected_prompt_tokens=4
    )
    with pytest.raises(CompactionContractError):
        operation.before("new", prompt, computed, count, [slot()])


def test_preemption_or_request_switch_cannot_silently_restart_a_carry():
    operation = BoundaryOperation(
        SnapshotStore(), capture_name=None, restore_name=None, expected_prompt_tokens=4
    )
    operation.before("r", 4, 0, 2, [slot()])
    operation.after(2, [slot()])
    for request_id, cursor in [("r", 0), ("other", 2)]:
        with pytest.raises(CompactionContractError, match="preemption"):
            operation.before(request_id, 4, cursor, 2, [slot()])


def test_invalid_first_boundary_does_not_import_state_before_rejection():
    store = SnapshotStore()
    store.capture("H", [slot(value=7.0)], {})
    current = slot(value=-1.0)
    operation = BoundaryOperation(
        store, capture_name=None, restore_name="H", expected_prompt_tokens=4
    )
    with pytest.raises(CompactionContractError, match="crosses"):
        operation.before("new", 4, 0, 5, [current])
    assert torch.all(current.states[1] == -1)
    assert not current.metadata.has_initial_state.item()
    assert operation.request_id is None


def config():
    return NS(
        parallel_config=NS(
            tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1
        ),
        scheduler_config=NS(
            max_num_seqs=1, async_scheduling=False, enable_chunked_prefill=True
        ),
        model_config=NS(enforce_eager=True, hf_config=NS(model_type="qwen3_5")),
        cache_config=NS(
            enable_prefix_caching=False, mamba_cache_mode="none", use_replayssm=False
        ),
        speculative_config=None,
        mamba_config=NS(
            state_quant_bits=None,
            state_trace_dir=None,
            enable_stochastic_rounding=False,
        ),
    )


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("cache_config", "use_replayssm", True),
        ("cache_config", "enable_prefix_caching", True),
        ("cache_config", "mamba_cache_mode", "align"),
        ("scheduler_config", "async_scheduling", True),
        ("scheduler_config", "async_scheduling", None),
        ("mamba_config", "state_quant_bits", 8),
        ("mamba_config", "state_trace_dir", "/tmp/trace"),
    ],
)
def test_unsupported_lifecycle_configuration_fails_closed(section, key, value):
    supported = config()
    validate_configuration(supported)
    setattr(getattr(supported, section), key, value)
    with pytest.raises(CompactionContractError):
        validate_configuration(supported)


def test_snapshot_names_and_exposed_metadata_cannot_overwrite_saved_state():
    store = SnapshotStore()
    receipt = store.capture("H", [slot()], {"nested": {"position": 4}})
    receipt["boundary"]["nested"]["position"] = 999
    assert store.describe("H")["boundary"]["nested"]["position"] == 4
    with pytest.raises(CompactionContractError, match="new and nonempty"):
        store.capture("H", [slot(value=3.0)], {})
    store.drop("H")
    assert store.list()["total_cpu_snapshot_bytes"] == 0


def test_fa_zero_context_receipt_comes_from_layer_lengths():
    metadata = {
        "fa": NS(seq_lens=torch.tensor([3]), query_start_loc=torch.tensor([0, 3]))
    }
    evidence = verify_fa_context(metadata, {"fa"}, query_tokens=3, computed_tokens=0)
    operation = BoundaryOperation(
        SnapshotStore(), capture_name=None, restore_name=None, expected_prompt_tokens=3
    )
    operation.before("r", 3, 0, 3, [slot()], evidence)
    operation.after(3, [slot()])
    result = operation.result()
    assert result["zero_fa_context_first_chunk"]
    assert result["first_fa_context"]["fa"]["seq_len"] == 3
    metadata["fa"].seq_lens.fill_(8)
    with pytest.raises(CompactionContractError, match="FA context"):
        verify_fa_context(metadata, {"fa"}, query_tokens=3, computed_tokens=0)


def test_unknown_fa_metadata_does_not_emit_a_false_empty_context_receipt():
    with pytest.raises(CompactionContractError, match="Unsupported FA"):
        verify_fa_context({"fa": NS()}, {"fa"}, query_tokens=3, computed_tokens=0)


@pytest.mark.parametrize("component,value", [(0, float("nan")), (1, float("inf"))])
def test_nonfinite_conv_or_ssm_cannot_be_saved_as_a_usable_checkpoint(component, value):
    store = SnapshotStore()
    current = slot()
    current.states[component][current.index].fill_(value)
    with pytest.raises(CompactionContractError, match="Nonfinite"):
        store.capture("bad", [current], {})
    assert store.list()["snapshots"] == {}


def test_continuation_captures_cumulative_boundary_without_reprocessing_history():
    store = SnapshotStore()
    current = slot(value=7.0)
    operation = BoundaryOperation(
        store,
        capture_name="continued",
        restore_name=None,
        expected_prompt_tokens=9,
        start_cursor=5,
        expected_request_id="session",
    )
    fa_context = {"fa": {"seq_len": 7, "query_tokens": 2, "context_tokens": 5}}
    operation.before("session", 9, 5, 2, [current], fa_context)
    assert current.states[1][1, 0, 0, 0] == 7
    assert not current.metadata.has_initial_state.item()
    operation.after(2, [current])
    current.states[1][1].fill_(11)
    operation.before("session", 9, 7, 2, [current], fa_context)
    operation.after(2, [current])
    receipt = operation.result()
    assert receipt["continuation"]
    assert receipt["start_cursor"] == receipt["first_computed_tokens"] == 5
    assert receipt["first_chunk_num_computed_tokens"] == 5
    assert receipt["new_tokens_processed"] == 4
    assert receipt["consumed_prompt_tokens"] == 9
    assert receipt["processed_cache_cursor"] == 9
    assert receipt["forward_chunks"] == [
        {"start": 5, "end": 7, "query_tokens": 2},
        {"start": 7, "end": 9, "query_tokens": 2},
    ]
    assert not receipt["zero_fa_context_first_chunk"]
    assert receipt["snapshot"]["boundary"]["processed_prompt_tokens"] == 9
    assert receipt["snapshot"]["boundary"]["start_cursor"] == 5
    assert receipt["restored_from"] is None


@pytest.mark.parametrize("start_cursor", [-1, True, 4.0, 5, 6])
def test_invalid_continuation_cursor_cannot_be_armed(start_cursor):
    with pytest.raises(CompactionContractError, match="start_cursor"):
        BoundaryOperation(
            SnapshotStore(),
            capture_name=None,
            restore_name=None,
            expected_prompt_tokens=5,
            start_cursor=start_cursor,
        )


def test_restore_into_a_nonempty_continuation_is_rejected_before_mutation():
    store = SnapshotStore()
    original = store.capture("H", [slot(value=7.0)], {})
    with pytest.raises(CompactionContractError, match="Restore requires"):
        BoundaryOperation(
            store,
            capture_name=None,
            restore_name="H",
            expected_prompt_tokens=5,
            start_cursor=2,
        )
    assert store.describe("H")["digest"] == original["digest"]


def test_wrong_expected_request_cannot_receive_a_restore():
    store = SnapshotStore()
    store.capture("H", [slot(value=7.0)], {})
    current = slot(value=-1.0)
    operation = BoundaryOperation(
        store,
        capture_name=None,
        restore_name="H",
        expected_prompt_tokens=4,
        expected_request_id="expected",
    )
    with pytest.raises(CompactionContractError, match="request identity"):
        operation.before("another", 4, 0, 2, [current])
    assert torch.all(current.states[1] == -1)
    assert not current.metadata.has_initial_state.item()
    assert operation.request_id is None


def test_continuation_rejects_wrong_initial_cursor_without_saving_a_snapshot():
    store = SnapshotStore()
    operation = BoundaryOperation(
        store,
        capture_name="continued",
        restore_name=None,
        expected_prompt_tokens=7,
        start_cursor=4,
    )
    with pytest.raises(CompactionContractError, match="start_cursor"):
        operation.before("session", 7, 0, 3, [slot()])
    assert operation.processed_tokens == 4
    assert store.list()["snapshots"] == {}


def test_after_requires_matching_forward_and_cannot_double_count_tokens():
    current = slot()
    operation = BoundaryOperation(
        SnapshotStore(),
        capture_name=None,
        restore_name=None,
        expected_prompt_tokens=3,
    )
    with pytest.raises(CompactionContractError, match="matching"):
        operation.after(3, [current])
    operation.before("request", 3, 0, 3, [current])
    with pytest.raises(CompactionContractError, match="pending"):
        operation.before("request", 3, 0, 3, [current])
    with pytest.raises(CompactionContractError, match="matching"):
        operation.after(2, [current])
    assert operation.processed_tokens == 0
    operation.after(3, [current])
    assert operation.result()["new_tokens_processed"] == 3
    with pytest.raises(CompactionContractError, match="matching"):
        operation.after(3, [current])
    assert operation.processed_tokens == 3


def controller_fixture():
    """Exercise controller hooks on CPU without creating a GPU model runner."""
    current = slot(value=2.0)
    current.metadata.num_prefill_tokens = 3
    current.metadata.num_decode_tokens = 0
    controller = object.__new__(NativeCompactionController)
    controller.store = SnapshotStore()
    controller.operation = None
    controller.operations = {}
    controller._descriptors = {}
    controller._retired_bindings = {}
    controller._seen_operation_ids = set()
    controller._position_rules = {}
    controller.layer_groups = {}
    controller.layers = {"layer0": ("gdn", NS(kv_cache=current.states))}
    controller.fa_layers = {"fa"}
    controller.metadata_types = {"gdn": NS}
    controller.runner = NS(
        execute_model_state=None,
        device=torch.device("cpu"),
        input_batch=NS(
            num_reqs=1,
            req_ids=["request"],
            num_computed_tokens_cpu=[0],
        ),
        requests={
            "request": NS(
                mm_features=[],
                prompt_embeds=None,
                lora_request=None,
                num_computed_tokens=0,
                num_prompt_tokens=3,
            )
        },
    )
    metadata = {
        "layer0": current.metadata,
        "fa": NS(seq_lens=torch.tensor([3]), query_start_loc=torch.tensor([0, 3])),
    }
    return controller, current, metadata


def test_controller_disarms_after_receipt_and_leaves_uninstrumented_calls_alone():
    controller, current, metadata = controller_fixture()
    assert controller.before_forward(None, None) is None
    controller.arm(capture_name="H", expected_prompt_tokens=3)
    boundary = controller.before_forward(metadata, torch.arange(3))
    assert controller.snapshots() == {}
    current.states[1][current.index].fill_(8)
    controller.after_forward(boundary)
    receipt = controller.result()
    assert receipt["backend"] == "vllm_native"
    assert receipt["zero_fa_context_first_chunk"]
    assert receipt["snapshot"]["digest"] == controller.snapshots()["H"]["digest"]
    assert controller.before_forward(None, None) is None
    controller.after_forward(None)
    controller.drop("H")
    assert controller.snapshots() == {}


@pytest.mark.parametrize("wrong_input", ["positions", "fa_context"])
def test_controller_rejects_inconsistent_forward_before_restoring_state(wrong_input):
    controller, current, metadata = controller_fixture()
    controller.store.capture("H", [slot(value=9.0)], {})
    controller.arm(restore_name="H", expected_prompt_tokens=3)
    positions = torch.arange(3)
    if wrong_input == "positions":
        positions += 7
    else:
        metadata["fa"].seq_lens += 1
    with pytest.raises(CompactionContractError):
        controller.before_forward(metadata, positions)
    assert torch.all(current.states[1] == 2)
    assert not current.metadata.has_initial_state.item()
    with pytest.raises(CompactionContractError):
        controller.result()


def test_failed_model_forward_cannot_publish_a_successful_boundary_receipt():
    controller, _, metadata = controller_fixture()
    controller.arm(capture_name="failed", expected_prompt_tokens=3)
    controller.before_forward(metadata, torch.arange(3))
    controller.fail_forward(RuntimeError("kernel failed"))
    with pytest.raises(CompactionContractError, match="kernel failed"):
        controller.result()
    assert controller.snapshots() == {}


def test_fresh_one_token_decode_zeros_stale_state_before_consuming_first_token():
    """Fresh budget-limited decode rows must not inherit the retired slot."""
    store = SnapshotStore()
    op = BoundaryOperation(
        store, capture_name=None, restore_name=None, expected_prompt_tokens=1
    )
    current = slot(prefill=False, value=99.0)
    op.before("r1", 1, 0, 1, [current])
    for state in current.states:
        assert torch.all(state[current.index] == 0)
        assert torch.all(state[0] == 99)
    op.after(1, [current])
    assert op.result()["fresh_decode_state_zeroed"]
    # A restored decode row has a defined initial state.
    op2 = BoundaryOperation(
        store, capture_name=None, restore_name=None, expected_prompt_tokens=4
    )
    op2.before("r2", 4, 0, 4, [slot(prefill=True)])
    store.capture("H", [slot(value=2.0)], {"processed_prompt_tokens": 4})
    op3 = BoundaryOperation(
        store, capture_name=None, restore_name="H", expected_prompt_tokens=1
    )
    op3.before("r3", 1, 0, 1, [slot(prefill=False)])
    assert op3.restore_receipt is not None


def test_compare_reports_relative_frobenius_without_mutation():
    store = SnapshotStore()
    a = slot(value=2.0)
    store.capture("A", [a], {"processed_prompt_tokens": 4})
    b = slot(value=2.0)
    b.states[1][b.index] += 0.02  # 1% relative perturbation of the SSM tensor only
    store.capture("B", [b], {"processed_prompt_tokens": 4})
    report = store.compare("A", "B")
    assert report["per_layer"]["layer0"][0] == 0.0
    assert abs(report["per_layer"]["layer0"][1] - 0.01) < 1e-6
    assert abs(report["max_relative_frobenius"] - 0.01) < 1e-6
    assert store.describe("A")["digest"] == store.describe("A")["digest"]


def test_graph_and_batch_configuration_retains_lifecycle_guards():
    supported = config()
    supported.scheduler_config.max_num_seqs = 32
    supported.model_config.enforce_eager = False
    validate_configuration(supported)
    supported.parallel_config.use_ubatching = True
    with pytest.raises(CompactionContractError, match="microbatching"):
        validate_configuration(supported)


@pytest.mark.parametrize("family", ["gdn", "mamba2"])
@pytest.mark.parametrize("components,selected", [("matrix", 1), ("conv", 0)])
@pytest.mark.parametrize("fresh", [False, True])
def test_component_restore_preserves_live_other_component_or_zeros_fresh_slot(
    family, components, selected, fresh
):
    store = SnapshotStore()
    store.capture("H", [slot(family=family, value=7.0)], {})
    current = slot(family=family, value=3.0)
    receipt = store.restore(
        "H", [current], components, zero_unselected=fresh, audit_restore=True
    )
    assert torch.all(current.states[selected][current.index] == 7)
    assert torch.all(current.states[1 - selected][current.index] == (0 if fresh else 3))
    assert receipt["copy_audit"]["selected_components_exact"]
    field = "fresh_unselected_zero" if fresh else "unselected_components_preserved"
    assert receipt["copy_audit"][field]
    assert store.audit("H")["immutable_digest_verified"]


def test_summary_graft_intervenes_once_at_boundary_before_new_query():
    store = SnapshotStore()
    store.capture("H", [slot(value=7.0)], {})
    current = slot(value=1.0)
    operation = BoundaryOperation(
        store,
        capture_name="end",
        restore_name="H",
        expected_prompt_tokens=5,
        restore_at=3,
        restore_components="matrix",
        audit_restore=True,
    )
    with pytest.raises(CompactionContractError, match="restore_at"):
        operation.before("r", 5, 0, 4, [current])
    assert torch.all(current.states[1] == 1)
    operation.before("r", 5, 0, 3, [current])
    current.states[0][current.index].fill_(4)
    current.states[1][current.index].fill_(5)
    operation.after(3, [current])
    operation.before("r", 5, 3, 2, [current])
    assert torch.all(current.states[0][current.index] == 4)
    assert torch.all(current.states[1][current.index] == 7)
    operation.after(2, [current])
    result = operation.result()
    assert result["restore"]["at_cursor"] == 3
    assert result["restore"]["copy_audit"]["unselected_components_preserved"]


@pytest.mark.parametrize("boundary", [0, -1, 5, 6, True, 2.0])
def test_prefill_boundary_must_be_an_exact_interior_token_cursor(boundary):
    with pytest.raises(CompactionContractError, match="prefill_boundary"):
        BoundaryOperation(
            SnapshotStore(),
            capture_name=None,
            restore_name=None,
            expected_prompt_tokens=5,
            prefill_boundary=boundary,
        )


def test_negative_offset_cannot_defer_failure_until_kv_capture_after_forward():
    with pytest.raises(CompactionContractError, match="nonnegative"):
        BoundaryOperation(
            SnapshotStore(),
            capture_name=None,
            restore_name=None,
            expected_prompt_tokens=5,
            restore_at=3,
            position_offset=-1,
            capture_kv={"name": "invalid", "token_indices": [0]},
        )


def test_prefill_barrier_does_not_move_restore_and_rejects_crossing_before_writes():
    store = SnapshotStore()
    store.capture("H", [slot(value=7.0)], {})
    current = slot(value=-1.0)
    operation = BoundaryOperation(
        store,
        capture_name=None,
        restore_name="H",
        expected_prompt_tokens=5,
        restore_at=0,
        prefill_boundary=3,
    )
    with pytest.raises(CompactionContractError, match="prefill_boundary"):
        operation.before("r", 5, 0, 4, [current])
    assert torch.all(current.states[1] == -1)
    assert operation.request_id is None
    operation.before("r", 5, 0, 3, [current])
    assert torch.all(current.states[1][current.index] == 7)
    current.states[1][current.index].fill_(9)
    operation.after(3, [current])
    operation.before("r", 5, 3, 2, [current])
    assert torch.all(current.states[1][current.index] == 9)
    operation.after(2, [current])
    receipt = operation.result()
    assert receipt["prefill_boundary"] == 3
    assert receipt["restore"]["at_cursor"] == receipt["restore_at"] == 0


def test_metadata_is_cached_and_explicit_audit_detects_corrupt_source(monkeypatch):
    import vllm.v1.worker.compaction as module

    store = SnapshotStore()
    store.capture("H", [slot(value=2.0)], {})
    original_hash = module._tensor_digest
    calls = []

    def count_hash(layers):
        calls.append(True)
        return original_hash(layers)

    monkeypatch.setattr(module, "_tensor_digest", count_hash)
    store.describe("H")
    store.list()
    store.restore("H", [slot()])
    assert not calls
    assert not store.describe("H")["immutable_digest_verified"]
    assert store.audit("H")["immutable_digest_verified"]
    assert len(calls) == 1
    store._entries["H"]["layers"]["layer0"][1][1].add_(1)
    with pytest.raises(CompactionContractError, match="mutated"):
        store.audit("H")


def batch_controller(family, requests, *, num_decodes=0, padding=0):
    """Requests: (id, cursor, query_count, descriptor, allocated_state_index)."""
    controller, _, _ = controller_fixture()
    states = (torch.full((8, 2, 3), -1.0), torch.full((8, 2, 4, 4), -1.0))
    count = len(requests)
    indices = torch.tensor([r[4] for r in requests] + [0] * padding)
    prefills = count - num_decodes
    metadata = NS(
        num_prefills=prefills,
        num_decodes=num_decodes + padding,
        num_spec_decodes=0,
        num_prefill_tokens=sum(r[2] for r in requests[num_decodes:]),
        num_decode_tokens=sum(r[2] for r in requests[:num_decodes]) + padding,
    )
    assert padding == 0 or prefills == 0
    if family == "gdn":
        metadata.prefill_state_indices = indices[num_decodes:] if prefills else None
        metadata.non_spec_state_indices_tensor = indices
        metadata.has_initial_state = (
            torch.zeros(count, dtype=torch.bool) if prefills else None
        )
        metadata.prefill_has_initial_state = (
            torch.zeros(prefills, dtype=torch.bool) if prefills else None
        )
    else:
        metadata.state_indices_tensor_p = indices[num_decodes:] if prefills else None
        metadata.state_indices_tensor_d = indices[: num_decodes + padding, None]
        metadata.has_initial_states_p = (
            torch.zeros(prefills, dtype=torch.bool) if prefills else None
        )
        metadata.prep_initial_states = False
    controller.layers = {"layer0": (family, NS(kv_cache=states))}
    controller.metadata_types = {family: NS}
    batch = controller.runner.input_batch
    batch.num_reqs = count
    batch.req_ids = [r[0] for r in requests]
    batch.num_computed_tokens_cpu = [r[1] for r in requests]
    controller.runner.requests = {
        request_id: NS(
            mm_features=[],
            prompt_embeds=None,
            lora_request=None,
            num_computed_tokens=cursor,
            num_prompt_tokens=descriptor["expected_prompt_tokens"]
            if descriptor
            else cursor + query,
            sampling_params=NS(
                extra_args={DESCRIPTOR_KEY: descriptor} if descriptor else {}
            ),
        )
        for request_id, cursor, query, descriptor, _ in requests
    }
    starts = [0]
    for r in requests:
        starts.append(starts[-1] + r[2])
    starts += [starts[-1]] * padding
    fa = NS(
        seq_lens=torch.tensor([r[1] + r[2] for r in requests] + [0] * padding),
        query_start_loc=torch.tensor(starts),
    )
    positions = torch.cat(
        [torch.arange(r[1], r[1] + r[2]) for r in requests]
        + [torch.full((padding,), 99)]
    )
    return (
        controller,
        states,
        {"layer0": metadata, "fa": fa},
        positions,
        {r[0]: r[2] for r in requests},
    )


@pytest.mark.parametrize("family", ["gdn", "mamba2"])
def test_request_descriptors_route_mixed_rows_and_touch_only_selected_prefill_flag(
    family,
):
    requests = [
        (
            "dec",
            2,
            1,
            dict(operation_id="d", expected_prompt_tokens=3, start_cursor=2),
            3,
        ),
        ("fresh", 0, 2, dict(operation_id="f", expected_prompt_tokens=2), 4),
        (
            "carry",
            0,
            3,
            dict(operation_id="c", expected_prompt_tokens=3, restore_name="H"),
            1,
        ),
    ]
    controller, states, metadata, positions, counts = batch_controller(
        family, requests, num_decodes=1
    )
    controller.store.capture("H", [slot(family=family, value=7.0)], {})
    boundary = controller.before_forward(metadata, positions, counts, "PIECEWISE", 3)
    assert torch.all(states[1][1] == 7)
    assert torch.all(states[1][[0, 2, 3, 4, 5, 6, 7]] == -1)
    current = metadata["layer0"]
    flag = (
        current.prefill_has_initial_state
        if family == "gdn"
        else current.has_initial_states_p
    )
    assert flag.tolist() == [False, True]
    if family == "gdn":
        assert current.has_initial_state.tolist() == [False, False, True]
    controller.after_forward(boundary)
    receipts = controller.results(["d", "f", "c"])
    assert receipts["c"]["request_id"] == "carry"
    assert receipts["c"]["forward_chunks"][0]["batch_row"] == 2
    assert receipts["f"]["restored_from"] is None


@pytest.mark.parametrize("family", ["gdn", "mamba2"])
def test_graph_padded_decode_never_selects_null_state_or_modifies_padding_positions(
    family,
):
    requests = [
        (
            "b",
            4,
            1,
            dict(operation_id="b", expected_prompt_tokens=5, start_cursor=4),
            2,
        ),
        (
            "a",
            0,
            1,
            dict(operation_id="a", expected_prompt_tokens=1, restore_name="H"),
            5,
        ),
    ]
    controller, states, metadata, positions, counts = batch_controller(
        family, requests, num_decodes=2, padding=2
    )
    controller.store.capture("H", [slot(family=family, value=7.0)], {})
    boundary = controller.before_forward(metadata, positions, counts, "FULL", 4)
    assert torch.all(states[1][0] == -1)
    assert torch.all(states[1][5] == 7)
    assert positions[2:].tolist() == [99, 99]
    controller.after_forward(boundary)
    result = controller.results(["a", "b"])
    chunk = result["a"]["forward_chunks"][0]
    assert chunk["graph_mode"] == "FULL"
    assert (chunk["batch_size"], chunk["padded_batch_size"]) == (2, 4)


def test_invalid_second_request_rejects_whole_forward_before_first_restore():
    requests = [
        (
            "a",
            0,
            2,
            dict(operation_id="a", expected_prompt_tokens=2, restore_name="H"),
            1,
        ),
        ("b", 1, 2, dict(operation_id="b", expected_prompt_tokens=3), 2),
    ]
    controller, states, metadata, positions, counts = batch_controller("gdn", requests)
    controller.store.capture("H", [slot(value=7.0)], {})
    with pytest.raises(CompactionContractError, match="start_cursor"):
        controller.before_forward(metadata, positions, counts)
    assert torch.all(states[1] == -1)
    assert not metadata["layer0"].has_initial_state.any()


def test_operation_id_cannot_be_shared_across_requests():
    descriptor = dict(operation_id="same", expected_prompt_tokens=2)
    requests = [("a", 0, 2, descriptor, 1), ("b", 0, 2, descriptor, 2)]
    controller, states, metadata, positions, counts = batch_controller("gdn", requests)
    with pytest.raises(CompactionContractError, match="reused"):
        controller.before_forward(metadata, positions, counts)
    assert torch.all(states[1] == -1)


def test_requests_cannot_restore_into_an_aliased_native_slot():
    requests = [
        (
            "a",
            0,
            2,
            dict(operation_id="a", expected_prompt_tokens=2, restore_name="H"),
            1,
        ),
        (
            "b",
            0,
            2,
            dict(operation_id="b", expected_prompt_tokens=2, restore_name="H"),
            1,
        ),
    ]
    controller, states, metadata, positions, counts = batch_controller("gdn", requests)
    controller.store.capture("H", [slot(value=7.0)], {})
    with pytest.raises(CompactionContractError, match="alias"):
        controller.before_forward(metadata, positions, counts)
    assert torch.all(states[1] == -1)


def test_aborted_request_marks_only_its_operation_failed():
    requests = [
        ("a", 0, 2, dict(operation_id="a", expected_prompt_tokens=4), 1),
        ("b", 0, 2, dict(operation_id="b", expected_prompt_tokens=4), 2),
    ]
    controller, _, metadata, positions, counts = batch_controller("gdn", requests)
    controller.after_forward(controller.before_forward(metadata, positions, counts))
    controller.finish_requests({"a"})
    assert "aborted" in controller.operations["a"].failed
    assert controller.operations["b"].failed is None


@pytest.mark.parametrize("index", [0, -1, 8])
def test_null_and_out_of_range_native_slots_are_rejected(index):
    with pytest.raises(CompactionContractError, match="Invalid native state slot"):
        slot(index=index)


def test_reordering_between_chunks_keeps_state_and_capture_owned_by_request():
    requests = [
        (
            "a",
            0,
            2,
            dict(operation_id="a", expected_prompt_tokens=5, capture_name="A"),
            1,
        ),
        (
            "b",
            0,
            2,
            dict(operation_id="b", expected_prompt_tokens=4, capture_name="B"),
            2,
        ),
    ]
    controller, states, metadata, positions, counts = batch_controller("gdn", requests)
    controller.after_forward(controller.before_forward(metadata, positions, counts))
    for component in states:
        component[1].fill_(11)
        component[2].fill_(22)
    batch = controller.runner.input_batch
    batch.req_ids = ["b", "a"]
    batch.num_computed_tokens_cpu = [2, 2]
    for request in controller.runner.requests.values():
        request.num_computed_tokens = 2
    metadata["layer0"].prefill_state_indices = torch.tensor([2, 1])
    metadata["layer0"].non_spec_state_indices_tensor = torch.tensor([2, 1])
    metadata["fa"].seq_lens = torch.tensor([4, 5])
    metadata["fa"].query_start_loc = torch.tensor([0, 2, 5])
    second = controller.before_forward(
        metadata, torch.tensor([2, 3, 2, 3, 4]), {"a": 3, "b": 2}
    )
    controller.after_forward(second)
    result = controller.results(["a", "b"])
    assert result["a"]["forward_chunks"][1]["batch_row"] == 1
    assert result["b"]["forward_chunks"][1]["batch_row"] == 0
    target = slot()
    controller.store.restore("A", [target])
    assert torch.all(target.states[1][target.index] == 11)
    controller.store.restore("B", [target])
    assert torch.all(target.states[1][target.index] == 22)


@pytest.mark.parametrize("finished_first", [False, True])
def test_delivered_results_retire_large_operations_and_live_tombstones(finished_first):
    descriptor = dict(operation_id="a", expected_prompt_tokens=2)
    controller, _, metadata, positions, counts = batch_controller(
        "gdn", [("request", 0, 2, descriptor, 1)]
    )
    controller.after_forward(controller.before_forward(metadata, positions, counts))
    if finished_first:
        controller.finish_requests({"request"})
    result = controller.results(["a"])
    assert result["a"]["forward_calls"] == 1
    assert controller.operations == {}
    if not finished_first:
        assert controller.before_forward(metadata, positions, counts) is None
        controller.finish_requests({"request"})
    assert controller._descriptors == {}
    assert controller._retired_bindings == {}
    with pytest.raises(CompactionContractError, match="reused"):
        controller.before_forward(metadata, positions, counts)


@pytest.mark.parametrize("wrong_offset", [False, True])
def test_kv_and_state_import_share_boundary_and_original_query_positions(wrong_offset):
    from vllm.v1.worker.compaction_kv import NativeKVSnapshotStore

    descriptor = dict(
        operation_id="target",
        expected_prompt_tokens=5,
        restore_name="H",
        restore_at=2,
        kv_restore_name="K",
        position_offset=3 if wrong_offset else 4,
        capture_kv={"name": "roundtrip", "token_indices": [0, 1]},
    )
    controller, states, metadata, positions, counts = batch_controller(
        "gdn", [("target", 0, 2, descriptor, 1)]
    )
    controller.store.capture("H", [slot(value=7.0)], {})
    kv_store = object.__new__(NativeKVSnapshotStore)
    kv_store._entries, kv_store._staged = {}, {}
    cache = torch.arange(8 * 2 * 4 * 6, dtype=torch.float32).reshape(8, 2, 4, 6)
    kv_store.layers = {
        "fa": NS(
            kv_cache=cache,
            head_size=3,
            num_kv_heads=2,
            get_attn_backend=lambda: NS(get_name=lambda: "FLASH_ATTN"),
        )
    }
    controller.kv_store = kv_store
    metadata["fa"].block_table = torch.tensor([[1, 2]])
    source = kv_store.capture(
        {"name": "K", "token_indices": [0, 5]}, "history", 0, metadata, 6
    )
    cache.fill_(-9)
    controller.after_forward(controller.before_forward(metadata, positions, counts))
    controller.runner.requests["target"].num_computed_tokens = 2
    controller.runner.input_batch.num_computed_tokens_cpu = [2]
    metadata["fa"].seq_lens = torch.tensor([5])
    metadata["fa"].query_start_loc = torch.tensor([0, 3])
    positions = torch.arange(2, 5)
    if wrong_offset:
        with pytest.raises(CompactionContractError, match="original next-query"):
            controller.before_forward(metadata, positions, {"target": 3})
        assert torch.all(states[1] == -1)
        assert torch.all(cache == -9)
        assert positions.tolist() == [2, 3, 4]
        return
    boundary = controller.before_forward(metadata, positions, {"target": 3})
    assert torch.all(states[1][1] == 7)
    assert positions.tolist() == [6, 7, 8]
    controller.after_forward(boundary)
    result = controller.results(["target"])["target"]
    assert result["fa_kv_imported"]
    assert result["kv_capture"]["digest"] == source["digest"]
    assert result["kv_capture"]["source_position_offset"] == 4
    assert result["kv_restore"]["query_position_offset"] == 4
    with pytest.raises(ValueError, match="Repeated selected-KV"):
        kv_store.validate_restore("roundtrip", "second-retention", 0, metadata, 2)
    # Offset remains owned by the live request after the receipt is delivered.
    controller.runner.requests["target"].num_computed_tokens = 5
    controller.runner.input_batch.num_computed_tokens_cpu = [5]
    positions = torch.tensor([5])
    controller.before_forward(metadata, positions, {"target": 1}, "FULL", 1)
    assert positions.item() == 9
    controller.finish_requests({"target"})
    assert not controller._position_rules


@pytest.mark.parametrize("family", ["gdn", "mamba2"])
def test_uninstrumented_fresh_one_token_decode_row_is_zeroed_at_runner_level(family):
    """Summary generation and validation natives carry no descriptor; a
    budget-limited 1-token first slice on the decode path must still start
    from a zero slot, while prefill rows and continuing rows are untouched."""
    controller, states, metadata, positions, counts = batch_controller(
        family,
        [("fresh", 0, 1, None, 3), ("continuing", 7, 1, None, 5)],
        num_decodes=2,
    )
    assert controller.before_forward(metadata, positions, counts, "FULL", 2) is None
    for state in states:
        assert torch.all(state[3] == 0)
        assert torch.all(state[5] == -1)
    assert controller.fresh_decode_rows_zeroed == 1
    assert controller.info()["fresh_decode_rows_zeroed"] == 1
    # A fresh multi-token slice takes the prefill path; its slot is left alone.
    controller2, states2, metadata2, positions2, counts2 = batch_controller(
        family, [("fresh_prefill", 0, 4, None, 2)]
    )
    assert controller2.before_forward(metadata2, positions2, counts2) is None
    for state in states2:
        assert torch.all(state[2] == -1)
    assert controller2.fresh_decode_rows_zeroed == 0
