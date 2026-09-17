# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU ownership/routing tests; these do not validate CUDA state consumption."""

from types import SimpleNamespace as NS

import pytest
import torch

from vllm.v1.worker.compaction import (
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
    assert receipt["snapshot_unchanged_after_restore"]
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
        ("scheduler_config", "max_num_seqs", 2),
        ("mamba_config", "state_quant_bits", 8),
        ("mamba_config", "state_trace_dir", "/tmp/trace"),
        ("model_config", "enforce_eager", False),
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
    controller.layers = {"layer0": ("gdn", NS(kv_cache=current.states))}
    controller.fa_layers = {"fa"}
    controller.metadata_types = {"gdn": NS}
    controller.runner = NS(
        execute_model_state=None,
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


def test_fresh_session_without_restore_rejects_a_decode_first_forward():
    """Recurrent slots are never zeroed on allocation; a fresh decode row would consume stale state."""
    store = SnapshotStore()
    op = BoundaryOperation(store, capture_name=None, restore_name=None, expected_prompt_tokens=1)
    with pytest.raises(CompactionContractError, match="prefill forward"):
        op.before("r1", 1, 0, 1, [slot(prefill=False)])
    # a prefill first forward is accepted, and so is a decode row when a snapshot is restored
    op2 = BoundaryOperation(store, capture_name=None, restore_name=None, expected_prompt_tokens=4)
    op2.before("r2", 4, 0, 4, [slot(prefill=True)])
    store.capture("H", [slot(value=2.0)], {"processed_prompt_tokens": 4})
    op3 = BoundaryOperation(store, capture_name=None, restore_name="H", expected_prompt_tokens=1)
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
