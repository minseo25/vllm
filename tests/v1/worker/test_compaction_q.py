# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for query export routing and in-worker compaction compute ops.

The attention custom op is modelled by calling each FA layer's
``compaction_q_export`` hook exactly as ``unified_attention_with_output`` does.
Methods-library calls go to small fakes injected through the registry (they
mirror the committed ``profiling.compaction_methods`` result types); one test
resolves the real package from the repository root so contract drift fails.
"""

import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from vllm.v1.worker import compaction_q
from vllm.v1.worker.compaction import (
    DESCRIPTOR_KEY,
    BoundaryOperation,
    CompactionContractError,
    NativeCompactionController,
    SnapshotStore,
)
from vllm.v1.worker.compaction_kv import NativeKVSnapshotStore
from vllm.v1.worker.compaction_q import (
    QExportStore,
    QueryExportPlan,
    ScoreStore,
    SelectionStore,
)

FA = ("fa0", "fa1")
HEAD, KV_HEADS, Q_HEADS, BLOCK = 3, 2, 4, 4
CTX = 6  # consumed cursor of the resident request in compute-op tests
LAGGING = 3  # the worker's num_computed_tokens copy after the last chunk
REPO_ROOT = Path(__file__).resolve().parents[5]


@pytest.fixture(autouse=True)
def _clear_methods():
    yield
    compaction_q.clear_methods()


def geometry(dtype=torch.float32, heads=Q_HEADS, kv_heads=KV_HEADS, layers=FA):
    return {
        name: {
            "num_heads": heads,
            "num_kv_heads": kv_heads,
            "head_size": HEAD,
            "dtype": dtype,
            "pinned": False,
        }
        for name in layers
    }


def fa_layer(index, name, dtype=torch.float32):
    cache = (
        torch.arange(8 * KV_HEADS * BLOCK * 2 * HEAD, dtype=torch.float32).reshape(
            8, KV_HEADS, BLOCK, 2 * HEAD
        )
        + 1000.0 * index
    ).to(dtype)
    return NS(
        layer_name=name,
        kv_cache=cache,
        head_size=HEAD,
        num_kv_heads=KV_HEADS,
        num_heads=Q_HEADS,
        dtype=dtype,
        impl=NS(scale=HEAD**-0.5),
        get_attn_backend=lambda: NS(get_name=lambda: "FLASH_ATTN"),
        compaction_q_export=None,
    )


def gdn_metadata(requests, num_decodes=0):
    count = len(requests)
    indices = torch.tensor([r[4] for r in requests])
    prefills = count - num_decodes
    return NS(
        num_prefills=prefills,
        num_decodes=num_decodes,
        num_spec_decodes=0,
        num_prefill_tokens=sum(r[2] for r in requests[num_decodes:]),
        num_decode_tokens=sum(r[2] for r in requests[:num_decodes]),
        prefill_state_indices=indices[num_decodes:] if prefills else None,
        non_spec_state_indices_tensor=indices,
        has_initial_state=torch.zeros(count, dtype=torch.bool) if prefills else None,
        prefill_has_initial_state=(
            torch.zeros(prefills, dtype=torch.bool) if prefills else None
        ),
    )


def make_controller(requests, *, consumed=None, dtype=torch.float32):
    """Requests: (id, cursor, query_count, descriptor|None, state_index).

    ``consumed`` seeds the controller's tracked cursors (what it saw consumed);
    the batch copies hold the request cursors given in ``requests``.
    """
    controller = object.__new__(NativeCompactionController)
    controller.store = SnapshotStore()
    controller.operation = None
    controller.operations = {}
    controller._descriptors = {}
    controller._retired_bindings = {}
    controller._seen_operation_ids = set()
    controller._position_rules = {}
    controller.layer_groups = {}
    states = (torch.full((8, 2, 3), -1.0), torch.full((8, 2, 4, 4), -1.0))
    controller.layers = {"layer0": ("gdn", NS(kv_cache=states))}
    controller.metadata_types = {"gdn": NS}
    controller.fa_layers = set(FA)
    controller.fa_layer_groups = {name: 0 for name in FA}
    layers = {name: fa_layer(i, name, dtype) for i, name in enumerate(FA)}
    kv_store = object.__new__(NativeKVSnapshotStore)
    kv_store._entries, kv_store._staged = {}, {}
    kv_store.layers = layers
    controller.kv_store = kv_store
    controller.q_store = QExportStore(FA, controller._export_geometry())
    controller.score_store = ScoreStore()
    controller.selection_store = SelectionStore()
    controller._export_plan = None
    controller._consumed = dict(consumed or {})
    controller._pending_consumed = None
    table = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32)
    controller.runner = NS(
        execute_model_state=None,
        device=torch.device("cpu"),
        input_batch=NS(
            num_reqs=len(requests),
            req_ids=[r[0] for r in requests],
            num_computed_tokens_cpu=[r[1] for r in requests],
            req_id_to_index={r[0]: i for i, r in enumerate(requests)},
            block_table=[NS(get_cpu_tensor=lambda: table)],
        ),
        requests={
            rid: NS(
                mm_features=[],
                prompt_embeds=None,
                lora_request=None,
                num_computed_tokens=cursor,
                num_prompt_tokens=(
                    desc["expected_prompt_tokens"] if desc else cursor + count
                ),
                sampling_params=NS(extra_args={DESCRIPTOR_KEY: desc} if desc else {}),
            )
            for rid, cursor, count, desc, _ in requests
        },
    )
    return controller, layers, states


def advance(controller, requests):
    batch = controller.runner.input_batch
    batch.num_reqs = len(requests)
    batch.req_ids = [r[0] for r in requests]
    batch.num_computed_tokens_cpu = [r[1] for r in requests]
    batch.req_id_to_index = {r[0]: i for i, r in enumerate(requests)}
    for rid, cursor, _, _, _ in requests:
        controller.runner.requests[rid].num_computed_tokens = cursor


def forward_inputs(requests, controller, num_decodes=0):
    table = controller.runner.input_batch.block_table[0].get_cpu_tensor()
    starts = [0]
    for r in requests:
        starts.append(starts[-1] + r[2])
    metadata = {"layer0": gdn_metadata(requests, num_decodes)}
    for name in FA:
        metadata[name] = NS(
            seq_lens=torch.tensor([r[1] + r[2] for r in requests]),
            query_start_loc=torch.tensor(starts),
            block_table=table,
            num_actual_tokens=starts[-1],
        )
    positions = torch.cat([torch.arange(r[1], r[1] + r[2]) for r in requests])
    return metadata, positions, {r[0]: r[2] for r in requests}


def query_batch(total, dtype=torch.float32):
    return (
        torch.arange(total * Q_HEADS * HEAD, dtype=torch.float32)
        .reshape(total, Q_HEADS, HEAD)
        .to(dtype)
    )


def run_attention(layers, query, metadata):
    """Model side of ``unified_attention_with_output``: call installed hooks."""
    for name, layer in layers.items():
        hook = layer.compaction_q_export
        if hook is not None:
            hook(layer, query, metadata[name])


def hooks_installed(layers):
    return [name for name in FA if layers[name].compaction_q_export is not None]


# ------------------------------------------------------------- descriptor


@pytest.mark.parametrize(
    "export_q,message",
    [
        ({"name": "Q"}, "name and token_range"),
        ({"name": "Q", "token_range": [1, 3], "extra": 1}, "name and token_range"),
        ("Q", "name and token_range"),
        ({"name": "", "token_range": [0, 2]}, "nonempty"),
        ({"name": "Q", "token_range": [2, 2]}, "token_range"),
        ({"name": "Q", "token_range": [0, 6]}, "token_range"),
        ({"name": "Q", "token_range": [0, True]}, "token_range"),
        ({"name": "Q", "token_range": [0]}, "token_range"),
    ],
)
def test_export_q_descriptor_is_validated_when_the_operation_is_created(
    export_q, message
):
    with pytest.raises(CompactionContractError, match=message):
        BoundaryOperation(
            SnapshotStore(),
            capture_name=None,
            restore_name=None,
            expected_prompt_tokens=5,
            export_q=export_q,
        )


def test_export_q_range_must_lie_within_the_new_tokens_of_a_continuation():
    with pytest.raises(CompactionContractError, match="start_cursor <= start"):
        BoundaryOperation(
            SnapshotStore(),
            capture_name=None,
            restore_name=None,
            expected_prompt_tokens=5,
            start_cursor=2,
            export_q={"name": "Q", "token_range": (1, 3)},
        )
    operation = BoundaryOperation(
        SnapshotStore(),
        capture_name=None,
        restore_name=None,
        expected_prompt_tokens=5,
        start_cursor=2,
        export_q={"name": "Q", "token_range": (2, 5)},
    )
    assert operation.export_q == {"name": "Q", "token_range": [2, 5]}
    assert operation.q_export_receipt is None and operation.q_export_chunks == []


# ------------------------------------------------------------- export flow


def test_mixed_batch_exports_only_the_flagged_requests_rows_for_every_layer():
    flagged = dict(
        operation_id="f",
        expected_prompt_tokens=3,
        export_q={"name": "Q", "token_range": [1, 3]},
    )
    requests = [("other", 0, 2, None, 2), ("flag", 0, 3, flagged, 3)]
    controller, layers, _ = make_controller(requests)
    metadata, positions, counts = forward_inputs(requests, controller)
    boundary = controller.before_forward(metadata, positions, counts, "PIECEWISE", 2)
    assert hooks_installed(layers) == list(FA)
    opened = controller.q_exports()["exports"]["Q"]
    assert opened["complete"] is False and opened["request_id"] == "flag"
    # Buffers exist before the forward: geometry-derived size, allocation timed.
    assert opened["bytes"] == 2 * opened["bytes_per_token"]
    assert opened["bytes_per_token"] == len(FA) * Q_HEADS * HEAD * 4
    assert opened["allocation_seconds"] >= 0.0 and opened["copy_seconds"] == 0.0
    query = query_batch(5)
    run_attention(layers, query, metadata)
    controller.after_forward(boundary)
    assert hooks_installed(layers) == [] and controller._export_plan is None
    # Consumed cursors are tracked although the worker copies still lag.
    assert controller._consumed == {"other": 2, "flag": 3}
    assert controller.runner.input_batch.num_computed_tokens_cpu == [0, 0]
    export = controller.q_store.describe("Q")
    assert export["complete"] and export["rows"] == export["rows_exported"] == 2
    assert export["dtype"] == "torch.float32" and export["pinned"] is False
    assert export["shapes"] == {name: [2, Q_HEADS, HEAD] for name in FA}
    assert export["layouts"] == {
        name: {
            "num_heads": Q_HEADS,
            "num_kv_heads": KV_HEADS,
            "head_size": HEAD,
            "group_size": Q_HEADS // KV_HEADS,
        }
        for name in FA
    }
    assert export["query_convention"]["stage"] == "post_qk_norm_post_rope_unscaled"
    assert export["query_convention"]["head_order"] == "contiguous_gqa_blocks"
    chunk = export["chunks"][0]
    assert (chunk["batch_row"], chunk["graph_mode"]) == (1, "PIECEWISE")
    assert (chunk["cursor_start"], chunk["cursor_end"], chunk["rows"]) == (1, 3, 2)
    assert chunk["hook_seconds"] >= 0.0 and export["copy_seconds"] >= 0.0
    tensors = controller.q_store.tensors("Q")
    for name in FA:
        # "flag" occupies batch rows 2..4; its cursors 1..2 are rows 3 and 4.
        assert torch.equal(tensors[name], query[3:5])
    assert controller.q_store.positions("Q").tolist() == [1, 2]
    assert controller.q_store.cursors("Q").tolist() == [1, 2]
    assert torch.equal(query, query_batch(5))  # the hook never mutates queries
    receipt = controller.results(["f"])["f"]
    assert receipt["query_exported"] and receipt["q_export"]["complete"]
    assert receipt["export_q"] == {"name": "Q", "token_range": [1, 3]}
    assert receipt["q_export_chunks"][0]["rows"] == 2
    assert receipt["query_export"] and "kv_score" in receipt["compute_ops"]
    assert receipt["compute_ops"] == [
        "kv_capture",
        "kv_subset",
        "kv_score",
        "kv_select",
        "kv_fit_am",
    ]
    assert (
        receipt["store_ops"]
        == controller.info()["store_ops"]
        == [
            "kv_subset",
            "kv_selection_drop",
            "kv_describe",
            "q_describe",
            "kv_drop",
            "q_drop",
            "score_drop",
        ]
    )
    # Every advertised op is a real controller method.
    for name in receipt["compute_ops"] + receipt["store_ops"]:
        assert callable(getattr(controller, name))


def test_padded_hook_call_exports_only_the_real_rows():
    desc = dict(
        operation_id="p",
        expected_prompt_tokens=3,
        export_q={"name": "Q", "token_range": [0, 3]},
    )
    requests = [("r", 0, 3, desc, 2)]
    controller, layers, _ = make_controller(requests)
    metadata, positions, counts = forward_inputs(requests, controller)
    boundary = controller.before_forward(metadata, positions, counts)
    padded = torch.cat([query_batch(3), torch.full((5, Q_HEADS, HEAD), 7.0)])
    run_attention(layers, padded, metadata)
    controller.after_forward(boundary)
    for name in FA:
        assert torch.equal(controller.q_store.tensors("Q")[name], query_batch(3))


def test_multi_chunk_prefill_fills_the_range_across_forwards_then_decode_is_silent():
    desc = dict(
        operation_id="m",
        expected_prompt_tokens=5,
        export_q={"name": "Q", "token_range": [1, 4]},
    )
    requests = [("r", 0, 2, desc, 2)]
    controller, layers, _ = make_controller(requests)
    metadata, positions, counts = forward_inputs(requests, controller)
    boundary = controller.before_forward(metadata, positions, counts)
    first = query_batch(2) + 100
    run_attention(layers, first, metadata)
    controller.after_forward(boundary)
    entry = controller.q_store.describe("Q")
    assert entry["rows_exported"] == 1 and not entry["complete"]
    assert entry["first_position"] is None
    with pytest.raises(CompactionContractError, match="incomplete"):
        controller.q_store.tensors("Q")
    requests = [("r", 2, 3, desc, 2)]
    advance(controller, requests)
    metadata, positions, counts = forward_inputs(requests, controller)
    boundary = controller.before_forward(metadata, positions, counts)
    second = query_batch(3) + 200
    run_attention(layers, second, metadata)
    controller.after_forward(boundary)
    entry = controller.q_store.describe("Q")
    assert entry["complete"] and entry["rows_exported"] == 3
    assert [c["rows"] for c in entry["chunks"]] == [1, 2]
    assert (entry["first_position"], entry["last_position"]) == (1, 3)
    tensors = controller.q_store.tensors("Q")
    for name in FA:
        assert torch.equal(tensors[name][0], first[1])
        assert torch.equal(tensors[name][1:], second[0:2])
    assert controller._consumed["r"] == 5
    # A FULL-graph decode step after the prompt installs nothing and is allowed.
    requests = [("r", 5, 1, desc, 2)]
    advance(controller, requests)
    metadata, positions, counts = forward_inputs(requests, controller, num_decodes=1)
    boundary = controller.before_forward(metadata, positions, counts, "FULL", 1)
    assert boundary is not None and hooks_installed(layers) == []
    assert controller._export_plan is None
    run_attention(layers, query_batch(1), metadata)
    controller.after_forward(boundary)
    assert controller._consumed["r"] == 6
    result = controller.results(["m"])["m"]
    assert len(result["q_export_chunks"]) == 2 and result["q_export"]["complete"]
    assert result["forward_calls"] == 3
    controller.finish_requests({"r"})
    assert "r" not in controller._consumed


def test_full_graph_refusal_precedes_co_scheduled_restores_and_state_writes():
    carry = dict(
        operation_id="c", expected_prompt_tokens=3, restore_name="H", restore_at=0
    )
    flagged = dict(
        operation_id="g",
        expected_prompt_tokens=3,
        export_q={"name": "Q", "token_range": [0, 3]},
    )
    requests = [("carry", 0, 3, carry, 2), ("flag", 0, 3, flagged, 3)]
    controller, layers, states = make_controller(requests)
    donor = torch.full((3, 2, 3), 7.0), torch.full((3, 2, 4, 4), 7.0)
    from vllm.v1.worker.compaction import resolve_slot

    slot_md = NS(
        num_prefills=1,
        num_decodes=0,
        num_spec_decodes=0,
        prefill_state_indices=torch.tensor([1]),
        non_spec_state_indices_tensor=torch.tensor([1]),
        has_initial_state=torch.tensor([False]),
        prefill_has_initial_state=torch.tensor([False]),
    )
    controller.store.capture("H", [resolve_slot("layer0", "gdn", donor, slot_md)], {})
    metadata, positions, counts = forward_inputs(requests, controller)
    with pytest.raises(CompactionContractError, match="FULL"):
        controller.before_forward(metadata, positions, counts, "FULL", 2)
    # Refused before the co-scheduled state restore: nothing staged or written.
    assert torch.all(states[1] == -1) and torch.all(states[0] == -1)
    assert not metadata["layer0"].has_initial_state.any()
    assert controller.store._staged == {}
    assert hooks_installed(layers) == [] and controller._export_plan is None
    assert "FULL" in controller.operations["g"].failed
    assert "FULL" in controller.operations["c"].failed
    assert controller.q_store.describe("Q")["rows_exported"] == 0
    assert controller._pending_consumed is None and controller._consumed == {}


def test_export_requires_query_start_loc_to_match_the_batch_token_start():
    desc = dict(
        operation_id="s",
        expected_prompt_tokens=3,
        export_q={"name": "Q", "token_range": [0, 3]},
    )
    requests = [("other", 0, 2, None, 2), ("r", 0, 3, desc, 3)]
    controller, layers, _ = make_controller(requests)
    metadata, positions, counts = forward_inputs(requests, controller)
    metadata["fa1"].query_start_loc = torch.tensor([0, 1, 4])
    metadata["fa1"].seq_lens = torch.tensor([1, 3])
    with pytest.raises(CompactionContractError, match="query_start_loc"):
        controller.before_forward(metadata, positions, counts)
    assert hooks_installed(layers) == []


def test_after_forward_fails_closed_when_a_layer_skipped_the_hook():
    desc = dict(
        operation_id="m",
        expected_prompt_tokens=3,
        export_q={"name": "Q", "token_range": [0, 3]},
        capture_kv={"name": "K", "token_indices": [0]},
    )
    requests = [("r", 0, 3, desc, 2)]
    controller, layers, _ = make_controller(requests)
    metadata, positions, counts = forward_inputs(requests, controller)
    boundary = controller.before_forward(metadata, positions, counts)
    query = query_batch(3)
    layers["fa0"].compaction_q_export(layers["fa0"], query, metadata["fa0"])
    with pytest.raises(CompactionContractError, match="did not run.*fa1"):
        controller.after_forward(boundary)
    assert hooks_installed(layers) == [] and controller._export_plan is None
    assert "did not run" in controller.operations["m"].failed
    assert controller.q_store.describe("Q")["rows_exported"] == 0
    # The forward itself ran: the consumed cursor was still committed.
    assert controller._consumed == {"r": 3}
    with pytest.raises(CompactionContractError, match="did not run"):
        controller.results(["m"])
    # A failed operation must be closable: its export buffers can be dropped.
    assert controller.q_drop("Q")["exports"] == {}
    assert controller.kv_store.list()["snapshots"] == {}


@pytest.mark.parametrize("fault", ["twice", "foreign", "short", "actual", "heads"])
def test_hook_rejects_inconsistent_calls_and_fails_the_operation(fault):
    desc = dict(
        operation_id="h",
        expected_prompt_tokens=3,
        export_q={"name": "Q", "token_range": [0, 3]},
    )
    requests = [("r", 0, 3, desc, 2)]
    controller, layers, _ = make_controller(requests)
    metadata, positions, counts = forward_inputs(requests, controller)
    boundary = controller.before_forward(metadata, positions, counts)
    hook = layers["fa0"].compaction_q_export
    query = query_batch(3)
    if fault == "twice":
        hook(layers["fa0"], query, metadata["fa0"])
        call = lambda: hook(layers["fa0"], query, metadata["fa0"])  # noqa: E731
        message = "twice"
    elif fault == "foreign":
        foreign = NS(layer_name="zz", num_heads=Q_HEADS, head_size=HEAD)
        call = lambda: hook(foreign, query, metadata["fa0"])  # noqa: E731
        message = "unexpected layer"
    elif fault == "short":
        call = lambda: hook(layers["fa0"], query[:2], metadata["fa0"])  # noqa: E731
        message = "shorter"
    elif fault == "actual":
        wrong = NS(num_actual_tokens=4)
        call = lambda: hook(layers["fa0"], query, wrong)  # noqa: E731
        message = "num_actual_tokens"
    else:
        bad = torch.zeros(3, Q_HEADS - 1, HEAD)
        call = lambda: hook(layers["fa0"], bad, metadata["fa0"])  # noqa: E731
        message = "head layout"
    with pytest.raises(CompactionContractError, match=message):
        call()
    with pytest.raises(CompactionContractError, match=message):
        controller.after_forward(boundary)
    assert hooks_installed(layers) == []
    assert message in controller.operations["h"].failed


def test_export_names_are_reserved_at_arm_and_dropped_only_when_inactive():
    requests = [("r", 0, 3, None, 2)]
    controller, layers, _ = make_controller(requests)
    controller.q_store.open("Q", token_range=[0, 3])
    with pytest.raises(CompactionContractError, match="reused"):
        controller.arm(
            expected_prompt_tokens=3, export_q={"name": "Q", "token_range": [0, 3]}
        )
    assert controller.operation is None
    controller.q_store.drop("Q")
    armed = controller.arm(
        expected_prompt_tokens=3, export_q={"name": "Q", "token_range": [0, 3]}
    )
    assert armed["export_q"] == {"name": "Q", "token_range": [0, 3]}
    with pytest.raises(CompactionContractError, match="active operation"):
        controller.q_drop("Q")
    metadata, positions, counts = forward_inputs(requests, controller)
    boundary = controller.before_forward(metadata, positions, counts)
    run_attention(layers, query_batch(3), metadata)
    controller.after_forward(boundary)
    receipt = controller.result()
    assert receipt["q_export"]["complete"]
    assert receipt["q_export"]["request_id"] == "r"
    assert controller.q_drop("Q")["exports"] == {}


def test_failed_model_forward_uninstalls_the_hook_and_keeps_rows_uncommitted():
    desc = dict(
        operation_id="k",
        expected_prompt_tokens=3,
        export_q={"name": "Q", "token_range": [0, 3]},
    )
    requests = [("r", 0, 3, desc, 2)]
    controller, layers, _ = make_controller(requests)
    metadata, positions, counts = forward_inputs(requests, controller)
    controller.before_forward(metadata, positions, counts)
    assert hooks_installed(layers) == list(FA)
    assert controller._pending_consumed == {"r": 3}
    controller.fail_forward(RuntimeError("kernel failed"))
    assert hooks_installed(layers) == [] and controller._export_plan is None
    assert controller.q_store.describe("Q")["rows_exported"] == 0
    assert controller._pending_consumed is None and controller._consumed == {}


def test_consumed_cursor_tracking_covers_uninstrumented_rows_and_clears_on_finish():
    requests = [("plain", 4, 2, None, 2)]
    controller, _, _ = make_controller(requests)
    metadata, positions, counts = forward_inputs(requests, controller)
    assert controller.before_forward(metadata, positions, counts) is None
    assert controller._pending_consumed == {"plain": 6}
    controller.after_forward(None)
    assert controller._consumed == {"plain": 6}
    controller.finish_requests({"plain"})
    assert controller._consumed == {}


# ------------------------------------------------------------ store units


def test_export_store_allocates_from_geometry_and_validates_writes():
    with pytest.raises(CompactionContractError, match="cover exactly"):
        QExportStore(FA, {"fa0": geometry()["fa0"]})
    with pytest.raises(CompactionContractError, match="Invalid query export geometry"):
        QExportStore(FA, {n: {**geometry()[n], "num_heads": 3} for n in FA})
    with pytest.raises(CompactionContractError, match="needs num_heads"):
        QExportStore(FA, {n: {"num_heads": 2} for n in FA})
    store = QExportStore(FA, geometry(heads=2, kv_heads=1))
    assert store.bytes_per_token() == 2 * 2 * 3 * 4
    with pytest.raises(CompactionContractError, match="start < end"):
        store.open("bad", token_range=[3, 3])
    opened = store.open("Q", token_range=[2, 4])
    assert opened["bytes"] == 2 * store.bytes_per_token()
    assert opened["shapes"] == {n: [2, 2, 3] for n in FA}
    assert opened["layouts"]["fa0"]["group_size"] == 2
    assert opened["allocation_seconds"] >= 0.0
    with pytest.raises(CompactionContractError, match="reused"):
        store.open("Q", token_range=[0, 1])
    with pytest.raises(CompactionContractError, match="Unsupported query dtype"):
        store.write("Q", "fa0", torch.zeros(1, 2, 3, dtype=torch.float8_e4m3fn), 2)
    with pytest.raises(CompactionContractError, match="dtype .* differs"):
        store.write("Q", "fa0", torch.zeros(1, 2, 3, dtype=torch.float16), 2)
    with pytest.raises(CompactionContractError, match="head layout .* differs"):
        store.write("Q", "fa0", torch.zeros(1, 2, 4), 2)
    with pytest.raises(CompactionContractError, match="inconsistent with the geometry"):
        store.write(
            "Q",
            "fa0",
            torch.ones(2, 2, 3),
            2,
            layout={"num_heads": 2, "num_kv_heads": 2, "head_size": 3},
        )
    with pytest.raises(CompactionContractError, match="outside token_range"):
        store.write("Q", "fa0", torch.zeros(3, 2, 3), 2)
    with pytest.raises(CompactionContractError, match="Unexpected query export layer"):
        store.write("Q", "zz", torch.zeros(1, 2, 3), 2)
    store.write(
        "Q",
        "fa0",
        torch.ones(2, 2, 3),
        2,
        layout={"num_heads": 2, "num_kv_heads": 1, "head_size": 3},
    )
    with pytest.raises(CompactionContractError, match="missed FA layers"):
        store.commit(
            "Q",
            cursor_start=2,
            cursor_end=4,
            positions=torch.tensor([2, 3]),
            layers_written={"fa0"},
            receipt={},
        )
    store.write("Q", "fa1", torch.ones(2, 2, 3), 2)
    store.commit(
        "Q",
        cursor_start=2,
        cursor_end=4,
        positions=torch.tensor([7, 8]),
        layers_written=set(FA),
        receipt={"hook_seconds": 0.25},
    )
    with pytest.raises(CompactionContractError, match="already exported"):
        store.write("Q", "fa0", torch.ones(1, 2, 3), 3)
    described = store.describe("Q")
    assert described["copy_seconds"] == 0.25 and described["complete"]
    assert store.positions("Q").tolist() == [7, 8]
    assert store.list()["total_host_bytes"] == 2 * 2 * 2 * 3 * 4
    with pytest.raises(CompactionContractError, match="Unknown query export"):
        store.describe("nope")


def test_bf16_export_path_writes_into_preallocated_bf16_buffers():
    desc = dict(
        operation_id="b",
        expected_prompt_tokens=2,
        export_q={"name": "Q", "token_range": [0, 2]},
    )
    requests = [("r", 0, 2, desc, 2)]
    controller, layers, _ = make_controller(requests, dtype=torch.bfloat16)
    assert controller.q_store.geometry["fa0"]["dtype"] == torch.bfloat16
    metadata, positions, counts = forward_inputs(requests, controller)
    boundary = controller.before_forward(metadata, positions, counts)
    opened = controller.q_store.describe("Q")
    assert opened["dtype"] == "torch.bfloat16" and opened["pinned"] is False
    assert opened["bytes"] == 2 * len(FA) * Q_HEADS * HEAD * 2
    with pytest.raises(CompactionContractError, match="dtype .* differs"):
        layers["fa0"].compaction_q_export(
            layers["fa0"], query_batch(2), metadata["fa0"]
        )
    controller.fail_forward(RuntimeError("stop"))
    assert controller.q_drop("Q")["exports"] == {}
    # A matching bf16 query lands in the preallocated bf16 buffers.
    controller, layers, _ = make_controller(requests, dtype=torch.bfloat16)
    metadata, positions, counts = forward_inputs(requests, controller)
    boundary = controller.before_forward(metadata, positions, counts)
    query = query_batch(2, torch.bfloat16)
    run_attention(layers, query, metadata)
    controller.after_forward(boundary)
    tensors = controller.q_store.tensors("Q")
    for name in FA:
        assert tensors[name].dtype == torch.bfloat16
        assert torch.equal(tensors[name], query)


def test_export_plan_rejects_rows_outside_the_token_batch():
    store = QExportStore(FA, geometry())
    store.open("Q", token_range=[0, 4])
    item = compaction_q.ExportItem(
        name="Q",
        operation=None,
        row_start=2,
        row_end=6,
        cursor_start=0,
        cursor_end=4,
        positions=torch.arange(4),
    )
    with pytest.raises(CompactionContractError, match="exceed the token batch"):
        QueryExportPlan(store, FA, [item], total_tokens=5)
    with pytest.raises(CompactionContractError, match="without rows"):
        QueryExportPlan(store, FA, [], total_tokens=5)


# ------------------------------------------------------------- compute ops


def score_result(scores, variant, **params):
    return NS(
        scores=scores, variant=variant, params=params, warnings=["fake-score-warning"]
    )


class FakeScores:
    """Mirrors ``compaction_methods.scores``: returns ScoreResult-like objects."""

    def __init__(self, bare=False):
        self.calls = []
        self.bare = bare

    def h2o_scores(
        self,
        q,
        k,
        *,
        scale,
        causal,
        q_positions,
        k_positions,
        chunk=2048,
        memory_budget_bytes=None,
    ):
        self.calls.append(
            dict(
                method="h2o",
                q=q.clone(),
                k=k.clone(),
                scale=scale,
                causal=causal,
                q_positions=q_positions.clone(),
                k_positions=k_positions.clone(),
                chunk=chunk,
                memory_budget_bytes=memory_budget_bytes,
            )
        )
        tokens, heads = k.shape[0], k.shape[1]
        scores = (
            torch.arange(tokens, dtype=torch.float32)[None, :].repeat(heads, 1)
            + 0.5 * torch.arange(heads, dtype=torch.float32)[:, None]
        )
        if self.bare:
            return scores
        return score_result(
            scores,
            "h2o_uniform_prefill",
            chunk=chunk,
            memory_budget_bytes=memory_budget_bytes,
        )

    def kvzip_scores(
        self,
        q_ref,
        k,
        *,
        scale,
        k_ref,
        chunk=2048,
        normalisation="paper",
        memory_budget_bytes=None,
        repeat_prompt=None,
    ):
        self.calls.append(
            dict(
                method="kvzip",
                q=q_ref.clone(),
                k=k.clone(),
                k_ref=None if k_ref is None else k_ref.clone(),
                scale=scale,
                chunk=chunk,
                normalisation=normalisation,
                memory_budget_bytes=memory_budget_bytes,
                repeat_prompt=repeat_prompt,
            )
        )
        variant = "kvzip_uniform_perlayer_fullctx"
        if normalisation == "context_only":
            variant += "_ctxonly"
        return score_result(
            k.float().abs().sum(-1).T,
            variant,
            chunk=chunk,
            normalisation=normalisation,
            repeat_prompt=repeat_prompt,
        )


class FakeSelect:
    def __init__(self):
        self.calls = []

    def uniform_token_budget(
        self, scores, budget_tokens, *, protected, policy, aggregate
    ):
        self.calls.append(dict(budget=budget_tokens, protected=list(protected)))
        out = []
        for layer_scores in scores:
            agg = (
                layer_scores.max(0).values
                if aggregate == "max"
                else layer_scores.mean(0)
            )
            order = torch.argsort(agg, descending=True, stable=True).tolist()
            chosen = list(protected)
            for i in order:
                if len(chosen) >= budget_tokens:
                    break
                if i not in chosen:
                    chosen.append(i)
            out.append(sorted(chosen))
        return [out[0]] * len(out) if policy == "shared" else out


class FakeAMResult:
    """Mirrors ``compaction_methods.am.AMResult`` for the shared case."""

    def __init__(
        self, chosen, k, v, fixed, *, chunk, budget, max_fit_workspace_bytes=None
    ):
        index = torch.tensor(chosen)
        heads = k.shape[1]
        self.indices = index[None, :].repeat(heads, 1)  # [Hkv, t]
        self.k_c = k[index].transpose(0, 1).contiguous()  # [Hkv, t, D]
        values = v[index] * 2
        fixed_rows = torch.tensor([c in fixed for c in chosen], dtype=torch.bool)
        values[fixed_rows] = v[index][fixed_rows]  # frozen frame rows keep V
        self.v_c = values.transpose(0, 1).contiguous()
        self.beta = None
        self.fixed_mask = fixed_rows[None, :].repeat(heads, 1)
        self.shared = True
        self.warnings = ["fake-am-warning"]
        policy = "fixed" if fixed else "refit"
        self.variant = (
            f"fake_am_rmskeys_ols_nobias_uniform_offpolicy_frame{policy}_chol64"
        )
        self.diagnostics = {
            "frame_policy": policy,
            "n_fixed": len(fixed),
            "n_protected": len(chosen) - len(fixed) if policy == "refit" else 0,
            "solver": "cholesky",
            "accumulate_dtype": "torch.float64",
            "compute_dtype": "torch.float32",
            "chunk": chunk,
            "identity_shortcut": budget == k.shape[0],
            "output_error_after_rel": [0.0] * heads,
            "max_fit_workspace_bytes": max_fit_workspace_bytes,
            "fit_workspace_admission": (
                None
                if budget == k.shape[0]
                else {
                    "additional_live_tensor_lower_bound_bytes": 8 * budget * budget,
                    "checked_capacity_bytes": max_fit_workspace_bytes,
                    "scope": "lower_bound_only_not_peak_certification",
                }
            ),
            "warnings": [],
        }

    def token_indices(self):
        return self.indices[0].tolist()

    def cache_layout(self):
        return (
            self.k_c.transpose(0, 1).contiguous(),
            self.v_c.transpose(0, 1).contiguous(),
        )


class FakeAM:
    def __init__(self):
        self.calls = []

    def compact(
        self,
        q_ref,
        k,
        v,
        budget,
        *,
        bias,
        head_budget,
        protected,
        fixed,
        scale,
        ridge,
        chunk=2048,
        memory_budget_bytes=None,
        max_fit_workspace_bytes=None,
    ):
        self.calls.append(
            dict(
                budget=budget,
                bias=bias,
                head_budget=head_budget,
                protected=list(protected),
                fixed=list(fixed),
                scale=scale,
                ridge=ridge,
                chunk=chunk,
                memory_budget_bytes=memory_budget_bytes,
                max_fit_workspace_bytes=max_fit_workspace_bytes,
                v_dtype=v.dtype,
            )
        )
        chosen = list(protected) + list(fixed)
        for i in range(k.shape[0] - 1, -1, -1):
            if len(chosen) >= budget:
                break
            if i not in chosen:
                chosen.append(i)
        return FakeAMResult(
            sorted(chosen),
            k,
            v,
            list(fixed),
            chunk=chunk,
            budget=budget,
            max_fit_workspace_bytes=max_fit_workspace_bytes,
        )


@pytest.fixture
def fakes():
    bundle = NS(scores=FakeScores(), select=FakeSelect(), am=FakeAM())
    compaction_q.register_methods(
        scores=bundle.scores, select=bundle.select, am=bundle.am
    )
    return bundle


def resident_controller():
    """One resident request whose worker cursor copy lags the consumed cursor."""
    controller, layers, _ = make_controller(
        [("ctx", LAGGING, 1, None, 2)], consumed={"ctx": CTX}
    )
    return controller, layers


def fill_export(controller, name, token_range, base=0.0, request_id="ctx"):
    store = controller.q_store
    store.open(name, token_range=token_range, request_id=request_id)
    rows = token_range[1] - token_range[0]
    queries = {}
    for i, layer in enumerate(FA):
        queries[layer] = query_batch(rows) + base + 10.0 * i
        store.write(name, layer, queries[layer], token_range[0])
    store.commit(
        name,
        cursor_start=token_range[0],
        cursor_end=token_range[1],
        positions=torch.arange(*token_range),
        layers_written=set(FA),
        receipt={"test": True},
    )
    return queries


def cache_rows(layer, tokens):
    blocks = torch.tensor([[1, 2, 3, 4, 5, 6][t // BLOCK] for t in tokens])
    offsets = torch.tensor([t % BLOCK for t in tokens])
    return layer.kv_cache[blocks, :, offsets, :]


def test_resident_ops_use_the_tracked_cursor_and_require_the_expected_cursor(fakes):
    controller, layers = resident_controller()
    fill_export(controller, "Q", [2, 6])
    assert controller.runner.input_batch.num_computed_tokens_cpu == [LAGGING]
    receipt = controller.kv_score(
        "S", q_export="Q", method="h2o", request_id="ctx", expected_cursor=CTX
    )
    # The default key range covers the consumed prefix, including the last chunk.
    assert receipt["source"]["key_range"] == [0, CTX]
    assert (
        receipt["source"]["cursor"] == CTX
        and receipt["source"]["expected_cursor"] == CTX
    )
    assert fakes.scores.calls[0]["k"].shape[0] == CTX
    with pytest.raises(CompactionContractError, match="expected_cursor 5 differs"):
        controller.kv_score(
            "S2", q_export="Q", method="h2o", request_id="ctx", expected_cursor=5
        )
    with pytest.raises(CompactionContractError, match="required for resident"):
        controller.kv_score("S2", q_export="Q", method="h2o", request_id="ctx")
    with pytest.raises(CompactionContractError, match="must be an integer"):
        controller.kv_capture(
            {"name": "c", "token_indices": [0]}, "ctx", expected_cursor=None
        )
    controller._consumed.pop("ctx")
    with pytest.raises(CompactionContractError, match="No consumed cursor"):
        controller.kv_score(
            "S2", q_export="Q", method="h2o", request_id="ctx", expected_cursor=CTX
        )
    controller._consumed["ctx"] = LAGGING - 1
    with pytest.raises(CompactionContractError, match="behind the worker copies"):
        controller.kv_capture(
            {"name": "c", "token_indices": [0]}, "ctx", expected_cursor=LAGGING - 1
        )


def test_kv_score_h2o_reads_cache_keys_and_stores_float32_cpu_scores(fakes):
    controller, layers = resident_controller()
    queries = fill_export(controller, "Q", [2, 6])
    receipt = controller.kv_score(
        "S",
        q_export="Q",
        method="h2o",
        request_id="ctx",
        expected_cursor=CTX,
        params={"chunk": 8},
    )
    assert receipt["method"] == "h2o" and receipt["variant"] == "h2o_uniform_prefill"
    assert receipt["params"]["chunk"] == 8
    assert receipt["params"]["blocking"] == {name: {"chunk": 8} for name in FA}
    assert receipt["params"]["memory_budget"]["policy"] == "default_1GiB"
    assert receipt["library_params"] == {
        name: {"chunk": 8, "memory_budget_bytes": None} for name in FA
    }
    assert receipt["shapes"] == {name: [KV_HEADS, 6] for name in FA}
    assert receipt["source"]["kind"] == "resident_request"
    assert receipt["source"]["key_range"] == [0, 6]
    assert receipt["source"]["q_token_range"] == [2, 6]
    assert receipt["source"]["scale"] == {name: HEAD**-0.5 for name in FA}
    assert receipt["source"]["k_ref"] is None
    assert receipt["source"]["normalisation"] == "causal_over_scored_keys"
    assert receipt["source"]["dropped_prefix_keys"] is None
    assert (
        receipt["source"]["query_convention"]["head_order"] == "contiguous_gqa_blocks"
    )
    assert receipt["seconds"] >= 0.0
    tensors = controller.score_store.tensors("S")
    for name in FA:
        assert tensors[name].dtype == torch.float32
        assert tensors[name].device.type == "cpu"
        assert tensors[name].shape == (KV_HEADS, 6)
    assert [c["method"] for c in fakes.scores.calls] == ["h2o", "h2o"]
    for call, name in zip(fakes.scores.calls, FA):
        assert torch.equal(call["k"], cache_rows(layers[name], range(6))[..., :HEAD])
        assert torch.equal(call["q"], queries[name])
        assert (call["scale"], call["causal"], call["chunk"]) == (HEAD**-0.5, True, 8)
        assert call["memory_budget_bytes"] is None
        assert call["q_positions"].tolist() == [2, 3, 4, 5]
        assert call["k_positions"].tolist() == list(range(6))
    assert controller.score_store.key_token_indices("S") == {
        name: list(range(6)) for name in FA
    }
    listed = controller.scores()["scores"]["S"]
    assert listed["keys_per_layer"] == {name: 6 for name in FA}
    assert "layers" not in listed and listed["variant"] == "h2o_uniform_prefill"
    assert listed["library_warnings"] == {name: ["fake-score-warning"] for name in FA}
    # Without an explicit chunk the memory budget is forwarded instead.
    budgeted = controller.kv_score(
        "B", q_export="Q", method="h2o", request_id="ctx", expected_cursor=CTX
    )
    assert budgeted["params"]["blocking"] == {
        name: {"memory_budget_bytes": 1 << 30} for name in FA
    }
    assert fakes.scores.calls[-1]["memory_budget_bytes"] == 1 << 30
    explicit = controller.kv_score(
        "E",
        q_export="Q",
        method="h2o",
        request_id="ctx",
        expected_cursor=CTX,
        params={"memory_budget_bytes": 4096},
    )
    assert explicit["params"]["memory_budget"] == {
        "bytes": 4096,
        "policy": "caller",
        "free_bytes": None,
    }
    assert fakes.scores.calls[-1]["memory_budget_bytes"] == 4096
    assert controller.score_drop("S")["scores"].keys() == {"B", "E"}


def test_bare_score_tensors_are_accepted_and_named_by_method(fakes):
    compaction_q.register_methods(scores=FakeScores(bare=True))
    controller, layers = resident_controller()
    fill_export(controller, "Q", [0, 6])
    receipt = controller.kv_score(
        "S", q_export="Q", method="h2o", request_id="ctx", expected_cursor=CTX
    )
    assert receipt["variant"] is None and receipt["library_params"] == {
        name: None for name in FA
    }
    selected = controller.kv_select(
        "sel", scores="S", budget_tokens=2, policy="shared", aggregate="max"
    )
    assert selected["variant"] == "h2o_uniform"
    assert selected["method"]["name"] == "h2o_uniform"


def test_kv_score_kvzip_paper_normaliser_needs_the_contiguous_prefix(fakes):
    controller, layers = resident_controller()
    fill_export(controller, "Q", [4, 6])  # repeat rows 4, 5 of "ctx"
    for bad in ([0, 3], [0, 6], [1, 4]):
        with pytest.raises(
            CompactionContractError, match="key_range == \\[0, repeat_start\\]"
        ):
            controller.kv_score(
                "Z",
                q_export="Q",
                method="kvzip",
                request_id="ctx",
                expected_cursor=CTX,
                key_range=bad,
            )
    receipt = controller.kv_score(
        "Z", q_export="Q", method="kvzip", request_id="ctx", expected_cursor=CTX
    )
    assert receipt["source"]["key_range"] == [0, 4]
    assert receipt["source"]["normalisation"] == "context_plus_causal_repeat_keys"
    assert receipt["source"]["k_ref"] == {"request_id": "ctx", "token_range": [4, 6]}
    assert receipt["variant"] == "kvzip_uniform_perlayer_fullctx"
    for call, name in zip(fakes.scores.calls, FA):
        assert call["method"] == "kvzip" and call["normalisation"] == "paper"
        assert torch.equal(call["k"], cache_rows(layers[name], range(4))[..., :HEAD])
        # k_ref reaches the last chunk (tokens 4, 5) beyond the lagging worker copy.
        assert torch.equal(call["k_ref"], cache_rows(layers[name], [4, 5])[..., :HEAD])
    explicit = controller.kv_score(
        "Z2",
        q_export="Q",
        method="kvzip",
        request_id="ctx",
        expected_cursor=CTX,
        key_range=[0, 4],
    )
    assert explicit["source"]["key_range"] == [0, 4]


def test_kv_score_kvzip_from_a_snapshot_labels_the_normaliser_and_counts_dropped_keys(
    fakes,
):
    controller, layers = resident_controller()
    fill_export(controller, "Q", [4, 6])
    row, metadata, computed = controller._resident("ctx", expected_cursor=CTX)
    controller.kv_store.capture(
        {"name": "snap", "layer_token_indices": {"fa0": [0, 1, 3], "fa1": [1, 2, 3]}},
        "ctx",
        row,
        metadata,
        computed,
    )
    with pytest.raises(CompactionContractError, match="required for snapshot"):
        controller.kv_score("S", q_export="Q", method="kvzip", kv_snapshot="snap")
    with pytest.raises(CompactionContractError, match="differs from the snapshot"):
        controller.kv_score(
            "S", q_export="Q", method="kvzip", kv_snapshot="snap", expected_cursor=5
        )
    receipt = controller.kv_score(
        "S", q_export="Q", method="kvzip", kv_snapshot="snap", expected_cursor=CTX
    )
    assert receipt["source"]["kind"] == "kv_snapshot"
    assert receipt["source"]["expected_cursor"] == CTX
    assert receipt["library_warnings"] == {name: ["fake-score-warning"] for name in FA}
    assert receipt["source"]["name"] == "snap" and receipt["source"]["cursor"] == 6
    assert receipt["source"]["normalisation"] == "snapshot_plus_causal_repeat_keys"
    assert receipt["source"]["dropped_prefix_keys"] == {"fa0": 1, "fa1": 1}
    assert controller.score_store.key_token_indices("S") == {
        "fa0": [0, 1, 3],
        "fa1": [1, 2, 3],
    }
    for call, name in zip(fakes.scores.calls, FA):
        assert torch.equal(
            call["k"], controller.kv_store.snapshot_rows("snap")[name][..., :HEAD]
        )
        assert torch.equal(call["k_ref"], cache_rows(layers[name], [4, 5])[..., :HEAD])
    with pytest.raises(CompactionContractError, match="same key tokens"):
        controller.kv_select(
            "sel", scores="S", budget_tokens=2, policy="shared", aggregate="max"
        )
    # Snapshot keys inside the repeat range would be counted twice: refused.
    controller.kv_store.capture(
        {"name": "overlap", "token_indices": [0, 5]}, "ctx", row, metadata, computed
    )
    with pytest.raises(CompactionContractError, match="overlap the repeat rows"):
        controller.kv_score(
            "S2",
            q_export="Q",
            method="kvzip",
            kv_snapshot="overlap",
            expected_cursor=CTX,
        )
    # The context-only deviation must be opted into explicitly and is labelled.
    controller.q_store.drop("Q")
    fill_export(controller, "Q", [4, 6], request_id=None)
    with pytest.raises(CompactionContractError, match="k_ref"):
        controller.kv_score(
            "S3", q_export="Q", method="kvzip", kv_snapshot="snap", expected_cursor=CTX
        )
    deviation = controller.kv_score(
        "S3",
        q_export="Q",
        method="kvzip",
        kv_snapshot="snap",
        expected_cursor=CTX,
        params={"context_only_normalisation": True},
    )
    assert deviation["source"]["normalisation"] == "context_only"
    assert deviation["variant"] == "kvzip_uniform_perlayer_fullctx_ctxonly"
    assert fakes.scores.calls[-1]["k_ref"] is None
    assert fakes.scores.calls[-1]["normalisation"] == "context_only"


def test_h2o_with_a_snapshot_source_requires_the_exports_request(fakes):
    controller, layers = resident_controller()
    fill_export(controller, "Q", [0, 6])
    row, metadata, computed = controller._resident("ctx", expected_cursor=CTX)
    controller.kv_store.capture(
        {"name": "mine", "token_indices": [0, 1, 2]}, "ctx", row, metadata, computed
    )
    controller.kv_store.capture(
        {"name": "theirs", "token_indices": [0, 1, 2]}, "other", row, metadata, computed
    )
    ok = controller.kv_score(
        "S", q_export="Q", method="h2o", kv_snapshot="mine", expected_cursor=CTX
    )
    assert ok["source"]["kind"] == "kv_snapshot" and ok["source"]["request_id"] == "ctx"
    with pytest.raises(CompactionContractError, match="same .known. request"):
        controller.kv_score(
            "S2", q_export="Q", method="h2o", kv_snapshot="theirs", expected_cursor=CTX
        )


def test_kv_select_requires_policy_and_aggregate_and_returns_a_method_record(fakes):
    controller, layers = resident_controller()
    fill_export(controller, "Q", [0, 6])
    controller.kv_score(
        "S", q_export="Q", method="h2o", request_id="ctx", expected_cursor=CTX
    )
    with pytest.raises(TypeError):
        controller.kv_select("x", scores="S", budget_tokens=3)
    with pytest.raises(CompactionContractError, match="Unknown selection policy"):
        controller.kv_select(
            "x", scores="S", budget_tokens=3, policy="top", aggregate="max"
        )
    shared = controller.kv_select(
        "shared",
        scores="S",
        budget_tokens=3,
        policy="shared",
        aggregate="max",
        protected=[0],
    )
    assert shared["token_indices"] == [0, 4, 5]
    assert shared["retained_tokens"] == 3 and shared["scored_tokens"] == 6
    assert shared["layer_token_indices"] == {name: [0, 4, 5] for name in FA}
    assert (
        shared["score_method"] == "h2o" and shared["variant"] == "h2o_uniform_prefill"
    )
    assert shared["layer_order"] == list(FA)
    method = shared["method"]
    assert method["name"] == "h2o_uniform_prefill"
    assert (
        method["params"]["policy"] == "shared"
        and method["params"]["aggregate"] == "max"
    )
    assert (
        method["params"]["protected"] == [0] and method["params"]["budget_tokens"] == 3
    )
    assert method["params"]["score_method"] == "h2o"
    assert method["params"]["library_params"]["fa0"]["memory_budget_bytes"] == 1 << 30
    assert method["inputs"]["scores"] == "S" and method["inputs"]["q_export"] == "Q"
    assert method["inputs"]["source"]["kind"] == "resident_request"
    library = method["inputs"]["library"]
    assert set(library) == {
        "package",
        "path",
        "commit",
        "library_commit",
        "dirty",
        "registered_override",
        "error",
    }
    assert library["registered_override"] == ["am", "scores", "select"]
    assert shared["capture_spec"] == {
        "name": "shared",
        "token_indices": [0, 4, 5],
        "method": method,
    }
    assert fakes.select.calls[-1] == {"budget": 3, "protected": [0]}
    snapshot = controller.kv_capture(shared["capture_spec"], "ctx", expected_cursor=CTX)
    assert snapshot["method"] == method and snapshot["synthetic"] is False
    assert snapshot["selection_policy"] == "shared" and snapshot["source_cursor"] == CTX
    per_layer = controller.kv_select(
        "pl", scores="S", budget_tokens=2, policy="per_layer", aggregate="mean"
    )
    assert per_layer["token_indices"] is None
    assert per_layer["capture_spec"]["layer_token_indices"] == {
        name: [4, 5] for name in FA
    }
    assert per_layer["capture_spec"]["method"]["params"]["policy"] == "per_layer"
    snapshot = controller.kv_capture(
        per_layer["capture_spec"], "ctx", expected_cursor=CTX
    )
    assert (
        snapshot["selection_policy"] == "per_layer" and snapshot["retained_tokens"] == 2
    )
    for name in FA:
        assert torch.equal(
            controller.kv_store.snapshot_rows("pl")[name],
            cache_rows(layers[name], [4, 5]),
        )
    assert set(controller.kv_selections()["selections"]) == {"shared", "pl"}
    with pytest.raises(CompactionContractError, match="not among the scored keys"):
        controller.kv_select(
            "x",
            scores="S",
            budget_tokens=2,
            policy="shared",
            aggregate="max",
            protected=[9],
        )
    with pytest.raises(CompactionContractError, match="overwritten"):
        controller.kv_select(
            "shared", scores="S", budget_tokens=2, policy="shared", aggregate="max"
        )
    with pytest.raises(CompactionContractError, match="exceeds budget"):
        controller.kv_select(
            "z",
            scores="S",
            budget_tokens=1,
            policy="shared",
            aggregate="max",
            protected=[0, 1],
        )
    with pytest.raises(CompactionContractError, match="exceeds the scored keys"):
        controller.kv_select(
            "big", scores="S", budget_tokens=7, policy="shared", aggregate="max"
        )


def test_kv_fit_am_freezes_frame_tokens_and_records_fit_metadata(fakes):
    controller, layers = resident_controller()
    fill_export(controller, "Q", [0, 6])
    receipt = controller.kv_fit_am(
        "AM",
        q_export="Q",
        budget_tokens=3,
        request_id="ctx",
        expected_cursor=CTX,
        protected=[1],
        params={"ridge": 0.1},
    )
    assert receipt["synthetic"] is True
    method = receipt["method"]
    assert method["name"].endswith("_framefixed_chol64")
    assert method["alias"] == "am_nobias_uniform"
    params = method["params"]
    assert (params["ridge"], params["bias"], params["head_budget"]) == (
        0.1,
        False,
        "uniform",
    )
    assert params["protected"] == [1] and params["budget_tokens"] == 3
    assert params["frame_policy"] == "fixed"
    assert params["fixed_tokens"] == {name: [1] for name in FA}
    assert params["n_fixed"] == {name: 1 for name in FA}
    assert params["n_protected"] == {name: 0 for name in FA}
    assert (params["solver"], params["accumulate_dtype"]) == (
        "cholesky",
        "torch.float64",
    )
    assert params["compute_dtype"] == "torch.float32"
    assert params["fit_value_dtype"] == "torch.float32"
    assert params["values_cast"] == ["torch.float32->torch.float32"]
    assert params["cast_error"] == {
        name: {"max_abs": 0.0, "rel_fro": 0.0} for name in FA
    }
    assert params["blocking"] == {name: {"memory_budget_bytes": 1 << 30} for name in FA}
    assert params["identity_shortcut"] == {name: False for name in FA}
    for name in FA:
        error = params["output_error_after_cast_in_sample"][name]
        assert len(error["rel"]) == KV_HEADS and error["max_rel"] >= 0.0
    assert params["output_error_after_rel_library"] == {name: [0.0, 0.0] for name in FA}
    assert params["library_warnings"] == {name: ["fake-am-warning"] for name in FA}
    assert params["warnings"] == {name: [] for name in FA}
    inputs = method["inputs"]
    assert inputs["q_export"] == "Q" and inputs["kv_digests"] is None
    assert inputs["q_digest"] is None and inputs["q_digest_policy"] == "not_computed"
    assert inputs["source"]["kind"] == "resident_request"
    assert inputs["source"]["expected_cursor"] == CTX
    assert inputs["query_convention"]["stage"] == "post_qk_norm_post_rope_unscaled"
    assert inputs["library"]["registered_override"] == ["am", "scores", "select"]
    assert receipt["layer_token_indices"] == {name: [1, 4, 5] for name in FA}
    assert receipt["selection_policy"] == "per_layer"
    assert (receipt["retained_tokens"], receipt["source_cursor"]) == (3, CTX)
    assert receipt["source_position_offset"] == 0 and receipt["request_id"] == "ctx"
    call = fakes.am.calls[0]
    assert (call["fixed"], call["protected"]) == ([1], [])
    assert (call["budget"], call["ridge"], call["bias"]) == (3, 0.1, False)
    assert (call["scale"], call["memory_budget_bytes"]) == (HEAD**-0.5, 1 << 30)
    assert call["v_dtype"] == torch.float32
    assert call["max_fit_workspace_bytes"] is None  # CPU: no admission limit
    for name in FA:
        workspace = params["fit_workspace"][name]
        assert workspace["capacity"]["policy"] == "cpu_no_limit"
        assert workspace["capacity"]["bytes"] is None
        assert (
            workspace["admission"]["scope"] == "lower_bound_only_not_peak_certification"
        )
        assert workspace["admission"]["checked_capacity_bytes"] is None
    rows = controller.kv_store.snapshot_rows("AM")
    for name in FA:
        source = cache_rows(layers[name], [1, 4, 5])
        assert torch.equal(rows[name][..., :HEAD], source[..., :HEAD])
        # Frozen token 1 keeps its original values; the others carry the fit.
        assert torch.equal(rows[name][0, :, HEAD:], source[0, :, HEAD:])
        assert torch.equal(rows[name][1:, :, HEAD:], source[1:, :, HEAD:] * 2)
    # Import through the unchanged restore path into a fresh row.
    metadata = {
        name: NS(
            block_table=controller.runner.input_batch.block_table[0].get_cpu_tensor()
        )
        for name in FA
    }
    for layer in layers.values():
        layer.kv_cache[3:5].fill_(-7)
    result = controller.kv_store.restore("AM", "fresh", 1, metadata, 3)
    assert result["synthetic"] is True
    for name in FA:
        assert torch.equal(
            layers[name].kv_cache[3, :, :3, :].transpose(0, 1), rows[name]
        )
        assert torch.all(layers[name].kv_cache[3, :, 3:, :] == -7)
    refit = controller.kv_fit_am(
        "AMrefit",
        q_export="Q",
        budget_tokens=2,
        request_id="ctx",
        expected_cursor=CTX,
        protected=[1],
        params={"refit_protected": True, "digest_inputs": True, "output_error": False},
    )
    assert fakes.am.calls[-1]["fixed"] == [] and fakes.am.calls[-1]["protected"] == [1]
    assert refit["method"]["params"]["frame_policy"] == "refit"
    assert refit["method"]["params"]["fixed_tokens"] == {name: [] for name in FA}
    assert refit["method"]["params"]["output_error_after_cast_in_sample"] is None
    assert set(refit["method"]["inputs"]["kv_digests"]) == set(FA)
    assert isinstance(refit["method"]["inputs"]["q_digest"], str)
    assert refit["method"]["inputs"]["q_digest_policy"] == "computed"
    for name in FA:
        source = cache_rows(layers[name], [1, 5])
        refit_rows = controller.kv_store.snapshot_rows("AMrefit")[name]
        assert torch.equal(refit_rows[..., HEAD:], source[..., HEAD:] * 2)


def test_compute_ops_fail_closed_on_missing_library_and_bad_outputs(monkeypatch):
    controller, layers = resident_controller()
    fill_export(controller, "Q", [0, 6])
    common = dict(q_export="Q", request_id="ctx", expected_cursor=CTX)
    monkeypatch.setattr(
        compaction_q, "METHODS_PACKAGE", "profiling_missing_for_test.compaction_methods"
    )
    with pytest.raises(CompactionContractError, match="unavailable"):
        controller.kv_score("S", method="h2o", **common)
    assert controller.scores()["scores"] == {}
    bad_outputs = {
        "shape": lambda q, k, **kw: torch.zeros(3, 3),
        "nan": lambda q, k, **kw: torch.full((KV_HEADS, k.shape[0]), float("nan")),
        "dtype": lambda q, k, **kw: torch.zeros(KV_HEADS, k.shape[0], dtype=torch.long),
    }
    for label, fake in bad_outputs.items():
        compaction_q.register_methods(scores=NS(h2o_scores=fake, kvzip_scores=fake))
        with pytest.raises(CompactionContractError, match="finite"):
            controller.kv_score(label, method="h2o", **common)
    compaction_q.register_methods(
        scores=NS(
            h2o_scores=lambda q, k, **kw: NS(scores=torch.zeros(2, 6), variant="")
        )
    )
    with pytest.raises(CompactionContractError, match="ScoreResult"):
        controller.kv_score("S", method="h2o", **common)
    assert controller.scores()["scores"] == {}
    compaction_q.register_methods(scores=FakeScores())
    with pytest.raises(CompactionContractError, match="Unknown scoring method"):
        controller.kv_score("S", method="snapkv", **common)
    for bad_params in ({"temperature": 1}, {"causal": False}):
        with pytest.raises(CompactionContractError, match="Unknown score params"):
            controller.kv_score("S", method="h2o", params=bad_params, **common)
    with pytest.raises(CompactionContractError, match="exactly one of"):
        controller.kv_score("S", q_export="Q", method="h2o")
    with pytest.raises(CompactionContractError, match="not resident"):
        controller.kv_score(
            "S", q_export="Q", method="h2o", request_id="ghost", expected_cursor=CTX
        )
    with pytest.raises(CompactionContractError, match="exceeds the consumed prefix"):
        controller.kv_score("S", method="h2o", key_range=[0, 7], **common)
    controller.q_store.open("P", token_range=[0, 6], request_id="ctx")
    with pytest.raises(CompactionContractError, match="incomplete"):
        controller.kv_score(
            "S", q_export="P", method="h2o", request_id="ctx", expected_cursor=CTX
        )
    controller.runner.execute_model_state = object()
    with pytest.raises(CompactionContractError, match="between forward and sampling"):
        controller.kv_score("S", method="h2o", **common)
    controller.runner.execute_model_state = None
    controller.kv_score("S", method="h2o", **common)
    with pytest.raises(CompactionContractError, match="overwritten"):
        controller.kv_score("S", method="h2o", **common)
    # Selection library outputs are validated too.
    compaction_q.register_methods(
        select=NS(uniform_token_budget=lambda s, b, **kw: [[0] for _ in s])
    )
    with pytest.raises(CompactionContractError, match="match the budget"):
        controller.kv_select(
            "sel", scores="S", budget_tokens=2, policy="shared", aggregate="max"
        )
    compaction_q.register_methods(
        select=NS(uniform_token_budget=lambda s, b, **kw: [[0, 1], [2, 3]])
    )
    with pytest.raises(CompactionContractError, match="identical layer lists"):
        controller.kv_select(
            "sel", scores="S", budget_tokens=2, policy="shared", aggregate="max"
        )
    compaction_q.register_methods(
        select=NS(
            uniform_token_budget=lambda *a, **kw: (_ for _ in ()).throw(
                ValueError("boom")
            )
        )
    )
    with pytest.raises(
        CompactionContractError, match="uniform_token_budget failed.*boom"
    ):
        controller.kv_select(
            "sel", scores="S", budget_tokens=2, policy="shared", aggregate="max"
        )
    assert controller.kv_selections()["selections"] == {}

    # AM outputs are validated: beta, shared layout, counts, keys, fixed rows, values.
    def am_result(
        k,
        v,
        chosen,
        *,
        beta=None,
        k_shift=0.0,
        v_cols=None,
        shared=True,
        fixed_mask="auto",
        fixed=(),
        v_shift_fixed=0.0,
    ):
        rows_k = k[chosen] + k_shift
        rows_v = v[chosen].clone() if v_cols is None else v[chosen][..., :v_cols]
        mask_rows = torch.tensor([c in fixed for c in chosen], dtype=torch.bool)
        if v_cols is None and v_shift_fixed:
            rows_v[mask_rows] += v_shift_fixed
        mask = (
            mask_rows[None, :].repeat(k.shape[1], 1)
            if fixed_mask == "auto"
            else fixed_mask
        )
        return NS(
            beta=beta,
            shared=shared,
            variant="fake",
            fixed_mask=mask,
            diagnostics={},
            token_indices=lambda: list(chosen),
            cache_layout=lambda: (rows_k, rows_v),
        )

    faults = {
        "beta": (lambda k, v: am_result(k, v, [0, 1], beta=torch.zeros(2)), "beta"),
        "shared": (lambda k, v: am_result(k, v, [0, 1], shared=False), "AMResult"),
        "count": (lambda k, v: am_result(k, v, [0]), "match the budget"),
        "keys": (lambda k, v: am_result(k, v, [0, 1], k_shift=1.0), "original keys"),
        "values": (lambda k, v: am_result(k, v, [0, 1], v_cols=1), "AM v_c"),
        "nomask": (
            lambda k, v: am_result(k, v, [0, 1], fixed=(0,), fixed_mask=None),
            "lacks fixed_mask",
        ),
        "maskset": (
            lambda k, v: am_result(k, v, [0, 1], fixed=(1,)),
            "fixed_mask marks",
        ),
        "frozen": (
            lambda k, v: am_result(k, v, [0, 1], fixed=(0,), v_shift_fixed=1.0),
            "fixed rows",
        ),
    }
    for build, message in faults.values():
        compaction_q.register_methods(
            am=NS(compact=lambda q, k, v, b, build=build, **kw: build(k, v))
        )
        with pytest.raises(CompactionContractError, match=message):
            controller.kv_fit_am("AM", budget_tokens=2, protected=[0], **common)
    compaction_q.register_methods(am=FakeAM())
    with pytest.raises(CompactionContractError, match="exceeds the source keys"):
        controller.kv_fit_am("AM", budget_tokens=7, **common)
    for bad_params in ({"iters": 3}, {"mass_weighting": "shifted"}):
        with pytest.raises(CompactionContractError, match="Unknown am params"):
            controller.kv_fit_am("AM", budget_tokens=2, params=bad_params, **common)
    assert controller.kv_store.list()["snapshots"] == {}


def test_kv_capture_of_a_resident_request_matches_boundary_capture_semantics():
    controller, layers = resident_controller()
    receipt = controller.kv_capture(
        {"name": "res", "token_indices": [0, 3, 5]}, "ctx", expected_cursor=CTX
    )
    assert receipt["request_id"] == "ctx" and receipt["source_cursor"] == CTX
    assert receipt["selection_policy"] == "shared" and receipt["synthetic"] is False
    assert receipt["method"] is None
    for name in FA:
        assert torch.equal(
            controller.kv_store.snapshot_rows("res")[name],
            cache_rows(layers[name], [0, 3, 5]),
        )
    controller._position_rules["ctx"] = (2, 40)
    method = {"name": "streamingllm", "params": {"recent": 2}, "inputs": {}}
    offset = controller.kv_capture(
        {"name": "off", "token_indices": [0], "method": method},
        "ctx",
        expected_cursor=CTX,
    )
    assert offset["source_position_offset"] == 40 and offset["method"] == method
    with pytest.raises(ValueError, match="method must carry"):
        controller.kv_capture(
            {"name": "bad", "token_indices": [0], "method": {"name": "x"}},
            "ctx",
            expected_cursor=CTX,
        )
    with pytest.raises(ValueError, match="indices"):
        controller.kv_capture(
            {"name": "late", "token_indices": [6]}, "ctx", expected_cursor=CTX
        )
    controller.arm(
        expected_prompt_tokens=9,
        capture_kv={"name": "pending", "token_indices": [0]},
    )
    with pytest.raises(CompactionContractError, match="reserved by another operation"):
        controller.kv_capture(
            {"name": "pending", "token_indices": [0]}, "ctx", expected_cursor=CTX
        )
    # A failed operation releases its reservation.
    controller.operation.failed = "RuntimeError: kernel failed"
    controller.kv_capture(
        {"name": "pending", "token_indices": [0]}, "ctx", expected_cursor=CTX
    )
    assert controller.kv_drop("pending")["snapshots"].keys() == {"res", "off"}


def test_scale_is_head_size_rule_cross_checked_against_the_kernel(fakes):
    controller, layers = resident_controller()
    fill_export(controller, "Q", [0, 6])
    layers["fa0"].impl.scale = 0.5  # a kernel scale that is not head_size ** -0.5
    common = dict(q_export="Q", method="h2o", request_id="ctx", expected_cursor=CTX)
    with pytest.raises(CompactionContractError, match="differs from head_size"):
        controller.kv_score("S", **common)
    assert controller.scores()["scores"] == {} and fakes.scores.calls == []
    receipt = controller.kv_score("S", params={"scale": 0.5}, **common)
    assert receipt["source"]["scale"] == {name: 0.5 for name in FA}
    assert all(call["scale"] == 0.5 for call in fakes.scores.calls)
    with pytest.raises(CompactionContractError, match="positive number"):
        controller.kv_score("S2", params={"scale": -1}, **common)


# ---------------------------------------------------------- real library


def test_real_methods_library_contract_on_tiny_tensors():
    """Resolve the committed ``profiling.compaction_methods`` from the repo root.

    This is the contract-drift guard: it fails (not skips) when the package is
    missing or its API no longer matches what the fork passes.
    """
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        library = compaction_q.resolve_methods("scores")
    except CompactionContractError as error:
        pytest.fail(f"real methods library unavailable at {REPO_ROOT}: {error}")
    assert library.__name__ == "profiling.compaction_methods.scores"
    controller, layers = resident_controller()
    generator = torch.Generator().manual_seed(0)
    for layer in layers.values():
        layer.kv_cache.copy_(torch.randn(layer.kv_cache.shape, generator=generator))
    fill_export(controller, "Q", [0, 6])
    for name in FA:
        controller.q_store._entries["Q"]["tensors"][name].copy_(
            torch.randn(6, Q_HEADS, HEAD, generator=generator)
        )
    h2o = controller.kv_score(
        "S", q_export="Q", method="h2o", request_id="ctx", expected_cursor=CTX
    )
    assert h2o["variant"] == "h2o_uniform_prefill"
    assert h2o["library_params"]["fa0"]["memory_budget_bytes"] == 1 << 30
    for tensor in controller.score_store.tensors("S").values():
        # Each query contributes unit mass per kv head (mean over the G heads).
        assert torch.allclose(
            tensor.sum(dim=1), torch.full((KV_HEADS,), 6.0), atol=1e-4
        )
    selection = controller.kv_select(
        "sel",
        scores="S",
        budget_tokens=3,
        policy="per_layer",
        aggregate="mean",
        protected=[0],
    )
    assert selection["method"]["name"] == "h2o_uniform_prefill"
    assert selection["method"]["inputs"]["library"]["path"].endswith(
        "compaction_methods"
    )
    snapshot = controller.kv_capture(
        selection["capture_spec"], "ctx", expected_cursor=CTX
    )
    assert snapshot["method"]["name"] == "h2o_uniform_prefill"
    fill_export(controller, "R", [4, 6])
    kvzip = controller.kv_score(
        "Z", q_export="R", method="kvzip", request_id="ctx", expected_cursor=CTX
    )
    assert kvzip["variant"] == "kvzip_uniform_perlayer_fullctx"
    assert kvzip["source"]["key_range"] == [0, 4]
    for tensor in controller.score_store.tensors("Z").values():
        assert ((tensor >= 0) & (tensor <= 1)).all()
    full = controller.kv_fit_am(
        "AMfull",
        q_export="R",
        budget_tokens=4,
        request_id="ctx",
        expected_cursor=CTX,
        key_range=[0, 4],
        protected=[0],
    )
    assert full["method"]["name"].startswith(
        "am_rmskeys_ols_nobias_uniform_offpolicy_framefixed"
    )
    assert full["method"]["params"]["identity_shortcut"] == {name: True for name in FA}
    assert full["method"]["params"]["fit_workspace"]["fa0"]["admission"] is None
    assert (
        full["method"]["params"]["solver"]
        and full["method"]["params"]["accumulate_dtype"]
    )
    rows = controller.kv_store.snapshot_rows("AMfull")
    for name in FA:
        assert torch.equal(rows[name], cache_rows(layers[name], range(4)))
        assert (
            full["method"]["params"]["output_error_after_cast_in_sample"][name][
                "max_rel"
            ]
            < 1e-5
        )
    partial = controller.kv_fit_am(
        "AM2",
        q_export="R",
        budget_tokens=2,
        request_id="ctx",
        expected_cursor=CTX,
        key_range=[0, 4],
        protected=[0],
    )
    assert partial["method"]["params"]["frame_policy"] == "fixed"
    admission = partial["method"]["params"]["fit_workspace"]["fa0"]["admission"]
    assert admission["scope"] == "lower_bound_only_not_peak_certification"
    assert admission["gram_width"] == 2 and admission["checked_capacity_bytes"] is None
    assert partial["method"]["params"]["fixed_tokens"] == {name: [0] for name in FA}
    for name in FA:
        stored = controller.kv_store.snapshot_rows("AM2")[name]
        assert torch.equal(
            stored[0], cache_rows(layers[name], [0])[0]
        )  # frozen frame row
        error = partial["method"]["params"]["output_error_after_cast_in_sample"][name]
        assert 0.0 <= error["max_rel"] < 10.0
    metadata = {
        name: NS(
            block_table=controller.runner.input_batch.block_table[0].get_cpu_tensor()
        )
        for name in FA
    }
    controller.kv_store.restore("AM2", "fresh", 1, metadata, 2)


# ------------------------------------------------------------------ subset


def test_kv_subset_slices_an_existing_snapshot_without_a_cache_pass(fakes):
    controller, layers = resident_controller()
    full = controller.kv_capture(
        {"name": "full", "token_indices": list(range(CTX))}, "ctx", expected_cursor=CTX
    )
    controller._position_rules["ctx"] = (0, 0)
    method = {
        "name": "h2o_uniform_prefill",
        "params": {"budget_tokens": 3},
        "inputs": {},
    }
    for layer in layers.values():
        layer.kv_cache.fill_(
            -5
        )  # the subset must come from the snapshot, not the cache
    receipt = controller.kv_subset(
        "sub", source_snapshot="full", token_indices=[0, 4, 5], method=method
    )
    assert receipt["selection_policy"] == "shared" and receipt["token_indices"] == [
        0,
        4,
        5,
    ]
    assert receipt["retained_tokens"] == 3 and receipt["synthetic"] is False
    assert (receipt["source_cursor"], receipt["source_position_offset"]) == (CTX, 0)
    assert receipt["request_id"] == "ctx" and receipt["method"] == method
    assert receipt["derived_from"] == {
        "name": "full",
        "digest": full["digest"],
        "retained_tokens": CTX,
        "selection_policy": "shared",
        "synthetic": False,
    }
    assert receipt["digest_verified_at"] == "subset"
    for name in FA:
        rows = controller.kv_store.snapshot_rows("full")[name][torch.tensor([0, 4, 5])]
        assert torch.equal(controller.kv_store.snapshot_rows("sub")[name], rows)
    per_layer = controller.kv_subset(
        "pl",
        source_snapshot="full",
        layer_token_indices={"fa0": [1, 2], "fa1": [3, 5]},
        method=method,
    )
    assert (
        per_layer["selection_policy"] == "per_layer"
        and per_layer["token_indices"] is None
    )
    assert torch.equal(
        controller.kv_store.snapshot_rows("pl")["fa1"],
        controller.kv_store.snapshot_rows("full")["fa1"][torch.tensor([3, 5])],
    )
    # A subset of a subset keeps the chain and the mutation-free source.
    nested = controller.kv_subset(
        "nested", source_snapshot="sub", token_indices=[4], method=method
    )
    assert nested["derived_from"]["name"] == "sub" and nested["retained_tokens"] == 1
    assert controller.kv_store.audit("full")["digest"] == full["digest"]
    # Import through the unchanged restore path.
    metadata = {
        name: NS(
            block_table=controller.runner.input_batch.block_table[0].get_cpu_tensor()
        )
        for name in FA
    }
    controller.kv_store.restore("sub", "fresh", 1, metadata, 3)
    for name in FA:
        assert torch.equal(
            layers[name].kv_cache[3, :, :3, :].transpose(0, 1),
            controller.kv_store.snapshot_rows("sub")[name],
        )
    # The selection's ready-made method record can be used directly.
    fill_export(controller, "Q", [0, 6])
    for layer in layers.values():
        layer.kv_cache.copy_(fa_layer(0, "x").kv_cache)
    controller.kv_score(
        "S", q_export="Q", method="h2o", request_id="ctx", expected_cursor=CTX
    )
    selection = controller.kv_select(
        "sel", scores="S", budget_tokens=2, policy="shared", aggregate="max"
    )
    materialised = controller.kv_subset(
        "sel_rows",
        source_snapshot="full",
        token_indices=selection["token_indices"],
        method=selection["method"],
    )
    assert materialised["method"]["name"] == "h2o_uniform_prefill"
    assert materialised["layer_token_indices"] == selection["layer_token_indices"]


@pytest.mark.parametrize(
    "kwargs,message",
    [
        (dict(token_indices=[0, 9]), "indices"),
        (dict(token_indices=[0, 3]), "not in the source snapshot"),
        (dict(layer_token_indices={"fa0": [0, 1], "fa1": [0]}), "equal counts"),
        (dict(layer_token_indices={"fa0": [0, 1]}), "every FA layer"),
        (
            dict(token_indices=[0], layer_token_indices={"fa0": [0], "fa1": [0]}),
            "not both",
        ),
        (dict(), "not both"),
        (dict(token_indices=[0], method={"name": "x"}), "method must carry"),
        (dict(token_indices=[0], source_snapshot="ghost"), "Unknown KV snapshot"),
        (dict(token_indices=[0], name_out="src"), "overwritten"),
    ],
)
def test_kv_subset_rejects_bad_selections_sources_and_methods(kwargs, message):
    controller, layers = resident_controller()
    controller.kv_capture(
        {"name": "src", "token_indices": [0, 1, 2, 4, 5]}, "ctx", expected_cursor=CTX
    )
    call = dict(
        name_out="sub",
        source_snapshot="src",
        method={"name": "ev", "params": {}, "inputs": {}},
    )
    call.update(kwargs)
    name_out = call.pop("name_out")
    with pytest.raises(ValueError, match=message):
        controller.kv_subset(name_out, **call)
    assert set(controller.kv_store.list()["snapshots"]) == {"src"}
    controller.arm(
        expected_prompt_tokens=9, capture_kv={"name": "pending", "token_indices": [0]}
    )
    with pytest.raises(CompactionContractError, match="reserved by another operation"):
        controller.kv_subset(
            "pending",
            source_snapshot="src",
            token_indices=[0],
            method={"name": "ev", "params": {}, "inputs": {}},
        )


def test_subset_of_a_synthetic_snapshot_stays_synthetic(fakes):
    controller, layers = resident_controller()
    fill_export(controller, "Q", [0, 6])
    controller.kv_fit_am(
        "AM",
        q_export="Q",
        budget_tokens=3,
        request_id="ctx",
        expected_cursor=CTX,
        protected=[1],
    )
    sub = controller.kv_subset(
        "AMsub",
        source_snapshot="AM",
        token_indices=[1, 5],
        method={"name": "am_then_evict", "params": {}, "inputs": {}},
    )
    assert sub["synthetic"] is True and sub["derived_from"]["synthetic"] is True
    for name in FA:
        rows = controller.kv_store.snapshot_rows("AM")[name][torch.tensor([0, 2])]
        assert torch.equal(controller.kv_store.snapshot_rows("AMsub")[name], rows)


# ------------------------------------------------------ drops and describes


def test_kv_selection_drop_frees_the_name_for_a_retried_task(fakes):
    controller, layers = resident_controller()
    fill_export(controller, "Q", [0, 6])
    controller.kv_score(
        "S", q_export="Q", method="h2o", request_id="ctx", expected_cursor=CTX
    )
    first = controller.kv_select(
        "task", scores="S", budget_tokens=2, policy="shared", aggregate="max"
    )
    with pytest.raises(CompactionContractError, match="overwritten"):
        controller.kv_select(
            "task", scores="S", budget_tokens=3, policy="shared", aggregate="max"
        )
    # A pending boundary capture of the selection's spec reserves the name.
    controller.arm(expected_prompt_tokens=9, capture_kv=first["capture_spec"])
    with pytest.raises(CompactionContractError, match="reserved by an operation"):
        controller.kv_selection_drop("task")
    controller.operation.failed = "RuntimeError: kernel failed"
    dropped = controller.kv_selection_drop("task")
    assert dropped["dropped"] == "task" and dropped["selections"] == {}
    retried = controller.kv_select(
        "task", scores="S", budget_tokens=3, policy="shared", aggregate="max"
    )
    assert retried["retained_tokens"] == 3
    with pytest.raises(CompactionContractError, match="Unknown selection"):
        controller.kv_selection_drop("never")
    assert set(controller.kv_selections()["selections"]) == {"task"}


def test_per_name_describe_returns_one_entry_without_listing(fakes):
    controller, layers = resident_controller()
    fill_export(controller, "Q", [0, 6])
    controller.kv_capture(
        {"name": "full", "token_indices": list(range(CTX))}, "ctx", expected_cursor=CTX
    )
    controller.kv_capture(
        {"name": "tiny", "token_indices": [1]}, "ctx", expected_cursor=CTX
    )
    described = controller.kv_describe("tiny")
    assert described["name"] == "tiny" and described["retained_tokens"] == 1
    assert described["layer_token_indices"] == {name: [1] for name in FA}
    assert "tensors" not in described
    assert described == controller.kv_store.list()["snapshots"]["tiny"]
    with pytest.raises(ValueError, match="Unknown KV snapshot"):
        controller.kv_describe("ghost")
    export = controller.q_describe("Q")
    assert export["name"] == "Q" and export["complete"] and export["rows"] == 6
    assert export == controller.q_exports()["exports"]["Q"]
    with pytest.raises(CompactionContractError, match="Unknown query export"):
        controller.q_describe("ghost")


# ------------------------------------------------------------- final pass


def test_failed_legacy_operation_closes_on_result_so_the_controller_can_rearm():
    requests = [("r", 0, 3, None, 2)]
    controller, layers, _ = make_controller(requests)
    controller.arm(expected_prompt_tokens=3)
    metadata, positions, counts = forward_inputs(requests, controller)
    controller.before_forward(metadata, positions, counts)
    controller.fail_forward(RuntimeError("kernel failed"))
    with pytest.raises(CompactionContractError, match="Call cc_result"):
        controller.arm(expected_prompt_tokens=3)
    failed = controller.operation
    with pytest.raises(CompactionContractError, match="kernel failed"):
        controller.result()
    assert failed.closed
    with pytest.raises(CompactionContractError, match="already closed"):
        failed.result()
    # The reviewer's reproduction: cc_arm must work again without a restart.
    armed = controller.arm(expected_prompt_tokens=3)
    assert armed["armed"] and controller.operation is not failed
    # An ingest aborted before its prompt end takes the same path.
    controller.before_forward(metadata, positions, counts)
    controller.finish_requests({"r"})
    with pytest.raises(CompactionContractError, match="aborted"):
        controller.result()
    assert controller.arm(expected_prompt_tokens=3)["armed"]
    # A live, unfinished operation still blocks re-arming.
    boundary = controller.before_forward(metadata, positions, counts)
    with pytest.raises(CompactionContractError, match="Call cc_result"):
        controller.arm(expected_prompt_tokens=3)
    controller.after_forward(boundary)
    assert controller.result()["forward_calls"] == 1


def test_failed_descriptor_operations_are_retired_once_by_results():
    desc = dict(operation_id="a", expected_prompt_tokens=5)
    requests = [("r", 0, 2, desc, 2)]
    controller, layers, _ = make_controller(requests)
    metadata, positions, counts = forward_inputs(requests, controller)
    controller.before_forward(metadata, positions, counts)
    controller.fail_forward(RuntimeError("kernel failed"))
    with pytest.raises(
        CompactionContractError, match="retired: a: RuntimeError: kernel"
    ):
        controller.results(["a"])
    assert controller.operations == {}
    with pytest.raises(CompactionContractError, match="Unknown operation"):
        controller.results(["a"])
    # The request lives on: its next forward with the same descriptor runs
    # uninstrumented (tombstone) instead of raising.
    requests = [("r", 2, 3, desc, 2)]
    advance(controller, requests)
    metadata, positions, counts = forward_inputs(requests, controller)
    assert controller.before_forward(metadata, positions, counts) is None
    controller.after_forward(None)
    assert controller._consumed == {"r": 5}
    controller.finish_requests({"r"})
    assert controller._retired_bindings == {} and controller._descriptors == {}
    # An unfailed, incomplete operation is still reported as such (not retired).
    other = dict(operation_id="b", expected_prompt_tokens=5)
    requests = [("s", 0, 2, other, 3)]
    controller.runner.requests["s"] = NS(
        mm_features=[],
        prompt_embeds=None,
        lora_request=None,
        num_computed_tokens=0,
        num_prompt_tokens=5,
        sampling_params=NS(extra_args={DESCRIPTOR_KEY: other}),
    )
    advance(controller, requests)
    metadata, positions, counts = forward_inputs(requests, controller)
    controller.after_forward(controller.before_forward(metadata, positions, counts))
    with pytest.raises(CompactionContractError, match="not reached its prompt end: b"):
        controller.results(["b"])
    assert "b" in controller.operations


def test_output_error_blocks_follow_the_budget_and_wrap_device_oom(monkeypatch):
    generator = torch.Generator().manual_seed(1)
    q = torch.randn(7, Q_HEADS, HEAD, generator=generator)
    k = torch.randn(9, KV_HEADS, HEAD, generator=generator)
    v = torch.randn(9, KV_HEADS, HEAD, generator=generator)
    k_c, v_c = k[[0, 3, 8]], v[[0, 3, 8]] * 1.1
    scale = HEAD**-0.5
    wide = compaction_q.attention_output_error(q, k, v, k_c, v_c, scale=scale)
    tiny = compaction_q.attention_output_error(
        q, k, v, k_c, v_c, scale=scale, budget_bytes=3 * (9 + 3) * 4 * 5
    )
    # rows_per_block = budget // (3 * (T + t) * 4) = 5 grouped rows -> 2 queries * G.
    assert (
        tiny["rows_per_block"] == 2 * (Q_HEADS // KV_HEADS)
        and tiny["budget_bytes"] == 720
    )
    assert wide["rows_per_block"] == max(1, (1 << 30) // (3 * 12 * 4)) // 2 * 2
    assert wide["rel"] == pytest.approx(tiny["rel"], rel=1e-5)
    assert wide["max_rel"] > 0.0 and len(wide["abs"]) == KV_HEADS
    with pytest.raises(CompactionContractError, match="budget_bytes must be"):
        compaction_q.attention_output_error(
            q, k, v, k_c, v_c, scale=scale, budget_bytes=0
        )

    def oom(*args, **kwargs):
        raise RuntimeError("CUDA out of memory. Tried to allocate 512.00 MiB")

    monkeypatch.setattr(compaction_q.torch, "softmax", oom)
    with pytest.raises(CompactionContractError, match="ran out of memory"):
        compaction_q.attention_output_error(q, k, v, k_c, v_c, scale=scale)

    def other(*args, **kwargs):
        raise RuntimeError("some other kernel failure")

    monkeypatch.setattr(compaction_q.torch, "softmax", other)
    with pytest.raises(RuntimeError, match="some other kernel failure"):
        compaction_q.attention_output_error(q, k, v, k_c, v_c, scale=scale)


def test_output_error_default_depends_on_the_context_length(fakes, monkeypatch):
    controller, layers = resident_controller()
    fill_export(controller, "Q", [0, 6])
    common = dict(q_export="Q", budget_tokens=2, request_id="ctx", expected_cursor=CTX)
    monkeypatch.setattr(compaction_q, "OUTPUT_ERROR_MAX_TOKENS", 4)
    off = controller.kv_fit_am("off", **common)["method"]["params"]
    assert off["output_error_after_cast_in_sample"] is None
    assert off["output_error_policy"] == "default_off_threshold_4_tokens"
    assert off["output_error_note"].startswith("in-sample")
    forced = controller.kv_fit_am("on", params={"output_error": True}, **common)
    params = forced["method"]["params"]
    assert params["output_error_policy"] == "caller"
    for name in FA:
        error = params["output_error_after_cast_in_sample"][name]
        assert error["memory_budget"]["policy"] == "default_1GiB"
        assert error["rows_per_block"] >= 1 and len(error["rel"]) == KV_HEADS
    monkeypatch.setattr(compaction_q, "OUTPUT_ERROR_MAX_TOKENS", 6)
    default_on = controller.kv_fit_am("on2", **common)["method"]["params"]
    assert default_on["output_error_policy"] == "default_on_threshold_6_tokens"
    assert default_on["output_error_after_cast_in_sample"] is not None
    with pytest.raises(CompactionContractError, match="bool or None"):
        controller.kv_fit_am("bad", params={"output_error": 1}, **common)


def test_library_provenance_is_cached_per_controller_and_names_the_library_commit(
    fakes, monkeypatch
):
    controller, layers = resident_controller()
    fill_export(controller, "Q", [0, 6])
    controller.kv_score(
        "S", q_export="Q", method="h2o", request_id="ctx", expected_cursor=CTX
    )
    calls = []
    real_run = compaction_q.subprocess.run

    def counting_run(*args, **kwargs):
        calls.append(args[0][:3])
        return real_run(*args, **kwargs)

    monkeypatch.setattr(compaction_q.subprocess, "run", counting_run)
    first = controller.kv_select(
        "a", scores="S", budget_tokens=2, policy="shared", aggregate="max"
    )
    controller.kv_select(
        "b", scores="S", budget_tokens=3, policy="shared", aggregate="max"
    )
    library = first["method"]["inputs"]["library"]
    assert set(library) >= {"package", "path", "commit", "library_commit", "dirty"}
    assert library["registered_override"] == ["am", "scores", "select"]
    git_calls = [c for c in calls if c and c[0] == "git"]
    # Shelled out once per controller (rev-parse, status, log), not per call.
    assert len(git_calls) in (0, 3)
    if library["commit"] is not None:
        assert len(library["commit"]) == 40 and len(library["library_commit"]) == 40
    controller._provenance_cache = {"package": "x", "commit": "cached"}
    cached = controller.kv_select(
        "c", scores="S", budget_tokens=2, policy="shared", aggregate="max"
    )
    assert cached["method"]["inputs"]["library"]["commit"] == "cached"


def test_h2o_snapshot_check_treats_unknown_request_ids_as_a_mismatch(fakes):
    controller, layers = resident_controller()
    fill_export(controller, "Q", [0, 6], request_id=None)  # unbound export
    row, metadata, computed = controller._resident("ctx", expected_cursor=CTX)
    controller.kv_store.capture(
        {"name": "known", "token_indices": [0, 1]}, "ctx", row, metadata, computed
    )
    controller.kv_store.register(
        "anon",
        {name: torch.zeros(2, KV_HEADS, 2 * HEAD) for name in FA},
        source_cursor=CTX,
        source_position_offset=0,
        retained_tokens=2,
        method={"name": "x", "params": {}, "inputs": {}},
        token_indices=[0, 1],
    )
    for snapshot in ("known", "anon"):
        with pytest.raises(CompactionContractError, match="same \\(known\\) request"):
            controller.kv_score(
                "S",
                q_export="Q",
                method="h2o",
                kv_snapshot=snapshot,
                expected_cursor=CTX,
            )


def test_kvzip_repeat_range_cannot_start_at_zero(fakes):
    controller, layers = resident_controller()
    fill_export(controller, "Q", [0, 6])
    with pytest.raises(CompactionContractError, match="cannot start at 0"):
        controller.kv_score(
            "Z", q_export="Q", method="kvzip", request_id="ctx", expected_cursor=CTX
        )


def test_store_errors_surface_as_the_single_contract_error_type(fakes):
    controller, layers = resident_controller()
    controller.kv_capture(
        {"name": "src", "token_indices": [0, 1]}, "ctx", expected_cursor=CTX
    )
    with pytest.raises(CompactionContractError, match="indices") as capture_error:
        controller.kv_capture(
            {"name": "bad", "token_indices": [9]}, "ctx", expected_cursor=CTX
        )
    assert capture_error.type is CompactionContractError
    with pytest.raises(
        CompactionContractError, match="Unknown KV snapshot"
    ) as subset_error:
        controller.kv_subset(
            "sub",
            source_snapshot="ghost",
            token_indices=[0],
            method={"name": "x", "params": {}, "inputs": {}},
        )
    assert subset_error.type is CompactionContractError
    with pytest.raises(
        CompactionContractError, match="Unknown KV snapshot"
    ) as describe_error:
        controller.kv_describe("ghost")
    assert describe_error.type is CompactionContractError


def test_repeat_prompt_is_forwarded_verbatim_for_kvzip_only(fakes):
    controller, layers = resident_controller()
    fill_export(controller, "Q", [4, 6])
    wording = "Repeat the previous context exactly:"
    receipt = controller.kv_score(
        "Z",
        q_export="Q",
        method="kvzip",
        request_id="ctx",
        expected_cursor=CTX,
        params={"repeat_prompt": wording},
    )
    assert receipt["params"]["repeat_prompt"] == wording
    assert receipt["library_params"]["fa0"]["repeat_prompt"] == wording
    assert all(call["repeat_prompt"] == wording for call in fakes.scores.calls)
    fill_export(controller, "H", [0, 6])
    with pytest.raises(CompactionContractError, match="kvzip only"):
        controller.kv_score(
            "S",
            q_export="H",
            method="h2o",
            request_id="ctx",
            expected_cursor=CTX,
            params={"repeat_prompt": wording},
        )
    with pytest.raises(CompactionContractError, match="nonempty string"):
        controller.kv_score(
            "Z2",
            q_export="Q",
            method="kvzip",
            request_id="ctx",
            expected_cursor=CTX,
            params={"repeat_prompt": ""},
        )


# ------------------------------------------------------- Codex M2 / R1 units


def test_fit_workspace_admission_rejection_surfaces_as_a_contract_error(fakes):
    controller, layers = resident_controller()
    fill_export(controller, "Q", [0, 6])

    def refuse(q_ref, k, v, budget, **kwargs):
        raise RuntimeError(
            "AM fit needs at least 5497390368 additional bytes, but admission "
            "capacity is 4096; query chunking cannot remove this floor"
        )

    compaction_q.register_methods(am=NS(compact=refuse))
    with pytest.raises(CompactionContractError, match="AM fit needs at least") as info:
        controller.kv_fit_am(
            "AM", q_export="Q", budget_tokens=2, request_id="ctx", expected_cursor=CTX
        )
    assert "am.compact failed" in str(info.value)
    assert controller.kv_store.list()["snapshots"] == {}
    assert compaction_q.fit_workspace_capacity(torch.device("cpu")) == {
        "bytes": None,
        "policy": "cpu_no_limit",
        "free_bytes": None,
        "allocator_cached_bytes": None,
    }


def test_requires_query_export_inspects_rows_without_binding_or_allocating():
    desc = dict(
        operation_id="e",
        expected_prompt_tokens=5,
        start_cursor=4,
        export_q={"name": "Q", "token_range": [4, 5]},
    )
    requests = [("dec", 10, 1, None, 3), ("flag", 4, 1, desc, 2)]
    controller, layers, _ = make_controller(requests)
    counts = np.array([1, 1])
    assert controller.requires_query_export(2, counts) is True
    # Inspection has no side effects: nothing bound, seen or allocated.
    assert controller.operations == {} and controller._seen_operation_ids == set()
    assert controller.q_exports()["exports"] == {}
    # Only the decode row: no export in this forward.
    assert controller.requires_query_export(1, counts) is False
    # Row outside its range (range already exported).
    advance(controller, [("dec", 10, 1, None, 3), ("flag", 5, 1, desc, 2)])
    assert controller.requires_query_export(2, counts) is False
    # Malformed descriptor fields are ignored here (validated at the forward).
    advance(controller, [("dec", 10, 1, None, 3), ("flag", 4, 1, desc, 2)])
    controller.runner.requests["flag"].sampling_params.extra_args[DESCRIPTOR_KEY] = {
        **desc,
        "export_q": {"name": "Q", "token_range": [4]},
    }
    assert controller.requires_query_export(2, counts) is False
    controller.runner.requests["flag"].sampling_params.extra_args[DESCRIPTOR_KEY] = {
        **desc,
        "operation_id": 7,
    }
    assert controller.requires_query_export(2, counts) is False
    controller.runner.requests["flag"].sampling_params.extra_args[DESCRIPTOR_KEY] = desc
    # A bound, open operation is inspected through its export_q; closed ones are not.
    metadata, positions, counts_map = forward_inputs(requests, controller)
    boundary = controller.before_forward(metadata, positions, counts_map)
    assert controller.requires_query_export(2, counts) is True
    run_attention(layers, query_batch(2), metadata)
    controller.after_forward(boundary)
    controller.results(["e"])
    assert controller.requires_query_export(2, counts) is False
    # Legacy arm: bound to the single request without a descriptor.
    plain = [("r", 4, 1, None, 2)]
    controller, layers, _ = make_controller(plain)
    controller.arm(
        expected_prompt_tokens=5,
        start_cursor=4,
        export_q={"name": "L", "token_range": [4, 5]},
    )
    assert controller.requires_query_export(1, np.array([1])) is True
    controller.operation.closed = True
    assert controller.requires_query_export(1, np.array([1])) is False


# ------------------------------------------------------------------- disarm


def test_disarm_closes_an_operation_that_never_bound_so_arming_works_again():
    requests = [("r", 0, 3, None, 2)]
    controller, layers, _ = make_controller(requests)
    controller.arm(
        expected_prompt_tokens=3, export_q={"name": "Q", "token_range": [0, 3]}
    )
    # Admission failed after arming: no forward ever ran for this operation.
    with pytest.raises(CompactionContractError, match="boundary was not reached"):
        controller.result()
    with pytest.raises(CompactionContractError, match="Call cc_result"):
        controller.arm(expected_prompt_tokens=3)
    receipt = controller.disarm()
    assert receipt["disarmed"] and receipt["never_bound"] and receipt["legacy"]
    assert receipt["failed"] is None and receipt["forward_calls"] == 0
    assert receipt["export_q"] == {"name": "Q", "token_range": [0, 3]}
    assert receipt["export_dropped"] and controller.q_exports()["exports"] == {}
    with pytest.raises(CompactionContractError, match="No armed operation"):
        controller.disarm()
    # The engine is idle: arming succeeds and the export name is free again.
    assert controller.arm(
        expected_prompt_tokens=3, export_q={"name": "Q", "token_range": [0, 3]}
    )["armed"]
    metadata, positions, counts = forward_inputs(requests, controller)
    boundary = controller.before_forward(metadata, positions, counts)
    # A bound, live operation cannot be disarmed.
    with pytest.raises(CompactionContractError, match="bound live operation"):
        controller.disarm()
    run_attention(layers, query_batch(3), metadata)
    controller.after_forward(boundary)
    assert controller.result()["q_export"]["complete"]
    # A bound but failed operation may be disarmed (closed) as well.
    controller.arm(expected_prompt_tokens=3)
    advance(controller, requests)
    controller.before_forward(metadata, positions, counts)
    controller.fail_forward(RuntimeError("kernel failed"))
    receipt = controller.disarm()
    assert not receipt["never_bound"] and "kernel failed" in receipt["failed"]
    assert controller.arm(expected_prompt_tokens=3)["armed"]


def test_disarm_addresses_descriptor_operations_by_id():
    desc = dict(operation_id="d", expected_prompt_tokens=3)
    requests = [("r", 0, 3, desc, 2)]
    controller, layers, _ = make_controller(requests)
    with pytest.raises(CompactionContractError, match="Unknown operation"):
        controller.disarm("d")
    metadata, positions, counts = forward_inputs(requests, controller)
    boundary = controller.before_forward(metadata, positions, counts)
    with pytest.raises(CompactionContractError, match="bound live operation"):
        controller.disarm("d")
    controller.after_forward(boundary)
    assert controller.results(["d"])["d"]["forward_calls"] == 1
    # A descriptor operation whose forward failed before binding is retired.
    other = dict(operation_id="f", expected_prompt_tokens=3, restore_name="missing")
    requests = [("s", 0, 3, other, 3)]
    controller.runner.requests["s"] = NS(
        mm_features=[],
        prompt_embeds=None,
        lora_request=None,
        num_computed_tokens=0,
        num_prompt_tokens=3,
        sampling_params=NS(extra_args={DESCRIPTOR_KEY: other}),
    )
    advance(controller, requests)
    metadata, positions, counts = forward_inputs(requests, controller)
    with pytest.raises(CompactionContractError):
        controller.before_forward(metadata, positions, counts)
    assert "f" not in controller.operations  # refused before it was registered
    with pytest.raises(CompactionContractError, match="Unknown operation"):
        controller.disarm("f")
