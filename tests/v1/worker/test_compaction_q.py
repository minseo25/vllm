# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for query export routing and in-worker compaction compute ops.

The attention custom op is modelled by calling each FA layer's
``compaction_q_export`` hook exactly as ``unified_attention_with_output`` does.
Methods-library calls go to small fakes injected through the registry; the
tests check the fork's routing, validation and bookkeeping, not the math.
"""

from types import SimpleNamespace as NS

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


@pytest.fixture(autouse=True)
def _clear_methods():
    yield
    compaction_q.clear_methods()


def fa_layer(index, name):
    cache = (
        torch.arange(8 * KV_HEADS * BLOCK * 2 * HEAD, dtype=torch.float32).reshape(
            8, KV_HEADS, BLOCK, 2 * HEAD
        )
        + 1000.0 * index
    )
    return NS(
        layer_name=name,
        kv_cache=cache,
        head_size=HEAD,
        num_kv_heads=KV_HEADS,
        num_heads=Q_HEADS,
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


def make_controller(requests):
    """Requests: (id, cursor, query_count, descriptor|None, state_index)."""
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
    layers = {name: fa_layer(i, name) for i, name in enumerate(FA)}
    kv_store = object.__new__(NativeKVSnapshotStore)
    kv_store._entries, kv_store._staged = {}, {}
    kv_store.layers = layers
    controller.kv_store = kv_store
    controller.q_store = QExportStore(FA)
    controller.score_store = ScoreStore()
    controller.selection_store = SelectionStore()
    controller._export_plan = None
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


def query_batch(total):
    return torch.arange(total * Q_HEADS * HEAD, dtype=torch.float32).reshape(
        total, Q_HEADS, HEAD
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
    query = query_batch(5)
    run_attention(layers, query, metadata)
    controller.after_forward(boundary)
    assert hooks_installed(layers) == [] and controller._export_plan is None
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
    assert chunk["hook_seconds"] >= 0.0
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
    # A FULL-graph decode step after the prompt installs nothing and is allowed.
    requests = [("r", 5, 1, desc, 2)]
    advance(controller, requests)
    metadata, positions, counts = forward_inputs(requests, controller, num_decodes=1)
    boundary = controller.before_forward(metadata, positions, counts, "FULL", 1)
    assert boundary is not None and hooks_installed(layers) == []
    assert controller._export_plan is None
    run_attention(layers, query_batch(1), metadata)
    controller.after_forward(boundary)
    result = controller.results(["m"])["m"]
    assert len(result["q_export_chunks"]) == 2 and result["q_export"]["complete"]
    assert result["forward_calls"] == 3


def test_full_graph_forward_with_export_rows_is_refused_before_any_write():
    desc = dict(
        operation_id="g",
        expected_prompt_tokens=3,
        export_q={"name": "Q", "token_range": [0, 3]},
    )
    requests = [("r", 0, 3, desc, 2)]
    controller, layers, _ = make_controller(requests)
    metadata, positions, counts = forward_inputs(requests, controller)
    with pytest.raises(CompactionContractError, match="FULL"):
        controller.before_forward(metadata, positions, counts, "FULL", 1)
    assert hooks_installed(layers) == [] and controller._export_plan is None
    assert "FULL" in controller.operations["g"].failed
    assert controller.q_store.describe("Q")["rows_exported"] == 0
    with pytest.raises(CompactionContractError, match="FULL"):
        controller.results(["g"])


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
    with pytest.raises(CompactionContractError, match="did not run"):
        controller.results(["m"])


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
    controller.fail_forward(RuntimeError("kernel failed"))
    assert hooks_installed(layers) == [] and controller._export_plan is None
    assert controller.q_store.describe("Q")["rows_exported"] == 0


# ------------------------------------------------------------ store units


def test_export_store_validates_ranges_dtypes_and_double_writes():
    store = QExportStore(FA)
    with pytest.raises(CompactionContractError, match="start < end"):
        store.open("bad", token_range=[3, 3])
    store.open("Q", token_range=[2, 4])
    with pytest.raises(CompactionContractError, match="reused"):
        store.open("Q", token_range=[0, 1])
    with pytest.raises(CompactionContractError, match="Unsupported query dtype"):
        store.write("Q", "fa0", torch.zeros(1, 2, 3, dtype=torch.float8_e4m3fn), 2)
    with pytest.raises(CompactionContractError, match="outside token_range"):
        store.write("Q", "fa0", torch.zeros(3, 2, 3), 2)
    with pytest.raises(CompactionContractError, match="Unexpected query export layer"):
        store.write("Q", "zz", torch.zeros(1, 2, 3), 2)
    with pytest.raises(CompactionContractError, match="head layout is inconsistent"):
        store.write(
            "Q",
            "fa0",
            torch.ones(2, 2, 3),
            2,
            layout={"num_heads": 3, "num_kv_heads": 2, "head_size": 3},
        )
    store.write(
        "Q",
        "fa0",
        torch.ones(2, 2, 3),
        2,
        layout={"num_heads": 2, "num_kv_heads": 1, "head_size": 3},
    )
    assert store.describe("Q")["layouts"] == {
        "fa0": {"num_heads": 2, "num_kv_heads": 1, "head_size": 3, "group_size": 2}
    }
    with pytest.raises(CompactionContractError, match="shape/dtype changed"):
        store.write("Q", "fa0", torch.ones(1, 2, 4), 2)
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
        receipt={},
    )
    with pytest.raises(CompactionContractError, match="already exported"):
        store.write("Q", "fa0", torch.ones(1, 2, 3), 3)
    assert store.positions("Q").tolist() == [7, 8]
    assert store.list()["total_host_bytes"] == 2 * 2 * 2 * 3 * 4
    with pytest.raises(CompactionContractError, match="Unknown query export"):
        store.describe("nope")


def test_export_plan_rejects_rows_outside_the_token_batch():
    store = QExportStore(FA)
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


class FakeScores:
    def __init__(self):
        self.calls = []

    def h2o_scores(self, q, k, *, scale, causal, q_positions, k_positions, chunk):
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
            )
        )
        tokens, heads = k.shape[0], k.shape[1]
        return (
            torch.arange(tokens, dtype=torch.float32)[None, :].repeat(heads, 1)
            + 0.5 * torch.arange(heads, dtype=torch.float32)[:, None]
        )

    def kvzip_scores(self, q_ref, k, *, scale, k_ref=None, chunk):
        self.calls.append(
            dict(
                method="kvzip",
                q=q_ref.clone(),
                k=k.clone(),
                k_ref=None if k_ref is None else k_ref.clone(),
                scale=scale,
                chunk=chunk,
            )
        )
        return k.float().abs().sum(-1).T


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
    """Mirrors ``profiling.compaction_methods.am.AMResult`` for the shared case."""

    def __init__(self, chosen, k, v):
        index = torch.tensor(chosen)
        self.indices = index[None, :].repeat(k.shape[1], 1)  # [Hkv, t]
        self.k_c = k[index].transpose(0, 1).contiguous()  # [Hkv, t, D]
        self.v_c = (v[index] * 2).transpose(0, 1).contiguous()
        self.beta = None
        self.shared = True
        self.variant = "fake_am_rmskeys_ols_nobias_uniform"
        self.diagnostics = {"output_error_after": torch.zeros(k.shape[1])}

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

    def compact(self, q_ref, k, v, budget, *, bias, head_budget, protected, **kwargs):
        self.calls.append(
            dict(
                budget=budget,
                bias=bias,
                head_budget=head_budget,
                protected=list(protected),
                **kwargs,
            )
        )
        chosen = list(protected)
        for i in range(k.shape[0] - 1, -1, -1):
            if len(chosen) >= budget:
                break
            if i not in chosen:
                chosen.append(i)
        return FakeAMResult(sorted(chosen), k, v)


@pytest.fixture
def fakes():
    bundle = NS(scores=FakeScores(), select=FakeSelect(), am=FakeAM())
    compaction_q.register_methods(
        scores=bundle.scores, select=bundle.select, am=bundle.am
    )
    return bundle


def resident_controller(cursor=6):
    controller, layers, _ = make_controller([("ctx", cursor, 1, None, 2)])
    return controller, layers


def fill_export(controller, name, token_range, base=0.0):
    store = controller.q_store
    store.open(name, token_range=token_range, request_id="ctx")
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


def test_kv_score_h2o_reads_cache_keys_and_stores_float32_cpu_scores(fakes):
    controller, layers = resident_controller()
    queries = fill_export(controller, "Q", [2, 6])
    receipt = controller.kv_score(
        "S", q_export="Q", method="h2o", request_id="ctx", params={"chunk": 8}
    )
    assert receipt["method"] == "h2o" and receipt["params"] == {"chunk": 8}
    assert receipt["shapes"] == {name: [KV_HEADS, 6] for name in FA}
    assert receipt["source"]["kind"] == "resident_request"
    assert receipt["source"]["key_range"] == [0, 6]
    assert receipt["source"]["q_token_range"] == [2, 6]
    assert receipt["source"]["scale"] == {name: HEAD**-0.5 for name in FA}
    assert receipt["source"]["k_ref"] is None
    assert receipt["source"]["normalisation"] == "causal_over_scored_keys"
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
        assert call["q_positions"].tolist() == [2, 3, 4, 5]
        assert call["k_positions"].tolist() == list(range(6))
    assert controller.score_store.key_token_indices("S") == {
        name: list(range(6)) for name in FA
    }
    listed = controller.scores()["scores"]["S"]
    assert listed["keys_per_layer"] == {name: 6 for name in FA}
    assert "layers" not in listed
    assert controller.score_drop("S")["scores"] == {}


def test_kv_score_kvzip_from_per_layer_snapshot_keeps_each_layers_key_indices(fakes):
    controller, layers = resident_controller()
    fill_export(controller, "Q", [4, 6])
    row, metadata, computed = controller._resident("ctx")
    controller.kv_store.capture(
        {"name": "snap", "layer_token_indices": {"fa0": [0, 1, 5], "fa1": [2, 3, 4]}},
        "ctx",
        row,
        metadata,
        computed,
    )
    receipt = controller.kv_score("S", q_export="Q", method="kvzip", kv_snapshot="snap")
    assert receipt["source"]["kind"] == "kv_snapshot"
    assert receipt["source"]["name"] == "snap" and receipt["source"]["cursor"] == 6
    assert controller.score_store.key_token_indices("S") == {
        "fa0": [0, 1, 5],
        "fa1": [2, 3, 4],
    }
    # k_ref = the repeat request's own keys for its exported rows (tokens 4, 5).
    assert receipt["source"]["k_ref"] == {"request_id": "ctx", "token_range": [4, 6]}
    assert receipt["source"]["normalisation"] == "context_plus_causal_repeat_keys"
    assert receipt["params"] == {"chunk": compaction_q.DEFAULT_CHUNK}
    for call, name in zip(fakes.scores.calls, FA):
        assert call["method"] == "kvzip" and call["chunk"] == compaction_q.DEFAULT_CHUNK
        assert torch.equal(
            call["k"], controller.kv_store.snapshot_rows("snap")[name][..., :HEAD]
        )
        assert torch.equal(call["k_ref"], cache_rows(layers[name], [4, 5])[..., :HEAD])
    with pytest.raises(CompactionContractError, match="same key tokens"):
        controller.kv_select("sel", scores="S", budget_tokens=2)
    # Without k_ref the library's named deviation must be opted into explicitly.
    controller.q_store.drop("Q")
    controller.q_store.open("Q", token_range=[4, 6])  # no request bound
    for layer in FA:
        controller.q_store.write("Q", layer, query_batch(2), 4)
    controller.q_store.commit(
        "Q",
        cursor_start=4,
        cursor_end=6,
        positions=torch.arange(4, 6),
        layers_written=set(FA),
        receipt={},
    )
    with pytest.raises(CompactionContractError, match="k_ref"):
        controller.kv_score("S3", q_export="Q", method="kvzip", kv_snapshot="snap")
    deviation = controller.kv_score(
        "S3",
        q_export="Q",
        method="kvzip",
        kv_snapshot="snap",
        params={"context_only_normalisation": True},
    )
    assert deviation["source"]["normalisation"] == "context_only"
    assert fakes.scores.calls[-1]["k_ref"] is None
    with pytest.raises(CompactionContractError, match="resident requests only"):
        controller.kv_score(
            "S2", q_export="Q", method="kvzip", kv_snapshot="snap", key_range=[0, 2]
        )


def test_kv_select_returns_equal_counts_capture_spec_and_keeps_protected(fakes):
    controller, layers = resident_controller()
    fill_export(controller, "Q", [0, 6])
    controller.kv_score("S", q_export="Q", method="h2o", request_id="ctx")
    shared = controller.kv_select("shared", scores="S", budget_tokens=3, protected=[0])
    assert shared["token_indices"] == [0, 4, 5]
    assert shared["retained_tokens"] == 3 and shared["scored_tokens"] == 6
    assert shared["layer_token_indices"] == {name: [0, 4, 5] for name in FA}
    assert shared["capture_spec"] == {"name": "shared", "token_indices": [0, 4, 5]}
    assert shared["method"] == "h2o" and shared["layer_order"] == list(FA)
    assert fakes.select.calls[-1] == {"budget": 3, "protected": [0]}
    per_layer = controller.kv_select(
        "pl", scores="S", budget_tokens=2, policy="per_layer", aggregate="mean"
    )
    assert per_layer["token_indices"] is None
    assert per_layer["capture_spec"] == {
        "name": "pl",
        "layer_token_indices": {name: [4, 5] for name in FA},
    }
    snapshot = controller.kv_capture(per_layer["capture_spec"], "ctx")
    assert snapshot["selection_policy"] == "per_layer"
    assert snapshot["retained_tokens"] == 2 and snapshot["source_cursor"] == 6
    assert snapshot["source_position_offset"] == 0
    for name in FA:
        assert torch.equal(
            controller.kv_store.snapshot_rows("pl")[name],
            cache_rows(layers[name], [4, 5]),
        )
    assert set(controller.kv_selections()["selections"]) == {"shared", "pl"}
    with pytest.raises(CompactionContractError, match="not among the scored keys"):
        controller.kv_select("x", scores="S", budget_tokens=2, protected=[9])
    with pytest.raises(CompactionContractError, match="overwritten"):
        controller.kv_select("shared", scores="S", budget_tokens=2)
    with pytest.raises(CompactionContractError, match="policy or aggregate"):
        controller.kv_select("y", scores="S", budget_tokens=2, policy="top")
    with pytest.raises(CompactionContractError, match="exceeds budget"):
        controller.kv_select("z", scores="S", budget_tokens=1, protected=[0, 1])


def test_kv_fit_am_registers_original_keys_with_fitted_values_that_import(fakes):
    controller, layers = resident_controller()
    fill_export(controller, "Q", [0, 6])
    receipt = controller.kv_fit_am(
        "AM",
        q_export="Q",
        budget_tokens=3,
        request_id="ctx",
        protected=[1],
        params={"ridge": 0.1},
    )
    assert receipt["synthetic"] is True
    assert receipt["method"]["name"] == "fake_am_rmskeys_ols_nobias_uniform"
    assert receipt["method"]["alias"] == "am_nobias_uniform"
    params = receipt["method"]["params"]
    assert (params["ridge"], params["bias"], params["head_budget"]) == (
        0.1,
        False,
        "uniform",
    )
    assert params["protected"] == [1] and params["budget_tokens"] == 3
    assert params["chunk"] == compaction_q.DEFAULT_CHUNK
    assert params["scale"] == {name: HEAD**-0.5 for name in FA}
    assert params["values_cast"] == ["torch.float32->torch.float32"]
    inputs = receipt["method"]["inputs"]
    assert inputs["q_export"] == "Q" and inputs["kv_digests"] is None
    assert inputs["source"]["kind"] == "resident_request"
    assert inputs["query_convention"]["stage"] == "post_qk_norm_post_rope_unscaled"
    assert receipt["method"]["diagnostics"] == {
        name: {"output_error_after": [0.0, 0.0]} for name in FA
    }
    assert receipt["layer_token_indices"] == {name: [1, 4, 5] for name in FA}
    assert receipt["selection_policy"] == "per_layer"
    assert (receipt["retained_tokens"], receipt["source_cursor"]) == (3, 6)
    assert receipt["source_position_offset"] == 0 and receipt["request_id"] == "ctx"
    assert params["protected_values"] == "original"
    rows = controller.kv_store.snapshot_rows("AM")
    for name in FA:
        source = cache_rows(layers[name], [1, 4, 5])
        assert torch.equal(rows[name][..., :HEAD], source[..., :HEAD])
        # Protected token 1 keeps its original values; the others carry the fit.
        assert torch.equal(rows[name][0, :, HEAD:], source[0, :, HEAD:])
        assert torch.equal(rows[name][1:, :, HEAD:], source[1:, :, HEAD:] * 2)
    refit = controller.kv_fit_am(
        "AMrefit",
        q_export="Q",
        budget_tokens=2,
        request_id="ctx",
        protected=[1],
        params={"refit_protected": True},
    )
    assert refit["method"]["params"]["protected_values"] == "fitted"
    for name in FA:
        source = cache_rows(layers[name], [1, 5])
        refit_rows = controller.kv_store.snapshot_rows("AMrefit")[name]
        assert torch.equal(refit_rows[..., HEAD:], source[..., HEAD:] * 2)
    call = fakes.am.calls[0]
    assert (call["budget"], call["protected"], call["ridge"]) == (3, [1], 0.1)
    assert (call["bias"], call["head_budget"]) == (False, "uniform")
    assert (call["scale"], call["chunk"]) == (HEAD**-0.5, compaction_q.DEFAULT_CHUNK)
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
    digested = controller.kv_fit_am(
        "AM2",
        q_export="Q",
        budget_tokens=2,
        request_id="ctx",
        params={"digest_inputs": True},
    )
    assert set(digested["method"]["inputs"]["kv_digests"]) == set(FA)


def test_compute_ops_fail_closed_on_missing_library_and_bad_outputs(monkeypatch):
    controller, layers = resident_controller()
    fill_export(controller, "Q", [0, 6])
    monkeypatch.setattr(
        compaction_q, "METHODS_PACKAGE", "profiling_missing_for_test.compaction_methods"
    )
    with pytest.raises(CompactionContractError, match="unavailable"):
        controller.kv_score("S", q_export="Q", method="h2o", request_id="ctx")
    assert controller.scores()["scores"] == {}
    bad_outputs = {
        "shape": lambda q, k, **kw: torch.zeros(3, 3),
        "nan": lambda q, k, **kw: torch.full((KV_HEADS, k.shape[0]), float("nan")),
        "dtype": lambda q, k, **kw: torch.zeros(KV_HEADS, k.shape[0], dtype=torch.long),
    }
    for label, fake in bad_outputs.items():
        compaction_q.register_methods(scores=NS(h2o_scores=fake, kvzip_scores=fake))
        with pytest.raises(CompactionContractError, match="finite"):
            controller.kv_score(label, q_export="Q", method="h2o", request_id="ctx")
    assert controller.scores()["scores"] == {}
    compaction_q.register_methods(scores=FakeScores())
    with pytest.raises(CompactionContractError, match="Unknown scoring method"):
        controller.kv_score("S", q_export="Q", method="snapkv", request_id="ctx")
    with pytest.raises(CompactionContractError, match="Unknown score params"):
        controller.kv_score(
            "S", q_export="Q", method="h2o", request_id="ctx", params={"temperature": 1}
        )
    with pytest.raises(CompactionContractError, match="exactly one of"):
        controller.kv_score("S", q_export="Q", method="h2o")
    with pytest.raises(CompactionContractError, match="not resident"):
        controller.kv_score("S", q_export="Q", method="h2o", request_id="ghost")
    with pytest.raises(CompactionContractError, match="exceeds the consumed prefix"):
        controller.kv_score(
            "S", q_export="Q", method="h2o", request_id="ctx", key_range=[0, 7]
        )
    controller.q_store.open("P", token_range=[0, 6], request_id="ctx")
    with pytest.raises(CompactionContractError, match="incomplete"):
        controller.kv_score("S", q_export="P", method="h2o", request_id="ctx")
    controller.runner.execute_model_state = object()
    with pytest.raises(CompactionContractError, match="between forward and sampling"):
        controller.kv_score("S", q_export="Q", method="h2o", request_id="ctx")
    controller.runner.execute_model_state = None
    controller.kv_score("S", q_export="Q", method="h2o", request_id="ctx")
    with pytest.raises(CompactionContractError, match="overwritten"):
        controller.kv_score("S", q_export="Q", method="h2o", request_id="ctx")
    # Selection library outputs are validated too.
    compaction_q.register_methods(
        select=NS(uniform_token_budget=lambda s, b, **kw: [[0] for _ in s])
    )
    with pytest.raises(CompactionContractError, match="match the budget"):
        controller.kv_select("sel", scores="S", budget_tokens=2)
    compaction_q.register_methods(
        select=NS(uniform_token_budget=lambda s, b, **kw: [[0, 1], [2, 3]])
    )
    with pytest.raises(CompactionContractError, match="identical layer lists"):
        controller.kv_select("sel", scores="S", budget_tokens=2, policy="shared")
    compaction_q.register_methods(select=FakeSelect())
    with pytest.raises(CompactionContractError, match="exceeds the scored keys"):
        controller.kv_select("big", scores="S", budget_tokens=7)
    # Library exceptions surface as contract errors with their origin.
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
        controller.kv_select("sel", scores="S", budget_tokens=2)
    assert controller.kv_selections()["selections"] == {}

    # AM outputs are validated: beta, shared layout, counts, keys, value shapes.
    def am_result(k, v, chosen, *, beta=None, k_shift=0.0, v_cols=None, shared=True):
        rows_k = k[chosen] + k_shift
        rows_v = v[chosen] if v_cols is None else v[chosen][..., :v_cols]
        return NS(
            beta=beta,
            shared=shared,
            variant="fake",
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
    }
    for build, message in faults.values():
        compaction_q.register_methods(
            am=NS(compact=lambda q, k, v, b, build=build, **kw: build(k, v))
        )
        with pytest.raises(CompactionContractError, match=message):
            controller.kv_fit_am("AM", q_export="Q", budget_tokens=2, request_id="ctx")
    compaction_q.register_methods(am=FakeAM())
    with pytest.raises(CompactionContractError, match="exceeds the source keys"):
        controller.kv_fit_am("AM", q_export="Q", budget_tokens=7, request_id="ctx")
    with pytest.raises(CompactionContractError, match="Unknown am params"):
        controller.kv_fit_am(
            "AM", q_export="Q", budget_tokens=2, request_id="ctx", params={"iters": 3}
        )
    assert controller.kv_store.list()["snapshots"] == {}


def test_kv_capture_of_a_resident_request_matches_boundary_capture_semantics():
    controller, layers = resident_controller()
    receipt = controller.kv_capture({"name": "res", "token_indices": [0, 3, 5]}, "ctx")
    assert receipt["request_id"] == "ctx" and receipt["source_cursor"] == 6
    assert receipt["selection_policy"] == "shared" and receipt["synthetic"] is False
    for name in FA:
        assert torch.equal(
            controller.kv_store.snapshot_rows("res")[name],
            cache_rows(layers[name], [0, 3, 5]),
        )
    controller._position_rules["ctx"] = (2, 40)
    offset = controller.kv_capture({"name": "off", "token_indices": [0]}, "ctx")
    assert offset["source_position_offset"] == 40
    with pytest.raises(ValueError, match="indices"):
        controller.kv_capture({"name": "late", "token_indices": [6]}, "ctx")
    controller.arm(
        expected_prompt_tokens=9,
        capture_kv={"name": "pending", "token_indices": [0]},
    )
    with pytest.raises(CompactionContractError, match="reserved by another operation"):
        controller.kv_capture({"name": "pending", "token_indices": [0]}, "ctx")


def test_scale_is_head_size_rule_cross_checked_against_the_kernel(fakes):
    controller, layers = resident_controller()
    fill_export(controller, "Q", [0, 6])
    layers["fa0"].impl.scale = 0.5  # a kernel scale that is not head_size ** -0.5
    with pytest.raises(CompactionContractError, match="differs from head_size"):
        controller.kv_score("S", q_export="Q", method="h2o", request_id="ctx")
    assert controller.scores()["scores"] == {} and fakes.scores.calls == []
    receipt = controller.kv_score(
        "S", q_export="Q", method="h2o", request_id="ctx", params={"scale": 0.5}
    )
    assert receipt["source"]["scale"] == {name: 0.5 for name in FA}
    assert all(call["scale"] == 0.5 for call in fakes.scores.calls)
    with pytest.raises(CompactionContractError, match="positive number"):
        controller.kv_score(
            "S2", q_export="Q", method="h2o", request_id="ctx", params={"scale": -1}
        )
