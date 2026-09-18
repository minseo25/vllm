# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Protect exact KV retention from layout, alias, and partial-write mistakes."""

from types import SimpleNamespace as NS

import pytest
import torch

from vllm.v1.worker.compaction_kv import NativeKVSnapshotStore


def fixture_store():
    backend = NS(get_name=lambda: "FLASH_ATTN")
    # Logical BHND cache with physically NHD strides.
    layers = {
        key: NS(
            kv_cache=torch.arange(8 * 4 * 2 * 6, dtype=torch.float32)
            .reshape(8, 4, 2, 6)
            .transpose(1, 2)
            .clone(),
            get_attn_backend=lambda: backend,
            head_size=3,
            num_kv_heads=2,
        )
        for key in ("a", "b")
    }
    runner = NS(
        compilation_config=NS(static_forward_context=layers),
        kv_cache_config=NS(
            kv_cache_groups=[
                NS(
                    kv_cache_spec=None,
                    layer_names=list(layers),
                )
            ]
        ),
    )
    metadata = {k: NS(block_table=torch.tensor([[2, 3, 4], [5, 6, 7]])) for k in layers}
    return NativeKVSnapshotStore(runner), layers, metadata


def test_selected_original_values_survive_release_and_different_destination_pages():
    store, layers, metadata = fixture_store()
    selected = [0, 1, 8, 9]
    source = layers["a"].kv_cache.clone()
    receipt = store.capture(
        {"name": "selected", "token_indices": selected},
        "source",
        0,
        metadata,
        10,
    )
    expected = source[torch.tensor([2, 2, 4, 4]), :, torch.tensor([0, 1, 0, 1]), :]
    for layer in layers.values():
        layer.kv_cache.fill_(-99)
    result = store.restore("selected", "target", 1, metadata, 4)
    assert result["scratch_prefill_tokens"] == 4
    assert result["source_cursor"] == 10
    for layer in layers.values():
        assert torch.equal(layer.kv_cache[5, :, :4, :].transpose(0, 1), expected)
        assert (layer.kv_cache[2] == -99).all()
    assert store.audit("selected")["digest"] == receipt["digest"]
    # Roundtrip captures the destination, rather than trusting the copy receipt.
    copied = store.capture(
        {"name": "copied", "token_indices": list(range(4))},
        "target",
        1,
        metadata,
        4,
    )
    assert receipt["digest"] == copied["digest"]
    assert store.list()["total_cpu_snapshot_bytes"] == 2 * receipt["bytes"]


def test_invalid_second_layer_causes_no_first_layer_write():
    store, layers, metadata = fixture_store()
    store.capture({"name": "s", "token_indices": [0, 1]}, "s", 0, metadata, 2)
    before = layers["a"].kv_cache.clone()
    metadata["b"].block_table[1, 0] = 0
    with pytest.raises(ValueError, match="NULL"):
        store.restore("s", "d", 1, metadata, 2)
    assert torch.equal(layers["a"].kv_cache, before)


@pytest.mark.parametrize("indices", [[1, 0], [0, 0], [-1], [4], [True], []])
def test_selection_rejects_duplicates_future_tokens_and_noninteger_indices(indices):
    store, _, metadata = fixture_store()
    with pytest.raises(ValueError, match="indices"):
        store.capture({"name": "s", "token_indices": indices}, "s", 0, metadata, 4)


def test_import_cannot_silently_change_boundary_or_overwrite_snapshot():
    store, _, metadata = fixture_store()
    spec = {"name": "s", "token_indices": [0, 1]}
    store.capture(spec, "s", 0, metadata, 4)
    with pytest.raises(ValueError, match="scratch prefix"):
        store.restore("s", "d", 1, metadata, 3)
    with pytest.raises(ValueError, match="overwritten"):
        store.capture(spec, "s", 0, metadata, 4)


def test_staging_is_owned_and_explicit_audit_detects_mutation():
    store, layers, metadata = fixture_store()
    store.capture({"name": "s", "token_indices": [0, 1]}, "s", 0, metadata, 4)
    store.stage("s", "cpu")
    layers["a"].kv_cache.zero_()
    store.audit("s")
    store._staged["s"]["a"].zero_()
    with pytest.raises(ValueError, match="modified"):
        store.audit("s")


def test_retain_all_roundtrip_is_identity():
    store, layers, metadata = fixture_store()
    indices = list(range(10))
    before = {name: layer.kv_cache.clone() for name, layer in layers.items()}
    store.capture({"name": "all", "token_indices": indices}, "s", 0, metadata, 10)
    store.restore("all", "s", 0, metadata, 10)
    assert all(
        torch.equal(layer.kv_cache, before[name]) for name, layer in layers.items()
    )


def test_offset_source_imports_and_records_its_absolute_next_position():
    """Chained imports: rows keep their RoPE positions; the controller checks
    position_offset == source_absolute_next_position - computed at import."""
    store, layers, metadata = fixture_store()
    source = layers["a"].kv_cache.clone()
    entry = store.capture(
        {"name": "offset", "token_indices": [0, 1]},
        "s",
        0,
        metadata,
        4,
        source_position_offset=100,
    )
    assert entry["source_absolute_next_position"] == 104
    store.audit("offset")
    receipt = store.restore("offset", "d", 1, metadata, 2)
    assert receipt["source_cursor"] == 4
    expected = source[torch.tensor([2, 2]), :, torch.tensor([0, 1]), :]
    assert torch.equal(layers["a"].kv_cache[5, :, :2, :].transpose(0, 1), expected)


def test_capture_spec_may_carry_a_self_describing_method_record():
    store, _, metadata = fixture_store()
    method = {
        "name": "streamingllm",
        "params": {"recent": 1},
        "inputs": {"scores": None},
    }
    receipt = store.capture(
        {"name": "ev", "token_indices": [0, 3], "method": method}, "s", 0, metadata, 4
    )
    assert receipt["method"] == method and receipt["synthetic"] is False
    method["params"]["recent"] = 99  # the entry holds its own copy
    assert store.describe("ev")["method"]["params"]["recent"] == 1
    with pytest.raises(ValueError, match="method must carry"):
        store.capture(
            {"name": "bad", "token_indices": [0], "method": {"name": "x"}},
            "s",
            0,
            metadata,
            4,
        )
    with pytest.raises(ValueError, match="JSON"):
        store.capture(
            {
                "name": "bad",
                "token_indices": [0],
                "method": {"name": "x", "params": {"t": object()}, "inputs": {}},
            },
            "s",
            0,
            metadata,
            4,
        )
    assert set(store.list()["snapshots"]) == {"ev"}


# ---------------------------------------------------------------- per-layer


def test_per_layer_capture_restores_each_layers_own_rows_and_records_policy():
    store, layers, metadata = fixture_store()
    source = {name: layer.kv_cache.clone() for name, layer in layers.items()}
    receipt = store.capture(
        {"name": "pl", "layer_token_indices": {"a": [0, 1], "b": [8, 9]}},
        "source",
        0,
        metadata,
        10,
    )
    assert receipt["selection_policy"] == "per_layer"
    assert receipt["token_indices"] is None
    assert receipt["layer_token_indices"] == {"a": [0, 1], "b": [8, 9]}
    assert receipt["retained_tokens"] == 2
    assert receipt["synthetic"] is False and receipt["method"] is None
    expected = {
        "a": source["a"][torch.tensor([2, 2]), :, torch.tensor([0, 1]), :],
        "b": source["b"][torch.tensor([4, 4]), :, torch.tensor([0, 1]), :],
    }
    for layer in layers.values():
        layer.kv_cache.fill_(-99)
    result = store.restore("pl", "target", 1, metadata, 2)
    assert result["selection_policy"] == "per_layer" and result["synthetic"] is False
    for name, layer in layers.items():
        assert torch.equal(layer.kv_cache[5, :, :2, :].transpose(0, 1), expected[name])
    legacy = store.capture(
        {"name": "shared", "token_indices": [0, 1]}, "source", 1, metadata, 2
    )
    assert legacy["selection_policy"] == "shared"
    assert legacy["layer_token_indices"] == {"a": [0, 1], "b": [0, 1]}


@pytest.mark.parametrize(
    "per_layer,message",
    [
        ({"a": [0, 1], "b": [0]}, "equal counts"),
        ({"a": [0, 1]}, "every FA layer"),
        ({"a": [0, 1], "b": [0, 1], "c": [0, 1]}, "every FA layer"),
        ({"a": [1, 0], "b": [0, 1]}, "indices"),
        ({"a": [0, 12], "b": [0, 1]}, "indices"),
        ([0, 1], "every FA layer"),
    ],
)
def test_per_layer_capture_rejects_unequal_missing_or_invalid_layer_lists(
    per_layer, message
):
    store, layers, metadata = fixture_store()
    before = {name: layer.kv_cache.clone() for name, layer in layers.items()}
    with pytest.raises(ValueError, match=message):
        store.capture(
            {"name": "bad", "layer_token_indices": per_layer}, "s", 0, metadata, 10
        )
    assert store.list()["snapshots"] == {}
    assert all(torch.equal(layers[n].kv_cache, before[n]) for n in layers)


def test_capture_spec_must_use_exactly_one_selection_form():
    store, _, metadata = fixture_store()
    with pytest.raises(ValueError, match="token_indices or layer_token_indices"):
        store.capture(
            {"name": "x", "token_indices": [0], "layer_token_indices": {}},
            "s",
            0,
            metadata,
            4,
        )


# ----------------------------------------------------------------- register


def method_record(**extra):
    return {
        "name": "fake_fit",
        "params": {"ridge": 0.0},
        "inputs": {"q": "abc"},
        **extra,
    }


def test_registered_synthetic_rows_import_exactly_like_a_capture():
    store, layers, metadata = fixture_store()
    captured = store.capture(
        {"name": "cap", "token_indices": [0, 5]}, "history", 0, metadata, 6
    )
    rows = {name: t.clone() for name, t in store.snapshot_rows("cap").items()}
    receipt = store.register(
        "syn",
        rows,
        source_cursor=6,
        source_position_offset=0,
        retained_tokens=2,
        method=method_record(),
        token_indices=[0, 5],
        request_id="history",
    )
    assert receipt["synthetic"] is True
    assert receipt["method"] == method_record()
    assert receipt["digest"] == captured["digest"]
    assert receipt["digest_verified_at"] == "register"
    assert receipt["selection_policy"] == "shared"
    assert receipt["token_indices"] == [0, 5]
    assert receipt["source_absolute_next_position"] == 6
    # Source mutation after register cannot leak into the immutable entry.
    rows["a"].fill_(123)
    assert store.audit("syn")["digest"] == captured["digest"]
    for layer in layers.values():
        layer.kv_cache.fill_(-1)
    store.restore("cap", "t1", 0, metadata, 2)
    from_capture = {n: layer.kv_cache.clone() for n, layer in layers.items()}
    for layer in layers.values():
        layer.kv_cache.fill_(-1)
    result = store.restore("syn", "t2", 0, metadata, 2)
    assert result["synthetic"] is True
    assert all(torch.equal(layers[n].kv_cache, from_capture[n]) for n in layers)


def test_registered_rows_can_be_fitted_values_with_per_layer_indices():
    store, layers, metadata = fixture_store()
    tensors = {
        name: torch.full((3, 2, 6), float(i + 1), dtype=torch.float32)
        for i, name in enumerate(layers)
    }
    receipt = store.register(
        "fit",
        tensors,
        source_cursor=10,
        source_position_offset=0,
        retained_tokens=3,
        method=method_record(),
        layer_token_indices={"a": [0, 1, 9], "b": [2, 3, 4]},
    )
    assert receipt["selection_policy"] == "per_layer"
    assert receipt["layer_token_indices"] == {"a": [0, 1, 9], "b": [2, 3, 4]}
    store.restore("fit", "t", 1, metadata, 3)
    assert torch.all(layers["a"].kv_cache[5, :, :3, :] == 1)
    assert torch.all(layers["b"].kv_cache[5, :, :3, :] == 2)
    assert torch.all(layers["a"].kv_cache[5, :, 3:, :] != 1)


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda t: {**t, "a": t["a"][:, :1]}, "shape differs"),
        (lambda t: {**t, "a": t["a"][:1]}, "shape differs"),
        (lambda t: {**t, "a": t["a"].to(torch.float16)}, "dtype differs"),
        (lambda t: {**t, "a": t["a"].clone().fill_(float("nan"))}, "Non-finite"),
        (lambda t: {**t, "a": t["a"].clone().fill_(float("inf"))}, "Non-finite"),
        (lambda t: {"a": t["a"]}, "exactly the FA layers"),
        (lambda t: {**t, "c": t["a"]}, "exactly the FA layers"),
        (lambda t: {**t, "a": t["a"].tolist()}, "shape differs"),
    ],
)
def test_register_rejects_layout_dtype_and_finiteness_mismatches(mutate, message):
    store, layers, metadata = fixture_store()
    good = {name: torch.zeros((2, 2, 6), dtype=torch.float32) for name in layers}
    with pytest.raises(ValueError, match=message):
        store.register(
            "bad",
            mutate(good),
            source_cursor=4,
            source_position_offset=0,
            retained_tokens=2,
            method=method_record(),
        )
    assert store.list()["snapshots"] == {}


@pytest.mark.parametrize(
    "kwargs,message",
    [
        (dict(method={"name": "x"}), "name, params and inputs"),
        (dict(method=method_record(params={"t": object()})), "JSON"),
        (dict(retained_tokens=0), "1..source_cursor"),
        (dict(retained_tokens=5, source_cursor=4), "1..source_cursor"),
        (dict(source_position_offset=-1), "nonnegative"),
        (dict(token_indices=[0, 1, 2]), "count retained_tokens"),
        (dict(layer_token_indices={"a": [0, 1], "b": [0]}), "equal counts"),
        (
            dict(token_indices=[0, 1], layer_token_indices={"a": [0, 1], "b": [0, 1]}),
            "not both",
        ),
        (dict(name=""), "nonempty"),
        (dict(name="cap"), "overwritten"),
    ],
)
def test_register_rejects_bad_bookkeeping(kwargs, message):
    store, layers, metadata = fixture_store()
    store.capture({"name": "cap", "token_indices": [0]}, "s", 0, metadata, 2)
    good = {name: torch.zeros((2, 2, 6), dtype=torch.float32) for name in layers}
    call = dict(
        name="syn",
        source_cursor=4,
        source_position_offset=0,
        retained_tokens=2,
        method=method_record(),
    )
    call.update(kwargs)
    name = call.pop("name")
    with pytest.raises(ValueError, match=message):
        store.register(name, good, **call)
    assert set(store.list()["snapshots"]) == {"cap"}


# ---------------------------------------------------------------- read_rows


def test_read_rows_returns_device_resident_rows_equal_to_capture():
    store, layers, metadata = fixture_store()
    captured = store.capture(
        {"name": "cap", "token_indices": [1, 4, 9]}, "s", 1, metadata, 10
    )
    rows = store.read_rows(1, metadata, [1, 4, 9], computed_tokens=10)
    assert set(rows) == {"a", "b"}
    for name, tensor in rows.items():
        assert tensor.device == layers[name].kv_cache.device
        assert tensor.shape == (3, 2, 6)
        assert torch.equal(tensor, store.snapshot_rows("cap")[name])
    # Gathers are copies: mutating them leaves the cache alone.
    before = layers["a"].kv_cache.clone()
    rows["a"].fill_(-5)
    assert torch.equal(layers["a"].kv_cache, before)
    subset = store.read_rows(
        1, metadata, {"a": [0], "b": [9]}, computed_tokens=10, layers=["b"]
    )
    assert set(subset) == {"b"}
    # Row 1 maps logical blocks to physical [5, 6, 7]; token 9 is block 7 slot 1.
    assert torch.equal(subset["b"][0], layers["b"].kv_cache[7, :, 1, :])
    assert store.list()["snapshots"]["cap"]["digest"] == captured["digest"]


@pytest.mark.parametrize(
    "indices,kwargs,message",
    [
        ([10], dict(computed_tokens=10), "indices"),
        ([2, 1], {}, "indices"),
        ([-1], {}, "indices"),
        ({"a": [0]}, {}, "exactly the FA layers"),
        ([0], dict(layers=["zz"]), "distinct FA layers"),
        ([0], dict(layers=["a", "a"]), "distinct FA layers"),
        ([12], {}, "exceeds allocated"),
    ],
)
def test_read_rows_rejects_unconsumed_or_malformed_selections(indices, kwargs, message):
    store, _, metadata = fixture_store()
    with pytest.raises(ValueError, match=message):
        store.read_rows(0, metadata, indices, **kwargs)
