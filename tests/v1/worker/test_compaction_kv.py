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


# --------------------------------------------------------------------- bias


def bias_store(with_bias=True):
    """Layers on a bias-capable backend carrying ``bias_cache`` [blocks, kv, slots]."""
    store, layers, metadata = fixture_store()
    backend = NS(get_name=lambda: "TRITON_ATTN")
    for layer in layers.values():
        layer.get_attn_backend = lambda backend=backend: backend
        if with_bias:
            layer.bias_cache = torch.zeros(8, 2, 4)
    return store, layers, metadata


def beta_rows(layers, value=1.0):
    """Token-major [3, kv_heads] biases with the frame row (row 0) at zero."""
    return {
        name: torch.tensor([[0.0, 0.0], [0.5, -1.0], [2.0, 3.0]]) * value
        for name in layers
    }


def test_registered_beta_imports_into_the_bias_buffer_at_the_row_slots():
    store, layers, metadata = bias_store()
    assert store.bias_supported() is True and store.bias_bytes() == 2 * 8 * 2 * 4 * 4
    tensors = {
        name: torch.full((3, 2, 6), float(i + 1), dtype=torch.float32)
        for i, name in enumerate(layers)
    }
    beta = beta_rows(layers)
    receipt = store.register(
        "fit",
        tensors,
        source_cursor=10,
        source_position_offset=0,
        retained_tokens=3,
        method=method_record(),
        layer_token_indices={"a": [0, 1, 9], "b": [2, 3, 4]},
        beta=beta,
    )
    assert receipt["has_bias"] is True and receipt["bias_bytes"] == 2 * 3 * 2 * 4
    assert "beta" not in receipt and "tensors" not in receipt
    assert store.list()["total_cpu_bias_bytes"] == 48
    # ``digest`` covers the K/V rows only; the bias has its own digest.
    plain = store.register(
        "same_rows_no_bias",
        {name: t.clone() for name, t in tensors.items()},
        source_cursor=10,
        source_position_offset=0,
        retained_tokens=3,
        method=method_record(),
    )
    assert plain["digest"] == receipt["digest"]
    assert plain["bias_digest"] is None and isinstance(receipt["bias_digest"], str)
    # The entry holds its own copy of beta.
    beta["a"].fill_(99.0)
    assert torch.all(store.snapshot_bias("fit")["a"][0] == 0)
    for layer in layers.values():
        layer.bias_cache.fill_(7.0)
    result = store.restore("fit", "target", 1, metadata, 3)  # row 1 -> block 5
    assert result["has_bias"] is True and result["bias_bytes"] == 48
    for name, layer in layers.items():
        assert torch.equal(
            layer.bias_cache[5, :, :3].T, store.snapshot_bias("fit")[name]
        )
        assert torch.all(layer.bias_cache[5, :, 3:] == 7.0)
        assert torch.all(layer.bias_cache[:5] == 7.0) and torch.all(
            layer.bias_cache[6:] == 7.0
        )
        assert torch.all(
            layer.kv_cache[5, :, :3, :] == float(list(layers).index(name) + 1)
        )
    store.audit("fit")


def test_capture_keeps_imported_bias_so_chained_imports_carry_it():
    store, layers, metadata = bias_store()
    tensors = {name: torch.full((3, 2, 6), 4.0) for name in layers}
    beta = beta_rows(layers)
    receipt = store.register(
        "fit",
        tensors,
        source_cursor=10,
        source_position_offset=0,
        retained_tokens=3,
        method=method_record(),
        beta=beta,
    )
    store.restore("fit", "target", 1, metadata, 3)
    chained = store.capture(
        {"name": "chain", "token_indices": [0, 1, 2]}, "target", 1, metadata, 3
    )
    assert chained["has_bias"] is True and chained["bias_bytes"] == 48
    assert chained["digest"] == receipt["digest"]  # K/V rows round-trip
    assert chained["bias_digest"] == receipt["bias_digest"]  # and so does beta
    for name in layers:
        assert torch.equal(store.snapshot_bias("chain")[name], beta[name])
    # Rows whose bias reads zero are captured without a bias record.
    plain = store.capture(
        {"name": "plain", "token_indices": [0, 1]}, "s", 0, metadata, 4
    )
    assert plain["has_bias"] is False and plain["bias_bytes"] == 0
    assert store.snapshot_bias("plain") is None
    # A snapshot without bias imports as bias 0 where the buffer held values.
    for layer in layers.values():
        layer.bias_cache.fill_(5.0)
    result = store.restore("plain", "t2", 1, metadata, 2)
    assert result["has_bias"] is False
    for layer in layers.values():
        assert torch.all(layer.bias_cache[5, :, :2] == 0)
        assert torch.all(layer.bias_cache[5, :, 2:] == 5.0)
    # Subsets of a bias snapshot slice the bias rows with the KV rows.
    subset = store.subset("sub", "chain", method=method_record(), token_indices=[0, 2])
    assert subset["has_bias"] is True and subset["bias_bytes"] == 2 * 2 * 2 * 4
    for name in layers:
        assert torch.equal(store.snapshot_bias("sub")[name], beta[name][[0, 2]])


def test_staged_bias_is_audited_with_the_rows():
    store, layers, metadata = bias_store()
    tensors = {name: torch.full((3, 2, 6), 4.0) for name in layers}
    store.register(
        "fit",
        tensors,
        source_cursor=10,
        source_position_offset=0,
        retained_tokens=3,
        method=method_record(),
        beta=beta_rows(layers),
    )
    store.stage("fit", "cpu")
    assert set(store._staged["fit"]) == {"a", "b", "a#beta", "b#beta"}
    audited = store.audit("fit")
    assert audited["staged_digest"] == audited["digest"]
    assert audited["staged_bias_digest"] == audited["bias_digest"]
    store._staged["fit"]["a#beta"].fill_(0.0)
    with pytest.raises(ValueError, match="modified"):
        store.audit("fit")
    store.stage("fit", "cpu")  # already staged: the mutated copy stays
    store._staged["fit"]["a#beta"].copy_(store.snapshot_bias("fit")["a"])
    store.audit("fit")
    store._entries["fit"]["beta"]["b"].fill_(0.0)
    with pytest.raises(ValueError, match="modified"):
        store.audit("fit")


def test_beta_snapshot_is_refused_when_the_backend_has_no_bias_buffer():
    store, layers, metadata = fixture_store()  # FLASH_ATTN layers, no buffer
    assert store.bias_supported() is False and store.bias_bytes() == 0
    tensors = {name: torch.full((3, 2, 6), 4.0) for name in layers}
    receipt = store.register(
        "fit",
        tensors,
        source_cursor=10,
        source_position_offset=0,
        retained_tokens=3,
        method=method_record(),
        beta=beta_rows(layers),
    )
    assert receipt["has_bias"] is True  # registering is allowed, importing is not
    before = {name: layer.kv_cache.clone() for name, layer in layers.items()}
    with pytest.raises(ValueError, match="cannot apply per-key bias"):
        store.validate_restore("fit", "target", 1, metadata, 3)
    with pytest.raises(ValueError, match="cannot apply per-key bias"):
        store.restore("fit", "target", 1, metadata, 3)
    assert all(torch.equal(layers[n].kv_cache, before[n]) for n in layers)
    # Only one layer with a buffer is still unsupported (all-or-nothing).
    layers["a"].bias_cache = torch.zeros(8, 2, 4)
    assert store.bias_supported() is False
    with pytest.raises(ValueError, match=r"cannot apply per-key bias.*: b$"):
        store.restore("fit", "target", 1, metadata, 3)
    assert all(torch.equal(layers[n].kv_cache, before[n]) for n in layers)
    assert not layers["a"].bias_cache.any()


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda b: {**b, "a": b["a"].T.contiguous()}, "beta shape"),
        (lambda b: {**b, "a": b["a"][:2]}, "beta shape"),
        (lambda b: {**b, "a": b["a"].to(torch.float64)}, "beta dtype"),
        (lambda b: {**b, "a": b["a"].clone().fill_(float("inf"))}, "Non-finite beta"),
        (lambda b: {"a": b["a"]}, "exactly the FA layers"),
        (lambda b: {**b, "c": b["a"]}, "exactly the FA layers"),
        (lambda b: {**b, "a": b["a"].tolist()}, "beta shape"),
        (lambda b: [b["a"], b["b"]], "exactly the FA layers"),
    ],
)
def test_register_rejects_malformed_beta(mutate, message):
    store, layers, metadata = bias_store()
    tensors = {name: torch.full((3, 2, 6), 4.0) for name in layers}
    with pytest.raises(ValueError, match=message):
        store.register(
            "bad",
            tensors,
            source_cursor=10,
            source_position_offset=0,
            retained_tokens=3,
            method=method_record(),
            beta=mutate(beta_rows(layers)),
        )
    assert store.list()["snapshots"] == {}


def test_bias_buffer_layout_mismatch_is_rejected_before_any_write():
    store, layers, metadata = bias_store()
    layers["b"].bias_cache = torch.zeros(8, 2, 4, dtype=torch.bfloat16)
    tensors = {name: torch.full((3, 2, 6), 4.0) for name in layers}
    receipt = store.register(
        "fit",
        tensors,
        source_cursor=10,
        source_position_offset=0,
        retained_tokens=3,
        method=method_record(),
        beta=beta_rows(layers),
    )
    assert receipt["has_bias"] is True
    # A malformed buffer is a fault, reported loudly rather than as "no bias".
    with pytest.raises(ValueError, match="bias_cache must be float32.*: b"):
        store.bias_supported()
    before_kv = {name: layer.kv_cache.clone() for name, layer in layers.items()}
    before_bias = {name: layer.bias_cache.clone() for name, layer in layers.items()}
    with pytest.raises(ValueError, match="bias_cache must be float32.*: b"):
        store.validate_restore("fit", "target", 1, metadata, 3)
    with pytest.raises(ValueError, match="bias_cache must be float32.*: b"):
        store.restore("fit", "target", 1, metadata, 3)
    for name, layer in layers.items():
        assert torch.equal(layer.kv_cache, before_kv[name])
        assert torch.equal(layer.bias_cache, before_bias[name])


def test_block_reused_after_an_import_reads_zero_bias():
    """The normal KV write zeroes the bias of exactly the slots it writes, so a
    page that later serves another request never exposes the imported bias."""
    from vllm.v1.worker.kv_bias import zero_kv_bias_slots

    store, layers, metadata = bias_store()
    tensors = {name: torch.full((3, 2, 6), 4.0) for name in layers}
    store.register(
        "fit",
        tensors,
        source_cursor=10,
        source_position_offset=0,
        retained_tokens=3,
        method=method_record(),
        beta=beta_rows(layers),
    )
    store.restore("fit", "target", 1, metadata, 3)  # block 5, slots 0..2
    for layer in layers.values():
        assert layer.bias_cache[5, :, 1:3].any()
        # The next request's prefill writes block 5 slots 0..1 (and a padded
        # token): only those slots are reset, slot 2 keeps its value until it
        # is written too.
        zero_kv_bias_slots(layer.bias_cache, torch.tensor([20, 21, -1]))
        assert not layer.bias_cache[5, :, :2].any()
        assert layer.bias_cache[5, :, 2].any()
        zero_kv_bias_slots(layer.bias_cache, torch.tensor([22, 23]))
        assert not layer.bias_cache[5].any()
        assert layer.kv_cache[5, :, :3, :].eq(4.0).all()  # rows are untouched


# ---------------------------------------------------------------- bias mask


def test_bias_mask_sets_beta_on_listed_rows_and_keeps_kv_rows_and_digest():
    store, layers, metadata = bias_store()
    full = store.capture(
        {"name": "full", "token_indices": list(range(6))}, "s", 0, metadata, 6
    )
    assert full["has_bias"] is False
    masked = store.bias_mask("mask", "full", token_indices=[1, 4], value=-20.0)
    assert masked["selection_policy"] == "bias_mask" and masked["synthetic"] is True
    assert masked["has_bias"] is True and masked["bias_bytes"] == 2 * 6 * 2 * 4
    assert masked["digest"] == full["digest"]  # K/V rows untouched
    assert masked["bias_digest"] != full["bias_digest"] and masked["bias_digest"]
    assert masked["digest_verified_at"] == "bias_mask"
    assert masked["token_indices"] == list(range(6))
    assert masked["layer_token_indices"] == full["layer_token_indices"]
    assert masked["retained_tokens"] == 6 and masked["bytes"] == full["bytes"]
    assert (masked["source_cursor"], masked["request_id"]) == (6, "s")
    assert masked["derived_from"] == {
        "name": "full",
        "digest": full["digest"],
        "bias_digest": None,
        "retained_tokens": 6,
        "selection_policy": "shared",
        "synthetic": False,
    }
    assert masked["method"]["name"] == "bias_mask"
    assert masked["method"]["params"] == {
        "token_indices": [1, 4],
        "value": -20.0,
        "heads": [0, 1],
        "all_heads": True,
        "rows_per_layer": {"a": [1, 4], "b": [1, 4]},
    }
    expected = torch.zeros(6, 2)
    expected[[1, 4]] = -20.0
    for name in layers:
        assert torch.equal(
            store.snapshot_rows("mask")[name], store.snapshot_rows("full")[name]
        )
        assert torch.equal(store.snapshot_bias("mask")[name], expected)
    # A bias source keeps its other rows; a head subset sets only those heads.
    head0 = store.bias_mask("h0", "mask", token_indices=[0, 4], value=3.0, heads=[0])
    assert head0["method"]["params"]["heads"] == [0]
    assert head0["method"]["params"]["all_heads"] is False
    assert head0["derived_from"]["bias_digest"] == masked["bias_digest"]
    expected_h0 = expected.clone()
    expected_h0[[0, 4], 0] = 3.0
    for name in layers:
        assert torch.equal(store.snapshot_bias("h0")[name], expected_h0)
        assert torch.equal(store.snapshot_bias("mask")[name], expected)  # unchanged
    # Import writes the masked bias at the rows' slots; K/V equal the source.
    before = {n: layer.kv_cache.clone() for n, layer in layers.items()}
    for layer in layers.values():
        layer.bias_cache.fill_(9.0)
    store.restore("h0", "target", 1, metadata, 6)  # row 1 -> blocks 5, 6
    for name, layer in layers.items():
        got = torch.cat([layer.bias_cache[5].T, layer.bias_cache[6, :, :2].T])
        assert torch.equal(got, expected_h0)
        assert torch.all(layer.bias_cache[6, :, 2:] == 9.0)
        source_rows = store.snapshot_rows("full")[name]
        written = torch.cat(
            [
                layer.kv_cache[5].transpose(0, 1),
                layer.kv_cache[6, :, :2].transpose(0, 1),
            ]
        )
        assert torch.equal(written, source_rows)
    del before


def test_bias_mask_works_on_per_layer_sources_and_survives_subsets():
    store, layers, metadata = bias_store()
    store.capture(
        {"name": "pl", "layer_token_indices": {"a": [0, 1, 5], "b": [1, 3, 4]}},
        "s",
        0,
        metadata,
        6,
    )
    masked = store.bias_mask("m", "pl", token_indices=[1], value=-20.0)
    assert masked["token_indices"] is None  # inherited from the per-layer source
    assert masked["method"]["params"]["rows_per_layer"] == {"a": [1], "b": [0]}
    for name, row in (("a", 1), ("b", 0)):
        beta = store.snapshot_bias("m")[name]
        assert torch.all(beta[row] == -20.0) and int(beta.ne(0).sum()) == 2
    # A token recorded for one layer only is refused before anything is created.
    with pytest.raises(ValueError, match=r"not in the source snapshot \(b\)"):
        store.bias_mask("m2", "pl", token_indices=[5], value=-20.0)
    assert "m2" not in store.list()["snapshots"]
    sub = store.subset(
        "s2", "m", method=method_record(), layer_token_indices={"a": [1], "b": [3]}
    )
    assert sub["has_bias"] is True
    assert torch.equal(store.snapshot_bias("s2")["a"], torch.full((1, 2), -20.0))
    assert not store.snapshot_bias("s2")["b"].any()


@pytest.mark.parametrize(
    "kwargs,message",
    [
        (dict(token_indices=[7]), "indices"),
        (dict(token_indices=[1, 0]), "indices"),
        (dict(token_indices=[]), "indices"),
        (dict(token_indices=[9], value=1.0), "indices"),
        (dict(value=float("nan")), "finite"),
        (dict(value=float("inf")), "finite"),
        (dict(value=True), "finite"),
        (dict(value="-20"), "finite"),
        (dict(heads=[2]), "heads must be"),
        (dict(heads=[1, 0]), "heads must be"),
        (dict(heads=[0, 0]), "heads must be"),
        (dict(heads=[]), "heads must be"),
        (dict(heads=(0,)), "heads must be"),
        (dict(name="full"), "overwritten"),
        (dict(name=""), "nonempty"),
        (dict(source="nope"), "Unknown KV snapshot"),
    ],
)
def test_bias_mask_rejects_bad_tokens_values_heads_and_names(kwargs, message):
    store, layers, metadata = bias_store()
    store.capture(
        {"name": "full", "token_indices": list(range(6))}, "s", 0, metadata, 6
    )
    call = dict(name="m", source="full", token_indices=[1], value=-20.0, heads=None)
    call.update(kwargs)
    name, source = call.pop("name"), call.pop("source")
    with pytest.raises(ValueError, match=message):
        store.bias_mask(name, source, **call)
    assert set(store.list()["snapshots"]) == {"full"}


def test_bias_mask_is_refused_without_a_bias_buffer_or_recorded_tokens():
    store, layers, metadata = fixture_store()  # FLASH_ATTN, no buffer
    store.capture(
        {"name": "full", "token_indices": list(range(6))}, "s", 0, metadata, 6
    )
    with pytest.raises(ValueError, match="cannot apply per-key bias"):
        store.bias_mask("m", "full", token_indices=[1], value=-20.0)
    store2, layers2, _ = bias_store()
    store2.register(
        "noidx",
        {name: torch.zeros((2, 2, 6)) for name in layers2},
        source_cursor=4,
        source_position_offset=0,
        retained_tokens=2,
        method=method_record(),
    )
    with pytest.raises(ValueError, match="records no token indices"):
        store2.bias_mask("m", "noidx", token_indices=[0], value=-20.0)
