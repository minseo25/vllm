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


def test_offset_source_is_auditable_but_cannot_be_silently_rebased_on_import():
    store, layers, metadata = fixture_store()
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
    before = layers["a"].kv_cache.clone()
    with pytest.raises(ValueError, match="offset source"):
        store.restore("offset", "d", 1, metadata, 2)
    assert torch.equal(layers["a"].kv_cache, before)
