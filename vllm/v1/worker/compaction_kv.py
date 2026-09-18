# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact selected-KV export/import for controlled compaction experiments.

The scheduler still owns every cache page. Import replaces an already computed
scratch prefix, at its exact boundary, before the next token is consumed. It
does not reconstruct KV from a recurrent state or merely mask a full cache.

Snapshot entries come from two sources and import through one ``restore`` path:

* ``capture``: rows copied from the paged cache of a live request, selected by
  one shared token list (``{"name", "token_indices"}``) or by one list per
  full-attention layer (``{"name", "layer_token_indices"}``, equal counts).
* ``register``: rows computed outside the cache (for example attention-matching
  values), validated against each layer's cache layout exactly as a capture of
  ``retained_tokens`` rows would be. Such entries carry ``synthetic: True`` and a
  ``method`` record (name, params, input digests).

Row layout per layer is the cache's ``[tokens, num_kv_heads, 2 * head_size]``
with keys in ``[..., :head_size]`` and values in ``[..., head_size:]``.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import torch

SUPPORTED_BACKENDS = {"FLASH_ATTN", "TRITON_ATTN"}
SUPPORTED_CACHE_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


def _digest(tensors: dict[str, torch.Tensor]) -> str:
    result = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        result.update(
            json.dumps([name, list(tensor.shape), str(tensor.dtype)]).encode()
        )
        result.update(tensor.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def _validate_index_list(indices: Any, computed_tokens: int | None) -> list[int]:
    """Unique, chronological, nonnegative ints below the consumed cursor."""
    limit = computed_tokens if computed_tokens is not None else float("inf")
    if (
        not isinstance(indices, list)
        or not indices
        or any(type(i) is not int or not 0 <= i < limit for i in indices)
        or indices != sorted(set(indices))
    ):
        raise ValueError("KV indices must be unique, chronological, and consumed")
    return list(indices)


def split_kv(rows: torch.Tensor, head_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split ``[tokens, num_kv_heads, 2 * head_size]`` rows into (keys, values)."""
    if rows.ndim != 3 or rows.shape[-1] != 2 * head_size:
        raise ValueError("KV rows must be [tokens, kv_heads, 2 * head_size]")
    return rows[..., :head_size], rows[..., head_size:]


def validate_method(method: Any) -> dict:
    """A self-describing method record: ``name``, ``params``, ``inputs``; JSON-safe."""
    if (
        not isinstance(method, dict)
        or not isinstance(method.get("name"), str)
        or not method["name"]
        or not isinstance(method.get("params"), dict)
        or not isinstance(method.get("inputs"), dict)
    ):
        raise ValueError("method must carry name, params and inputs")
    try:
        json.dumps(method)
    except (TypeError, ValueError) as error:
        raise ValueError("method must be JSON-serializable") from error
    return copy.deepcopy(method)


class NativeKVSnapshotStore:
    """Immutable CPU snapshots, with optional reusable GPU staging copies."""

    def __init__(self, runner: Any):
        from vllm.v1.kv_cache_interface import MambaSpec

        context = runner.compilation_config.static_forward_context
        self.layers = {
            name: context[name]
            for group in runner.kv_cache_config.kv_cache_groups
            if not isinstance(group.kv_cache_spec, MambaSpec)
            for name in group.layer_names
        }
        self._entries: dict[str, dict] = {}
        self._staged: dict[str, dict[str, torch.Tensor]] = {}

    # ------------------------------------------------------------------ specs
    def _layer_indices(
        self, spec: dict, computed_tokens: int
    ) -> tuple[str, dict[str, list[int]]]:
        """Resolve a capture spec into per-layer index lists and its policy.

        ``spec`` may also carry an optional ``method`` record (validated like
        ``register``'s) describing how the selection was produced.
        """
        if not isinstance(spec, dict) or "name" not in spec:
            raise ValueError("KV capture requires name and token_indices")
        if not isinstance(spec["name"], str) or not spec["name"]:
            raise ValueError("KV snapshot name must be nonempty")
        if "method" in spec:
            validate_method(spec["method"])
        keys = set(spec) - {"method"}
        if keys == {"name", "token_indices"}:
            indices = _validate_index_list(spec["token_indices"], computed_tokens)
            return "shared", {name: list(indices) for name in self.layers}
        if keys == {"name", "layer_token_indices"}:
            per_layer = spec["layer_token_indices"]
            if not isinstance(per_layer, dict) or set(per_layer) != set(self.layers):
                raise ValueError(
                    "layer_token_indices must name every FA layer exactly once"
                )
            validated = {
                name: _validate_index_list(per_layer[name], computed_tokens)
                for name in self.layers
            }
            if len({len(v) for v in validated.values()}) != 1:
                raise ValueError(
                    "layer_token_indices must have equal counts across layers"
                )
            return "per_layer", validated
        raise ValueError(
            "KV capture requires name and token_indices or layer_token_indices"
        )

    @staticmethod
    def _indices(spec: dict, computed_tokens: int) -> list[int]:
        """Legacy shared-selection validation (kept for callers and tests)."""
        if not isinstance(spec, dict) or set(spec) != {"name", "token_indices"}:
            raise ValueError("KV capture requires name and token_indices")
        if not isinstance(spec["name"], str) or not spec["name"]:
            raise ValueError("KV snapshot name must be nonempty")
        return _validate_index_list(spec["token_indices"], computed_tokens)

    def _check_layout(self, name: str, layer: Any) -> torch.Tensor:
        cache = layer.kv_cache
        if (
            layer.get_attn_backend().get_name() not in SUPPORTED_BACKENDS
            or not isinstance(cache, torch.Tensor)
            or cache.ndim != 4
            or cache.dtype not in SUPPORTED_CACHE_DTYPES
            or cache.shape[3] != 2 * layer.head_size
            or cache.shape[1] != layer.num_kv_heads
        ):
            raise ValueError(f"Unsupported exact-KV cache layout: {name}")
        return cache

    def _locations(
        self,
        row: int,
        metadata: dict,
        indices: list[int] | dict[str, list[int]],
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Validate every layer before a caller can mutate any layer.

        ``indices`` is one shared list or one list per FA layer.
        """
        if type(row) is not int or row < 0 or not self.layers:
            raise ValueError("KV import requires a real request row and FA layers")
        per_layer = (
            indices
            if isinstance(indices, dict)
            else {name: indices for name in self.layers}
        )
        if set(per_layer) != set(self.layers):
            raise ValueError("KV selection must cover exactly the FA layers")
        locations = {}
        for name, layer in self.layers.items():
            cache = self._check_layout(name, layer)
            selected = per_layer[name]
            if not selected:
                raise ValueError(f"Empty KV selection: {name}")
            table = getattr(metadata.get(name), "block_table", None)
            if not isinstance(table, torch.Tensor) or table.ndim != 2:
                raise ValueError(f"Missing kernel block table: {name}")
            block_size = cache.shape[2]
            logical = torch.tensor(selected, dtype=torch.long)
            block_columns = logical // block_size
            if row >= table.shape[0] or int(block_columns.max()) >= table.shape[1]:
                raise ValueError(f"KV selection exceeds allocated block table: {name}")
            physical = table[row].cpu()[block_columns].to(dtype=torch.long)
            offsets = logical % block_size
            # Block zero is the scheduler's reserved NULL page.
            if bool(((physical <= 0) | (physical >= cache.shape[0])).any()):
                raise ValueError(f"KV selection points to NULL/invalid pages: {name}")
            addresses = physical * block_size + offsets
            if addresses.unique().numel() != len(selected):
                raise ValueError(f"Aliased KV destination addresses: {name}")
            locations[name] = (
                cache,
                physical.to(cache.device),
                offsets.to(cache.device),
            )
        return locations

    # --------------------------------------------------------------- capture
    def validate_capture(
        self,
        spec: dict,
        request_id: str,
        row: int,
        metadata: dict,
        computed_tokens: int,
    ) -> None:
        _, per_layer = self._layer_indices(spec, computed_tokens)
        if spec["name"] in self._entries:
            raise ValueError("KV snapshot names cannot be overwritten")
        self._locations(row, metadata, per_layer)

    def capture(
        self,
        spec: dict,
        request_id: str,
        row: int,
        metadata: dict,
        computed_tokens: int,
        source_position_offset: int = 0,
    ) -> dict:
        if type(source_position_offset) is not int or source_position_offset < 0:
            raise ValueError("Invalid source position offset")
        self.validate_capture(spec, request_id, row, metadata, computed_tokens)
        policy, per_layer = self._layer_indices(spec, computed_tokens)
        locations = self._locations(row, metadata, per_layer)
        tensors = {
            name: cache[blocks, :, offsets, :].detach().cpu().clone()
            for name, (cache, blocks, offsets) in locations.items()
        }
        if not all(bool(torch.isfinite(t).all()) for t in tensors.values()):
            raise ValueError("Non-finite selected KV values")
        first = next(iter(per_layer.values()))
        self._entries[spec["name"]] = {
            "name": spec["name"],
            "request_id": request_id,
            "source_cursor": computed_tokens,
            "source_position_offset": source_position_offset,
            "source_absolute_next_position": computed_tokens + source_position_offset,
            "selection_policy": policy,
            "token_indices": list(first) if policy == "shared" else None,
            "layer_token_indices": {name: list(v) for name, v in per_layer.items()},
            "retained_tokens": len(first),
            "synthetic": False,
            "method": (validate_method(spec["method"]) if "method" in spec else None),
            "bytes": sum(t.numel() * t.element_size() for t in tensors.values()),
            "digest": _digest(tensors),
            "digest_verified_at": "capture",
            "tensors": tensors,
        }
        return self.describe(spec["name"])

    # -------------------------------------------------------------- register
    def register(
        self,
        name: str,
        tensors: dict[str, torch.Tensor],
        *,
        source_cursor: int,
        source_position_offset: int,
        retained_tokens: int,
        method: dict,
        layer_token_indices: dict[str, list[int]] | None = None,
        token_indices: list[int] | None = None,
        request_id: str | None = None,
    ) -> dict:
        """Store rows computed outside the cache as an importable snapshot.

        Every layer tensor must look exactly like a capture of ``retained_tokens``
        rows from that layer's cache: shape ``[retained_tokens, num_kv_heads,
        2 * head_size]``, the cache dtype, all values finite. ``method`` records
        how the rows were produced (``name``, ``params``, ``inputs`` digests) and
        must be JSON-serializable. Optional ``layer_token_indices`` or
        ``token_indices`` record which source tokens the rows stand for.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("KV snapshot name must be nonempty")
        if name in self._entries:
            raise ValueError("KV snapshot names cannot be overwritten")
        for value, label in (
            (source_cursor, "source_cursor"),
            (source_position_offset, "source_position_offset"),
            (retained_tokens, "retained_tokens"),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{label} must be a nonnegative integer")
        if retained_tokens < 1 or retained_tokens > source_cursor:
            raise ValueError("retained_tokens must lie in 1..source_cursor")
        if request_id is not None and (
            not isinstance(request_id, str) or not request_id
        ):
            raise ValueError("request_id must be a nonempty string or None")
        if not self.layers:
            raise ValueError("KV register requires FA layers")
        if not isinstance(tensors, dict) or set(tensors) != set(self.layers):
            raise ValueError("Synthetic KV must provide exactly the FA layers")
        method = validate_method(method)
        if layer_token_indices is not None and token_indices is not None:
            raise ValueError("Give layer_token_indices or token_indices, not both")
        policy = None
        per_layer = None
        if layer_token_indices is not None:
            policy, per_layer = self._layer_indices(
                {"name": name, "layer_token_indices": layer_token_indices},
                source_cursor,
            )
        elif token_indices is not None:
            policy, per_layer = self._layer_indices(
                {"name": name, "token_indices": token_indices}, source_cursor
            )
        if per_layer is not None and any(
            len(v) != retained_tokens for v in per_layer.values()
        ):
            raise ValueError("Recorded token indices must count retained_tokens")
        copied = {}
        for layer_name, layer in self.layers.items():
            cache = self._check_layout(layer_name, layer)
            tensor = tensors[layer_name]
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != (
                retained_tokens,
                cache.shape[1],
                cache.shape[3],
            ):
                raise ValueError(
                    f"Synthetic KV shape differs from the cache layout: {layer_name}"
                )
            if tensor.dtype != cache.dtype:
                raise ValueError(
                    f"Synthetic KV dtype differs from the cache: {layer_name}"
                )
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"Non-finite synthetic KV values: {layer_name}")
            copied[layer_name] = tensor.detach().to("cpu", copy=True).contiguous()
        self._entries[name] = {
            "name": name,
            "request_id": request_id,
            "source_cursor": source_cursor,
            "source_position_offset": source_position_offset,
            "source_absolute_next_position": source_cursor + source_position_offset,
            "selection_policy": policy,
            "token_indices": (
                list(next(iter(per_layer.values())))
                if policy == "shared" and per_layer
                else None
            ),
            "layer_token_indices": (
                {k: list(v) for k, v in per_layer.items()} if per_layer else None
            ),
            "retained_tokens": retained_tokens,
            "synthetic": True,
            "method": copy.deepcopy(method),
            "bytes": sum(t.numel() * t.element_size() for t in copied.values()),
            "digest": _digest(copied),
            "digest_verified_at": "register",
            "tensors": copied,
        }
        return self.describe(name)

    # -------------------------------------------------------------- reading
    def read_rows(
        self,
        row: int,
        metadata: dict,
        indices: list[int] | dict[str, list[int]],
        *,
        computed_tokens: int | None = None,
        layers: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Gather selected cache rows per layer on the cache device (no host copy).

        Returns ``{layer: Tensor[len(indices), num_kv_heads, 2 * head_size]}``
        for ``layers`` (default: every FA layer); every layer is validated even
        when only a subset is gathered. ``computed_tokens`` bounds the indices to
        the consumed prefix; without it only structural validation applies.
        """
        per_layer = (
            {
                name: _validate_index_list(indices[name], computed_tokens)
                for name in indices
            }
            if isinstance(indices, dict)
            else _validate_index_list(indices, computed_tokens)
        )
        locations = self._locations(row, metadata, per_layer)
        selected = list(self.layers) if layers is None else list(layers)
        if set(selected) - set(self.layers) or len(set(selected)) != len(selected):
            raise ValueError("read_rows layers must be distinct FA layers")
        return {
            name: locations[name][0][locations[name][1], :, locations[name][2], :]
            for name in selected
        }

    def snapshot_rows(self, name: str) -> dict[str, torch.Tensor]:
        """The immutable CPU rows of a snapshot (callers must not mutate them)."""
        self.describe(name)
        return dict(self._entries[name]["tensors"])

    # ---------------------------------------------------------------- subset
    def subset(
        self,
        name: str,
        source: str,
        *,
        method: dict,
        token_indices: list[int] | None = None,
        layer_token_indices: dict[str, list[int]] | None = None,
    ) -> dict:
        """Materialise a selection by slicing an existing snapshot on the CPU.

        The requested tokens (one shared list or one list per FA layer, equal
        counts) must be a subset of the source snapshot's recorded token indices
        for each layer; rows are gathered from the source's immutable CPU tensors,
        so no cache pass is needed. ``source_cursor``, ``source_position_offset``
        and ``request_id`` are inherited, ``derived_from`` names the source and
        its digest, and ``method`` records how the selection was produced.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("KV snapshot name must be nonempty")
        if name in self._entries:
            raise ValueError("KV snapshot names cannot be overwritten")
        origin = self._entries.get(source)
        if origin is None:
            raise ValueError(f"Unknown KV snapshot: {source}")
        if origin["layer_token_indices"] is None:
            raise ValueError("Source snapshot records no token indices to subset")
        method = validate_method(method)
        if (token_indices is None) == (layer_token_indices is None):
            raise ValueError("Give token_indices or layer_token_indices, not both")
        spec = {"name": name}
        if token_indices is not None:
            spec["token_indices"] = token_indices
        else:
            spec["layer_token_indices"] = layer_token_indices
        policy, per_layer = self._layer_indices(spec, origin["source_cursor"])
        tensors = {}
        for layer_name, requested in per_layer.items():
            available = origin["layer_token_indices"][layer_name]
            position_of = {token: row for row, token in enumerate(available)}
            missing = [t for t in requested if t not in position_of]
            if missing:
                raise ValueError(
                    f"Subset tokens are not in the source snapshot ({layer_name}): "
                    f"{missing[:8]}"
                )
            rows = torch.tensor([position_of[t] for t in requested], dtype=torch.long)
            tensors[layer_name] = (
                origin["tensors"][layer_name][rows].contiguous().clone()
            )
        first = next(iter(per_layer.values()))
        self._entries[name] = {
            "name": name,
            "request_id": origin["request_id"],
            "source_cursor": origin["source_cursor"],
            "source_position_offset": origin["source_position_offset"],
            "source_absolute_next_position": origin["source_absolute_next_position"],
            "selection_policy": policy,
            "token_indices": list(first) if policy == "shared" else None,
            "layer_token_indices": {k: list(v) for k, v in per_layer.items()},
            "retained_tokens": len(first),
            "synthetic": origin["synthetic"],
            "method": method,
            "derived_from": {
                "name": source,
                "digest": origin["digest"],
                "retained_tokens": origin["retained_tokens"],
                "selection_policy": origin["selection_policy"],
                "synthetic": origin["synthetic"],
            },
            "bytes": sum(t.numel() * t.element_size() for t in tensors.values()),
            "digest": _digest(tensors),
            "digest_verified_at": "subset",
            "tensors": tensors,
        }
        return self.describe(name)

    # ------------------------------------------------------------- metadata
    def describe(self, name: str) -> dict:
        if name not in self._entries:
            raise ValueError(f"Unknown KV snapshot: {name}")
        result = {}
        for key, value in self._entries[name].items():
            if key == "tensors":
                continue
            if key == "token_indices":
                result[key] = list(value) if value is not None else None
            elif key == "layer_token_indices":
                result[key] = (
                    {k: list(v) for k, v in value.items()}
                    if value is not None
                    else None
                )
            else:
                result[key] = copy.deepcopy(value)
        result["staged_gpu_bytes"] = sum(
            t.numel() * t.element_size()
            for t in self._staged.get(name, {}).values()
            if t.is_cuda
        )
        return result

    def list(self) -> dict:
        snapshots = {name: self.describe(name) for name in self._entries}
        return {
            "snapshots": snapshots,
            "total_cpu_snapshot_bytes": sum(s["bytes"] for s in snapshots.values()),
            "total_gpu_staging_bytes": sum(
                s["staged_gpu_bytes"] for s in snapshots.values()
            ),
        }

    def stage(self, name: str, device: Any = None) -> dict:
        self.describe(name)
        if device is not None and any(
            torch.device(device) != layer.kv_cache.device
            for layer in self.layers.values()
        ):
            raise ValueError("KV staging device must match the cache device")
        if name not in self._staged:
            self._staged[name] = {
                key: tensor.to(self.layers[key].kv_cache.device, copy=True)
                for key, tensor in self._entries[name]["tensors"].items()
            }
        return self.describe(name)

    def audit(self, name: str) -> dict:
        entry = self._entries[name]
        cpu_digest = _digest(entry["tensors"])
        gpu_digest = _digest(self._staged[name]) if name in self._staged else None
        if cpu_digest != entry["digest"] or (
            gpu_digest is not None and gpu_digest != entry["digest"]
        ):
            raise ValueError("An immutable KV snapshot was modified")
        return {**self.describe(name), "audited": True, "staged_digest": gpu_digest}

    # --------------------------------------------------------------- restore
    def validate_restore(
        self,
        name: str,
        request_id: str,
        row: int,
        metadata: dict,
        computed_tokens: int,
    ) -> None:
        entry = self.describe(name)
        # Offset sources (captured from an imported-then-continued request) are
        # importable: their rows keep original RoPE positions and the controller
        # requires position_offset == source_absolute_next_position - computed.
        if computed_tokens != entry["retained_tokens"]:
            raise ValueError(
                "KV import must follow exactly the retained scratch prefix"
            )
        locations = self._locations(row, metadata, list(range(computed_tokens)))
        tensors = self._entries[name]["tensors"]
        if set(tensors) != set(locations):
            raise ValueError("KV snapshot layer set differs")
        for key, (cache, _, _) in locations.items():
            if (
                tensors[key].shape != (computed_tokens, cache.shape[1], cache.shape[3])
                or tensors[key].dtype != cache.dtype
            ):
                raise ValueError(f"KV import shape/dtype differs: {key}")

    def restore(
        self,
        name: str,
        request_id: str,
        row: int,
        metadata: dict,
        computed_tokens: int,
    ) -> dict:
        self.validate_restore(name, request_id, row, metadata, computed_tokens)
        self.stage(name)
        locations = self._locations(row, metadata, list(range(computed_tokens)))
        page_bytes = 0
        for key, (cache, blocks, offsets) in locations.items():
            cache[blocks, :, offsets, :] = self._staged[name][key]
            page_bytes += (
                blocks.unique().numel() * cache[0].numel() * cache.element_size()
            )
        entry = self._entries[name]
        return {
            "name": name,
            "request_id": request_id,
            "cursor": computed_tokens,
            "source_cursor": entry["source_cursor"],
            "retained_tokens": computed_tokens,
            "payload_bytes": entry["bytes"],
            "occupied_kernel_page_bytes": page_bytes,
            "source_digest": entry["digest"],
            "selection_policy": entry["selection_policy"],
            "synthetic": entry["synthetic"],
            "copy_kind": "exact_indexed_assignment",
            "scratch_prefill_tokens": computed_tokens,
        }

    def drop(self, name: str) -> dict:
        result = self.describe(name)
        del self._entries[name]
        self._staged.pop(name, None)
        return result
