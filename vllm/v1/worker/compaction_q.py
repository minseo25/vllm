# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Query export and in-worker KV scoring/selection/fitting for compaction.

Query export
------------
A request armed with ``export_q = {"name": str, "token_range": [start, end]}``
has the post-RoPE queries of its prefill rows inside ``[start, end)`` copied,
for every full-attention (FA) layer, into a host store. The controller builds
one :class:`QueryExportPlan` per forward that contains such rows and installs
``plan.hook`` on the FA ``Attention`` modules (attribute ``compaction_q_export``,
class default ``None``). The attention custom op calls the hook before the
kernel and the controller removes it after the forward. Nothing is installed
for dummy runs, CUDA-graph capture, or decode steps, the controller refuses
FULL-graph forwards for export rows, and the hook never mutates a model tensor.

Compute ops
-----------
:func:`score_layer`, :func:`select_layers` and :func:`fit_am_layer` call the
methods library ``profiling.compaction_methods`` (``scores``, ``select``, ``am``),
resolved lazily at call time. Tests inject small fakes with
:func:`register_methods`. Results are validated for shape, dtype and finiteness
before they enter :class:`ScoreStore`, :class:`SelectionStore` or a synthetic KV
snapshot.

Failure semantics
-----------------
A hook or contract violation detected *inside* a forward (the hook raising, a
missing layer at ``after_forward``) propagates out of ``execute_model`` and is
fatal to the EngineCore: the operation is marked failed, the hook is removed,
but the engine process does not survive the exception. Pre-forward refusals
(FULL-graph mode, metadata mismatches) fail the operation before any write.
Compute ops on a resident request read the paged cache outside a forward; they
require the request to have run no further step since the cursor the caller
expects (``expected_cursor``), because the scheduler may reuse pages once the
request finishes. Residency after ``ingest`` is incidental to the harness's
resumable sessions, not a property this module can guarantee; prefer
``kv_snapshot`` sources where the extra host copy is affordable.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm.v1.worker.compaction import CompactionContractError
from vllm.v1.worker.compaction_kv import split_kv

__all__ = [
    "QExportStore",
    "QueryExportPlan",
    "ExportItem",
    "ScoreStore",
    "SelectionStore",
    "register_methods",
    "clear_methods",
    "resolve_methods",
    "score_layer",
    "select_layers",
    "fit_am_layer",
    "split_kv",
]

METHOD_MODULES = ("scores", "select", "am")
METHODS_PACKAGE = "profiling.compaction_methods"
EXPORT_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
SCORE_METHODS = ("h2o", "kvzip")
SELECT_POLICIES = ("shared", "per_layer")
SELECT_AGGREGATES = ("max", "mean")
# Reference queries per block; about 2 GB peak per kv head at Qwen3.5-9B geometry.
DEFAULT_CHUNK = 1024
# What the hook point delivers: the model applies q_norm and RoPE before
# ``Attention.forward``; the softmax scale is applied inside the kernel, so the
# exported rows are unscaled. ``QKVParallelLinear`` (TP=1) emits query heads in
# contiguous GQA blocks: query head i belongs to kv head i // (Hq // Hkv).
QUERY_CONVENTION = {
    "stage": "post_qk_norm_post_rope_unscaled",
    "head_order": "contiguous_gqa_blocks",
    "kv_head_of_query_head": "i // group_size",
    "scale_convention": "head_size ** -0.5 passed explicitly to the methods library",
}

_REGISTRY: dict[str, Any] = {}


# --------------------------------------------------------------------- methods
def register_methods(**modules: Any) -> None:
    """Inject module-like objects for ``scores``, ``select`` and/or ``am``.

    Intended for tests and for callers that already imported the library; a
    registered object takes precedence over the importable package.
    """
    unknown = set(modules) - set(METHOD_MODULES)
    if unknown:
        raise CompactionContractError(f"Unknown methods modules: {sorted(unknown)}")
    _REGISTRY.update(modules)


def clear_methods() -> None:
    _REGISTRY.clear()


def resolve_methods(kind: str) -> Any:
    """Return the ``kind`` module (registered fake or the real package)."""
    if kind not in METHOD_MODULES:
        raise CompactionContractError(f"Unknown methods module: {kind}")
    if kind in _REGISTRY:
        return _REGISTRY[kind]
    try:
        return importlib.import_module(f"{METHODS_PACKAGE}.{kind}")
    except ImportError as error:
        raise CompactionContractError(
            f"{METHODS_PACKAGE}.{kind} is unavailable ({error}); put the repository "
            "root on sys.path or register an implementation"
        ) from error


def _digest_tensors(tensors: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        digest.update(
            json.dumps([name, list(tensor.shape), str(tensor.dtype)]).encode()
        )
        digest.update(
            tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def _check_params(params: Any, allowed: set[str], label: str) -> dict:
    if params is None:
        return {}
    if not isinstance(params, dict):
        raise CompactionContractError(f"{label} params must be a dict")
    unknown = set(params) - allowed
    if unknown:
        raise CompactionContractError(f"Unknown {label} params: {sorted(unknown)}")
    try:
        json.dumps(params)
    except (TypeError, ValueError) as error:
        raise CompactionContractError(
            f"{label} params must be JSON-serializable"
        ) from error
    return dict(params)


def _call_library(kind: str, function: str, *args: Any, **kwargs: Any) -> Any:
    """Call ``compaction_methods.<kind>.<function>``; failures become contract errors.

    The library's own ``ValueError``s (shape, budget, protected) are re-raised as
    :class:`CompactionContractError` with the function named, so RPC callers see
    one error type.
    """
    library = resolve_methods(kind)
    target = getattr(library, function, None)
    if not callable(target):
        raise CompactionContractError(f"{METHODS_PACKAGE}.{kind}.{function} is missing")
    try:
        return target(*args, **kwargs)
    except CompactionContractError:
        raise
    except Exception as error:
        raise CompactionContractError(
            f"{kind}.{function} failed: {type(error).__name__}: {error}"
        ) from error


def _jsonable(value: Any) -> Any:
    """Convert library diagnostics (tensors, numpy scalars) into JSON-safe data."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return value.item()
    return repr(value)


def _check_token_range(value: Any, label: str) -> tuple[int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(type(x) is not int for x in value)
        or not 0 <= value[0] < value[1]
    ):
        raise CompactionContractError(f"{label} must be [start, end] with start < end")
    return int(value[0]), int(value[1])


def validate_protected(protected: Any, num_keys: int) -> list[int]:
    if protected is None:
        return []
    if (
        not isinstance(protected, list)
        or any(type(i) is not int or not 0 <= i < num_keys for i in protected)
        or protected != sorted(set(protected))
    ):
        raise CompactionContractError(
            "protected must be unique, sorted key positions inside the scored range"
        )
    return list(protected)


# ---------------------------------------------------------------- query export
class QExportStore:
    """Host copies of post-RoPE queries for flagged prefill rows, per FA layer.

    ``geometry`` fixes every layer's ``(num_heads, num_kv_heads, head_size,
    dtype, pinned)`` when the store is built, so ``open`` (called when the
    operation is armed) allocates the full host buffers up front and fails loudly
    before any forward runs; ``write`` then only checks that the rows match.
    Buffers are pinned when the model runs on CUDA. Sizes: Qwen3.5-9B has 8 FA
    layers of 16 x 256 bf16 = 8 KiB per token per layer, 64 KiB per token in
    total (2 GiB for a 32K export, 8 GiB for 128K); Nemotron-Nano-9B-v2 has 4 FA
    layers of 40 x 128 bf16 = 40 KiB per token (1.25 GiB at 32K). ``describe``
    reports ``bytes_per_token``, ``allocation_seconds`` and ``copy_seconds``.
    """

    def __init__(self, layer_names: Any, geometry: dict[str, dict]):
        names = list(layer_names)
        if not names or len(set(names)) != len(names):
            raise CompactionContractError("Query export needs distinct FA layers")
        if not isinstance(geometry, dict) or set(geometry) != set(names):
            raise CompactionContractError(
                "Query export geometry must cover exactly the FA layers"
            )
        self.layer_names = tuple(names)
        self.geometry = {
            name: self._check_geometry(name, geometry[name]) for name in names
        }
        if len({(g["dtype"], g["pinned"]) for g in self.geometry.values()}) != 1:
            raise CompactionContractError(
                "FA layers must share the query dtype and device for export"
            )
        self._entries: dict[str, dict] = {}

    @staticmethod
    def _check_geometry(name: str, spec: Any) -> dict:
        try:
            heads, kv_heads = spec["num_heads"], spec["num_kv_heads"]
            head_size, dtype, pinned = spec["head_size"], spec["dtype"], spec["pinned"]
        except (KeyError, TypeError) as error:
            raise CompactionContractError(
                f"Query export geometry for {name} needs num_heads, num_kv_heads, "
                "head_size, dtype and pinned"
            ) from error
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype.removeprefix("torch."), None)
        if (
            any(type(x) is not int or x < 1 for x in (heads, kv_heads, head_size))
            or heads % kv_heads
            or not isinstance(dtype, torch.dtype)
            or dtype not in EXPORT_DTYPES
            or not isinstance(pinned, bool)
        ):
            raise CompactionContractError(f"Invalid query export geometry: {name}")
        return {
            "num_heads": heads,
            "num_kv_heads": kv_heads,
            "head_size": head_size,
            "group_size": heads // kv_heads,
            "dtype": dtype,
            "pinned": pinned,
        }

    def bytes_per_token(self) -> int:
        return sum(
            g["num_heads"] * g["head_size"] * torch.finfo(g["dtype"]).bits // 8
            for g in self.geometry.values()
        )

    def open(
        self,
        name: str,
        *,
        token_range: Any,
        request_id: str | None = None,
        operation_id: str | None = None,
    ) -> dict:
        """Reserve ``name`` and allocate every layer's host buffer for the range."""
        if not isinstance(name, str) or not name:
            raise CompactionContractError("Query export name must be nonempty")
        if name in self._entries:
            raise CompactionContractError("Query export names cannot be reused")
        start, end = _check_token_range(token_range, "export_q token_range")
        rows = end - start
        started = time.perf_counter()
        tensors = {}
        try:
            for layer, g in self.geometry.items():
                tensors[layer] = torch.empty(
                    (rows, g["num_heads"], g["head_size"]),
                    dtype=g["dtype"],
                    device="cpu",
                    pin_memory=g["pinned"],
                )
        except Exception as error:
            raise CompactionContractError(
                f"Query export buffer allocation failed for {name}: {rows} rows, "
                f"{rows * self.bytes_per_token()} bytes: "
                f"{type(error).__name__}: {error}"
            ) from error
        first = next(iter(self.geometry.values()))
        self._entries[name] = {
            "name": name,
            "token_range": [start, end],
            "request_id": request_id,
            "operation_id": operation_id,
            "rows": rows,
            "layer_names": list(self.layer_names),
            "tensors": tensors,
            "shapes": {layer: list(t.shape) for layer, t in tensors.items()},
            "layouts": {
                layer: {
                    key: g[key]
                    for key in ("num_heads", "num_kv_heads", "head_size", "group_size")
                }
                for layer, g in self.geometry.items()
            },
            "query_convention": dict(QUERY_CONVENTION),
            "dtype": str(first["dtype"]),
            "pinned": first["pinned"],
            "positions": torch.full((rows,), -1, dtype=torch.long),
            "cursors": torch.arange(start, end, dtype=torch.long),
            "filled": torch.zeros(rows, dtype=torch.bool),
            "chunks": [],
            "bytes": sum(t.numel() * t.element_size() for t in tensors.values()),
            "allocation_seconds": time.perf_counter() - started,
            "copy_seconds": 0.0,
        }
        return self.describe(name)

    def bind(self, name: str, request_id: str) -> None:
        entry = self._entry(name)
        if entry["request_id"] is None:
            entry["request_id"] = request_id
        elif entry["request_id"] != request_id:
            raise CompactionContractError("Query export is bound to another request")

    def _entry(self, name: str) -> dict:
        if name not in self._entries:
            raise CompactionContractError(f"Unknown query export: {name}")
        return self._entries[name]

    def write(
        self,
        name: str,
        layer_name: str,
        rows: torch.Tensor,
        cursor_start: int,
        *,
        layout: dict | None = None,
    ) -> None:
        """Copy ``rows`` ([n, Hq, D], device) into the preallocated host buffer.

        The rows must match the layer's geometry (dtype, heads, head size and
        device class); ``layout`` (the hook's view of the layer: ``num_heads``,
        ``num_kv_heads``, ``head_size``) must agree with the geometry too, so the
        GQA grouping recorded in the receipt is the one the rows really have.
        """
        entry = self._entry(name)
        if layer_name not in self.layer_names:
            raise CompactionContractError(
                f"Unexpected query export layer: {layer_name}"
            )
        if not isinstance(rows, torch.Tensor) or rows.ndim != 3 or rows.shape[0] < 1:
            raise CompactionContractError("Exported queries must be [rows, heads, dim]")
        if rows.dtype not in EXPORT_DTYPES:
            raise CompactionContractError(
                f"Unsupported query dtype for export: {rows.dtype}"
            )
        geometry = self.geometry[layer_name]
        destination = entry["tensors"][layer_name]
        if rows.dtype != destination.dtype:
            raise CompactionContractError(
                f"Query dtype {rows.dtype} differs from the export geometry "
                f"{destination.dtype}: {layer_name}"
            )
        if tuple(rows.shape[1:]) != tuple(destination.shape[1:]):
            raise CompactionContractError(
                f"Query head layout {tuple(rows.shape[1:])} differs from the export "
                f"geometry {tuple(destination.shape[1:])}: {layer_name}"
            )
        if bool(rows.is_cuda) != geometry["pinned"]:
            raise CompactionContractError(
                f"Query device differs from the export geometry: {layer_name}"
            )
        if layout is not None and any(
            layout.get(key) != geometry[key]
            for key in ("num_heads", "num_kv_heads", "head_size")
        ):
            raise CompactionContractError(
                f"Query head layout is inconsistent with the geometry: {layer_name}"
            )
        start, _ = entry["token_range"]
        first = cursor_start - start
        last = first + rows.shape[0]
        if first < 0 or last > entry["rows"]:
            raise CompactionContractError("Exported rows fall outside token_range")
        if bool(entry["filled"][first:last].any()):
            raise CompactionContractError("Query rows were already exported")
        # Synchronous device-to-host copy into owned memory; model tensors untouched.
        destination[first:last].copy_(rows)

    def commit(
        self,
        name: str,
        *,
        cursor_start: int,
        cursor_end: int,
        positions: torch.Tensor,
        layers_written: set[str],
        receipt: dict,
    ) -> dict:
        entry = self._entry(name)
        if layers_written != set(self.layer_names):
            raise CompactionContractError(
                "Query export chunk missed FA layers: "
                f"{sorted(set(self.layer_names) - layers_written)}"
            )
        start, _ = entry["token_range"]
        first, last = cursor_start - start, cursor_end - start
        if positions.shape != (last - first,):
            raise CompactionContractError("Chunk positions do not match its rows")
        entry["positions"][first:last] = positions.to(torch.long)
        entry["filled"][first:last] = True
        entry["copy_seconds"] += float(receipt.get("hook_seconds", 0.0))
        entry["chunks"].append(
            {
                **copy.deepcopy(receipt),
                "cursor_start": cursor_start,
                "cursor_end": cursor_end,
                "rows": last - first,
            }
        )
        return self.describe(name)

    def describe(self, name: str) -> dict:
        entry = self._entry(name)
        filled = entry["filled"]
        complete = bool(filled.all())
        return {
            "name": name,
            "token_range": list(entry["token_range"]),
            "request_id": entry["request_id"],
            "operation_id": entry["operation_id"],
            "rows": entry["rows"],
            "rows_exported": int(filled.sum()),
            "complete": complete,
            "layer_names": list(entry["layer_names"]),
            "shapes": copy.deepcopy(entry["shapes"]),
            "layouts": copy.deepcopy(entry["layouts"]),
            "query_convention": dict(entry["query_convention"]),
            "dtype": entry["dtype"],
            "pinned": entry["pinned"],
            "bytes": entry["bytes"],
            "bytes_per_token": self.bytes_per_token(),
            "allocation_seconds": entry["allocation_seconds"],
            "copy_seconds": entry["copy_seconds"],
            "first_position": int(entry["positions"][0]) if complete else None,
            "last_position": int(entry["positions"][-1]) if complete else None,
            "chunks": copy.deepcopy(entry["chunks"]),
        }

    def list(self) -> dict:
        exports = {name: self.describe(name) for name in self._entries}
        return {
            "exports": exports,
            "total_host_bytes": sum(e["bytes"] for e in exports.values()),
        }

    def tensors(self, name: str) -> dict[str, torch.Tensor]:
        """Complete per-layer host tensors ``[rows, Hq, D]`` (do not mutate)."""
        entry = self._entry(name)
        if not bool(entry["filled"].all()):
            raise CompactionContractError(f"Query export is incomplete: {name}")
        return dict(entry["tensors"])

    def cursors(self, name: str) -> torch.Tensor:
        """Request token indices (cursor positions) of the exported rows."""
        return self._entry(name)["cursors"].clone()

    def positions(self, name: str) -> torch.Tensor:
        """Absolute RoPE positions of the exported rows (complete exports only)."""
        entry = self._entry(name)
        if not bool(entry["filled"].all()):
            raise CompactionContractError(f"Query export is incomplete: {name}")
        return entry["positions"].clone()

    def drop(self, name: str) -> dict:
        result = self.describe(name)
        del self._entries[name]
        return result


@dataclass
class ExportItem:
    """One request's exported row slice inside one forward."""

    name: str
    operation: Any
    row_start: int
    row_end: int
    cursor_start: int
    cursor_end: int
    positions: torch.Tensor
    receipt: dict = field(default_factory=dict)


class QueryExportPlan:
    """Per-forward hook state: which token rows go to which export entries.

    ``hook(layer, query, attn_metadata)`` is called by the attention custom op
    once per FA layer. It slices ``query`` by request rows (verified against the
    FA ``query_start_loc`` by the controller before the forward), copies them to
    the host store and never touches ``query``, ``output`` or the cache.
    """

    def __init__(
        self,
        store: QExportStore,
        layer_names: Any,
        items: list[ExportItem],
        total_tokens: int,
    ):
        self.store = store
        self.layer_names = set(layer_names)
        self.items = list(items)
        self.total_tokens = int(total_tokens)
        self.visited: list[str] = []
        self.failed: str | None = None
        self.hook_seconds = 0.0
        if not self.items:
            raise CompactionContractError("Query export plan without rows")
        for item in self.items:
            if not 0 <= item.row_start < item.row_end <= self.total_tokens:
                raise CompactionContractError("Export rows exceed the token batch")
            if item.row_end - item.row_start != item.cursor_end - item.cursor_start:
                raise CompactionContractError("Export rows and cursors disagree")
            if item.positions.shape != (item.row_end - item.row_start,):
                raise CompactionContractError("Export positions do not match rows")

    def hook(self, layer: Any, query: torch.Tensor, attn_metadata: Any) -> None:
        try:
            name = getattr(layer, "layer_name", None)
            if name not in self.layer_names:
                raise CompactionContractError(
                    f"Query export hook reached an unexpected layer: {name}"
                )
            if name in self.visited:
                raise CompactionContractError(f"Query export hook ran twice: {name}")
            if not isinstance(query, torch.Tensor) or query.ndim != 3:
                raise CompactionContractError("Query export expects [tokens, Hq, D]")
            if query.is_cuda and torch.cuda.is_current_stream_capturing():
                raise CompactionContractError(
                    "Query export must not run inside CUDA-graph capture"
                )
            if query.shape[0] < self.total_tokens:
                raise CompactionContractError(
                    "Query buffer is shorter than the real token batch"
                )
            actual = getattr(attn_metadata, "num_actual_tokens", None)
            if actual is not None and int(actual) != self.total_tokens:
                raise CompactionContractError(
                    "FA num_actual_tokens differs from the scheduled token batch"
                )
            heads = getattr(layer, "num_heads", None)
            head_size = getattr(layer, "head_size", None)
            if (heads is not None and query.shape[1] != heads) or (
                head_size is not None and query.shape[2] != head_size
            ):
                raise CompactionContractError(f"Query head layout differs: {name}")
            layout = None
            if heads is not None and head_size is not None:
                layout = {
                    "num_heads": int(heads),
                    "num_kv_heads": int(getattr(layer, "num_kv_heads", 0)),
                    "head_size": int(head_size),
                }
            started = time.perf_counter()
            for item in self.items:
                self.store.write(
                    item.name,
                    name,
                    query[item.row_start : item.row_end],
                    item.cursor_start,
                    layout=layout,
                )
            self.hook_seconds += time.perf_counter() - started
            self.visited.append(name)
        except Exception as error:
            self.failed = f"{type(error).__name__}: {error}"
            raise

    def finish(self) -> list[dict]:
        """Verify every FA layer ran the hook once, then commit chunk receipts."""
        if self.failed:
            raise CompactionContractError(self.failed)
        missing = self.layer_names - set(self.visited)
        if missing:
            raise CompactionContractError(
                f"Query export hook did not run for FA layers: {sorted(missing)}"
            )
        receipts = []
        for item in self.items:
            receipts.append(
                self.store.commit(
                    item.name,
                    cursor_start=item.cursor_start,
                    cursor_end=item.cursor_end,
                    positions=item.positions,
                    layers_written=set(self.visited),
                    receipt={**item.receipt, "hook_seconds": self.hook_seconds},
                )
            )
        return receipts


# --------------------------------------------------------------------- scores
class ScoreStore:
    """Per-``(layer, kv_head, token)`` scores as float32 CPU tensors."""

    def __init__(self):
        self._entries: dict[str, dict] = {}

    def put(
        self,
        name: str,
        *,
        layers: dict[str, torch.Tensor],
        key_token_indices: dict[str, list[int]],
        method: str,
        params: dict,
        q_export: str,
        source: dict,
        variant: str | None = None,
        library_params: dict | None = None,
        library_warnings: dict | None = None,
    ) -> dict:
        if not isinstance(name, str) or not name:
            raise CompactionContractError("Score names must be nonempty")
        if variant is not None and (not isinstance(variant, str) or not variant):
            raise CompactionContractError("Score variant must be a nonempty string")
        if name in self._entries:
            raise CompactionContractError("Score names cannot be overwritten")
        if not layers or set(layers) != set(key_token_indices):
            raise CompactionContractError("Scores and key indices must share layers")
        stored = {}
        for layer, tensor in layers.items():
            indices = key_token_indices[layer]
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.ndim != 2
                or tensor.shape[1] != len(indices)
                or not bool(torch.isfinite(tensor).all())
            ):
                raise CompactionContractError(
                    f"Scores must be finite [kv_heads, keys] tensors: {layer}"
                )
            stored[layer] = tensor.detach().to(device="cpu", dtype=torch.float32)
            stored[layer] = stored[layer].contiguous()
        self._entries[name] = {
            "name": name,
            "method": method,
            "variant": variant,
            "library_params": copy.deepcopy(library_params),
            "library_warnings": copy.deepcopy(library_warnings),
            "params": copy.deepcopy(params),
            "q_export": q_export,
            "source": copy.deepcopy(source),
            "layers": stored,
            "key_token_indices": {k: list(v) for k, v in key_token_indices.items()},
            "shapes": {k: list(v.shape) for k, v in stored.items()},
            "bytes": sum(t.numel() * t.element_size() for t in stored.values()),
            "digest": _digest_tensors(stored),
        }
        return self.describe(name)

    def _entry(self, name: str) -> dict:
        if name not in self._entries:
            raise CompactionContractError(f"Unknown scores: {name}")
        return self._entries[name]

    def describe(self, name: str) -> dict:
        entry = self._entry(name)
        return {
            key: copy.deepcopy(value)
            for key, value in entry.items()
            if key not in {"layers", "key_token_indices"}
        } | {
            "layer_names": list(entry["layers"]),
            "keys_per_layer": {
                k: len(v) for k, v in entry["key_token_indices"].items()
            },
        }

    def list(self) -> dict:
        scores = {name: self.describe(name) for name in self._entries}
        return {
            "scores": scores,
            "total_cpu_bytes": sum(s["bytes"] for s in scores.values()),
        }

    def tensors(self, name: str) -> dict[str, torch.Tensor]:
        return dict(self._entry(name)["layers"])

    def key_token_indices(self, name: str) -> dict[str, list[int]]:
        return {k: list(v) for k, v in self._entry(name)["key_token_indices"].items()}

    def drop(self, name: str) -> dict:
        result = self.describe(name)
        del self._entries[name]
        return result


class SelectionStore:
    """Named token selections produced by ``kv_select`` (RPC-returnable)."""

    def __init__(self):
        self._entries: dict[str, dict] = {}

    def put(self, name: str, selection: dict) -> dict:
        if not isinstance(name, str) or not name:
            raise CompactionContractError("Selection names must be nonempty")
        if name in self._entries:
            raise CompactionContractError("Selection names cannot be overwritten")
        self._entries[name] = copy.deepcopy(selection)
        return self.describe(name)

    def describe(self, name: str) -> dict:
        if name not in self._entries:
            raise CompactionContractError(f"Unknown selection: {name}")
        return copy.deepcopy(self._entries[name])

    def list(self) -> dict:
        return {"selections": {n: self.describe(n) for n in self._entries}}

    def drop(self, name: str) -> dict:
        result = self.describe(name)
        del self._entries[name]
        return result


# ---------------------------------------------------------------- compute ops
def _as_long_list(values: Any, label: str) -> list[int]:
    if isinstance(values, torch.Tensor):
        if values.ndim != 1 or values.dtype not in {torch.int32, torch.int64}:
            raise CompactionContractError(f"{label} must be a 1-D integer tensor")
        return [int(v) for v in values.tolist()]
    if isinstance(values, (list, tuple)) and all(type(v) is int for v in values):
        return list(values)
    raise CompactionContractError(f"{label} must be a list of ints or an int tensor")


DEFAULT_MEMORY_BUDGET_BYTES = 1 << 30  # transient score/fit blocks per library call


def memory_budget(device: Any, requested: int | None = None) -> dict:
    """Transient-block budget handed to the library's ``chunk_for_budget``.

    ``requested`` (``params['memory_budget_bytes']``) wins when given. Otherwise
    the 1 GiB default is clamped to half of the free device memory
    (``torch.cuda.mem_get_info``) on CUDA devices and used as is on CPU. The
    library derives the query chunk from it (``chunk_for_budget``).
    """
    if requested is not None:
        if type(requested) is not int or requested < 1:
            raise CompactionContractError(
                "memory_budget_bytes must be a positive integer"
            )
        return {"bytes": requested, "policy": "caller", "free_bytes": None}
    device = torch.device(device)
    if device.type == "cuda":
        free, _total = torch.cuda.mem_get_info(device)
        return {
            "bytes": max(1, min(DEFAULT_MEMORY_BUDGET_BYTES, int(free) // 2)),
            "policy": "min(default_1GiB, free_device_memory/2)",
            "free_bytes": int(free),
        }
    return {
        "bytes": DEFAULT_MEMORY_BUDGET_BYTES,
        "policy": "default_1GiB",
        "free_bytes": None,
    }


def _blocking(params: dict, budget: dict | None) -> dict:
    """Either an explicit ``chunk`` or a ``memory_budget_bytes`` for the library."""
    chunk = params.get("chunk")
    if chunk is not None:
        if type(chunk) is not int or chunk < 1:
            raise CompactionContractError("chunk must be a positive integer")
        return {"chunk": chunk}
    if budget is not None:
        return {"memory_budget_bytes": int(budget["bytes"])}
    return {"chunk": DEFAULT_CHUNK}


def library_provenance() -> dict:
    """Package path, git commit and dirtiness of the methods library in use."""
    info: dict[str, Any] = {
        "package": METHODS_PACKAGE,
        "path": None,
        "commit": None,
        "library_commit": None,
        "dirty": None,
        "registered_override": sorted(_REGISTRY),
        "error": None,
    }
    try:
        module = importlib.import_module(METHODS_PACKAGE)
    except ImportError as error:
        info["error"] = f"{type(error).__name__}: {error}"
        return info
    path = os.path.dirname(os.path.abspath(module.__file__))
    info["path"] = path
    try:
        commit = subprocess.run(
            ["git", "-C", path, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        status = subprocess.run(
            ["git", "-C", path, "status", "--porcelain", "--", "."],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        last = subprocess.run(
            ["git", "-C", path, "log", "-1", "--format=%H", "--", "."],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        info["error"] = f"{type(error).__name__}: {error}"
        return info
    if commit.returncode == 0:
        info["commit"] = commit.stdout.strip()
    else:
        info["error"] = commit.stderr.strip() or "git rev-parse failed"
    if status.returncode == 0:
        info["dirty"] = bool(status.stdout.strip())
    if last.returncode == 0 and last.stdout.strip():
        info["library_commit"] = last.stdout.strip()
    return info


def _unwrap_scores(
    result: Any, method: str
) -> tuple[torch.Tensor, str | None, dict | None, list | None]:
    """Accept a ``ScoreResult`` (scores, variant, params, warnings) or a tensor."""
    if isinstance(result, torch.Tensor):
        return result, None, None, None
    scores = getattr(result, "scores", None)
    variant = getattr(result, "variant", None)
    if (
        not isinstance(scores, torch.Tensor)
        or not isinstance(variant, str)
        or not variant
    ):
        raise CompactionContractError(
            f"{method}_scores must return a ScoreResult(scores, variant, params) "
            "or a score tensor"
        )
    params = getattr(result, "params", None)
    warnings = getattr(result, "warnings", None)
    return (
        scores,
        variant,
        _jsonable(dict(params or {})),
        _jsonable(list(warnings)) if warnings is not None else None,
    )


def score_layer(
    method: str,
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    scale: float,
    q_positions: torch.Tensor,
    k_positions: torch.Tensor,
    params: dict,
    k_ref: torch.Tensor | None = None,
    budget: dict | None = None,
) -> dict:
    """Run one scoring method on one layer and validate its ``[Hkv, T]`` output.

    ``h2o``: ``scores.h2o_scores(q, k, scale=, causal=True, q_positions=,
    k_positions=, chunk= | memory_budget_bytes=)`` with request token indices as
    positions, so the causal mask (``k_pos <= q_pos``) follows chronological cache
    order. ``kvzip``: ``scores.kvzip_scores(q_ref, k, scale=, k_ref=,
    normalisation=, chunk= | memory_budget_bytes=)`` where ``k_ref`` are the repeat
    input's own keys (row-aligned with ``q_ref``); ``normalisation='paper'`` needs
    them, ``params['context_only_normalisation']`` selects the library's named
    ``'context_only'`` deviation. ``scale`` is ``head_size ** -0.5``, passed
    explicitly. Returns ``{"scores" (float32 CPU), "variant", "library_params",
    "normalisation", "blocking"}``; ``variant``/``library_params`` are ``None`` when
    the library returned a bare tensor.
    """
    if method not in SCORE_METHODS:
        raise CompactionContractError(f"Unknown scoring method: {method}")
    if q.ndim != 3 or k.ndim != 3 or q.shape[2] != k.shape[2]:
        raise CompactionContractError("q must be [R, Hq, D] and k [T, Hkv, D]")
    if q.shape[1] % k.shape[1] != 0:
        raise CompactionContractError("Query heads must group onto kv heads")
    blocking = _blocking(params, budget)
    normalisation = None
    if method == "h2o":
        result = _call_library(
            "scores",
            "h2o_scores",
            q,
            k,
            scale=scale,
            causal=True,
            q_positions=q_positions.to(k.device),
            k_positions=k_positions.to(k.device),
            **blocking,
        )
    else:
        normalisation = (
            "context_only" if params.get("context_only_normalisation") else "paper"
        )
        if normalisation == "paper" and k_ref is None:
            raise CompactionContractError(
                "kvzip needs the repeat input's own keys (k_ref); pass "
                "params={'context_only_normalisation': True} to accept the deviation"
            )
        if k_ref is not None and tuple(k_ref.shape) != (
            q.shape[0],
            k.shape[1],
            k.shape[2],
        ):
            raise CompactionContractError(
                "k_ref must be [R, Hkv, D] aligned with q_ref"
            )
        extra = {}
        if params.get("repeat_prompt") is not None:
            extra["repeat_prompt"] = params["repeat_prompt"]  # recorded verbatim
        result = _call_library(
            "scores",
            "kvzip_scores",
            q,
            k,
            scale=scale,
            k_ref=k_ref,
            normalisation=normalisation,
            **blocking,
            **extra,
        )
    scores, variant, library_params, warnings = _unwrap_scores(result, method)
    if (
        tuple(scores.shape) != (k.shape[1], k.shape[0])
        or not scores.dtype.is_floating_point
        or not bool(torch.isfinite(scores).all())
    ):
        raise CompactionContractError(
            f"{method}_scores must return a finite [kv_heads, keys] float tensor"
        )
    return {
        "scores": scores.detach().to(device="cpu", dtype=torch.float32),
        "variant": variant,
        "library_params": library_params,
        "library_warnings": warnings,
        "normalisation": normalisation,
        "blocking": blocking,
    }


def select_layers(
    scores: list[torch.Tensor],
    *,
    budget_tokens: int,
    protected: list[int],
    policy: str,
    aggregate: str,
) -> list[list[int]]:
    """Call ``select.uniform_token_budget`` and validate equal-count key lists.

    Returns one sorted list of key positions (indices into each layer's scored
    key axis) per layer, each of length exactly ``budget_tokens`` and containing
    ``protected``. ``budget_tokens`` above the number of scored keys is refused
    (the library requires exact counts). ``policy='shared'`` requires identical
    lists across layers.
    """
    if policy not in SELECT_POLICIES or aggregate not in SELECT_AGGREGATES:
        raise CompactionContractError("Unknown selection policy or aggregate")
    if type(budget_tokens) is not int or budget_tokens < 1:
        raise CompactionContractError("budget_tokens must be a positive integer")
    if not scores or any(t.ndim != 2 for t in scores):
        raise CompactionContractError("Selection needs [kv_heads, keys] scores")
    num_keys = {int(t.shape[1]) for t in scores}
    if len(num_keys) != 1:
        raise CompactionContractError("Layers must score the same key set")
    keys = num_keys.pop()
    if len(protected) > budget_tokens:
        raise CompactionContractError("protected exceeds budget_tokens")
    if budget_tokens > keys:
        raise CompactionContractError(
            f"budget_tokens ({budget_tokens}) exceeds the scored keys ({keys})"
        )
    result = _call_library(
        "select",
        "uniform_token_budget",
        scores,
        budget_tokens,
        protected=list(protected),
        policy=policy,
        aggregate=aggregate,
    )
    if not isinstance(result, (list, tuple)) or len(result) != len(scores):
        raise CompactionContractError(
            "uniform_token_budget must return one list per layer"
        )
    selections = []
    for chosen in result:
        chosen = _as_long_list(chosen, "selection")
        if (
            len(chosen) != budget_tokens
            or chosen != sorted(set(chosen))
            or any(not 0 <= i < keys for i in chosen)
            or not set(protected) <= set(chosen)
        ):
            raise CompactionContractError(
                "Selections must be sorted, unique, inside the key range, keep "
                "protected keys and match the budget"
            )
        selections.append(chosen)
    if policy == "shared" and any(s != selections[0] for s in selections):
        raise CompactionContractError("shared policy requires identical layer lists")
    return selections


AM_SUMMARY_KEYS = (
    "frame_policy",
    "n_fixed",
    "n_protected",
    "solver",
    "accumulate_dtype",
    "compute_dtype",
    "chunk",
    "memory_budget_bytes",
    "identity_shortcut",
    "output_error_before_rel",
    "output_error_after_rel",
    "n_dead_columns",
    "value_norm_ratio",
    "condition_estimate",
    "value_guard_rounds",
    "max_fit_workspace_bytes",
    "fit_workspace_admission",
    "warnings",
    "mass_weighting",
    "mass_error_before",
    "mass_error_after",
    "bias_fallback",
    "any_bias_fallback",
)


def fit_workspace_capacity(device: Any) -> dict:
    """Admission capacity for the AM fit workspace (``max_fit_workspace_bytes``).

    On CUDA: driver-free memory plus the allocator's cached-but-unallocated
    reserve, sampled when called (the caller does so with the layer's inputs
    already on the device). On CPU: ``None`` (no admission limit). The library
    compares its lower bound of unavoidable new tensors against this; passing
    does not certify the peak (Codex R1).
    """
    device = torch.device(device)
    if device.type != "cuda":
        return {
            "bytes": None,
            "policy": "cpu_no_limit",
            "free_bytes": None,
            "allocator_cached_bytes": None,
        }
    free, _total = torch.cuda.mem_get_info(device)
    cached = max(
        0, torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    )
    return {
        "bytes": int(free) + int(cached),
        "policy": "device_free_plus_allocator_cached",
        "free_bytes": int(free),
        "allocator_cached_bytes": int(cached),
    }


def fit_am_layer(
    *,
    q_ref: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    budget: int,
    protected: list[int],
    fixed: list[int],
    scale: float,
    params: dict,
    budget_bytes: dict | None = None,
    max_fit_workspace_bytes: int | None = None,
    bias: bool = False,
    mass_weighting: str = "uniform",
) -> dict:
    """Call ``am.compact`` (uniform head budget, optional bias) for one layer.

    Contract (``profiling.compaction_methods.am``): ``compact(q_ref, k, v, budget,
    bias=, head_budget='uniform', protected=, fixed=, scale=, ridge=,
    chunk= | memory_budget_bytes=[, mass_weighting=])`` returns an ``AMResult``:
    ``token_indices()`` lists the shared ascending key positions (``len ==
    budget``, containing ``protected`` and ``fixed``), ``cache_layout()`` gives
    ``(k_c, v_c)`` as ``[t, Hkv, D]`` rows aligned with them, ``fixed_mask``
    (``bool [Hkv, t]``) marks frozen frame rows. ``fixed`` tokens keep their
    original K *and* V (frame policy ``fixed``; asserted here row by row);
    ``protected`` tokens are kept but refit-able (``refit``). ``k_c`` must equal
    the original keys. With ``bias=False`` (the ``*_nobias`` variant) ``beta``
    must be ``None``; with ``bias=True`` (``mass_weighting`` is passed through)
    ``beta`` must be a finite float ``[Hkv, t]`` that is exactly zero on fixed
    rows, returned here as float32 on the CPU. Per-head budgets and ``iters``
    stay refused by the caller's param allowlist. ``v_c`` comes back in ``v``'s
    dtype; pass a float32 ``v`` to obtain pre-cast values. Returns
    ``{"indices", "k_c", "v_c", "beta", "fixed_positions", "variant",
    "diagnostics", "summary", "blocking"}``.
    """
    if not isinstance(bias, bool):
        raise CompactionContractError("bias must be a bool")
    if mass_weighting not in ("uniform", "shifted"):
        raise CompactionContractError("mass_weighting must be 'uniform' or 'shifted'")
    if (
        k.ndim != 3
        or v.shape != k.shape
        or q_ref.ndim != 3
        or q_ref.shape[2] != k.shape[2]
    ):
        raise CompactionContractError("AM needs q_ref [R, Hq, D], k and v [T, Hkv, D]")
    if type(budget) is not int or budget < 1:
        raise CompactionContractError("budget must be a positive integer")
    tokens = int(k.shape[0])
    if budget > tokens:
        raise CompactionContractError(
            f"budget ({budget}) exceeds the source keys ({tokens})"
        )
    if set(protected) & set(fixed):
        raise CompactionContractError("protected and fixed positions must be disjoint")
    if len(protected) + len(fixed) > budget:
        raise CompactionContractError("protected + fixed exceeds budget")
    ridge = params.get("ridge", 0.0)
    if isinstance(ridge, bool) or not isinstance(ridge, (int, float)) or ridge < 0:
        raise CompactionContractError("ridge must be a nonnegative number")
    blocking = _blocking(params, budget_bytes)
    bias_kwargs = {"mass_weighting": mass_weighting} if bias else {}
    result = _call_library(
        "am",
        "compact",
        q_ref,
        k,
        v,
        budget,
        bias=bias,
        head_budget="uniform",
        protected=list(protected),
        fixed=list(fixed),
        scale=scale,
        ridge=ridge,
        max_fit_workspace_bytes=max_fit_workspace_bytes,
        **blocking,
        **bias_kwargs,
    )
    beta = getattr(result, "beta", None)
    if not bias:
        if beta is not None:
            raise CompactionContractError(
                "AM beta is not importable in M1 (bias=False)"
            )
    elif (
        not isinstance(beta, torch.Tensor)
        or tuple(beta.shape) != (int(k.shape[1]), budget)
        or not beta.dtype.is_floating_point
        or not bool(torch.isfinite(beta).all())
    ):
        raise CompactionContractError("AM beta must be finite [Hkv, budget] with bias")
    else:
        beta = beta.detach().to(dtype=torch.float32).cpu()
    if (
        not getattr(result, "shared", False)
        or not callable(getattr(result, "token_indices", None))
        or not callable(getattr(result, "cache_layout", None))
    ):
        raise CompactionContractError(
            "AM must return a shared-token AMResult with token_indices()/cache_layout()"
        )
    indices = _as_long_list(result.token_indices(), "AM token_indices")
    if (
        len(indices) != budget
        or indices != sorted(set(indices))
        or any(not 0 <= i < tokens for i in indices)
        or not (set(protected) | set(fixed)) <= set(indices)
    ):
        raise CompactionContractError(
            "AM indices must be sorted, unique, inside the key range, keep "
            "protected and fixed keys and match the budget"
        )
    layout = result.cache_layout()
    if not isinstance(layout, tuple) or len(layout) != 2:
        raise CompactionContractError("AM cache_layout() must return (k_c, v_c)")
    k_c, v_c = layout
    expected = (budget, int(k.shape[1]), int(k.shape[2]))
    for label, tensor in (("k_c", k_c), ("v_c", v_c)):
        if (
            not isinstance(tensor, torch.Tensor)
            or tuple(tensor.shape) != expected
            or not tensor.dtype.is_floating_point
            or not bool(torch.isfinite(tensor).all())
        ):
            raise CompactionContractError(f"AM {label} must be finite [budget, Hkv, D]")
    index = torch.tensor(indices, dtype=torch.long, device=k.device)
    if not torch.equal(k_c.to(device=k.device, dtype=k.dtype), k[index]):
        raise CompactionContractError("AM k_c rows differ from the original keys")
    fixed_mask = getattr(result, "fixed_mask", None)
    if fixed_mask is None:
        if fixed:
            raise CompactionContractError("AM result lacks fixed_mask for fixed rows")
        fixed_rows = torch.zeros(budget, dtype=torch.bool)
    else:
        if (
            not isinstance(fixed_mask, torch.Tensor)
            or fixed_mask.dtype != torch.bool
            or tuple(fixed_mask.shape) != (int(k.shape[1]), budget)
            or not torch.equal(fixed_mask.all(dim=0), fixed_mask.any(dim=0))
        ):
            raise CompactionContractError("AM fixed_mask must be bool [Hkv, t], shared")
        fixed_rows = fixed_mask[0].cpu()
    flagged = {indices[i] for i in range(budget) if bool(fixed_rows[i])}
    if flagged != set(fixed):
        raise CompactionContractError(
            f"AM fixed_mask marks {sorted(flagged)} but fixed = {sorted(fixed)}"
        )
    if flagged:
        rows = fixed_rows.to(k.device)
        original = v[index][rows].to(device=v_c.device, dtype=v_c.dtype)
        if not torch.equal(v_c.to(v_c.device)[rows.to(v_c.device)], original):
            raise CompactionContractError(
                "AM v_c differs from the original values on fixed rows"
            )
        if beta is not None and bool(beta[:, fixed_rows].ne(0).any()):
            raise CompactionContractError("AM beta must be zero on fixed rows")
    diagnostics = _jsonable(getattr(result, "diagnostics", {}))
    return {
        "indices": indices,
        "k_c": k_c.detach(),
        "v_c": v_c.detach(),
        "beta": beta,
        "fixed_positions": sorted(flagged),
        "variant": str(getattr(result, "variant", "unknown")),
        "warnings": _jsonable(list(getattr(result, "warnings", None) or [])),
        "diagnostics": diagnostics,
        "summary": {
            key: diagnostics.get(key) if isinstance(diagnostics, dict) else None
            for key in AM_SUMMARY_KEYS
        },
        "blocking": blocking,
    }


def cast_error(reference: torch.Tensor, cast: torch.Tensor) -> dict:
    """Value-space error introduced by casting fitted values to the cache dtype."""
    reference = reference.detach().float()
    difference = cast.detach().float() - reference
    norm = float(reference.norm())
    return {
        "max_abs": float(difference.abs().max()) if difference.numel() else 0.0,
        "rel_fro": float(difference.norm()) / norm if norm > 0 else 0.0,
    }


OUTPUT_ERROR_MAX_TOKENS = 65536  # default the post-cast error pass off above this


def attention_output_error(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_c: torch.Tensor,
    v_c: torch.Tensor,
    *,
    scale: float,
    budget_bytes: int = DEFAULT_MEMORY_BUDGET_BYTES,
    beta: torch.Tensor | None = None,
) -> dict:
    """In-sample output error of the *stored* compacted rows against the cache.

    Per kv head ``h`` (query heads ``h*G .. (h+1)*G-1``, contiguous GQA blocks) in
    float32: ``Y_ref = softmax(scale q K_h^T) V_h`` over all ``T`` keys and
    ``Y_c = softmax(scale q K_c^T + beta_h) V_c`` over the retained rows exactly
    as they will sit in the cache (after any dtype cast; ``beta`` is the
    ``[Hkv, t]`` per-key bias the kernel adds, zero when None), for the same
    reference queries the fit used. Memory: only per-head key/value slices and one query
    block are cast to float32; the block holds ``rows_per_block =
    budget_bytes // (3 * (T + t) * 4)`` grouped rows (two float32 logits blocks
    plus slack, the library's ``chunk_for_budget`` rule with ``g = 1`` because
    the grouped rows already fold ``G`` in). Device OOM becomes a contract error.
    Returns per-head absolute/relative (``/ ||Y_ref||_F``) Frobenius errors,
    ``max_rel``, ``rows_per_block`` and ``budget_bytes``.
    """
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape or k_c.shape != v_c.shape:
        raise CompactionContractError("output error needs q [R,Hq,D], k/v and k_c/v_c")
    heads, kv_heads = int(q.shape[1]), int(k.shape[1])
    if heads % kv_heads or k_c.shape[1] != kv_heads or k_c.shape[2] != k.shape[2]:
        raise CompactionContractError("output error: head layout mismatch")
    if type(budget_bytes) is not int or budget_bytes < 1:
        raise CompactionContractError("budget_bytes must be a positive integer")
    if beta is not None and (
        not isinstance(beta, torch.Tensor)
        or tuple(beta.shape) != (kv_heads, int(k_c.shape[0]))
    ):
        raise CompactionContractError("output error: beta must be [Hkv, t]")
    group = heads // kv_heads
    tokens, retained = int(k.shape[0]), int(k_c.shape[0])
    rows_per_block = max(1, budget_bytes // (3 * (tokens + retained) * 4))
    queries_per_block = max(1, rows_per_block // group)
    device = k.device
    absolute, relative = [], []
    try:
        for h in range(kv_heads):
            keys_h = k[:, h, :].detach().float()
            values_h = v[:, h, :].detach().float()
            keys_c = k_c[:, h, :].detach().to(device).float()
            values_c = v_c[:, h, :].detach().to(device).float()
            beta_h = None if beta is None else beta[h].detach().to(device).float()
            numerator = denominator = 0.0
            for start in range(0, int(q.shape[0]), queries_per_block):
                block = (
                    q[start : start + queries_per_block, h * group : (h + 1) * group, :]
                    .reshape(-1, int(q.shape[2]))
                    .detach()
                    .float()
                )
                y_ref = torch.softmax(block @ keys_h.T * scale, dim=-1) @ values_h
                logits_c = block @ keys_c.T * scale
                if beta_h is not None:
                    logits_c = logits_c + beta_h[None, :]
                y_c = torch.softmax(logits_c, dim=-1) @ values_c
                numerator += float(((y_ref - y_c) ** 2).sum())
                denominator += float((y_ref**2).sum())
            absolute.append(math.sqrt(numerator))
            relative.append(
                math.sqrt(numerator) / math.sqrt(denominator)
                if denominator > 0
                else 0.0
            )
    except RuntimeError as error:
        if (
            type(error).__name__ != "OutOfMemoryError"
            and "out of memory" not in str(error).lower()
        ):
            raise
        raise CompactionContractError(
            f"output error pass ran out of memory (T={tokens}, t={retained}, "
            f"rows_per_block={queries_per_block * group}, budget={budget_bytes} "
            f"bytes): {error}"
        ) from error
    return {
        "abs": absolute,
        "rel": relative,
        "max_rel": max(relative),
        "rows_per_block": queries_per_block * group,
        "budget_bytes": budget_bytes,
    }
