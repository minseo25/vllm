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
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
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

    An entry is opened when the operation is armed (reserving the name), filled
    chunk by chunk by :class:`QueryExportPlan`, and ``complete`` once every row
    of ``token_range`` was written for every layer. Buffers are pinned when the
    source query lives on a CUDA device; ``describe`` records ``pinned``.
    """

    def __init__(self, layer_names: Any):
        names = list(layer_names)
        if not names or len(set(names)) != len(names):
            raise CompactionContractError("Query export needs distinct FA layers")
        self.layer_names = tuple(names)
        self._entries: dict[str, dict] = {}

    def open(
        self,
        name: str,
        *,
        token_range: Any,
        request_id: str | None = None,
        operation_id: str | None = None,
    ) -> dict:
        if not isinstance(name, str) or not name:
            raise CompactionContractError("Query export name must be nonempty")
        if name in self._entries:
            raise CompactionContractError("Query export names cannot be reused")
        start, end = _check_token_range(token_range, "export_q token_range")
        rows = end - start
        self._entries[name] = {
            "name": name,
            "token_range": [start, end],
            "request_id": request_id,
            "operation_id": operation_id,
            "rows": rows,
            "layer_names": list(self.layer_names),
            "tensors": {},
            "shapes": {},
            "layouts": {},
            "query_convention": dict(QUERY_CONVENTION),
            "dtype": None,
            "pinned": None,
            "positions": torch.full((rows,), -1, dtype=torch.long),
            "cursors": torch.arange(start, end, dtype=torch.long),
            "filled": torch.zeros(rows, dtype=torch.bool),
            "chunks": [],
            "bytes": 0,
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
        """Copy ``rows`` ([n, Hq, D], device) into host rows starting at cursor.

        ``layout`` records the layer's head geometry (``num_heads``,
        ``num_kv_heads``, ``head_size``, ``group_size``) in the receipt so the
        GQA grouping of the exported heads is explicit.
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
        if layout is not None:
            heads, kv_heads = layout.get("num_heads"), layout.get("num_kv_heads")
            if (
                type(heads) is not int
                or type(kv_heads) is not int
                or kv_heads < 1
                or heads % kv_heads
                or heads != rows.shape[1]
                or layout.get("head_size") != rows.shape[2]
            ):
                raise CompactionContractError(
                    f"Query head layout is inconsistent with the rows: {layer_name}"
                )
            layout = {**layout, "group_size": heads // kv_heads}
            previous = entry["layouts"].get(layer_name)
            if previous is not None and previous != layout:
                raise CompactionContractError(
                    f"Query head layout changed between chunks: {layer_name}"
                )
            entry["layouts"][layer_name] = layout
        start, _ = entry["token_range"]
        first = cursor_start - start
        last = first + rows.shape[0]
        if first < 0 or last > entry["rows"]:
            raise CompactionContractError("Exported rows fall outside token_range")
        if bool(entry["filled"][first:last].any()):
            raise CompactionContractError("Query rows were already exported")
        destination = entry["tensors"].get(layer_name)
        if destination is None:
            pinned = bool(rows.is_cuda)
            destination = torch.empty(
                (entry["rows"], rows.shape[1], rows.shape[2]),
                dtype=rows.dtype,
                device="cpu",
                pin_memory=pinned,
            )
            if entry["dtype"] is None:
                entry["dtype"] = str(rows.dtype)
                entry["pinned"] = pinned
            elif entry["dtype"] != str(rows.dtype) or entry["pinned"] != pinned:
                raise CompactionContractError(
                    "FA layers disagree on query dtype or device for export"
                )
            entry["tensors"][layer_name] = destination
            entry["shapes"][layer_name] = list(destination.shape)
            entry["bytes"] += destination.numel() * destination.element_size()
        elif (
            tuple(destination.shape[1:]) != tuple(rows.shape[1:])
            or destination.dtype != rows.dtype
        ):
            raise CompactionContractError(
                f"Query shape/dtype changed between chunks: {layer_name}"
            )
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
    ) -> dict:
        if not isinstance(name, str) or not name:
            raise CompactionContractError("Score names must be nonempty")
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
) -> torch.Tensor:
    """Run one scoring method on one layer and validate its ``[Hkv, T]`` output.

    ``h2o``: ``scores.h2o_scores(q, k, scale=, causal=, q_positions=,
    k_positions=, chunk=)`` with request token indices as positions, so the
    causal mask (``k_pos <= q_pos``) follows chronological cache order.
    ``kvzip``: ``scores.kvzip_scores(q_ref, k, scale=, k_ref=, chunk=)`` where
    ``k_ref`` are the repeat input's own keys (row-aligned with ``q_ref``);
    ``k_ref=None`` is the library's named "context-only normalisation"
    deviation and is only allowed with ``params['context_only_normalisation']``.
    ``scale`` is the model's ``head_size ** -0.5``, passed explicitly.
    """
    if method not in SCORE_METHODS:
        raise CompactionContractError(f"Unknown scoring method: {method}")
    if q.ndim != 3 or k.ndim != 3 or q.shape[2] != k.shape[2]:
        raise CompactionContractError("q must be [R, Hq, D] and k [T, Hkv, D]")
    if q.shape[1] % k.shape[1] != 0:
        raise CompactionContractError("Query heads must group onto kv heads")
    chunk = params.get("chunk", DEFAULT_CHUNK)
    if type(chunk) is not int or chunk < 1:
        raise CompactionContractError("chunk must be a positive integer")
    if method == "h2o":
        result = _call_library(
            "scores",
            "h2o_scores",
            q,
            k,
            scale=scale,
            causal=bool(params.get("causal", True)),
            q_positions=q_positions.to(k.device),
            k_positions=k_positions.to(k.device),
            chunk=chunk,
        )
    else:
        if k_ref is None and not params.get("context_only_normalisation"):
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
        result = _call_library(
            "scores", "kvzip_scores", q, k, scale=scale, k_ref=k_ref, chunk=chunk
        )
    if (
        not isinstance(result, torch.Tensor)
        or tuple(result.shape) != (k.shape[1], k.shape[0])
        or not result.dtype.is_floating_point
        or not bool(torch.isfinite(result).all())
    ):
        raise CompactionContractError(
            f"{method}_scores must return a finite [kv_heads, keys] float tensor"
        )
    return result.detach().to(device="cpu", dtype=torch.float32)


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
    key axis) per layer, all of length ``min(budget_tokens, keys)`` and all
    containing ``protected``. ``policy='shared'`` requires identical lists.
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
    expected = budget_tokens
    if not isinstance(result, (list, tuple)) or len(result) != len(scores):
        raise CompactionContractError(
            "uniform_token_budget must return one list per layer"
        )
    selections = []
    for chosen in result:
        chosen = _as_long_list(chosen, "selection")
        if (
            len(chosen) != expected
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


def fit_am_layer(
    *,
    q_ref: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    budget: int,
    protected: list[int],
    scale: float,
    params: dict,
) -> dict:
    """Call ``am.compact`` (no bias, uniform head budget) for one layer.

    Contract (``profiling.compaction_methods.am``): ``compact(q_ref, k, v, budget,
    bias=False, head_budget='uniform', protected=, scale=, chunk=, ridge=)`` returns
    an ``AMResult`` whose ``token_indices()`` lists the shared, ascending key
    positions (``len == budget``, containing ``protected``) and whose
    ``cache_layout()`` gives ``(k_c, v_c)`` as ``[t, Hkv, D]`` rows aligned with
    them; ``k_c`` are the original keys (checked exactly here), ``v_c`` the fitted
    values; ``beta`` must be ``None`` (bias and per-head budgets are M2).
    Returns ``{"indices", "k_c", "v_c", "variant", "diagnostics"}``.
    """
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
    if len(protected) > budget:
        raise CompactionContractError("protected exceeds budget")
    chunk = params.get("chunk", DEFAULT_CHUNK)
    if type(chunk) is not int or chunk < 1:
        raise CompactionContractError("chunk must be a positive integer")
    ridge = params.get("ridge", 0.0)
    if isinstance(ridge, bool) or not isinstance(ridge, (int, float)) or ridge < 0:
        raise CompactionContractError("ridge must be a nonnegative number")
    result = _call_library(
        "am",
        "compact",
        q_ref,
        k,
        v,
        budget,
        bias=False,
        head_budget="uniform",
        protected=list(protected),
        scale=scale,
        chunk=chunk,
        ridge=ridge,
    )
    if getattr(result, "beta", None) is not None:
        raise CompactionContractError("AM beta is not importable in M1 (bias=False)")
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
        or not set(protected) <= set(indices)
    ):
        raise CompactionContractError(
            "AM indices must be sorted, unique, inside the key range, keep "
            "protected keys and match the budget"
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
    return {
        "indices": indices,
        "k_c": k_c.detach(),
        "v_c": v_c.detach(),
        "variant": str(getattr(result, "variant", "unknown")),
        "diagnostics": _jsonable(getattr(result, "diagnostics", {})),
    }
