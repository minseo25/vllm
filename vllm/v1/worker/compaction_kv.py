# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact selected-KV export/import for controlled compaction experiments.

The scheduler still owns every cache page. Import replaces an already computed
scratch prefix, at its exact boundary, before the next token is consumed. It
does not reconstruct KV from a recurrent state or merely mask a full cache.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import torch


def _digest(tensors: dict[str, torch.Tensor]) -> str:
    result = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        result.update(
            json.dumps([name, list(tensor.shape), str(tensor.dtype)]).encode()
        )
        result.update(tensor.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


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

    @staticmethod
    def _indices(spec: dict, computed_tokens: int) -> list[int]:
        if not isinstance(spec, dict) or set(spec) != {"name", "token_indices"}:
            raise ValueError("KV capture requires name and token_indices")
        if not isinstance(spec["name"], str) or not spec["name"]:
            raise ValueError("KV snapshot name must be nonempty")
        indices = spec["token_indices"]
        if (
            not isinstance(indices, list)
            or not indices
            or any(type(i) is not int or not 0 <= i < computed_tokens for i in indices)
            or indices != sorted(set(indices))
        ):
            raise ValueError("KV indices must be unique, chronological, and consumed")
        return indices

    def _locations(
        self, row: int, metadata: dict, indices: list[int]
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Validate every layer before a caller can mutate any layer."""
        if type(row) is not int or row < 0 or not self.layers:
            raise ValueError("KV import requires a real request row and FA layers")
        locations = {}
        for name, layer in self.layers.items():
            cache = layer.kv_cache
            if (
                layer.get_attn_backend().get_name() not in {"FLASH_ATTN", "TRITON_ATTN"}
                or not isinstance(cache, torch.Tensor)
                or cache.ndim != 4
                or cache.dtype not in {torch.float16, torch.bfloat16, torch.float32}
                or cache.shape[3] != 2 * layer.head_size
                or cache.shape[1] != layer.num_kv_heads
            ):
                raise ValueError(f"Unsupported exact-KV cache layout: {name}")
            table = getattr(metadata.get(name), "block_table", None)
            if not isinstance(table, torch.Tensor) or table.ndim != 2:
                raise ValueError(f"Missing kernel block table: {name}")
            block_size = cache.shape[2]
            logical = torch.tensor(indices, dtype=torch.long)
            block_columns = logical // block_size
            if row >= table.shape[0] or int(block_columns.max()) >= table.shape[1]:
                raise ValueError(f"KV selection exceeds allocated block table: {name}")
            physical = table[row].cpu()[block_columns].to(dtype=torch.long)
            offsets = logical % block_size
            # Block zero is the scheduler's reserved NULL page.
            if bool(((physical <= 0) | (physical >= cache.shape[0])).any()):
                raise ValueError(f"KV selection points to NULL/invalid pages: {name}")
            addresses = physical * block_size + offsets
            if addresses.unique().numel() != len(indices):
                raise ValueError(f"Aliased KV destination addresses: {name}")
            locations[name] = (
                cache,
                physical.to(cache.device),
                offsets.to(cache.device),
            )
        return locations

    def validate_capture(
        self,
        spec: dict,
        request_id: str,
        row: int,
        metadata: dict,
        computed_tokens: int,
    ) -> None:
        indices = self._indices(spec, computed_tokens)
        if spec["name"] in self._entries:
            raise ValueError("KV snapshot names cannot be overwritten")
        self._locations(row, metadata, indices)

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
        locations = self._locations(row, metadata, spec["token_indices"])
        tensors = {
            name: cache[blocks, :, offsets, :].detach().cpu().clone()
            for name, (cache, blocks, offsets) in locations.items()
        }
        if not all(bool(torch.isfinite(t).all()) for t in tensors.values()):
            raise ValueError("Non-finite selected KV values")
        entry = {
            "name": spec["name"],
            "request_id": request_id,
            "source_cursor": computed_tokens,
            "source_position_offset": source_position_offset,
            "source_absolute_next_position": computed_tokens + source_position_offset,
            "token_indices": list(spec["token_indices"]),
            "retained_tokens": len(spec["token_indices"]),
            "bytes": sum(t.numel() * t.element_size() for t in tensors.values()),
            "digest": _digest(tensors),
            "digest_verified_at": "capture",
            "tensors": tensors,
        }
        self._entries[spec["name"]] = entry
        return self.describe(spec["name"])

    def describe(self, name: str) -> dict:
        if name not in self._entries:
            raise ValueError(f"Unknown KV snapshot: {name}")
        result = {
            key: copy.deepcopy(value)
            for key, value in self._entries[name].items()
            if key != "tensors"
        }
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

    def validate_restore(
        self,
        name: str,
        request_id: str,
        row: int,
        metadata: dict,
        computed_tokens: int,
    ) -> None:
        entry = self.describe(name)
        if entry["source_position_offset"]:
            raise ValueError(
                "Repeated selected-KV import from an offset source is unsupported; "
                "this snapshot is available for audit only"
            )
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
        return {
            "name": name,
            "request_id": request_id,
            "cursor": computed_tokens,
            "source_cursor": self._entries[name]["source_cursor"],
            "retained_tokens": computed_tokens,
            "payload_bytes": self._entries[name]["bytes"],
            "occupied_kernel_page_bytes": page_bytes,
            "source_digest": self._entries[name]["digest"],
            "copy_kind": "exact_indexed_assignment",
            "scratch_prefill_tokens": computed_tokens,
        }

    def drop(self, name: str) -> dict:
        result = self.describe(name)
        del self._entries[name]
        self._staged.pop(name, None)
        return result
