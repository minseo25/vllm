# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional per-request compaction at scheduler-owned V1 token boundaries."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import torch

from vllm.v1.worker.kv_bias import UNSUPPORTED_MESSAGE as KV_BIAS_UNSUPPORTED

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

DESCRIPTOR_KEY = "native_compaction_v1"
COMPONENTS = {"all": (0, 1), "conv": (0,), "matrix": (1,)}
EXPORT_GRAPH_MODES = {"NONE", "PIECEWISE"}


class CompactionContractError(ValueError):
    """The requested operation is outside the supported diagnostic contract."""


def _q_module():
    """Lazy import: compaction_q imports this module for the error type."""
    from vllm.v1.worker import compaction_q

    return compaction_q


def validate_export_q(export_q: Any, start_cursor: int, prompt_tokens: int) -> dict:
    """Normalize ``{"name", "token_range": [start, end]}`` inside the new tokens."""
    if not isinstance(export_q, dict) or set(export_q) != {"name", "token_range"}:
        raise CompactionContractError("export_q requires name and token_range")
    name, token_range = export_q["name"], export_q["token_range"]
    if not isinstance(name, str) or not name:
        raise CompactionContractError("export_q name must be nonempty")
    if (
        not isinstance(token_range, (list, tuple))
        or len(token_range) != 2
        or any(type(x) is not int for x in token_range)
        or not start_cursor <= token_range[0] < token_range[1] <= prompt_tokens
    ):
        raise CompactionContractError(
            "export_q token_range must satisfy start_cursor <= start < end <= "
            "expected_prompt_tokens"
        )
    return {"name": name, "token_range": [int(token_range[0]), int(token_range[1])]}


def validate_configuration(config: Any) -> None:
    """Reject cache lifecycles not covered by this worker integration."""
    required = (
        (config.parallel_config.tensor_parallel_size == 1, "TP must be 1"),
        (config.parallel_config.pipeline_parallel_size == 1, "PP must be 1"),
        (config.parallel_config.data_parallel_size == 1, "DP must be 1"),
        (
            config.scheduler_config.async_scheduling is False,
            "async scheduling is unsupported",
        ),
        (
            config.scheduler_config.enable_chunked_prefill,
            "chunked prefill must be enabled",
        ),
        (
            not getattr(config.parallel_config, "use_ubatching", False),
            "microbatching is unsupported",
        ),
        (
            not config.cache_config.enable_prefix_caching,
            "prefix caching is unsupported",
        ),
        (
            config.cache_config.mamba_cache_mode == "none",
            "mamba_cache_mode must be none",
        ),
        (not config.cache_config.use_replayssm, "ReplaySSM is unsupported"),
        (config.speculative_config is None, "speculative decoding is unsupported"),
        (
            config.mamba_config.state_quant_bits is None,
            "state quantization is unsupported",
        ),
        (config.mamba_config.state_trace_dir is None, "state tracing is unsupported"),
        (not config.mamba_config.enable_stochastic_rounding, "state SR is unsupported"),
        (getattr(config, "lora_config", None) is None, "LoRA is unsupported"),
        (
            getattr(config, "kv_transfer_config", None) is None,
            "KV transfer is unsupported",
        ),
        (
            getattr(config, "ec_transfer_config", None) is None,
            "encoder transfer is unsupported",
        ),
    )
    for passed, reason in required:
        if not passed:
            raise CompactionContractError(reason)
    model_type = config.model_config.hf_config.model_type
    if model_type not in {"qwen3_5", "qwen3_5_text", "nemotron_h"}:
        raise CompactionContractError(f"Unsupported model_type: {model_type}")


@dataclass(frozen=True)
class StateSlot:
    """A scheduler-owned native slot resolved from current forward metadata."""

    name: str
    family: str
    states: tuple[torch.Tensor, ...]
    index: int
    metadata: Any
    prefill: bool
    row: int = 0
    prefill_row: int = 0


def resolve_slot(
    name: str,
    family: str,
    states: tuple,
    metadata: Any,
    *,
    row: int = 0,
    index: int | None = None,
) -> StateSlot:
    """Resolve a real batch row, excluding graph padding and speculative rows.

    The controller supplies the scheduler's CPU block-table index to avoid a
    device synchronization for every layer. Standalone callers can resolve the
    index from metadata instead.
    """
    if family not in {"gdn", "mamba2"} or len(states) != 2:
        raise CompactionContractError(
            "Only native conv+SSM GDN/Mamba2 layouts are supported"
        )
    prefills, decodes = int(metadata.num_prefills), int(metadata.num_decodes)
    if not 0 <= row < prefills + decodes:
        raise CompactionContractError("State row is outside the current batch")
    prefill = row >= decodes
    local_row = row - decodes if prefill else row
    if family == "gdn":
        if getattr(metadata, "num_spec_decodes", 0) != 0:
            raise CompactionContractError("Speculative GDN rows are unsupported")
        indices = (
            metadata.prefill_state_indices
            if prefill
            else metadata.non_spec_state_indices_tensor
        )
    else:
        indices = (
            metadata.state_indices_tensor_p
            if prefill
            else metadata.state_indices_tensor_d
        )
    if (
        not isinstance(indices, torch.Tensor)
        or indices.ndim < 1
        or local_row >= indices.shape[0]
        or indices[local_row].numel() != 1
    ):
        raise CompactionContractError(f"Unexpected state index layout in {name}")
    if index is None or indices.device.type == "cpu":
        metadata_index = int(indices[local_row].item())
        if index is not None and index != metadata_index:
            raise CompactionContractError(f"CPU/metadata state slots differ: {name}")
        index = metadata_index
    for state in states:
        if (
            not isinstance(state, torch.Tensor)
            or state.ndim < 2
            or not 0 < index < state.shape[0]
        ):
            raise CompactionContractError(f"Invalid native state slot in {name}")
    slot = StateSlot(
        name, family, tuple(states), index, metadata, prefill, row, local_row
    )
    _initial_state_flags(slot)  # Validate before any restore writes.
    return slot


def _initial_state_flags(slot: StateSlot) -> list[torch.Tensor]:
    if not slot.prefill:
        return []  # Native single-token decode unconditionally consumes its slot.
    keys = (
        (
            ("has_initial_state", slot.row),
            ("prefill_has_initial_state", slot.prefill_row),
        )
        if slot.family == "gdn"
        else (("has_initial_states_p", slot.prefill_row),)
    )
    flags = []
    for key, row in keys:
        flag = getattr(slot.metadata, key, None)
        if (
            not isinstance(flag, torch.Tensor)
            or flag.dtype != torch.bool
            or flag.ndim != 1
            or not 0 <= row < flag.numel()
        ):
            raise CompactionContractError(f"Invalid {key} in {slot.name}")
        flags.append(flag[row : row + 1])
    if slot.family == "mamba2" and not hasattr(slot.metadata, "prep_initial_states"):
        raise CompactionContractError(
            f"Missing Mamba2 prep_initial_states in {slot.name}"
        )
    return flags


def verify_fa_context(
    metadata: dict,
    layer_names: set[str],
    *,
    query_tokens: int,
    computed_tokens: int,
    row: int = 0,
) -> dict:
    """Read actual FA lengths; reject unknown layouts instead of inventing proof."""
    if not layer_names:
        raise CompactionContractError("No full-attention layers were found")
    evidence = {}
    for name in sorted(layer_names):
        current = metadata.get(name)
        lengths = getattr(current, "seq_lens", None)
        location = "seq_lens"
        local_row = row
        starts = getattr(current, "query_start_loc", None)
        if lengths is None:
            decodes = int(getattr(current, "num_decodes", 0))
            part = "decode" if row < decodes else "prefill"
            local_row = row if row < decodes else row - decodes
            nested = getattr(current, part, None)
            lengths = getattr(nested, "seq_lens", None)
            starts = getattr(nested, "cum_seq_lens_q", None)
            location = f"{part}.seq_lens"
        if (
            not isinstance(lengths, torch.Tensor)
            or lengths.ndim != 1
            or not 0 <= local_row < lengths.numel()
        ):
            raise CompactionContractError(
                f"Unsupported FA sequence-length metadata: {name}"
            )
        seq_len = int(lengths[local_row].item())
        context_len = seq_len - query_tokens
        if context_len != computed_tokens:
            raise CompactionContractError(
                f"FA context and worker cursor disagree: {name}"
            )
        if starts is not None and (
            starts.numel() < local_row + 2
            or int((starts[local_row + 1] - starts[local_row]).item()) != query_tokens
        ):
            raise CompactionContractError(
                f"FA query and recurrent token counts disagree: {name}"
            )
        evidence[name] = {
            "seq_len": seq_len,
            "query_tokens": query_tokens,
            "context_tokens": context_len,
            "metadata_field": location,
            "batch_row": row,
        }
    return evidence


def _tensor_digest(layers: dict[str, tuple[str, tuple[torch.Tensor, ...]]]) -> str:
    digest = hashlib.sha256()
    for name, (family, states) in sorted(layers.items()):
        digest.update(json.dumps([name, family]).encode())
        for state in states:
            digest.update(json.dumps([list(state.shape), str(state.dtype)]).encode())
            digest.update(state.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class SnapshotStore:
    """Immutable CPU snapshots with optional per-device restore staging.

    Digests are computed at capture and explicit audit, never on metadata lookup.
    GPU staging is owned by this store, not aliased to any request's live cache.
    """

    def __init__(self):
        self._entries: dict[str, dict[str, Any]] = {}
        self._staged: dict[str, dict[str, dict]] = {}

    def capture(self, name: str, slots: list[StateSlot], boundary: dict) -> dict:
        if not isinstance(name, str) or not name or name in self._entries:
            raise CompactionContractError("Snapshot names must be new and nonempty")
        if not slots or len({slot.name for slot in slots}) != len(slots):
            raise CompactionContractError("Missing or duplicate state layers")
        layers = {
            slot.name: (
                slot.family,
                tuple(
                    state[slot.index].detach().to(device="cpu", copy=True).contiguous()
                    for state in slot.states
                ),
            )
            for slot in slots
        }
        for layer_name, (_, states) in layers.items():
            if any(not bool(torch.isfinite(state).all()) for state in states):
                raise CompactionContractError(f"Nonfinite native state in {layer_name}")
        self._entries[name] = {
            "layers": layers,
            "digest": _tensor_digest(layers),
            "boundary": copy.deepcopy(boundary),
            "bytes": sum(
                state.numel() * state.element_size()
                for _, states in layers.values()
                for state in states
            ),
        }
        return {
            **self.describe(name),
            "immutable_digest_verified": True,
            "digest_validation": "capture",
        }

    def describe(self, name: str) -> dict:
        if name not in self._entries:
            raise CompactionContractError(f"Unknown snapshot: {name}")
        entry = self._entries[name]
        staged = self._staged.get(name, {})
        return {
            "name": name,
            "sha256": entry["digest"],
            "immutable_digest_verified": False,
            "digest_validation": "cached_capture_digest",
            "all_native_values_finite_at_capture": True,
            "digest": entry["digest"],
            "bytes": entry["bytes"],
            "layer_count": len(entry["layers"]),
            "storage_device": "cpu",
            "staged_devices": sorted(staged),
            "gpu_staging_bytes": entry["bytes"]
            * sum(torch.device(device).type == "cuda" for device in staged),
            "boundary": copy.deepcopy(entry["boundary"]),
            "native_bytes": entry["bytes"],
            "layers": {
                key: {
                    "family": family,
                    "shapes": [list(x.shape) for x in states],
                    "dtypes": [str(x.dtype) for x in states],
                }
                for key, (family, states) in entry["layers"].items()
            },
        }

    def stage(self, name: str, device: torch.device | str) -> dict:
        self.describe(name)
        device = torch.device(device)
        if device.type != "cpu":
            key = str(device)
            staged = self._staged.setdefault(name, {})
            if key not in staged:
                staged[key] = {
                    layer: (
                        family,
                        tuple(t.to(device=device, copy=True) for t in states),
                    )
                    for layer, (family, states) in self._entries[name]["layers"].items()
                }
        return self.describe(name)

    def unstage(self, name: str) -> dict:
        self.describe(name)
        self._staged.pop(name, None)
        return self.describe(name)

    def audit(self, name: str) -> dict:
        entry = self._entries.get(name)
        if entry is None:
            raise CompactionContractError(f"Unknown snapshot: {name}")
        verified = ["cpu"]
        for device, layers in [
            ("cpu", entry["layers"]),
            *self._staged.get(name, {}).items(),
        ]:
            cpu_layers = {
                layer: (family, tuple(t.cpu() for t in states))
                for layer, (family, states) in layers.items()
            }
            if (
                any(
                    not bool(torch.isfinite(t).all())
                    for _, states in cpu_layers.values()
                    for t in states
                )
                or _tensor_digest(cpu_layers) != entry["digest"]
            ):
                raise CompactionContractError(f"Snapshot mutated: {name}/{device}")
            if device != "cpu":
                verified.append(device)
        return {
            **self.describe(name),
            "immutable_digest_verified": True,
            "digest_validation": "explicit_audit",
            "all_native_values_finite": True,
            "audited_devices": verified,
        }

    def validate_restore(
        self, name: str, slots: list[StateSlot], components: str = "all"
    ) -> None:
        self.describe(name)
        if components not in COMPONENTS:
            raise CompactionContractError("Unknown restore_components")
        layers = self._entries[name]["layers"]
        if set(layers) != {slot.name for slot in slots} or len(slots) != len(layers):
            raise CompactionContractError("Snapshot/target layer identities differ")
        for slot in slots:
            family, source = layers[slot.name]
            if family != slot.family or len(source) != len(slot.states):
                raise CompactionContractError(
                    f"Snapshot family/layout differs: {slot.name}"
                )
            _initial_state_flags(slot)
            for old, target in zip(source, slot.states):
                if old.shape != target[slot.index].shape or old.dtype != target.dtype:
                    raise CompactionContractError(
                        f"Snapshot shape/dtype differs: {slot.name}"
                    )

    def restore(
        self,
        name: str,
        slots: list[StateSlot],
        components: str = "all",
        *,
        zero_unselected: bool = False,
        audit_restore: bool = False,
    ) -> dict:
        self.validate_restore(name, slots, components)
        for device in {state.device for slot in slots for state in slot.states}:
            self.stage(name, device)
        selected = COMPONENTS[components]
        before_unselected = {
            (slot.name, component): target[slot.index].clone()
            for slot in slots
            for component, target in enumerate(slot.states)
            if audit_restore and component not in selected and not zero_unselected
        }
        with torch.no_grad():
            for slot in slots:
                for component, target in enumerate(slot.states):
                    if component in selected:
                        device = str(target.device)
                        layers = self._staged.get(name, {}).get(
                            device, self._entries[name]["layers"]
                        )
                        target[slot.index].copy_(layers[slot.name][1][component])
                    elif zero_unselected:
                        target[slot.index].zero_()
                for flag in _initial_state_flags(slot):
                    flag.fill_(True)
                if slot.family == "mamba2" and slot.prefill:
                    slot.metadata.prep_initial_states = True
        copy_audit = None
        if audit_restore:
            selected_equal = True
            untouched_equal = True
            zero_equal = True
            for slot in slots:
                for component, target in enumerate(slot.states):
                    current = target[slot.index]
                    if component in selected:
                        layers = self._staged.get(name, {}).get(
                            str(target.device), self._entries[name]["layers"]
                        )
                        selected_equal &= torch.equal(
                            current, layers[slot.name][1][component]
                        )
                    elif zero_unselected:
                        zero_equal &= bool((current == 0).all())
                    else:
                        untouched_equal &= torch.equal(
                            current, before_unselected[(slot.name, component)]
                        )
            if not (selected_equal and untouched_equal and zero_equal):
                raise CompactionContractError("Native state copy audit failed")
            copy_audit = {
                "selected_components_exact": selected_equal,
                "unselected_components_preserved": (
                    untouched_equal
                    if components != "all" and not zero_unselected
                    else None
                ),
                "fresh_unselected_zero": (
                    zero_equal if components != "all" and zero_unselected else None
                ),
            }
        return {
            **self.describe(name),
            "restore_components": components,
            "unselected_components_policy": (
                "none"
                if components == "all"
                else "zero_fresh_state"
                if zero_unselected
                else "keep_live_state"
            ),
            "restore_modes": {
                slot.name: "prefill_initial_state_enabled"
                if slot.prefill
                else "native_decode_state"
                for slot in slots
            },
            "snapshot_unchanged_after_restore": None,
            "snapshot_isolation": "private_source_copied_into_scheduler_owned_slots",
            "copy_audit": copy_audit,
        }

    def compare(self, name_a: str, name_b: str) -> dict:
        """Explicit CPU numerical diagnostic, outside the forward hot path."""
        self.audit(name_a)
        self.audit(name_b)
        layers_a = self._entries[name_a]["layers"]
        layers_b = self._entries[name_b]["layers"]
        if set(layers_a) != set(layers_b):
            raise CompactionContractError("Snapshots have different layer identities")
        per_layer = {}
        max_rel = 0.0
        for name in sorted(layers_a):
            fam_a, ta = layers_a[name]
            fam_b, tb = layers_b[name]
            if fam_a != fam_b or len(ta) != len(tb):
                raise CompactionContractError(f"Snapshot layouts differ: {name}")
            rels = []
            for x, y in zip(ta, tb):
                if x.shape != y.shape:
                    raise CompactionContractError(f"Snapshot shapes differ: {name}")
                xf, yf = x.float(), y.float()
                denom = float(torch.linalg.vector_norm(xf))
                rel = float(torch.linalg.vector_norm(xf - yf)) / (denom or 1.0)
                rels.append(rel)
                max_rel = max(max_rel, rel)
            per_layer[name] = rels
        return {
            "a": name_a,
            "b": name_b,
            "max_relative_frobenius": max_rel,
            "per_layer": per_layer,
        }

    def list(self) -> dict:
        snapshots = {name: self.describe(name) for name in self._entries}
        return {
            "snapshots": snapshots,
            "total_cpu_snapshot_bytes": sum(
                s["native_bytes"] for s in snapshots.values()
            ),
            "total_gpu_staging_bytes": sum(
                s["gpu_staging_bytes"] for s in snapshots.values()
            ),
        }

    def drop(self, name: str) -> None:
        self.describe(name)
        self._staged.pop(name, None)
        del self._entries[name]


class BoundaryOperation:
    """One request's capture and optional intervention at an exact token cursor."""

    def __init__(
        self,
        store: SnapshotStore,
        *,
        capture_name: str | None,
        restore_name: str | None,
        expected_prompt_tokens: int,
        start_cursor: int = 0,
        expected_request_id: str | None = None,
        operation_id: str | None = None,
        restore_at: int = 0,
        restore_components: str = "all",
        kv_restore_name: str | None = None,
        position_offset: int = 0,
        capture_kv: dict | None = None,
        audit_restore: bool = False,
        prefill_boundary: int | None = None,
        export_q: dict | None = None,
    ):
        if (
            isinstance(expected_prompt_tokens, bool)
            or not isinstance(expected_prompt_tokens, int)
            or expected_prompt_tokens < 1
        ):
            raise CompactionContractError(
                "expected_prompt_tokens must be a positive integer"
            )
        if (
            isinstance(start_cursor, bool)
            or not isinstance(start_cursor, int)
            or not 0 <= start_cursor < expected_prompt_tokens
        ):
            raise CompactionContractError(
                "start_cursor must be an integer below the cumulative prompt length"
            )
        if expected_request_id is not None and (
            not isinstance(expected_request_id, str) or not expected_request_id
        ):
            raise CompactionContractError("expected_request_id must be nonempty")
        if (
            isinstance(restore_at, bool)
            or not isinstance(restore_at, int)
            or not 0 <= restore_at < expected_prompt_tokens
        ):
            raise CompactionContractError("restore_at must precede the prompt end")
        if (
            restore_name is not None or kv_restore_name is not None
        ) and restore_at < start_cursor:
            raise CompactionContractError("Restore requires restore_at >= start_cursor")
        if restore_components not in COMPONENTS:
            raise CompactionContractError("Unknown restore_components")
        if (
            isinstance(position_offset, bool)
            or not isinstance(position_offset, int)
            or position_offset < 0
        ):
            raise CompactionContractError(
                "position_offset must be a nonnegative integer"
            )
        if operation_id is not None and (
            not isinstance(operation_id, str) or not operation_id
        ):
            raise CompactionContractError("operation_id must be nonempty")
        if capture_kv is not None and not isinstance(capture_kv, dict):
            raise CompactionContractError("capture_kv must be a descriptor")
        if not isinstance(audit_restore, bool):
            raise CompactionContractError("audit_restore must be a bool")
        if prefill_boundary is not None and (
            type(prefill_boundary) is not int
            or not 0 < prefill_boundary < expected_prompt_tokens
        ):
            raise CompactionContractError(
                "prefill_boundary must be an integer inside the prompt"
            )
        if export_q is not None:
            export_q = validate_export_q(export_q, start_cursor, expected_prompt_tokens)
        if capture_name is not None and (
            not isinstance(capture_name, str)
            or not capture_name
            or capture_name in store._entries
        ):
            raise CompactionContractError("Capture name must be new and nonempty")
        if restore_name is not None:
            store.describe(restore_name)
        self.store = store
        self.capture_name = capture_name
        self.restore_name = restore_name
        self.prompt_tokens = expected_prompt_tokens
        self.start_cursor = start_cursor
        self.expected_request_id = expected_request_id
        self.operation_id = operation_id
        self.restore_at = restore_at
        self.restore_components = restore_components
        self.kv_restore_name = kv_restore_name
        self.position_offset = position_offset
        self.capture_kv = copy.deepcopy(capture_kv)
        self.audit_restore = audit_restore
        self.prefill_boundary = prefill_boundary
        self.export_q = export_q
        self.q_export_chunks: list[dict] = []
        self.q_export_receipt: dict | None = None
        self.kv_restore_receipt: dict | None = None
        self.kv_capture_receipt: dict | None = None
        self.request_id: str | None = None
        self.first_computed_tokens: int | None = None
        self.processed_tokens = start_cursor
        self.prompt_complete = False
        self.closed = False
        self.request_finished = False
        self.fresh_decode_state_zeroed = False
        self.failed: str | None = None
        self.chunks: list[dict] = []
        self.restore_receipt: dict | None = None
        self.capture_receipt: dict | None = None
        self.first_fa_context: dict | None = None
        self._pending_tokens: int | None = None
        self._forward_details: dict = {}

    def restore_due(self, computed_tokens: int) -> bool:
        return computed_tokens == self.restore_at and self.restore_receipt is None

    def validate_before(
        self,
        request_id: str,
        prompt_tokens: int,
        computed_tokens: int,
        query_tokens: int,
        slots: list[StateSlot],
    ) -> None:
        """Perform every contract check before any request's state is modified."""
        if self.closed or self.failed:
            raise CompactionContractError("Operation is closed or failed")
        if self._pending_tokens is not None:
            raise CompactionContractError("A forward is already pending")
        if prompt_tokens != self.prompt_tokens or query_tokens < 1:
            raise CompactionContractError(
                "Prompt length or scheduled token count differs"
            )
        first_forward = self.request_id is None
        if first_forward and computed_tokens != self.start_cursor:
            raise CompactionContractError(
                "First forward cursor differs from start_cursor"
            )
        if (
            self.expected_request_id is not None
            and request_id != self.expected_request_id
        ):
            raise CompactionContractError("Unexpected request identity")
        if (
            not first_forward and request_id != self.request_id
        ) or computed_tokens != self.processed_tokens:
            raise CompactionContractError(
                "Unexpected request, preemption, or noncontiguous cursor"
            )
        end = computed_tokens + query_tokens
        if computed_tokens < prompt_tokens < end:
            raise CompactionContractError(
                "A forward crosses the declared prompt boundary"
            )
        if computed_tokens < self.restore_at < end:
            raise CompactionContractError("A forward crosses the restore_at boundary")
        if (
            self.prefill_boundary is not None
            and computed_tokens < self.prefill_boundary < end
        ):
            raise CompactionContractError("A forward crosses the prefill_boundary")
        if computed_tokens > self.restore_at and (
            (self.restore_name is not None and self.restore_receipt is None)
            or (self.kv_restore_name is not None and self.kv_restore_receipt is None)
        ):
            raise CompactionContractError("The intervention boundary was missed")
        if self.restore_name is not None and self.restore_due(computed_tokens):
            self.store.validate_restore(
                self.restore_name, slots, self.restore_components
            )

    def before(
        self,
        request_id: str,
        prompt_tokens: int,
        computed_tokens: int,
        query_tokens: int,
        slots: list[StateSlot],
        fa_context: dict | None = None,
        forward_details: dict | None = None,
    ) -> None:
        self.validate_before(
            request_id, prompt_tokens, computed_tokens, query_tokens, slots
        )
        if (
            self.request_id is None
            and computed_tokens == 0
            and (self.restore_name is None or self.restore_at != 0)
        ):
            # A budget-limited first token may use the decode kernel, which
            # consumes its slot unconditionally instead of zero-initializing it.
            for slot in slots:
                if not slot.prefill:
                    for state in slot.states:
                        state[slot.index].zero_()
                    self.fresh_decode_state_zeroed = True
        if self.restore_name is not None and self.restore_due(computed_tokens):
            self.restore_receipt = self.store.restore(
                self.restore_name,
                slots,
                self.restore_components,
                zero_unselected=computed_tokens == 0,
                audit_restore=self.audit_restore,
            )
            self.restore_receipt["at_cursor"] = computed_tokens
        if self.request_id is None:
            self.request_id = request_id
            self.first_computed_tokens = computed_tokens
            self.first_fa_context = copy.deepcopy(fa_context)
        self._pending_tokens = query_tokens
        self._forward_details = dict(forward_details or {})

    def after(self, query_tokens: int, slots: list[StateSlot]) -> None:
        if self.closed or self.failed or self._pending_tokens != query_tokens:
            raise CompactionContractError("No matching successful forward is pending")
        start = self.processed_tokens
        self.processed_tokens += query_tokens
        self._pending_tokens = None
        self.chunks.append(
            {
                "start": start,
                "end": self.processed_tokens,
                "query_tokens": query_tokens,
                **self._forward_details,
            }
        )
        if self.processed_tokens == self.prompt_tokens and not self.prompt_complete:
            if self.capture_name is not None:
                self.capture_receipt = self.store.capture(
                    self.capture_name,
                    slots,
                    {
                        "request_id": self.request_id,
                        "processed_prompt_tokens": self.prompt_tokens,
                        "start_cursor": self.start_cursor,
                        "position_policy": self.position_policy,
                        "position_offset": self.position_offset,
                        "restore_at": self.restore_at,
                        "prefill_boundary": self.prefill_boundary,
                        "snapshot_boundary": "prompt_end_before_first_sample",
                        "sampled_output_tokens_consumed": 0,
                    },
                )
            self.prompt_complete = True

    @property
    def position_policy(self) -> str:
        return "offset_after_intervention" if self.position_offset else "reset"

    def result(self) -> dict:
        if self.failed:
            # Report the failure, then close so the controller can arm again;
            # later calls say the operation is already closed.
            message = (
                f"Operation already closed after failure: {self.failed}"
                if self.closed
                else self.failed
            )
            self.closed = True
            raise CompactionContractError(message)
        if not self.prompt_complete or self._pending_tokens is not None:
            raise CompactionContractError("The prompt-end boundary was not reached")
        self.closed = True
        return copy.deepcopy(
            {
                "request_id": self.request_id,
                "operation_id": self.operation_id,
                "expected_request_id": self.expected_request_id,
                "expected_prompt_tokens": self.prompt_tokens,
                "start_cursor": self.start_cursor,
                "first_computed_tokens": self.first_computed_tokens,
                "new_tokens_processed": self.processed_tokens - self.start_cursor,
                "continuation": self.start_cursor > 0,
                "prompt_tokens_seen": min(self.processed_tokens, self.prompt_tokens),
                "forward_calls": len(self.chunks),
                "consumed_prompt_tokens": min(
                    self.processed_tokens, self.prompt_tokens
                ),
                "processed_cache_cursor": self.processed_tokens,
                "consumed_generation_tokens": max(
                    0, self.processed_tokens - self.prompt_tokens
                ),
                "forward_chunks": self.chunks,
                "position_policy": self.position_policy,
                "position_offset": self.position_offset,
                "restore_at": self.restore_at,
                "prefill_boundary": self.prefill_boundary,
                "restore_components": self.restore_components,
                "first_chunk_num_computed_tokens": self.first_computed_tokens,
                "first_fa_context": self.first_fa_context,
                "zero_fa_context_first_chunk": bool(self.first_fa_context)
                and all(
                    layer["context_tokens"] == 0
                    for layer in self.first_fa_context.values()
                ),
                "fa_kv_imported": self.kv_restore_receipt is not None,
                "kv_restore": self.kv_restore_receipt,
                "kv_capture": self.kv_capture_receipt,
                "restored_initial_state": self.restore_receipt is not None,
                "fresh_decode_state_zeroed": self.fresh_decode_state_zeroed,
                "restore": self.restore_receipt,
                "capture": self.capture_receipt,
                "restored_from": self.restore_name,
                "snapshot": self.capture_receipt,
                "export_q": self.export_q,
                "query_exported": self.q_export_receipt is not None,
                "q_export": self.q_export_receipt,
                "q_export_chunks": self.q_export_chunks,
                "returned_final_sample_is_not_consumed": True,
                "scope": "native_per_request_boundary_intervention",
                "total_cpu_snapshot_bytes": self.store.list()[
                    "total_cpu_snapshot_bytes"
                ],
                "total_gpu_staging_bytes": self.store.list()["total_gpu_staging_bytes"],
            }
        )


@dataclass(frozen=True)
class ForwardBoundary:
    """A real request row and its current scheduler-owned state slots."""

    operation: BoundaryOperation
    query_tokens: int
    slots: list[StateSlot]
    request_id: str
    row: int
    computed_tokens: int
    token_start: int
    fa_context: dict | None
    details: dict


def _fa_metadata_on_cpu(metadata: dict, layer_names: set[str]) -> dict:
    """Copy only FA length metadata, in one transfer per device/dtype."""
    result = {}
    fields = []
    for name in layer_names:
        current = metadata.get(name)
        copied = copy.copy(current)
        result[name] = copied
        if current is None:
            continue
        nodes = [(current, copied)]
        for part in ("prefill", "decode"):
            nested = getattr(current, part, None)
            if nested is not None:
                nested_copy = copy.copy(nested)
                setattr(copied, part, nested_copy)
                nodes.append((nested, nested_copy))
        for original, target in nodes:
            for key in ("seq_lens", "query_start_loc", "cum_seq_lens_q"):
                tensor = getattr(original, key, None)
                if isinstance(tensor, torch.Tensor):
                    fields.append((target, key, tensor))
    groups: dict[tuple, list] = {}
    for target, key, tensor in fields:
        if tensor.device.type == "cpu":
            setattr(target, key, tensor)
        else:
            groups.setdefault((tensor.device, tensor.dtype), []).append(
                (target, key, tensor)
            )
    for group in groups.values():
        copied = torch.cat([tensor.reshape(-1) for _, _, tensor in group]).cpu()
        cursor = 0
        for target, key, tensor in group:
            end = cursor + tensor.numel()
            setattr(target, key, copied[cursor:end].view(tensor.shape))
            cursor = end
    return result


class NativeCompactionController:
    """Default-off, per-request state/KV operations outside captured graphs."""

    # Class defaults: fixtures may bypass __init__.
    fresh_decode_rows_zeroed = 0
    q_store: Any = None
    score_store: Any = None
    selection_store: Any = None
    fa_layer_groups: dict[str, int] = {}
    _export_plan: Any = None
    _consumed: Any = None  # request_id -> cursor after its last successful forward
    _pending_consumed: Any = None
    _provenance_cache: Any = None

    def __init__(self, runner: GPUModelRunner):
        validate_configuration(runner.vllm_config)
        if type(runner).__module__ != "vllm.v1.worker.gpu_model_runner":
            raise CompactionContractError("Only V1 GPUModelRunner is supported")

        from vllm.model_executor.layers.mamba.abstract import MambaBase
        from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
        from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadata
        from vllm.v1.kv_cache_interface import MambaSpec
        from vllm.v1.worker.compaction_kv import NativeKVSnapshotStore

        self.runner = runner
        self.layers = {}
        for name, layer in runner.compilation_config.static_forward_context.items():
            if isinstance(layer, MambaBase):
                kind = layer.mamba_type.name
                family = {"GDN_ATTN": "gdn", "MAMBA2": "mamba2"}.get(kind)
                if family is None or len(layer.kv_cache) != 2:
                    raise CompactionContractError(
                        f"Unsupported native cache: {name}/{kind}"
                    )
                self.layers[name] = (family, layer)
        self.layer_groups = {
            name: group_id
            for group_id, group in enumerate(runner.kv_cache_config.kv_cache_groups)
            if isinstance(group.kv_cache_spec, MambaSpec)
            for name in group.layer_names
        }
        self.fa_layers = {
            name
            for group in runner.kv_cache_config.kv_cache_groups
            if not isinstance(group.kv_cache_spec, MambaSpec)
            for name in group.layer_names
        }
        self.fa_layer_groups = {
            name: group_id
            for group_id, group in enumerate(runner.kv_cache_config.kv_cache_groups)
            if not isinstance(group.kv_cache_spec, MambaSpec)
            for name in group.layer_names
        }
        if not self.layers or not self.fa_layers:
            raise CompactionContractError(
                "Expected a hybrid with recurrent and FA layers"
            )
        self.store = SnapshotStore()
        self.kv_store = NativeKVSnapshotStore(runner)
        compaction_q = _q_module()
        self.q_store = compaction_q.QExportStore(
            list(self.kv_store.layers), self._export_geometry()
        )
        self.score_store = compaction_q.ScoreStore()
        self.selection_store = compaction_q.SelectionStore()
        self._export_plan = None
        self._consumed = {}
        self._pending_consumed = None
        self.operation: BoundaryOperation | None = None
        self.operations: dict[str, BoundaryOperation] = {}
        self._descriptors: dict[str, dict] = {}
        self._retired_bindings: dict[str, str] = {}
        self._seen_operation_ids: set[str] = set()
        self._position_rules: dict[str, tuple[int, int]] = {}
        self.fresh_decode_rows_zeroed = 0
        self.metadata_types = {
            "gdn": GDNAttentionMetadata,
            "mamba2": Mamba2AttentionMetadata,
        }

    def info(self) -> dict:
        # Paged per-key bias (AM bias variant): only on a bias-capable backend.
        kv_store = getattr(self, "kv_store", None)
        kv_bias = kv_store is not None and kv_store.bias_supported()
        return {
            "backend": "vllm_native",
            "implementation": "vllm.v1.worker.compaction.NativeCompactionController",
            "descriptor_key": DESCRIPTOR_KEY,
            "recurrent_layers": len(self.layers),
            "fa_layers": len(self.fa_layers),
            "families": sorted({family for family, _ in self.layers.values()}),
            "native_state_bytes_per_request": sum(
                state[0].numel() * state.element_size()
                for _, layer in self.layers.values()
                for state in layer.kv_cache
            ),
            "position_policy": "per_request_offset_after_intervention",
            "snapshot_storage": "cpu_with_optional_gpu_staging",
            "kv_ownership": "vllm_scheduler",
            "prefix_caching": False,
            "batch_operations": True,
            "fresh_decode_rows_zeroed": self.fresh_decode_rows_zeroed,
            "fresh_decode_zeroing_scope": "every_request_row",
            "query_export": self.q_store is not None,
            "query_export_hook": "attention.compaction_q_export_prefill_only",
            "compute_ops": [
                "kv_capture",
                "kv_subset",
                "kv_score",
                "kv_select",
                "kv_fit_am",
                *(["kv_bias"] if kv_bias else []),
            ],
            "store_ops": [
                "kv_subset",
                "kv_selection_drop",
                "kv_describe",
                "q_describe",
                "kv_drop",
                "q_drop",
                "score_drop",
                *(["bias", "bias_mask"] if kv_bias else []),
            ],
            "kv_bias": kv_bias,
            "kv_bias_bytes": kv_store.bias_bytes() if kv_bias else 0,
            # No environment switch: the buffer follows the resolved backend's
            # supports_kv_bias() and an unquantized fp16/bf16/fp32 KV cache.
            "kv_bias_policy": "resolved_backend_and_kv_dtype_only_no_env_flag",
            # Resolved attention backend of the FA layers (what kv_bias follows).
            "fa_backends": sorted(
                {
                    layer.get_attn_backend().get_name()
                    for layer in (kv_store.layers.values() if kv_store else ())
                }
            ),
            "consumed_cursor_tracking": "controller_tracked_expected_cursor_required",
            "scope": "native_per_request_boundary_intervention",
        }

    def _zero_fresh_decode_rows(
        self,
        metadata: Any,
        scheduled_tokens: dict[str, int] | None,
        bound: list[tuple[int, str, BoundaryOperation | None]],
    ) -> list[dict]:
        """Zero native slots of fresh rows whose 1-token first slice hits the decode kernel.

        GDN classifies every single-token query as decode regardless of its
        cursor, the decode kernel consumes the slot unconditionally, and the
        block pool never zeroes retired recurrent pages. A budget-limited first
        slice of an uninstrumented request (a summary generation, a validation
        control) would therefore inherit a retired request's state. This runs
        for every request row, before any operation restores into its slot.
        """
        if not isinstance(metadata, dict):
            return []
        batch = self.runner.input_batch
        zeroed = []
        for row, request_id, _ in bound:
            if int(batch.num_computed_tokens_cpu[row]) != 0:
                continue
            if scheduled_tokens is not None:
                count = int(scheduled_tokens.get(request_id, 0))
            elif batch.num_reqs == 1:
                current = metadata[next(iter(self.layers))]
                count = int(current.num_prefill_tokens + current.num_decode_tokens)
            else:
                count = 0
            if count != 1:
                continue
            for slot in self._slots(row, metadata):
                if slot.prefill:
                    continue
                for state in slot.states:
                    state[slot.index].zero_()
                zeroed.append(
                    {"row": row, "request_id": request_id, "layer": slot.name}
                )
        if zeroed:
            self.fresh_decode_rows_zeroed += len({z["request_id"] for z in zeroed})
        return zeroed

    def _all_operations(self) -> list[BoundaryOperation]:
        return [
            *self.operations.values(),
            *([self.operation] if self.operation else []),
        ]

    def _check_capture_names(self, operation: BoundaryOperation) -> None:
        for other in self._all_operations():
            if other is operation or other.closed:
                continue
            if operation.capture_name and operation.capture_name == other.capture_name:
                raise CompactionContractError(
                    "Capture name is reserved by another operation"
                )
            if (
                operation.capture_kv
                and other.capture_kv
                and (operation.capture_kv.get("name") == other.capture_kv.get("name"))
            ):
                raise CompactionContractError(
                    "KV capture name is reserved by another operation"
                )

    def arm(
        self,
        capture_name: str | None = None,
        restore_name: str | None = None,
        expected_prompt_tokens: int | None = None,
        start_cursor: int = 0,
        expected_request_id: str | None = None,
        restore_at: int = 0,
        restore_components: str = "all",
        kv_restore_name: str | None = None,
        position_offset: int = 0,
        capture_kv: dict | None = None,
        audit_restore: bool = False,
        prefill_boundary: int | None = None,
        export_q: dict | None = None,
    ) -> dict:
        if self.operation is not None and not self.operation.closed:
            raise CompactionContractError(
                "Call cc_result before arming another operation"
            )
        if self.runner.execute_model_state is not None:
            raise CompactionContractError("Cannot arm between forward and sampling")
        operation = BoundaryOperation(
            self.store,
            capture_name=capture_name,
            restore_name=restore_name,
            expected_prompt_tokens=expected_prompt_tokens,
            start_cursor=start_cursor,
            expected_request_id=expected_request_id,
            restore_at=restore_at,
            restore_components=restore_components,
            kv_restore_name=kv_restore_name,
            position_offset=position_offset,
            capture_kv=capture_kv,
            audit_restore=audit_restore,
            prefill_boundary=prefill_boundary,
            export_q=export_q,
        )
        self._check_capture_names(operation)
        self._open_export(operation, expected_request_id, None)
        self.operation = operation
        return {
            "armed": True,
            "capture_name": capture_name,
            "restore_name": restore_name,
            "expected_prompt_tokens": expected_prompt_tokens,
            "start_cursor": start_cursor,
            "expected_request_id": expected_request_id,
            "restore_at": restore_at,
            "restore_components": restore_components,
            "kv_restore_name": kv_restore_name,
            "capture_kv": capture_kv,
            "position_offset": position_offset,
            "audit_restore": audit_restore,
            "prefill_boundary": prefill_boundary,
            "export_q": operation.export_q,
        }

    def _q_store(self):
        if self.q_store is None:
            raise CompactionContractError("Query export store is unavailable")
        return self.q_store

    def _export_geometry(self) -> dict[str, dict]:
        """Per-FA-layer query geometry so export buffers are allocated at arm time."""
        geometry = {}
        for name, layer in self.kv_store.layers.items():
            geometry[name] = {
                "num_heads": getattr(layer, "num_heads", None),
                "num_kv_heads": getattr(layer, "num_kv_heads", None),
                "head_size": getattr(layer, "head_size", None),
                "dtype": getattr(layer, "dtype", None),
                "pinned": bool(layer.kv_cache.is_cuda),
            }
        return geometry

    def _consumed_map(self) -> dict[str, int]:
        if self._consumed is None:
            self._consumed = {}
        return self._consumed

    def _open_export(
        self,
        operation: BoundaryOperation,
        request_id: str | None,
        operation_id: str | None,
    ) -> None:
        """Reserve the export name when the operation is created."""
        if operation.export_q is None:
            return
        self._q_store().open(
            operation.export_q["name"],
            token_range=operation.export_q["token_range"],
            request_id=request_id,
            operation_id=operation_id,
        )

    def requires_query_export(self, num_reqs: int, scheduled_tokens: Any) -> bool:
        """Does this forward carry rows of an active query-export range?

        Inspection only, for cudagraph dispatch: no operation is bound, no
        export buffer is allocated and no state changes. A one-token slice
        inside an export range would otherwise be classified as uniform decode
        and dispatched FULL, which the forward-time guard then refuses.
        Malformed descriptors are ignored here; normal validation rejects them.
        """
        batch = self.runner.input_batch
        for row, request_id in enumerate(batch.req_ids[:num_reqs]):
            request = self.runner.requests[request_id]
            params = getattr(request, "sampling_params", None)
            descriptor = (getattr(params, "extra_args", None) or {}).get(DESCRIPTOR_KEY)
            export = None
            if descriptor is None:
                operation = self.operation
                if (
                    operation is not None
                    and not operation.closed
                    and operation.expected_request_id in (None, request_id)
                ):
                    export = operation.export_q
            elif isinstance(descriptor, dict):
                operation_id = descriptor.get("operation_id")
                if not isinstance(operation_id, str):
                    continue
                operation = self.operations.get(operation_id)
                if operation is not None:
                    if not operation.closed:
                        export = operation.export_q
                elif operation_id not in self._seen_operation_ids:
                    export = descriptor.get("export_q")
            if not isinstance(export, dict):
                continue
            span = export.get("token_range")
            if (
                not isinstance(span, (list, tuple))
                or len(span) != 2
                or any(type(value) is not int for value in span)
            ):
                continue
            computed = int(batch.num_computed_tokens_cpu[row])
            count = int(scheduled_tokens[row])
            if computed < span[1] and computed + count > span[0]:
                return True
        return False

    def _bound_operation(
        self, request_id: str, request: Any
    ) -> BoundaryOperation | None:
        params = getattr(request, "sampling_params", None)
        descriptor = (getattr(params, "extra_args", None) or {}).get(DESCRIPTOR_KEY)
        legacy = self.operation
        legacy_active = legacy is not None and not legacy.closed
        if descriptor is None:
            if legacy_active and (
                legacy.expected_request_id is None
                or legacy.expected_request_id == request_id
            ):
                if (
                    legacy.expected_request_id is None
                    and self.runner.input_batch.num_reqs != 1
                ):
                    raise CompactionContractError(
                        "Unbound cc_arm requires a single request"
                    )
                return legacy
            return None
        if not isinstance(descriptor, dict):
            raise CompactionContractError("native_compaction_v1 must be a descriptor")
        operation_id = descriptor.get("operation_id")
        if not isinstance(operation_id, str) or not operation_id:
            raise CompactionContractError("Descriptor requires a unique operation_id")
        if legacy_active and (
            legacy.expected_request_id is None
            or legacy.expected_request_id == request_id
        ):
            raise CompactionContractError("Request has both a descriptor and cc_arm")
        if operation_id in self.operations:
            operation = self.operations[operation_id]
            if (
                operation.expected_request_id != request_id
                or descriptor != self._descriptors[operation_id]
            ):
                raise CompactionContractError(
                    "operation_id or descriptor was reused/changed"
                )
            return None if operation.closed else operation
        if operation_id in self._seen_operation_ids:
            if self._retired_bindings.get(
                request_id
            ) == operation_id and descriptor == self._descriptors.get(operation_id):
                return None
            raise CompactionContractError("operation_id was already delivered/reused")
        fields = {
            "operation_id",
            "capture_name",
            "restore_name",
            "expected_prompt_tokens",
            "start_cursor",
            "restore_at",
            "restore_components",
            "kv_restore_name",
            "position_offset",
            "capture_kv",
            "audit_restore",
            "prefill_boundary",
            "export_q",
        }
        if set(descriptor) - fields:
            raise CompactionContractError("Unknown native_compaction_v1 fields")
        kwargs = {"capture_name": None, "restore_name": None, **descriptor}
        if "expected_prompt_tokens" not in kwargs:
            raise CompactionContractError("Descriptor requires expected_prompt_tokens")
        operation = BoundaryOperation(
            self.store, expected_request_id=request_id, **kwargs
        )
        self._check_capture_names(operation)
        self._open_export(operation, request_id, operation_id)
        self.operations[operation_id] = operation
        self._descriptors[operation_id] = copy.deepcopy(descriptor)
        self._seen_operation_ids.add(operation_id)
        return operation

    def result(self) -> dict:
        if self.operation is None:
            raise CompactionContractError("No operation has been armed")
        return {**self.info(), **self.operation.result()}

    def results(self, operation_ids: list[str]) -> dict:
        """Close and return completed operations; failed ones are retired once.

        A failed operation is reported in the raised error and removed, leaving
        a tombstone so later forwards of a still-live request run
        uninstrumented instead of wedging the controller until restart.
        """
        if len(set(operation_ids)) != len(operation_ids):
            raise CompactionContractError("Duplicate operation_ids")
        failed = {}
        for operation_id in operation_ids:
            operation = self.operations.get(operation_id)
            if operation is None:
                raise CompactionContractError(f"Unknown operation: {operation_id}")
            if operation.failed:
                failed[operation_id] = operation.failed
            elif not operation.prompt_complete or operation._pending_tokens is not None:
                raise CompactionContractError(
                    f"Operation has not reached its prompt end: {operation_id}"
                )
        if failed:
            for operation_id in failed:
                operation = self.operations[operation_id]
                operation.closed = True
                self._retire(operation_id, operation)
            raise CompactionContractError(
                "Failed operations retired: "
                + "; ".join(f"{key}: {text}" for key, text in failed.items())
            )
        receipts = {
            key: {**self.info(), **self.operations[key].result()}
            for key in operation_ids
        }
        for key in operation_ids:
            self._retire(key, self.operations[key])
        return receipts

    def disarm(self, operation_id: str | None = None) -> dict:
        """Close an armed operation that never bound to a request.

        Admission can fail after ``cc_arm``; the operation then never sees a
        forward, ``result()`` keeps raising "boundary was not reached" and the
        next ``cc_arm`` is refused although the engine is idle. Disarming closes
        such an operation (and a failed one) and drops the export reservation it
        opened, which holds no rows. A bound, live operation is refused: it must
        finish through ``cc_result``/``cc_results``.
        """
        if operation_id is None:
            operation = self.operation
            if operation is None or operation.closed:
                raise CompactionContractError("No armed operation to disarm")
        else:
            operation = self.operations.get(operation_id)
            if operation is None:
                raise CompactionContractError(f"Unknown operation: {operation_id}")
        bound = operation.request_id is not None
        if bound and not operation.failed and not operation.closed:
            raise CompactionContractError(
                "Cannot disarm a bound live operation; finish it through "
                "cc_result/cc_results"
            )
        operation.closed = True
        export_dropped = False
        if operation.export_q is not None and self.q_store is not None:
            name = operation.export_q["name"]
            entries = self.q_store.list()["exports"]
            if name in entries and entries[name]["rows_exported"] == 0:
                self.q_store.drop(name)
                export_dropped = True
        if operation_id is not None:
            self._retire(operation_id, operation)
        return {
            "disarmed": True,
            "operation_id": operation_id,
            "legacy": operation_id is None,
            "never_bound": not bound,
            "expected_request_id": operation.expected_request_id,
            "failed": operation.failed,
            "export_q": operation.export_q,
            "export_dropped": export_dropped,
            "forward_calls": len(operation.chunks),
        }

    def _retire(self, key: str, operation: BoundaryOperation) -> None:
        self.operations.pop(key, None)
        request_id = operation.request_id or operation.expected_request_id
        if operation.request_finished or request_id is None:
            self._descriptors.pop(key, None)
        else:
            self._retired_bindings[request_id] = key

    def snapshots(self) -> dict:
        return self.store.list()["snapshots"]

    def compare(self, name_a: str, name_b: str) -> dict:
        return self.store.compare(name_a, name_b)

    def audit(self, names: list[str] | None = None) -> dict:
        return {
            name: self.store.audit(name)
            for name in (names if names is not None else self.store._entries)
        }

    def stage(self, names: list[str]) -> dict:
        return {name: self.store.stage(name, self.runner.device) for name in names}

    def unstage(self, names: list[str]) -> dict:
        return {name: self.store.unstage(name) for name in names}

    def drop(self, name: str) -> dict:
        if any(
            not op.closed and name in (op.restore_name, op.capture_name)
            for op in self._all_operations()
        ):
            raise CompactionContractError("Cannot drop an active operation's snapshot")
        self.store.drop(name)
        return {"dropped": name, **self.store.list()}

    def kv_drop(self, name: str) -> dict:
        if any(
            not op.closed
            and not op.failed
            and (
                name == op.kv_restore_name
                or (op.capture_kv and name == op.capture_kv.get("name"))
            )
            for op in self._all_operations()
        ):
            raise CompactionContractError(
                "Cannot drop an active operation's KV snapshot"
            )
        self.kv_store.drop(name)
        return {"dropped": name, **self.kv_store.list()}

    def memory_stats(self, reset_peak: bool = False) -> dict:
        device = self.runner.device
        stats = {
            "native_snapshots": self.store.list(),
            "kv_snapshots": self.kv_store.list(),
        }
        if torch.device(device).type == "cuda":
            if reset_peak:
                torch.cuda.reset_peak_memory_stats(device)
            stats.update(
                cuda_allocated_bytes=torch.cuda.memory_allocated(device),
                cuda_reserved_bytes=torch.cuda.memory_reserved(device),
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            )
        return stats

    def finish_requests(self, request_ids: set[str]) -> None:
        for operation in self._all_operations():
            if operation.request_id in request_ids:
                operation.request_finished = True
                if not operation.closed and not operation.prompt_complete:
                    operation.failed = (
                        "Request finished/aborted before its prompt-end boundary"
                    )
        for request_id in request_ids:
            self._position_rules.pop(request_id, None)
            self._consumed_map().pop(request_id, None)
            operation_id = self._retired_bindings.pop(request_id, None)
            if operation_id is not None:
                self._descriptors.pop(operation_id, None)

    def _slots(self, row: int, metadata: dict) -> list[StateSlot]:
        slots = []
        for name, (family, layer) in self.layers.items():
            current = metadata.get(name)
            if not isinstance(current, self.metadata_types[family]):
                raise CompactionContractError(f"Unexpected metadata type for {name}")
            index = None
            group = self.layer_groups.get(name)
            if group is not None:
                index = int(
                    self.runner.input_batch.block_table[group].get_cpu_tensor()[row, 0]
                )
            slots.append(
                resolve_slot(
                    name, family, layer.kv_cache, current, row=row, index=index
                )
            )
        return slots

    def before_forward(
        self,
        metadata: Any,
        positions: torch.Tensor | None,
        scheduled_tokens: dict[str, int] | None = None,
        graph_mode: str = "NONE",
        padded_batch_size: int | None = None,
    ) -> list[ForwardBoundary] | None:
        boundaries = []
        try:
            runner, batch = self.runner, self.runner.input_batch
            bound = [
                (
                    row,
                    request_id,
                    self._bound_operation(request_id, runner.requests[request_id]),
                )
                for row, request_id in enumerate(batch.req_ids[: batch.num_reqs])
            ]
            # Runner-level guard for every row, instrumented or not.
            self._zero_fresh_decode_rows(metadata, scheduled_tokens, bound)
            # Cursor every row will have consumed once this forward succeeds; the
            # worker's own copies are not advanced after the forward (see
            # _resident), so the controller tracks them for compute ops.
            self._pending_consumed = None
            if scheduled_tokens is not None:
                self._pending_consumed = {
                    request_id: int(batch.num_computed_tokens_cpu[row])
                    + int(scheduled_tokens[request_id])
                    for row, request_id, _ in bound
                }
            elif isinstance(metadata, dict) and batch.num_reqs == 1 and self.layers:
                current = metadata.get(next(iter(self.layers)))
                if current is not None:
                    self._pending_consumed = {
                        batch.req_ids[0]: int(batch.num_computed_tokens_cpu[0])
                        + int(current.num_prefill_tokens + current.num_decode_tokens)
                    }
            if not any(
                op is not None or req in self._position_rules for _, req, op in bound
            ):
                return None
            if not isinstance(metadata, dict) or positions is None:
                raise CompactionContractError(
                    "Expected per-layer metadata and forward positions"
                )
            if scheduled_tokens is None:
                if batch.num_reqs != 1:
                    raise CompactionContractError(
                        "Batched forwards require scheduler token counts"
                    )
                current = metadata[next(iter(self.layers))]
                scheduled_tokens = {
                    batch.req_ids[0]: int(
                        current.num_prefill_tokens + current.num_decode_tokens
                    )
                }
            total = sum(scheduled_tokens[request_id] for _, request_id, _ in bound)
            if positions.ndim not in (1, 2) or positions.shape[-1] < total:
                raise CompactionContractError(
                    "Position buffer is shorter than the real token batch"
                )
            fa_cpu = None
            token_start = 0
            position_updates = []
            export_items = []
            for row, request_id, operation in bound:
                request = runner.requests[request_id]
                count = int(scheduled_tokens[request_id])
                computed = int(batch.num_computed_tokens_cpu[row])
                if count < 1 or int(request.num_computed_tokens) != computed:
                    raise CompactionContractError(
                        "Worker request and scheduler cursors disagree"
                    )
                if operation is not None and (
                    request.mm_features
                    or request.prompt_embeds is not None
                    or request.lora_request
                ):
                    raise CompactionContractError(
                        "Only unadapted text-token requests are supported"
                    )
                rule = self._position_rules.get(request_id, (0, 0))
                if operation is not None and operation.position_offset:
                    rule = (operation.restore_at, operation.position_offset)
                if computed < rule[0] < computed + count:
                    raise CompactionContractError(
                        "A forward crosses the position-offset boundary"
                    )
                offset = rule[1] if computed >= rule[0] else 0
                if positions.device.type == "cpu":
                    expected = torch.arange(
                        computed, computed + count, dtype=positions.dtype
                    )
                    if not bool(
                        (
                            positions[..., token_start : token_start + count]
                            == expected
                        ).all()
                    ):
                        raise CompactionContractError(
                            "Forward positions differ from scheduler cursors"
                        )
                if offset:
                    position_updates.append((token_start, count, offset))
                if operation is not None:
                    slots = self._slots(row, metadata)
                    operation.validate_before(
                        request_id, request.num_prompt_tokens, computed, count, slots
                    )
                    export_span = None
                    if operation.export_q is not None:
                        start, end = operation.export_q["token_range"]
                        if computed < end and computed + count > start:
                            export_span = (
                                max(start, computed),
                                min(end, computed + count),
                            )
                    needs_fa = (
                        operation.request_id is None
                        or computed == operation.restore_at
                        or computed + count == operation.prompt_tokens
                        or export_span is not None
                    )
                    context = None
                    if needs_fa:
                        if fa_cpu is None:
                            fa_cpu = _fa_metadata_on_cpu(metadata, self.fa_layers)
                        context = verify_fa_context(
                            fa_cpu,
                            self.fa_layers,
                            row=row,
                            query_tokens=count,
                            computed_tokens=computed,
                        )
                    if export_span is not None:
                        # Ground the row slice in every FA layer's query_start_loc.
                        for name in sorted(self.fa_layers):
                            starts = getattr(fa_cpu.get(name), "query_start_loc", None)
                            if (
                                not isinstance(starts, torch.Tensor)
                                or starts.numel() < row + 2
                                or int(starts[row].item()) != token_start
                            ):
                                raise CompactionContractError(
                                    "Query export requires FA query_start_loc matching "
                                    f"the batch token start: {name}"
                                )
                        self._q_store().bind(operation.export_q["name"], request_id)
                        cursor_start, cursor_end = export_span
                        export_items.append(
                            _q_module().ExportItem(
                                name=operation.export_q["name"],
                                operation=operation,
                                row_start=token_start + cursor_start - computed,
                                row_end=token_start + cursor_end - computed,
                                cursor_start=cursor_start,
                                cursor_end=cursor_end,
                                positions=torch.arange(
                                    cursor_start + offset,
                                    cursor_end + offset,
                                    dtype=torch.long,
                                ),
                                receipt={
                                    "request_id": request_id,
                                    "operation_id": operation.operation_id,
                                    "batch_row": row,
                                    "graph_mode": str(graph_mode),
                                    "position_offset": offset,
                                },
                            )
                        )
                    if (
                        operation.kv_restore_name is not None
                        and computed == operation.restore_at
                    ):
                        self.kv_store.validate_restore(
                            operation.kv_restore_name,
                            request_id,
                            row,
                            metadata,
                            computed,
                        )
                        source = self.kv_store.describe(operation.kv_restore_name)
                        # Chained imports: the imported rows carry their original
                        # RoPE positions, so the new tokens continue from the
                        # source's absolute next position (offset sources included).
                        if (
                            source["source_absolute_next_position"] - computed
                            != operation.position_offset
                        ):
                            raise CompactionContractError(
                                "KV import requires original next-query positions: "
                                "position_offset must equal "
                                "source_absolute_next_position - computed "
                                f"({source['source_absolute_next_position']} - "
                                f"{computed})"
                            )
                    if (
                        operation.capture_kv is not None
                        and computed + count == operation.prompt_tokens
                    ):
                        self.kv_store.validate_capture(
                            operation.capture_kv,
                            request_id,
                            row,
                            metadata,
                            computed + count,
                        )
                    details = {
                        "batch_row": row,
                        "batch_size": batch.num_reqs,
                        "padded_batch_size": padded_batch_size or batch.num_reqs,
                        "position_buffer_tokens": positions.shape[-1],
                        "graph_mode": str(graph_mode),
                        "position_offset": offset,
                        "first_position": computed + offset,
                        "last_position": computed + count - 1 + offset,
                        "position_evidence": "native_runner_cursor_and_inplace_offset",
                    }
                    boundaries.append(
                        ForwardBoundary(
                            operation,
                            count,
                            slots,
                            request_id,
                            row,
                            computed,
                            token_start,
                            context,
                            details,
                        )
                    )
                token_start += count
            if export_items and str(graph_mode) not in EXPORT_GRAPH_MODES:
                # Refused before any co-scheduled stage/restore/state write below.
                raise CompactionContractError(
                    "Query export requires an eager or piecewise forward; a FULL "
                    f"CUDA-graph forward ({graph_mode}) cannot copy rows to the host"
                )
            # Every row/layout/boundary has passed before any cache or position write.
            allocated = [
                (slot.name, slot.index)
                for boundary in boundaries
                for slot in boundary.slots
            ]
            if len(allocated) != len(set(allocated)):
                raise CompactionContractError("Requests alias a native state slot")
            for boundary in boundaries:
                op = boundary.operation
                if op.restore_name is not None and op.restore_due(
                    boundary.computed_tokens
                ):
                    for device in {
                        state.device for slot in boundary.slots for state in slot.states
                    }:
                        self.store.stage(op.restore_name, device)
                if (
                    op.kv_restore_name is not None
                    and boundary.computed_tokens == op.restore_at
                ):
                    self.kv_store.stage(op.kv_restore_name, runner.device)
            for boundary in boundaries:
                op = boundary.operation
                if (
                    op.kv_restore_name is not None
                    and boundary.computed_tokens == op.restore_at
                ):
                    op.kv_restore_receipt = self.kv_store.restore(
                        op.kv_restore_name,
                        boundary.request_id,
                        boundary.row,
                        metadata,
                        boundary.computed_tokens,
                    )
                    op.kv_restore_receipt["query_position_offset"] = op.position_offset
                    op.kv_restore_receipt["recurrence_policy"] = (
                        op.restore_components
                        if op.restore_name is not None
                        else "keep_scratch_prefix_state"
                    )
                op.before(
                    boundary.request_id,
                    runner.requests[boundary.request_id].num_prompt_tokens,
                    boundary.computed_tokens,
                    boundary.query_tokens,
                    boundary.slots,
                    boundary.fa_context,
                    boundary.details,
                )
                if op.position_offset:
                    self._position_rules[boundary.request_id] = (
                        op.restore_at,
                        op.position_offset,
                    )
            for start, count, offset in position_updates:
                positions[..., start : start + count].add_(offset)
            self._forward_metadata = metadata
            if export_items:
                self._install_export(
                    _q_module().QueryExportPlan(
                        self._q_store(), self.fa_layers, export_items, total
                    )
                )
            return boundaries
        except Exception as error:
            self.fail_forward(error)
            raise

    def _install_export(self, plan: Any) -> None:
        """Point every FA layer's ``compaction_q_export`` at this forward's plan."""
        layers = self.kv_store.layers
        missing = [name for name in self.fa_layers if name not in layers]
        if missing or self._export_plan is not None:
            raise CompactionContractError(
                f"Cannot install query export (missing layers {missing}, "
                f"pending plan {self._export_plan is not None})"
            )
        for name in self.fa_layers:
            layers[name].compaction_q_export = plan.hook
        self._export_plan = plan

    def _uninstall_export(self) -> None:
        plan = self._export_plan
        if plan is None:
            return
        for name in self.fa_layers:
            layer = self.kv_store.layers.get(name)
            if layer is not None:
                layer.compaction_q_export = None
        self._export_plan = None

    def after_forward(self, boundaries: list[ForwardBoundary] | None) -> None:
        # The forward succeeded: every row consumed its scheduled tokens.
        if self._pending_consumed is not None:
            self._consumed_map().update(self._pending_consumed)
            self._pending_consumed = None
        plan = self._export_plan
        if plan is not None:
            try:
                receipts = plan.finish()
            except Exception as error:
                self.fail_forward(error)
                raise
            self._uninstall_export()
            for item, receipt in zip(plan.items, receipts):
                item.operation.q_export_chunks.append(receipt["chunks"][-1])
                if receipt["complete"]:
                    item.operation.q_export_receipt = receipt
        if boundaries is None:
            return
        try:
            for boundary in boundaries:
                operation = boundary.operation
                capture_kv = (
                    operation.capture_kv is not None
                    and not operation.prompt_complete
                    and (
                        boundary.computed_tokens + boundary.query_tokens
                        == operation.prompt_tokens
                    )
                )
                operation.after(boundary.query_tokens, boundary.slots)
                if capture_kv:
                    operation.kv_capture_receipt = self.kv_store.capture(
                        operation.capture_kv,
                        boundary.request_id,
                        boundary.row,
                        self._forward_metadata,
                        operation.processed_tokens,
                        source_position_offset=boundary.details["position_offset"],
                    )
        except Exception as error:
            self.fail_forward(error)
            raise

    def fail_forward(self, error: Exception) -> None:
        self._pending_consumed = None
        self._uninstall_export()
        for operation in self._all_operations():
            if not operation.closed:
                operation.failed = f"{type(error).__name__}: {error}"

    # ------------------------------------------------------------ query export
    def q_exports(self) -> dict:
        return self._q_store().list()

    def q_drop(self, name: str) -> dict:
        """Release an export; failed operations release theirs (no leaked pins)."""
        if any(
            not op.closed
            and not op.failed
            and op.export_q is not None
            and op.export_q["name"] == name
            for op in self._all_operations()
        ):
            raise CompactionContractError(
                "Cannot drop an active operation's query export"
            )
        store = self._q_store()
        store.drop(name)
        return {"dropped": name, **store.list()}

    # -------------------------------------------------------- resident access
    def _resident(
        self, request_id: str, *, expected_cursor: int | None
    ) -> tuple[int, dict, int]:
        """Row, FA block-table metadata and consumed cursor of a resident request.

        The worker's ``num_computed_tokens`` copies hold the scheduler's
        *pre-step* cursor (``_update_states`` sets them before the forward and
        nothing advances them afterwards), so after the last prefill chunk they
        lag by that chunk. The controller therefore tracks the cursor it saw
        consumed in ``before_forward``/``after_forward`` and uses that; the
        caller states ``expected_cursor`` (the end cursor it knows) and any
        mismatch is refused. ``expected_cursor=None`` is only for internal
        lookups that bound the range themselves.
        """
        runner = self.runner
        if runner.execute_model_state is not None:
            raise CompactionContractError(
                "Cannot read caches between forward and sampling"
            )
        if not isinstance(request_id, str) or not request_id:
            raise CompactionContractError("request_id must be a nonempty string")
        batch = runner.input_batch
        if request_id not in runner.requests or request_id not in batch.req_id_to_index:
            raise CompactionContractError(f"Request is not resident: {request_id}")
        row = int(batch.req_id_to_index[request_id])
        consumed = self._consumed_map().get(request_id)
        if consumed is None:
            raise CompactionContractError(
                f"No consumed cursor is tracked for {request_id}: it ran no forward "
                "under this controller"
            )
        batch_cursor = int(batch.num_computed_tokens_cpu[row])
        request_cursor = int(runner.requests[request_id].num_computed_tokens)
        if consumed < max(batch_cursor, request_cursor):
            raise CompactionContractError(
                f"Tracked consumed cursor {consumed} is behind the worker copies "
                f"({batch_cursor}, {request_cursor}) for {request_id}"
            )
        if expected_cursor is not None and (
            type(expected_cursor) is not int or expected_cursor != consumed
        ):
            raise CompactionContractError(
                f"expected_cursor {expected_cursor} differs from the consumed cursor "
                f"{consumed} of {request_id} (worker copy {batch_cursor})"
            )
        computed = consumed
        if computed < 1:
            raise CompactionContractError("Request has no consumed tokens")
        metadata = {}
        for name in self.fa_layers:
            group = self.fa_layer_groups.get(name)
            if group is None:
                raise CompactionContractError(f"Unknown KV group for FA layer: {name}")
            metadata[name] = SimpleNamespace(
                block_table=batch.block_table[group].get_cpu_tensor()
            )
        return row, metadata, computed

    def _position_offset(self, request_id: str, computed: int) -> int:
        rule = self._position_rules.get(request_id, (0, 0))
        return rule[1] if computed >= rule[0] else 0

    def kv_capture(self, spec: dict, request_id: str, *, expected_cursor: int) -> dict:
        """Capture selected rows from a resident request's cache, outside a forward.

        Same spec forms as ``capture_kv`` (shared or per-layer indices, optional
        ``method`` record); the source cursor is the request's tracked consumed
        prefix, which must equal ``expected_cursor``, and the recorded position
        offset is the request's current rule, as an after-forward capture would
        record.
        """
        name = spec.get("name") if isinstance(spec, dict) else None
        if any(
            not op.closed
            and not op.failed
            and op.capture_kv is not None
            and op.capture_kv.get("name") == name
            for op in self._all_operations()
        ):
            raise CompactionContractError(
                "KV capture name is reserved by another operation"
            )
        if type(expected_cursor) is not int:
            raise CompactionContractError("expected_cursor must be an integer")
        row, metadata, computed = self._resident(
            request_id, expected_cursor=expected_cursor
        )
        with self._store_errors():
            return self.kv_store.capture(
                spec,
                request_id,
                row,
                metadata,
                computed,
                source_position_offset=self._position_offset(request_id, computed),
            )

    @staticmethod
    @contextmanager
    def _store_errors():
        """Surface the KV store's ``ValueError``s as the one RPC error type."""
        try:
            yield
        except CompactionContractError:
            raise
        except ValueError as error:
            raise CompactionContractError(str(error)) from error

    def kv_subset(
        self,
        name_out: str,
        *,
        source_snapshot: str,
        method: dict,
        token_indices: list[int] | None = None,
        layer_token_indices: dict[str, list[int]] | None = None,
    ) -> dict:
        """Materialise a selection from an existing snapshot without a cache pass.

        Slices the source's immutable CPU rows (typically a full-KV capture at
        the boundary); inherits ``source_cursor`` / ``source_position_offset`` /
        ``request_id``; records ``derived_from`` and ``method``.
        """
        if any(
            not op.closed
            and not op.failed
            and op.capture_kv is not None
            and op.capture_kv.get("name") == name_out
            for op in self._all_operations()
        ):
            raise CompactionContractError(
                "KV capture name is reserved by another operation"
            )
        with self._store_errors():
            return self.kv_store.subset(
                name_out,
                source_snapshot,
                method=method,
                token_indices=token_indices,
                layer_token_indices=layer_token_indices,
            )

    def kv_bias_mask(
        self,
        name_out: str,
        *,
        source_snapshot: str,
        token_indices: list[int],
        value: float,
        heads: list[int] | None = None,
    ) -> dict:
        """Copy a snapshot with its per-key bias set to ``value`` on ``token_indices``.

        Kernel-level control for the bias path (harness): with ``value = -20``
        on a token set ``D`` the import must generate like ``kv_subset`` of the
        complement while differing from the full snapshot; ``heads`` (kv head
        indices, default all) must differ from both. K/V rows, cursor, positions
        and ``request_id`` are the source's; ``selection_policy`` is
        ``bias_mask``. Refused unless every FA layer carries a paged bias buffer.
        """
        if not self.kv_store.bias_supported():
            raise CompactionContractError(
                f"kv_bias_mask refused: {KV_BIAS_UNSUPPORTED}"
            )
        if any(
            not op.closed
            and not op.failed
            and op.capture_kv is not None
            and op.capture_kv.get("name") == name_out
            for op in self._all_operations()
        ):
            raise CompactionContractError(
                "KV capture name is reserved by another operation"
            )
        with self._store_errors():
            return self.kv_store.bias_mask(
                name_out,
                source_snapshot,
                token_indices=token_indices,
                value=value,
                heads=heads,
            )

    def _kv_layer_source(
        self,
        *,
        request_id: str | None,
        kv_snapshot: str | None,
        key_range: Any,
        expected_cursor: int | None,
    ) -> tuple[dict, Any]:
        """Describe a K/V source and return a per-layer row iterator.

        The iterator yields ``(layer_name, rows[T, Hkv, 2D] on the cache device,
        key_token_indices)``. Key token indices are request cursor positions,
        which order the cache chronologically; they are what the causal mask uses.
        ``expected_cursor`` is required for both kinds: the tracked consumed
        cursor of a resident request, or the recorded ``source_cursor`` of a
        snapshot.
        """
        if (request_id is None) == (kv_snapshot is None):
            raise CompactionContractError(
                "Give exactly one of request_id or kv_snapshot"
            )
        layers = self.kv_store.layers
        if request_id is not None:
            if type(expected_cursor) is not int:
                raise CompactionContractError(
                    "expected_cursor (the request's end cursor) is required for "
                    "resident sources"
                )
            row, metadata, computed = self._resident(
                request_id, expected_cursor=expected_cursor
            )
            if key_range is None:
                start, end = 0, computed
            else:
                start, end = _q_module()._check_token_range(key_range, "key_range")
            if end > computed:
                raise CompactionContractError("key_range exceeds the consumed prefix")
            indices = list(range(start, end))
            source = {
                "kind": "resident_request",
                "request_id": request_id,
                "cursor": computed,
                "key_range": [start, end],
                "position_offset": self._position_offset(request_id, computed),
                "expected_cursor": expected_cursor,
            }

            def rows():
                for name in layers:
                    gathered = self.kv_store.read_rows(
                        row, metadata, indices, computed_tokens=computed, layers=[name]
                    )
                    yield name, gathered[name], indices

            return source, rows
        if key_range is not None:
            raise CompactionContractError("key_range applies to resident requests only")
        entry = self.kv_store.describe(kv_snapshot)
        if type(expected_cursor) is not int:
            raise CompactionContractError(
                "expected_cursor (the snapshot's source cursor) is required for "
                "snapshot sources"
            )
        if expected_cursor != entry["source_cursor"]:
            raise CompactionContractError(
                f"expected_cursor {expected_cursor} differs from the snapshot's "
                f"source_cursor {entry['source_cursor']} ({kv_snapshot})"
            )
        if entry["layer_token_indices"] is None:
            raise CompactionContractError("Snapshot records no token indices to score")
        tensors = self.kv_store.snapshot_rows(kv_snapshot)
        source = {
            "kind": "kv_snapshot",
            "name": kv_snapshot,
            "request_id": entry["request_id"],
            "digest": entry["digest"],
            "cursor": entry["source_cursor"],
            "synthetic": entry["synthetic"],
            "position_offset": entry["source_position_offset"],
            "expected_cursor": expected_cursor,
        }

        def snapshot_rows():
            for name in layers:
                yield (
                    name,
                    tensors[name].to(layers[name].kv_cache.device),
                    list(entry["layer_token_indices"][name]),
                )

        return source, snapshot_rows

    def _q_source(self, q_export: str) -> tuple[dict, dict, torch.Tensor]:
        store = self._q_store()
        entry = store.describe(q_export)
        if not entry["complete"]:
            raise CompactionContractError(f"Query export is incomplete: {q_export}")
        if set(entry["layer_names"]) != set(self.kv_store.layers):
            raise CompactionContractError("Query export layers differ from FA layers")
        return entry, store.tensors(q_export), store.cursors(q_export)

    @staticmethod
    def _scale(layer: Any, params: dict) -> float:
        """The softmax scale passed explicitly to the methods library.

        Default ``head_size ** -0.5`` (the models' ``scaling``); the layer's own
        kernel scale must agree unless the caller overrides with ``params['scale']``.
        """
        override = params.get("scale")
        if override is not None:
            if (
                isinstance(override, bool)
                or not isinstance(override, (int, float))
                or override <= 0
            ):
                raise CompactionContractError(
                    "params['scale'] must be a positive number"
                )
            return float(override)
        head_size = getattr(layer, "head_size", None)
        if type(head_size) is not int or head_size < 1:
            raise CompactionContractError("Attention head_size unavailable for scaling")
        scale = float(head_size) ** -0.5
        kernel_scale = getattr(getattr(layer, "impl", None), "scale", None)
        if kernel_scale is not None and abs(float(kernel_scale) - scale) > 1e-6 * scale:
            raise CompactionContractError(
                f"Layer kernel scale {kernel_scale} differs from head_size ** -0.5 "
                f"({scale}); pass params['scale'] explicitly"
            )
        return scale

    @staticmethod
    def _protected_tokens(protected: Any) -> list[int]:
        if protected is None:
            return []
        if (
            not isinstance(protected, list)
            or any(type(i) is not int or i < 0 for i in protected)
            or protected != sorted(set(protected))
        ):
            raise CompactionContractError(
                "protected must be unique, sorted, nonnegative token indices"
            )
        return list(protected)

    # ------------------------------------------------------------ compute ops
    def _provenance(self) -> dict:
        """Library provenance computed once per controller (git is shelled out)."""
        compaction_q = _q_module()
        if self._provenance_cache is None:
            self._provenance_cache = compaction_q.library_provenance()
        return {
            **self._provenance_cache,
            "registered_override": sorted(compaction_q._REGISTRY),
        }

    def _budget(self, params: dict) -> dict:
        device = next(iter(self.kv_store.layers.values())).kv_cache.device
        return _q_module().memory_budget(device, params.get("memory_budget_bytes"))

    def kv_score(
        self,
        name_out: str,
        *,
        q_export: str,
        method: str,
        request_id: str | None = None,
        kv_snapshot: str | None = None,
        expected_cursor: int | None = None,
        params: dict | None = None,
        key_range: Any = None,
    ) -> dict:
        """Score every key of the source per ``(layer, kv_head, token)``.

        ``h2o`` (library variant ``h2o_uniform_prefill``): accumulated causal
        attention mass of the exported queries over the keys, positions = request
        token indices; with a ``kv_snapshot`` source the export must come from the
        snapshot's request. ``kvzip`` (``kvzip_uniform_perlayer_*``): the paper
        normaliser needs the repeat input's own keys ``k_ref`` (read from the
        exporting request's cache); on a resident source the scored keys must be
        exactly ``[0, repeat_start)`` (frame + context, no gap, no overlap); on a
        snapshot source the normaliser is labelled
        ``snapshot_plus_causal_repeat_keys`` and dropped prefix keys are counted.
        ``params['context_only_normalisation']`` selects the library's
        ``context_only`` deviation. Blocks are sized from ``params['chunk']`` or a
        memory budget (``params['memory_budget_bytes']`` or the device-derived
        default). Resident sources require ``expected_cursor``. Scores are float32
        CPU tensors in ``ScoreStore[name_out]``; the receipt carries the library
        ``variant`` and ``library_params``.
        """
        compaction_q = _q_module()
        if method not in compaction_q.SCORE_METHODS:
            raise CompactionContractError(f"Unknown scoring method: {method}")
        params = compaction_q._check_params(
            params,
            {
                "chunk",
                "scale",
                "context_only_normalisation",
                "memory_budget_bytes",
                "repeat_prompt",
            },
            "score",
        )
        repeat_prompt = params.get("repeat_prompt")
        if repeat_prompt is not None and (
            method != "kvzip" or not isinstance(repeat_prompt, str) or not repeat_prompt
        ):
            raise CompactionContractError(
                "repeat_prompt must be a nonempty string and applies to kvzip only"
            )
        if not isinstance(name_out, str) or not name_out:
            raise CompactionContractError("Score names must be nonempty")
        if name_out in self.score_store._entries:
            raise CompactionContractError("Score names cannot be overwritten")
        q_entry, q_tensors, q_cursors = self._q_source(q_export)
        q_start, q_end = q_entry["token_range"]
        context_only = bool(params.get("context_only_normalisation"))
        if method == "kvzip" and q_start == 0:
            raise CompactionContractError(
                "kvzip repeat range cannot start at 0: there are no cached keys "
                "before the repeat input to score"
            )
        if method == "kvzip" and request_id is not None and not context_only:
            if key_range is None:
                key_range = [0, q_start]
            elif list(key_range) != [0, q_start]:
                raise CompactionContractError(
                    "kvzip paper normalisation requires key_range == "
                    f"[0, repeat_start] = [0, {q_start}] (frame + context, no gap, "
                    "no overlap)"
                )
        source, rows = self._kv_layer_source(
            request_id=request_id,
            kv_snapshot=kv_snapshot,
            key_range=key_range,
            expected_cursor=expected_cursor,
        )
        if (
            method == "h2o"
            and kv_snapshot is not None
            and (
                q_entry["request_id"] is None
                or source.get("request_id") is None
                or q_entry["request_id"] != source.get("request_id")
            )
        ):
            raise CompactionContractError(
                "h2o with a kv_snapshot source requires the query export and the "
                "snapshot to come from the same (known) request"
            )
        k_ref_reader = None
        k_ref_info = None
        if method == "kvzip" and not context_only:
            ref_request = q_entry["request_id"]
            if ref_request is None:
                raise CompactionContractError(
                    "kvzip needs the query export's request for k_ref"
                )
            ref_row, ref_metadata, ref_computed = self._resident(
                ref_request,
                expected_cursor=expected_cursor if ref_request == request_id else None,
            )
            if q_end > ref_computed:
                raise CompactionContractError(
                    "kvzip k_ref range exceeds the exporting request's consumed prefix"
                )
            ref_indices = list(range(q_start, q_end))

            def k_ref_reader(name):
                return self.kv_store.read_rows(
                    ref_row,
                    ref_metadata,
                    ref_indices,
                    computed_tokens=ref_computed,
                    layers=[name],
                )[name]

            k_ref_info = {"request_id": ref_request, "token_range": [q_start, q_end]}
        budget = self._budget(params)
        started = time.perf_counter()
        scores, key_indices, scales = {}, {}, {}
        variants, library_params, blockings, dropped_prefix = set(), {}, {}, {}
        library_warnings = {}
        for name, kv_rows, indices in rows():
            layer = self.kv_store.layers[name]
            keys, _ = compaction_q.split_kv(kv_rows, layer.head_size)
            k_ref = None
            if k_ref_reader is not None:
                k_ref, _ = compaction_q.split_kv(k_ref_reader(name), layer.head_size)
                overlapping = sum(1 for t in indices if t >= q_start)
                if overlapping:
                    raise CompactionContractError(
                        f"kvzip source keys overlap the repeat rows [{q_start}, "
                        f"{q_end}) in {name}: the normaliser would count them twice"
                    )
                dropped_prefix[name] = q_start - len(indices)
            scales[name] = self._scale(layer, params)
            result = compaction_q.score_layer(
                method,
                q=q_tensors[name].to(keys.device),
                k=keys,
                scale=scales[name],
                q_positions=q_cursors,
                k_positions=torch.tensor(indices, dtype=torch.long),
                params=params,
                k_ref=k_ref,
                budget=budget,
            )
            scores[name] = result["scores"]
            key_indices[name] = list(indices)
            variants.add(result["variant"])
            library_params[name] = result["library_params"]
            library_warnings[name] = result["library_warnings"]
            blockings[name] = result["blocking"]
        if len(variants) != 1:
            raise CompactionContractError(
                f"FA layers disagree on the score variant: {sorted(map(str, variants))}"
            )
        if method != "kvzip":
            normaliser = "causal_over_scored_keys"
        elif context_only:
            normaliser = "context_only"
        elif kv_snapshot is not None:
            normaliser = "snapshot_plus_causal_repeat_keys"
        else:
            normaliser = "context_plus_causal_repeat_keys"
        receipt = self.score_store.put(
            name_out,
            layers=scores,
            key_token_indices=key_indices,
            method=method,
            params={**params, "blocking": blockings, "memory_budget": budget},
            q_export=q_export,
            source={
                **source,
                "q_token_range": q_entry["token_range"],
                "q_rows": q_entry["rows"],
                "query_convention": q_entry["query_convention"],
                "scale": scales,
                "k_ref": k_ref_info,
                "normalisation": normaliser,
                "dropped_prefix_keys": (
                    dropped_prefix
                    if normaliser == "snapshot_plus_causal_repeat_keys"
                    else None
                ),
            },
            variant=variants.pop(),
            library_params=library_params,
            library_warnings=library_warnings,
        )
        receipt["seconds"] = time.perf_counter() - started
        return receipt

    def kv_select(
        self,
        name_out: str,
        *,
        scores: str,
        budget_tokens: int,
        policy: str,
        aggregate: str,
        protected: list[int] | None = None,
    ) -> dict:
        """Uniform token budget per layer from stored scores.

        ``policy`` (``'shared'`` | ``'per_layer'``) and ``aggregate`` (``'max'`` |
        ``'mean'``) are required. Returns (and stores under ``name_out``)
        ``layer_token_indices`` (request token indices per layer, equal counts),
        ``token_indices`` for the shared policy, the score ``variant`` /
        ``library_params``, a self-describing ``method`` record (variant, params,
        protected, policy, aggregate, scores name, q_export, source and the
        methods-library provenance) and a ``capture_spec`` carrying that record,
        ready for ``capture_kv`` / ``kv_capture``. ``protected`` are token indices
        that must be kept.
        """
        compaction_q = _q_module()
        if not isinstance(name_out, str) or not name_out:
            raise CompactionContractError("Selection names must be nonempty")
        if name_out in self.selection_store._entries:
            raise CompactionContractError("Selection names cannot be overwritten")
        if policy not in compaction_q.SELECT_POLICIES:
            raise CompactionContractError(f"Unknown selection policy: {policy}")
        if aggregate not in compaction_q.SELECT_AGGREGATES:
            raise CompactionContractError(f"Unknown selection aggregate: {aggregate}")
        described = self.score_store.describe(scores)
        tensors = self.score_store.tensors(scores)
        key_indices = self.score_store.key_token_indices(scores)
        order = [name for name in self.kv_store.layers if name in tensors]
        if set(order) != set(tensors):
            raise CompactionContractError("Scores name layers outside the FA set")
        if len({tuple(v) for v in key_indices.values()}) != 1:
            raise CompactionContractError(
                "Selection requires every layer to score the same key tokens"
            )
        tokens = key_indices[order[0]]
        position_of = {token: position for position, token in enumerate(tokens)}
        protected_tokens = self._protected_tokens(protected)
        missing = [t for t in protected_tokens if t not in position_of]
        if missing:
            raise CompactionContractError(
                f"protected tokens are not among the scored keys: {missing}"
            )
        started = time.perf_counter()
        selections = compaction_q.select_layers(
            [tensors[name] for name in order],
            budget_tokens=budget_tokens,
            protected=[position_of[t] for t in protected_tokens],
            policy=policy,
            aggregate=aggregate,
        )
        layer_tokens = {
            name: [tokens[p] for p in chosen] for name, chosen in zip(order, selections)
        }
        shared = policy == "shared"
        variant = described["variant"] or f"{described['method']}_uniform"
        method = {
            "name": variant,
            "params": {
                "score_method": described["method"],
                "score_params": described["params"],
                "library_params": described["library_params"],
                "budget_tokens": budget_tokens,
                "protected": protected_tokens,
                "policy": policy,
                "aggregate": aggregate,
            },
            "inputs": {
                "scores": scores,
                "scores_digest": described["digest"],
                "q_export": described["q_export"],
                "source": described["source"],
                "library": self._provenance(),
            },
        }
        capture_spec = (
            {"name": name_out, "token_indices": layer_tokens[order[0]]}
            if shared
            else {"name": name_out, "layer_token_indices": layer_tokens}
        )
        capture_spec["method"] = method
        result = {
            "name": name_out,
            "scores": scores,
            "score_method": described["method"],
            "variant": variant,
            "library_params": described["library_params"],
            "method": method,
            "policy": policy,
            "aggregate": aggregate,
            "budget_tokens": budget_tokens,
            "protected": protected_tokens,
            "retained_tokens": len(selections[0]),
            "scored_tokens": len(tokens),
            "layer_order": order,
            "layer_token_indices": layer_tokens,
            "token_indices": layer_tokens[order[0]] if shared else None,
            "capture_spec": capture_spec,
            "seconds": time.perf_counter() - started,
        }
        return self.selection_store.put(name_out, result)

    def kv_selections(self) -> dict:
        return self.selection_store.list()

    def kv_selection_drop(self, name: str) -> dict:
        """Release a selection so a retried task may reuse its name.

        Refused only while a live (not closed, not failed) operation reserves a
        KV capture of that name, i.e. a boundary capture of the selection's
        ``capture_spec`` is still pending.
        """
        if any(
            not op.closed
            and not op.failed
            and op.capture_kv is not None
            and op.capture_kv.get("name") == name
            for op in self._all_operations()
        ):
            raise CompactionContractError(
                "Cannot drop a selection whose capture is reserved by an operation"
            )
        self.selection_store.drop(name)
        return {"dropped": name, **self.selection_store.list()}

    def kv_describe(self, name: str) -> dict:
        """One KV snapshot's metadata (no tensors), without listing the store."""
        with self._store_errors():
            return self.kv_store.describe(name)

    def q_describe(self, name: str) -> dict:
        """One query export's metadata (no tensors), without listing the store."""
        return self._q_store().describe(name)

    def kv_fit_am(
        self,
        name_out: str,
        *,
        q_export: str,
        budget_tokens: int,
        request_id: str | None = None,
        kv_snapshot: str | None = None,
        expected_cursor: int | None = None,
        protected: list[int] | None = None,
        params: dict | None = None,
        key_range: Any = None,
    ) -> dict:
        """Attention matching: selected original keys, OLS-fitted values, optional bias.

        Per FA layer ``am.compact(q_ref, k, v, budget, bias=<params['bias']>,
        head_budget='uniform', fixed=<frame positions>, protected=<refit-able
        positions>, scale=, ridge=, chunk= | memory_budget_bytes=[,
        mass_weighting=])``. ``protected`` (token indices) are the frame: frozen
        with their original K and V (``fixed=``, beta = 0) unless
        ``params['refit_protected']`` makes them refit-able (``protected=``).
        Values are fitted in float32 (``v`` is passed as float32) and cast once to
        the cache dtype here; the value-space cast error and, unless
        ``params['output_error']`` is False, the post-cast attention-output error
        of the stored rows (with beta when fitted) against the full cache are
        recorded. ``method.name`` is the library variant (for example
        ``am_rmskeys_ols_nobias_uniform_offpolicy_framefixed_chol64``), ``alias``
        ``am_nobias_uniform`` or ``am_bias_uniform``. ``params['bias']`` (default
        False) fits the per-key logit bias by NNLS mass matching with
        ``params['mass_weighting']`` ('uniform' | 'shifted', default 'uniform')
        and registers it beside the rows as float32 ``[t, num_kv_heads]`` per
        layer (frame rows 0); it needs every FA layer to carry a paged bias
        buffer (Triton attention backend, ``info()['compute_ops']`` lists
        ``kv_bias``) and is refused otherwise. ``mass_weighting`` without
        ``bias=True``, ``iters`` and per-head budgets are refused. Resident
        sources require ``expected_cursor``.
        """
        compaction_q = _q_module()
        params = compaction_q._check_params(
            params,
            {
                "ridge",
                "chunk",
                "scale",
                "digest_inputs",
                "refit_protected",
                "memory_budget_bytes",
                "output_error",
                "bias",
                "mass_weighting",
            },
            "am",
        )
        bias = params.get("bias", False)
        if not isinstance(bias, bool):
            raise CompactionContractError("bias must be a bool")
        if "mass_weighting" in params and bias is not True:
            raise CompactionContractError("mass_weighting requires bias=True")
        mass_weighting = params.get("mass_weighting", "uniform")
        if mass_weighting not in ("uniform", "shifted"):
            raise CompactionContractError(
                "mass_weighting must be 'uniform' or 'shifted'"
            )
        if bias and not self.kv_store.bias_supported():
            raise CompactionContractError(
                f"kv_fit_am(bias=True) refused: {KV_BIAS_UNSUPPORTED}"
            )
        if type(budget_tokens) is not int or budget_tokens < 1:
            raise CompactionContractError("budget_tokens must be a positive integer")
        if not isinstance(name_out, str) or not name_out:
            raise CompactionContractError("KV snapshot name must be nonempty")
        if name_out in self.kv_store._entries:
            raise CompactionContractError("KV snapshot names cannot be overwritten")
        refit_protected = params.get("refit_protected", False)
        output_error_setting = params.get("output_error")
        if not isinstance(refit_protected, bool) or not isinstance(
            output_error_setting, (bool, type(None))
        ):
            raise CompactionContractError(
                "refit_protected must be a bool and output_error a bool or None"
            )
        want_output_error = None
        output_error_policy = None
        protected_tokens = self._protected_tokens(protected)
        q_entry, q_tensors, _ = self._q_source(q_export)
        source, rows = self._kv_layer_source(
            request_id=request_id,
            kv_snapshot=kv_snapshot,
            key_range=key_range,
            expected_cursor=expected_cursor,
        )
        budget = self._budget(params)
        started = time.perf_counter()
        synthetic, layer_tokens, kv_digests, betas = {}, {}, {}, {}
        casts, variants, diagnostics, summaries, scales = set(), set(), {}, {}, {}
        cast_errors, output_errors, blockings, fixed_positions = {}, {}, {}, {}
        library_warnings, fit_workspace = {}, {}
        for name, kv_rows, indices in rows():
            layer = self.kv_store.layers[name]
            keys, values = compaction_q.split_kv(kv_rows, layer.head_size)
            position_of = {token: position for position, token in enumerate(indices)}
            missing = [t for t in protected_tokens if t not in position_of]
            if missing:
                raise CompactionContractError(
                    f"protected tokens are not among the source keys: {missing}"
                )
            scales[name] = self._scale(layer, params)
            positions = [position_of[t] for t in protected_tokens]
            fixed, refit = ([], positions) if refit_protected else (positions, [])
            q_ref = q_tensors[name].to(keys.device)
            # Fit admission (Codex R1): sampled with this layer's inputs on device.
            capacity = compaction_q.fit_workspace_capacity(keys.device)
            fit = compaction_q.fit_am_layer(
                q_ref=q_ref,
                k=keys,
                v=values.float(),
                budget=budget_tokens,
                protected=refit,
                fixed=fixed,
                scale=scales[name],
                params=params,
                budget_bytes=budget,
                max_fit_workspace_bytes=capacity["bytes"],
                bias=bias,
                mass_weighting=mass_weighting,
            )
            if bias:
                # Library beta is [Hkv, t]; the store keeps token-major rows.
                betas[name] = fit["beta"].transpose(0, 1).contiguous()
            fit_workspace[name] = {
                "capacity": capacity,
                "admission": fit["summary"].get("fit_workspace_admission"),
            }
            fitted32 = fit["v_c"].to(device=keys.device)
            fitted = fitted32.to(dtype=values.dtype)
            casts.add(f"{fitted32.dtype}->{values.dtype}")
            cast_errors[name] = compaction_q.cast_error(fitted32, fitted)
            k_c = fit["k_c"].to(device=keys.device, dtype=keys.dtype)
            if want_output_error is None:
                threshold = compaction_q.OUTPUT_ERROR_MAX_TOKENS
                if output_error_setting is None:
                    want_output_error = int(keys.shape[0]) <= threshold
                    output_error_policy = (
                        f"default_{'on' if want_output_error else 'off'}"
                        f"_threshold_{threshold}_tokens"
                    )
                else:
                    want_output_error = output_error_setting
                    output_error_policy = "caller"
            if want_output_error:
                # Free memory is sampled now, with the layer's inputs on device.
                error_budget = compaction_q.memory_budget(
                    keys.device, params.get("memory_budget_bytes")
                )
                output_errors[name] = compaction_q.attention_output_error(
                    q_ref,
                    keys,
                    values,
                    k_c,
                    fitted,
                    scale=scales[name],
                    budget_bytes=int(error_budget["bytes"]),
                    **({"beta": fit["beta"].to(keys.device)} if bias else {}),
                )
                output_errors[name]["memory_budget"] = error_budget
            variants.add(fit["variant"])
            diagnostics[name] = fit["diagnostics"]
            summaries[name] = fit["summary"]
            library_warnings[name] = fit["warnings"]
            blockings[name] = fit["blocking"]
            fixed_positions[name] = [indices[p] for p in fit["fixed_positions"]]
            synthetic[name] = torch.cat([k_c, fitted], dim=-1).cpu()
            layer_tokens[name] = [indices[p] for p in fit["indices"]]
            if params.get("digest_inputs"):
                kv_digests[name] = compaction_q._digest_tensors({name: kv_rows})
        retained = {len(v) for v in layer_tokens.values()}
        frame_policies = {s.get("frame_policy") for s in summaries.values()}
        if len(retained) != 1 or len(variants) != 1 or len(frame_policies) != 1:
            raise CompactionContractError(
                "AM layers disagree on count, variant or frame policy"
            )
        first_summary = next(iter(summaries.values()))
        digest_inputs = bool(params.get("digest_inputs"))
        bias_record = (
            {
                "mass_weighting": mass_weighting,
                "beta_layout": (
                    "[retained_tokens, num_kv_heads] float32, frame rows 0"
                ),
                "mass_error_before": {
                    n: s.get("mass_error_before") for n, s in summaries.items()
                },
                "mass_error_after": {
                    n: s.get("mass_error_after") for n, s in summaries.items()
                },
                "bias_fallback": {
                    n: s.get("bias_fallback") for n, s in summaries.items()
                },
                "any_bias_fallback": {
                    n: s.get("any_bias_fallback") for n, s in summaries.items()
                },
            }
            if bias
            else {}
        )
        method = {
            "name": variants.pop(),
            "alias": "am_bias_uniform" if bias else "am_nobias_uniform",
            "params": {
                **params,
                "blocking": blockings,
                "memory_budget": budget,
                "ridge": params.get("ridge", 0.0),
                "scale": scales,
                "budget_tokens": budget_tokens,
                "protected": protected_tokens,
                "frame_policy": frame_policies.pop(),
                "fixed_tokens": fixed_positions,
                "n_fixed": {n: s.get("n_fixed") for n, s in summaries.items()},
                "n_protected": {n: s.get("n_protected") for n, s in summaries.items()},
                "bias": bias,
                **bias_record,
                "head_budget": "uniform",
                "solver": first_summary.get("solver"),
                "compute_dtype": first_summary.get("compute_dtype"),
                "accumulate_dtype": first_summary.get("accumulate_dtype"),
                "identity_shortcut": {
                    n: s.get("identity_shortcut") for n, s in summaries.items()
                },
                "fit_value_dtype": "torch.float32",
                "values_cast": sorted(casts),
                "cast_error": cast_errors,
                "output_error_after_cast_in_sample": (
                    output_errors if want_output_error else None
                ),
                "output_error_policy": output_error_policy,
                "output_error_note": (
                    "in-sample: measured on the same reference queries as the fit"
                ),
                "output_error_after_rel_library": {
                    n: s.get("output_error_after_rel") for n, s in summaries.items()
                },
                "warnings": {n: s.get("warnings") for n, s in summaries.items()},
                "library_warnings": library_warnings,
                "fit_workspace": fit_workspace,
            },
            "inputs": {
                "q_export": q_export,
                "q_token_range": q_entry["token_range"],
                "q_digest": (
                    compaction_q._digest_tensors(q_tensors) if digest_inputs else None
                ),
                "q_digest_policy": "computed" if digest_inputs else "not_computed",
                "query_convention": q_entry["query_convention"],
                "source": source,
                "kv_digests": kv_digests if digest_inputs else None,
                "library": self._provenance(),
            },
            "diagnostics": diagnostics,
        }
        receipt = self.kv_store.register(
            name_out,
            synthetic,
            source_cursor=int(source["cursor"]),
            source_position_offset=int(source["position_offset"]),
            retained_tokens=retained.pop(),
            method=method,
            layer_token_indices=layer_tokens,
            request_id=source.get("request_id"),
            beta=betas if bias else None,
        )
        receipt["variant"] = method["name"]
        receipt["bias"] = bias
        receipt["seconds"] = time.perf_counter() - started
        return receipt

    def scores(self) -> dict:
        if self.score_store is None:
            raise CompactionContractError("Score store is unavailable")
        return self.score_store.list()

    def score_drop(self, name: str) -> dict:
        if self.score_store is None:
            raise CompactionContractError("Score store is unavailable")
        self.score_store.drop(name)
        return {"dropped": name, **self.score_store.list()}
