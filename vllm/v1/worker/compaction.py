# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional per-request compaction at scheduler-owned V1 token boundaries."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

DESCRIPTOR_KEY = "native_compaction_v1"
COMPONENTS = {"all": (0, 1), "conv": (0,), "matrix": (1,)}


class CompactionContractError(ValueError):
    """The requested operation is outside the supported diagnostic contract."""


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
        if self.failed or not self.prompt_complete or self._pending_tokens is not None:
            raise CompactionContractError(
                self.failed or "The prompt-end boundary was not reached"
            )
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

    fresh_decode_rows_zeroed = 0  # class default: fixtures may bypass __init__

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
        if not self.layers or not self.fa_layers:
            raise CompactionContractError(
                "Expected a hybrid with recurrent and FA layers"
            )
        self.store = SnapshotStore()
        self.kv_store = NativeKVSnapshotStore(runner)
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
        )
        self._check_capture_names(operation)
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
        }

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
        self.operations[operation_id] = operation
        self._descriptors[operation_id] = copy.deepcopy(descriptor)
        self._seen_operation_ids.add(operation_id)
        return operation

    def result(self) -> dict:
        if self.operation is None:
            raise CompactionContractError("No operation has been armed")
        return {**self.info(), **self.operation.result()}

    def results(self, operation_ids: list[str]) -> dict:
        if len(set(operation_ids)) != len(operation_ids):
            raise CompactionContractError("Duplicate operation_ids")
        for operation_id in operation_ids:
            operation = self.operations.get(operation_id)
            if operation is None:
                raise CompactionContractError(f"Unknown operation: {operation_id}")
            if (
                operation.failed
                or not operation.prompt_complete
                or operation._pending_tokens is not None
            ):
                raise CompactionContractError(
                    operation.failed or "Operation has not reached its prompt end"
                )
        receipts = {
            key: {**self.info(), **self.operations[key].result()}
            for key in operation_ids
        }
        for key in operation_ids:
            operation = self.operations.pop(key)
            if operation.request_finished:
                self._descriptors.pop(key, None)
            else:
                self._retired_bindings[operation.request_id] = key
        return receipts

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
                    needs_fa = (
                        operation.request_id is None
                        or computed == operation.restore_at
                        or computed + count == operation.prompt_tokens
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
                        if (
                            source["source_cursor"] - computed
                            != operation.position_offset
                        ):
                            raise CompactionContractError(
                                "KV import requires original next-query positions"
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
            return boundaries
        except Exception as error:
            self.fail_forward(error)
            raise

    def after_forward(self, boundaries: list[ForwardBoundary] | None) -> None:
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
        for operation in self._all_operations():
            if not operation.closed:
                operation.failed = f"{type(error).__name__}: {error}"
