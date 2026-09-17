# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native-state snapshots and request-boundary compaction for V1 workers.

This optional diagnostic keeps scheduler ownership of KV allocation. CPU contract
checks do not establish GPU kernel parity or a streaming memory bound.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class CompactionContractError(ValueError):
    """The requested operation is outside the supported diagnostic contract."""


def validate_configuration(config: Any) -> None:
    """Reject modes whose cache lifecycle the first backend does not support."""
    required = (
        (config.parallel_config.tensor_parallel_size == 1, "TP must be 1"),
        (config.parallel_config.pipeline_parallel_size == 1, "PP must be 1"),
        (config.parallel_config.data_parallel_size == 1, "DP must be 1"),
        (config.scheduler_config.max_num_seqs == 1, "max_num_seqs must be 1"),
        (
            config.scheduler_config.async_scheduling is False,
            "async scheduling is unsupported",
        ),
        (
            config.scheduler_config.enable_chunked_prefill,
            "chunked prefill must be enabled",
        ),
        (config.model_config.enforce_eager, "eager execution is required"),
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


def resolve_slot(name: str, family: str, states: tuple, metadata: Any) -> StateSlot:
    """Resolve the one active GDN/Mamba2 row without assuming a block ID."""
    if family not in {"gdn", "mamba2"} or len(states) != 2:
        raise CompactionContractError(
            "Only native conv+SSM GDN/Mamba2 layouts are supported"
        )
    prefills, decodes = int(metadata.num_prefills), int(metadata.num_decodes)
    if prefills + decodes != 1 or prefills not in (0, 1):
        raise CompactionContractError(
            "Exactly one non-speculative state row is required"
        )
    if family == "gdn":
        if getattr(metadata, "num_spec_decodes", 0) != 0:
            raise CompactionContractError("Speculative GDN rows are unsupported")
        indices = (
            metadata.prefill_state_indices
            if prefills
            else metadata.non_spec_state_indices_tensor
        )
    else:
        indices = (
            metadata.state_indices_tensor_p
            if prefills
            else metadata.state_indices_tensor_d
        )
    if not isinstance(indices, torch.Tensor) or indices.numel() != 1:
        raise CompactionContractError(f"Unexpected state index layout in {name}")
    index = int(indices.reshape(-1)[0].item())
    for state in states:
        if (
            not isinstance(state, torch.Tensor)
            or state.ndim < 2
            or not 0 <= index < state.shape[0]
        ):
            raise CompactionContractError(f"Invalid native state slot in {name}")
    slot = StateSlot(name, family, tuple(states), index, metadata, bool(prefills))
    _initial_state_flags(slot)  # Validate before any restore writes.
    return slot


def _initial_state_flags(slot: StateSlot) -> list[torch.Tensor]:
    if not slot.prefill:
        return []  # Native single-token decode unconditionally consumes its slot.
    keys = (
        ("has_initial_state", "prefill_has_initial_state")
        if slot.family == "gdn"
        else ("has_initial_states_p",)
    )
    flags = []
    for key in keys:
        flag = getattr(slot.metadata, key, None)
        if (
            not isinstance(flag, torch.Tensor)
            or flag.dtype != torch.bool
            or flag.numel() != 1
        ):
            raise CompactionContractError(f"Invalid {key} in {slot.name}")
        flags.append(flag)
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
) -> dict:
    """Read actual FA lengths; reject unknown layouts instead of inventing proof."""
    if not layer_names:
        raise CompactionContractError("No full-attention layers were found")
    evidence = {}
    for name in sorted(layer_names):
        current = metadata.get(name)
        lengths = getattr(current, "seq_lens", None)
        location = "seq_lens"
        if lengths is None:
            for part in ("prefill", "decode"):
                nested = getattr(current, part, None)
                lengths = getattr(nested, "seq_lens", None)
                if lengths is not None:
                    location = f"{part}.seq_lens"
                    break
        if not isinstance(lengths, torch.Tensor) or lengths.numel() != 1:
            raise CompactionContractError(
                f"Unsupported FA sequence-length metadata: {name}"
            )
        seq_len = int(lengths.reshape(-1)[0].item())
        context_len = seq_len - query_tokens
        if context_len != computed_tokens:
            raise CompactionContractError(
                f"FA context and worker cursor disagree: {name}"
            )
        starts = getattr(current, "query_start_loc", None)
        if starts is not None and (
            starts.numel() != 2 or int((starts[-1] - starts[0]).item()) != query_tokens
        ):
            raise CompactionContractError(
                f"FA query and recurrent token counts disagree: {name}"
            )
        evidence[name] = {
            "seq_len": seq_len,
            "query_tokens": query_tokens,
            "context_tokens": context_len,
            "metadata_field": location,
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
    """Process-local, private CPU snapshots; RPC clients receive metadata only."""

    def __init__(self):
        self._entries: dict[str, dict[str, Any]] = {}

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
        }
        return self.describe(name)

    def describe(self, name: str) -> dict:
        if name not in self._entries:
            raise CompactionContractError(f"Unknown snapshot: {name}")
        entry = self._entries[name]
        actual_digest = _tensor_digest(entry["layers"])
        if actual_digest != entry["digest"]:
            raise CompactionContractError(f"Snapshot mutated: {name}")
        native_bytes = sum(
            state.numel() * state.element_size()
            for _, states in entry["layers"].values()
            for state in states
        )
        return {
            "name": name,
            "sha256": actual_digest,
            "immutable_digest_verified": True,
            "all_native_values_finite": True,
            "digest": actual_digest,
            "bytes": native_bytes,
            "layer_count": len(entry["layers"]),
            "storage_device": "cpu",
            "boundary": copy.deepcopy(entry["boundary"]),
            "native_bytes": native_bytes,
            "layers": {
                key: {
                    "family": family,
                    "shapes": [list(x.shape) for x in states],
                    "dtypes": [str(x.dtype) for x in states],
                }
                for key, (family, states) in entry["layers"].items()
            },
        }

    def restore(self, name: str, slots: list[StateSlot]) -> dict:
        receipt = self.describe(name)
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
        with torch.no_grad():
            for slot in slots:
                for source, target in zip(layers[slot.name][1], slot.states):
                    target[slot.index].copy_(source)
                for flag in _initial_state_flags(slot):
                    flag.fill_(True)
                if slot.family == "mamba2" and slot.prefill:
                    slot.metadata.prep_initial_states = True
        receipt["restore_modes"] = {
            slot.name: "prefill_initial_state_enabled"
            if slot.prefill
            else "native_decode_state"
            for slot in slots
        }
        receipt["snapshot_unchanged_after_restore"] = (
            self.describe(name)["sha256"] == receipt["sha256"]
        )
        return receipt

    def compare(self, name_a: str, name_b: str) -> dict:
        """Relative Frobenius differences between two snapshots (per layer, per tensor); no mutation."""
        self.describe(name_a)
        self.describe(name_b)
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
                rel = float(torch.linalg.vector_norm(xf - yf)) / (denom if denom > 0 else 1.0)
                rels.append(rel)
                max_rel = max(max_rel, rel)
            per_layer[name] = rels
        return {"a": name_a, "b": name_b, "max_relative_frobenius": max_rel, "per_layer": per_layer}

    def list(self) -> dict:
        snapshots = {name: self.describe(name) for name in self._entries}
        return {
            "snapshots": snapshots,
            "total_cpu_snapshot_bytes": sum(
                s["native_bytes"] for s in snapshots.values()
            ),
        }

    def drop(self, name: str) -> None:
        self.describe(name)
        del self._entries[name]


class BoundaryOperation:
    """One request or continuation; only a fresh epoch can restore native state."""

    def __init__(
        self,
        store: SnapshotStore,
        *,
        capture_name: str | None,
        restore_name: str | None,
        expected_prompt_tokens: int,
        start_cursor: int = 0,
        expected_request_id: str | None = None,
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
        if restore_name is not None and start_cursor != 0:
            raise CompactionContractError("Restore requires start_cursor == 0")
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
        self.request_id: str | None = None
        self.first_computed_tokens: int | None = None
        self.processed_tokens = start_cursor
        self.prompt_complete = False
        self.closed = False
        self.failed: str | None = None
        self.chunks: list[dict] = []
        self.restore_receipt: dict | None = None
        self.capture_receipt: dict | None = None
        self.first_fa_context: dict | None = None
        self._pending_tokens: int | None = None

    def before(
        self,
        request_id: str,
        prompt_tokens: int,
        computed_tokens: int,
        query_tokens: int,
        slots: list[StateSlot],
        fa_context: dict | None = None,
    ) -> None:
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
        if computed_tokens < prompt_tokens < computed_tokens + query_tokens:
            raise CompactionContractError(
                "A forward crosses the declared prompt boundary"
            )
        if first_forward:
            if (
                self.start_cursor == 0
                and self.restore_name is None
                and any(not slot.prefill for slot in slots)
            ):
                # Recurrent slots are not zeroed on allocation; only the prefill kernels
                # zero rows without an initial-state flag. A fresh session whose first
                # forward is a decode row would read a stale slot.
                raise CompactionContractError(
                    "A fresh session without restore must start with a prefill forward"
                )
            if self.restore_name is not None:
                self.restore_receipt = self.store.restore(self.restore_name, slots)
            self.request_id = request_id
            self.first_computed_tokens = computed_tokens
            self.first_fa_context = copy.deepcopy(fa_context)
        self._pending_tokens = query_tokens

    def after(self, query_tokens: int, slots: list[StateSlot]) -> None:
        if self.closed or self.failed or self._pending_tokens != query_tokens:
            raise CompactionContractError("No matching successful forward is pending")
        start = self.processed_tokens
        self.processed_tokens += query_tokens
        self._pending_tokens = None
        self.chunks.append(
            {"start": start, "end": self.processed_tokens, "query_tokens": query_tokens}
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
                        "position_policy": "reset",
                        "snapshot_boundary": "prompt_end_before_first_sample",
                        "sampled_output_tokens_consumed": 0,
                    },
                )
            self.prompt_complete = True

    def result(self) -> dict:
        if self.failed or not self.prompt_complete or self._pending_tokens is not None:
            raise CompactionContractError(
                self.failed or "The prompt-end boundary was not reached"
            )
        if self.restore_name is not None:
            self.store.describe(self.restore_name)
        self.closed = True
        return copy.deepcopy(
            {
                "request_id": self.request_id,
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
                "position_policy": "reset",
                "first_chunk_num_computed_tokens": self.first_computed_tokens,
                "first_fa_context": self.first_fa_context,
                "zero_fa_context_first_chunk": bool(self.first_fa_context)
                and all(
                    layer["context_tokens"] == 0
                    for layer in self.first_fa_context.values()
                ),
                "fa_kv_imported": False,
                "restored_initial_state": self.restore_receipt is not None,
                "restore": self.restore_receipt,
                "capture": self.capture_receipt,
                "restored_from": self.restore_name,
                "snapshot": self.capture_receipt,
                "returned_final_sample_is_not_consumed": True,
                "scope": "native_state_boundary_and_serial_continuation",
                "total_cpu_snapshot_bytes": self.store.list()[
                    "total_cpu_snapshot_bytes"
                ],
            }
        )


@dataclass(frozen=True)
class ForwardBoundary:
    """The exact operation and allocated slots belonging to one model forward."""

    operation: BoundaryOperation
    query_tokens: int
    slots: list[StateSlot]


class NativeCompactionController:
    """Optional state operations invoked directly by GPUModelRunner.

    Created by Worker.cc_install after cache initialization. No model method is
    replaced, and an unarmed controller leaves ordinary forwards untouched.
    """

    def __init__(self, runner: GPUModelRunner):
        validate_configuration(runner.vllm_config)
        if type(runner).__module__ != "vllm.v1.worker.gpu_model_runner":
            raise CompactionContractError("Only V1 GPUModelRunner is supported")

        from vllm.model_executor.layers.mamba.abstract import MambaBase
        from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
        from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadata
        from vllm.v1.kv_cache_interface import MambaSpec

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
        self.operation: BoundaryOperation | None = None
        self.metadata_types = {
            "gdn": GDNAttentionMetadata,
            "mamba2": Mamba2AttentionMetadata,
        }

    def info(self) -> dict:
        return {
            "backend": "vllm_native",
            "implementation": ("vllm.v1.worker.compaction.NativeCompactionController"),
            "recurrent_layers": len(self.layers),
            "fa_layers": len(self.fa_layers),
            "families": sorted({family for family, _ in self.layers.values()}),
            "native_state_bytes_per_request": sum(
                state[0].numel() * state.element_size()
                for _, layer in self.layers.values()
                for state in layer.kv_cache
            ),
            "position_policy": "reset",
            "snapshot_storage": "cpu",
            "kv_ownership": "vllm_scheduler",
            "prefix_caching": False,
            "scope": "native_state_boundary_and_serial_continuation",
        }

    def arm(
        self,
        capture_name: str | None = None,
        restore_name: str | None = None,
        expected_prompt_tokens: int | None = None,
        start_cursor: int = 0,
        expected_request_id: str | None = None,
    ) -> dict:
        if self.operation is not None and not self.operation.closed:
            raise CompactionContractError(
                "Call cc_result after the previous request before arming"
            )
        if self.runner.execute_model_state is not None:
            raise CompactionContractError("Cannot arm between forward and sampling")
        self.operation = BoundaryOperation(
            self.store,
            capture_name=capture_name,
            restore_name=restore_name,
            expected_prompt_tokens=expected_prompt_tokens,
            start_cursor=start_cursor,
            expected_request_id=expected_request_id,
        )
        return {
            "armed": True,
            "capture_name": capture_name,
            "restore_name": restore_name,
            "expected_prompt_tokens": expected_prompt_tokens,
            "start_cursor": start_cursor,
            "expected_request_id": expected_request_id,
        }

    def result(self) -> dict:
        if self.operation is None:
            raise CompactionContractError("No operation has been armed")
        return {**self.info(), **self.operation.result()}

    def snapshots(self) -> dict:
        return self.store.list()["snapshots"]

    def compare(self, name_a: str, name_b: str) -> dict:
        return self.store.compare(name_a, name_b)

    def drop(self, name: str) -> dict:
        active = self.operation
        if (
            active is not None
            and not active.closed
            and name in (active.restore_name, active.capture_name)
        ):
            raise CompactionContractError("Cannot drop an active operation's snapshot")
        self.store.drop(name)
        return {"dropped": name, **self.store.list()}

    def before_forward(
        self, metadata: Any, positions: torch.Tensor | None
    ) -> ForwardBoundary | None:
        operation = self.operation
        if operation is None or operation.closed:
            return None
        try:
            runner = self.runner
            batch = runner.input_batch
            if batch.num_reqs != 1:
                raise CompactionContractError("Exactly one request must be scheduled")
            request_id = batch.req_ids[0]
            request = runner.requests[request_id]
            if (
                request.mm_features
                or request.prompt_embeds is not None
                or request.lora_request
            ):
                raise CompactionContractError(
                    "Only unadapted text-token requests are supported"
                )
            if not isinstance(metadata, dict):
                raise CompactionContractError(
                    "Expected eager per-layer attention metadata"
                )
            slots = []
            token_counts = set()
            for name, (family, layer) in self.layers.items():
                current = metadata.get(name)
                if not isinstance(current, self.metadata_types[family]):
                    raise CompactionContractError(
                        f"Unexpected metadata type for {name}"
                    )
                slots.append(resolve_slot(name, family, layer.kv_cache, current))
                token_counts.add(
                    int(current.num_prefill_tokens + current.num_decode_tokens)
                )
            if len(token_counts) != 1:
                raise CompactionContractError(
                    "Recurrent layers disagree on processed token count"
                )
            query_tokens = token_counts.pop()
            computed = int(batch.num_computed_tokens_cpu[0])
            if int(request.num_computed_tokens) != computed:
                raise CompactionContractError(
                    "Worker request and batch cursors disagree"
                )
            if positions is None:
                raise CompactionContractError("Forward positions were not supplied")
            expected_positions = torch.arange(
                computed,
                computed + query_tokens,
                dtype=positions.dtype,
                device=positions.device,
            )
            if positions.shape[-1] != query_tokens or not bool(
                (positions == expected_positions).all()
            ):
                raise CompactionContractError("Only reset text positions are supported")
            fa_context = verify_fa_context(
                metadata,
                self.fa_layers,
                query_tokens=query_tokens,
                computed_tokens=computed,
            )
            operation.before(
                request_id,
                request.num_prompt_tokens,
                computed,
                query_tokens,
                slots,
                fa_context,
            )
            return ForwardBoundary(operation, query_tokens, slots)
        except Exception as error:
            self.fail_forward(error)
            raise

    def after_forward(self, boundary: ForwardBoundary | None) -> None:
        if boundary is None:
            return
        try:
            if boundary.operation is not self.operation:
                raise CompactionContractError(
                    "Compaction operation changed mid-forward"
                )
            boundary.operation.after(boundary.query_tokens, boundary.slots)
        except Exception as error:
            self.fail_forward(error)
            raise

    def fail_forward(self, error: Exception) -> None:
        if self.operation is not None and not self.operation.closed:
            self.operation.failed = f"{type(error).__name__}: {error}"
