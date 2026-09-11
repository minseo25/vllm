# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Research hook: emulated low-bit checkpoint quantization of recurrent state.

Scope (EffHybridAttn SSM state quantization study, 2026-09):

* The persistent recurrent state of a Mamba2 / Gated DeltaNet layer is kept in
  its configured dtype (FP32 for the reference lane).  At *flush* steps the
  state rows of the flushing requests are replaced in place by
  ``dequant(quant(state))``.  Between flushes the working state is untouched.
  With ``--use-replayssm`` the kernel only materializes the checkpoint at its
  own flush steps, so quantizing exactly those rows reproduces
  "ReplaySSM + low-bit checkpoint" semantics.  Without ReplaySSM the same
  schedule is the flush-only dense emulation of that scheme.
* This is *fake* quantization: no packed storage, no bandwidth or latency claim.
* Decode-update index ``t`` counts recurrent updates after prompt end
  (``t = 0`` is the prefill-end checkpoint).  A row flushes when
  ``t > 0 and t % window == 0``; the prefill-end checkpoint is quantized once
  (``Q0``) when ``q0`` is enabled.  Update -> readout -> quantize/store order is
  preserved because the hooks run after the decode kernel has produced its
  output from the pre-quantization state.
* Quantizer: symmetric integer grid with ``qmax = 2**(bits-1) - 1``, one FP32
  absmax scale per head (``scale = absmax / qmax``; an all-zero head uses
  ``scale = 1``), round-to-nearest ties-to-even (``torch.round``) or
  stochastic rounding from an explicit ``torch.Generator``.  Values are
  clamped to ``[-qmax, qmax]`` after rounding.

The optional tracer dumps per-step native factors, kernel readouts and state
snapshots for a single request at a time so that frozen-factor analyses can
be run offline.  Tracing forces eager execution and batch size one.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_VALID_BITS = (4, 8)
_VALID_ROUNDING = ("rtn", "sr")


@dataclass(frozen=True)
class StateQuantSpec:
    """Fully specified emulated checkpoint quantizer + flush schedule."""

    bits: int
    window: int
    rounding: str = "rtn"
    seed: int = 0
    scale_axis: str = "head"
    q0: bool = True
    layers: frozenset[int] | None = None

    def __post_init__(self) -> None:
        if self.bits not in _VALID_BITS:
            raise ValueError(f"state_quant_bits must be one of {_VALID_BITS}")
        if self.window < 1:
            raise ValueError("state_quant_window must be >= 1")
        if self.rounding not in _VALID_ROUNDING:
            raise ValueError(f"state_quant_rounding must be one of {_VALID_ROUNDING}")
        if self.scale_axis != "head":
            raise ValueError("only per-head absmax scaling is implemented")

    @property
    def qmax(self) -> int:
        return 2 ** (self.bits - 1) - 1

    def applies_to_layer(self, layer_index: int) -> bool:
        return self.layers is None or layer_index in self.layers

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["layers"] = None if self.layers is None else sorted(self.layers)
        return data

    @classmethod
    def from_mamba_config(cls, mamba_config: Any) -> "StateQuantSpec | None":
        bits = getattr(mamba_config, "state_quant_bits", None)
        if bits is None:
            return None
        layers = getattr(mamba_config, "state_quant_layers", None)
        return cls(
            bits=int(bits),
            window=int(getattr(mamba_config, "state_quant_window", 1)),
            rounding=str(getattr(mamba_config, "state_quant_rounding", "rtn")),
            seed=int(getattr(mamba_config, "state_quant_seed", 0)),
            q0=bool(getattr(mamba_config, "state_quant_q0", True)),
            layers=None if layers is None else frozenset(int(x) for x in layers),
        )


def per_head_absmax_scale(x: torch.Tensor, qmax: int) -> torch.Tensor:
    """FP32 scale of shape ``x.shape[:2] + (1,) * (x.dim() - 2)``.

    ``x`` is ``[rows, heads, *state_dims]``; the scale is one absmax per
    (row, head).  A head whose state is entirely zero gets ``scale = 1`` so the
    codes (all zero) reconstruct exactly to zero.
    """
    absmax = x.detach().abs().reshape(x.shape[0], x.shape[1], -1).amax(dim=-1)
    scale = absmax.to(torch.float32) / float(qmax)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return scale.reshape(x.shape[0], x.shape[1], *([1] * (x.dim() - 2)))


def quantize_codes(
    x: torch.Tensor,
    spec: StateQuantSpec,
    generator: torch.Generator | None = None,
    scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(codes, scale)``; codes are int8 in ``[-qmax, qmax]``.

    ``x`` must be ``[rows, heads, *state_dims]``.  All arithmetic is FP32.
    """
    if x.dim() < 3:
        raise ValueError("expected [rows, heads, *state_dims]")
    xf = x.detach().to(torch.float32)
    if scale is None:
        scale = per_head_absmax_scale(xf, spec.qmax)
    y = xf / scale
    if spec.rounding == "rtn":
        r = torch.round(y)  # half-to-even, matching the protocol
    else:
        if generator is None:
            raise ValueError("stochastic rounding requires an explicit torch.Generator")
        u = torch.rand(y.shape, generator=generator, device=y.device, dtype=torch.float32)
        r = torch.floor(y + u)
    r = torch.clamp(r, -float(spec.qmax), float(spec.qmax))
    return r.to(torch.int8), scale


def dequantize_codes(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return codes.to(torch.float32) * scale


def quantize_dequantize(
    x: torch.Tensor,
    spec: StateQuantSpec,
    generator: torch.Generator | None = None,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fake-quantize ``x`` (``[rows, heads, *dims]``); returns ``x.dtype``."""
    codes, scale = quantize_codes(x, spec, generator, scale)
    return dequantize_codes(codes, scale).to(x.dtype)


def state_hooks_enabled(mamba_config: Any) -> bool:
    """True when either the quantizer or the tracer is requested."""
    return bool(
        getattr(mamba_config, "state_quant_bits", None) is not None
        or getattr(mamba_config, "state_trace_dir", None)
    )


def layer_index_from_prefix(prefix: str) -> int:
    """Model layer index from a parameter prefix like ``model.layers.7.mixer``.

    Mirrors ``vllm.model_executor.models.utils.extract_layer_index`` (first
    integer path component) without importing the models package.  Returns -1
    when the prefix carries no integer component.
    """
    ints = [int(p) for p in prefix.split(".") if p.isdigit()]
    if not ints:
        return -1
    return ints[0]


def compute_state_hook_schedule_cpu(
    common_attn_metadata: Any, num_decodes: int, num_prefills: int
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """CPU decode-update indices and prefill-end flags for one batch.

    Rows are assumed reordered as ``[decodes..., (padding)..., prefills...]``
    with prefills occupying the last ``num_prefills`` rows of ``num_reqs``.
    For every row ``t_after = num_computed + query_len - num_prompt`` is the
    number of decode updates after prompt end once this step is applied.

    Returns ``(t_d, prefill_end)``: ``t_d`` is int32 ``[num_decodes]`` (rows
    that are not real decodes, i.e. zero-length padding or still-prefilling
    rows, get ``-1``); ``prefill_end`` is int8 ``[num_prefills]`` and is 1 when
    the prefill row reaches its prompt end in this step.  Either is ``None``
    when the corresponding count is zero.
    """
    num_computed = common_attn_metadata._num_computed_tokens_cpu
    num_prompt = common_attn_metadata.num_prompt_tokens_cpu
    if num_computed is None or num_prompt is None:
        raise ValueError(
            "recurrent-state hooks need CPU computed/prompt token counts; async "
            "speculative decoding is not supported"
        )
    qsl = common_attn_metadata.query_start_loc_cpu
    query_lens = (qsl[1:] - qsl[:-1]).to(torch.int64)
    num_reqs = int(common_attn_metadata.num_reqs)
    n = min(num_reqs, query_lens.numel(), num_computed.numel(), num_prompt.numel())
    t_after = (
        num_computed[:n].to(torch.int64) + query_lens[:n] - num_prompt[:n].to(torch.int64)
    )
    t_d = None
    if num_decodes > 0:
        t = t_after[:num_decodes].clone()
        invalid = (query_lens[:num_decodes] <= 0) | (t < 0)
        t[invalid] = -1
        t_d = t.to(torch.int32)
    prefill_end = None
    if num_prefills > 0:
        start = num_reqs - num_prefills
        prefill_end = (t_after[start:num_reqs] >= 0).to(torch.int8)
    return t_d, prefill_end


def flush_mask_from_update_index(
    update_index: torch.Tensor, spec: StateQuantSpec
) -> torch.Tensor:
    """Boolean mask of rows to quantize given their decode-update index ``t``.

    ``t == 0`` is the prefill-end checkpoint (flushed iff ``q0``); ``t > 0``
    flushes when ``t % window == 0``; negative values mark invalid/padded rows.
    """
    t = update_index.to(torch.int64)
    positive = (t > 0) & (torch.remainder(t, spec.window) == 0)
    if spec.q0:
        return positive | (t == 0)
    return positive


class StateQuantizer:
    """Applies the emulated quantizer to selected rows of a state tensor."""

    def __init__(self, spec: StateQuantSpec, device: torch.device | str) -> None:
        self.spec = spec
        self._generator: torch.Generator | None = None
        if spec.rounding == "sr":
            self._generator = torch.Generator(device=device)
            self._generator.manual_seed(spec.seed)
        self.num_flushes = 0

    def apply_rows(
        self,
        state: torch.Tensor,
        rows: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> None:
        """In place: ``state[rows[i]] <- deq(q(state[rows[i]]))`` where ``mask[i]``.

        ``state`` is ``[slots, heads, *dims]``; ``rows`` is a 1-D slot index
        tensor.  Rows with ``mask == False`` are written back unchanged, so the
        operation is free of host synchronisation (CUDA-graph friendly for RTN).
        """
        rows = rows.reshape(-1).to(torch.long)  # cache indices arrive as int32
        sel = state.index_select(0, rows)
        deq = quantize_dequantize(sel, self.spec, self._generator)
        if mask is not None:
            view = mask.to(torch.bool).reshape(-1, *([1] * (sel.dim() - 1)))
            deq = torch.where(view, deq, sel)
        state.index_copy_(0, rows, deq)
        self.num_flushes += 1


# --------------------------------------------------------------------------- #
# Tracing (single request, eager only)
# --------------------------------------------------------------------------- #


def _sha256_of_tensor(t: torch.Tensor) -> str:
    raw = t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


@dataclass
class _LayerTraceBuffer:
    factors: dict[str, list[torch.Tensor]] = field(default_factory=dict)
    update_index: list[int] = field(default_factory=list)
    chunk_id: int = 0
    steps_in_chunk: int = 0


class StateTracer:
    """Dumps per-step native factors, readouts and state snapshots to disk.

    Layout (one directory per request, requests numbered in arrival order)::

        <root>/req0000/layer{L:02d}/factors_{chunk:04d}.safetensors
        <root>/req0000/layer{L:02d}/state_t{t:06d}.safetensors
        <root>/req0000/trace_manifest.json

    Factor tensors are stacked over steps along dim 0 and stored in their
    native dtype.  State snapshots are stored in the state dtype using the
    physical layout of the vLLM cache (``[heads, *dims]`` for one slot).
    """

    def __init__(
        self,
        root: str,
        snapshot_every: int = 256,
        max_steps: int = 8192,
        chunk_steps: int = 256,
        extra_snapshot_steps: tuple[int, ...] = (1, 2, 8, 16, 32, 64, 128),
    ) -> None:
        self.root = root
        self.snapshot_every = int(snapshot_every)
        self.max_steps = int(max_steps)
        self.chunk_steps = int(chunk_steps)
        self.extra_snapshot_steps = frozenset(int(x) for x in extra_snapshot_steps)
        self.request_index = -1
        self._layers: dict[int, _LayerTraceBuffer] = {}
        self._layer_meta: dict[int, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._manifest: dict[str, Any] = {}
        os.makedirs(root, exist_ok=True)

    # -- request lifecycle -------------------------------------------------- #
    def _request_dir(self) -> str:
        return os.path.join(self.root, f"req{self.request_index:04d}")

    def begin_request(self, meta: dict[str, Any] | None = None) -> None:
        with self._lock:
            if self.request_index >= 0 and not self._manifest.get("complete"):
                self._finalize_locked()
            self.request_index += 1
            self._layers = {}
            self._layer_meta = {}
            self._manifest = {
                "schema": "ssmq-vllm-trace-v1",
                "request_index": self.request_index,
                "started_unix": time.time(),
                "snapshot_every": self.snapshot_every,
                "chunk_steps": self.chunk_steps,
                "layers": {},
                "meta": meta or {},
            }
            os.makedirs(self._request_dir(), exist_ok=True)
            self._write_manifest()

    def end_request(self) -> None:
        """Flush the active request.  vLLM gives layers no end-of-request
        signal and the engine-core process may exit without running atexit
        handlers, so callers should issue a tiny sentinel request after the
        last traced request: its prefill finalizes the previous trace."""
        with self._lock:
            if self.request_index >= 0 and not self._manifest.get("complete"):
                self._finalize_locked()

    def _finalize_locked(self) -> None:
        self._flush_all_chunks()
        self._manifest["ended_unix"] = time.time()
        self._manifest["complete"] = True
        self._write_manifest()

    def _write_manifest(self) -> None:
        path = os.path.join(self._request_dir(), "trace_manifest.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._manifest, f, indent=1, sort_keys=True, default=str)
        os.replace(tmp, path)

    # -- recording ---------------------------------------------------------- #
    def should_snapshot(self, t: int) -> bool:
        if t == 0 or t in self.extra_snapshot_steps:
            return True
        return self.snapshot_every > 0 and t % self.snapshot_every == 0

    def record_layer_meta(self, layer: int, **meta: Any) -> None:
        if layer not in self._layer_meta:
            self._layer_meta[layer] = dict(meta)
            self._manifest["layers"][str(layer)] = self._layer_meta[layer]

    def record_constants(self, layer: int, tensors: dict[str, torch.Tensor],
                         **meta: Any) -> None:
        """Store per-layer constant parameters once per trace root."""
        from safetensors.torch import save_file

        cdir = os.path.join(self.root, "constants")
        os.makedirs(cdir, exist_ok=True)
        path = os.path.join(cdir, f"layer{layer:02d}.safetensors")
        if os.path.exists(path):
            return
        cpu = {k: v.detach().contiguous().cpu() for k, v in tensors.items()}
        save_file(cpu, path, metadata={"layer": str(layer),
                                       **{k: str(v) for k, v in meta.items()}})
        with open(os.path.join(cdir, f"layer{layer:02d}.json"), "w") as f:
            json.dump({"layer": layer, "meta": meta,
                       "tensors": {k: {"dtype": str(v.dtype), "shape": list(v.shape),
                                       "sha256": _sha256_of_tensor(v)}
                                   for k, v in cpu.items()}}, f, indent=1, default=str)

    def record_state(self, layer: int, t: int, state_row: torch.Tensor,
                     suffix: str = "") -> None:
        """Store one slot's state (``[heads, *dims]``) as ``state_t{t}{suffix}``.

        ``suffix=""`` is the pre-quantization state after update ``t`` (the
        state the readout of token ``t`` was produced from); ``"_post"`` is the
        persistent state after the emulated quantizer ran at a flush.
        """
        from safetensors.torch import save_file

        ldir = os.path.join(self._request_dir(), f"layer{layer:02d}")
        os.makedirs(ldir, exist_ok=True)
        cpu = state_row.detach().contiguous().cpu()
        path = os.path.join(ldir, f"state_t{t:06d}{suffix}.safetensors")
        save_file({"state": cpu}, path, metadata={"t": str(t), "layer": str(layer),
                                                  "kind": "post_quant" if suffix else "pre_quant",
                                                  "dtype": str(cpu.dtype),
                                                  "shape": json.dumps(list(cpu.shape))})
        lm = self._manifest["layers"].setdefault(str(layer), {})
        lm.setdefault("state_snapshots", []).append(
            {"t": t, "kind": "post_quant" if suffix else "pre_quant",
             "file": os.path.relpath(path, self._request_dir()),
             "sha256": _sha256_of_tensor(cpu), "dtype": str(cpu.dtype),
             "shape": list(cpu.shape)})

    def record_step(self, layer: int, t: int, factors: dict[str, torch.Tensor]) -> None:
        """Append one decode step's factors (each ``[...]`` for one request)."""
        if t > self.max_steps:
            return
        buf = self._layers.setdefault(layer, _LayerTraceBuffer())
        for name, val in factors.items():
            buf.factors.setdefault(name, []).append(val.detach().contiguous().cpu())
        buf.update_index.append(int(t))
        buf.steps_in_chunk += 1
        if buf.steps_in_chunk >= self.chunk_steps:
            self._flush_chunk(layer, buf)

    def _flush_chunk(self, layer: int, buf: _LayerTraceBuffer) -> None:
        from safetensors.torch import save_file

        if buf.steps_in_chunk == 0:
            return
        ldir = os.path.join(self._request_dir(), f"layer{layer:02d}")
        os.makedirs(ldir, exist_ok=True)
        tensors = {name: torch.stack(vals, dim=0) for name, vals in buf.factors.items()}
        tensors["update_index"] = torch.tensor(buf.update_index, dtype=torch.int32)
        path = os.path.join(ldir, f"factors_{buf.chunk_id:04d}.safetensors")
        save_file(tensors, path, metadata={"layer": str(layer),
                                           "t_first": str(buf.update_index[0]),
                                           "t_last": str(buf.update_index[-1])})
        lm = self._manifest["layers"].setdefault(str(layer), {})
        lm.setdefault("factor_chunks", []).append(
            {"chunk": buf.chunk_id, "file": os.path.relpath(path, self._request_dir()),
             "t_first": buf.update_index[0], "t_last": buf.update_index[-1],
             "n": len(buf.update_index),
             "tensors": {k: {"dtype": str(v.dtype), "shape": list(v.shape)}
                         for k, v in tensors.items()}})
        buf.factors = {}
        buf.update_index = []
        buf.steps_in_chunk = 0
        buf.chunk_id += 1
        self._write_manifest()

    def _flush_all_chunks(self) -> None:
        for layer, buf in list(self._layers.items()):
            self._flush_chunk(layer, buf)


# --------------------------------------------------------------------------- #
# Process-wide registry (one tracer / quantizer set per worker process)
# --------------------------------------------------------------------------- #

_tracer: StateTracer | None = None
_tracer_lock = threading.Lock()


def get_tracer(mamba_config: Any) -> StateTracer | None:
    """Return the process-wide tracer configured by ``mamba_config``."""
    global _tracer
    root = getattr(mamba_config, "state_trace_dir", None)
    if not root:
        return None
    with _tracer_lock:
        if _tracer is None:
            _tracer = StateTracer(
                root,
                snapshot_every=getattr(mamba_config, "state_trace_snapshot_every", 256),
                max_steps=getattr(mamba_config, "state_trace_max_steps", 8192),
            )
            import atexit

            atexit.register(_tracer.end_request)
            logger.info("SSM state tracer enabled at %s", root)
    return _tracer


class LayerStateHook:
    """Per-layer glue used by the Mamba2 and GDN layers.

    Created at layer construction from ``vllm_config.mamba_config``.  Holds the
    quantizer (if enabled) and the tracer handle (if enabled); ``enabled`` is
    False when neither is requested so the hot path pays nothing.
    """

    def __init__(self, mamba_config: Any, layer_index: int, family: str,
                 device: torch.device | str | None = None) -> None:
        self.layer_index = int(layer_index)
        self.family = family
        spec = StateQuantSpec.from_mamba_config(mamba_config)
        if spec is not None and not spec.applies_to_layer(self.layer_index):
            spec = None
        self.spec = spec
        self.quantizer: StateQuantizer | None = None
        self._device = device
        self.tracer = get_tracer(mamba_config)
        self._first_request_seen = False

    @property
    def enabled(self) -> bool:
        return self.spec is not None or self.tracer is not None

    def _ensure_quantizer(self, device: torch.device) -> StateQuantizer | None:
        if self.spec is None:
            return None
        if self.quantizer is None:
            self.quantizer = StateQuantizer(self.spec, device)
        return self.quantizer

    # -- prefill end (t = 0) ---------------------------------------------- #
    def on_prefill_end(self, state: torch.Tensor, rows: torch.Tensor,
                       prefill_end_mask: torch.Tensor | None) -> None:
        """Called after the prefill kernel wrote ``state[rows]``.

        ``prefill_end_mask[i]`` says whether row ``i`` completed its prompt in
        this step (chunked prefill may store intermediate states, which are not
        checkpoints of the study and are neither quantized nor traced).
        """
        if self.tracer is not None:
            self._trace_prefill_end(state, rows, prefill_end_mask)
        q = self._ensure_quantizer(state.device)
        if q is not None and self.spec is not None and self.spec.q0:
            q.apply_rows(state, rows, prefill_end_mask)

    def _trace_prefill_end(self, state, rows, prefill_end_mask) -> None:
        assert self.tracer is not None
        rows = rows.reshape(-1)
        if rows.numel() != 1:
            raise RuntimeError("state tracing requires max_num_seqs=1 (one prefill row)")
        done = True if prefill_end_mask is None else bool(prefill_end_mask.reshape(-1)[0].item())
        if not done:
            return
        if self.layer_index == self._first_layer_marker():
            self.tracer.begin_request()
        self.tracer.record_layer_meta(self.layer_index, family=self.family,
                                      state_dtype=str(state.dtype),
                                      state_shape_per_slot=list(state.shape[1:]))
        self.tracer.record_state(self.layer_index, 0, state[rows[0]])

    _first_layer_index: int | None = None

    def _first_layer_marker(self) -> int:
        # The tracer must start a new request exactly once per prefill.  The
        # lowest layer index that registers itself acts as the marker.
        if LayerStateHook._first_layer_index is None or \
                self.layer_index < LayerStateHook._first_layer_index:
            LayerStateHook._first_layer_index = self.layer_index
        return LayerStateHook._first_layer_index

    # -- decode step -------------------------------------------------------- #
    def record_constants(self, tensors: dict[str, torch.Tensor], **meta: Any) -> None:
        """Store constant per-layer parameters once (tracer only)."""
        if self.tracer is not None:
            self.tracer.record_constants(self.layer_index, tensors, **meta)

    def on_decode_step(self, state: torch.Tensor, rows: torch.Tensor,
                       update_index: torch.Tensor | None,
                       factors: dict[str, torch.Tensor] | None = None,
                       kernel_flush_mask: torch.Tensor | None = None) -> None:
        """Called after the decode kernel updated ``state[rows]`` and produced
        this token's readout.  ``update_index[i]`` is the decode-update index
        ``t >= 1`` of row ``i`` after this step (``< 0`` for padded rows).

        ``kernel_flush_mask`` (ReplaySSM ``is_flush_d``) lets trace runs verify
        that the kernel's own flush schedule coincides with ``t % W == 0``.
        """
        if update_index is None:
            raise RuntimeError("state quantization/tracing needs decode_update_index metadata")
        rows = rows.reshape(-1)
        update_index = update_index.reshape(-1)
        n = min(rows.numel(), update_index.numel())
        rows = rows[:n]
        update_index = update_index[:n]
        snapshot_t = None
        if self.tracer is not None and factors is not None:
            snapshot_t = self._trace_decode(state, rows, update_index, factors)
        q = self._ensure_quantizer(state.device)
        if q is not None and self.spec is not None:
            mask = flush_mask_from_update_index(update_index, self.spec)
            if kernel_flush_mask is not None and self.tracer is not None:
                kmask = kernel_flush_mask.reshape(-1)[:n].to(torch.bool)
                if not bool(torch.equal(kmask, mask)):
                    raise RuntimeError(
                        "ReplaySSM flush schedule disagrees with state_quant_window "
                        f"schedule: kernel={kmask.tolist()} quant={mask.tolist()} "
                        f"t={update_index.tolist()}"
                    )
            q.apply_rows(state, rows, mask)
            if snapshot_t is not None and bool(mask[0].item()):
                assert self.tracer is not None
                self.tracer.record_state(self.layer_index, snapshot_t,
                                         state[rows[0]], suffix="_post")

    def _trace_decode(self, state, rows, update_index, factors) -> int | None:
        assert self.tracer is not None
        if rows.numel() != 1:
            raise RuntimeError("state tracing requires max_num_seqs=1 (one decode row)")
        t = int(update_index[0].item())
        if t < 0:
            return None
        self.tracer.record_step(self.layer_index, t,
                                {k: v[0] if v.dim() > 0 and v.shape[0] == 1 else v
                                 for k, v in factors.items()})
        if self.tracer.should_snapshot(t):
            # Pre-quantization state after this update (the readout was produced
            # from exactly this state).
            self.tracer.record_state(self.layer_index, t, state[rows[0]])
            return t
        return None
