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

Precision context (vLLM 0.27): the state tensor this hook rewrites has the
dtype selected by ``mamba_ssm_cache_dtype``; "auto" (the default) resolves to
the model dtype (BF16).  The EffHybridAttn study always runs with
``mamba_ssm_cache_dtype=float32`` (FP32 state, BF16 weights), the same setting
the ReplaySSM authors report; the ReplaySSM ring stores x/B in BF16 and dt in
FP32.  The hook itself is dtype-agnostic: it quantizes FP32 copies of the
selected rows and writes back in the state dtype.
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
# Scale grouping over one slot's state ``[heads, A, B]`` (physical layout):
#   head   one FP32 absmax scale per head (the protocol's naive quantizer)
#   dim1   one scale per (head, a): absmax over the last axis B
#          (Mamba2 [H,P,N]: per head_dim channel; GDN [HV,V,K]: per value dim)
#   dim2   one scale per (head, b): absmax over axis A
#          (Mamba2: per state index n; GDN: per key dim k)
#   rowcol two-axis scale r_a * c_b (row absmax, then column absmax of the
#          row-normalised matrix); our variant, not a published recipe
#   static calibrated (offline) scale table per layer, loaded from
#          ``state_quant_static_scales_dir/layer{L:02d}.safetensors`` (key
#          "scale", shape broadcastable to [heads, A, B]); this is how
#          Quamba2 stores its 8-bit cached SSM states (static grouped scales)
_VALID_SCALE_AXES = ("head", "dim1", "dim2", "rowcol", "static")


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
    static_scales_dir: str | None = None  # required when scale_axis == "static"
    # The prefill-end checkpoint Q0 is always encoded with RTN so that every
    # low-precision condition (RTN and SR windows alike) starts from the same
    # checkpoint, as in the offline sweep; SR only affects later flushes.
    q0_rounding: str = "rtn"
    # Integer range: "symmetric" = [-qmax, qmax] (the study's naive codec);
    # "twos_complement" = [-2^(b-1), 2^(b-1)-1] as in Quamba2's Python codec.
    int_range: str = "symmetric"

    def __post_init__(self) -> None:
        if self.bits not in _VALID_BITS:
            raise ValueError(f"state_quant_bits must be one of {_VALID_BITS}")
        if self.window < 1:
            raise ValueError("state_quant_window must be >= 1")
        if self.rounding not in _VALID_ROUNDING:
            raise ValueError(f"state_quant_rounding must be one of {_VALID_ROUNDING}")
        if self.scale_axis not in _VALID_SCALE_AXES:
            raise ValueError(f"state_quant_scale_axis must be one of {_VALID_SCALE_AXES}")
        if self.scale_axis == "static" and not self.static_scales_dir:
            raise ValueError("scale_axis='static' needs state_quant_static_scales_dir")
        if self.q0_rounding not in _VALID_ROUNDING:
            raise ValueError(f"q0_rounding must be one of {_VALID_ROUNDING}")
        if self.int_range not in ("symmetric", "twos_complement"):
            raise ValueError("int_range must be 'symmetric' or 'twos_complement'")

    @property
    def qmax(self) -> int:
        return 2 ** (self.bits - 1) - 1

    @property
    def qmin(self) -> int:
        return -self.qmax if self.int_range == "symmetric" else -(2 ** (self.bits - 1))

    def with_rounding(self, rounding: str) -> "StateQuantSpec":
        """Same quantizer with another rounding rule (used for the RTN Q0)."""
        if rounding == self.rounding:
            return self
        return StateQuantSpec(bits=self.bits, window=self.window, rounding=rounding, seed=self.seed,
                              scale_axis=self.scale_axis, q0=self.q0, layers=self.layers,
                              static_scales_dir=self.static_scales_dir, q0_rounding=self.q0_rounding,
                              int_range=self.int_range)

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
            scale_axis=str(getattr(mamba_config, "state_quant_scale_axis", "head")),
            q0=bool(getattr(mamba_config, "state_quant_q0", True)),
            layers=None if layers is None else frozenset(int(x) for x in layers),
            static_scales_dir=getattr(mamba_config, "state_quant_static_scales_dir", None),
            q0_rounding=str(getattr(mamba_config, "state_quant_q0_rounding", "rtn")),
            int_range=str(getattr(mamba_config, "state_quant_int_range", "symmetric")),
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


def _nonzero(scale: torch.Tensor) -> torch.Tensor:
    return torch.where(scale > 0, scale, torch.ones_like(scale))


def absmax_scale(x: torch.Tensor, qmax: int, axis: str) -> torch.Tensor:
    """Dynamic absmax scale for ``x`` = ``[rows, heads, A, B]`` under ``axis``.

    Returns an FP32 tensor broadcastable to ``x`` such that ``|x / scale| <= qmax``
    (up to FP32 rounding, which the caller clamps).  Zero groups get scale 1.
    """
    if axis == "head":
        return per_head_absmax_scale(x, qmax)
    if x.dim() != 4:
        raise ValueError("axis scaling needs [rows, heads, A, B] states")
    xf = x.detach().abs().to(torch.float32)
    if axis == "dim1":
        return _nonzero(xf.amax(dim=-1, keepdim=True) / float(qmax))
    if axis == "dim2":
        return _nonzero(xf.amax(dim=-2, keepdim=True) / float(qmax))
    if axis == "rowcol":
        r = _nonzero(xf.amax(dim=-1, keepdim=True))            # [rows, heads, A, 1]
        c = _nonzero((xf / r).amax(dim=-2, keepdim=True))      # [rows, heads, 1, B], <= 1
        return r * c / float(qmax)
    if axis == "static":
        raise ValueError("static scales must be passed explicitly (loaded per layer)")
    raise ValueError(f"unknown scale axis {axis!r}")


def load_static_scales(directory: str, layer_index: int, device: torch.device | str) -> torch.Tensor:
    """Load a calibrated per-layer scale table ``[heads, A, B]`` (broadcastable).

    File: ``<directory>/layer{L:02d}.safetensors`` with key ``scale`` (FP32).
    Values already include the ``/ qmax`` division (i.e. they are step sizes).
    """
    from safetensors.torch import load_file

    path = os.path.join(directory, f"layer{layer_index:02d}.safetensors")
    table = load_file(path)["scale"].to(torch.float32)
    if table.dim() != 3:
        raise ValueError(f"{path}: expected a 3-D [heads, A, B]-broadcastable scale, got {tuple(table.shape)}")
    return _nonzero(table).to(device)


def scale_numel_per_head(shape_ab: tuple[int, int], axis: str) -> int:
    """Number of scale values stored per head for a state of shape ``[A, B]``."""
    a, b = shape_ab
    return {"head": 1, "dim1": a, "dim2": b, "rowcol": a + b}[axis]


def sr_uniform_noise(shape: tuple[int, ...], seed: int, keys: torch.Tensor,
                     device: torch.device | str) -> torch.Tensor:
    """Counter-based uniform noise in [0, 1) for stochastic rounding.

    ``shape`` is ``[rows, heads, *dims]`` and ``keys`` an int64 tensor of
    shape ``[rows]`` (one key per row, e.g. ``layer * 2**32 + decode_update_index``).
    Each element's draw is a 32-bit integer hash of ``(seed, key, flat element
    index)``, so the noise is reproducible, independent of batch composition
    and of any RNG state, and safe under CUDA-graph capture (pure integer
    tensor arithmetic; no ``torch.Generator``).
    """
    rows = shape[0]
    per_row = 1
    for s in shape[1:]:
        per_row *= int(s)
    idx = torch.arange(per_row, device=device, dtype=torch.int64).view(1, -1)
    key = keys.to(device=device, dtype=torch.int64).view(-1, 1)
    mask32 = (1 << 32) - 1
    h = (idx * 0x9E3779B1 + (key & mask32) * 0x85EBCA77 + (seed & mask32) * 0xC2B2AE3D) & mask32
    # murmur3-style finalizer on 32-bit lanes
    h = (h ^ (h >> 16)) & mask32
    h = (h * 0x85EBCA6B) & mask32
    h = (h ^ (h >> 13)) & mask32
    h = (h * 0xC2B2AE35) & mask32
    h = (h ^ (h >> 16)) & mask32
    u = h.to(torch.float32) * (1.0 / 4294967296.0)
    return u.view(rows, *shape[1:])


def quantize_codes(
    x: torch.Tensor,
    spec: StateQuantSpec,
    generator: torch.Generator | None = None,
    scale: torch.Tensor | None = None,
    sr_keys: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(codes, scale)``; codes are int8 in ``[qmin, qmax]``.

    ``x`` must be ``[rows, heads, *state_dims]`` (``[rows, heads, A, B]`` for
    axis scaling).  All arithmetic is FP32; ``scale`` follows ``spec.scale_axis``
    unless given explicitly (fixed-grid experiments).  Stochastic rounding uses
    counter-based noise keyed by ``sr_keys`` (one int64 per row) and
    ``spec.seed``; a ``torch.Generator`` is accepted only as an offline
    fallback when no keys are given.
    """
    if x.dim() < 3:
        raise ValueError("expected [rows, heads, *state_dims]")
    xf = x.detach().to(torch.float32)
    if scale is None:
        scale = absmax_scale(xf, spec.qmax, spec.scale_axis)
    y = xf / scale
    if spec.rounding == "rtn":
        r = torch.round(y)  # half-to-even, matching the protocol
    else:
        if sr_keys is not None:
            u = sr_uniform_noise(tuple(y.shape), spec.seed, sr_keys, y.device)
        elif generator is not None:
            u = torch.rand(y.shape, generator=generator, device=y.device, dtype=torch.float32)
        else:
            raise ValueError("stochastic rounding needs sr_keys (in-situ) or a torch.Generator (offline)")
        r = torch.floor(y + u)
    r = torch.clamp(r, float(spec.qmin), float(spec.qmax))
    return r.to(torch.int8), scale


def dequantize_codes(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return codes.to(torch.float32) * scale


def quantize_dequantize(
    x: torch.Tensor,
    spec: StateQuantSpec,
    generator: torch.Generator | None = None,
    scale: torch.Tensor | None = None,
    sr_keys: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fake-quantize ``x`` (``[rows, heads, *dims]``); returns ``x.dtype``."""
    codes, scale = quantize_codes(x, spec, generator, scale, sr_keys)
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
        # Exactly the prompt end.  A prefill row with t_after > 0 is a
        # recomputation after preemption (prompt + generated tokens replayed as
        # one prefill); its stored state is exact and must not receive a new Q0.
        prefill_end = (t_after[start:num_reqs] == 0).to(torch.int8)
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

    def __init__(self, spec: StateQuantSpec, device: torch.device | str,
                 layer_index: int | None = None) -> None:
        self.spec = spec
        self.layer_index = -1 if layer_index is None else int(layer_index)
        self.static_scale: torch.Tensor | None = None
        if spec.scale_axis == "static":
            if layer_index is None or spec.static_scales_dir is None:
                raise ValueError("static scales need the layer index and a scales directory")
            # [heads, A, B] -> [1, heads, A, B] to broadcast over rows
            self.static_scale = load_static_scales(spec.static_scales_dir, layer_index, device).unsqueeze(0)
        self.num_flushes = 0
        self.num_clipped = 0

    def apply_rows(
        self,
        state: torch.Tensor,
        rows: torch.Tensor,
        mask: torch.Tensor | None,
        q0_mask: torch.Tensor | None = None,
        update_index: torch.Tensor | None = None,
    ) -> None:
        """In place: ``state[rows[i]] <- deq(q(state[rows[i]]))`` where ``mask[i]``.

        ``state`` is ``[slots, heads, *dims]``; ``rows`` is a 1-D slot index
        tensor.  Rows with ``mask == False`` are written back unchanged, so the
        operation is free of host synchronisation and CUDA-graph safe (the SR
        noise is counter-based, keyed by layer and decode-update index, so it
        does not depend on batch composition or RNG state).  Rows flagged in
        ``q0_mask`` (the prefill-end checkpoint) are encoded with
        ``spec.q0_rounding`` (RTN by default) instead of ``spec.rounding``.
        """
        rows = rows.reshape(-1).to(torch.long)  # cache indices arrive as int32
        sel = state.index_select(0, rows)
        sr_keys = None
        if self.spec.rounding == "sr":
            t = (update_index.reshape(-1)[: rows.numel()].to(torch.int64)
                 if update_index is not None else torch.zeros(rows.numel(), dtype=torch.int64, device=state.device))
            sr_keys = (self.layer_index & 0xFFFF) * (1 << 32) + torch.clamp(t, min=0)
        deq = quantize_dequantize(sel, self.spec, None, scale=self.static_scale, sr_keys=sr_keys)
        if q0_mask is not None and self.spec.q0_rounding != self.spec.rounding:
            deq_q0 = quantize_dequantize(sel, self.spec.with_rounding(self.spec.q0_rounding),
                                         None, scale=self.static_scale)
            view0 = q0_mask.to(torch.bool).reshape(-1, *([1] * (sel.dim() - 1)))
            deq = torch.where(view0, deq_q0, deq)
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
        snapshot_steps: tuple[int, ...] | None = None,
        record_factors: bool = True,
    ) -> None:
        self.root = root
        self.snapshot_every = int(snapshot_every)
        self.max_steps = int(max_steps)
        self.chunk_steps = int(chunk_steps)
        self.extra_snapshot_steps = frozenset(int(x) for x in extra_snapshot_steps)
        # Explicit anchor list (plus t = 0) overrides the periodic schedule.
        self.snapshot_steps = (
            None if snapshot_steps is None else frozenset(int(x) for x in snapshot_steps)
        )
        self.record_factors = bool(record_factors)
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
                "snapshot_steps": None if self.snapshot_steps is None else sorted(self.snapshot_steps),
                "record_factors": self.record_factors,
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
        if not os.path.isdir(self._request_dir()):
            # The runner removed this (sentinel) request's directory on purpose;
            # do not resurrect it at shutdown.
            self._layers = {}
            self._manifest["complete"] = True
            return
        self._flush_all_chunks()
        self._manifest["ended_unix"] = time.time()
        self._manifest["complete"] = True
        self._write_manifest()

    def _write_manifest(self) -> None:
        # The runner may already have removed a sentinel request's directory.
        os.makedirs(self._request_dir(), exist_ok=True)
        path = os.path.join(self._request_dir(), "trace_manifest.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._manifest, f, indent=1, sort_keys=True, default=str)
        os.replace(tmp, path)

    # -- recording ---------------------------------------------------------- #
    def should_snapshot(self, t: int) -> bool:
        if t == 0:
            return True
        if self.snapshot_steps is not None:
            return t in self.snapshot_steps
        if t in self.extra_snapshot_steps:
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
        if not self.record_factors or t > self.max_steps:
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
            steps = getattr(mamba_config, "state_trace_snapshot_steps", None)
            _tracer = StateTracer(
                root,
                snapshot_every=getattr(mamba_config, "state_trace_snapshot_every", 256),
                max_steps=getattr(mamba_config, "state_trace_max_steps", 8192),
                snapshot_steps=None if steps is None else tuple(steps),
                record_factors=getattr(mamba_config, "state_trace_factors", True),
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
            self.quantizer = StateQuantizer(self.spec, device, layer_index=self.layer_index)
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
            all_q0 = torch.ones(rows.reshape(-1).numel(), dtype=torch.bool, device=state.device)
            q.apply_rows(state, rows, prefill_end_mask, q0_mask=all_q0)

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

        ``kernel_flush_mask`` (ReplaySSM ``is_flush_d``) marks the rows whose
        persistent state the kernel actually materialized this step.  With
        ReplaySSM the non-flush rows still hold the previous checkpoint, so
        state snapshots are only recorded at flush steps and trace runs verify
        that the kernel's schedule coincides with ``t % W == 0``.
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
            materialized = True
            if kernel_flush_mask is not None:
                materialized = bool(kernel_flush_mask.reshape(-1)[0].item())
                self.tracer.record_layer_meta(self.layer_index, family=self.family,
                                              state_dtype=str(state.dtype),
                                              state_shape_per_slot=list(state.shape[1:]),
                                              replay_state_snapshots_only_at_flush=True)
            # A one-token final prefill chunk is scheduled as a decode row and
            # reaches this hook with t == 0: treat it as the prefill-end
            # lifecycle (new request, t = 0 snapshot) instead of a decode step.
            if int(update_index[0].item()) == 0:
                self._trace_prefill_end(state, rows, None)
            else:
                snapshot_t = self._trace_decode(state, rows, update_index, factors,
                                                materialized)
        q = self._ensure_quantizer(state.device)
        if q is not None and self.spec is not None:
            mask = flush_mask_from_update_index(update_index, self.spec)
            q0_mask = update_index == 0
            if kernel_flush_mask is not None:
                # ReplaySSM materializes the checkpoint only at its own flush
                # rows (which re-anchor after preemption/resume), so the kernel
                # mask is the source of truth for what can be quantized.  The
                # t % W schedule is checked against it; a disagreement means
                # the run's conditions changed (e.g. a preempted request).
                if kernel_flush_mask.numel() < n:
                    raise RuntimeError(
                        f"kernel flush mask has {kernel_flush_mask.numel()} rows, hook has {n}"
                    )
                kmask = kernel_flush_mask.reshape(-1)[:n].to(torch.bool)
                # A t == 0 row (one-token final prefill chunk run as a decode
                # row) is a *forced* kernel flush that only materializes the
                # prefill-end checkpoint; whether it is quantized is the Q0
                # decision, not the window schedule.  Compare and follow the
                # kernel mask on decode rows (t > 0) only.
                decode_rows = update_index > 0
                if self.tracer is not None and not bool(
                        torch.equal(kmask & decode_rows, mask & decode_rows)):
                    raise RuntimeError(
                        "ReplaySSM flush schedule disagrees with state_quant_window "
                        f"schedule: kernel={kmask.tolist()} quant={mask.tolist()} "
                        f"t={update_index.tolist()}"
                    )
                mask = (kmask & decode_rows) | (q0_mask & mask)
            q.apply_rows(state, rows, mask, q0_mask=q0_mask, update_index=update_index)
            if snapshot_t is not None and bool(mask[0].item()):
                assert self.tracer is not None
                self.tracer.record_state(self.layer_index, snapshot_t,
                                         state[rows[0]], suffix="_post")

    def _trace_decode(self, state, rows, update_index, factors,
                      materialized: bool = True) -> int | None:
        assert self.tracer is not None
        if rows.numel() != 1:
            raise RuntimeError("state tracing requires max_num_seqs=1 (one decode row)")
        t = int(update_index[0].item())
        if t < 0:
            return None
        self.tracer.record_step(self.layer_index, t,
                                {k: v[0] if v.dim() > 0 and v.shape[0] == 1 else v
                                 for k, v in factors.items()})
        if materialized and self.tracer.should_snapshot(t):
            # Pre-quantization state after this update (the readout was produced
            # from exactly this state).
            self.tracer.record_state(self.layer_index, t, state[rows[0]])
            return t
        return None
