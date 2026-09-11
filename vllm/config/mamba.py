# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum, EnumMeta
from typing import Any, Literal, get_args

from pydantic import field_validator

from vllm.config.utils import config


class _MambaBackendEnumMeta(EnumMeta):
    """Metaclass for MambaBackendEnum to provide better error messages."""

    def __getitem__(cls, name: str):
        try:
            return super().__getitem__(name)
        except KeyError:
            valid = ", ".join(cls.__members__.keys())
            raise ValueError(
                f"Unknown Mamba SSU backend: '{name}'. Valid options are: {valid}"
            ) from None


class MambaBackendEnum(Enum, metaclass=_MambaBackendEnumMeta):
    """Enumeration of supported Mamba SSU (selective state update) backends."""

    TRITON = "triton"
    FLASHINFER = "flashinfer"
    CPU = "cpu"


MambaSSUAlgorithm = Literal["auto", "simple", "vertical", "horizontal"]


@config
class MambaConfig:
    """Configuration for Mamba SSM backends."""

    backend: MambaBackendEnum = MambaBackendEnum.TRITON
    """Mamba SSU backend to use."""

    enable_stochastic_rounding: bool = False
    """Enable stochastic rounding when writing SSM state to fp16 cache.
    Uses random bits to unbias the rounding error, which can improve
    numerical stability for long sequences."""
    stochastic_rounding_philox_rounds: int = 0
    """Number of Philox PRNG rounds for stochastic rounding random number
    generation. 0 uses the Triton default. Higher values improve randomness
    quality at the cost of compute."""

    ssu_algorithm: MambaSSUAlgorithm | None = None
    """Selective state update algorithm to use with the FlashInfer backend.
    None defaults to FlashInfer's "auto" algorithm. Forced algorithms must
    be supported by FlashInfer for the active GPU, state dtype, and decoding
    mode."""

    # ---- Research: emulated low-bit checkpoint quantization of the recurrent
    # state (Mamba2 / Gated DeltaNet).  See
    # vllm/model_executor/layers/mamba/state_quant.py.  Disabled unless
    # ``state_quant_bits`` is set.  This is fake quantization: the state stays
    # in its configured dtype and is replaced by dequant(quant(state)) at flush
    # steps; no packed storage, no speed claim.
    state_quant_bits: int | None = None
    """Integer bit width (4 or 8) of the emulated checkpoint quantizer; None
    disables the hook."""
    state_quant_window: int = 1
    """Flush window W: a request's checkpoint is (re)quantized after decode
    updates t = W, 2W, ... counted from prompt end.  With --use-replayssm this
    must equal --replayssm-buffer-len."""
    state_quant_rounding: Literal["rtn", "sr"] = "rtn"
    """Rounding: round-to-nearest ties-to-even, or stochastic rounding drawn
    from a dedicated generator seeded by ``state_quant_seed``."""
    state_quant_seed: int = 0
    """Seed of the stochastic-rounding generator (independent of sampling)."""
    state_quant_scale_axis: Literal["head", "dim1", "dim2", "rowcol", "static"] = "head"
    """Scale grouping over one slot's state [heads, A, B]: 'head' = one absmax
    scale per head (naive); 'dim1' = one per (head, a) over B (Mamba2: per
    head_dim channel; GDN: per value dim); 'dim2' = one per (head, b) over A
    (Mamba2: per state index; GDN: per key dim); 'rowcol' = two-axis r_a*c_b
    (our variant); 'static' = calibrated per-layer scale tables loaded from
    ``state_quant_static_scales_dir`` (Quamba2-style cached-state scales).
    Dynamic axes recompute the scale at every encoding."""
    state_quant_static_scales_dir: str | None = None
    """Directory with ``layer{L:02d}.safetensors`` (key ``scale``, FP32 step
    sizes broadcastable to [heads, A, B]) for ``state_quant_scale_axis='static'``."""
    state_quant_q0: bool = True
    """Also quantize the prefill-end checkpoint (t = 0) once."""
    state_quant_layers: list[int] | None = None
    """Restrict the hook to these model layer indices; None means every
    recurrent layer."""
    state_trace_dir: str | None = None
    """If set, dump per-step native recurrence factors, kernel readouts and
    state snapshots of each request to this directory (requires eager mode
    and max_num_seqs=1).  Works with or without quantization."""
    state_trace_snapshot_every: int = 256
    """Store a full state snapshot every this many decode updates (plus a
    fixed set of early anchors)."""
    state_trace_max_steps: int = 8192
    """Stop recording per-step factors after this many decode updates."""
    state_trace_snapshot_steps: list[int] | None = None
    """Explicit decode-update anchors at which to store full state snapshots
    (t = 0 is always stored).  Overrides ``state_trace_snapshot_every`` and the
    built-in early anchors when set, e.g. ``[32, 128, 256, 512, 1024, 2048,
    4096, 8192]``."""
    state_trace_factors: bool = True
    """Record per-step native factors and kernel readouts.  Set False for a
    state-snapshot-only trace (much smaller; no offline replay possible)."""

    @field_validator("backend", mode="before")
    @classmethod
    def validate_backend_before(cls, value: Any) -> Any:
        """Enable parsing of the `backend` enum type from string."""
        if isinstance(value, str):
            return MambaBackendEnum[value.upper()]
        return value

    def validate_ssu_algorithm(self) -> None:
        if self.ssu_algorithm is None:
            return
        valid_algorithms = get_args(MambaSSUAlgorithm)
        if self.ssu_algorithm not in valid_algorithms:
            valid = ", ".join(valid_algorithms)
            raise ValueError(
                f"Unknown Mamba SSU algorithm: '{self.ssu_algorithm}'. "
                f"Valid options are: {valid}"
            )
        if self.backend != MambaBackendEnum.FLASHINFER:
            raise ValueError(
                "Mamba SSU algorithm selection is only supported with the "
                "FlashInfer backend. Please set `--mamba-backend flashinfer`, "
                "or omit `--mamba-ssu-algorithm`."
            )

    def validate_state_quant(self) -> None:
        if self.state_quant_bits is not None and self.state_quant_bits not in (4, 8):
            raise ValueError("state_quant_bits must be 4 or 8 (or None to disable)")
        if self.state_quant_window < 1:
            raise ValueError("state_quant_window must be >= 1")
        if self.state_quant_rounding not in ("rtn", "sr"):
            raise ValueError("state_quant_rounding must be 'rtn' or 'sr'")
        if self.state_quant_scale_axis not in ("head", "dim1", "dim2", "rowcol", "static"):
            raise ValueError("state_quant_scale_axis must be head, dim1, dim2, rowcol or static")
        if self.state_quant_scale_axis == "static" and not self.state_quant_static_scales_dir:
            raise ValueError("state_quant_scale_axis='static' requires state_quant_static_scales_dir")
        if self.state_trace_snapshot_every < 0 or self.state_trace_max_steps < 0:
            raise ValueError("state_trace_* values must be non-negative")

    def __post_init__(self):
        self.validate_ssu_algorithm()
        self.validate_state_quant()
        if self.enable_stochastic_rounding:
            from vllm.platforms import current_platform

            if not current_platform.is_cuda():
                raise ValueError(
                    "Stochastic rounding for Mamba cache is only supported "
                    "on NVIDIA CUDA platforms. Please do not specify  "
                    "`--enable-mamba-cache-stochastic-rounding`."
                )
            if (
                self.backend == MambaBackendEnum.TRITON
                and not current_platform.is_device_capability_family(100)
            ):
                raise ValueError(
                    "Stochastic rounding for Mamba cache with triton backend requires "
                    "compute capability 10.0 (data center Blackwell). The `cvt.rs` "
                    "PTX instruction is not supported on your GPU. Please do not "
                    "specify `--enable-mamba-cache-stochastic-rounding`, "
                    "or set `--mamba-backend flashinfer`."
                )
