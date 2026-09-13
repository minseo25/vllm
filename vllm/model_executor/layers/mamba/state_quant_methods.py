# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Method-search codecs in physical [slot, head, value, key] coordinates.

These return dense FP32 reconstructions. Packed-byte ledgers describe proposed
representations; this module does not allocate or measure packed state storage.
"""

import math

import torch

METHODS = (
    "native", "block32", "block16", "key_hadamard",
    "key_hadamard_identity", "head_budget", "head_budget_control", "residual4",
    "row_outlier1",
)


def rtn(x: torch.Tensor, qmax=7.0) -> torch.Tensor:
    """One symmetric dynamic scale along the last dimension, FP32 arithmetic."""
    xf = x.to(torch.float32)
    scale = xf.abs().amax(dim=-1, keepdim=True) / qmax
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    codes = torch.round(xf / scale)
    if isinstance(qmax, torch.Tensor):
        codes = torch.minimum(torch.maximum(codes, -qmax), qmax)
    else:
        codes = torch.clamp(codes, -qmax, qmax)
    return codes * scale


def hadamard(x: torch.Tensor) -> torch.Tensor:
    """Normalized Sylvester transform on the last axis, without padding."""
    n = x.shape[-1]
    if n < 1 or n & (n - 1):
        raise ValueError("key Hadamard needs a power-of-two key dimension")
    shape = x.shape
    y = x
    stride = 1
    while stride < n:
        pair = y.reshape(*shape[:-1], n // (2 * stride), 2, stride)
        a, b = pair[..., 0, :], pair[..., 1, :]
        y = torch.stack((a + b, a - b), dim=-2).reshape(shape)
        stride *= 2
    return y / math.sqrt(n)


def key_signs(x: torch.Tensor) -> torch.Tensor:
    """One fixed diagonal; no runtime RNG or rotation-seed search."""
    index = torch.arange(x.shape[-1], device=x.device, dtype=torch.int64)
    bit = ((index * 1103515245 + 12345) >> 16) & 1
    return (1 - 2 * bit).to(x.dtype)


def encode_decode(x: torch.Tensor, method: str,
                  table: torch.Tensor | None = None) -> torch.Tensor:
    """Apply a declared INT4 method, preserving the original state layout."""
    if x.ndim != 4:
        raise ValueError("method codecs require [slot, head, value, key]")
    xf = x.to(torch.float32)
    if method == "row_outlier1":
        if xf.shape[-1] != 128 or table is not None:
            raise ValueError("row_outlier1 requires K128 and no calibration table")
        # Argmax selects the first index on ties. The exception stays FP32.
        index = xf.abs().argmax(dim=-1, keepdim=True)
        exception = xf.gather(-1, index)
        remainder = xf.scatter(-1, index, 0.0)
        return rtn(remainder).scatter(-1, index, exception).to(x.dtype)
    if method in ("block32", "block16"):
        group = 32 if method == "block32" else 16
        if xf.shape[-1] % group:
            raise ValueError("key dimension must be divisible by block size")
        grouped = xf.reshape(*xf.shape[:-1], xf.shape[-1] // group, group)
        return rtn(grouped).reshape(xf.shape).to(x.dtype)
    if method in ("key_hadamard", "key_hadamard_identity"):
        signs = key_signs(xf)
        rotated = hadamard(xf * signs)
        coded = rotated if method.endswith("identity") else rtn(rotated)
        return (hadamard(coded) * signs).to(x.dtype)
    if method in ("head_budget", "head_budget_control"):
        if table is None or table.shape != (xf.shape[1],):
            raise ValueError("head budget requires one frozen bit selector per head")
        qmax = torch.where(table, 127.0, 7.0).reshape(1, -1, 1, 1)
        return rtn(xf, qmax).to(x.dtype)
    if method == "residual4":
        if table is None or table.shape != (xf.shape[1], xf.shape[-1], 4):
            raise ValueError("residual4 needs a frozen [head, key, 4] basis")
        base = rtn(xf)
        coefficients = torch.matmul(xf - base, table).to(torch.bfloat16)
        correction = torch.matmul(coefficients.to(torch.float32), table.transpose(-1, -2))
        return (base + correction).to(x.dtype)
    raise ValueError(f"unknown method codec {method!r}")
