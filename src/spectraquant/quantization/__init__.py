"""Quantization package (Milestone 2 — not implemented).

Planned responsibilities: uniform fake quantization (class 2), int8/int4 packing with measured
byte counts (class 3), quantization-error metrics, and the QAT loop that jointly optimises ranks
and bit widths.

CUDA-only kernels (bitsandbytes, TorchAO CUDA paths, GPTQ/AWQ GPU kernels) are deliberately absent:
they are deferred behind a documented GPU prerequisite (ADR-0002). Any function that would require
CUDA must raise rather than fall back to different numerics.
"""

from __future__ import annotations

from spectraquant.quantization.fake_quant import (
    dequantize,
    fake_quantize,
    quantization_error,
)
from spectraquant.quantization.packing import pack_int4, pack_int8, unpack

__all__ = [
    "dequantize",
    "fake_quantize",
    "pack_int4",
    "pack_int8",
    "quantization_error",
    "unpack",
]
