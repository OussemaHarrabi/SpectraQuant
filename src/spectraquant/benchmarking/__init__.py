"""Benchmarking package (wave 2 — not implemented).

Planned responsibilities: analytical memory accounting (class 1), measured packed sizes (class 3)
and a latency/throughput harness for a GPU that is not yet available (classes 4-5).

Every entry point raises ``NotImplementedError``; see
:mod:`spectraquant.benchmarking.measurement`.
"""

from __future__ import annotations

from spectraquant.benchmarking.measurement import (
    analytical_memory_model,
    measure_inference_latency,
    measure_packed_size,
)

__all__ = [
    "analytical_memory_model",
    "measure_inference_latency",
    "measure_packed_size",
]
