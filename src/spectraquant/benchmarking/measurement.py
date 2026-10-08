"""Benchmarking API: memory accounting and latency measurement.

Measurement classes 4 (kernel-backed inference) and 5 (end-to-end service) are **unavailable** on
this workstation: there is no CUDA device and no supported low-bit kernel (``AGENTS.md`` section 2,
``docs/research/environment.md``). Any latency measurement would therefore be a class-2 float
number dressed up as a speed claim, which section 4.4 forbids. The functions below raise instead of
producing such a number.

Planned owner: benchmarking stream (wave 2).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from torch import nn

__all__ = ["analytical_memory_model", "measure_inference_latency", "measure_packed_size"]


def analytical_memory_model(
    layer_shapes: Mapping[str, Sequence[int]],
    *,
    bits: Mapping[str, int],
    ranks: Mapping[str, int] | None = None,
) -> Mapping[str, Any]:
    """Class-1 analytical memory model: bits per parameter and total bytes from shapes alone.

    Raises:
        NotImplementedError: Milestone 2 has not landed (quantization/benchmarking streams).
    """
    raise NotImplementedError(
        "analytical_memory_model is a Milestone 2 deliverable; the bootstrap scaffold ships no "
        "memory model"
    )


def measure_packed_size(path: str) -> int:
    """Measured (class 3) size in bytes of a serialized compressed checkpoint.

    Raises:
        NotImplementedError: Milestone 2 has not landed (quantization stream).
    """
    raise NotImplementedError("measure_packed_size is a Milestone 2 deliverable")


def measure_inference_latency(
    model: nn.Module,
    *,
    batch_size: int,
    seq_len: int,
    warmup: int = 5,
    repetitions: int = 20,
    backend: str = "torch-cpu-fp32",
) -> Mapping[str, float]:
    """Measure inference latency (class 4/5).

    Raises:
        NotImplementedError: always. Latency claims require a real supported low-bit kernel and a
            controlled protocol, neither of which exists on this CPU-only workstation. Reporting a
            number here would be a fabricated speed claim.
    """
    raise NotImplementedError(
        "measure_inference_latency requires measurement class 4/5 (a real low-bit kernel); this "
        "workstation has no CUDA device and no supported low-bit kernel, so no latency number may "
        "be produced. See AGENTS.md sections 2 and 4.4 and ADR-0002"
    )
