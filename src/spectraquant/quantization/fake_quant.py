"""Quantization API: fake quantization, dequantization and quality accounting.

Milestone 2 (owner: quantization stream). Fake quantization is measurement class 2 and must never
be reported as low-bit storage (class 3) or accelerated inference (class 4) — see ``AGENTS.md``
section 4.3. Nothing is implemented in the bootstrap scaffold.
"""

from __future__ import annotations

import torch

__all__ = ["dequantize", "fake_quantize", "quantization_error"]


def fake_quantize(
    tensor: torch.Tensor,
    *,
    bits: int,
    group_size: int | None = None,
    symmetric: bool = True,
    axis: int = -1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simulate uniform quantization in float precision (measurement class 2).

    Shapes:
        ``tensor``: ``(..., in_features)`` float tensor.
        returns ``(dequantized, scale, zero_point)`` where ``dequantized`` matches ``tensor``'s
        shape and dtype and the parameters carry the per-group ranges.

    Args:
        bits: target bit width (2, 3, 4 or 8).
        group_size: number of elements sharing one scale; ``None`` means per-tensor.
        symmetric: symmetric (zero_point == 0) or asymmetric quantization.
        axis: axis along which groups are formed.

    Raises:
        NotImplementedError: Milestone 2 has not landed (quantization stream).
    """
    raise NotImplementedError(
        "fake_quantize is a Milestone 2 deliverable (quantization stream); the bootstrap scaffold "
        "ships no quantization implementation"
    )


def dequantize(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    *,
    group_size: int | None = None,
    axis: int = -1,
) -> torch.Tensor:
    """Reconstruct float values from integer codes (inverse of :func:`fake_quantize`).

    Raises:
        NotImplementedError: Milestone 2 has not landed (quantization stream).
    """
    raise NotImplementedError("dequantize is a Milestone 2 deliverable (quantization stream)")


def quantization_error(
    reference: torch.Tensor,
    reconstructed: torch.Tensor,
    *,
    norm: str = "fro",
) -> dict[str, float]:
    """Error metrics between a float tensor and its quantized reconstruction.

    Raises:
        NotImplementedError: Milestone 2 has not landed (quantization stream).
    """
    raise NotImplementedError(
        "quantization_error is a Milestone 2 deliverable (quantization stream)"
    )
