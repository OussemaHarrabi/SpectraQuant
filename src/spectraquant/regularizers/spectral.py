"""Spectral and rank regularizers.

Milestone 5 (owner: regularizers/training stream). These penalties are how low-rank preparation and
low-bit training are coupled: they shape the spectrum of the weights that quantization then sees.
Not implemented in the bootstrap scaffold.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

__all__ = ["effective_rank", "orthogonality_penalty", "spectral_penalty"]


def spectral_penalty(
    model: nn.Module,
    *,
    target_rank: int,
    strength: float = 1.0,
    exclude: Iterable[str] = (),
) -> torch.Tensor:
    """Penalty that pushes the singular-value spectrum of selected weights towards ``target_rank``.

    Returns:
        Scalar tensor to add to the task loss.

    Raises:
        NotImplementedError: Milestone 5 has not landed (regularizers stream).
    """
    raise NotImplementedError(
        "spectral_penalty is a Milestone 5 deliverable (regularizers stream); the bootstrap "
        "scaffold ships no regularizer implementation"
    )


def orthogonality_penalty(
    factors: tuple[torch.Tensor, torch.Tensor],
    *,
    strength: float = 1.0,
) -> torch.Tensor:
    """Penalty encouraging orthonormal low-rank factors.

    Raises:
        NotImplementedError: Milestone 5 has not landed (regularizers stream).
    """
    raise NotImplementedError("orthogonality_penalty is a Milestone 5 deliverable")


def effective_rank(weight: torch.Tensor, *, tolerance: float = 1e-3) -> float:
    """Effective (entropy-based) rank of a weight matrix.

    Raises:
        NotImplementedError: Milestone 5 has not landed (regularizers stream).
    """
    raise NotImplementedError("effective_rank is a Milestone 5 deliverable (regularizers stream)")
