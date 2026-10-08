"""Low-rank factorization API.

Milestone 2 (owner: factorization stream) implements truncated SVD, LoftQ-style initialisation and
the reconstruction-error accounting used to *prepare* a model for quantization. Nothing is
implemented in the bootstrap scaffold: every entry point raises ``NotImplementedError`` so that no
caller can mistake an empty package for a working method (see ``AGENTS.md`` section 4.13).
"""

from __future__ import annotations

import torch

__all__ = [
    "factorize_linear",
    "reconstruction_error",
    "truncated_svd",
]


def truncated_svd(
    weight: torch.Tensor,
    *,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Truncated singular value decomposition of a 2-D weight matrix.

    Shapes:
        ``weight``: ``(out_features, in_features)`` float tensor.
        returns ``(A, B)`` with ``A`` of shape ``(out_features, rank)`` and ``B`` of shape
        ``(rank, in_features)`` such that ``A @ B`` approximates ``weight``.

    Raises:
        NotImplementedError: Milestone 2 has not landed (factorization stream).
    """
    raise NotImplementedError(
        "truncated_svd is a Milestone 2 deliverable (factorization stream); the bootstrap "
        "scaffold ships no factorization implementation"
    )


def factorize_linear(
    weight: torch.Tensor,
    *,
    rank: int,
    init: str = "svd",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Factorize one linear layer, optionally with a LoftQ-style residual-aware initialisation.

    Raises:
        NotImplementedError: Milestone 2 has not landed (factorization stream).
    """
    raise NotImplementedError(
        "factorize_linear is a Milestone 2 deliverable (factorization stream)"
    )


def reconstruction_error(
    weight: torch.Tensor,
    factors: tuple[torch.Tensor, torch.Tensor],
    *,
    norm: str = "fro",
) -> float:
    """Relative reconstruction error of a factorization.

    Raises:
        NotImplementedError: Milestone 2 has not landed (factorization stream).
    """
    raise NotImplementedError(
        "reconstruction_error is a Milestone 2 deliverable (factorization stream)"
    )
