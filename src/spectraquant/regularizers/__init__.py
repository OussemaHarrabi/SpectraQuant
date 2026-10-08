"""Regularizers package (Milestone 5 — not implemented).

Planned responsibilities: spectral/rank penalties, orthogonality penalties on low-rank factors, and
effective-rank measurement of trained weights.

Every entry point raises ``NotImplementedError``; see
:mod:`spectraquant.regularizers.spectral`.
"""

from __future__ import annotations

from spectraquant.regularizers.spectral import (
    effective_rank,
    orthogonality_penalty,
    spectral_penalty,
)

__all__ = ["effective_rank", "orthogonality_penalty", "spectral_penalty"]
