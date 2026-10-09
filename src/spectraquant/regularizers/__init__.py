"""Regularizers package (Milestone 5 — implemented).

Responsibilities: the rounding-aware spectral preparation objective that couples low-rank
preparation to the quantization grid (hypothesis H3), plus the retained spectral / orthogonality
penalties and the effective-rank measurement.

All numbers produced here are measurement class 2 (``AGENTS.md`` section 5): float arithmetic that
simulates quantization numerics. Nothing here is low-bit storage or accelerated inference.

See :mod:`spectraquant.regularizers.spectral` for which terms of the pre-registered objective are
actually needed and why.
"""

from __future__ import annotations

from spectraquant.regularizers.spectral import (
    PreparationObjective,
    PreparationTerms,
    effective_rank,
    factorization_proxy,
    measured_factor_rounding_residual,
    measured_product_rounding_residual,
    orthogonality_penalty,
    rounding_grid_penalty,
    rounding_residual_ratio,
    spectral_penalty,
    spectral_tail_energy,
)

__all__ = [
    "PreparationObjective",
    "PreparationTerms",
    "effective_rank",
    "factorization_proxy",
    "measured_factor_rounding_residual",
    "measured_product_rounding_residual",
    "orthogonality_penalty",
    "rounding_grid_penalty",
    "rounding_residual_ratio",
    "spectral_penalty",
    "spectral_tail_energy",
]
