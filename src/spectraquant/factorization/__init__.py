"""Factorization and spectral analysis of weight matrices (Milestone 2).

Convention (frozen, load-bearing — stated once in
:mod:`spectraquant.factorization.decomposition`): ``W ~= B @ A`` with
``A: (rank, in_features)``, ``B: (out_features, rank)``, ``W: (out_features, in_features)``.

Public API:

* :mod:`spectraquant.factorization.decomposition` — :class:`LowRankFactors`,
  :func:`~spectraquant.factorization.decomposition.truncated_svd`,
  :func:`~spectraquant.factorization.decomposition.randomized_svd`,
  :func:`~spectraquant.factorization.decomposition.reconstruction_error`,
  :func:`~spectraquant.factorization.decomposition.factor_bytes`.
* :mod:`spectraquant.factorization.spectral` — :func:`~spectraquant.factorization.spectral.singular_values`,
  :func:`~spectraquant.factorization.spectral.effective_rank`,
  :func:`~spectraquant.factorization.spectral.stable_rank`,
  :func:`~spectraquant.factorization.spectral.spectral_summary`.
* :mod:`spectraquant.factorization.initialization` — the ``svd`` / ``svd_residual`` / ``random``
  factor-initialization strategies (LoftQ alternating init is deferred to Milestone 3).
"""

from __future__ import annotations

from spectraquant.factorization.decomposition import (
    LowRankFactors,
    factor_bytes,
    randomized_svd,
    reconstruction_error,
    truncated_svd,
)
from spectraquant.factorization.initialization import (
    initialize_random,
    initialize_svd,
    initialize_svd_residual,
)
from spectraquant.factorization.spectral import (
    effective_rank,
    singular_values,
    spectral_summary,
    stable_rank,
)

__all__ = [
    "LowRankFactors",
    "effective_rank",
    "factor_bytes",
    "initialize_random",
    "initialize_svd",
    "initialize_svd_residual",
    "randomized_svd",
    "reconstruction_error",
    "singular_values",
    "spectral_summary",
    "stable_rank",
    "truncated_svd",
]
