"""Factorization package (Milestone 2 — not implemented).

Planned responsibilities: truncated SVD, LoftQ-style residual-aware initialisation, spectral
summaries of weight matrices, and the reconstruction-error accounting that links low-rank
preparation to downstream quantization quality.

Every entry point currently raises ``NotImplementedError``; see
:mod:`spectraquant.factorization.decomposition`.
"""

from __future__ import annotations

from spectraquant.factorization.decomposition import (
    factorize_linear,
    reconstruction_error,
    truncated_svd,
)

__all__ = ["factorize_linear", "reconstruction_error", "truncated_svd"]
