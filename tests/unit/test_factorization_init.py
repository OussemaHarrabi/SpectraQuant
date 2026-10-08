"""Tests for the Milestone-2 factor-initialization strategies."""

from __future__ import annotations

import pytest
import torch

from spectraquant.factorization.decomposition import (
    LowRankFactors,
    reconstruction_error,
    truncated_svd,
)
from spectraquant.factorization.initialization import (
    initialize_random,
    initialize_svd,
    initialize_svd_residual,
)


def test_initialize_svd_matches_truncated_svd() -> None:
    w = torch.randn(30, 12, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    plain = initialize_svd(w, 4)
    direct = truncated_svd(w, 4)

    assert torch.equal(plain.A, direct.A)
    assert torch.equal(plain.B, direct.B)
    assert plain.reconstruction_error == direct.reconstruction_error


def test_initialize_svd_residual_splits_weight_exactly() -> None:
    w = torch.randn(30, 12, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    factors, residual = initialize_svd_residual(w, 4)

    assert isinstance(factors, LowRankFactors)
    assert residual.shape == w.shape
    assert residual.dtype == w.dtype
    assert torch.allclose(factors.reconstruct() + residual, w, atol=1e-12)


def test_initialize_svd_residual_norm_equals_the_recorded_error() -> None:
    w = torch.randn(20, 15, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    factors, residual = initialize_svd_residual(w, 3)

    assert float(torch.linalg.matrix_norm(residual.to(torch.float64), ord="fro")) == pytest.approx(
        factors.reconstruction_error, rel=1e-9
    )
    assert factors.reconstruction_error == pytest.approx(
        reconstruction_error(w, factors), rel=1e-12
    )


def test_initialize_svd_residual_captures_the_dominant_subspace() -> None:
    """The residual of a near-rank-r matrix is tiny in relative terms (the LoftQ first step)."""
    generator = torch.Generator().manual_seed(3)
    A0 = torch.randn(2, 20, generator=generator, dtype=torch.float64)
    B0 = torch.randn(25, 2, generator=generator, dtype=torch.float64)
    w = B0 @ A0
    factors, residual = initialize_svd_residual(w, 4, energy_keep=0.999)

    assert factors.rank >= 2
    assert factors.reconstruction_error < 1e-9
    assert float(residual.norm()) < 1e-9


def test_initialize_random_is_bit_identical_at_a_fixed_seed() -> None:
    w = torch.randn(30, 12, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
    first = initialize_random(w, 4, seed=7)
    second = initialize_random(w, 4, seed=7)

    assert torch.equal(first.A, second.A)
    assert torch.equal(first.B, second.B)
    assert first.reconstruction_error == second.reconstruction_error


def test_initialize_random_depends_on_the_seed() -> None:
    w = torch.randn(30, 12, dtype=torch.float64, generator=torch.Generator().manual_seed(5))

    assert not torch.equal(initialize_random(w, 4, seed=1).A, initialize_random(w, 4, seed=2).A)


def test_initialize_random_matches_the_weight_norm_in_expectation() -> None:
    """E ||B @ A||_F^2 == ||W||_F^2 for the default scale; check the sample mean over 40 seeds."""
    w = torch.randn(40, 24, dtype=torch.float64, generator=torch.Generator().manual_seed(6))
    target = float(torch.linalg.matrix_norm(w, ord="fro")) ** 2
    squared_norms = [
        float(torch.linalg.matrix_norm(initialize_random(w, 6, seed=seed).reconstruct(), ord="fro"))
        ** 2
        for seed in range(40)
    ]
    mean = sum(squared_norms) / len(squared_norms)

    assert mean == pytest.approx(target, rel=0.15)


def test_initialize_random_explicit_scale_is_used_verbatim() -> None:
    w = torch.zeros(10, 5, dtype=torch.float64)
    factors = initialize_random(w, 2, seed=0, scale=0.5)
    generator = torch.Generator().manual_seed(0)
    expected_A = torch.randn(2, 5, generator=generator, dtype=torch.float64) * 0.5

    assert torch.equal(factors.A, expected_A)


def test_initialize_random_on_the_zero_matrix_matches_the_zero_norm() -> None:
    w = torch.zeros(8, 6, dtype=torch.float64)
    factors = initialize_random(w, 3, seed=0)

    assert factors.rank == 3
    assert factors.reconstruction_error == 0.0


def test_initialize_random_rank_zero_is_empty() -> None:
    w = torch.randn(8, 6, dtype=torch.float64)
    factors = initialize_random(w, 0, seed=0)

    assert factors.rank == 0
    assert factors.reconstruction_error == pytest.approx(float(w.norm()), rel=1e-12)


def test_initialize_random_rejects_a_non_positive_scale() -> None:
    with pytest.raises(ValueError, match="scale"):
        initialize_random(torch.randn(4, 4), 2, seed=0, scale=0.0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_strategies_preserve_dtype(dtype: torch.dtype) -> None:
    w = torch.randn(20, 10, dtype=dtype, generator=torch.Generator().manual_seed(7))

    assert initialize_svd(w, 3).dtype is dtype
    assert initialize_svd_residual(w, 3)[1].dtype is dtype
    assert initialize_random(w, 3, seed=0).dtype is dtype
