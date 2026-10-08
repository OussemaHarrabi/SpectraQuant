"""Tests for the randomized SVD: determinism, accuracy bound and degenerate inputs."""

from __future__ import annotations

import math

import pytest
import torch

from spectraquant.factorization.decomposition import (
    randomized_svd,
    reconstruction_error,
    truncated_svd,
)


def _halko_frobenius_bound(exact_error: float, rank: int, sketch: int) -> float:
    """Halko–Martinsson–Tropp Frobenius bound sqrt(1 + k/(l - k + 1)) * ||W - W_k||_F."""
    return math.sqrt(1.0 + rank / max(sketch - rank + 1, 1)) * exact_error


def test_randomized_factors_follow_the_frozen_convention() -> None:
    w = torch.randn(25, 9, dtype=torch.float64)
    factors = randomized_svd(w, 4, n_oversamples=3, n_iter=2, seed=0)

    assert factors.A.shape == (4, 9)
    assert factors.B.shape == (25, 4)
    assert factors.reconstruct().shape == w.shape


def test_randomized_svd_is_bit_identical_at_a_fixed_seed() -> None:
    w = torch.randn(30, 20, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    first = randomized_svd(w, 4, n_oversamples=5, n_iter=2, seed=1234)
    second = randomized_svd(w, 4, n_oversamples=5, n_iter=2, seed=1234)

    assert torch.equal(first.A, second.A)
    assert torch.equal(first.B, second.B)
    assert first.reconstruction_error == second.reconstruction_error


def test_different_seeds_change_the_factors() -> None:
    w = torch.randn(30, 20, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    first = randomized_svd(w, 4, n_oversamples=5, n_iter=2, seed=1)
    second = randomized_svd(w, 4, n_oversamples=5, n_iter=2, seed=2)

    assert not torch.equal(first.A, second.A)


def test_randomized_error_is_not_below_the_optimal_truncation_error() -> None:
    """Any rank-r factor has error >= the optimal rank-r SVD error; the sketch cannot beat it."""
    w = torch.randn(50, 30, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    exact = truncated_svd(w, 6)

    for oversamples, n_iter in ((0, 0), (5, 2), (10, 4)):
        randomized = randomized_svd(w, 6, n_oversamples=oversamples, n_iter=n_iter, seed=0)
        assert randomized.reconstruction_error >= exact.reconstruction_error - 1e-9


def test_randomized_error_stays_within_the_halko_frobenius_bound() -> None:
    """Measured randomized error vs the Halko et al. (2011) Frobenius bound on a fixed sweep.

    The worst measured ratio of ``randomized_error / bound`` over the sweep is 0.9396
    (see ``docs/research/spectral-notes.md`` §3); the bound is asserted here on the same fixed
    seed set, so the check is deterministic.
    """
    shapes = [(60, 40), (200, 120), (40, 200), (128, 128)]
    worst = 0.0
    for shape_index, shape in enumerate(shapes):
        for seed in range(8):
            w = torch.randn(
                *shape, dtype=torch.float64, generator=torch.Generator().manual_seed(seed)
            )
            for rank in (2, 5, 10, 20):
                if rank >= min(shape):
                    continue
                exact = truncated_svd(w, rank)
                for oversamples in (0, 5, 10):
                    sketch = min(rank + oversamples, min(shape))
                    for n_iter in (0, 1, 2, 4):
                        randomized = randomized_svd(
                            w, rank, n_oversamples=oversamples, n_iter=n_iter, seed=100 + seed
                        )
                        bound = _halko_frobenius_bound(exact.reconstruction_error, rank, sketch)
                        ratio = randomized.reconstruction_error / bound
                        worst = max(worst, ratio)
                        assert ratio <= 1.0, (
                            f"shape_index={shape_index} rank={rank} oversamples={oversamples} "
                            f"n_iter={n_iter} seed={seed}: ratio={ratio:.6f} > 1"
                        )
    assert worst < 1.0  # documented measured worst is 0.9396


def test_recommended_configuration_is_within_ten_percent_of_the_exact_error() -> None:
    """With oversampling and power iterations the sketch error tracks the exact error closely.

    Measured worst ratio for ``n_oversamples >= 5`` and ``n_iter >= 2`` over the sweep:
    1.0323 (``docs/research/spectral-notes.md`` §3).
    """
    worst = 0.0
    for shape in [(60, 40), (200, 120), (40, 200), (128, 128)]:
        for seed in range(8):
            w = torch.randn(
                *shape, dtype=torch.float64, generator=torch.Generator().manual_seed(seed)
            )
            for rank in (2, 5, 10, 20):
                if rank >= min(shape):
                    continue
                exact = truncated_svd(w, rank)
                for oversamples in (5, 10):
                    for n_iter in (2, 4):
                        randomized = randomized_svd(
                            w, rank, n_oversamples=oversamples, n_iter=n_iter, seed=100 + seed
                        )
                        ratio = randomized.reconstruction_error / max(
                            exact.reconstruction_error, 1e-300
                        )
                        worst = max(worst, ratio)
                        assert ratio < 1.10, (
                            f"shape={shape} rank={rank} oversamples={oversamples} "
                            f"n_iter={n_iter} seed={seed}: ratio={ratio:.6f}"
                        )
    assert worst < 1.10


def test_power_iterations_do_not_hurt_and_help_on_hard_cases() -> None:
    w = torch.randn(200, 120, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    exact_error = truncated_svd(w, 10).reconstruction_error
    no_iterations = randomized_svd(w, 10, n_oversamples=5, n_iter=0, seed=0).reconstruction_error
    with_iterations = randomized_svd(w, 10, n_oversamples=5, n_iter=2, seed=0).reconstruction_error

    assert exact_error <= with_iterations <= no_iterations


def test_randomized_error_is_reported_by_reconstruction_error() -> None:
    w = torch.randn(20, 12, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
    factors = randomized_svd(w, 3, n_oversamples=4, n_iter=2, seed=0)
    residual = w - factors.reconstruct()

    assert reconstruction_error(w, factors) == pytest.approx(
        float(torch.linalg.matrix_norm(residual, ord="fro")), rel=1e-12
    )
    assert factors.reconstruction_error == pytest.approx(
        reconstruction_error(w, factors), rel=1e-12
    )


def test_zero_matrix_returns_zero_valued_factors_at_the_requested_rank() -> None:
    """The sketch of a zero matrix yields zero rows in ``A`` and an arbitrary orthonormal ``B``.

    ``QR`` of a zero matrix returns an arbitrary orthonormal basis, so ``B`` is not itself zero;
    the product ``B @ A`` is zero because ``A`` is. Unlike the energy-selecting
    ``truncated_svd`` path, ``rank`` here is the requested (clamped) rank, since the randomized
    path performs no energy selection.
    """
    factors = randomized_svd(torch.zeros(6, 4), 3, n_oversamples=2, n_iter=1, seed=0)

    assert factors.rank == 3
    assert factors.reconstruction_error == 0.0
    assert not factors.A.any()
    assert not factors.reconstruct().any()


def test_rank_zero_returns_empty_factors() -> None:
    w = torch.randn(7, 5, dtype=torch.float64)
    factors = randomized_svd(w, 0, n_oversamples=0, n_iter=0, seed=0)

    assert factors.rank == 0
    assert factors.reconstruction_error == pytest.approx(float(w.norm()), rel=1e-12)


def test_rank_above_minimum_shape_is_clamped() -> None:
    w = torch.randn(6, 9, dtype=torch.float64, generator=torch.Generator().manual_seed(5))
    factors = randomized_svd(w, 999, n_oversamples=4, n_iter=3, seed=0)

    assert factors.rank == 6
    assert factors.reconstruction_error < 1e-9


def test_non_square_shapes_are_respected() -> None:
    w = torch.randn(3, 40, dtype=torch.float64, generator=torch.Generator().manual_seed(6))
    factors = randomized_svd(w, 2, n_oversamples=3, n_iter=2, seed=0)

    assert factors.A.shape == (2, 40)
    assert factors.B.shape == (3, 2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_supported_dtypes_are_preserved(dtype: torch.dtype) -> None:
    w = torch.randn(20, 10, dtype=dtype, generator=torch.Generator().manual_seed(7))
    factors = randomized_svd(w, 3, n_oversamples=3, n_iter=2, seed=0)

    assert factors.dtype is dtype
    assert factors.A.dtype is dtype and factors.B.dtype is dtype


def test_invalid_arguments_are_rejected() -> None:
    w = torch.randn(4, 4)
    with pytest.raises(ValueError, match="rank"):
        randomized_svd(w, -1, n_oversamples=1, n_iter=1, seed=0)
    with pytest.raises(ValueError, match="n_oversamples"):
        randomized_svd(w, 1, n_oversamples=-1, n_iter=1, seed=0)
    with pytest.raises(ValueError, match="n_iter"):
        randomized_svd(w, 1, n_oversamples=1, n_iter=-1, seed=0)


def test_keyword_only_parameters_are_enforced() -> None:
    with pytest.raises(TypeError):
        randomized_svd(torch.randn(4, 4), 2, 1, 1, 0)  # type: ignore[misc]
