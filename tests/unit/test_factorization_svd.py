"""Tests for the exact truncated SVD, the factor convention and the error accounting."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from itertools import pairwise

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from spectraquant.factorization.decomposition import (
    LowRankFactors,
    factor_bytes,
    reconstruction_error,
    truncated_svd,
)


def _rank_r_matrix(
    out: int, inner: int, rank: int, *, seed: int, dtype: torch.dtype
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    A = torch.randn(rank, inner, generator=generator, dtype=dtype)
    B = torch.randn(out, rank, generator=generator, dtype=dtype)
    return B @ A


# ------------------------------------------------------------------------------------------------
# Convention and exactness
# ------------------------------------------------------------------------------------------------


def test_factor_shapes_follow_the_frozen_convention() -> None:
    w = torch.randn(12, 7, dtype=torch.float64)
    factors = truncated_svd(w, 4)

    assert factors.A.shape == (4, 7)  # A: (rank, in_features)
    assert factors.B.shape == (12, 4)  # B: (out_features, rank)
    assert factors.reconstruct().shape == w.shape
    assert factors.rank == 4
    assert factors.in_features == 7
    assert factors.out_features == 12
    assert factors.num_parameters == 4 * (12 + 7)


def test_exact_reconstruction_of_a_rank_r_matrix() -> None:
    for dtype in (torch.float64, torch.float32):
        w = _rank_r_matrix(30, 17, 3, seed=5, dtype=dtype)
        factors = truncated_svd(w, 3)
        scale = float(w.norm())
        tolerance = 1e-10 * scale if dtype == torch.float64 else 1e-4 * scale

        assert factors.rank == 3
        assert factors.reconstruction_error < tolerance
        assert torch.allclose(factors.reconstruct(), w, atol=tolerance, rtol=1e-3)


def test_low_rank_matrix_is_exactly_recovered_by_its_own_rank() -> None:
    w = _rank_r_matrix(40, 20, 2, seed=0, dtype=torch.float64)
    factors = truncated_svd(w, 2)

    assert factors.reconstruction_error == pytest.approx(0.0, abs=1e-12)


def test_error_is_non_increasing_in_rank() -> None:
    w = torch.randn(60, 40, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    errors = [truncated_svd(w, rank).reconstruction_error for rank in range(0, 41, 4)]

    assert all(later <= earlier + 1e-12 for earlier, later in pairwise(errors)), errors
    assert errors[0] == pytest.approx(float(w.norm()), rel=1e-12)


def test_full_rank_reconstructs_exactly() -> None:
    w = torch.randn(9, 5, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    factors = truncated_svd(w, 5)

    assert factors.rank == 5
    assert factors.reconstruction_error < 1e-12


# ------------------------------------------------------------------------------------------------
# energy_keep
# ------------------------------------------------------------------------------------------------


def test_energy_keep_reduces_rank_monotonically() -> None:
    w = torch.randn(50, 30, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    ranks = [truncated_svd(w, 30, energy_keep=e).rank for e in (0.5, 0.9, 0.99, 1.0)]

    assert all(later >= earlier for earlier, later in pairwise(ranks)), ranks
    assert ranks[-1] == 30


def test_energy_keep_cannot_exceed_the_requested_rank() -> None:
    w = torch.randn(50, 30, dtype=torch.float64, generator=torch.Generator().manual_seed(4))

    assert truncated_svd(w, 7, energy_keep=0.999).rank <= 7


def test_energy_keep_captures_at_least_the_requested_energy() -> None:
    w = torch.randn(80, 60, dtype=torch.float64, generator=torch.Generator().manual_seed(6))
    s = torch.linalg.svdvals(w)
    total = float((s * s).sum())

    for fraction in (0.25, 0.75, 0.99):
        rank = truncated_svd(w, 60, energy_keep=fraction).rank
        captured = float((s[:rank] * s[:rank]).sum()) / total
        assert captured >= fraction
        if rank > 0:
            previous = float((s[: rank - 1] * s[: rank - 1]).sum()) / total
            assert previous < fraction


def test_energy_keep_on_the_zero_matrix_selects_rank_zero() -> None:
    factors = truncated_svd(torch.zeros(6, 4), 3, energy_keep=0.99)

    assert factors.rank == 0
    assert factors.reconstruction_error == 0.0


def test_energy_keep_out_of_range_is_rejected() -> None:
    w = torch.randn(4, 4)
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="energy_keep"):
            truncated_svd(w, 2, energy_keep=bad)


# ------------------------------------------------------------------------------------------------
# Degenerate inputs
# ------------------------------------------------------------------------------------------------


def test_rank_zero_returns_empty_factors_with_the_full_norm_as_error() -> None:
    w = torch.randn(8, 5, dtype=torch.float64, generator=torch.Generator().manual_seed(7))
    factors = truncated_svd(w, 0)

    assert factors.rank == 0
    assert factors.A.shape == (0, 5)
    assert factors.B.shape == (8, 0)
    assert factors.reconstruction_error == pytest.approx(float(w.norm()), rel=1e-12)


def test_rank_above_minimum_shape_is_clamped() -> None:
    w = torch.randn(6, 9, dtype=torch.float64)
    factors = truncated_svd(w, 999)

    assert factors.rank == 6
    assert factors.reconstruction_error < 1e-12


def test_non_square_orientation_is_respected() -> None:
    tall = torch.randn(40, 7, dtype=torch.float64, generator=torch.Generator().manual_seed(8))
    wide = tall.mT
    f_tall = truncated_svd(tall, 3)
    f_wide = truncated_svd(wide, 3)

    assert f_tall.A.shape == (3, 7) and f_tall.B.shape == (40, 3)
    assert f_wide.A.shape == (3, 40) and f_wide.B.shape == (7, 3)
    assert f_tall.reconstruction_error == pytest.approx(f_wide.reconstruction_error, rel=1e-12)


def test_rank_one_matrix_round_trips() -> None:
    w = _rank_r_matrix(11, 13, 1, seed=9, dtype=torch.float64)
    factors = truncated_svd(w, 1)

    assert factors.rank == 1
    assert factors.reconstruction_error < 1e-12


def test_negative_rank_is_rejected() -> None:
    with pytest.raises(ValueError, match="rank"):
        truncated_svd(torch.randn(3, 3), -1)


# ------------------------------------------------------------------------------------------------
# dtypes / device guards
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_supported_dtypes_are_preserved(dtype: torch.dtype) -> None:
    w = torch.randn(10, 6, dtype=dtype, generator=torch.Generator().manual_seed(10))
    factors = truncated_svd(w, 2)

    assert factors.dtype is dtype
    assert factors.A.dtype is dtype and factors.B.dtype is dtype


def test_half_precision_is_rejected_with_an_actionable_message() -> None:
    with pytest.raises(ValueError, match=r"float32|CPU"):
        truncated_svd(torch.randn(4, 4, dtype=torch.float16), 2)


def test_non_cpu_tensor_is_rejected() -> None:
    with pytest.raises(ValueError, match="CPU"):
        truncated_svd(torch.empty(4, 4, device="meta"), 2)


def test_non_matrix_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="2-D"):
        truncated_svd(torch.randn(4), 1)


# ------------------------------------------------------------------------------------------------
# reconstruction_error
# ------------------------------------------------------------------------------------------------


def test_reconstruction_error_is_absolute_frobenius_by_default() -> None:
    w = torch.randn(20, 12, dtype=torch.float64, generator=torch.Generator().manual_seed(11))
    factors = truncated_svd(w, 4)
    residual = w - factors.reconstruct()
    expected = float(torch.linalg.matrix_norm(residual, ord="fro"))

    assert reconstruction_error(w, factors) == pytest.approx(expected, rel=1e-12)


def test_relative_reconstruction_error_normalises_by_the_weight_norm() -> None:
    w = torch.randn(20, 12, dtype=torch.float64, generator=torch.Generator().manual_seed(12))
    factors = truncated_svd(w, 4)
    absolute = reconstruction_error(w, factors)
    relative = reconstruction_error(w, factors, norm="relative_fro")

    assert relative == pytest.approx(absolute / float(w.norm()), rel=1e-12)
    assert reconstruction_error(w, factors, norm="fro_relative") == pytest.approx(relative)


def test_relative_error_of_the_zero_matrix_is_zero() -> None:
    zero = torch.zeros(3, 3)
    factors = truncated_svd(zero, 1)

    assert reconstruction_error(zero, factors, norm="relative_fro") == 0.0


def test_unknown_norm_is_rejected() -> None:
    w = torch.randn(4, 4)
    factors = truncated_svd(w, 1)
    with pytest.raises(ValueError, match="norm"):
        reconstruction_error(w, factors, norm="nuclear")


def test_shape_mismatch_between_weight_and_factors_is_rejected() -> None:
    w = torch.randn(4, 4)
    factors = truncated_svd(torch.randn(6, 3), 2)
    with pytest.raises(ValueError, match="do not match"):
        reconstruction_error(w, factors)


def test_stored_error_matches_the_discarded_singular_values() -> None:
    """Property: for exact truncation the error is the norm of the discarded singular values."""
    w = torch.randn(30, 18, dtype=torch.float64, generator=torch.Generator().manual_seed(13))
    singular_values = torch.linalg.svdvals(w)

    for rank in (0, 1, 5, 18):
        factors = truncated_svd(w, rank)
        discarded = float(singular_values[factors.rank :].norm())
        assert factors.reconstruction_error == pytest.approx(discarded, rel=1e-10, abs=1e-12)


@settings(max_examples=60, deadline=None)
@given(
    out=st.integers(min_value=1, max_value=8),
    inner=st.integers(min_value=1, max_value=8),
    rank=st.integers(min_value=0, max_value=8),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_property_error_equals_norm_of_discarded_singular_values(
    out: int, inner: int, rank: int, seed: int
) -> None:
    generator = torch.Generator().manual_seed(seed)
    w = torch.randn(out, inner, generator=generator, dtype=torch.float64)
    factors = truncated_svd(w, rank)
    discarded = torch.linalg.svdvals(w)[factors.rank :]
    expected = float(discarded.norm()) if discarded.numel() else 0.0

    assert factors.rank == min(rank, min(out, inner))
    assert factors.reconstruction_error == pytest.approx(expected, rel=1e-9, abs=1e-12)


# ------------------------------------------------------------------------------------------------
# LowRankFactors container
# ------------------------------------------------------------------------------------------------


def test_low_rank_factors_is_frozen() -> None:
    factors = truncated_svd(torch.randn(4, 4), 2)
    with pytest.raises(FrozenInstanceError):
        factors.rank = 3  # type: ignore[misc]


def test_inconsistent_rank_is_rejected() -> None:
    A = torch.randn(2, 4)
    B = torch.randn(4, 3)  # B.shape[1] != A.shape[0]
    with pytest.raises(ValueError, match="rank"):
        LowRankFactors(A=A, B=B, rank=2, dtype=torch.float32, reconstruction_error=0.0)


def test_from_weight_measures_the_error() -> None:
    w = torch.randn(5, 3, dtype=torch.float64)
    factors = LowRankFactors.from_weight(w, w.new_zeros(1, 3), w.new_zeros(5, 1))

    assert factors.reconstruction_error == pytest.approx(float(w.norm()), rel=1e-12)


# ------------------------------------------------------------------------------------------------
# Class-1 analytical cost
# ------------------------------------------------------------------------------------------------


def test_factor_bytes_matches_the_memory_accounting_worked_example() -> None:
    # docs/protocols/memory-accounting.md §7.3: r=64, in=out=4096, fp16 factors -> 1 048 576 B.
    assert factor_bytes(4096, 4096, 64) == 1_048_576
    assert factor_bytes(4096, 4096, 64, 4) == 2_097_152


def test_factor_bytes_scales_with_rank_and_dtype() -> None:
    assert factor_bytes(10, 20, 0) == 0
    assert factor_bytes(10, 20, 3) == 3 * 30 * 2
    assert factor_bytes(10, 20, 3, dtype_bytes=1) == 3 * 30


def test_factor_bytes_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="rank"):
        factor_bytes(4, 4, -1)
    with pytest.raises(ValueError, match="dtype_bytes"):
        factor_bytes(4, 4, 1, dtype_bytes=0)
    with pytest.raises(TypeError, match="in_features"):
        factor_bytes(4.0, 4, 1)  # type: ignore[arg-type]


def test_factor_bytes_matches_num_parameters_of_a_real_factorization() -> None:
    w = torch.randn(32, 16, dtype=torch.float64)
    factors = truncated_svd(w, 4)

    assert factor_bytes(16, 32, factors.rank, 4) == factors.num_parameters * 4
    assert math.isclose(
        factors.reconstruction_error, reconstruction_error(w, factors), rel_tol=1e-12
    )
