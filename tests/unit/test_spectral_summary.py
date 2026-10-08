"""Tests for the spectral analysis module (singular values, ranks, summary features)."""

from __future__ import annotations

import math
from itertools import pairwise

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from spectraquant.factorization.spectral import (
    TOP_K_SUMMARY,
    effective_rank,
    singular_values,
    spectral_summary,
    stable_rank,
)


def _from_spectrum(s: torch.Tensor, *, seed: int = 0) -> torch.Tensor:
    """Build a matrix with the prescribed singular values ``s`` (random orthonormal bases)."""
    n = int(s.numel())
    gen0 = torch.Generator().manual_seed(seed)
    gen1 = torch.Generator().manual_seed(seed + 1)
    q0, _ = torch.linalg.qr(torch.randn(n, n, generator=gen0, dtype=torch.float64))
    q1, _ = torch.linalg.qr(torch.randn(n, n, generator=gen1, dtype=torch.float64))
    return q0 @ torch.diag(s) @ q1.mT


def _power_law(n: int, alpha: float) -> torch.Tensor:
    idx = torch.arange(1, n + 1, dtype=torch.float64)
    return idx ** (-alpha)


# ------------------------------------------------------------------------------------------------
# singular_values
# ------------------------------------------------------------------------------------------------


def test_singular_values_are_descending_and_match_the_reference() -> None:
    w = torch.randn(20, 12, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    values = singular_values(w)
    reference = torch.linalg.svdvals(w)

    assert values.shape == (12,)
    assert torch.allclose(values, reference, atol=1e-12)
    assert bool(torch.all(values[:-1] >= values[1:]))


def test_singular_values_top_k_truncates() -> None:
    w = torch.randn(20, 12, dtype=torch.float64, generator=torch.Generator().manual_seed(1))

    assert singular_values(w, top_k=3).shape == (3,)
    assert torch.allclose(singular_values(w, top_k=3), singular_values(w)[:3])
    assert singular_values(w, top_k=0).shape == (0,)
    assert singular_values(w, top_k=999).shape == (12,)


def test_singular_values_of_a_projected_matrix_match_the_spectrum() -> None:
    s = _power_law(16, 1.0)
    w = _from_spectrum(s)

    assert torch.allclose(singular_values(w), s, rtol=1e-9, atol=1e-12)


def test_singular_values_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="2-D"):
        singular_values(torch.randn(5))
    with pytest.raises(ValueError, match="top_k"):
        singular_values(torch.randn(3, 3), top_k=-1)
    with pytest.raises(ValueError, match="float32/float64"):
        singular_values(torch.randn(3, 3, dtype=torch.float16))
    with pytest.raises(ValueError, match="CPU"):
        singular_values(torch.empty(3, 3, device="meta"))


# ------------------------------------------------------------------------------------------------
# effective_rank / stable_rank
# ------------------------------------------------------------------------------------------------


def test_effective_rank_matches_the_measured_power_law_ranks() -> None:
    # docs/research/spectral-notes.md §2 (n=256, float64): rank@99% = 241 / 49 / 3.
    for alpha, expected in ((0.5, 241), (1.0, 49), (2.0, 3)):
        w = _from_spectrum(_power_law(256, alpha))
        assert effective_rank(w, energy=0.99) == expected


def test_effective_rank_threshold_ranks_are_internally_consistent() -> None:
    w = _from_spectrum(_power_law(256, 1.0))
    s = singular_values(w)
    total = float((s * s).sum())

    for energy in (0.5, 0.9, 0.99, 0.999):
        rank = int(effective_rank(w, energy=energy))
        captured = float((s[:rank] * s[:rank]).sum()) / total
        assert captured >= energy
        assert float((s[: rank - 1] * s[: rank - 1]).sum()) / total < energy


def test_effective_rank_of_a_flat_spectrum_is_full_rank() -> None:
    w = _from_spectrum(torch.ones(32, dtype=torch.float64))

    assert effective_rank(w, energy=0.99) == 32
    assert stable_rank(w) == pytest.approx(32.0)


def test_effective_rank_of_the_zero_matrix_is_zero() -> None:
    assert effective_rank(torch.zeros(5, 7)) == 0.0


def test_stable_rank_matches_the_measured_regimes() -> None:
    # docs/research/spectral-notes.md §2: power-law 2.0 -> 1.082; near-rank-1 -> 1.000.
    assert stable_rank(_from_spectrum(_power_law(256, 2.0))) == pytest.approx(1.082, abs=5e-3)
    s = torch.full((256,), 1e-3, dtype=torch.float64)
    s[0] = 1.0
    # 255 non-zero singular values of 1e-3 contribute 2.55e-4 to the energy, so the exact stable
    # rank is 1.000255 — the notes round it to 1.000.
    assert stable_rank(_from_spectrum(s)) == pytest.approx(1.0, abs=5e-3)


def test_stable_rank_of_the_zero_matrix_is_zero() -> None:
    assert stable_rank(torch.zeros(4, 4)) == 0.0


def test_rank_functions_reject_invalid_energy() -> None:
    w = torch.randn(4, 4)
    for bad in (0.0, 1.5, -0.5):
        with pytest.raises(ValueError, match="energy"):
            effective_rank(w, energy=bad)


# ------------------------------------------------------------------------------------------------
# spectral_summary
# ------------------------------------------------------------------------------------------------


def test_spectral_summary_contains_every_required_feature_family() -> None:
    w = torch.randn(64, 48, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    summary = spectral_summary(w)

    required = {
        "n_singular_values",
        "spectral_decay_exponent",
        "spectral_decay_r2",
        "effective_rank",
        "stable_rank",
        "condition_number",
        "condition_number_nonzero",
        "energy_top_1pct",
        "energy_top_1pct_count",
        "sv_min",
        "sv_q25",
        "sv_median",
        "sv_q75",
        "sv_max",
        "sv_iqr",
        "sv_iqr_over_median",
        "sv_coefficient_of_variation",
        "frobenius_norm",
        "spectral_norm",
        "nuclear_norm",
    }
    assert required <= set(summary)
    # top-k singular values are surfaced individually, under a stable key scheme
    for i in range(TOP_K_SUMMARY):
        assert f"sv_{i}" in summary
    assert all(isinstance(value, float) for value in summary.values())


def test_spectral_summary_top_k_values_match_singular_values() -> None:
    w = torch.randn(64, 48, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    summary = spectral_summary(w)
    values = singular_values(w, top_k=TOP_K_SUMMARY)

    for i, value in enumerate(values):
        assert summary[f"sv_{i}"] == pytest.approx(float(value), rel=1e-12)


def test_spectral_summary_recovers_the_prescribed_power_law_exponent() -> None:
    for alpha in (0.5, 1.0, 2.0):
        summary = spectral_summary(_from_spectrum(_power_law(256, alpha)))
        assert summary["spectral_decay_exponent"] == pytest.approx(alpha, abs=1e-6)
        assert summary["spectral_decay_r2"] == pytest.approx(1.0, abs=1e-6)


def test_spectral_summary_matches_the_measured_regime_table() -> None:
    # docs/research/spectral-notes.md §2, n = 256.
    expected = {
        0.5: {"effective_rank": 241.0, "stable_rank": 6.124, "energy_top_1pct": 0.2994},
        1.0: {"effective_rank": 49.0, "stable_rank": 1.641, "energy_top_1pct": 0.8294},
        2.0: {"effective_rank": 3.0, "stable_rank": 1.082, "energy_top_1pct": 0.9931},
    }
    for alpha, values in expected.items():
        summary = spectral_summary(_from_spectrum(_power_law(256, alpha)))
        assert summary["effective_rank"] == values["effective_rank"]
        assert summary["stable_rank"] == pytest.approx(values["stable_rank"], abs=5e-3)
        assert summary["energy_top_1pct"] == pytest.approx(values["energy_top_1pct"], abs=1e-3)


def test_near_rank_one_regime_is_the_concentration_extreme() -> None:
    s = torch.full((256,), 1e-3, dtype=torch.float64)
    s[0] = 1.0
    summary = spectral_summary(_from_spectrum(s))

    assert summary["effective_rank"] == 1.0
    assert summary["stable_rank"] == pytest.approx(1.0, abs=5e-3)
    assert summary["condition_number"] == pytest.approx(1000.0, rel=1e-9)
    assert summary["energy_top_1pct"] > 0.99


def test_condition_number_is_infinite_for_a_rank_deficient_matrix() -> None:
    w = torch.zeros(6, 6, dtype=torch.float64)
    w[0, 0] = 2.0
    summary = spectral_summary(w)

    assert summary["condition_number"] == math.inf
    assert summary["condition_number_nonzero"] == pytest.approx(1.0)


def test_quantile_spread_keys_are_consistent() -> None:
    w = torch.randn(40, 40, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
    summary = spectral_summary(w)

    assert summary["sv_min"] <= summary["sv_q25"] <= summary["sv_median"]
    assert summary["sv_median"] <= summary["sv_q75"] <= summary["sv_max"]
    assert summary["sv_iqr"] == pytest.approx(summary["sv_q75"] - summary["sv_q25"])
    assert summary["sv_iqr_over_median"] == pytest.approx(summary["sv_iqr"] / summary["sv_median"])


def test_zero_matrix_summary_is_defined_and_does_not_raise() -> None:
    summary = spectral_summary(torch.zeros(5, 7))

    assert summary["effective_rank"] == 0.0
    assert summary["stable_rank"] == 0.0
    assert summary["condition_number"] == 0.0
    assert summary["energy_top_1pct"] == 0.0
    assert summary["frobenius_norm"] == 0.0
    assert summary["nuclear_norm"] == 0.0
    assert math.isnan(summary["spectral_decay_exponent"])


def test_rank_one_matrix_summary() -> None:
    w = torch.zeros(4, 6, dtype=torch.float64)
    w[0, 0] = 3.0
    summary = spectral_summary(w)

    assert summary["spectral_norm"] == 3.0
    assert summary["nuclear_norm"] == 3.0
    assert summary["effective_rank"] == 1.0
    assert summary["stable_rank"] == pytest.approx(1.0)


def test_non_square_summary_uses_the_smaller_dimension() -> None:
    w = torch.randn(9, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(5))
    summary = spectral_summary(w)

    assert summary["n_singular_values"] == 4.0
    assert summary["stable_rank"] <= 4.0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_summary_supports_both_float_dtypes(dtype: torch.dtype) -> None:
    w = torch.randn(16, 8, dtype=dtype, generator=torch.Generator().manual_seed(6))
    summary = spectral_summary(w)

    assert summary["n_singular_values"] == 8.0
    assert summary["spectral_norm"] > 0.0


def test_frobenius_and_spectral_norms_agree_with_torch() -> None:
    w = torch.randn(20, 13, dtype=torch.float64, generator=torch.Generator().manual_seed(7))
    summary = spectral_summary(w)

    assert summary["frobenius_norm"] == pytest.approx(float(torch.linalg.matrix_norm(w, ord="fro")))
    assert summary["spectral_norm"] == pytest.approx(float(torch.linalg.matrix_norm(w, ord=2)))


# ------------------------------------------------------------------------------------------------
# Properties
# ------------------------------------------------------------------------------------------------


@settings(max_examples=60, deadline=None)
@given(
    out=st.integers(min_value=1, max_value=10),
    inner=st.integers(min_value=1, max_value=10),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_property_ranks_are_bounded_by_the_smaller_dimension(
    out: int, inner: int, seed: int
) -> None:
    generator = torch.Generator().manual_seed(seed)
    w = torch.randn(out, inner, generator=generator, dtype=torch.float64)
    bound = float(min(out, inner))

    assert 0.0 <= effective_rank(w, energy=0.99) <= bound
    assert 0.0 <= stable_rank(w) <= bound + 1e-9
    summary = spectral_summary(w)
    assert 0.0 <= summary["energy_top_1pct"] <= 1.0 + 1e-12


@settings(max_examples=40, deadline=None)
@given(
    dim=st.integers(min_value=2, max_value=8),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_property_effective_rank_is_monotone_in_the_energy_threshold(dim: int, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    w = torch.randn(dim, dim, generator=generator, dtype=torch.float64)
    ranks = [effective_rank(w, energy=e) for e in (0.25, 0.5, 0.75, 0.99)]

    assert all(later >= earlier for earlier, later in pairwise(ranks)), ranks
