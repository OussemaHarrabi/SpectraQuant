"""Tests for the M4 proxy-validation statistics and the local-fixture sweep artifact.

Covers: the DerSimonian-Laird Fisher-z combination against a hand-computed example; the pre-declared
MDE formula against the numbers frozen in `preregistration.md` section 7.5; determinism of the sweep
at fixed seeds; structural validation of the committed artifact (keys present, no NaN/None hiding a
non-finite measurement); and the requirement that the candidate's model-level advantage over the
naive baseline is reported with its confidence interval rather than as a bare number.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from spectraquant.proxies.analysis import (
    PREDECLARED_MDE_Z_CONSERVATIVE,
    PREDECLARED_MDE_Z_OPTIMISTIC,
    cluster_contrast,
    contrast_verdict,
    fisher_z,
    fisher_z_random_effects,
    inverse_fisher_z,
    mde_fisher_z,
    within_model_neff,
)
from spectraquant.proxies.validation import (
    CANDIDATE,
    DEFAULT_CELLS,
    DEFAULT_SEEDS,
    NAIVE_BASELINE,
    PREDECLARED_COMPARATORS,
    build_document,
    run_sweep,
)

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts" / "sample-results" / "proxy-validation" / "proxy-validation.json"


# ------------------------------------------------------------------ Fisher-z combination


def test_random_effects_matches_a_hand_computed_example() -> None:
    """DerSimonian-Laird on effects [0.2, 0.5, 0.9] with variances [0.04, 0.09, 0.16].

    The expected values were computed independently (exact rational arithmetic on the DL formulas,
    no code from this package):

    * fixed-effect mean = 0.3819672131, Q = 2.6598360656, C = 23.7704918033, tau^2 = 0.0277586207
    * random-effects pooled = 0.4196154239, SE = 0.1870672363
    * 95% CI = [0.0529703781, 0.7862604697], z = 2.2431262269, two-sided p = 0.0248886759
    * I^2 = 0.2480739599
    """
    result = fisher_z_random_effects([0.2, 0.5, 0.9], [0.04, 0.09, 0.16])
    assert result.k == 3
    assert result.pooled == pytest.approx(0.4196154239, rel=1e-6)
    assert result.tau2 == pytest.approx(0.0277586207, rel=1e-6)
    assert result.q == pytest.approx(2.6598360656, rel=1e-6)
    assert result.df == 2
    assert result.se == pytest.approx(0.1870672363, rel=1e-6)
    assert result.ci_low == pytest.approx(0.0529703781, rel=1e-6)
    assert result.ci_high == pytest.approx(0.7862604697, rel=1e-6)
    assert result.z_stat == pytest.approx(2.2431262269, rel=1e-6)
    assert result.p_value == pytest.approx(0.0248886759, rel=1e-6)
    assert result.i2 == pytest.approx(0.2480739599, rel=1e-6)
    assert math.isclose(result.pooled, 0.4196154239, rel_tol=1e-6)
    assert result.ci_low < result.pooled < result.ci_high


def test_homogeneous_effects_reduce_to_the_inverse_variance_mean() -> None:
    """With identical effects there is no between-study variance and tau^2 is 0."""
    result = fisher_z_random_effects([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    assert result.pooled == pytest.approx(1.0, rel=1e-12)
    assert result.tau2 == 0.0
    assert result.q == pytest.approx(0.0, abs=1e-12)
    assert result.i2 == 0.0
    # inverse-variance mean of three unit-variance estimates has SE = 1/sqrt(3)
    assert result.se == pytest.approx(1.0 / math.sqrt(3.0), rel=1e-12)


def test_single_unit_is_returned_with_its_own_interval() -> None:
    result = fisher_z_random_effects([0.5], [0.25])
    assert result.k == 1 and result.tau2 == 0.0
    assert result.pooled == 0.5
    assert result.se == pytest.approx(0.5, rel=1e-12)
    assert result.ci_low == pytest.approx(0.5 - 1.9599639845 * 0.5, rel=1e-6)


def test_random_effects_rejects_non_finite_or_non_positive_inputs() -> None:
    with pytest.raises(ValueError):
        fisher_z_random_effects([], [])
    with pytest.raises(ValueError):
        fisher_z_random_effects([0.1, 0.2], [0.1])
    with pytest.raises(ValueError):
        fisher_z_random_effects([float("nan")], [0.1])
    with pytest.raises(ValueError):
        fisher_z_random_effects([0.1], [0.0])


def test_fisher_z_is_finite_at_the_boundaries_and_invertible() -> None:
    assert fisher_z(0.0) == 0.0
    assert math.isfinite(fisher_z(1.0)) and math.isfinite(fisher_z(-1.0))
    assert fisher_z(1.0) > fisher_z(0.999)
    assert math.isnan(fisher_z(float("nan")))
    for r in (-0.5, 0.0, 0.3, 0.75):
        assert inverse_fisher_z(fisher_z(r)) == pytest.approx(r, rel=1e-12)


def test_mde_formula_reproduces_the_predeclared_section_7_5_numbers() -> None:
    """`preregistration.md` section 7.5 declares MDE(z) ~ 0.33 (n_eff=32) and ~ 0.79 (n_eff=8) at 5 seeds."""
    assert mde_fisher_z(32, 5) == pytest.approx(PREDECLARED_MDE_Z_OPTIMISTIC, abs=5e-3)
    assert mde_fisher_z(8, 5) == pytest.approx(PREDECLARED_MDE_Z_CONSERVATIVE, abs=5e-3)
    assert mde_fisher_z(32, 5) < mde_fisher_z(8, 5)
    assert mde_fisher_z(32, 10) < mde_fisher_z(32, 5)
    assert math.isnan(mde_fisher_z(3.0, 5))
    assert math.isnan(mde_fisher_z(20.0, 0))


# ------------------------------------------------------------------ within-model n_eff


def _proxy_and_damage(n: int = 13) -> tuple[list[float], list[float]]:
    """A deterministic, *imperfectly* correlated fixture pair.

    Rank-perfect data would make every bootstrap resample have ``rho = 1``, a zero bootstrap variance
    and therefore an undefined ``n_eff``; the deterministic pseudo-random construction below keeps
    ``rho`` strictly inside ``(-1, 1)`` while staying reproducible.
    """
    import random

    rng = random.Random(20261009)
    damage = [0.5 + rng.random() * 3.0 for _ in range(n)]
    proxy = [d * (0.8 + 0.4 * rng.random()) for d in damage]
    return proxy, damage


def test_within_model_neff_is_deterministic_and_finite() -> None:
    proxy, damage = _proxy_and_damage()
    first = within_model_neff(proxy, damage, n_boot=64, seed=7)
    second = within_model_neff(proxy, damage, n_boot=64, seed=7)
    assert first == second
    assert first.n_modules == len(proxy)
    assert math.isfinite(first.n_eff)
    assert first.n_draws_accepted == 64


def test_within_model_neff_reports_a_zero_variance_bootstrap_as_undefined() -> None:
    """A rank-perfect pair has every resample at ``|rho| = 1``: the transform is clipped, the
    bootstrap variance is 0 and ``n_eff`` is undefined rather than silently large."""
    damage = [1.0 + 0.7 * i for i in range(13)]
    proxy = [2.0 * d for d in damage]
    estimate = within_model_neff(proxy, damage, n_boot=32, seed=1)
    assert estimate.clipped
    assert math.isnan(estimate.n_eff)


def test_within_model_neff_is_undefined_for_a_constant_proxy() -> None:
    _, damage = _proxy_and_damage()
    constant = [3.0] * len(damage)
    estimate = within_model_neff(constant, damage, n_boot=32, seed=1)
    assert math.isnan(estimate.n_eff)


# ------------------------------------------------------------------ contrast reporting


def _synthetic_clusters(
    effect: float, *, spread: float = 0.01
) -> dict[str, list[tuple[float, float]]]:
    return {f"model{i}": [(effect + spread * (i % 3), 0.05 + 0.001 * i)] for i in range(5)}


def test_contrast_reports_an_interval_not_a_bare_number() -> None:
    contrast = cluster_contrast(
        _synthetic_clusters(0.6), label="candidate - naive", n_boot=200, seed=3
    )
    payload = contrast.to_dict()
    for key in ("delta_z", "ci_low", "ci_high", "boot_ci_low", "boot_ci_high", "p_value", "k"):
        assert key in payload, f"contrast payload is missing {key}"
    assert payload["ci_low"] < payload["delta_z"] < payload["ci_high"]
    assert payload["boot_ci_low"] <= payload["boot_ci_high"]
    assert payload["boot_ci_high"] - payload["boot_ci_low"] > 0.0
    assert contrast.n_clusters == 5


def test_verdict_uses_the_falsifier_rule_in_both_directions() -> None:
    positive = cluster_contrast(_synthetic_clusters(0.6), n_boot=100, seed=5)
    assert contrast_verdict(positive).verdict == "supported"
    negative = cluster_contrast(_synthetic_clusters(-0.6), n_boot=100, seed=5)
    assert contrast_verdict(negative).verdict == "refuted"
    # a wide, zero-centred contrast is inconclusive, never a refutation
    wide = {f"model{i}": [(0.2 if i % 2 else -0.2, 1.5)] for i in range(4)}
    verdict = contrast_verdict(cluster_contrast(wide, n_boot=100, seed=5))
    assert verdict.verdict == "inconclusive"
    assert not verdict.decisive


# ------------------------------------------------------------------ the committed artifact


def _load_artifact() -> dict:
    assert ARTIFACT.exists(), f"missing artifact: {ARTIFACT}"
    text = ARTIFACT.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    return json.loads(text, parse_constant=_reject_constant)


def _reject_constant(name: str) -> float:
    raise AssertionError(f"artifact contains the non-JSON constant {name}")


def test_committed_artifact_has_the_expected_skeleton() -> None:
    document = _load_artifact()
    assert document["schema"] == "spectraquant.proxy-validation/1"
    assert document["substrate"] == "LOCAL-FIXTURE"
    assert document["measurement_class"] == 2
    assert document["statistical_unit"] == "model"
    assert isinstance(document["git_commit"], str) and document["git_commit"]
    for key in ("plan", "predeclared_mde", "runs", "per_cell", "aggregate", "cloud_cells"):
        assert key in document, f"artifact is missing {key}"


def test_committed_artifact_respects_the_preregistered_floor_and_grid() -> None:
    document = _load_artifact()
    assert document["plan"]["seeds"] == list(DEFAULT_SEEDS)
    assert len(document["plan"]["seeds"]) >= 5, "Tier-0 seed floor is >= 5"
    assert [list(c) for c in document["plan"]["cells"]] == [list(c) for c in DEFAULT_CELLS]
    assert len(document["plan"]["cells"]) >= 4
    ranks = {c[0] for c in document["plan"]["cells"]}
    bits = {c[1] for c in document["plan"]["cells"]}
    assert ranks == {8, 16, 32} and bits == {4, 8}
    assert set(document["plan"]["fixtures"]) == {"layernorm", "no_layernorm"}
    assert document["plan"]["candidate_proxies"][0] == CANDIDATE
    assert document["plan"]["naive_baseline"] == NAIVE_BASELINE
    assert document["plan"]["predeclared_comparators"] == list(PREDECLARED_COMPARATORS)
    assert document["runtime"]["n_models"] == len(DEFAULT_SEEDS) * 2
    assert len(document["per_cell"]) == len(DEFAULT_CELLS) * 2


def test_committed_artifact_has_no_missing_measurements() -> None:
    """A ``None`` where a measured number belongs means a non-finite value was silently dropped."""
    document = _load_artifact()
    checked = (*PREDECLARED_COMPARATORS, CANDIDATE, NAIVE_BASELINE)
    for fixture, cells in document["runs"]["cells"].items():
        for cell in cells:
            for variant in checked:
                rhos = cell["rho"][variant]
                n_effs = cell["n_eff"][variant]
                assert len(rhos) == len(document["runs"]["seeds"]) == 5, (
                    fixture,
                    cell["rank"],
                    variant,
                )
                assert all(isinstance(r, float) for r in rhos), (
                    fixture,
                    cell["rank"],
                    variant,
                    rhos,
                )
                assert all(isinstance(n, float) and n > 3.0 for n in n_effs), (
                    fixture,
                    cell["rank"],
                    variant,
                    n_effs,
                )
    for cell in document["per_cell"]:
        assert cell["mde"]["achieved_z"] is not None
        for variant, row in cell["variants"].items():
            assert row["rho_mean"] is not None, (cell["fixture"], variant)
            assert row["n_eff_mean"] is not None, (cell["fixture"], variant)
        assert cell["damage_mean"]["joint_hidden"] > 0.0


def test_committed_artifact_reports_the_candidate_advantage_with_its_ci() -> None:
    document = _load_artifact()
    for group in ("layernorm", "no_layernorm", "all_fixtures"):
        contrasts = document["aggregate"][group]["contrasts"][CANDIDATE]
        assert NAIVE_BASELINE in contrasts
        assert PREDECLARED_COMPARATORS[0] in contrasts
        for comparator in (*PREDECLARED_COMPARATORS, NAIVE_BASELINE):
            contrast = contrasts[comparator]
            assert isinstance(contrast, dict), "a bare scalar is not an acceptable contrast report"
            ci = contrast["ci"]
            boot = contrast["boot_ci"]
            assert len(ci) == len(boot) == 2
            assert all(isinstance(v, float) for v in (*ci, *boot))
            assert ci[0] < ci[1] and boot[0] <= boot[1]
            assert ci[0] < contrast["delta_z"] < ci[1]
            assert isinstance(contrast["p_value"], float)
            assert contrast["verdict"]["verdict"] in {"supported", "refuted", "inconclusive"}
            assert "mde_achieved_z" in contrast and contrast["mde_achieved_z"] is not None
    per_cell_contrasts = document["per_cell"][0]["contrasts"][CANDIDATE][PREDECLARED_COMPARATORS[0]]
    assert len(per_cell_contrasts["ci"]) == 2
    assert len(per_cell_contrasts["boot_ci"]) == 2


def test_committed_artifact_marks_the_cloud_cells_not_run() -> None:
    document = _load_artifact()
    assert document["cloud_cells"], "the cloud cells must be listed"
    for cell in document["cloud_cells"].values():
        assert cell["substrate"] == "CLOUD-COLAB"
        assert cell["status"] == "NOT RUN"
        assert cell["hypothesis"] in {"H2", "H4"}


def test_committed_artifact_records_the_achieved_and_predeclared_mde() -> None:
    document = _load_artifact()
    assert document["predeclared_mde"]["mde_z_optimistic"] == PREDECLARED_MDE_Z_OPTIMISTIC
    assert document["predeclared_mde"]["mde_z_conservative"] == PREDECLARED_MDE_Z_CONSERVATIVE
    achieved = document["aggregate"]["all_fixtures"]["mde"]["achieved_z"]
    assert achieved is not None and math.isfinite(achieved)
    assert achieved > 0.0


# ------------------------------------------------------------------ sweep determinism


@pytest.mark.integration
def test_sweep_is_deterministic_at_a_fixed_seed() -> None:
    """Two sweeps at the same seeds reproduce the same scientific document bit-for-bit.

    Only the wall-clock fields (``runtime.wall_time_s`` and the per-run ``training.wall_time_s``) are
    allowed to differ; every measured and derived number must be bit-identical.
    """
    kwargs = {"seeds": (0, 1), "cells": ((8, 4), (32, 8)), "n_boot_neff": 20}

    def science(document: dict) -> str:
        clone = json.loads(json.dumps(document, sort_keys=True))
        clone.pop("runtime")
        for fixture in clone["runs"]["training"].values():
            for record in fixture.values():
                record.pop("wall_time_s")
        return json.dumps(clone, sort_keys=True)

    first = build_document(run_sweep(**kwargs), wall_time_s=0.0, **kwargs, n_boot_ci=50)
    second = build_document(run_sweep(**kwargs), wall_time_s=0.0, **kwargs, n_boot_ci=50)
    assert science(first) == science(second)
    assert first["runtime"]["n_models"] == 4

    # and the measured content does depend on the seeds
    other = build_document(
        run_sweep(seeds=(0, 2), cells=kwargs["cells"], n_boot_neff=20),
        seeds=(0, 2),
        cells=kwargs["cells"],
        n_boot_neff=20,
        n_boot_ci=50,
    )
    assert other["per_cell"][0]["variants"] != first["per_cell"][0]["variants"]
