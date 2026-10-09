"""Tests for the Milestone-8 ablation helpers and the committed ablation artifact.

Covers: the plug-in estimator's relative standard error against the exact Gaussian formula and its
``1/sqrt(n)`` scaling; deterministic outlier injection (plan reproducibility, exact channel scaling,
identity at severity 1.0); the Fisher-z random-effects combination on a hand-computed example; a
determinism test for the component sweep; and structural validation of the committed artifact
(component/calibration/outlier/seed tables present, the cloud cells marked NOT RUN, the H1/H2/H4
verdicts carrying the frozen section 11 falsifier wording).
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
import torch

from spectraquant.evaluation.toy import TinyConfig, TinyTransformer
from spectraquant.proxies.ablations import (
    CANDIDATE,
    COMPONENT_VARIANTS,
    H1_FALSIFIER,
    H2_FALSIFIER,
    H4_FALSIFIER,
    NEW_VARIANT,
    REFERENCE_COMPARATOR,
    apply_outlier_injection,
    calibration_ids,
    calibration_shape,
    gaussian_relative_se,
    measurement_ids,
    outlier_injection_plan,
    plugin_relative_se,
    run_component_sweep,
)
from spectraquant.proxies.analysis import fisher_z_random_effects

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts" / "sample-results" / "ablations" / "ablations.json"


# ------------------------------------------------------------------ estimator relative SE


def test_plugin_relative_se_matches_the_exact_gaussian_formula() -> None:
    """For ``x ~ N(0, I)`` and ``delta = I`` the empirical relative SE tracks ``sqrt(2 d / n) / d``."""
    torch.manual_seed(0)
    n, d = 40_000, 8
    samples = torch.randn(n, d, dtype=torch.float64)
    # delta = I so the per-sample squared error is ||x_i||^2; G = I, tr(G) = d, tr(G^2) = d.
    squared = torch.sum(samples * samples, dim=-1)
    empirical = plugin_relative_se(squared)
    exact = gaussian_relative_se(trace_g=float(d), trace_g2=float(d), n=n)
    assert empirical == pytest.approx(exact, rel=0.05)
    assert exact == pytest.approx(math.sqrt(2.0 / (d * n)), rel=1e-12)


def test_plugin_relative_se_scales_as_one_over_sqrt_n() -> None:
    torch.manual_seed(1)
    base = torch.rand(4000, dtype=torch.float64) + 1.0
    small = plugin_relative_se(base[:250])
    large = plugin_relative_se(base[:1000])
    assert small == pytest.approx(2.0 * large, rel=0.2)


def test_plugin_relative_se_is_undefined_for_degenerate_input() -> None:
    assert math.isnan(plugin_relative_se([1.0]))
    assert math.isnan(plugin_relative_se([0.0, 0.0]))  # zero mean variance


# ------------------------------------------------------------------ outlier injection


def _fresh_model() -> TinyTransformer:
    return TinyTransformer(TinyConfig())


def test_outlier_plan_is_deterministic_and_well_formed() -> None:
    shapes = {name: layer.weight.shape[1] for name, layer in _fresh_model().linear_layers().items()}
    first = outlier_injection_plan(shapes, fraction=0.2, seed=7)
    second = outlier_injection_plan(shapes, fraction=0.2, seed=7)
    assert first == second
    assert set(first) == set(shapes)
    for name, channels in first.items():
        assert channels == sorted(channels)
        assert len(channels) == max(1, round(0.2 * shapes[name]))
        assert all(0 <= c < shapes[name] for c in channels)
    assert outlier_injection_plan(shapes, fraction=0.2, seed=8) != first


def test_apply_outlier_injection_is_deterministic_and_exact() -> None:
    model = _fresh_model()
    shapes = {name: layer.weight.shape[1] for name, layer in model.linear_layers().items()}
    plan = outlier_injection_plan(shapes, fraction=0.25, seed=3)

    one = copy.deepcopy(model)
    two = copy.deepcopy(model)
    apply_outlier_injection(one, plan, 8.0)
    apply_outlier_injection(two, plan, 8.0)
    for name in plan:
        assert torch.equal(one.linear_layers()[name].weight, two.linear_layers()[name].weight)

    # exact scaling on the selected columns, everything else untouched
    for name, channels in plan.items():
        before = model.linear_layers()[name].weight.detach()
        after = one.linear_layers()[name].weight.detach()
        index = torch.tensor(channels)
        non_spiked = torch.ones(before.shape[1], dtype=torch.bool)
        non_spiked[index] = False
        assert torch.equal(after[:, non_spiked], before[:, non_spiked])
        assert torch.allclose(after[:, index], before[:, index] * 8.0, rtol=0, atol=0)


def test_apply_outlier_injection_at_severity_one_is_the_identity() -> None:
    model = _fresh_model()
    shapes = {name: layer.weight.shape[1] for name, layer in model.linear_layers().items()}
    plan = outlier_injection_plan(shapes, seed=11)
    reference = copy.deepcopy(model)
    apply_outlier_injection(model, plan, 1.0)
    for name in plan:
        assert torch.equal(
            model.linear_layers()[name].weight, reference.linear_layers()[name].weight
        )


# ------------------------------------------------------------------ calibration ids


def test_calibration_shape_and_ids_are_deterministic() -> None:
    assert calibration_shape(8) == (1, 8)
    assert calibration_shape(512) == (32, 16)
    with pytest.raises(ValueError):
        calibration_shape(100)
    first = calibration_ids(24, 4, 16, seed=5)
    assert torch.equal(first, calibration_ids(24, 4, 16, seed=5))
    assert first.shape == (4, 16)
    assert int(first.min()) >= 0 and int(first.max()) < 24
    assert measurement_ids(24).shape == (2, 24)


# ------------------------------------------------------------------ Fisher-z combination


def test_fisher_z_combination_reduces_to_the_inverse_variance_mean() -> None:
    """With identical effects there is no between-study variance, so the pool is that effect."""
    result = fisher_z_random_effects([0.4, 0.4, 0.4], [0.04, 0.09, 0.16])
    assert result.pooled == pytest.approx(0.4, rel=1e-12)
    assert result.tau2 == 0.0
    weights = [1.0 / v for v in (0.04, 0.09, 0.16)]
    assert result.se == pytest.approx(1.0 / math.sqrt(sum(weights)), rel=1e-12)


def test_fisher_z_combination_reports_a_wider_interval_for_heterogeneous_effects() -> None:
    heterogeneous = fisher_z_random_effects([0.1, 0.6, 1.2], [0.05, 0.05, 0.05])
    assert heterogeneous.tau2 > 0.0
    assert heterogeneous.ci_low < heterogeneous.pooled < heterogeneous.ci_high


# ------------------------------------------------------------------ sweep determinism


@pytest.mark.integration
def test_component_sweep_is_deterministic_at_fixed_seeds() -> None:
    """Two sweeps at the same seeds reproduce the same measured document bit-for-bit.

    Only the wall-clock fields are allowed to differ.
    """
    kwargs = {
        "seeds": (0,),
        "cells": ((8, 4),),
        "fixtures": (("layernorm", True),),
        "train_steps": 20,
        "n_boot_neff": 10,
    }

    def science(runs: list[dict]) -> str:
        clone = json.loads(json.dumps(runs, sort_keys=True))
        for run in clone:
            run["training"].pop("wall_time_s")
        return json.dumps(clone, sort_keys=True)

    first = run_component_sweep(**kwargs)
    second = run_component_sweep(**kwargs)
    assert science(first) == science(second)
    assert NEW_VARIANT in first[0]["cells"][0]["variants"]
    assert REFERENCE_COMPARATOR in first[0]["cells"][0]["variants"]


# ------------------------------------------------------------------ the committed artifact


def _load_artifact() -> dict:
    assert ARTIFACT.exists(), f"missing artifact: {ARTIFACT}"
    text = ARTIFACT.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    return json.loads(text)


def test_committed_artifact_has_the_expected_skeleton() -> None:
    document = _load_artifact()
    assert document["schema"] == "spectraquant.proxy-ablations/1"
    assert document["substrate"] == "LOCAL-FIXTURE"
    assert document["measurement_class"] == 2
    assert document["statistical_unit"] == "model"
    for key in (
        "plan",
        "component_ablation",
        "calibration_size",
        "outliers",
        "seed_count",
        "hypotheses",
        "cloud_cells",
    ):
        assert key in document, f"artifact is missing {key}"


def test_committed_artifact_component_table_has_contrasts_with_cis() -> None:
    document = _load_artifact()
    analysis = document["component_ablation"]["analysis"]
    assert set(document["plan"]["component_variants"]) == set(COMPONENT_VARIANTS)
    assert document["plan"]["candidate"] == CANDIDATE
    assert document["plan"]["reference_comparator"] == REFERENCE_COMPARATOR
    for group in ("layernorm", "no_layernorm", "all_fixtures"):
        payload = analysis[group]
        for variant in COMPONENT_VARIANTS:
            assert payload["variants"][variant]["rho_mean"] is not None
        assert REFERENCE_COMPARATOR in payload["contrasts"]
        contrast = payload["contrasts"][REFERENCE_COMPARATOR]
        assert len(contrast["ci"]) == 2 and len(contrast["boot_ci"]) == 2
        assert contrast["ci"][0] < contrast["delta_z"] < contrast["ci"][1]
        assert contrast["verdict"]["verdict"] in {"supported", "refuted", "inconclusive"}


def test_committed_artifact_calibration_curve_has_at_least_four_sizes() -> None:
    document = _load_artifact()
    sizes = document["plan"]["calibration_sizes"]
    assert len(sizes) >= 4
    assert min(sizes) <= 8 and max(sizes) >= 512
    means = document["calibration_size"]["mean"]
    assert set(means) == {str(s) for s in sizes}
    ses = [means[str(s)]["estimator_rel_se_mean"] for s in sizes]
    assert all(v is not None and v > 0 for v in ses)
    assert ses[-1] < ses[0], "relative SE must fall as calibration grows"


def test_committed_artifact_outlier_table_increases_with_severity() -> None:
    document = _load_artifact()
    payload = document["outliers"]
    assert len(document["plan"]["outlier_severities"]) >= 3
    severities = sorted(float(s) for s in payload["mean"])
    assert severities[0] == 1.0
    first = payload["mean"][str(severities[0])]
    last = payload["mean"][str(severities[-1])]
    assert last["damage_sum"] > first["damage_sum"]
    assert first["rho_gain_aware_composed"] is not None
    assert last["rho_gain_aware_composed"] is not None


def test_committed_artifact_seed_count_grows_to_five() -> None:
    document = _load_artifact()
    rows = document["seed_count"]
    assert [row["seeds"] for row in rows] == [1, 2, 3, 4, 5]
    for row in rows:
        assert row["n_clusters"] == row["seeds"] * 2, "two fixtures per seed"
        assert len(row["ci"]) == 2 and row["ci"][0] < row["delta_z"] < row["ci"][1]
        assert row["verdict"]["verdict"] in {"supported", "refuted", "inconclusive"}


def test_committed_artifact_states_the_falsifiers_and_marks_tier12_not_run() -> None:
    document = _load_artifact()
    hypotheses = document["hypotheses"]
    assert set(hypotheses) == {"H1", "H2", "H4"}
    assert hypotheses["H1"]["falsifier"] == H1_FALSIFIER
    assert hypotheses["H2"]["falsifier"] == H2_FALSIFIER
    assert hypotheses["H4"]["falsifier"] == H4_FALSIFIER
    for payload in hypotheses.values():
        assert payload["verdict"] in {"supported", "contradicted", "inconclusive"}
    assert hypotheses["H2"]["tested_by_this_slice"] is True
    assert hypotheses["H1"]["verdict"] == "inconclusive"
    assert hypotheses["H4"]["verdict"] == "inconclusive"

    assert document["cloud_cells"], "the cloud cells must be listed"
    for cell in document["cloud_cells"].values():
        assert cell["status"] == "NOT RUN"
        assert cell["tier"] in {1, 2}
        assert cell["substrate"] in {"CLOUD-COLAB", "CLOUD-GPU"}
    tiers = {cell["tier"] for cell in document["cloud_cells"].values()}
    assert tiers == {1, 2}, "both Tier-1 and Tier-2 cells must be marked"
