"""Unit tests for the real-model H2 ranking measurement (``spectraquant.proxies.h2_ranking``).

The tests use a **tiny synthetic model fixture** (a handful of small linear layers) so they are fast
and deterministic on the CPU, and they assert the ranking statistic against hand-constructed cases
whose true ordering is known exactly. No model download, no GPU, no training.
"""

from __future__ import annotations

import math

import pytest
import torch

from spectraquant.proxies.analysis import fisher_z, inverse_fisher_z
from spectraquant.proxies.base import LayerInputs
from spectraquant.proxies.h2_ranking import (
    CANDIDATE,
    DEFAULT_VARIANTS,
    MEASUREMENT_CLASS,
    SCHEMA_ID,
    CompressionVariant,
    build_document,
    compressed_weight,
    contrast_payload,
    layer_output_damage,
    measure_cell,
    rank_cell,
    variant_spec,
)

N_LAYERS = 24


def _fixture(seed: int = 0) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """A tiny synthetic model: seeded weights and seeded activations, one per 'layer'.

    Shapes and per-layer scales are drawn from the seeded generator, so the per-layer damages (and
    the proxies) are genuinely heterogeneous and the ranking statistic is not degenerate.
    """
    generator = torch.Generator().manual_seed(seed)
    weights: dict[str, torch.Tensor] = {}
    activations: dict[str, torch.Tensor] = {}
    for index in range(N_LAYERS):
        out_features = int(torch.randint(6, 16, (1,), generator=generator).item())
        in_features = 16
        weight_scale = 0.05 * math.exp(float(torch.randn(1, generator=generator).item()))
        activation_scale = 0.4 * math.exp(float(torch.randn(1, generator=generator).item()))
        weights[f"layer{index}"] = (
            torch.randn(out_features, in_features, generator=generator, dtype=torch.float64)
            * weight_scale
        )
        activations[f"layer{index}"] = (
            torch.randn(40, in_features, generator=generator, dtype=torch.float64)
            * activation_scale
        )
    return weights, activations


def _layer_inputs(
    weights: dict[str, torch.Tensor], activations: dict[str, torch.Tensor]
) -> dict[str, LayerInputs]:
    """Layer inputs carrying a measured downstream gain, so the composed variants do not fall back."""
    return {
        name: LayerInputs(
            weight=weights[name],
            activations=activations[name],
            downstream_gain=0.5 + 0.4 * index,
        )
        for index, name in enumerate(sorted(weights))
    }


# ------------------------------------------------------------------------------------------------
# damage and compression operators
# ------------------------------------------------------------------------------------------------


def test_layer_output_damage_matches_hand_computation() -> None:
    """The measured damage equals ``mean_i ||x_i (W - W_hat)^T||^2`` computed by hand."""
    torch.manual_seed(0)
    w = torch.randn(6, 8, dtype=torch.float64)
    x = torch.randn(5, 8, dtype=torch.float64)
    variant = CompressionVariant(name="v", kind="rank_quant", rank=3, bits=4)
    w_hat = compressed_weight(w, variant)
    delta = w - w_hat
    proj = x @ delta.transpose(0, 1)
    expected = float(torch.mean(torch.sum(proj * proj, dim=-1)))
    assert layer_output_damage(w, x, variant) == pytest.approx(expected, rel=1e-12, abs=1e-15)
    # A different variant must produce a different (larger) damage than the near-lossless one.
    lossless = CompressionVariant(name="lossless", kind="rank", rank=0, bits=8)
    assert layer_output_damage(w, x, variant) > layer_output_damage(w, x, lossless)


def test_full_rank_variant_is_factor_quantization_only() -> None:
    """``rank=0`` means full rank: the truncation is exact, so the damage is the rounding error."""
    torch.manual_seed(1)
    w = torch.randn(7, 9, dtype=torch.float64)
    full = CompressionVariant(name="full", kind="quant", rank=0, bits=4)
    assert variant_spec(full).bits == 4
    w_hat = compressed_weight(w, full)
    # full-rank SVD reconstructs exactly, so W_hat differs from W only through the quantizer
    from spectraquant.factorization import truncated_svd
    from spectraquant.quantization import fake_quantize

    factors = truncated_svd(w, min(w.shape))
    expected = fake_quantize(factors.B, variant_spec(full)) @ fake_quantize(
        factors.A, variant_spec(full)
    )
    assert torch.allclose(w_hat, expected, atol=1e-12)


def test_variant_validation() -> None:
    with pytest.raises(ValueError):
        CompressionVariant(name="bad", kind="nope", rank=1, bits=4)
    with pytest.raises(ValueError):
        CompressionVariant(name="bad", kind="rank", rank=-1, bits=4)
    with pytest.raises(ValueError):
        CompressionVariant(name="bad", kind="rank", rank=1, bits=4, group_size=0)


# ------------------------------------------------------------------------------------------------
# the model-level ranking statistic
# ------------------------------------------------------------------------------------------------


def test_rank_cell_perfect_and_anti_correlation() -> None:
    damages = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    perfect = rank_cell(damages, damages, n_boot=32, seed=0)
    assert perfect.rho == pytest.approx(1.0)
    assert perfect.z == pytest.approx(fisher_z(1.0))
    assert perfect.clipped is True
    anti = rank_cell(list(reversed(damages)), damages, n_boot=32, seed=0)
    assert anti.rho == pytest.approx(-1.0)
    assert anti.z == pytest.approx(fisher_z(-1.0))


def test_rank_cell_hand_computed_spearman() -> None:
    """A known permutation: rho = 1 - 6 * sum(d^2) / (n(n^2-1)) = 0.9."""
    proxy = [1.0, 2.0, 3.0, 4.0, 5.0]
    damages = [1.0, 2.0, 3.0, 5.0, 4.0]
    result = rank_cell(proxy, damages, n_boot=64, seed=7)
    assert result.rho == pytest.approx(0.9, abs=1e-12)
    assert result.z == pytest.approx(fisher_z(0.9))
    assert result.n_modules == 5
    assert math.isfinite(result.n_eff)
    assert result.variance == pytest.approx(1.0 / (result.n_eff - 3.0))


def test_rank_cell_degenerate_inputs_are_unusable() -> None:
    constant = rank_cell([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0], n_boot=16, seed=0)
    assert math.isnan(constant.rho)
    assert constant.usable is False
    too_few = rank_cell([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], n_boot=16, seed=0)
    assert too_few.usable is False
    with pytest.raises(ValueError):
        rank_cell([1.0, 2.0], [1.0, 2.0, 3.0])


def test_rank_cell_is_deterministic() -> None:
    generator = torch.Generator().manual_seed(3)
    proxy = torch.randn(20, generator=generator).tolist()
    damages = torch.randn(20, generator=generator).tolist()
    first = rank_cell(proxy, damages, n_boot=128, seed=42)
    second = rank_cell(proxy, damages, n_boot=128, seed=42)
    assert first == second


# ------------------------------------------------------------------------------------------------
# per-cell measurement through the public proxy entry point
# ------------------------------------------------------------------------------------------------


def test_measure_cell_naive_proxy_is_tautological_and_weight_proxy_is_not() -> None:
    """``per_layer_output_error`` is the exact plug-in estimator of the measured damage."""
    weights, activations = _fixture()
    layers = _layer_inputs(weights, activations)
    variant = DEFAULT_VARIANTS[0]
    damages = {
        name: layer_output_damage(weights[name], activations[name], variant) for name in weights
    }
    cell = measure_cell(layers, damages, variant, n_boot_neff=64, neff_seed=1)
    assert cell["variant"] == variant.name
    assert cell["n_modules"] == N_LAYERS
    naive = cell["variants"]["per_layer_output_error"]
    assert naive["rho"] == pytest.approx(1.0)
    assert naive["fallback_fraction"] == 0.0
    # A weight-space comparator is not in the common unit, so it must not be a perfect ranker.
    frobenius = cell["variants"]["weight_frobenius"]
    assert math.isfinite(frobenius["rho"])
    assert abs(frobenius["rho"]) <= 1.0


def test_measure_cell_damage_summary_matches_hand_values() -> None:
    """The cell's damage summary is the hand-computed summary of the damage mapping it was given."""
    weights, activations = _fixture()
    layers = _layer_inputs(weights, activations)
    variant = DEFAULT_VARIANTS[2]
    names = sorted(weights)
    damages = {name: float(index + 1) for index, name in enumerate(names)}
    cell = measure_cell(layers, damages, variant, n_boot_neff=64, neff_seed=2)
    assert cell["damage"]["sum"] == pytest.approx(sum(damages.values()))
    assert cell["damage"]["min"] == pytest.approx(1.0)
    assert cell["damage"]["max"] == pytest.approx(float(len(names)))
    assert cell["damage"]["mean"] == pytest.approx(sum(damages.values()) / len(names))


def test_measure_cell_rejects_mismatched_keys() -> None:
    weights, activations = _fixture()
    layers = _layer_inputs(weights, activations)
    with pytest.raises(ValueError):
        measure_cell(layers, {"layer0": 1.0}, DEFAULT_VARIANTS[0])


# ------------------------------------------------------------------------------------------------
# model-level contrast and document
# ------------------------------------------------------------------------------------------------


def test_contrast_payload_hand_computed_difference() -> None:
    """Two cells with known Fisher-z differences give the hand-computed random-effects pool."""
    z_cand = [1.0, 1.2]
    z_comp = [0.4, 0.2]
    var = 0.25
    cells = [
        {
            "variant": f"cell{i}",
            "variants": {
                "cand": {"z": z, "variance": var, "usable": True, "rho": 0.0, "n_eff": 7.0},
                "comp": {"z": k, "variance": var, "usable": True, "rho": 0.0, "n_eff": 7.0},
            },
        }
        for i, (z, k) in enumerate(zip(z_cand, z_comp, strict=True))
    ]
    payload = contrast_payload(
        cells, "cand", "comp", model_id="m", n_boot_ci=64, boot_seed=0, n_eff_ref=7.0
    )
    assert payload["delta_z"] == pytest.approx((0.6 + 1.0) / 2.0)
    assert payload["k"] == 2
    assert payload["usable"] is True
    assert payload["model_bootstrap_degenerate"] is True
    low, high = payload["ci"]
    assert low < payload["delta_z"] < high
    assert payload["mde_achieved_z"] is not None


def test_contrast_payload_empty_when_no_usable_cells() -> None:
    payload = contrast_payload(
        [], "cand", "comp", model_id="m", n_boot_ci=16, boot_seed=0, n_eff_ref=None
    )
    assert payload["usable"] is False


def test_build_document_labels_and_shape() -> None:
    weights, activations = _fixture()
    layers = _layer_inputs(weights, activations)
    cells = []
    for index, variant in enumerate(DEFAULT_VARIANTS[:2]):
        damages = {
            name: layer_output_damage(weights[name], activations[name], variant) for name in weights
        }
        cells.append(measure_cell(layers, damages, variant, n_boot_neff=64, neff_seed=index))
    document = build_document(
        {"layer_output": cells},
        target_notes={"layer_output": "the layer's own output error"},
        model={"id": "fixture", "revision": "synthetic"},
        calibration={"role": "train", "documents": 1, "tokens": 40, "seed": 0},
        variants=DEFAULT_VARIANTS[:2],
        n_boot_neff=64,
        n_boot_ci=64,
        wall_time_s=0.5,
    )
    assert document["schema"] == SCHEMA_ID
    assert document["measurement_class"] == MEASUREMENT_CLASS == 2
    assert document["substrate"] == "LOCAL-CPU"
    assert document["statistical_unit"] == "model"
    assert document["runtime"]["n_models"] == 1
    assert document["runtime"]["n_modules"] == N_LAYERS
    target = document["targets"]["layer_output"]
    assert target["description"] == "the layer's own output error"
    assert len(target["per_cell"]) == 2
    assert document["plan"]["candidate_proxy"] == CANDIDATE
    assert "per_layer_output_error" in target["aggregate"]["variants"]
    assert "weight_magnitude" in target["aggregate"]["variants"]
    contrasts = target["aggregate"]["contrasts"][CANDIDATE]
    assert "weight_frobenius" in contrasts
    assert set(contrasts["weight_frobenius"]) >= {
        "delta_z",
        "ci",
        "cell_boot_ci",
        "usable",
        "verdict",
    }
    assert any("class 2" in line for line in document["limitations"])


def test_build_document_tolerates_partial_proxy_selection() -> None:
    """A partial ``--proxies`` selection is a smaller study, not a crash."""
    weights, activations = _fixture()
    layers = _layer_inputs(weights, activations)
    variant = DEFAULT_VARIANTS[0]
    damages = {
        name: layer_output_damage(weights[name], activations[name], variant) for name in weights
    }
    cells = [
        measure_cell(
            layers,
            damages,
            variant,
            proxies=["combined", "weight_frobenius"],
            n_boot_neff=32,
            neff_seed=0,
        )
    ]
    document = build_document(
        {"layer_output": cells},
        model={"id": "fixture"},
        calibration={},
        variants=DEFAULT_VARIANTS[:1],
        proxies=["combined", "weight_frobenius"],
        n_boot_ci=16,
    )
    aggregate = document["targets"]["layer_output"]["aggregate"]
    assert set(aggregate["contrasts"]) == {"combined"}  # gain_aware_composed was not scored
    assert set(aggregate["contrasts"]["combined"]) == {"weight_frobenius"}
    assert aggregate["primary_comparator"] == "weight_frobenius"
    assert aggregate["contrasts"]["combined"]["weight_frobenius"]["usable"] is True


def test_inverse_fisher_round_trip() -> None:
    for rho in (-0.9, -0.3, 0.0, 0.5, 0.99):
        assert inverse_fisher_z(fisher_z(rho)) == pytest.approx(rho, abs=1e-9)
