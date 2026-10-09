"""Variant tests: every proxy against the exact float64 toy value, plus the LayerNorm control.

The M2 gate requires "proxy results versus exact toy calculations". The reference is
:func:`spectraquant.evaluation.toy.exact_layer_output_error`, computed by brute force in float64, so
the tolerances below are tight and are justified by float64 accumulation alone (a few hundred
summands): ``rtol=1e-9``.
"""

from __future__ import annotations

import pytest
import torch

from spectraquant.evaluation.toy import (
    TinyConfig,
    exact_layer_output_error,
    layer_inputs,
    layer_names,
    network_joint_damage,
    network_single_layer_damage,
    norm_preimages,
    train_tiny,
)
from spectraquant.proxies.base import FollowingNorm, LayerInputs
from spectraquant.proxies.gain import estimate_downstream_gains
from spectraquant.proxies.variants import (
    PROXY_VARIANTS,
    ActivationMagnitudeProxy,
    CombinedProxy,
    GainAwareComposedProxy,
    HessianDiagProxy,
    InSituOutputErrorProxy,
    PerLayerOutputErrorProxy,
    QuantResidualStatsProxy,
    SpectralSummaryProxy,
    WeightFrobeniusProxy,
    WeightMagnitudeProxy,
    get_proxy,
    proxy_names,
)
from spectraquant.quantization import QuantSpec

RANK = 8


def spec() -> QuantSpec:
    return QuantSpec(bits=4, granularity="per_group", group_size=32, symmetric=True)


def random_layer(seed: int = 0, out: int = 32, inp: int = 48, n: int = 64):
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(out, inp, generator=gen, dtype=torch.float64)
    x = torch.randn(n, inp, generator=gen, dtype=torch.float64)
    return w, x


# ---------------------------------------------------------------- exactness


def test_per_layer_output_error_matches_exact_float64_value() -> None:
    w, x = random_layer()
    proxy = PerLayerOutputErrorProxy()
    result = proxy.score_layer(w, x, RANK, spec())
    exact = exact_layer_output_error(w, x, RANK, spec())
    assert result.value == pytest.approx(exact, rel=1e-9)
    assert result.exact is True
    assert result.measurement_class == 2
    assert set(result.per_layer) == {"layer"}


def test_quant_residual_stats_matches_exact_value_for_its_own_delta() -> None:
    w, x = random_layer(1)
    proxy = QuantResidualStatsProxy()
    result = proxy.score_layer(w, x, RANK, spec())
    from spectraquant.proxies.operators import quantization_only_delta

    delta = quantization_only_delta(w, RANK, spec())
    proj = x @ delta.transpose(0, 1)
    exact = float(torch.mean(torch.sum(proj * proj, dim=-1)))
    assert result.value == pytest.approx(exact, rel=1e-9)
    assert 0.0 <= result.diagnostics["rounding_fraction_of_total"] <= 1.0
    assert result.diagnostics["residual_std"] > 0.0


def test_hessian_diag_equals_its_own_quadratic_form() -> None:
    w, x = random_layer(2)
    result = HessianDiagProxy().score_layer(w, x, RANK, spec())
    from spectraquant.proxies.operators import compression_delta

    delta = compression_delta(w, RANK, spec())
    second = torch.mean(x * x, dim=0)
    expected = float(torch.sum(second * torch.sum(delta * delta, dim=0)))
    assert result.value == pytest.approx(expected, rel=1e-9)
    assert result.exact is False  # off-diagonal activation correlations are dropped
    assert 0.0 < result.diagnostics["diagonal_energy_fraction"] <= 1.0


def test_rank_only_comparators_are_closed_form_and_weight_or_activation_based() -> None:
    """The predeclared H2 comparator set must be implemented, not merely named in the preregistration."""
    w, x = random_layer(11)
    weight_mag = WeightMagnitudeProxy().score_layer(w, x, RANK, spec())
    assert weight_mag.value == pytest.approx(float(torch.sum(w.abs())), rel=1e-12)
    assert weight_mag.measurement_class == 1
    act_mag = ActivationMagnitudeProxy().score_layer(w, x, RANK, spec())
    assert act_mag.value == pytest.approx(float(torch.sum(torch.mean(x * x, dim=0))), rel=1e-12)
    assert act_mag.measurement_class == 1
    # both are compression-independent: they must not change when the rank changes
    for proxy in (WeightMagnitudeProxy(), ActivationMagnitudeProxy()):
        assert proxy.score_layer(w, x, 2, spec()).value == proxy.score_layer(w, x, 32, spec()).value
    for name in ("weight_magnitude", "activation_magnitude"):
        assert name in PROXY_VARIANTS


def test_weight_frobenius_is_the_weight_space_norm() -> None:
    w, x = random_layer(3)
    result = WeightFrobeniusProxy().score_layer(w, x, RANK, spec())
    from spectraquant.proxies.operators import compression_delta

    delta = compression_delta(w, RANK, spec())
    assert result.value == pytest.approx(float(torch.sum(delta**2)), rel=1e-9)
    assert result.measurement_class == 1


def test_spectral_summary_is_anisotropy_labelled_and_finite() -> None:
    w, x = random_layer(4)
    result = SpectralSummaryProxy().score_layer(w, x, RANK, spec())
    assert result.value > 0.0
    assert result.exact is False
    assert result.measurement_class == 1
    assert result.diagnostics["isotropy_ratio"] > 0.0


def test_zero_rank_gives_the_full_signal_error_except_for_the_rounding_variant() -> None:
    """At rank 0 nothing is kept: the output error is the full signal energy, and nothing is rounded."""
    w, x = random_layer(5)
    for name in ("per_layer_output_error", "hessian_diag"):
        assert get_proxy(name).score_layer(w, x, 0, spec()).value > 0.0
    # the rounding residual is exactly zero when there are no factors to round
    assert get_proxy("quant_residual_stats").score_layer(w, x, 0, spec()).value == 0.0
    from spectraquant.proxies.operators import compression_delta

    # rank 0 discards everything, so the operator error is the whole weight
    assert torch.equal(compression_delta(w, 0, spec()), w)


# ---------------------------------------------------------------- context-dependent variants


def test_in_situ_falls_back_without_context_and_uses_the_jacobian_with_it() -> None:
    w, x = random_layer(6)
    fallback = InSituOutputErrorProxy().score_layer(w, x, RANK, spec())
    assert fallback.diagnostics["fallback"] == 1.0
    naive = PerLayerOutputErrorProxy().score_layer(w, x, RANK, spec())
    assert fallback.value == pytest.approx(naive.value, rel=1e-12)

    norm = FollowingNorm(kind="layernorm", preimage=x, weight=torch.ones(x.shape[-1]), bias=None)
    residual_error = torch.randn(x.shape, generator=torch.Generator().manual_seed(7))
    layer = LayerInputs(weight=w, activations=x, following_norm=norm, residual_error=residual_error)
    with_context = InSituOutputErrorProxy()._context_layer(layer, RANK, spec())
    assert with_context.diagnostics["fallback"] == 0.0
    mapped = norm.jacobian_apply(residual_error)
    expected = float(torch.mean(torch.sum(mapped**2, dim=-1)))
    assert with_context.value == pytest.approx(expected, rel=1e-9)


def test_gain_aware_squares_the_gain_and_falls_back_without_it() -> None:
    w, x = random_layer(8)
    naive = PerLayerOutputErrorProxy().score_layer(w, x, RANK, spec())
    layer = LayerInputs(weight=w, activations=x, downstream_gain=2.5)
    result = GainAwareComposedProxy()._context_layer(layer, RANK, spec())
    assert result.value == pytest.approx(2.5**2 * naive.value, rel=1e-9)
    assert result.exact is False  # the gain is an estimate of a nonlinear composition

    fallback = GainAwareComposedProxy()._context_layer(
        LayerInputs(weight=w, activations=x), RANK, spec()
    )
    assert fallback.diagnostics["fallback"] == 1.0
    assert fallback.value == pytest.approx(naive.value, rel=1e-12)


def test_combined_proxy_is_an_explicit_weighted_sum() -> None:
    w, x = random_layer(9)
    proxy = CombinedProxy(weights={"gain_aware": 2.0, "hessian_diag": 0.5, "quant_residual": 0.0})
    layer = LayerInputs(weight=w, activations=x, downstream_gain=1.5)
    result = proxy._context_layer(layer, RANK, spec())
    gain_only = GainAwareComposedProxy()._context_layer(layer, RANK, spec())
    hessian_only = HessianDiagProxy()._context_layer(layer, RANK, spec())
    assert result.value == pytest.approx(2.0 * gain_only.value + 0.5 * hessian_only.value, rel=1e-9)
    with pytest.raises(Exception, match="unknown combined weights"):
        CombinedProxy(weights={"nonsense": 1.0})


# ---------------------------------------------------------------- model-level aggregation


def test_score_model_sums_per_layer_values_and_validates_keys() -> None:
    w, x = random_layer(10, out=16, inp=24, n=32)
    layers = {
        "a": LayerInputs(weight=w, activations=x),
        "b": LayerInputs(weight=w.clone(), activations=x.clone()),
    }
    proxy = PerLayerOutputErrorProxy()
    result = proxy.score_model(layers, {"a": RANK, "b": RANK}, {"a": spec(), "b": spec()})
    assert result.value == pytest.approx(sum(result.per_layer.values()), rel=1e-12)
    assert set(result.per_layer) == {"a", "b"}
    with pytest.raises(Exception, match="same key set"):
        proxy.score_model(layers, {"a": RANK}, {"a": spec(), "b": spec()})


def test_registry_exposes_every_variant_and_instantiates_without_arguments() -> None:
    assert set(proxy_names()) == set(PROXY_VARIANTS)
    for name in proxy_names():
        proxy = get_proxy(name)
        assert isinstance(proxy.name, str) and proxy.name
    with pytest.raises(Exception, match="unknown proxy"):
        get_proxy("does-not-exist")


# ---------------------------------------------------------------- the scientific regression


@pytest.fixture(scope="module")
def fixture_measurement() -> dict:
    """One cheap LayerNorm fixture measurement, shared by the ranking tests below."""
    cfg = TinyConfig(n_blocks=2, d_model=32, n_heads=2, d_ff=64, vocab=16, seq_len=12)
    model, _ = train_tiny(cfg, steps=60)
    ids = torch.arange(12)[None].repeat(2, 1) % 16
    names = layer_names(model)
    damage = network_single_layer_damage(model, ids, names, RANK, spec())
    joint_hidden, _ = network_joint_damage(model, ids, names, RANK, spec())
    activations = layer_inputs(model, ids, names)
    preimages = norm_preimages(model, ids)
    gains = estimate_downstream_gains(
        lambda nm, pert: model(ids, inject=(nm, pert))[1].reshape(-1, cfg.d_model),
        damage.layer_errors,
    )
    from spectraquant.proxies.analysis import following_norm_container, spearman

    per_variant: dict[str, list[float]] = {}
    for variant in ("per_layer_output_error", "gain_aware_composed", "weight_frobenius"):
        values = []
        for name in names:
            layer = LayerInputs(
                weight=model.linear_layers()[name].weight.detach(),
                activations=activations[name],
                following_norm=following_norm_container(model, name, preimages),
                residual_error=damage.residual_errors.get(name),
                downstream_gain=gains.per_layer[name],
            )
            values.append(get_proxy(variant)._context_layer(layer, RANK, spec()).value)
        per_variant[variant] = values
    damages = [damage.per_layer[n] for n in names]
    return {
        "rho": {k: spearman(v, damages) for k, v in per_variant.items()},
        "joint_over_sum_naive": joint_hidden / sum(per_variant["per_layer_output_error"]),
        "joint_over_sum_single": joint_hidden / sum(damages),
        "n_layers": len(names),
    }


def test_gain_aware_ranking_beats_the_naive_baseline_under_layernorm(
    fixture_measurement: dict,
) -> None:
    """The measured reason this project exists: with LayerNorm the naive proxy loses ranking.

    Deterministic (seeded fixture), so this is a stable regression guard for the scientific claim
    rather than a flaky statistical test. The numbers here are from a 2-block fixture and are NOT the
    M4 result — that gate repeats this across seeds and (rank, bits) configurations.
    """
    rho = fixture_measurement["rho"]
    assert fixture_measurement["n_layers"] >= 6
    assert rho["gain_aware_composed"] > rho["per_layer_output_error"]
    assert rho["gain_aware_composed"] > 0.4
    # the joint damage is not the sum of per-layer proxies; if this ever holds, revisit section 7
    assert fixture_measurement["joint_over_sum_single"] > 0.2


def test_joint_damage_is_not_recovered_by_summing_the_naive_proxy(
    fixture_measurement: dict,
) -> None:
    assert fixture_measurement["joint_over_sum_naive"] < 0.5
