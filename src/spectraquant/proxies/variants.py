"""Proxy variants (``design-m2-interfaces.md`` section 3 as amended by section 7).

Every variant returns :class:`~spectraquant.proxies.base.ProxyResult`. The **common unit** is the
plug-in average of the layer-output squared error
``mean_i ||x_i W^T - x_i W_hat^T||^2`` for the calibration activations, i.e.
``tr(delta Sigma_X delta^T)`` estimated on a finite batch. Two exceptions are documented in place and
never compared by value against the unit: ``weight_frobenius`` (weight space, the H2 comparator) and
``spectral_summary`` (an isotropic-activation analytical estimate).

Why there are several variants (adversarial review section 3, measured):
    With LayerNorm present, the naive pre-normalisation per-layer output error correlates 0.119
    (r=32, b=4) and 0.137 (r=32, b=8) with downstream damage, versus 0.813 with LayerNorm replaced
    by Identity; per-layer gain spreads 33-110x; and the summed proxy is 1e-4..4.9e-2 of the joint
    damage. The naive form is therefore a **baseline whose failure is published**, and the candidate
    variants add (i) the in-situ normalisation Jacobian and (ii) an estimated downstream gain.

Measurement classes (``AGENTS.md`` section 5):
    all variants are class 2 (fake-quantization quality) except where a variant is purely analytical
    over the weight spectrum, which is class 1. Nothing here is class 3-5.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from spectraquant.proxies.base import (
    FollowingNorm,
    LayerInputs,
    ProxyInputError,
    ProxyResult,
)
from spectraquant.proxies.operators import (
    compression_delta,
    low_rank_delta,
    quantization_only_delta,
)
from spectraquant.quantization import QuantSpec

__all__ = [
    "PROXY_VARIANTS",
    "ActivationMagnitudeProxy",
    "CombinedProxy",
    "GainAwareComposedProxy",
    "HessianDiagProxy",
    "InSituOutputErrorProxy",
    "PerLayerOutputErrorProxy",
    "QuantResidualStatsProxy",
    "SpectralSummaryProxy",
    "WeightFrobeniusProxy",
    "WeightMagnitudeProxy",
    "get_proxy",
]


def _output_error(w: torch.Tensor, x: torch.Tensor, delta: torch.Tensor) -> float:
    """``mean_i ||x_i delta^T||^2`` in float64, the plug-in estimator of ``tr(delta Sigma_X delta^T)``."""
    proj = x.to(torch.float64) @ delta.to(torch.float64).transpose(0, 1)
    return float(torch.mean(torch.sum(proj * proj, dim=-1)))


def _sampling_diagnostics(delta: torch.Tensor, x: torch.Tensor, value: float) -> dict[str, float]:
    """Per-layer sampling information (count, relative standard error) for the plug-in estimator.

    For ``x ~ N(0, Sigma)`` the exact relative standard error of the estimator is
    ``sqrt(2 * tr(G^2) / n) / tr(G)`` with ``G = delta Sigma delta^T``; we report the empirical
    counterpart computed from the batch, which is finite-sample safe and matches the exact value on
    Gaussian draws (verified in ``tests/unit/test_proxy_variants.py``).
    """
    proj = x.to(torch.float64) @ delta.to(torch.float64).transpose(0, 1)
    per_sample = torch.sum(proj * proj, dim=-1)
    n = int(per_sample.numel())
    out = {"n_samples": float(n)}
    if n > 1 and value > 0.0:
        # var of the per-sample squared norms / n, relative to the mean squared
        var = float(torch.var(per_sample, unbiased=True))
        out["sampling_rel_se"] = float((var / n) ** 0.5 / value)
    out["mean_output_error"] = value
    return out


class _BaseVariant:
    """Shared plumbing: model-level scoring and result construction.

    A plain class (not a dataclass) so every variant can declare its own ``name`` as a class
    attribute and be instantiated with no arguments — the registry in :data:`PROXY_VARIANTS` relies
    on that.
    """

    #: Stable variant name; overridden by each subclass and by ``CombinedProxy`` instances.
    name: str = "base"

    def score_layer(
        self, w: torch.Tensor, x: torch.Tensor, rank: int, spec: QuantSpec
    ) -> ProxyResult:
        raise NotImplementedError

    def _context_layer(self, layer: LayerInputs, rank: int, spec: QuantSpec) -> ProxyResult:
        """Context-rich per-layer scoring used by :meth:`score_model`."""
        return self.score_layer(layer.weight, layer.activations, rank, spec)

    def score_model(
        self,
        layers: Mapping[str, LayerInputs],
        ranks: Mapping[str, int],
        specs: Mapping[str, QuantSpec],
    ) -> ProxyResult:
        if set(layers) != set(ranks) or set(layers) != set(specs):
            raise ProxyInputError(
                "layers, ranks and specs must share the same key set: "
                f"{sorted(layers)} vs {sorted(ranks)} vs {sorted(specs)}"
            )
        per_layer: dict[str, float] = {}
        diagnostics: dict[str, float] = {}
        exact = True
        classes: set[int] = set()
        for name, layer in layers.items():
            result = self._context_layer(layer, ranks[name], specs[name])
            per_layer[name] = result.value
            exact = exact and result.exact
            classes.add(result.measurement_class)
            for key, value in result.diagnostics.items():
                diagnostics[f"{name}/{key}"] = value
        return ProxyResult(
            value=float(sum(per_layer.values())),
            exact=exact,
            measurement_class=max(classes),
            per_layer=per_layer,
            diagnostics=diagnostics,
        )


class WeightFrobeniusProxy(_BaseVariant):
    """``||W - W_hat||_F^2`` — the weight-space baseline (the predeclared H2 comparator).

    NOT in the common unit: this is a weight-space quantity. It is reported so its rank correlation
    with damage can be compared against the output-aware variants; its *value* must never be compared
    against an output-error figure (``design-m2-interfaces.md`` section 0.3).
    """

    name = "weight_frobenius"

    def score_layer(self, w, x, rank, spec):
        delta = compression_delta(w, rank, spec)
        value = float(torch.sum(delta.to(torch.float64) ** 2))
        return ProxyResult(
            value=value,
            exact=True,
            measurement_class=1,
            per_layer={"layer": value},
            diagnostics={"weight_delta_norm_sq": value, "n_params": float(w.numel())},
        )


class PerLayerOutputErrorProxy(_BaseVariant):
    """The naive pre-normalisation per-layer output error — the baseline whose failure we publish.

    ``mean_i ||x_i delta^T||^2`` with ``delta = W - Q(B) Q(A)``. Exact for the scored batch, and
    measured to lose its ranking ability once a normalisation layer sits between the layer and the
    model output (adversarial review section 3.2).
    """

    name = "per_layer_output_error"

    def score_layer(self, w, x, rank, spec):
        delta = compression_delta(w, rank, spec)
        value = _output_error(w, x, delta)
        return ProxyResult(
            value=value,
            exact=True,
            measurement_class=2,
            per_layer={"layer": value},
            diagnostics=_sampling_diagnostics(delta, x, value),
        )


class InSituOutputErrorProxy(_BaseVariant):
    """The layer error mapped through the **following normalisation's exact Jacobian**.

    ``mean_i ||J_N(v_i) e_i||^2`` where ``e`` is the residual-stream error the layer's own
    compression produced and ``v`` is the normalisation's preimage on the uncompressed forward pass
    (both supplied by the calibration driver). Falls back to the context-free
    :class:`PerLayerOutputErrorProxy` form — with ``fallback=True`` in ``diagnostics`` — when the
    context is absent, because ``score_layer``'s frozen signature cannot carry it.
    """

    name = "in_situ_output_error"

    def _context_layer(self, layer: LayerInputs, rank: int, spec: QuantSpec) -> ProxyResult:
        if layer.residual_error is None or layer.following_norm is None:
            result = PerLayerOutputErrorProxy().score_layer(
                layer.weight, layer.activations, rank, spec
            )
            diagnostics = dict(result.diagnostics)
            diagnostics["fallback"] = 1.0
            return ProxyResult(
                value=result.value,
                exact=result.exact,
                measurement_class=result.measurement_class,
                per_layer=result.per_layer,
                diagnostics=diagnostics,
            )
        norm: FollowingNorm = layer.following_norm
        mapped = norm.jacobian_apply(layer.residual_error)
        per_sample = torch.sum(mapped.to(torch.float64) ** 2, dim=-1)
        value = float(torch.mean(per_sample))
        n = int(per_sample.numel())
        diagnostics = {"n_samples": float(n), "fallback": 0.0}
        if n > 1 and value > 0.0:
            diagnostics["sampling_rel_se"] = float(
                (float(torch.var(per_sample, unbiased=True)) / n) ** 0.5 / value
            )
        # The activation-distribution shift the proxy is exposed to (section 7.5): ratio of the
        # joint-compression activation scale to the uncompressed one.
        if layer.joint_activations is not None:
            base = float(torch.mean(layer.activations.to(torch.float64) ** 2))
            joint = float(torch.mean(layer.joint_activations.to(torch.float64) ** 2))
            if base > 0.0:
                diagnostics["activation_shift_ratio"] = joint / base
        return ProxyResult(
            value=value,
            exact=True,
            measurement_class=2,
            per_layer={"layer": value},
            diagnostics=diagnostics,
        )

    def score_layer(self, w, x, rank, spec):
        # The frozen score_layer signature cannot carry the normalisation context, so the
        # context-free call is the naive form and must say so (the docstring promises `fallback=1`).
        result = PerLayerOutputErrorProxy().score_layer(w, x, rank, spec)
        diagnostics = dict(result.diagnostics)
        diagnostics["fallback"] = 1.0
        return ProxyResult(
            value=result.value,
            exact=result.exact,
            measurement_class=result.measurement_class,
            per_layer=result.per_layer,
            diagnostics=diagnostics,
        )


class GainAwareComposedProxy(_BaseVariant):
    """Per-layer error scaled by the **estimated downstream gain**: ``g^2 * mean_i ||x_i delta^T||^2``.

    ``g`` is the estimated RMS ratio between the model-output error and this layer's own error,
    produced by :func:`spectraquant.proxies.gain.estimate_downstream_gains`. Squaring is required
    because the unit is a squared error. Without ``downstream_gain`` the variant degrades to the
    naive form and says so (``fallback=1``).
    """

    name = "gain_aware_composed"

    def _context_layer(self, layer: LayerInputs, rank: int, spec: QuantSpec) -> ProxyResult:
        base = PerLayerOutputErrorProxy().score_layer(layer.weight, layer.activations, rank, spec)
        if layer.downstream_gain is None:
            diagnostics = dict(base.diagnostics)
            diagnostics["fallback"] = 1.0
            diagnostics["downstream_gain"] = 0.0
            return ProxyResult(
                value=base.value,
                exact=base.exact,
                measurement_class=base.measurement_class,
                per_layer=base.per_layer,
                diagnostics=diagnostics,
            )
        gain = float(layer.downstream_gain)
        value = gain * gain * base.value
        diagnostics = dict(base.diagnostics)
        diagnostics["fallback"] = 0.0
        diagnostics["downstream_gain"] = gain
        return ProxyResult(
            value=value,
            exact=False,  # the gain is an estimate of a nonlinear composition
            measurement_class=2,
            per_layer={"layer": value},
            diagnostics=diagnostics,
        )


class WeightMagnitudeProxy(_BaseVariant):
    """``sum_ij |W_ij|`` — the "weight magnitude" comparator of the predeclared H2 comparator set.

    NOT in the common unit: it is a weight-space statistic used only to rank layers against a proxy's
    ranking. Reported with ``exact=True`` for its own definition (it is a closed-form sum), class 1.
    """

    name = "weight_magnitude"

    def score_layer(self, w, x, rank, spec):
        value = float(torch.sum(w.to(torch.float64).abs()))
        return ProxyResult(
            value=value,
            exact=True,
            measurement_class=1,
            per_layer={"layer": value},
            diagnostics={"weight_l1": value, "n_params": float(w.numel())},
        )


class ActivationMagnitudeProxy(_BaseVariant):
    """``sum_j E[x_j^2]`` — the "activation magnitude" comparator of the predeclared H2 set.

    NOT in the common unit: it ignores the compression entirely and only measures how energetic the
    layer's inputs are. It exists because a proxy that cannot beat it carries no information about
    *damage* (adversarial review section 3). Class 1 for the plug-in mean of a second moment.
    """

    name = "activation_magnitude"

    def score_layer(self, w, x, rank, spec):
        x64 = x.to(torch.float64)
        value = float(torch.sum(torch.mean(x64 * x64, dim=0)))
        return ProxyResult(
            value=value,
            exact=True,
            measurement_class=1,
            per_layer={"layer": value},
            diagnostics={"n_samples": float(x.shape[0]), "mean_activation_energy": value},
        )


class HessianDiagProxy(_BaseVariant):
    """Diagonal activation-second-moment surrogate: ``sum_j E[x_j^2] * sum_k delta_kj^2``.

    This is ``tr(delta diag(Sigma_X) delta^T)``, the diagonal-Hessian (Fisher-diagonal) form of the
    exact quadratic ``tr(delta Sigma_X delta^T)``. It is an approximation — off-diagonal activation
    correlations are dropped — so ``exact=False`` and the diagonal fraction of the activation
    covariance is reported as a diagnostic so the approximation's size is visible.
    """

    name = "hessian_diag"

    def score_layer(self, w, x, rank, spec):
        delta = compression_delta(w, rank, spec).to(torch.float64)
        x64 = x.to(torch.float64)
        second = torch.mean(x64 * x64, dim=0)  # (in,)
        value = float(torch.sum(second * torch.sum(delta * delta, dim=0)))
        # how much of the activation covariance the diagonal keeps
        cov = (
            (x64 - x64.mean(dim=0, keepdim=True)).transpose(0, 1)
            @ (x64 - x64.mean(dim=0, keepdim=True))
            / max(1, x64.shape[0] - 1)
        )
        total = float(torch.sum(cov * cov))
        diag = float(torch.sum(torch.diagonal(cov) ** 2))
        return ProxyResult(
            value=value,
            exact=False,
            measurement_class=2,
            per_layer={"layer": value},
            diagnostics={
                "n_samples": float(x.shape[0]),
                "activation_second_moment": float(second.sum()),
                "diagonal_energy_fraction": (diag / total) if total > 0.0 else 0.0,
            },
        )


class QuantResidualStatsProxy(_BaseVariant):
    """Error contributed by **rounding the factors**, isolated from truncation.

    ``mean_i ||x_i (B A - Q(B) Q(A))^T||^2`` plus residual-distribution diagnostics (mean, max and
    kurtosis of the per-weight rounding residual, and the fraction of total error attributable to
    rounding). Exact for the batch; it answers "how much of the damage is the quantizer's fault",
    which is the quantity H1/H3 care about.
    """

    name = "quant_residual_stats"

    def score_layer(self, w, x, rank, spec):
        delta_q = quantization_only_delta(w, rank, spec)
        delta = compression_delta(w, rank, spec)
        value = _output_error(w, x, delta_q)
        total = _output_error(w, x, delta)
        residual = delta_q.to(torch.float64).flatten()
        mean = float(residual.mean())
        centred = residual - mean
        std = float(centred.std(unbiased=True)) if residual.numel() > 1 else 0.0
        kurtosis = float(torch.mean((centred / std) ** 4)) if std > 0.0 else 0.0
        diagnostics = _sampling_diagnostics(delta_q, x, value)
        diagnostics.update(
            {
                "residual_mean": mean,
                "residual_std": std,
                "residual_kurtosis": kurtosis,
                "residual_abs_max": float(residual.abs().max()) if residual.numel() else 0.0,
                "rounding_fraction_of_total": (value / total) if total > 0.0 else 0.0,
            }
        )
        return ProxyResult(
            value=value,
            exact=True,
            measurement_class=2,
            per_layer={"layer": value},
            diagnostics=diagnostics,
        )


class SpectralSummaryProxy(_BaseVariant):
    """Analytical estimate from the weight spectrum under an isotropic-activation assumption.

    ``(mean_i ||x_i||^2 / in_features) * sum_{k>rank} sigma_k^2``: the discarded singular energy
    scaled by the mean per-feature activation energy. Analytical (class 1) and isotropic
    (``exact=False``); the anisotropy is reported as the ratio of the mean squared activation norm to
    the isotropic prediction so the assumption's error is visible.
    """

    name = "spectral_summary"

    def score_layer(self, w, x, rank, spec):
        from spectraquant.factorization import singular_values

        sv = singular_values(w)
        keep = min(max(rank, 0), sv.numel())
        discarded = float(torch.sum(sv[keep:].to(torch.float64) ** 2))
        x64 = x.to(torch.float64)
        per_sample = torch.sum(x64 * x64, dim=-1)
        mean_energy = float(torch.mean(per_sample))
        in_features = w.shape[1]
        value = mean_energy / in_features * discarded
        exact_error = _output_error(w, x, low_rank_delta(w, rank))
        diagnostics = {
            "discarded_singular_energy": discarded,
            "mean_activation_energy": mean_energy,
            "isotropy_ratio": (value / exact_error) if exact_error > 0.0 else 0.0,
            "kept_rank": float(keep),
        }
        return ProxyResult(
            value=value,
            exact=False,
            measurement_class=1,
            per_layer={"layer": value},
            diagnostics=diagnostics,
        )


class CombinedProxy(_BaseVariant):
    """The candidate: an explicit, configurable combination of the informative components.

    ``value = w_gain * gain_aware + w_hessian * hessian_diag + w_quant * quant_residual`` with
    weights supplied by the caller (defaults are recorded in :data:`DEFAULT_COMBINED_WEIGHTS` and are
    never hidden constants). Each component is computed by its own variant, so an ablation of the
    combination is a change of weights, not a rewrite.
    """

    name = "combined"

    def __init__(self, weights: Mapping[str, float] | None = None) -> None:
        self.name = "combined"
        self.weights = dict(DEFAULT_COMBINED_WEIGHTS if weights is None else weights)
        unknown = set(self.weights) - {"gain_aware", "hessian_diag", "quant_residual"}
        if unknown:
            raise ProxyInputError(f"unknown combined weights: {sorted(unknown)}")
        self._gain = GainAwareComposedProxy()
        self._hessian = HessianDiagProxy()
        self._quant = QuantResidualStatsProxy()

    def _context_layer(self, layer: LayerInputs, rank: int, spec: QuantSpec) -> ProxyResult:
        parts = {
            "gain_aware": self._gain._context_layer(layer, rank, spec),
            "hessian_diag": self._hessian._context_layer(layer, rank, spec),
            "quant_residual": self._quant._context_layer(layer, rank, spec),
        }
        value = sum(self.weights.get(key, 0.0) * part.value for key, part in parts.items())
        diagnostics = {f"{key}_value": part.value for key, part in parts.items()}
        diagnostics.update({f"weight_{key}": w for key, w in self.weights.items()})
        diagnostics["gain_fallback"] = parts["gain_aware"].diagnostics.get("fallback", 0.0)
        return ProxyResult(
            value=float(value),
            exact=False,
            measurement_class=2,
            per_layer={"layer": float(value)},
            diagnostics=diagnostics,
        )

    def score_layer(self, w, x, rank, spec):
        layer = LayerInputs(weight=w, activations=x)
        return self._context_layer(layer, rank, spec)


#: Documented defaults for :class:`CombinedProxy`; equal weights, no tuning has been performed.
DEFAULT_COMBINED_WEIGHTS: dict[str, float] = {
    "gain_aware": 1.0,
    "hessian_diag": 1.0,
    "quant_residual": 1.0,
}

#: Registry used by the measurement script and by integration code.
PROXY_VARIANTS: dict[str, type] = {
    "weight_frobenius": WeightFrobeniusProxy,
    "weight_magnitude": WeightMagnitudeProxy,
    "activation_magnitude": ActivationMagnitudeProxy,
    "per_layer_output_error": PerLayerOutputErrorProxy,
    "in_situ_output_error": InSituOutputErrorProxy,
    "gain_aware_composed": GainAwareComposedProxy,
    "hessian_diag": HessianDiagProxy,
    "quant_residual_stats": QuantResidualStatsProxy,
    "spectral_summary": SpectralSummaryProxy,
    "combined": CombinedProxy,
}


def get_proxy(name: str, **kwargs: object) -> object:
    """Instantiate a variant by name (``combined`` accepts ``weights=``)."""
    if name not in PROXY_VARIANTS:
        raise ProxyInputError(f"unknown proxy {name!r}; known: {sorted(PROXY_VARIANTS)}")
    return PROXY_VARIANTS[name](**kwargs)  # type: ignore[arg-type]


def proxy_names() -> Sequence[str]:
    """Names of every registered variant, in registry order."""
    return tuple(PROXY_VARIANTS)
