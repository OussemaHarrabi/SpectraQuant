"""Component ablations, calibration-size sensitivity, outlier behaviour and seed-count sensitivity
for the declared gain-aware proxy — the Milestone-8 local (``LOCAL-FIXTURE``) half.

This module is the *library* half of the local ablation study; the CLI wrapper is
``scripts/experiments/ablations.py``. It answers four pre-registered engineering questions on the
Tier-0 fixture (`AGENTS.md` section 6), with the model as the resampling unit and the frozen
section 7 statistics of `preregistration.md` (Fisher-z, DerSimonian-Laird random effects, paired
contrasts with a cluster-bootstrap CI):

1. **Proxy component ablation** — the declared candidate ``gain_aware_composed`` against its parts
   removed. The candidate is ``g^2 * mean_i ||x_i delta^T||^2`` (the repo's real implementation: the
   gain term applied to the **naive** pre-normalisation per-layer error). The component factorial
   spans the error basis ``{naive, in-situ-normalisation-mapped}`` x ``{no gain, gain}`` plus the
   untuned ``combined`` variant and the pre-declared ``weight_frobenius`` comparator:

   ===================================  ===========  ======  ============================
   variant                              in-situ      gain    role
   ===================================  ===========  ======  ============================
   ``per_layer_output_error``           no           no      naive baseline
   ``in_situ_output_error``             yes          no      in-situ only (no gain term)
   ``gain_aware_composed``              no           yes     **the declared candidate**
   ``gain_aware_in_situ``               yes          yes     full composition (local variant)
   ``combined``                         no           yes*    untuned combination (extra terms)
   ===================================  ===========  ======  ============================

   ``*`` ``combined`` carries the gain term through its ``gain_aware`` component plus the
   unnormalised Hessian-diagonal and rounding-residual terms. ``gain_aware_in_situ`` is implemented
   **locally here** and is deliberately *not* added to the frozen :data:`PROXY_VARIANTS` registry, so
   the committed M4 validation artifact and its tests are not perturbed.

2. **Calibration-size sensitivity** — the model-level ``rho`` and the plug-in estimator's relative
   standard error as a function of the number of calibration tokens per layer (8 .. 512), showing
   where the ranking stops improving.

3. **Outlier behaviour** — controlled per-channel magnitude spikes at increasing severity
   (1, 2, 4, 8, 16): does the candidate's ranking degrade, how does per-group quantization confine
   the damage relative to per-tensor, and how does the damage / rounding-residual relationship move.

4. **Seed-count sensitivity** — the aggregate candidate-vs-comparator contrast and its CI as the seed
   count grows from 1 to 5.

Substrate: ``LOCAL-FIXTURE`` (CPU workstation). Every number is measurement class 1 or 2
(`AGENTS.md` section 5). The Tier-1/Tier-2 cells of the frozen section 8 matrix remain **NOT RUN**.

No CUDA: pure CPU torch, deterministic given the seeds.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from spectraquant.evaluation.toy import (
    TinyConfig,
    TinyTransformer,
    layer_inputs,
    network_single_layer_damage,
    norm_preimages,
    train_tiny,
)
from spectraquant.proxies.analysis import (
    PREDECLARED_MDE_Z_CONSERVATIVE,
    PREDECLARED_MDE_Z_OPTIMISTIC,
    PREDECLARED_NEFF_CONSERVATIVE,
    PREDECLARED_NEFF_OPTIMISTIC,
    cluster_contrast,
    contrast_verdict,
    fisher_z,
    following_norm_container,
    mde_fisher_z,
    spearman,
    within_model_neff,
)
from spectraquant.proxies.base import LayerInputs, ProxyResult
from spectraquant.proxies.gain import estimate_downstream_gains
from spectraquant.proxies.operators import compression_delta
from spectraquant.proxies.validation import (
    DEFAULT_CELLS,
    DEFAULT_FIXTURES,
    DEFAULT_SEEDS,
    GROUP_SIZE,
    TRAIN_STEPS,
    git_state,
)
from spectraquant.proxies.variants import (
    InSituOutputErrorProxy,
    get_proxy,
)
from spectraquant.quantization import QuantSpec

__all__ = [
    "CALIBRATION_SIZES",
    "CANDIDATE",
    "COMPONENT_LABELS",
    "COMPONENT_VARIANTS",
    "FULL_COMPOSITION",
    "H1_FALSIFIER",
    "H2_FALSIFIER",
    "H4_FALSIFIER",
    "NEW_VARIANT",
    "OUTLIER_SEVERITIES",
    "REFERENCE_COMPARATOR",
    "SCHEMA_ID",
    "SUBSTRATE",
    "apply_outlier_injection",
    "build_document",
    "calibration_ids",
    "calibration_shape",
    "component_analysis",
    "component_signal_verdict",
    "gaussian_relative_se",
    "git_state",
    "hypothesis_verdicts",
    "mean_layer_relative_se",
    "measurement_ids",
    "outlier_injection_plan",
    "plugin_relative_se",
    "run_calibration_size_sweep",
    "run_component_sweep",
    "run_outlier_sweep",
    "seed_count_sensitivity",
]

#: Artifact schema identifier; bumped when the document layout changes.
SCHEMA_ID = "spectraquant.proxy-ablations/1"
#: Execution substrate for every number in this document (`preregistration.md` section 8).
SUBSTRATE = "LOCAL-FIXTURE"

#: The declared candidate (`preregistration.md` section 13, freeze checklist item A-0007).
CANDIDATE = "gain_aware_composed"
#: The full in-situ x gain composition, implemented locally (NOT in the frozen registry).
NEW_VARIANT = "gain_aware_in_situ"
FULL_COMPOSITION = NEW_VARIANT
#: The pre-declared primary H2 comparator (`preregistration.md` section 7.2).
REFERENCE_COMPARATOR = "weight_frobenius"

#: Component-factorial variant order for every table.
COMPONENT_VARIANTS: tuple[str, ...] = (
    "per_layer_output_error",
    "in_situ_output_error",
    "gain_aware_composed",
    "gain_aware_in_situ",
    "combined",
)
#: Human labels stating each variant's role in the factorial.
COMPONENT_LABELS: dict[str, str] = {
    "per_layer_output_error": "naive per-layer output error (no in-situ, no gain)",
    "in_situ_output_error": "in-situ normalisation Jacobian only (no gain term)",
    "gain_aware_composed": "declared candidate: gain-only, no in-situ (the frozen interface)",
    "gain_aware_in_situ": "full composition: in-situ basis x gain (local variant)",
    "combined": "untuned combination (+ Hessian-diagonal and rounding-residual terms)",
}
#: Variants measured per cell: the component factorial plus the pre-declared H2 comparator.
MEASURED_VARIANTS: tuple[str, ...] = (*COMPONENT_VARIANTS, REFERENCE_COMPARATOR)

#: Calibration tokens per layer. 8 -> (batch 1, seq 8); the rest are exact multiples of seq 16.
CALIBRATION_SIZES: tuple[int, ...] = (8, 16, 32, 64, 128, 256, 512)
#: Fixed measurement set (the M4 validation fixture's batch): 2 sequences of 24 tokens = 48 tokens.
MEASUREMENT_BATCH = 2
MEASUREMENT_SEQ = 24
#: Outlier severities: the per-channel spike multiplier. 1.0 is the no-op control.
OUTLIER_SEVERITIES: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0)
#: Fraction of input channels spiked per layer by the outlier injection.
OUTLIER_CHANNEL_FRACTION = 0.1

#: Representative single cell for the calibration-size and outlier studies (H5-relevant b = 4).
STUDY_RANK = 8
STUDY_BITS = 4
#: Deterministic seed for the generated calibration batches (separate stream from training).
CALIBRATION_SEED = 20261009
#: Deterministic seed for the outlier channel selection.
OUTLIER_SEED = 20261010

FLOAT_SIGNIFICANT_DIGITS = 6


def _num(value: float | None) -> float | None:
    """JSON-safe float rounded to :data:`FLOAT_SIGNIFICANT_DIGITS`; non-finite becomes ``None``."""
    if value is None:
        return None
    v = float(value)
    if not math.isfinite(v):
        return None
    return float(f"{v:.{FLOAT_SIGNIFICANT_DIGITS}g}")


# ------------------------------------------------------------------ estimator standard error


def plugin_relative_se(per_sample: Sequence[float] | torch.Tensor) -> float:
    """Relative standard error of the plug-in estimator ``mean_i s_i`` of a per-sample quantity.

    ``s_i`` are the per-sample squared output errors ``||x_i delta^T||^2``. The sample mean has
    standard error ``sqrt(var_unbiased(s) / n)`` (the ``n - 1`` convention, matching
    :func:`spectraquant.proxies.variants._sampling_diagnostics`); the returned value is that divided
    by the mean, i.e. a *relative* SE.

    Returns ``nan`` for fewer than two samples or a non-positive mean. Numerical limitation: the
    ``n - 1`` variance is not the exact Gaussian sampling variance, which is
    ``sqrt(2 tr(G^2) / n) / tr(G)`` with ``G = delta Sigma_X delta^T`` — see
    :func:`gaussian_relative_se`.
    """
    if isinstance(per_sample, torch.Tensor):
        values = per_sample.detach().to(torch.float64).reshape(-1)
        n = int(values.numel())
        if n < 2:
            return float("nan")
        mean = float(values.mean())
        var = float(values.var(unbiased=True))
    else:
        n = len(per_sample)
        if n < 2:
            return float("nan")
        mean = sum(float(v) for v in per_sample) / n
        var = sum((float(v) - mean) ** 2 for v in per_sample) / (n - 1)
    if mean <= 0.0:
        return float("nan")
    return math.sqrt(var / n) / mean


def gaussian_relative_se(trace_g: float, trace_g2: float, n: int) -> float:
    """Exact relative SE of the plug-in estimator for Gaussian ``x``: ``sqrt(2 tr(G^2) / n) / tr(G)``.

    Args:
        trace_g: ``tr(G)`` with ``G = delta Sigma_X delta^T``.
        trace_g2: ``tr(G^2)``.
        n: sample count.

    Limitation: exact only when the activations are Gaussian (so ``sum_i ||x_i delta^T||^2`` has
    chi-square-like variance); it is a reference for the empirical :func:`plugin_relative_se`, not a
    replacement for it.
    """
    if n < 1 or trace_g <= 0.0 or trace_g2 < 0.0:
        return float("nan")
    return math.sqrt(2.0 * trace_g2 / n) / trace_g


def layer_output_error_samples(
    w: torch.Tensor, x: torch.Tensor, rank: int, spec: QuantSpec
) -> torch.Tensor:
    """Per-sample squared output errors ``||x_i delta^T||^2`` (float64) of the naive unit."""
    delta = compression_delta(w, rank, spec)
    proj = x.to(torch.float64) @ delta.to(torch.float64).transpose(0, 1)
    return torch.sum(proj * proj, dim=-1)


def mean_layer_relative_se(
    context: Mapping[str, LayerInputs], names: Sequence[str], rank: int, spec: QuantSpec
) -> float:
    """Mean over layers of the plug-in estimator's relative standard error (naive unit).

    Returns ``nan`` when no layer yields a finite value (e.g. a constant-score layer).
    """
    values = [
        plugin_relative_se(
            layer_output_error_samples(context[n].weight, context[n].activations, rank, spec)
        )
        for n in names
    ]
    finite = [v for v in values if math.isfinite(v)]
    return sum(finite) / len(finite) if finite else float("nan")


# ------------------------------------------------------------------ the local composed variant


def _score_gain_aware_in_situ(layer: LayerInputs, rank: int, spec: QuantSpec) -> ProxyResult:
    """``g^2 * (in-situ mapped error)`` — the full composition (in-situ basis x gain)."""
    in_situ = InSituOutputErrorProxy()._context_layer(layer, rank, spec)  # type: ignore[attr-defined]
    diagnostics = dict(in_situ.diagnostics)
    if layer.downstream_gain is None:
        diagnostics["fallback"] = 1.0
        diagnostics["downstream_gain"] = 0.0
        return ProxyResult(
            value=in_situ.value,
            exact=in_situ.exact,
            measurement_class=in_situ.measurement_class,
            per_layer=in_situ.per_layer,
            diagnostics=diagnostics,
        )
    gain = float(layer.downstream_gain)
    diagnostics["fallback"] = 0.0
    diagnostics["downstream_gain"] = gain
    value = gain * gain * in_situ.value
    return ProxyResult(
        value=value,
        exact=False,
        measurement_class=2,
        per_layer={"layer": value},
        diagnostics=diagnostics,
    )


def _score(variant: str, layer: LayerInputs, rank: int, spec: QuantSpec) -> ProxyResult:
    """Score one layer with one component-ablation variant (registry or the local composition)."""
    if variant == NEW_VARIANT:
        return _score_gain_aware_in_situ(layer, rank, spec)
    proxy = get_proxy(variant)
    return proxy._context_layer(layer, rank, spec)  # type: ignore[attr-defined]


# ------------------------------------------------------------------ calibration / measurement ids


def calibration_shape(size: int) -> tuple[int, int]:
    """``(batch, seq)`` for an exact ``size``-token batch with ``seq <= 16`` (and ``<= size``)."""
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")
    seq = min(16, size)
    if size % seq != 0:
        raise ValueError(f"size {size} is not representable as batch x seq with seq <= 16")
    return size // seq, seq


def calibration_ids(
    vocab: int, batch: int, seq: int, *, seed: int = CALIBRATION_SEED
) -> torch.Tensor:
    """Deterministic calibration token batch drawn from the fixture's synthetic train generator.

    The same construction as :func:`spectraquant.evaluation.toy.train_tiny`'s batch generator
    (``(base + arange(seq) + offset) % vocab``), with a dedicated RNG stream so calibration draws are
    disjoint from the training draws. Returns ``(batch, seq)`` int64.
    """
    gen = torch.Generator().manual_seed(seed)
    offsets = torch.randint(0, vocab, (batch, 1), generator=gen)
    base = torch.randint(0, vocab, (batch, 1), generator=gen)
    return (base + torch.arange(seq)[None] + offsets) % vocab


def measurement_ids(vocab: int) -> torch.Tensor:
    """The fixed measurement batch of the M4 fixture sweep: ``arange(24) % vocab`` repeated twice."""
    return torch.arange(MEASUREMENT_SEQ)[None].repeat(MEASUREMENT_BATCH, 1) % vocab


# ------------------------------------------------------------------ outlier injection


def outlier_injection_plan(
    weight_shapes: Mapping[str, int],
    *,
    fraction: float = OUTLIER_CHANNEL_FRACTION,
    seed: int = OUTLIER_SEED,
) -> dict[str, list[int]]:
    """Deterministic per-layer selection of input channels to spike.

    Procedure (exact, reproducible): for each layer, ``k = max(1, round(fraction * in_features))``
    input-channel indices are drawn **without replacement** from ``range(in_features)`` by a
    ``random.Random(seed)`` advanced layer by layer in the sorted layer order, and returned sorted.
    The same ``(weight_shapes, fraction, seed)`` always yields the same plan.

    Args:
        weight_shapes: ``layer name -> in_features`` for the model's compressible linear weights.
        fraction: fraction of input channels spiked per layer.
        seed: RNG seed for the channel draw.

    Returns:
        ``layer name -> sorted list of spiked input-channel indices``.
    """
    rng = random.Random(seed)
    plan: dict[str, list[int]] = {}
    for name in sorted(weight_shapes):
        in_features = int(weight_shapes[name])
        if in_features < 1:
            raise ValueError(f"layer {name!r} has {in_features} input features")
        k = max(1, round(fraction * in_features))
        k = min(k, in_features)
        plan[name] = sorted(rng.sample(range(in_features), k))
    return plan


def apply_outlier_injection(
    model: TinyTransformer, plan: Mapping[str, Sequence[int]], severity: float
) -> None:
    """Multiply the planned input channels of every weight by ``severity`` **in place**.

    This mutates the model's parameters; the caller must pass a private copy when the original must
    be preserved. ``severity = 1.0`` is an exact no-op (a float multiply by 1.0 is exact).
    """
    layers = model.linear_layers()
    with torch.no_grad():
        for name, channels in plan.items():
            weight = layers[name].weight
            index = torch.tensor(list(channels), dtype=torch.long)
            weight[:, index] = weight[:, index] * float(severity)


# ------------------------------------------------------------------ shared per-model measurement


def _layer_inputs_for_proxies(
    model: TinyTransformer,
    names: Sequence[str],
    rank: int,
    spec: QuantSpec,
    *,
    source: Mapping[str, Any],
    activations: Mapping[str, torch.Tensor],
    preimages: Mapping[str, torch.Tensor],
    gains: Any,
) -> dict[str, LayerInputs]:
    """Build the :class:`LayerInputs` context (residual errors + gains measured on ``source``)."""
    residual_errors = source.get("residual_errors", {})
    return {
        name: LayerInputs(
            weight=model.linear_layers()[name].weight.detach(),
            activations=activations[name],
            following_norm=following_norm_container(model, name, preimages),
            residual_error=residual_errors.get(name),
            downstream_gain=gains.per_layer[name],
        )
        for name in names
    }


def _measure_cell_extended(
    model: TinyTransformer,
    ids: torch.Tensor,
    cfg: TinyConfig,
    names: Sequence[str],
    *,
    rank: int,
    bits: int,
    group_size: int,
    n_boot_neff: int,
    neff_seed: int,
) -> dict[str, Any]:
    """Measure one ``(rank, bits)`` cell for every component variant, including the local one.

    Mirrors :func:`spectraquant.proxies.validation.measure_cell` for the shared quantities (damage,
    activations, gains) so the registered variants' ``rho`` values are identical to the committed M4
    validation artifact, and additionally scores :data:`NEW_VARIANT`. All class 2.
    """
    spec = QuantSpec(bits=bits, granularity="per_group", group_size=group_size, symmetric=True)
    damage = network_single_layer_damage(model, ids, names, rank, spec)
    activations = layer_inputs(model, ids, names)
    preimages = norm_preimages(model, ids)
    gains = estimate_downstream_gains(
        lambda nm, pert: model(ids, inject=(nm, pert))[1].reshape(-1, cfg.d_model),
        damage.layer_errors,
    )
    immediate: dict[str, Any] = {
        "residual_errors": damage.residual_errors,
        "per_layer": damage.per_layer,
    }
    context = _layer_inputs_for_proxies(
        model,
        names,
        rank,
        spec,
        source=immediate,
        activations=activations,
        preimages=preimages,
        gains=gains,
    )
    damages = [damage.per_layer[n] for n in names]
    variants: dict[str, Any] = {}
    for variant in MEASURED_VARIANTS:
        values = [float(_score(variant, context[n], rank, spec).value) for n in names]
        neff = within_model_neff(values, damages, n_boot=n_boot_neff, seed=neff_seed)
        variants[variant] = {
            "rho": _num(spearman(values, damages)),
            "n_eff": _num(neff.n_eff),
            "clipped": neff.clipped,
        }
    return {
        "rank": rank,
        "bits": bits,
        "group_size": group_size,
        "damage_sum": _num(sum(damage.per_layer.values())),
        "estimator_rel_se_mean": _num(mean_layer_relative_se(context, names, rank, spec)),
        "variants": variants,
    }


def run_component_sweep(
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    cells: Sequence[tuple[int, int]] = DEFAULT_CELLS,
    fixtures: Sequence[tuple[str, bool]] = DEFAULT_FIXTURES,
    train_steps: int = TRAIN_STEPS,
    group_size: int = GROUP_SIZE,
    n_boot_neff: int = 200,
    boot_seed: int = 20261009,
) -> list[dict[str, Any]]:
    """Train each fixture once per seed and measure every cell for every component variant."""
    runs: list[dict[str, Any]] = []
    for fixture, use_norm in fixtures:
        for seed in seeds:
            cfg = TinyConfig(use_norm=use_norm, seed=seed)
            model, training = train_tiny(cfg, steps=train_steps)
            names = tuple(model.linear_layers())
            ids = measurement_ids(cfg.vocab)
            records: list[dict[str, Any]] = []
            for index, (rank, bits) in enumerate(cells):
                records.append(
                    _measure_cell_extended(
                        model,
                        ids,
                        cfg,
                        names,
                        rank=rank,
                        bits=bits,
                        group_size=group_size,
                        n_boot_neff=n_boot_neff,
                        neff_seed=boot_seed + seed * 1_000 + index * 10,
                    )
                )
            runs.append(
                {
                    "fixture": fixture,
                    "use_norm": use_norm,
                    "seed": seed,
                    "modules": list(names),
                    "training": training,
                    "cells": records,
                }
            )
    return runs


# ------------------------------------------------------------------ contrast reduction


def _usable(record: Mapping[str, Any]) -> tuple[float, float] | None:
    """``(rho, n_eff)`` of a per-model variant record, or ``None`` when unusable."""
    rho, n_eff = record.get("rho"), record.get("n_eff")
    if rho is None or n_eff is None or float(n_eff) <= 3.0:
        return None
    return float(rho), float(n_eff)


def _paired_units(
    runs: Sequence[Mapping[str, Any]],
    candidate: str,
    comparator: str,
    *,
    fixture: str | None = None,
    rank: int | None = None,
    bits: int | None = None,
) -> dict[str, list[tuple[float, float]]]:
    """``cluster id -> [(d, var_d)]`` for the paired candidate-vs-comparator Fisher-z difference.

    One cluster is one trained model (``fixture:seed``); its units are the selected cells. ``d`` is
    the within-model Fisher-z difference, ``var_d`` the conservative sum of the two ``1/(n_eff-3)``
    variances (the same paired form as the frozen section 7.2 statistic).
    """
    out: dict[str, list[tuple[float, float]]] = {}
    for run in runs:
        if fixture is not None and run["fixture"] != fixture:
            continue
        units: list[tuple[float, float]] = []
        for cell in run["cells"]:
            if rank is not None and (cell["rank"] != rank or cell["bits"] != bits):
                continue
            cand = _usable(cell["variants"][candidate])
            comp = _usable(cell["variants"][comparator])
            if cand is None or comp is None:
                continue
            var_d = 1.0 / (cand[1] - 3.0) + 1.0 / (comp[1] - 3.0)
            units.append((fisher_z(cand[0]) - fisher_z(comp[0]), var_d))
        if units:
            out[f"{run['fixture']}:{run['seed']}"] = units
    return out


def _variant_units(
    runs: Sequence[Mapping[str, Any]], variant: str, *, fixture: str | None = None
) -> dict[str, list[tuple[float, float]]]:
    """``cluster id -> [(z, var)]`` for one variant over every cell of a fixture group."""
    out: dict[str, list[tuple[float, float]]] = {}
    for run in runs:
        if fixture is not None and run["fixture"] != fixture:
            continue
        units: list[tuple[float, float]] = []
        for cell in run["cells"]:
            usable = _usable(cell["variants"][variant])
            if usable is None:
                continue
            rho, n_eff = usable
            units.append((fisher_z(rho), 1.0 / (n_eff - 3.0)))
        if units:
            out[f"{run['fixture']}:{run['seed']}"] = units
    return out


def _contrast_payload(
    units: dict[str, list[tuple[float, float]]],
    *,
    label: str,
    n_boot_ci: int,
    boot_seed: int,
    n_eff_ref: float | None,
) -> dict[str, Any]:
    """Contrast payload: point estimate, analytic random-effects CI, cluster-bootstrap CI, verdict."""
    result = cluster_contrast(units, label=label, n_boot=n_boot_ci, seed=boot_seed)
    verdict = contrast_verdict(result).to_dict()
    n_eff_achieved = mde_fisher_z(
        float(n_eff_ref) if n_eff_ref is not None else float("nan"), result.n_clusters
    )
    return {
        "label": label,
        "delta_z": _num(result.delta_z),
        "ci": [_num(result.ci_low), _num(result.ci_high)],
        "boot_ci": [_num(result.boot_ci_low), _num(result.boot_ci_high)],
        "p_value": _num(result.p_value),
        "tau2": _num(result.tau2),
        "k": result.k,
        "n_clusters": result.n_clusters,
        "mde_achieved_z": _num(n_eff_achieved),
        "verdict": verdict,
    }


def _variant_table(runs: Sequence[Mapping[str, Any]], fixture: str | None) -> dict[str, Any]:
    """Mean ``rho`` and ``n_eff`` per variant over a fixture group (or the pooled group)."""
    out: dict[str, Any] = {}
    for variant in COMPONENT_VARIANTS:
        rhos: list[float] = []
        n_effs: list[float] = []
        for run in runs:
            if fixture is not None and run["fixture"] != fixture:
                continue
            for cell in run["cells"]:
                usable = _usable(cell["variants"][variant])
                if usable is None:
                    continue
                rhos.append(usable[0])
                n_effs.append(usable[1])
        out[variant] = {
            "rho_mean": _num(sum(rhos) / len(rhos)) if rhos else None,
            "n_eff_mean": _num(sum(n_effs) / len(n_effs)) if n_effs else None,
            "k": len(rhos),
        }
    return out


def component_analysis(
    runs: Sequence[Mapping[str, Any]], *, n_boot_ci: int, boot_seed: int
) -> dict[str, Any]:
    """Component-ablation aggregate: per-variant means and candidate-vs-part contrasts per group."""
    groups: list[str] = [name for name, _ in DEFAULT_FIXTURES] + ["all_fixtures"]
    analysis: dict[str, Any] = {}
    for group in groups:
        fixture = None if group == "all_fixtures" else group
        table = _variant_table(runs, fixture)
        n_eff_ref = table[CANDIDATE]["n_eff_mean"]
        contrasts: dict[str, Any] = {}
        for comparator in (*COMPONENT_VARIANTS, REFERENCE_COMPARATOR):
            if comparator == CANDIDATE:
                continue
            units = _paired_units(runs, CANDIDATE, comparator, fixture=fixture)
            if not units:
                continue
            contrasts[comparator] = _contrast_payload(
                units,
                label=f"{CANDIDATE} - {comparator}",
                n_boot_ci=n_boot_ci,
                boot_seed=boot_seed,
                n_eff_ref=n_eff_ref,
            )
        analysis[group] = {"variants": table, "contrasts": contrasts}
    return analysis


def component_signal_verdict(analysis: Mapping[str, Any]) -> dict[str, Any]:
    """State which component carries the signal, from the pooled (``all_fixtures``) contrasts.

    The gain term is isolated by the ``candidate - per_layer_output_error`` contrast (the declared
    candidate *is* gain-only). The in-situ term is isolated by the reversed contrast
    ``gain_aware_in_situ - gain_aware_composed`` (gain held fixed, error basis swapped to in-situ):
    that term *adds* signal only when its random-effects CI sits entirely above 0. A positive
    ``candidate - gain_aware_in_situ`` therefore means the in-situ basis does **not** help.
    """
    pooled = analysis["all_fixtures"]["contrasts"]
    table = analysis["all_fixtures"]["variants"]
    gain = pooled.get("per_layer_output_error", {})
    reversed_full = pooled.get(NEW_VARIANT, {})
    naive = pooled.get("in_situ_output_error", {})
    extra = pooled.get("combined", {})

    in_situ_ci = [-reversed_full["ci"][1], -reversed_full["ci"][0]] if reversed_full else None
    in_situ_delta = -reversed_full["delta_z"] if reversed_full else None
    in_situ_adds = bool(in_situ_ci) and in_situ_ci[0] > 0.0

    return {
        "gain_term": {
            "contrast": "gain_aware_composed - per_layer_output_error",
            "delta_z": gain.get("delta_z"),
            "ci": gain.get("ci"),
            "carries_signal": bool(gain) and gain["ci"][0] > 0.0,
        },
        "in_situ_term": {
            "contrast": "gain_aware_in_situ - gain_aware_composed (in-situ basis x gain, minus the "
            "gain-only candidate)",
            "delta_z": in_situ_delta,
            "ci": in_situ_ci,
            "carries_signal": in_situ_adds,
        },
        "candidate_minus_full": {
            "contrast": "gain_aware_composed - gain_aware_in_situ",
            "delta_z": reversed_full.get("delta_z"),
            "ci": reversed_full.get("ci"),
            "verdict": reversed_full.get("verdict", {}).get("verdict"),
        },
        "extra_terms": {
            "contrast": "gain_aware_composed - combined",
            "delta_z": extra.get("delta_z"),
            "ci": extra.get("ci"),
        },
        "naive_basis_vs_in_situ_only": {
            "contrast": "gain_aware_composed - in_situ_output_error",
            "delta_z": naive.get("delta_z"),
            "ci": naive.get("ci"),
        },
        "component_rho_mean": {
            variant: table[variant]["rho_mean"] for variant in COMPONENT_VARIANTS
        },
    }


# ------------------------------------------------------------------ calibration-size sensitivity


def run_calibration_size_sweep(
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    fixtures: Sequence[tuple[str, bool]] = DEFAULT_FIXTURES,
    sizes: Sequence[int] = CALIBRATION_SIZES,
    rank: int = STUDY_RANK,
    bits: int = STUDY_BITS,
    group_size: int = GROUP_SIZE,
    train_steps: int = TRAIN_STEPS,
    n_boot_neff: int = 100,
    boot_seed: int = 20261009,
) -> dict[str, Any]:
    """``rho`` and the plug-in relative SE versus the calibration token count per layer.

    The **measurement** target (single-layer damage used for the ranking) is always measured on the
    fixed :func:`measurement_ids` batch, so the target does not move with the calibration size; only
    the proxy's calibration set (activations, residual errors, gains) grows. ``rho`` is one
    model-level Spearman over the fixture's modules.
    """
    variants = ("gain_aware_composed", "per_layer_output_error", "in_situ_output_error")
    rows: list[dict[str, Any]] = []
    for fixture, use_norm in fixtures:
        for seed in seeds:
            cfg = TinyConfig(use_norm=use_norm, seed=seed)
            model, _training = train_tiny(cfg, steps=train_steps)
            names = tuple(model.linear_layers())
            spec = QuantSpec(
                bits=bits, granularity="per_group", group_size=group_size, symmetric=True
            )
            target = network_single_layer_damage(
                model, measurement_ids(cfg.vocab), names, rank, spec
            )
            damages = [target.per_layer[n] for n in names]
            for size in sizes:
                batch, seq = calibration_shape(size)
                ids = calibration_ids(cfg.vocab, batch, seq)
                cal = network_single_layer_damage(model, ids, names, rank, spec)
                activations = layer_inputs(model, ids, names)
                preimages = norm_preimages(model, ids)
                gains = estimate_downstream_gains(
                    lambda nm, pert, _model=model, _ids=ids, _d=cfg.d_model: _model(
                        _ids, inject=(nm, pert)
                    )[1].reshape(-1, _d),
                    cal.layer_errors,
                )
                context = _layer_inputs_for_proxies(
                    model,
                    names,
                    rank,
                    spec,
                    source={"residual_errors": cal.residual_errors, "per_layer": cal.per_layer},
                    activations=activations,
                    preimages=preimages,
                    gains=gains,
                )
                rhos: dict[str, float | None] = {}
                for variant in variants:
                    values = [float(_score(variant, context[n], rank, spec).value) for n in names]
                    rhos[variant] = _num(spearman(values, damages))
                rel_se = _num(mean_layer_relative_se(context, names, rank, spec))
                rows.append(
                    {
                        "fixture": fixture,
                        "seed": seed,
                        "size": size,
                        "batch": batch,
                        "seq": seq,
                        "tokens": batch * seq,
                        "rho": rhos,
                        "estimator_rel_se": rel_se,
                    }
                )
    return {"sizes": list(sizes), "rows": rows, "mean": _mean_by_size(rows, variants)}


def _mean_by_size(rows: Sequence[Mapping[str, Any]], variants: Sequence[str]) -> dict[str, Any]:
    """Mean over models of each size's ``rho`` and relative SE."""
    out: dict[str, Any] = {}
    for size in sorted({int(r["size"]) for r in rows}):
        selected = [r for r in rows if int(r["size"]) == size]
        entry: dict[str, Any] = {"n_models": len(selected)}
        for variant in variants:
            values = [r["rho"][variant] for r in selected if r["rho"][variant] is not None]
            entry[f"rho_{variant}"] = _num(sum(values) / len(values)) if values else None
        ses = [r["estimator_rel_se"] for r in selected if r["estimator_rel_se"] is not None]
        entry["estimator_rel_se_mean"] = _num(sum(ses) / len(ses)) if ses else None
        out[str(size)] = entry
    return out


# ------------------------------------------------------------------ outlier behaviour


def run_outlier_sweep(
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    fixtures: Sequence[tuple[str, bool]] = DEFAULT_FIXTURES,
    severities: Sequence[float] = OUTLIER_SEVERITIES,
    rank: int = STUDY_RANK,
    bits: int = STUDY_BITS,
    group_size: int = GROUP_SIZE,
    train_steps: int = TRAIN_STEPS,
    fraction: float = OUTLIER_CHANNEL_FRACTION,
    injection_seed: int = OUTLIER_SEED,
) -> dict[str, Any]:
    """Inject per-channel magnitude spikes and measure ranking, quantization and residuals.

    Procedure (exact): train the fixture normally; build the injection plan with
    :func:`outlier_injection_plan` (deterministic channel selection); for each severity deep-copy the
    trained model and multiply the planned input channels of every compressible weight by the
    severity (:func:`apply_outlier_injection`); measure, on the fixed :func:`measurement_ids` batch,
    the single-layer damage and, per variant, the model-level ranking ``rho``. Rounding residuals are
    measured per-group (the study spec) and per-tensor (the control) with
    :class:`spectraquant.proxies.variants.QuantResidualStatsProxy`. The injected channel list is
    recorded once, per layer.
    """
    import copy

    from spectraquant.proxies.variants import QuantResidualStatsProxy

    rows: list[dict[str, Any]] = []
    plan: dict[str, list[int]] | None = None
    for fixture, use_norm in fixtures:
        for seed in seeds:
            cfg = TinyConfig(use_norm=use_norm, seed=seed)
            model, _training = train_tiny(cfg, steps=train_steps)
            names = tuple(model.linear_layers())
            if plan is None:
                plan = outlier_injection_plan(
                    {name: model.linear_layers()[name].weight.shape[1] for name in names},
                    fraction=fraction,
                    seed=injection_seed,
                )
            per_group = QuantSpec(
                bits=bits, granularity="per_group", group_size=group_size, symmetric=True
            )
            per_tensor = QuantSpec(
                bits=bits, granularity="per_tensor", group_size=None, symmetric=True
            )
            for severity in severities:
                injected = copy.deepcopy(model)
                apply_outlier_injection(injected, plan, float(severity))
                ids = measurement_ids(cfg.vocab)
                damage = network_single_layer_damage(injected, ids, names, rank, per_group)
                activations = layer_inputs(injected, ids, names)
                preimages = norm_preimages(injected, ids)
                gains = estimate_downstream_gains(
                    lambda nm, pert, _model=injected, _ids=ids, _d=cfg.d_model: _model(
                        _ids, inject=(nm, pert)
                    )[1].reshape(-1, _d),
                    damage.layer_errors,
                )
                context = _layer_inputs_for_proxies(
                    injected,
                    names,
                    rank,
                    per_group,
                    source={
                        "residual_errors": damage.residual_errors,
                        "per_layer": damage.per_layer,
                    },
                    activations=activations,
                    preimages=preimages,
                    gains=gains,
                )
                damages = [damage.per_layer[n] for n in names]
                rho: dict[str, float | None] = {}
                for variant in ("gain_aware_composed", "per_layer_output_error"):
                    values = [
                        float(_score(variant, context[n], rank, per_group).value) for n in names
                    ]
                    rho[variant] = _num(spearman(values, damages))
                group_residuals = [
                    float(
                        QuantResidualStatsProxy()
                        .score_layer(context[n].weight, context[n].activations, rank, per_group)
                        .value
                    )
                    for n in names
                ]
                tensor_residuals = [
                    float(
                        QuantResidualStatsProxy()
                        .score_layer(context[n].weight, context[n].activations, rank, per_tensor)
                        .value
                    )
                    for n in names
                ]
                residual_damage_rho = spearman(group_residuals, damages)
                rows.append(
                    {
                        "fixture": fixture,
                        "seed": seed,
                        "severity": float(severity),
                        "damage_sum": _num(sum(damages)),
                        "rounding_residual_per_group": _num(sum(group_residuals)),
                        "rounding_residual_per_tensor": _num(sum(tensor_residuals)),
                        "per_tensor_over_per_group": _num(
                            sum(tensor_residuals) / sum(group_residuals)
                            if sum(group_residuals) > 0.0
                            else None
                        ),
                        "damage_over_rounding": _num(
                            sum(damages) / sum(group_residuals)
                            if sum(group_residuals) > 0.0
                            else None
                        ),
                        "rho_residual_damage": _num(residual_damage_rho),
                        "rho": rho,
                    }
                )
    return {
        "procedure": (
            "train the fixture normally; select k = max(1, round(0.1 * in_features)) input channels "
            "per layer without replacement by random.Random(20261010) over the sorted layer order; "
            "deep-copy the model and multiply those channels of every compressible weight by the "
            "severity; measure on the fixed 2x24-token batch. severity 1.0 is the no-op control."
        ),
        "severities": [float(s) for s in severities],
        "channel_fraction": fraction,
        "injection_seed": injection_seed,
        "plan": plan or {},
        "rows": rows,
        "mean": _mean_by_severity(rows),
    }


def _mean_by_severity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Mean over models of every outlier metric at each severity."""
    out: dict[str, Any] = {}
    for severity in sorted({round(float(r["severity"]), 6) for r in rows}):
        selected = [r for r in rows if round(float(r["severity"]), 6) == severity]
        entry: dict[str, Any] = {"n_models": len(selected)}
        for key in (
            "damage_sum",
            "rounding_residual_per_group",
            "rounding_residual_per_tensor",
            "per_tensor_over_per_group",
            "damage_over_rounding",
            "rho_residual_damage",
        ):
            values = [r[key] for r in selected if r[key] is not None]
            entry[key] = _num(sum(values) / len(values)) if values else None
        for variant in ("gain_aware_composed", "per_layer_output_error"):
            values = [r["rho"][variant] for r in selected if r["rho"][variant] is not None]
            entry[f"rho_{variant}"] = _num(sum(values) / len(values)) if values else None
        out[str(severity)] = entry
    return out


# ------------------------------------------------------------------ seed-count sensitivity


def seed_count_sensitivity(
    runs: Sequence[Mapping[str, Any]],
    *,
    all_seeds: Sequence[int] = DEFAULT_SEEDS,
    comparator: str = REFERENCE_COMPARATOR,
    n_boot_ci: int = 2000,
    boot_seed: int = 20261009,
) -> list[dict[str, Any]]:
    """The candidate-vs-comparator pooled contrast as the seed count grows from 1 to ``len``.

    For each ``S`` the runs of seeds ``all_seeds[:S]`` are pooled (both fixtures, every cell) and the
    paired Fisher-z contrast is reported with its analytic and bootstrap CI, the achieved MDE and the
    frozen verdict. This is a *sensitivity* view of the pre-registered no-optional-stopping rule
    (`preregistration.md` section 7.6): it shows what a smaller seed count would have reported.
    """
    rows: list[dict[str, Any]] = []
    for size in range(1, len(all_seeds) + 1):
        seeds = set(all_seeds[:size])
        subset = [r for r in runs if r["seed"] in seeds]
        units = _paired_units(subset, CANDIDATE, comparator)
        if not units:
            rows.append({"seeds": size, "seed_list": sorted(seeds), "n_clusters": 0})
            continue
        table = _variant_table(subset, None)
        contrast = _contrast_payload(
            units,
            label=f"{CANDIDATE} - {comparator}",
            n_boot_ci=n_boot_ci,
            boot_seed=boot_seed,
            n_eff_ref=table[CANDIDATE]["n_eff_mean"],
        )
        contrast["seeds"] = size
        contrast["seed_list"] = sorted(seeds)
        rows.append(contrast)
    return rows


# ------------------------------------------------------------------ falsifiers


H1_FALSIFIER = (
    "No regime found where truncation reduces reconstruction error while simultaneously increasing "
    "rounding sensitivity at equal stored bytes (95% CI of the sensitivity difference includes 0, or "
    "the two move together), on >=5 Tier-1 seeds (cloud)."
)
H2_FALSIFIER = (
    "The proxy's **model-level** correlation with measured degradation is **not** higher than "
    "weight-space Frobenius error — the Fisher-z random-effects CI on the paired difference includes "
    "0 and is narrower than the predeclared section 7.5 MDE (a CI wider than the MDE is reported "
    "**inconclusive**, not refuting)."
)
H4_FALSIFIER = (
    "Proxy-allocated $(r_\\ell,b_\\ell)$ does **not** dominate uniform rank/bit at any predeclared "
    "budget beyond CIs, or is matched by the HAWQ-V2-style bit-only allocator at equal bytes."
)


def hypothesis_verdicts(analysis: Mapping[str, Any]) -> dict[str, Any]:
    """Per-hypothesis fixture-scale verdicts with the frozen section 11 falsifier wording.

    * **H2** is the hypothesis this ablation slice tests: the candidate-vs-``weight_frobenius``
      contrast in the pooled group, reduced by the frozen section 11.1 rule.
    * **H1** and **H4** are **not** tested by this slice: H1 needs the factorization-error vs
      rounding-sensitivity regime comparison and H4 needs the Pareto hypervolume comparison, both
      owned elsewhere. Their Tier-1/Tier-2 cells are ``NOT RUN``, so their verdict is
      ``inconclusive`` at fixture scale — stated as *not tested by this slice*, never as a refutation.
    """
    pooled = analysis["all_fixtures"]["contrasts"]
    h2 = pooled.get(REFERENCE_COMPARATOR, {})
    h2_verdict = h2.get("verdict", {})
    return {
        "H1": {
            "falsifier": H1_FALSIFIER,
            "tested_by_this_slice": False,
            "verdict": "inconclusive",
            "reason": (
                "H1 requires the controlled factorization-error vs rounding-sensitivity regime "
                "comparison; this slice measures proxy components, calibration size, outlier "
                "robustness and seed sensitivity. No H1 contrast is computed here."
            ),
        },
        "H2": {
            "falsifier": H2_FALSIFIER,
            "tested_by_this_slice": True,
            "contrast": f"{CANDIDATE} - {REFERENCE_COMPARATOR}",
            "delta_z": h2.get("delta_z"),
            "ci": h2.get("ci"),
            "boot_ci": h2.get("boot_ci"),
            "p_value": h2.get("p_value"),
            "verdict": _map_verdict(h2_verdict.get("verdict")),
            "intrinsic_verdict": h2_verdict.get("verdict"),
            "decisive": h2_verdict.get("decisive"),
            "rationale": h2_verdict.get("rationale"),
        },
        "H4": {
            "falsifier": H4_FALSIFIER,
            "tested_by_this_slice": False,
            "verdict": "inconclusive",
            "reason": (
                "H4 requires the Pareto-hypervolume frontier comparison against uniform and the "
                "HAWQ-V2-style allocator; this slice supplies only the proxy-component evidence. No "
                "H4 dominance contrast is computed here."
            ),
        },
    }


def _map_verdict(verdict: object) -> str:
    """Map the frozen ``refuted`` vocabulary to the requested ``contradicted`` wording."""
    text = str(verdict)
    if text == "refuted":
        return "contradicted"
    return text


# ------------------------------------------------------------------ document assembly


def build_document(
    *,
    component_runs: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any],
    outliers: Mapping[str, Any],
    seed_count: Sequence[Mapping[str, Any]],
    seeds: Sequence[int] = DEFAULT_SEEDS,
    cells: Sequence[tuple[int, int]] = DEFAULT_CELLS,
    fixtures: Sequence[tuple[str, bool]] = DEFAULT_FIXTURES,
    group_size: int = GROUP_SIZE,
    train_steps: int = TRAIN_STEPS,
    n_boot_neff: int = 200,
    n_boot_ci: int = 2000,
    boot_seed: int = 20261009,
    wall_time_s: float = 0.0,
    git_commit: str = "unknown",
    git_dirty: bool = False,
) -> dict[str, Any]:
    """Assemble the machine-readable ablation document from the measured sections."""
    analysis = component_analysis(component_runs, n_boot_ci=n_boot_ci, boot_seed=boot_seed)
    return {
        "schema": SCHEMA_ID,
        "generated_by": "scripts/experiments/ablations.py",
        "substrate": SUBSTRATE,
        "measurement_class": 2,
        "statistical_unit": "model",
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "float_precision_significant_digits": FLOAT_SIGNIFICANT_DIGITS,
        "runtime": {
            "wall_time_s": _num(wall_time_s),
            "n_models": len(component_runs),
            "n_component_cells": sum(len(r["cells"]) for r in component_runs),
        },
        "plan": {
            "seeds": list(seeds),
            "cells": [list(c) for c in cells],
            "fixtures": [name for name, _ in fixtures],
            "modules": list(component_runs[0]["modules"]) if component_runs else [],
            "group_size": group_size,
            "train_steps": train_steps,
            "candidate": CANDIDATE,
            "component_variants": list(COMPONENT_VARIANTS),
            "component_labels": dict(COMPONENT_LABELS),
            "local_variant": NEW_VARIANT,
            "reference_comparator": REFERENCE_COMPARATOR,
            "calibration_sizes": list(calibration["sizes"]),
            "outlier_severities": list(outliers["severities"]),
            "outlier_channel_fraction": outliers["channel_fraction"],
            "outlier_injection_seed": outliers["injection_seed"],
            "study_cell": [STUDY_RANK, STUDY_BITS],
            "n_boot_neff": n_boot_neff,
            "n_boot_ci": n_boot_ci,
            "boot_seed": boot_seed,
        },
        "predeclared_mde": {
            "n_eff_optimistic": PREDECLARED_NEFF_OPTIMISTIC,
            "n_eff_conservative": PREDECLARED_NEFF_CONSERVATIVE,
            "seeds": list(seeds),
            "mde_z_optimistic": PREDECLARED_MDE_Z_OPTIMISTIC,
            "mde_z_conservative": PREDECLARED_MDE_Z_CONSERVATIVE,
            "source": "preregistration.md section 7.5 (frozen 2026-10-09)",
        },
        "component_ablation": {
            "analysis": analysis,
            "signal_verdict": component_signal_verdict(analysis),
        },
        "calibration_size": dict(calibration),
        "outliers": dict(outliers),
        "seed_count": list(seed_count),
        "hypotheses": hypothesis_verdicts(analysis),
        "cloud_cells": {
            "tier1_h1_factorization_vs_rounding": {
                "confirmatory_cell": "Factorization error vs rounding sensitivity, tiny LM, >=5 seeds",
                "hypothesis": "H1",
                "tier": 1,
                "substrate": "CLOUD-COLAB",
                "milestone": "M4",
                "status": "NOT RUN",
                "reason": "Tier-1 training runs only on the cloud notebook substrate "
                "(AGENTS.md section 2.3/2b); no validated run_manifest.json exists yet.",
            },
            "tier1_h2_proxy_ranking": {
                "confirmatory_cell": "Proxy vs weight-space Frobenius error rank correlation, "
                "activation covariance + quantization residual in the proxy, tiny LM, >=5 seeds",
                "hypothesis": "H2",
                "tier": 1,
                "substrate": "CLOUD-COLAB",
                "milestone": "M4",
                "status": "NOT RUN",
                "reason": "Same as above: the H2 confirmatory cell is CLOUD-COLAB and has not run.",
            },
            "tier1_h4_mixed_allocation": {
                "confirmatory_cell": "Layer-wise mixed rank + mixed precision vs uniform and vs "
                "HAWQ-V2-style, >=5 seeds",
                "hypothesis": "H4",
                "tier": 1,
                "substrate": "CLOUD-COLAB",
                "milestone": "M4",
                "status": "NOT RUN",
                "reason": "Same as above: the H4 confirmatory cell is CLOUD-COLAB and has not run.",
            },
            "tier2_h1_h5_tinyllama": {
                "confirmatory_cell": "TinyLlama-1.1B-class + WikiText-2, all arms, >=3 seeds",
                "hypothesis": "H1-H5",
                "tier": 2,
                "substrate": "CLOUD-GPU",
                "milestone": "M7",
                "status": "NOT RUN",
                "reason": "Tier-2 runs on the cloud GPU substrate; not started, no validated "
                "manifest.",
            },
        },
    }
