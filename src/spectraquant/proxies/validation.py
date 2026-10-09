"""Multi-seed, multi-configuration proxy-ranking sweep — the local-fixture half of the M4 gate.

This module is the *library* half of the Milestone-4 proxy-validation gate; the CLI wrapper is
``scripts/experiments/proxy_validation_sweep.py``. It implements the pre-registered statistical plan
(`preregistration.md` section 7.0/7.2, frozen 2026-10-09) on the Tier-0 fixture:

* every reported cell is measured on **>= 5 seeds** (the pre-registered floor for cheap cells);
* the cells span the pre-declared ``(rank, bits)`` grid on both the LayerNorm and the no-LayerNorm
  fixture;
* the **model** (one trained fixture) is the resampling unit: one Spearman ``rho`` per trained model
  per cell, transformed to Fisher-z, with ``n_eff`` from a within-model module bootstrap;
* models are combined with the DerSimonian-Laird random-effects model in
  :mod:`spectraquant.proxies.analysis`, and the candidate-vs-comparator contrast is reported with an
  analytic random-effects CI **and** a cluster-bootstrap CI (a cluster is one trained model, carrying
  all of its cells);
* the achieved MDE is reported next to the pre-declared one.

Substrate: ``LOCAL-FIXTURE`` (CPU workstation). Every number here is measurement class 1 or 2
(`AGENTS.md` section 5) and is a **fixture** result: the Tier-1 cells of the frozen section 8 matrix
that the H2/H4 verdicts ultimately rest on are ``CLOUD-COLAB`` and remain **NOT RUN**.

No CUDA: pure CPU torch, deterministic given the seeds.
"""

from __future__ import annotations

import math
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from spectraquant.evaluation.toy import (
    TinyConfig,
    TinyTransformer,
    layer_inputs,
    layer_names,
    network_joint_damage,
    network_single_layer_damage,
    norm_preimages,
    train_tiny,
)
from spectraquant.proxies.analysis import (
    PREDECLARED_MDE_Z_CONSERVATIVE,
    PREDECLARED_MDE_Z_OPTIMISTIC,
    PREDECLARED_NEFF_CONSERVATIVE,
    PREDECLARED_NEFF_OPTIMISTIC,
    PREDECLARED_SEEDS,
    cluster_contrast,
    contrast_units,
    contrast_verdict,
    fisher_z,
    following_norm_container,
    mde_fisher_z,
    spearman,
    within_model_neff,
)
from spectraquant.proxies.base import LayerInputs
from spectraquant.proxies.gain import estimate_downstream_gains
from spectraquant.proxies.variants import get_proxy, proxy_names
from spectraquant.quantization import QuantSpec

__all__ = [
    "CANDIDATE",
    "CANDIDATE_PROXIES",
    "DEFAULT_CELLS",
    "DEFAULT_FIXTURES",
    "DEFAULT_SEEDS",
    "NAIVE_BASELINE",
    "PREDECLARED_COMPARATORS",
    "SCHEMA_ID",
    "SUBSTRATE",
    "build_document",
    "git_state",
    "run_sweep",
]

#: Artifact schema identifier; bumped when the document layout changes.
SCHEMA_ID = "spectraquant.proxy-validation/1"
#: Execution substrate for every number in this document (`preregistration.md` section 8).
SUBSTRATE = "LOCAL-FIXTURE"

#: Pre-registered seed floor for cheap cells (`preregistration.md` section 6: Tier 0 >= 5 seeds).
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
#: `(rank, bits)` cells spanning the pre-declared `ranks {2,4,8,16,32}` x `bits {2,3,4,8}` grid;
#: `r in {8,16,32}`, `b in {4,8}` (the H5-relevant bit widths).
DEFAULT_CELLS: tuple[tuple[int, int], ...] = ((8, 4), (16, 4), (32, 4), (8, 8), (16, 8), (32, 8))
#: The realistic fixture and the adversarial review's no-LayerNorm control.
DEFAULT_FIXTURES: tuple[tuple[str, bool], ...] = (("layernorm", True), ("no_layernorm", False))

GROUP_SIZE = 32
TRAIN_STEPS = 120
CALIBRATION_TOKENS = 24

#: The pre-registered candidate interface is `combined` (`design-m2-interfaces.md` section 7 item 3:
#: "`combined` (the candidate; weights must be explicit config values, never hard-coded)"), which is
#: also the variant whose component set matches the frozen H2 wording ("activation covariance +
#: quantization residual"). The repository's own status record (`docs/coordination/status.md`, item
#: 16) calls the **gain-aware composed** variant "the candidate" (rho = +0.830). Both are carried as
#: candidates, in that order, and the report states the evidence for each rather than resolving the
#: naming ambiguity by fiat.
CANDIDATE_PROXIES: tuple[str, ...] = ("combined", "gain_aware_composed")
#: The candidate the headline H2 verdict is stated for (the pre-registered interface).
CANDIDATE = CANDIDATE_PROXIES[0]
#: The naive pre-normalisation baseline whose failure is published (`design-m2-interfaces.md` 7.1).
NAIVE_BASELINE = "per_layer_output_error"
#: The pre-declared H2 comparator set: Frobenius is primary (`preregistration.md` section 7.2), the
#: rest are the pre-declared / exploratory comparators of section 7.2 and the freeze checklist.
PREDECLARED_COMPARATORS: tuple[str, ...] = (
    "weight_frobenius",
    "weight_magnitude",
    "activation_magnitude",
    "hessian_diag",
)

#: Significant digits kept for every float written into the artifact. The *sweep* is exact and
#: deterministic; this is a reporting precision so the committed JSON stays a reviewable size. It is
#: recorded in the document as ``float_precision_significant_digits``.
FLOAT_SIGNIFICANT_DIGITS = 6


def _num(value: float | None) -> float | None:
    """JSON-safe float rounded to :data:`FLOAT_SIGNIFICANT_DIGITS`; non-finite becomes ``None``."""
    if value is None:
        return None
    v = float(value)
    if not math.isfinite(v):
        return None
    return float(f"{v:.{FLOAT_SIGNIFICANT_DIGITS}g}")


def git_state(root: Path | str | None = None) -> dict[str, Any]:
    """Current commit SHA and dirty flag, or placeholders outside a checkout."""
    cwd = str(root) if root is not None else None
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=cwd,
        ).stdout.strip()
        porcelain = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            cwd=cwd,
        ).stdout
        return {"git_commit": sha, "git_dirty": bool(porcelain.strip())}
    except Exception:
        return {"git_commit": "unknown", "git_dirty": False}


def _calibration_ids(vocab: int) -> torch.Tensor:
    """Deterministic calibration token batch (fixed, independent of the training generator)."""
    return torch.arange(CALIBRATION_TOKENS)[None].repeat(2, 1) % vocab


def measure_cell(
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
    """Measure one ``(rank, bits)`` cell on one trained fixture: damage, proxies, per-model ``rho``.

    Args:
        model: a trained fixture transformer.
        ids: ``(batch, seq)`` calibration token ids.
        cfg: the fixture configuration (its ``d_model`` reshapes the gain estimator's output).
        names: the compressible layer names.
        rank: low-rank budget.
        bits: fake-quantization bit width.
        group_size: per-group quantization group size.
        n_boot_neff: within-model bootstrap replicates for ``n_eff``.
        neff_seed: RNG seed for that bootstrap (derived per model and cell).

    Returns:
        A JSON-safe mapping with the measured single-layer and joint damage summaries, the
        per-variant Spearman ``rho`` against single-layer damage (and against logits damage), the
        per-model ``n_eff`` and the gain summary. All class 2. Per-layer proxy values are *not*
        carried in the artifact (size); the sweep is deterministic, so re-running the script
        reproduces them bit-for-bit.
    """
    spec = QuantSpec(bits=bits, granularity="per_group", group_size=group_size, symmetric=True)
    damage = network_single_layer_damage(model, ids, names, rank, spec)
    joint_hidden, joint_logits = network_joint_damage(model, ids, names, rank, spec)
    activations = layer_inputs(model, ids, names)
    preimages = norm_preimages(model, ids)
    gains = estimate_downstream_gains(
        lambda nm, pert: model(ids, inject=(nm, pert))[1].reshape(-1, cfg.d_model),
        damage.layer_errors,
    )

    damages = [damage.per_layer[n] for n in names]
    damages_logits = [damage.per_layer_logits[n] for n in names]
    variants: dict[str, Any] = {}
    for variant in proxy_names():
        proxy = get_proxy(variant)
        values: list[float] = []
        for name in names:
            layer = LayerInputs(
                weight=model.linear_layers()[name].weight.detach(),
                activations=activations[name],
                following_norm=following_norm_container(model, name, preimages),
                residual_error=damage.residual_errors.get(name),
                downstream_gain=gains.per_layer[name],
            )
            values.append(float(proxy._context_layer(layer, rank, spec).value))  # type: ignore[attr-defined]
        neff = within_model_neff(values, damages, n_boot=n_boot_neff, seed=neff_seed)
        variants[variant] = {
            "rho": _num(spearman(values, damages)),
            "rho_logits": _num(spearman(values, damages_logits)),
            "n_eff": _num(neff.n_eff),
            "clipped": neff.clipped,
        }

    sum_single = sum(damage.per_layer.values())
    gain_values = [g for g in gains.per_layer.values() if g > 0.0]
    return {
        "rank": rank,
        "bits": bits,
        "group_size": group_size,
        "damage": {
            "sum_single_layer": _num(sum_single),
            "joint_hidden": _num(joint_hidden),
            "joint_logits": _num(joint_logits),
            "joint_over_sum_single": _num(joint_hidden / sum_single) if sum_single else None,
        },
        "gains": {
            "forward_calls": gains.forward_calls,
            "min": _num(min(gain_values)) if gain_values else None,
            "max": _num(max(gain_values)) if gain_values else None,
            "spread": _num(max(gain_values) / min(gain_values))
            if gain_values and min(gain_values) > 0.0
            else None,
        },
        "variants": variants,
    }


def run_sweep(
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    cells: Sequence[tuple[int, int]] = DEFAULT_CELLS,
    fixtures: Sequence[tuple[str, bool]] = DEFAULT_FIXTURES,
    train_steps: int = TRAIN_STEPS,
    group_size: int = GROUP_SIZE,
    n_boot_neff: int = 200,
    boot_seed: int = 20261009,
) -> list[dict[str, Any]]:
    """Train every fixture model once and measure every cell on it.

    Returns one run record per ``(fixture, seed)`` carrying the training provenance, the module
    order and the per-cell measured payload. Training is deterministic given ``fixture`` and
    ``seed``; the whole Tier-0 sweep is designed to finish in well under a minute on the CPU
    workstation.
    """
    runs: list[dict[str, Any]] = []
    for fixture, use_norm in fixtures:
        for seed in seeds:
            cfg = TinyConfig(use_norm=use_norm, seed=seed)
            model, training = train_tiny(cfg, steps=train_steps)
            names = layer_names(model)
            ids = _calibration_ids(cfg.vocab)
            cell_records: list[dict[str, Any]] = []
            for index, (rank, bits) in enumerate(cells):
                cell_records.append(
                    measure_cell(
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
                    "cells": cell_records,
                }
            )
    return runs


def _cell_of(run: Mapping[str, Any], rank: int, bits: int) -> dict[str, Any] | None:
    """The cell record of one run, or ``None`` when the run does not carry that cell."""
    for cell in run["cells"]:
        if cell["rank"] == rank and cell["bits"] == bits:
            return cell
    return None


def _select_runs(runs: Sequence[Mapping[str, Any]], fixture: str | None) -> list[Mapping[str, Any]]:
    """Runs belonging to one fixture group (``None`` = every run)."""
    return [r for r in runs if fixture is None or r["fixture"] == fixture]


def _usable(rec: Mapping[str, Any]) -> tuple[float, float] | None:
    """``(rho, n_eff)`` from a per-model variant record, or ``None`` when either is unusable."""
    rho, n_eff = rec["rho"], rec["n_eff"]
    if rho is None or n_eff is None or float(n_eff) <= 3.0:
        return None
    return float(rho), float(n_eff)


def _variant_pool(
    runs: Sequence[Mapping[str, Any]],
    fixture: str | None,
    variant: str,
) -> dict[str, list[tuple[float, float]]]:
    """``cluster id -> [(z, var)]`` for one variant over a fixture group's cells.

    A cluster is one trained model (``fixture:seed``); its units are the cells measured on it. Models
    whose ``rho`` is undefined (constant proxy on modules) or whose ``n_eff`` is unusable (``<= 3``)
    contribute no unit.
    """
    out: dict[str, list[tuple[float, float]]] = {}
    for run in _select_runs(runs, fixture):
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


def _contrast_units(
    runs: Sequence[Mapping[str, Any]],
    fixture: str | None,
    candidate: str,
    comparator: str,
    *,
    rank: int | None = None,
    bits: int | None = None,
) -> dict[str, list[tuple[float, float]]]:
    """``cluster id -> [(d, var_d)]`` for the paired candidate-vs-comparator Fisher-z difference.

    One cluster is one trained model (``fixture:seed``); its units are the cells selected by the
    ``rank``/``bits`` filters (all cells when both are ``None``).
    """
    out: dict[str, list[tuple[float, float]]] = {}
    for run in _select_runs(runs, fixture):
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


def _n_clusters(runs: Sequence[Mapping[str, Any]], fixture: str | None) -> int:
    return len({f"{r['fixture']}:{r['seed']}" for r in _select_runs(runs, fixture)})


def _mde_block(n_eff: float | None, n_models: int) -> dict[str, Any]:
    """Achieved MDE next to the pre-declared band (`preregistration.md` section 7.5)."""
    achieved = mde_fisher_z(float(n_eff), n_models) if n_eff is not None else float("nan")
    return {
        "n_eff_reference": _num(n_eff),
        "n_models": n_models,
        "achieved_z": _num(achieved),
        "predeclared_optimistic_z": PREDECLARED_MDE_Z_OPTIMISTIC,
        "predeclared_conservative_z": PREDECLARED_MDE_Z_CONSERVATIVE,
    }


def _contrast_payload(
    units: dict[str, list[tuple[float, float]]],
    *,
    label: str,
    n_boot_ci: int,
    boot_seed: int,
    n_eff_ref: float | None,
    with_rationale: bool,
    compact: bool,
) -> dict[str, Any]:
    """Contrast payload: point estimate, analytic random-effects CI, cluster-bootstrap CI, verdict.

    ``compact=True`` returns the array form documented by :data:`CONTRAST_FIELDS` for the per-cell
    table; the aggregate table uses named fields so the headline numbers read directly.
    """
    result = cluster_contrast(units, label=label, n_boot=n_boot_ci, seed=boot_seed)
    verdict = contrast_verdict(result).to_dict()
    if not with_rationale:
        verdict.pop("rationale", None)
    n_eff_achieved = mde_fisher_z(
        float(n_eff_ref) if n_eff_ref is not None else float("nan"), result.n_clusters
    )
    analytic_sign = 1 if result.ci_low > 0 else (-1 if result.ci_high < 0 else 0)
    boot_sign = 1 if result.boot_ci_low > 0 else (-1 if result.boot_ci_high < 0 else 0)
    if compact:
        return {
            "delta_z": _num(result.delta_z),
            "ci": [_num(result.ci_low), _num(result.ci_high)],
            "boot_ci": [_num(result.boot_ci_low), _num(result.boot_ci_high)],
            "p_value": _num(result.p_value),
            "k": result.k,
            "ci_agreement": analytic_sign == boot_sign,
            "verdict": verdict,
        }
    return {
        "label": label,
        "delta_z": _num(result.delta_z),
        "ci": [_num(result.ci_low), _num(result.ci_high)],
        "boot_ci": [_num(result.boot_ci_low), _num(result.boot_ci_high)],
        "p_value": _num(result.p_value),
        "tau2": _num(result.tau2),
        "i2": _num(result.i2),
        "k": result.k,
        "n_clusters": result.n_clusters,
        "n_boot": result.n_boot,
        "ci_agreement": analytic_sign == boot_sign,
        "mde_achieved_z": _num(n_eff_achieved),
        "verdict": verdict,
    }


def build_document(
    runs: Sequence[Mapping[str, Any]],
    *,
    cells: Sequence[tuple[int, int]] = DEFAULT_CELLS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
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
    """Assemble the machine-readable validation document from measured runs.

    The document carries the raw per-model evidence (``runs``), the per-cell table (mean ``rho`` per
    variant, the model-level Fisher-z pool, achieved ``n_eff``, the paired contrasts with both CIs
    and the achieved MDE), the per-fixture and combined aggregates with the falsifier verdict against
    the frozen section 11.1 wording, and the explicit ``NOT RUN`` cloud cells.

    Arrays in ``runs`` are ordered by ``runs.seeds``; ``plan.modules`` gives the module order that the
    per-cell ``rho`` refers to.
    """
    variants = list(proxy_names())

    runs_section: dict[str, Any] = {
        "seeds": list(seeds),
        "training": {},
        "cells": {},
    }
    for fixture, _ in fixtures:
        fixture_runs = _select_runs(runs, fixture)
        runs_section["training"][fixture] = {
            str(run["seed"]): {k: _num(v) for k, v in run["training"].items()}
            for run in fixture_runs
        }
        runs_section["cells"][fixture] = [
            {
                "rank": rank,
                "bits": bits,
                "damage": {
                    key: [
                        _num(_cell_of(run, rank, bits)["damage"][key])  # type: ignore[index]
                        for run in fixture_runs
                    ]
                    for key in (
                        "sum_single_layer",
                        "joint_hidden",
                        "joint_logits",
                        "joint_over_sum_single",
                    )
                },
                "gains": {
                    "forward_calls": [
                        _cell_of(run, rank, bits)["gains"]["forward_calls"]  # type: ignore[index]
                        for run in fixture_runs
                    ],
                    "spread": [
                        _num(_cell_of(run, rank, bits)["gains"]["spread"])  # type: ignore[index]
                        for run in fixture_runs
                    ],
                },
                "rho": {
                    variant: [
                        _cell_of(run, rank, bits)["variants"][variant]["rho"]  # type: ignore[index]
                        for run in fixture_runs
                    ]
                    for variant in variants
                },
                "rho_logits": {
                    variant: [
                        _cell_of(run, rank, bits)["variants"][variant]["rho_logits"]  # type: ignore[index]
                        for run in fixture_runs
                    ]
                    for variant in variants
                },
                "n_eff": {
                    variant: [
                        _cell_of(run, rank, bits)["variants"][variant]["n_eff"]  # type: ignore[index]
                        for run in fixture_runs
                    ]
                    for variant in variants
                },
            }
            for rank, bits in cells
        ]

    per_cell: list[dict[str, Any]] = []
    for fixture, _ in fixtures:
        fixture_runs = _select_runs(runs, fixture)
        for rank, bits in cells:
            cell_variants: dict[str, Any] = {}
            for variant in variants:
                recs = [
                    _cell_of(run, rank, bits)["variants"][variant]  # type: ignore[index]
                    for run in fixture_runs
                ]
                usable = [u for u in (_usable(rec) for rec in recs) if u is not None]
                rhos = [u[0] for u in usable]
                n_effs = [u[1] for u in usable]
                cell_variants[variant] = {
                    "rho_mean": _num(sum(rhos) / len(rhos)) if rhos else None,
                    "n_eff_mean": _num(sum(n_effs) / len(n_effs)) if n_effs else None,
                    "n_models_used": len(usable),
                }
            contrasts: dict[str, Any] = {}
            for candidate in CANDIDATE_PROXIES:
                candidate_contrasts: dict[str, Any] = {}
                for comparator in (*PREDECLARED_COMPARATORS, NAIVE_BASELINE):
                    if comparator == candidate:
                        continue
                    units = _contrast_units(
                        runs, fixture, candidate, comparator, rank=rank, bits=bits
                    )
                    if not units:
                        continue
                    candidate_contrasts[comparator] = _contrast_payload(
                        units,
                        label=f"{candidate} - {comparator}",
                        n_boot_ci=n_boot_ci,
                        boot_seed=boot_seed,
                        n_eff_ref=cell_variants[candidate]["n_eff_mean"],
                        with_rationale=False,
                        compact=True,
                    )
                contrasts[candidate] = candidate_contrasts
            n_models = len(fixture_runs)
            per_cell.append(
                {
                    "fixture": fixture,
                    "rank": rank,
                    "bits": bits,
                    "group_size": group_size,
                    "n_models": n_models,
                    "damage_mean": {
                        key: _num(
                            sum(
                                float(_cell_of(run, rank, bits)["damage"][key])  # type: ignore[index]
                                for run in fixture_runs
                            )
                            / max(1, n_models)
                        )
                        for key in ("sum_single_layer", "joint_hidden", "joint_over_sum_single")
                    },
                    "variants": cell_variants,
                    "contrasts": contrasts,
                    "mde": _mde_block(cell_variants[CANDIDATE]["n_eff_mean"], n_models),
                }
            )

    aggregate: dict[str, Any] = {}
    groups: list[tuple[str, str | None]] = [(name, name) for name, _ in fixtures]
    groups.append(("all_fixtures", None))
    for group_name, fixture in groups:
        variant_table: dict[str, Any] = {}
        for variant in variants:
            pool = _variant_pool(runs, fixture, variant)
            usable = [
                u
                for run in _select_runs(runs, fixture)
                for cell in run["cells"]
                if (u := _usable(cell["variants"][variant])) is not None
            ]
            rhos = [u[0] for u in usable]
            n_effs = [u[1] for u in usable]
            pooled = contrast_units(pool) if pool else None
            variant_table[variant] = {
                "k": pooled.k if pooled else 0,
                "n_clusters": len(pool) if pool else 0,
                "pooled": _num(pooled.pooled) if pooled else None,
                "se": _num(pooled.se) if pooled else None,
                "ci": [_num(pooled.ci_low), _num(pooled.ci_high)] if pooled else None,
                "p_value": _num(pooled.p_value) if pooled else None,
                "tau2": _num(pooled.tau2) if pooled else None,
                "i2": _num(pooled.i2) if pooled else None,
                "rho_mean": _num(sum(rhos) / len(rhos)) if rhos else None,
                "n_eff_mean": _num(sum(n_effs) / len(n_effs)) if n_effs else None,
            }
        contrasts: dict[str, Any] = {}
        for candidate in CANDIDATE_PROXIES:
            candidate_contrasts: dict[str, Any] = {}
            for comparator in (*PREDECLARED_COMPARATORS, NAIVE_BASELINE):
                if comparator == candidate:
                    continue
                units = _contrast_units(runs, fixture, candidate, comparator)
                if not units:
                    continue
                candidate_contrasts[comparator] = _contrast_payload(
                    units,
                    label=f"{candidate} - {comparator}",
                    n_boot_ci=n_boot_ci,
                    boot_seed=boot_seed,
                    n_eff_ref=variant_table[candidate]["n_eff_mean"],
                    with_rationale=True,
                    compact=False,
                )
            contrasts[candidate] = candidate_contrasts
        n_clusters = _n_clusters(runs, fixture)
        aggregate[group_name] = {
            "variants": variant_table,
            "contrasts": contrasts,
            "primary_candidate": CANDIDATE,
            "primary_contrast": PREDECLARED_COMPARATORS[0],
            "primary_verdict": contrasts.get(CANDIDATE, {})
            .get(PREDECLARED_COMPARATORS[0], {})
            .get("verdict"),
            "n_clusters": n_clusters,
            "mde": _mde_block(variant_table[CANDIDATE]["n_eff_mean"], n_clusters),
        }

    return {
        "schema": SCHEMA_ID,
        "generated_by": "scripts/experiments/proxy_validation_sweep.py",
        "substrate": SUBSTRATE,
        "measurement_class": 2,
        "statistical_unit": "model",
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "float_precision_significant_digits": FLOAT_SIGNIFICANT_DIGITS,
        "runtime": {
            "wall_time_s": _num(wall_time_s),
            "n_models": len(runs),
            "n_cells": len(per_cell),
            "n_variants": len(variants),
        },
        "plan": {
            "seeds": list(seeds),
            "cells": [list(c) for c in cells],
            "fixtures": [name for name, _ in fixtures],
            "modules": list(runs[0]["modules"]) if runs else [],
            "group_size": group_size,
            "train_steps": train_steps,
            "candidate_proxy": CANDIDATE,
            "candidate_proxies": list(CANDIDATE_PROXIES),
            "naive_baseline": NAIVE_BASELINE,
            "predeclared_comparators": list(PREDECLARED_COMPARATORS),
            "n_boot_neff": n_boot_neff,
            "n_boot_ci": n_boot_ci,
            "boot_seed": boot_seed,
        },
        "predeclared_mde": {
            "n_eff_optimistic": PREDECLARED_NEFF_OPTIMISTIC,
            "n_eff_conservative": PREDECLARED_NEFF_CONSERVATIVE,
            "seeds": PREDECLARED_SEEDS,
            "mde_z_optimistic": PREDECLARED_MDE_Z_OPTIMISTIC,
            "mde_z_conservative": PREDECLARED_MDE_Z_CONSERVATIVE,
            "source": "preregistration.md section 7.5 (frozen 2026-10-09)",
        },
        "per_layer_proxy_evidence": {
            "included": False,
            "reason": "Per-layer proxy values are omitted from the committed artifact (size); the "
            "sweep is deterministic at the recorded seeds and re-running "
            "scripts/experiments/proxy_validation_sweep.py reproduces them bit-for-bit. Per-model "
            "`rho`, `rho_logits` and `n_eff` plus the measured damage summaries are kept, which is "
            "everything the Fisher-z aggregation consumes.",
        },
        "runs": runs_section,
        "per_cell": per_cell,
        "aggregate": aggregate,
        "cloud_cells": {
            "tier1_proxy_ranking_h2": {
                "confirmatory_cell": "Proxy vs weight-space Frobenius error rank correlation "
                "(activation covariance + quantization residual in the proxy), tiny LM, >=5 seeds",
                "hypothesis": "H2",
                "tier": 1,
                "substrate": "CLOUD-COLAB",
                "milestone": "M4",
                "status": "NOT RUN",
                "reason": "Tier-1 training/evaluation runs only on the cloud notebook substrate "
                "(AGENTS.md section 2.3/2b); no validated run_manifest.json exists yet. No Tier-1 "
                "number is implied anywhere in this document.",
            },
            "tier1_mixed_allocation_h4": {
                "confirmatory_cell": "Layer-wise mixed rank + mixed precision vs uniform and vs "
                "HAWQ-V2-style, >=5 seeds",
                "hypothesis": "H4",
                "tier": 1,
                "substrate": "CLOUD-COLAB",
                "milestone": "M4",
                "status": "NOT RUN",
                "reason": "Same as above: the H4 confirmatory cell is CLOUD-COLAB (section 8 row 7) "
                "and has not been executed.",
            },
        },
    }
