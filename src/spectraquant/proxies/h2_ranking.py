"""Real-model H2 ranking measurement on the CPU substrate — measurement class 2.

The project's core contribution is the output-aware sensitivity proxy that the allocator uses to
choose layer-wise ranks and bit widths. Its ranking ability was only ever measured on the 13-module
synthetic Tier-0 fixture (``spectraquant.proxies.validation``). This module builds the **real-model**
measurement: a real pretrained transformer, a real calibration batch drawn from a training-role
split, several compression variants per compressible linear layer, and the per-layer damage each
variant actually causes.

What is measured
----------------

For one model, one calibration batch and one :class:`CompressionVariant`:

* **measured damage** per compressible linear layer ``l``, on two targets, both computed from the
  real captured activations and the real uncompressed forward pass:

  * ``layer_output`` — the mean squared change of the layer's own output,
    ``mean_i ||x_i W_l^T - x_i W_hat_l^T||^2`` with ``W_hat = Q(B) Q(A)`` the frozen compression
    operator of :mod:`spectraquant.proxies.operators` (the frozen proxy unit);
  * ``model_output`` — the mean squared change of the final hidden state when that layer alone is
    compressed (the Tier-0 fixture's end-to-end target);

* **predicted damage** per layer — the value of every registered proxy variant
  (:func:`spectraquant.proxies.variants.proxy_names`) scored through the public
  :meth:`Proxy.score_model` entry point (``per_layer`` carries one value per layer), so no proxy is
  reimplemented here;
* the **model-level statistic** — one Spearman ``rho`` per proxy per cell per target across that
  model's compressible layers, transformed to Fisher-``z``, with ``var(z) = 1 / (n_eff - 3)`` and
  ``n_eff`` estimated by a within-model module bootstrap (the pre-registered convention,
  ``preregistration.md`` section 7.0/7.2: **the model is the unit; modules are a within-model
  nuisance**). The paired difference against each predeclared comparator is combined with the
  project's DerSimonian-Laird Fisher-``z`` random-effects helper
  (:func:`spectraquant.proxies.analysis.paired_fisher_z_contrast` /
  :func:`~spectraquant.proxies.analysis.cluster_contrast`), and the achieved MDE is reported next to
  the predeclared band.

The measurement class is **2** (fake quantization: float execution that simulates quantization
numerics). Nothing here is a packed-storage (class 3) or kernel-backed (class 4) measurement, and no
latency is claimed. The substrate is the local CPU workstation: no CUDA path exists in this module
(``AGENTS.md`` section 2).

Known structural properties (reported, not hidden)
--------------------------------------------------

* ``per_layer_output_error`` is the exact plug-in estimator of the ``layer_output`` damage by
  construction, so its ``rho`` is ``1`` on every cell of that target (and its bootstrap variance is
  zero, so it contributes no aggregation unit). It is a *sanity anchor*, not a competitor; the
  artifact states this. On the ``model_output`` target it is a genuine (and much weaker) predictor.
* ``in_situ_output_error`` and ``gain_aware_composed`` require context (a following normalisation, a
  measured residual error, an estimated downstream gain). When a :class:`LayerInputs` lacks that
  context the variants document their own fallback (``fallback = 1`` in ``diagnostics``) and the
  artifact records the per-cell fallback fraction; the measurement driver supplies the context it
  can measure.

Determinism
-----------

Every random choice is seeded: the calibration batch is a fixed, capped, revision-pinned read; the
``n_eff`` bootstrap takes a per-cell seed; the SVD is a deterministic LAPACK routine. Re-running the
measurement script on the same model revision and calibration cap reproduces the numbers.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from spectraquant.proxies.analysis import (
    cluster_contrast,
    contrast_verdict,
    fisher_z,
    mde_fisher_z,
    paired_fisher_z_contrast,
    spearman,
    within_model_neff,
)
from spectraquant.proxies.base import LayerInputs, ProxyInputError
from spectraquant.proxies.operators import compression_delta, quantized_factor_reconstruction
from spectraquant.proxies.validation import (
    CANDIDATE,
    CANDIDATE_PROXIES,
    NAIVE_BASELINE,
    PREDECLARED_COMPARATORS,
)
from spectraquant.proxies.variants import get_proxy, proxy_names
from spectraquant.quantization import QuantSpec

__all__ = [
    "CALIBRATION_ROLE",
    "CANDIDATE",
    "CANDIDATE_PROXIES",
    "DEFAULT_CALIBRATION_SEED",
    "DEFAULT_GROUP_SIZE",
    "DEFAULT_MODEL_ID",
    "DEFAULT_MODEL_REVISION",
    "DEFAULT_PROXIES",
    "DEFAULT_VARIANTS",
    "FLOAT_SIGNIFICANT_DIGITS",
    "MEASUREMENT_CLASS",
    "NAIVE_BASELINE",
    "PREDECLARED_COMPARATORS",
    "SCHEMA_ID",
    "STATISTICAL_UNIT",
    "SUBSTRATE",
    "CellRanking",
    "CompressionVariant",
    "build_document",
    "compressed_weight",
    "contrast_payload",
    "layer_output_damage",
    "mde_block",
    "measure_cell",
    "rank_cell",
    "variant_spec",
]

#: Artifact schema identifier; bumped when the document layout changes.
SCHEMA_ID = "spectraquant.h2-real-model/1"
#: Execution substrate. Every number is a local CPU float-execution measurement.
SUBSTRATE = "LOCAL-CPU"
#: ``AGENTS.md`` section 5: fake-quantization quality damage, never storage and never a kernel.
MEASUREMENT_CLASS = 2
#: Pre-registered resampling unit (`preregistration.md` section 7.0): the trained model.
STATISTICAL_UNIT = "model"

#: Default model and the frozen revision of the Tier-1 pilot plan (`configs/tier1/smollm2_135m.yaml`).
DEFAULT_MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
DEFAULT_MODEL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
#: Per-group symmetric fake quantization, the frozen M2 default (``validation.GROUP_SIZE``).
DEFAULT_GROUP_SIZE = 32
#: The plan's derived ``calibration_sampling`` seed (`configs/tier1/smollm2_135m.yaml`).
DEFAULT_CALIBRATION_SEED = 20261011
#: The calibration corpus is read from this **role**, never from the test role.
#: ``ROLE_SPLITS["calibration"]`` is the dataset's own ``train`` split (`preregistration.md` section
#: 7; no test-set peeking, ``AGENTS.md`` section 4.6).
CALIBRATION_ROLE = "train"

#: Reporting precision for the committed artifact (the measurement itself is full precision).
FLOAT_SIGNIFICANT_DIGITS = 6

#: Every registered proxy variant, scored through its public ``score_model`` entry point.
DEFAULT_PROXIES: tuple[str, ...] = tuple(proxy_names())


@dataclass(frozen=True)
class CompressionVariant:
    """One compression setting a proxy is asked to rank, expressed in the frozen ``(rank, spec)`` space.

    The frozen proxy interface scores ``(w, x, rank, spec)`` and the frozen compression operator is
    ``W_hat = Q(B) Q(A)`` (:mod:`spectraquant.proxies.operators`). A variant therefore *is* a
    ``(rank, spec)`` pair, and the measured damage and the proxy prediction are computed from the
    same ``W_hat`` — they cannot disagree about what was compressed.

    Args:
        name: stable key used in the artifact.
        kind: ``"rank"`` (rank-dominant; the highest available bit width, 8, is the near-lossless
            setting of the frozen grid), ``"quant"`` (quantization-dominant; full rank, so the SVD
            truncation is exact and only the factors are rounded), or ``"rank_quant"`` (both).
        rank: requested low-rank budget; ``rank == 0`` means "full rank" and is resolved against the
            layer's ``min(shape)`` at compression time.
        bits: fake-quantization bit width; must be one of
            :data:`spectraquant.quantization.fake_quant.SUPPORTED_BITS`.
        group_size: per-group quantization group size (used with ``granularity="per_group"``).
        description: human-readable statement of what the variant does.
    """

    name: str
    kind: str
    rank: int
    bits: int
    group_size: int = DEFAULT_GROUP_SIZE
    description: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("rank", "quant", "rank_quant"):
            raise ProxyInputError(f"kind must be rank|quant|rank_quant, got {self.kind!r}")
        if self.rank < 0:
            raise ProxyInputError(f"rank must be >= 0 (0 = full rank), got {self.rank}")
        if self.group_size <= 0:
            raise ProxyInputError(f"group_size must be > 0, got {self.group_size}")


#: The pre-declared minimum variant set: rank-dominant at two ranks (8-bit factors are the
#: near-lossless setting of the frozen grid), quantization-dominant at two bit widths (full rank, so
#: truncation is exact), and two low-rank-plus-quantized combinations.
DEFAULT_VARIANTS: tuple[CompressionVariant, ...] = (
    CompressionVariant(
        name="rank8_b8",
        kind="rank",
        rank=8,
        bits=8,
        description="rank-8 truncation with 8-bit factors (near-lossless quantization)",
    ),
    CompressionVariant(
        name="rank32_b8",
        kind="rank",
        rank=32,
        bits=8,
        description="rank-32 truncation with 8-bit factors (near-lossless quantization)",
    ),
    CompressionVariant(
        name="quant_b4",
        kind="quant",
        rank=0,
        bits=4,
        description="full-rank factor quantization at 4 bits (no truncation)",
    ),
    CompressionVariant(
        name="quant_b2",
        kind="quant",
        rank=0,
        bits=2,
        description="full-rank factor quantization at 2 bits (no truncation)",
    ),
    CompressionVariant(
        name="rank16_b4",
        kind="rank_quant",
        rank=16,
        bits=4,
        description="rank-16 truncation plus 4-bit factor quantization",
    ),
    CompressionVariant(
        name="rank8_b2",
        kind="rank_quant",
        rank=8,
        bits=2,
        description="rank-8 truncation plus 2-bit factor quantization",
    ),
)


def variant_spec(variant: CompressionVariant) -> QuantSpec:
    """The :class:`QuantSpec` a variant's factors are quantized with (per-group, symmetric)."""
    return QuantSpec(
        bits=variant.bits,
        granularity="per_group",
        group_size=variant.group_size,
        symmetric=True,
    )


def _effective_rank(w: torch.Tensor, variant: CompressionVariant) -> int:
    """Resolve ``variant.rank`` (``0`` = full rank) against the layer's ``min(shape)``."""
    full = int(min(w.shape))
    return full if variant.rank == 0 else min(variant.rank, full)


def compressed_weight(w: torch.Tensor, variant: CompressionVariant) -> torch.Tensor:
    """``Q(B) Q(A)`` — the variant's deployed weight, through the frozen compression operator.

    Shapes:
        ``w``: ``(out, in)`` 2-D CPU float tensor. Returns the same shape/dtype.
    """
    if w.dim() != 2:
        raise ProxyInputError(f"w must be 2-D (out, in), got {tuple(w.shape)}")
    return quantized_factor_reconstruction(w, _effective_rank(w, variant), variant_spec(variant))


def layer_output_damage(w: torch.Tensor, x: torch.Tensor, variant: CompressionVariant) -> float:
    """The **measured** damage a variant causes to one layer's output, on real activations.

    ``mean_i ||x_i W^T - x_i W_hat^T||^2`` in float64, with ``W_hat`` the variant's
    :func:`compressed_weight`. This is the frozen proxy unit evaluated on the real activation batch.

    Shapes:
        ``w``: ``(out, in)``; ``x``: ``(n_samples, in)``. Returns a Python float.
    """
    delta = compression_delta(w, _effective_rank(w, variant), variant_spec(variant))
    proj = x.to(torch.float64) @ delta.to(torch.float64).transpose(0, 1)
    return float(torch.mean(torch.sum(proj * proj, dim=-1)))


@dataclass(frozen=True)
class CellRanking:
    """The model-level ranking statistic for one (proxy, cell) pair.

    Attributes:
        rho: Spearman rank correlation between the proxy's per-layer values and the measured damage,
            over the model's compressible layers; ``nan`` when undefined.
        z: Fisher transform ``atanh(rho)`` (clipped at ``|rho| = 1 - 1e-6``).
        variance: ``1 / (n_eff - 3)``, the pre-registered within-model variance of ``z``.
        n_eff: bootstrap effective module count (:func:`within_model_neff`); ``nan`` when unusable.
        n_modules: number of layers the correlation was computed over.
        clipped: whether ``|rho|`` hit the Fisher clipping bound.
        usable: whether ``rho`` and ``n_eff`` are finite and ``n_eff > 3`` (a usable aggregation
            unit). Unusable cells contribute nothing to the random-effects pool.
    """

    rho: float
    z: float
    variance: float
    n_eff: float
    n_modules: int
    clipped: bool
    usable: bool

    def to_dict(self) -> dict[str, object]:
        """JSON-safe mapping (rounded at :data:`FLOAT_SIGNIFICANT_DIGITS`)."""
        return {
            "rho": _round(self.rho),
            "z": _round(self.z),
            "variance": _round(self.variance),
            "n_eff": _round(self.n_eff),
            "n_modules": self.n_modules,
            "clipped": self.clipped,
            "usable": self.usable,
        }


def rank_cell(
    proxy_values: Sequence[float],
    damages: Sequence[float],
    *,
    n_boot: int = 200,
    seed: int = 0,
) -> CellRanking:
    """Compute one (proxy, cell) model-level ranking statistic from per-layer values.

    This is the pre-registered model-level statistic (`preregistration.md` section 7.2): Spearman
    ``rho`` over the model's modules, Fisher-``z`` transform, and ``var(z) = 1 / (n_eff - 3)`` with
    ``n_eff`` from a within-model module bootstrap (resample with replacement, require at least
    ``ceil(n / 2)`` distinct indices, else redraw).

    Shapes:
        ``proxy_values`` and ``damages`` are equal-length sequences over the model's modules.

    Limitations:
        With a real model's module count (hundreds) the bootstrap is well behaved, but ``n_eff`` is
        still an estimate and is reported per cell, never as exact.
    """
    n = len(proxy_values)
    if len(damages) != n:
        raise ProxyInputError(f"length mismatch: {n} proxy values vs {len(damages)} damages")
    rho = spearman(proxy_values, damages)
    if n < 4 or not math.isfinite(rho):
        return CellRanking(float("nan"), float("nan"), float("nan"), float("nan"), n, False, False)
    neff = within_model_neff(proxy_values, damages, n_boot=n_boot, seed=seed)
    if not math.isfinite(neff.n_eff) or neff.n_eff <= 3.0:
        return CellRanking(rho, fisher_z(rho), float("nan"), neff.n_eff, n, neff.clipped, False)
    z = fisher_z(rho)
    return CellRanking(
        rho=rho,
        z=z,
        variance=1.0 / (neff.n_eff - 3.0),
        n_eff=neff.n_eff,
        n_modules=n,
        clipped=neff.clipped,
        usable=True,
    )


def measure_cell(
    layers: Mapping[str, LayerInputs],
    damages: Mapping[str, float],
    variant: CompressionVariant,
    *,
    proxies: Sequence[str] = DEFAULT_PROXIES,
    n_boot_neff: int = 200,
    neff_seed: int = 0,
) -> dict[str, object]:
    """Score every proxy on one cell and compute the per-proxy model-level ranking statistic.

    Args:
        layers: ``name -> LayerInputs`` for the model's compressible layers, on the calibration
            batch (``weight`` and real ``activations``; optional context is passed through to the
            variants that use it).
        damages: ``name -> measured damage`` for the same layers and the same variant.
        variant: the compression setting both the damage and the proxies refer to.
        proxies: proxy variant names to score (default: every registered variant).
        n_boot_neff: within-model bootstrap replicates for ``n_eff``.
        neff_seed: RNG seed for that bootstrap (derive one per cell for determinism).

    Returns:
        A JSON-safe cell record carrying the variant identity, the damage summary, and one
        :class:`CellRanking` per proxy. Per-layer proxy values are not carried (size); the
        measurement is deterministic and re-running the script reproduces them.
    """
    names = list(layers)
    if set(names) != set(damages):
        raise ProxyInputError(
            f"layers and damages must share the same key set: {sorted(names)} vs {sorted(damages)}"
        )
    if not names:
        raise ProxyInputError("no layers to score")
    order = sorted(names)
    ranks = {name: _effective_rank(layers[name].weight, variant) for name in order}
    specs = {name: variant_spec(variant) for name in order}
    ranked = {name: float(damages[name]) for name in order}

    variants: dict[str, dict[str, object]] = {}
    for proxy_name in proxies:
        proxy = get_proxy(proxy_name)
        result = proxy.score_model(  # type: ignore[attr-defined]
            {name: layers[name] for name in order}, ranks, specs
        )
        values = [float(result.per_layer[name]) for name in order]
        ranking = rank_cell(
            values, [ranked[name] for name in order], n_boot=n_boot_neff, seed=neff_seed
        )
        record = ranking.to_dict()
        record["fallback_fraction"] = _fallback_fraction(result.diagnostics, len(order))
        variants[proxy_name] = record

    values = [ranked[name] for name in order]
    return {
        "variant": variant.name,
        "kind": variant.kind,
        "rank": variant.rank,
        "bits": variant.bits,
        "group_size": variant.group_size,
        "description": variant.description,
        "n_modules": len(order),
        "damage": {
            "sum": _round(float(sum(values))),
            "mean": _round(float(sum(values) / len(values))),
            "min": _round(float(min(values))),
            "max": _round(float(max(values))),
        },
        "variants": variants,
    }


def _fallback_fraction(diagnostics: Mapping[str, float], n_modules: int) -> float:
    """Fraction of layers whose proxy call reported a context fallback (``0`` when none did)."""
    if n_modules <= 0:
        return 0.0
    fell_back = sum(
        1 for key, value in diagnostics.items() if key.endswith("/fallback") and value > 0.0
    )
    return fell_back / n_modules


def _usable(cell: Mapping[str, object], proxy: str) -> tuple[float, float] | None:
    """``(z, variance)`` from a cell's per-proxy record, or ``None`` when the cell is unusable."""
    record = cell["variants"][proxy]  # type: ignore[index]
    if not record["usable"]:  # type: ignore[index]
        return None
    return float(record["z"]), float(record["variance"])  # type: ignore[index]


def mde_block(n_eff: float | None, n_models: int) -> dict[str, object]:
    """Achieved MDE next to the pre-declared band (`preregistration.md` section 7.5)."""
    achieved = mde_fisher_z(float(n_eff), n_models) if n_eff is not None else float("nan")
    return {
        "n_eff_reference": _round(n_eff),
        "n_models": n_models,
        "achieved_z": _round(achieved),
    }


def contrast_payload(
    cells: Sequence[Mapping[str, object]],
    candidate: str,
    comparator: str,
    *,
    model_id: str,
    n_boot_ci: int,
    boot_seed: int,
    n_eff_ref: float | None,
) -> dict[str, object]:
    """Paired candidate-vs-comparator Fisher-z contrast with analytic and bootstrap intervals.

    The pre-registered interval resamples **models** (clusters); with one model that bootstrap is
    degenerate, so a **cell-level** bootstrap is reported beside it as an explicitly labelled
    robustness check. The decision interval is the analytic random-effects interval, as
    :func:`spectraquant.proxies.analysis.contrast_verdict` prescribes.
    """
    cand_z: list[float] = []
    comp_z: list[float] = []
    cand_var: list[float] = []
    comp_var: list[float] = []
    cell_units: dict[str, list[tuple[float, float]]] = {}
    for cell in cells:
        cand = _usable(cell, candidate)
        comp = _usable(cell, comparator)
        if cand is None or comp is None:
            continue
        cand_z.append(cand[0])
        comp_z.append(comp[0])
        cand_var.append(cand[1])
        comp_var.append(comp[1])
        cell_units[str(cell["variant"])] = [(cand[0] - comp[0], cand[1] + comp[1])]
    if not cell_units:
        return {"label": f"{candidate} - {comparator}", "k": 0, "n_clusters": 0, "usable": False}

    # The pre-registered paired contrast (`preregistration.md` section 7.2): one paired difference
    # per cell, combined by a DerSimonian-Laird random-effects model with `var(d) = v_cand + v_comp`.
    point = paired_fisher_z_contrast(cand_z, comp_z, cand_var, comp_var)
    # The pre-registered interval resamples models (clusters); with one model it is degenerate, so
    # the same units are also bootstrapped cell-by-cell as an explicitly labelled robustness check.
    model_boot = cluster_contrast(
        {
            model_id: [
                (c - k, vc + vk)
                for c, k, vc, vk in zip(cand_z, comp_z, cand_var, comp_var, strict=True)
            ]
        },
        label=f"{candidate} - {comparator}",
        n_boot=n_boot_ci,
        seed=boot_seed,
    )
    cell_boot = cluster_contrast(
        cell_units,
        label=f"{candidate} - {comparator} (cell bootstrap)",
        n_boot=n_boot_ci,
        seed=boot_seed,
    )
    verdict = contrast_verdict(model_boot).to_dict()
    analytic_sign = 1 if point.ci_low > 0 else (-1 if point.ci_high < 0 else 0)
    cell_sign = 1 if cell_boot.boot_ci_low > 0 else (-1 if cell_boot.boot_ci_high < 0 else 0)
    return {
        "label": f"{candidate} - {comparator}",
        "delta_z": _round(point.pooled),
        "ci": [_round(point.ci_low), _round(point.ci_high)],
        "p_value": _round(point.p_value),
        "tau2": _round(point.tau2),
        "i2": _round(point.i2),
        "k": point.k,
        "n_clusters": model_boot.n_clusters,
        "n_boot": model_boot.n_boot,
        "model_bootstrap_degenerate": model_boot.n_clusters < 2,
        "model_boot_ci": [_round(model_boot.boot_ci_low), _round(model_boot.boot_ci_high)],
        "cell_boot_ci": [_round(cell_boot.boot_ci_low), _round(cell_boot.boot_ci_high)],
        "ci_agreement": analytic_sign == cell_sign,
        "mde_achieved_z": _round(mde_fisher_z(float(n_eff_ref), 1) if n_eff_ref else None),
        "verdict": verdict,
        "usable": True,
    }


def build_document(
    per_cell: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    target_notes: Mapping[str, str] | None = None,
    model: Mapping[str, object],
    calibration: Mapping[str, object],
    variants: Sequence[CompressionVariant] = DEFAULT_VARIANTS,
    proxies: Sequence[str] = DEFAULT_PROXIES,
    candidate: str = CANDIDATE,
    comparators: Sequence[str] = PREDECLARED_COMPARATORS,
    naive_baseline: str = NAIVE_BASELINE,
    group_size: int = DEFAULT_GROUP_SIZE,
    n_boot_neff: int = 200,
    n_boot_ci: int = 2000,
    boot_seed: int = 20261009,
    wall_time_s: float = 0.0,
    git_commit: str = "unknown",
    git_dirty: bool = False,
    generated_by: str = "scripts/experiments/h2_real_model.py",
) -> dict[str, object]:
    """Assemble the machine-readable H2 real-model document from the measured per-cell records.

    Args:
        per_cell: ``target -> per-cell records``. A target is a damage definition (for example the
            layer's own output error and the end-to-end final-hidden-state change); each target gets
            its own aggregate and contrasts, so the two are never mixed inside one claim.
        target_notes: optional ``target -> human-readable description``.

    The document carries the measured evidence, the model-level aggregate per proxy (pooled
    Fisher-``z`` over cells), the paired candidate-vs-comparator contrasts with the analytic,
    model-bootstrap and cell-bootstrap intervals, the achieved MDE, and the explicit statement of the
    measurement class, the substrate and the structural limitations.
    """
    notes = dict(target_notes or {})
    targets: dict[str, object] = {}
    n_modules = 0
    n_cells = 0
    for target, cells in per_cell.items():
        cells = list(cells)
        n_cells += len(cells)
        n_modules = max(n_modules, int(cells[0]["n_modules"]) if cells else 0)  # type: ignore[index]
        variants_table: dict[str, dict[str, object]] = {}
        for proxy in proxies:
            usable = [u for cell in cells if (u := _usable(cell, proxy)) is not None]
            rhos = [
                value
                for cell in cells
                if (value := _finite(cell["variants"][proxy]["rho"])) is not None  # type: ignore[index]
            ]
            neffs = [
                value
                for cell in cells
                if (value := _finite(cell["variants"][proxy]["n_eff"])) is not None  # type: ignore[index]
            ]
            pooled = (
                cluster_contrast(
                    {str(model.get("id", "model")): usable},
                    label=proxy,
                    n_boot=n_boot_ci,
                    seed=boot_seed,
                )
                if usable
                else None
            )
            variants_table[proxy] = {
                "rho_mean": _round(sum(rhos) / len(rhos)) if rhos else None,
                "rho_min": _round(min(rhos)) if rhos else None,
                "rho_max": _round(max(rhos)) if rhos else None,
                "n_eff_mean": _round(sum(neffs) / len(neffs)) if neffs else None,
                "pooled_z": _round(pooled.delta_z) if pooled else None,
                "ci": [_round(pooled.ci_low), _round(pooled.ci_high)] if pooled else None,
                "k": pooled.k if pooled else 0,
                "tau2": _round(pooled.tau2) if pooled else None,
                "i2": _round(pooled.i2) if pooled else None,
            }

        contrasts: dict[str, dict[str, object]] = {}
        for cand in CANDIDATE_PROXIES:
            entry: dict[str, object] = {}
            for comp in (*comparators, naive_baseline):
                if comp == cand:
                    continue
                entry[comp] = contrast_payload(
                    cells,
                    cand,
                    comp,
                    model_id=str(model.get("id", "model")),
                    n_boot_ci=n_boot_ci,
                    boot_seed=boot_seed,
                    n_eff_ref=variants_table[cand]["n_eff_mean"],  # type: ignore[index]
                )
            contrasts[cand] = entry

        primary = contrasts.get(candidate, {}).get(comparators[0], {}) if comparators else {}
        targets[target] = {
            "description": notes.get(target, ""),
            "per_cell": cells,
            "aggregate": {
                "variants": variants_table,
                "contrasts": contrasts,
                "primary_candidate": candidate,
                "primary_comparator": comparators[0] if comparators else None,
                "primary_verdict": primary.get("verdict") if isinstance(primary, Mapping) else None,
                "mde": mde_block(
                    _finite(variants_table[candidate]["n_eff_mean"])
                    if candidate in variants_table
                    else None,
                    1,
                ),
            },
        }

    return {
        "schema": SCHEMA_ID,
        "generated_by": generated_by,
        "substrate": SUBSTRATE,
        "measurement_class": MEASUREMENT_CLASS,
        "statistical_unit": STATISTICAL_UNIT,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "float_precision_significant_digits": FLOAT_SIGNIFICANT_DIGITS,
        "model": dict(model),
        "calibration": dict(calibration),
        "runtime": {
            "wall_time_s": _round(wall_time_s),
            "n_models": 1,
            "n_cells": n_cells,
            "n_modules": n_modules,
            "n_variants": len(variants),
        },
        "plan": {
            "variants": [
                {
                    "name": v.name,
                    "kind": v.kind,
                    "rank": v.rank,
                    "bits": v.bits,
                    "group_size": v.group_size,
                    "description": v.description,
                }
                for v in variants
            ],
            "proxies": list(proxies),
            "candidate_proxy": candidate,
            "candidate_proxies": list(CANDIDATE_PROXIES),
            "naive_baseline": naive_baseline,
            "predeclared_comparators": list(comparators),
            "group_size": group_size,
            "n_boot_neff": n_boot_neff,
            "n_boot_ci": n_boot_ci,
            "boot_seed": boot_seed,
        },
        "predeclared_mde": {
            "n_eff_optimistic": 32,
            "n_eff_conservative": 8,
            "seeds": 5,
            "mde_z_optimistic": 0.33,
            "mde_z_conservative": 0.79,
            "source": "preregistration.md section 7.5 (frozen 2026-10-09)",
        },
        "targets": targets,
        "limitations": [
            "measurement class 2 (fake quantization, CPU float execution): not packed storage and "
            "not a kernel-backed measurement; no latency is claimed",
            "one model only (the pre-registered unit is the model; with n_models = 1 the "
            "model-cluster bootstrap is degenerate and a cell-level bootstrap is reported beside it)",
            "`layer_output` damage is the layer's own output error on one calibration batch (the "
            "frozen proxy unit); `model_output` damage is the end-to-end change of the final hidden "
            "state when that layer alone is compressed (the Tier-0 fixture's target)",
            f"`{naive_baseline}` is the exact plug-in estimator of the `layer_output` damage and is "
            "therefore tautological (rho = 1) on that target; it is a sanity anchor, not a "
            "competitor",
            "proxies whose context (following normalisation, measured residual error, estimated "
            "downstream gain) is absent fall back to their documented context-free form; the "
            "per-cell `fallback_fraction` records it",
            "the variant grid is a small predeclared subset of the frozen grid (ranks {8, 16, 32}, "
            "bits {2, 4, 8}); the rank-dominant cells use 8-bit factors, the near-lossless setting "
            "of the frozen grid, so the truncation dominates their damage",
        ],
    }


def _finite(value: object) -> float | None:
    """``float(value)`` when it is a finite number, else ``None`` (JSON ``null``)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _round(value: float | None, digits: int = FLOAT_SIGNIFICANT_DIGITS) -> float | None:
    """JSON-safe float rounded to ``digits`` significant digits; non-finite becomes ``None``."""
    if value is None:
        return None
    v = float(value)
    if not math.isfinite(v):
        return None
    return float(f"{v:.{digits}g}")
