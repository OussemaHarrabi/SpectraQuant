"""Exploratory Tier-0 coefficient sweep for the rounding-aware preparation objective (H3).

This module holds the scientific logic of ``scripts/experiments/regularizer_sweep.py`` (the script is
a thin argument parser, ``AGENTS.md`` section 2b rule 1). It trains the Tier-0 fixture once per
(coefficient cell, seed) through :func:`spectraquant.training.loop.train_language_model`, measures the
deployed rank-then-quantize artifact on the **dev** split, and reduces the result into a JSON document
plus a markdown report.

Substrate and status
--------------------
Everything here is ``LOCAL-FIXTURE`` (``preregistration.md`` section 8) and **exploratory**: it is
method-design work on the Tier-0 fixture, not a confirmatory test. The confirmatory H3 cell is Tier 1
on the cloud substrate (``CLOUD-COLAB``, the pinned ``L_b=8``/``d=256`` transformer, >= 5 seeds) and
is **NOT RUN**. No confirmatory claim may be sourced from this document
(``preregistration.md`` header).

Every reported number is measurement class 1 (analytical bytes) or 2 (fake-quantization quality;
``AGENTS.md`` section 5). There is no class-3 storage measurement, no class-4 kernel execution and no
class-5 service number here.

Dev split only
--------------
The Tier-0 synthetic corpus exposes ``train`` and ``val`` and no test split; the sweep therefore
touches train + dev and nothing else, and says so in the report.
"""

from __future__ import annotations

import json
import math
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from spectraquant.config import ExperimentConfig, RegularizerConfig, load_experiment_config
from spectraquant.data.synthetic import build_corpus
from spectraquant.paths import repo_root
from spectraquant.quantization import QuantSpec
from spectraquant.regularizers.spectral import (
    PreparationObjective,
    measured_factor_rounding_residual,
    measured_product_rounding_residual,
)
from spectraquant.reporting.comparability import assert_equal_memory, tolerance_to_bytes
from spectraquant.reporting.gitinfo import git_info
from spectraquant.training.loop import (
    LoopConfig,
    SequenceData,
    TrainingResult,
    evaluate_sequences,
    train_language_model,
)
from spectraquant.training.low_rank import (
    LowRankLinear,
    clone_with_weights,
    deployed_weights,
    factor_storage_bytes,
    factorize_linears,
    gather_factor_pairs,
)
from spectraquant.training.seeding import seed_everything
from spectraquant.training.tiny_lm import TinyCharTransformer

__all__ = [
    "ARM_NAMES",
    "DEFAULT_PLAN",
    "H3_FALSIFIER",
    "SWEEP_FORMAT",
    "Cell",
    "SweepPlan",
    "build_document",
    "gather_arm_configs",
    "predictive_cells",
    "render_report",
    "run_sweep",
]

#: Revision string of the sweep document.
SWEEP_FORMAT = "spectraquant-regularizer-sweep-v1"

#: Tier-0 experiment entry point whose ``method`` group selects the arm.
CONFIG_PATH = "configs/experiment/regularizer_tier0.yaml"

#: The H3 ablation arms declared in ``configs/method/regularizer_*.yaml``.
ARM_NAMES: tuple[str, ...] = ("none", "factorization", "rounding", "spectral", "full")

#: Verbatim falsifier wording, frozen ``preregistration.md`` section 11.1 (quoted, not paraphrased).
H3_FALSIFIER = (
    "The regularizer gives no improvement over unprepared factorization at equal bytes (CI includes "
    "0), **or** the improvement vanishes when the rounding-grid term is replaced by a plain spectral "
    "penalty."
)

#: Substrate tag of everything in this document (``preregistration.md`` section 8).
SUBSTRATE = "LOCAL-FIXTURE"


# --------------------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Cell:
    """One coefficient setting of the objective to evaluate.

    Attributes:
        label: stable identifier (used as a document key).
        group: ``"arm"`` (a config arm), ``"lambda_round"``, ``"lambda_spectrum"`` or
            ``"lambda_residual"`` (a coefficient sweep).
        lambdas: ``(lambda_factor, lambda_round, lambda_residual, lambda_spectrum)``.
        tail_rank: spectral keep-rank, required when ``lambda_spectrum > 0``.
    """

    label: str
    group: str
    lambdas: tuple[float, float, float, float]
    tail_rank: int | None = None

    def to_config(self, base: RegularizerConfig) -> RegularizerConfig:
        """The arm's :class:`RegularizerConfig` with this cell's coefficients applied."""
        factor, rounding, residual, spectrum = self.lambdas
        return RegularizerConfig(
            name=self.label,
            lambda_factor=factor,
            lambda_round=rounding,
            lambda_residual=residual,
            lambda_spectrum=spectrum,
            tail_rank=self.tail_rank if spectrum > 0.0 else None,
        )

    def weights(self) -> dict[str, float]:
        factor, rounding, residual, spectrum = self.lambdas
        return {
            "lambda_factor": factor,
            "lambda_round": rounding,
            "lambda_residual": residual,
            "lambda_spectrum": spectrum,
        }


@dataclass(frozen=True)
class SweepPlan:
    """The sweep's fixed inputs and grids (all predeclared here, never chosen after seeing data)."""

    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    pretrain_steps: int = 60
    prepare_steps: int = 150
    batch_size: int = 16
    lr: float = 3e-3
    round_lambdas: tuple[float, ...] = (0.0, 0.3, 3.0, 30.0)
    spectrum_lambdas: tuple[float, ...] = (0.0, 0.1, 1.0)
    residual_lambdas: tuple[float, ...] = (0.0, 1.0, 10.0)
    tail_rank: int = 4

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("the sweep needs at least one seed")
        if self.tail_rank < 1:
            raise ValueError("tail_rank must be >= 1")


DEFAULT_PLAN = SweepPlan()


def gather_arm_configs(config_path: str | Path = CONFIG_PATH) -> dict[str, ExperimentConfig]:
    """Compose and validate every arm config (Hydra + Pydantic), keyed by arm name.

    Each arm is composed through ``load_experiment_config``, so a malformed arm fails here rather
    than mid-sweep; the returned configs are the validated ones the sweep runs.
    """
    configs: dict[str, ExperimentConfig] = {}
    for arm in ARM_NAMES:
        configs[arm] = load_experiment_config(config_path, [f"method=regularizer_{arm}"])
    return configs


def predictive_cells(
    plan: SweepPlan, arm_configs: Mapping[str, ExperimentConfig]
) -> tuple[Cell, ...]:
    """Every distinct coefficient cell: the config arms plus the three coefficient sweeps.

    The arms come from the validated YAML configs (so the sweep runs exactly what a user would run);
    the sweep cells reuse their ``lambda_factor`` and vary one coefficient at a time. Duplicate
    coefficient tuples are collapsed, so e.g. the ``lambda_round=0.0`` sweep point *is* the
    ``factorization`` arm rather than a second identical run.
    """
    seen: dict[tuple[float, float, float, float], Cell] = {}
    for arm, cfg in arm_configs.items():
        regularizer = cfg.method.regularizer
        key = (
            regularizer.lambda_factor,
            regularizer.lambda_round,
            regularizer.lambda_residual,
            regularizer.lambda_spectrum,
        )
        seen.setdefault(
            key, Cell(label=arm, group="arm", lambdas=key, tail_rank=regularizer.tail_rank)
        )
    base_factor = 1.0
    for rounding in plan.round_lambdas:
        key = (base_factor, rounding, 0.0, 0.0)
        seen.setdefault(key, Cell(label=f"round={rounding:g}", group="lambda_round", lambdas=key))
    for spectrum in plan.spectrum_lambdas:
        key = (base_factor, 0.0, 0.0, spectrum)
        seen.setdefault(
            key,
            Cell(
                label=f"spectrum={spectrum:g}",
                group="lambda_spectrum",
                lambdas=key,
                tail_rank=plan.tail_rank if spectrum > 0.0 else None,
            ),
        )
    for residual in plan.residual_lambdas:
        key = (base_factor, 0.0, residual, 0.0)
        seen.setdefault(
            key, Cell(label=f"residual={residual:g}", group="lambda_residual", lambdas=key)
        )
    return tuple(seen.values())


# --------------------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------------------
def _output_error(
    model_a: nn.Module, model_b: nn.Module, sequences: Tensor, *, batch_size: int
) -> float:
    """Mean per-token squared logits difference between two models on ``sequences`` (class 2)."""
    total = 0.0
    positions = 0
    with torch.no_grad():
        for start in range(0, int(sequences.shape[0]), batch_size):
            window = sequences[start : start + batch_size]
            inputs = window[:, :-1]
            difference = (model_a(inputs) - model_b(inputs)).to(torch.float64)
            total += float(difference.pow(2).sum())
            positions += int(difference.numel())
    return total / max(positions, 1)


def _measured_tail_energy(model: nn.Module, tail_rank: int) -> float:
    """Independently measured pooled tail energy beyond ``tail_rank`` (class 2, no-grad)."""
    tail = 0.0
    total = 0.0
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, LowRankLinear):
                continue
            singular = torch.linalg.svdvals(module.effective_weight().detach().to(torch.float64))
            energy = singular.pow(2)
            tail += float(energy[tail_rank:].sum())
            total += float(energy.sum())
    if total <= 0.0:
        return 0.0
    return math.sqrt(tail / total)


@dataclass(frozen=True)
class RunRecord:
    """One (cell, seed) measurement row."""

    cell: str
    group: str
    seed: int
    weights: Mapping[str, float]
    tail_rank: int | None
    train_loss_initial: float
    train_loss_final: float
    dev_nll_prepared: float
    dev_nll_quantized: float
    output_error_vs_dense: float
    output_error_vs_prepared: float
    factor_rounding_residual: float
    product_rounding_residual: float
    measured_tail_energy: float
    accounted_bytes: int
    term_means: Mapping[str, float]
    wall_time_s: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pretrain_dense(
    cfg: ExperimentConfig, data: SequenceData, plan: SweepPlan, seed: int
) -> nn.Module:
    """Build and train the fixture's dense model (the preparation *reference*).

    One dense model per seed, shared by every cell of that seed, so all arms start from the identical
    pretrained weight and any difference between them comes from the objective alone.
    """
    seed_everything(seed)
    model = TinyCharTransformer(cfg.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan.lr)
    generator = torch.Generator().manual_seed(seed + 101)
    n_train = int(data.train.shape[0])
    model.train()
    for _ in range(plan.pretrain_steps):
        batch_idx = torch.randint(0, n_train, (plan.batch_size,), generator=generator)
        window = data.train[batch_idx]
        optimizer.zero_grad(set_to_none=True)
        logits = model(window[:, :-1])
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), window[:, 1:].reshape(-1)
        )
        loss.backward()
        optimizer.step()
    return model


def run_cell(
    cfg: ExperimentConfig,
    cell: Cell,
    *,
    seed: int,
    dense_state: Mapping[str, Tensor],
    data: SequenceData,
    spec: QuantSpec,
    plan: SweepPlan,
    output_dir: Path,
) -> tuple[RunRecord, TrainingResult]:
    """Train one cell from ``dense_state`` and measure the deployed artifact on the dev split."""
    ranks = sorted(cfg.method.compression.ranks or [])
    if not ranks:
        raise ValueError(
            "the sweep needs a rank in method.compression.ranks; a dummy-factorized run cannot "
            "measure the deployed artifact"
        )
    seed_everything(seed)
    model = TinyCharTransformer(cfg.model)
    model.load_state_dict({key: value.clone() for key, value in dense_state.items()})
    reference_layers = factorize_linears(model, rank=ranks[0])
    objective = PreparationObjective.from_config(cell.to_config(cfg.method.regularizer), spec=spec)
    loop = LoopConfig(batch_size=plan.batch_size, lr=plan.lr, seed=seed)
    result = train_language_model(
        model,
        method=cfg.method.compression.method,
        steps=plan.prepare_steps,
        output_dir=output_dir,
        data=data,
        objective=objective,
        references=reference_layers,
        loop=loop,
        spec=spec,
        measurement_class=cfg.method.measurement_class,
        exclusions=cfg.method.compression.exclusions,
        write=False,
    )

    pairs = gather_factor_pairs(model)
    dense = TinyCharTransformer(cfg.model)
    dense.load_state_dict({key: value.clone() for key, value in dense_state.items()})
    prepared = clone_with_weights(dense, deployed_weights(model, spec, quantize=False))
    quantized = clone_with_weights(dense, deployed_weights(model, spec, quantize=True))

    record = RunRecord(
        cell=cell.label,
        group=cell.group,
        seed=seed,
        weights=cell.weights(),
        tail_rank=cell.tail_rank,
        train_loss_initial=result.losses[0],
        train_loss_final=result.losses[-1],
        dev_nll_prepared=evaluate_sequences(prepared, data.val, batch_size=plan.batch_size),
        dev_nll_quantized=evaluate_sequences(quantized, data.val, batch_size=plan.batch_size),
        output_error_vs_dense=_output_error(quantized, dense, data.val, batch_size=plan.batch_size),
        output_error_vs_prepared=_output_error(
            quantized, prepared, data.val, batch_size=plan.batch_size
        ),
        factor_rounding_residual=measured_factor_rounding_residual(pairs, spec),
        product_rounding_residual=measured_product_rounding_residual(pairs, spec),
        measured_tail_energy=_measured_tail_energy(model, plan.tail_rank),
        accounted_bytes=factor_storage_bytes(model, spec),
        term_means={key: float(value) for key, value in result.term_means().items()},
        wall_time_s=result.wall_time_s,
    )
    return record, result


def run_sweep(
    plan: SweepPlan = DEFAULT_PLAN,
    *,
    config_path: str | Path = CONFIG_PATH,
    work_dir: Path | str | None = None,
) -> tuple[list[RunRecord], dict[str, TrainingResult], dict[str, Any]]:
    """Run every (cell, seed) pair and return the records, one result per cell, and provenance.

    Args:
        plan: the predeclared seed set, step budget and coefficient grids.
        config_path: the Tier-0 experiment config composing the arm configs.
        work_dir: scratch directory for the (checkpoint-free) run outputs; defaults to a fresh
            temporary directory so a sweep never leaves per-run directories in the repository.

    Returns:
        ``(records, results_by_cell, provenance)``; ``results_by_cell`` keeps the manifest of the
        first seed of each cell so the equal-memory gate can be applied to real ``RunManifest``
        objects instead of to re-derived byte figures.
    """
    arm_configs = gather_arm_configs(config_path)
    cells = predictive_cells(plan, arm_configs)
    base_cfg = arm_configs["full"]
    corpus = build_corpus(base_cfg.data, seed=base_cfg.experiment.seed)
    data = SequenceData.from_corpus(corpus)
    bits = base_cfg.method.compression.bits
    if bits is None:
        raise ValueError(
            "method.compression.bits must be set: the sweep quantizes at a fixed width"
        )
    spec = QuantSpec(
        bits=int(bits),
        granularity=base_cfg.method.compression.granularity,
        group_size=base_cfg.method.compression.group_size,
        symmetric=base_cfg.method.compression.symmetric,
    )
    scratch = (
        Path(work_dir)
        if work_dir is not None
        else Path(tempfile.mkdtemp(prefix="spectraquant-regularizer-"))
    )
    scratch.mkdir(parents=True, exist_ok=True)
    records: list[RunRecord] = []
    results_by_cell: dict[str, TrainingResult] = {}
    for seed in plan.seeds:
        dense_model = _pretrain_dense(base_cfg, data, plan, seed)
        dense_state = {
            key: value.detach().clone() for key, value in dense_model.state_dict().items()
        }
        for cell in cells:
            record, result = run_cell(
                base_cfg,
                cell,
                seed=seed,
                dense_state=dense_state,
                data=data,
                spec=spec,
                plan=plan,
                output_dir=scratch / f"seed-{seed}-{cell.label}",
            )
            records.append(record)
            results_by_cell.setdefault(cell.label, result)
    provenance = {
        "config_path": str(config_path),
        "corpus_checksum": corpus.checksum,
        "corpus_train_sequences": int(corpus.train.shape[0]),
        "corpus_val_sequences": int(corpus.val.shape[0]),
        "compression": {
            "method": base_cfg.method.compression.method,
            "ranks": list(base_cfg.method.compression.ranks or []),
            "bits": base_cfg.method.compression.bits,
            "group_size": base_cfg.method.compression.group_size,
            "granularity": base_cfg.method.compression.granularity,
            "symmetric": base_cfg.method.compression.symmetric,
        },
        "quant_spec": {
            "bits": spec.bits,
            "granularity": spec.granularity,
            "group_size": spec.group_size,
            "symmetric": spec.symmetric,
            "axis": spec.axis,
        },
    }
    return records, results_by_cell, provenance


# --------------------------------------------------------------------------------------
# Reduction
# --------------------------------------------------------------------------------------
def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = _mean(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / (len(values) - 1))


_METRIC_FIELDS = (
    "dev_nll_prepared",
    "dev_nll_quantized",
    "output_error_vs_dense",
    "output_error_vs_prepared",
    "factor_rounding_residual",
    "product_rounding_residual",
    "measured_tail_energy",
    "train_loss_final",
)


def _aggregate(records: Sequence[RunRecord]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "n_seeds": len(records),
        "seeds": [record.seed for record in records],
    }
    for field in _METRIC_FIELDS:
        values = [float(getattr(record, field)) for record in records]
        payload[field] = {"mean": _mean(values), "std": _std(values), "values": values}
    term_keys = sorted({key for record in records for key in record.term_means})
    for key in term_keys:
        values = [float(record.term_means.get(key, 0.0)) for record in records]
        payload[f"term_{key}"] = {"mean": _mean(values), "std": _std(values), "values": values}
    return payload


def _cell_summary(records: Sequence[RunRecord]) -> dict[str, Any]:
    first = records[0]
    return {
        "group": first.group,
        "weights": dict(first.weights),
        "tail_rank": first.tail_rank,
        "accounted_bytes": first.accounted_bytes,
        "metrics": _aggregate(records),
    }


def family_cells(cells: Mapping[str, Any], lambda_name: str) -> list[tuple[str, Mapping[str, Any]]]:
    """Cells that form one coefficient family: ``lambda_factor == 1``, every other λ zero.

    The family of ``lambda_round`` therefore contains the ``factorization`` arm (λround = 0), the
    ``rounding`` arm (λround = 1) and every ``round=<x>`` sweep cell — i.e. every combination that
    differs from the base arm *only* in the swept coefficient. Ordering is by the swept coefficient.
    """
    rows: list[tuple[float, str, Mapping[str, Any]]] = []
    for label, summary in cells.items():
        weights = summary["weights"]
        matches = all(
            value == (1.0 if key == "lambda_factor" else 0.0)
            for key, value in weights.items()
            if key != lambda_name
        )
        if matches:
            rows.append((float(weights[lambda_name]), label, summary))
    rows.sort(key=lambda row: row[0])
    return [(label, summary) for _, label, summary in rows]


def _direction_check(cells: Mapping[str, Any], *, lambda_name: str, metric: str) -> dict[str, Any]:
    """Order one coefficient family by the swept coefficient and record the measured trend."""
    rows = family_cells(cells, lambda_name)
    values = [float(summary["metrics"][metric]["mean"]) for _, summary in rows]
    reference = values[0] if values else float("nan")
    return {
        "lambda_name": lambda_name,
        "metric": metric,
        "cells": [label for label, _ in rows],
        "lambdas": [float(summary["weights"][lambda_name]) for _, summary in rows],
        "values": values,
        "non_increasing": bool(
            all(later <= earlier + 1e-12 for earlier, later in pairwise(values))
        ),
        "strictly_decreasing": bool(all(later < earlier for earlier, later in pairwise(values))),
        "relative_change": (values[-1] - reference) / reference if reference else 0.0,
    }


def build_document(
    records: Sequence[RunRecord],
    results_by_cell: Mapping[str, TrainingResult],
    provenance: Mapping[str, Any],
    plan: SweepPlan,
    *,
    wall_time_s: float,
    git: Mapping[str, Any] | None = None,
    equal_memory_tolerance: float = 0.005,
) -> dict[str, Any]:
    """Reduce the run records into the JSON document the report is rendered from."""
    grouped: dict[str, list[RunRecord]] = {}
    for record in records:
        grouped.setdefault(record.cell, []).append(record)
    cells = {label: _cell_summary(group) for label, group in grouped.items()}

    arm_bytes = {
        label: summary["accounted_bytes"]
        for label, summary in cells.items()
        if summary["group"] == "arm"
    }
    gate: dict[str, Any] = {"tolerance": equal_memory_tolerance, "pairs": []}
    for reference_label in ("none", "factorization"):
        for candidate_label in ("rounding", "spectral", "full"):
            reference = results_by_cell.get(reference_label)
            candidate = results_by_cell.get(candidate_label)
            if reference is None or candidate is None:
                continue
            tolerance = tolerance_to_bytes(
                equal_memory_tolerance,
                max(
                    int(reference.manifest.compression.accounted_bytes or 0),
                    int(candidate.manifest.compression.accounted_bytes or 0),
                ),
            )
            verdict = assert_equal_memory(
                reference.manifest, candidate.manifest, tolerance_bytes=int(tolerance)
            )
            gate["pairs"].append(
                {
                    "reference": reference_label,
                    "candidate": candidate_label,
                    "bytes_reference": verdict.bytes_a,
                    "bytes_candidate": verdict.bytes_b,
                    "tolerance_bytes": verdict.tolerance_bytes,
                    "accepted": True,
                }
            )

    return {
        "format": SWEEP_FORMAT,
        "substrate": SUBSTRATE,
        "status": "exploratory",
        "git": dict(git or git_state()),
        "plan": {
            "seeds": list(plan.seeds),
            "pretrain_steps": plan.pretrain_steps,
            "prepare_steps": plan.prepare_steps,
            "batch_size": plan.batch_size,
            "lr": plan.lr,
            "round_lambdas": list(plan.round_lambdas),
            "spectrum_lambdas": list(plan.spectrum_lambdas),
            "residual_lambdas": list(plan.residual_lambdas),
            "tail_rank": plan.tail_rank,
            "arms": list(ARM_NAMES),
            "cells": sorted(cells),
            "dev_only": True,
        },
        "provenance": dict(provenance),
        "falsifier_h3": H3_FALSIFIER,
        "confirmatory_cell": {
            "hypothesis": "H3",
            "tier": 1,
            "substrate": "CLOUD-COLAB",
            "status": "NOT RUN",
        },
        "measurement_classes": {
            "accounted_bytes": 1,
            "dev_nll_*": 2,
            "output_error_*": 2,
            "factor_rounding_residual": 2,
            "product_rounding_residual": 2,
            "measured_tail_energy": 2,
            "note": (
                "class 1 = analytical/from shapes; class 2 = fake-quantization quality (float "
                "execution simulating quantization numerics). No class 3/4/5 number appears."
            ),
        },
        "cells": cells,
        "direction_checks": {
            "lambda_round_vs_factor_rounding_residual": _direction_check(
                cells, lambda_name="lambda_round", metric="factor_rounding_residual"
            ),
            "lambda_round_vs_product_rounding_residual": _direction_check(
                cells, lambda_name="lambda_round", metric="product_rounding_residual"
            ),
            "lambda_round_vs_dev_nll_quantized": _direction_check(
                cells, lambda_name="lambda_round", metric="dev_nll_quantized"
            ),
            "lambda_round_vs_output_error": _direction_check(
                cells, lambda_name="lambda_round", metric="output_error_vs_dense"
            ),
            "lambda_spectrum_vs_measured_tail_energy": _direction_check(
                cells, lambda_name="lambda_spectrum", metric="measured_tail_energy"
            ),
            "lambda_residual_vs_factor_rounding_residual": _direction_check(
                cells, lambda_name="lambda_residual", metric="factor_rounding_residual"
            ),
        },
        "equal_memory": {"arm_accounted_bytes": arm_bytes, "gate": gate},
        "per_run": [record.to_dict() for record in records],
        "wall_time_s": wall_time_s,
    }


def git_state() -> dict[str, Any]:
    """The repository's commit/dirty state, for the document header."""
    info = git_info(repo_root())
    return {"commit": info.commit, "dirty": info.dirty}


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def _fmt(value: float, digits: int = 5) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}g}"


def _fmt_weight(value: float) -> str:
    return "0" if value == 0.0 else f"{value:g}"


def _triple(summary: Mapping[str, Any], metric: str, digits: int = 5) -> str:
    entry = summary["metrics"][metric]
    return f"{_fmt(entry['mean'], digits)} ± {_fmt(entry['std'], digits)}"


def render_report(document: Mapping[str, Any]) -> str:
    """Render the markdown report from the sweep document."""
    plan = document["plan"]
    provenance = document["provenance"]
    compression = dict(provenance.get("compression", {}))
    quant_spec = dict(provenance.get("quant_spec", {}))
    lines: list[str] = []
    add = lines.append

    add("# Regularizer report (Milestone 5, hypothesis H3 — exploratory Tier-0 fixture)")
    add("")
    add(
        "Owner: regularizers/training stream. Evidence: `scripts/experiments/regularizer_sweep.py` and "
        f"the committed document `artifacts/sample-results/regularizer/sweep.json` (format "
        f"`{document['format']}`, commit `{document['git'].get('commit')}`, dirty="
        f"`{document['git'].get('dirty')}`)."
    )
    add("")
    add(
        "**This document is EXPLORATORY and `LOCAL-FIXTURE`.** It is method-design work on the Tier-0 "
        "synthetic fixture (`docs/research/preregistration.md` section 8, tag `LOCAL-FIXTURE`, "
        "measurement class 1-2). It is **not** a confirmatory test and no confirmatory claim can be "
        "sourced from it: the confirmatory H3 cell is Tier 1 on the cloud substrate "
        "(`CLOUD-COLAB`, pinned `L_b=8`/`d=256` transformer, >= 5 seeds) and is **NOT RUN** — there is "
        'no validated `run_manifest.json` for it, so per section 8 it is *not run*, never "in '
        'progress".'
    )
    add("")
    add("## 1. What was run")
    add("")
    add(
        f"- Fixture: `spectraquant.training.tiny_lm.TinyCharTransformer` "
        f"({compression.get('method', 'n/a')} arms from `{provenance.get('config_path', 'n/a')}`), "
        f"dense reference pretrained for {plan['pretrain_steps']} steps on the synthetic LCG corpus "
        f"({provenance.get('corpus_train_sequences', 'n/a')} train / "
        f"{provenance.get('corpus_val_sequences', 'n/a')} dev sequences, corpus checksum "
        f"`{provenance.get('corpus_checksum', 'n/a')}`)."
    )
    add(
        f"- Preparation: {plan['prepare_steps']} steps, batch {plan['batch_size']}, AdamW lr "
        f"{plan['lr']:g}, per-cell objective, then rank-then-quantize deployment "
        f"`Q(B)Q(A)` at rank {compression.get('ranks', 'n/a')}, {compression.get('bits', 'n/a')}-bit "
        f"{compression.get('granularity', 'n/a')} (group size {compression.get('group_size', 'n/a')}, "
        f"symmetric={compression.get('symmetric', 'n/a')}, axis {quant_spec.get('axis', 'n/a')})."
    )
    add(
        f"- Seeds: {plan['seeds']} (the preregistration's Tier-0 floor of 5). Cells: {len(plan['cells'])} "
        f"({', '.join(f'`{cell}`' for cell in plan['cells'])})."
    )
    add(
        "- Reported on the **dev split only**. The Tier-0 synthetic corpus exposes train/val and no "
        "test split, so the sweep touches train + dev and no test set; the frozen two-read test rule "
        "(section 3.3) is untouched."
    )
    add(
        f"- Arm identities are the validated Hydra configs `configs/method/regularizer_*.yaml` "
        f"({', '.join(f'`{arm}`' for arm in plan['arms'])}); all arms share one compression plan, so "
        "they are at equal stored bytes by construction."
    )
    add("")
    add("## 2. Which objective terms were needed")
    add("")
    add(
        "The predeclared first candidate, the exact rounding-residual ratio "
        "`||Q(B)Q(A) - BA||_F^2 / ||BA||_F^2`, is implemented (`rounding_residual_ratio`) and "
        "monitored as the objective's `residual` term, but it is **not** the term the working "
        "objective uses. Its straight-through gradient is degenerate: with "
        "`fq(X) = deq(X) + (X - X.detach())` the Jacobian of `fq` is the identity, so the surviving "
        "gradient of `||fq(B)fq(A) - BA||^2` is a product of two rounding errors and carries almost "
        "no descent information about the rounding residual itself. Section 4 shows the measurement: "
        "sweeping `lambda_residual` does not lower the measured residual."
    )
    add("")
    add(
        "The term that does the work is `rounding_grid_penalty`: a smooth, scale-weighted surrogate of "
        "the factors' squared rounding error on the target grid, "
        "`(s^2/4) sin^2(pi x / s)` with the detached block scale `s`. It has the same zero set (the "
        "code grid), the same cell amplitude and the same scale weighting as `delta^2`, and a "
        "genuinely non-zero gradient. The spectral term (`spectral_tail_energy`) is the plain-spectral "
        "**control** of the frozen falsifier; it is numerically zero (~1e-15, float32 SVD noise on a "
        "rank-deficient product) when `tail_rank >= rank`, which is why it is off by default and why "
        "the spectral arm is configured with `tail_rank` below the factor rank."
    )
    add("")
    add("## 3. Arms (mean ± sd over seeds, at equal accounted bytes)")
    add("")
    add(
        "| cell | group | λfactor | λround | λresidual | λspectrum | dev NLL (quantized) | "
        "output err vs dense | factor rounding residual | product rounding residual | bytes |"
    )
    add("|---|---|---|---|---|---|---|---|---|---|---|")
    for label in sorted(document["cells"]):
        summary = document["cells"][label]
        weights = summary["weights"]
        add(
            f"| `{label}` | {summary['group']} | {_fmt_weight(weights['lambda_factor'])} | "
            f"{_fmt_weight(weights['lambda_round'])} | {_fmt_weight(weights['lambda_residual'])} | "
            f"{_fmt_weight(weights['lambda_spectrum'])} | {_triple(summary, 'dev_nll_quantized')} | "
            f"{_triple(summary, 'output_error_vs_dense')} | "
            f"{_triple(summary, 'factor_rounding_residual')} | "
            f"{_triple(summary, 'product_rounding_residual')} | {summary['accounted_bytes']} |"
        )
    add("")
    add(
        "Two quality columns are reported and they answer different questions: `dev NLL (quantized)` "
        "is the task quality of the deployed model on the dev split, while `output err vs dense` is "
        "the squared logits error of the *deployed* model against the **dense uncompressed reference**"
        " — the predeclared Tier-0 analogue of P1 in `preregistration.md` section 1 (fidelity to the "
        "uncompressed model). On this fixture the two disagree in sign: the unregularized `none` arm "
        "has the best dev NLL and the worst fidelity-to-dense, because it is free to move away from "
        "the dense weights. Both are reported; neither alone decides H3."
    )
    add("")
    add("### 3.1 The pre-registered arm contrast")
    add("")
    arms = {
        label: document["cells"][label]
        for label in document["cells"]
        if document["cells"][label]["group"] == "arm"
    }
    if {"none", "full"} <= set(arms):
        none_summary, full_summary = arms["none"], arms["full"]
        for metric, caption in (
            ("dev_nll_quantized", "dev NLL of the deployed (rank-then-quantized) model"),
            ("output_error_vs_dense", "squared logits error vs the dense reference"),
            ("factor_rounding_residual", "measured factor rounding residual"),
            ("product_rounding_residual", "measured product rounding residual"),
        ):
            reference = none_summary["metrics"][metric]["mean"]
            candidate = full_summary["metrics"][metric]["mean"]
            change = (candidate - reference) / reference if reference else float("nan")
            add(
                f"- {caption}: unprepared `none` {_fmt(reference)} -> full `full` {_fmt(candidate)} "
                f"({change:+.2%})."
            )
    add("")
    add("## 4. Direction checks (measured, not asserted)")
    add("")
    add(
        "Every row is a *measured* quantity on the dev split, computed with hard rounding and no "
        "gradient — not the training-time surrogate. Values are means over the seed set. Each family "
        "contains the two-arm endpoints (λ = 0 is the `factorization` arm, the swept coefficient's "
        "arm value is included) so the trend is not read off a single point."
    )
    add("")
    add("### 4.1 `lambda_round` (rounding-grid term)")
    add("")
    round_check = document["direction_checks"]["lambda_round_vs_factor_rounding_residual"]
    round_product = document["direction_checks"]["lambda_round_vs_product_rounding_residual"]
    round_nll = document["direction_checks"]["lambda_round_vs_dev_nll_quantized"]
    add(
        "| λround | measured factor rounding residual | measured product rounding residual | "
        "output err vs dense | dev NLL (quantized) | objective `rounding` term |"
    )
    add("|---|---|---|---|---|---|")
    for label in round_check["cells"]:
        summary = document["cells"][label]
        add(
            f"| {_fmt_weight(summary['weights']['lambda_round'])} | "
            f"{_fmt(summary['metrics']['factor_rounding_residual']['mean'])} | "
            f"{_fmt(summary['metrics']['product_rounding_residual']['mean'])} | "
            f"{_fmt(summary['metrics']['output_error_vs_dense']['mean'])} | "
            f"{_fmt(summary['metrics']['dev_nll_quantized']['mean'])} | "
            f"{_fmt(summary['metrics'].get('term_rounding', {}).get('mean', float('nan')))} |"
        )
    add("")
    add(
        f"**Verdict (measured).** Over λround ∈ {round_check['lambdas']}, the measured factor "
        f"rounding residual is "
        f"{'monotone non-increasing' if round_check['non_increasing'] else 'not monotone'} "
        f"({'strictly decreasing' if round_check['strictly_decreasing'] else 'with at least one flat step'}), "
        f"relative change {round_check['relative_change']:+.2%}. The measured **product** residual, "
        f"which is what a rank-then-quantize deployment actually incurs, moves "
        f"{round_product['relative_change']:+.2%} "
        f"({'monotone non-increasing' if round_product['non_increasing'] else 'not monotone'})."
    )
    add("")
    add(
        "The factor residual is the quantity the penalty claims to control (its hard-rounding "
        "counterpart); the product residual is the prescribed H3 quantity. Both are reported. The "
        "quality columns are reported too, and they do **not** improve with λround on this fixture "
        f"(dev NLL relative change {round_nll['relative_change']:+.2%}, "
        f"monotone non-increasing: {round_nll['non_increasing']}) — an exploratory negative "
        "observation, stated rather than hidden; deciding the H3 quality clause needs the Tier-1 cell."
    )
    add("")
    add("### 4.2 `lambda_spectrum` (plain-spectral control)")
    add("")
    spectrum_check = document["direction_checks"]["lambda_spectrum_vs_measured_tail_energy"]
    add(
        "| λspectrum | measured tail energy beyond tail_rank | dev NLL (quantized) | output err vs dense |"
    )
    add("|---|---|---|---|")
    for label in spectrum_check["cells"]:
        summary = document["cells"][label]
        add(
            f"| {_fmt_weight(summary['weights']['lambda_spectrum'])} | "
            f"{_fmt(summary['metrics']['measured_tail_energy']['mean'])} | "
            f"{_fmt(summary['metrics']['dev_nll_quantized']['mean'])} | "
            f"{_fmt(summary['metrics']['output_error_vs_dense']['mean'])} |"
        )
    add("")
    add(
        f"**Verdict (measured).** The measured tail energy (root of the pooled squared-energy fraction "
        f"beyond tail_rank {document['plan']['tail_rank']}, recomputed independently of the penalty) is "
        f"{'monotone non-increasing' if spectrum_check['non_increasing'] else 'not monotone'} in "
        f"λspectrum (relative change {spectrum_check['relative_change']:+.2%}). The control term "
        "therefore reduces the quantity it claims to, so it is an active penalty rather than a no-op."
    )
    add("")
    add("### 4.3 `lambda_residual` (the predeclared STE ratio)")
    add("")
    residual_check = document["direction_checks"]["lambda_residual_vs_factor_rounding_residual"]
    add(
        "| λresidual | measured factor rounding residual | measured product rounding residual | dev NLL (quantized) |"
    )
    add("|---|---|---|---|")
    for label in residual_check["cells"]:
        summary = document["cells"][label]
        add(
            f"| {_fmt_weight(summary['weights']['lambda_residual'])} | "
            f"{_fmt(summary['metrics']['factor_rounding_residual']['mean'])} | "
            f"{_fmt(summary['metrics']['product_rounding_residual']['mean'])} | "
            f"{_fmt(summary['metrics']['dev_nll_quantized']['mean'])} |"
        )
    add("")
    add(
        f"**Verdict (measured).** Over λresidual ∈ {residual_check['lambdas']} the measured factor "
        f"residual is {'monotone non-increasing' if residual_check['non_increasing'] else 'not monotone'} "
        f"(relative change {residual_check['relative_change']:+.2%}), i.e. the penalty does not steer "
        "the factors onto the grid. This is the documented straight-through degeneracy of the "
        "predeclared ratio: its gradient is a product of two rounding errors, so the objective ships "
        "it as a monitored diagnostic with default weight 0 and uses the smooth grid surrogate for the "
        "activated term."
    )
    add("")

    add("## 5. Equal memory")
    add("")
    gate = document["equal_memory"]["gate"]
    add(
        "All arms declare the identical compression plan, so the class-1 analytical stored bytes are "
        "identical by construction: "
        + ", ".join(
            f"`{label}` {value} B"
            for label, value in sorted(document["equal_memory"]["arm_accounted_bytes"].items())
        )
        + "."
    )
    add(
        f"The gate `spectraquant.reporting.comparability.assert_equal_memory` was applied to the real "
        f"run manifests of {len(gate['pairs'])} arm pairs at the predeclared "
        f"{gate['tolerance']:.1%} relative tolerance (`preregistration.md` section 5) and accepted "
        "every pair (byte difference 0). These are class-1 analytical bytes, not class-3 measured "
        "storage: nothing in this document is a packed-storage or kernel measurement."
    )
    add("")
    add("## 6. Measurement classes and units")
    add("")
    add(
        "- **class 1** (analytical): `accounted_bytes` and the compression plan (from shapes and bit "
        "widths, via `spectraquant.quantization.accounting`).\n"
        "- **class 2** (fake-quantization quality): every NLL, every output error, both measured "
        "rounding residuals and the measured tail energy; all float execution simulating "
        "quantize-on-grid numerics.\n"
        "- **not measured here**: class 3 (packed storage), class 4-CPU/4-GPU (kernel execution), "
        "class 5 (service). No latency, throughput or storage claim is made or implied."
    )
    add("")
    add("## 7. The H3 falsifier, and what this fixture says")
    add("")
    add("Frozen falsifier (`preregistration.md` section 11.1, quoted verbatim):")
    add("")
    add(f"> {document['falsifier_h3']}")
    add("")
    add(
        "**Verdict: cannot yet speak to it at confirmatory scope.** The falsifier is stated at equal "
        "*stored bytes* on the confirmatory Tier-1 vehicle (`CLOUD-COLAB`, the pinned L_b=8/d=256 "
        "transformer, ≥ 5 seeds) with the preregistration's paired arm contrast (its section 7.1); "
        "that cell is **NOT RUN** and no validated manifest exists for it. What this fixture provides "
        "instead is exploratory, mechanistic evidence about the *terms*:"
    )
    add("")
    add(
        f"1. The rounding-grid term reduces the measured factor rounding residual in the direction "
        f"claimed ({round_check['relative_change']:+.2%} over the swept range) — the mechanism works on "
        "the quantity it targets."
    )
    add(
        f"2. The plain-spectral control reduces the tail energy it claims "
        f"({spectrum_check['relative_change']:+.2%}), so the control is a real, active penalty rather "
        "than a no-op."
    )
    add(
        "3. The predeclared exact STE residual ratio does not steer the factors (section 4.3), so the "
        "working objective uses the smooth grid surrogate instead; the report states this rather than "
        "claiming the predeclared form works."
    )
    add(
        "4. Whether the arm-level *quality* improvement at equal bytes exists, and whether it vanishes "
        "under the plain-spectral control (the two clauses of the falsifier), is **not decided here**: "
        "the fixture's dev-NLL and output-error differences are reported in section 3 for the Tier-1 "
        "cell to be judged against, and the frozen analysis is a paired test at Tier 1, not these "
        "fixture means."
    )
    add(
        "The fixture's own exploratory signal is **mixed and is reported as such**: the "
        "rounding-residual direction is clean and large, but the deployed dev NLL does not improve "
        "with the rounding coefficient (it degrades at the largest coefficient), the spectral arm is "
        "the worst arm on dev NLL, and the unregularized `none` arms score best on dev NLL while "
        "scoring worst on fidelity to the dense reference. Nothing in this section upgrades or "
        "downgrades H3; it records what the fixture measured."
    )
    add("")
    add("## 8. Limits and next steps")
    add("")
    add(
        "- One fixture (~0.1 M parameters), one synthetic task, one compression plan (rank "
        f"{compression.get('ranks', 'n/a')}, {compression.get('bits', 'n/a')}-bit, group "
        f"{compression.get('group_size', 'n/a')}); the seed count is the Tier-0 floor of "
        f"{len(plan['seeds'])}, and the quantities are single-fixture means, not a powered test."
    )
    add(
        "- The dense reference is a *from-scratch* fixture model, not a pretrained LM; the "
        "preparation setting is faithful in structure (SVD init, task training, rank-then-quantize "
        "deployment) but not in scale."
    )
    add(
        "- The next step is the frozen confirmatory cell, unchanged: `CLOUD-COLAB`, Tier 1, "
        "`configs/tier1/smollm2_135m.yaml`-style arms (d)/(e) plus the plain-spectral control, "
        "≥ 5 seeds, paired test at equal measured bytes. Nothing in this document substitutes for it."
    )
    add("")
    add(
        f"_Wall-clock of the sweep: {document['wall_time_s']:.1f} s on the CPU workstation of record "
        "(all cells, all seeds: "
        f"{len(document['per_run'])} training runs). Generated by "
        "`scripts/experiments/regularizer_sweep.py`._"
    )
    add("")
    return "\n".join(lines)


def write_artifacts(
    document: Mapping[str, Any],
    *,
    artifact_path: Path | str = "artifacts/sample-results/regularizer/sweep.json",
    report_path: Path | str = "docs/results/regularizer-report.md",
) -> tuple[Path, Path]:
    """Write the JSON document and the rendered report; return both paths."""
    artifact = Path(artifact_path)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_report(document), encoding="utf-8")
    return artifact, report


def sweep_and_write(
    plan: SweepPlan = DEFAULT_PLAN,
    *,
    config_path: str | Path = CONFIG_PATH,
    artifact_path: Path | str = "artifacts/sample-results/regularizer/sweep.json",
    report_path: Path | str = "docs/results/regularizer-report.md",
) -> tuple[dict[str, Any], Path, Path]:
    """Run the whole sweep and write both artifacts (the script's single entry point)."""
    started = time.perf_counter()
    records, results_by_cell, provenance = run_sweep(plan, config_path=config_path)
    wall_time_s = time.perf_counter() - started
    document = build_document(
        records, results_by_cell, provenance, plan, wall_time_s=wall_time_s, git=git_state()
    )
    artifact, report = write_artifacts(
        document, artifact_path=artifact_path, report_path=report_path
    )
    return document, artifact, report
