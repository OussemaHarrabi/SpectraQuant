"""Pre-flight disk projection: refuse a submission the worker cannot fit.

Why this exists (2026-10-11 cost post-mortem): one Tier-1 method run trained all eight arms and then
died at **export** with ``OSError: [Errno 28] No space left on device`` because six trained arms had
written four full-model checkpoints each into a 20 GB worker. The science had completed; nothing was
claimable. That failure is arithmetic, and arithmetic belongs on the workstation, not on a billed
GPU: this module projects the footprint of a resolved plan and refuses it before any remote call.

What is projected
-----------------
Every byte the run writes to a worker volume, itemised as :class:`ProjectionTerm`:

* ``environment`` - the pinned install (``RunSpec.install_spec``); the CUDA torch build dominates.
* ``hub_model_snapshot:<model>`` - the pinned model's cached upstream files.
* ``hub_dataset_cache`` - the pinned datasets' cached payloads and the streamed-corpus buffer.
* ``run_artifacts:<arm>`` - one (arm, seed)'s manifest, ``metrics.json`` and log.
* ``checkpoints:<arm>`` - one (arm, seed)'s checkpoints (the runner's largest artifacts).
* ``notebook`` / ``export_manifest`` - the executed notebook and the exported run manifest.

Two totals are reported, and the guard decides on the first
----------------------------------------------------------
* ``peak_bytes`` - the maximum the volume ever holds **under the runner's documented retention
  policy**: ``plan_runner._prune_checkpoints`` deletes each (arm, seed)'s checkpoint directory as
  soon as that arm finishes and records the count in ``training.checkpoints_pruned``. At most one
  (arm, seed) checkpoint set is therefore resident, which is why the *largest* checkpoint term is the
  one counted into the peak and the others are counted only into the unpruned total.
* ``unpruned_total_bytes`` - the same terms with every checkpoint set counted. This is the historical
  failure mode: it is what the volume would hold if the per-arm pruning policy were skipped or did
  not exist. It is reported (and warned about when it exceeds the headroom) but it is not the
  decision quantity, because a pruning runner never materialises it.

Honesty rules (``AGENTS.md`` §4.13)
-----------------------------------
Every term is an **estimate**, labelled ``analytical_estimate``, with the arithmetic that produced it
in its ``basis``. Some constants are *measured anchors* - a figure measured on this workstation or in
a collected bundle - and the basis string says which; an anchor is still not a measurement of the
remote volume. Nothing here is ever presented as an exact figure, and the guard never claims to have
measured the worker.

Usage
-----
::

    from spectraquant.cloud.preflight import assert_plan_fits
    projection = assert_plan_fits(load_plan("configs/tier1/smollm2_135m.yaml"), platform="kaggle")

The CLI runs the same function from ``spectraquant cloud preflight`` and from every ``cloud submit``
of a plan spec (see ``spectraquant.cli.main``).
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spectraquant.cloud.method_arms import ALLOCATION_ARM_KINDS
from spectraquant.cloud.plan_data import is_plan_document
from spectraquant.cloud.spec import PLATFORMS, RunSpec, cast_platform
from spectraquant.experiment_plan import PlanConfig, load_plan, plan_path

__all__ = [
    "DISK_ESTIMATE_LABEL",
    "PLATFORM_DISK_HEADROOM_BYTES",
    "DiskProjection",
    "ProjectionTerm",
    "WorkerDiskExceeded",
    "assert_plan_fits",
    "checkpoints_per_seed",
    "headroom_for_platform",
    "project_plan_disk",
    "recorded_plan_file",
]

logger = logging.getLogger("spectraquant.cloud.preflight")

#: The label every figure in this module carries: none of them is a measurement of the worker.
DISK_ESTIMATE_LABEL = "analytical_estimate"

#: Documented disk headroom per platform, from the cloud run book (``scripts/cloud/README.md`` §3,
#: "Free-tier limits (documented 2026-10-08)"): "~78 GB ephemeral" for consumer Colab and
#: "~20 GB ``/kaggle/working``" for Kaggle.
#:
#: Both figures are approximate operational limits, and each is read here in its **decimal**
#: spelling (``78 GB`` = 78e9 B) rather than as 78 GiB: the decimal reading is the smaller one and is
#: therefore the conservative reading of an approximate figure.
PLATFORM_DISK_HEADROOM_BYTES: dict[str, int] = {
    "colab": 78_000_000_000,
    "kaggle": 20_000_000_000,
}

#: Platforms whose headroom the run book does not document. ``local_cpu`` is measured instead (the
#: volume in front of us is the one that would be filled); ``colab_enterprise`` runs in a container
#: whose disk the design note does not state, so the caller must supply the figure rather than let
#: this module invent one.
_MEASURED_HEADROOM_PLATFORMS: frozenset[str] = frozenset({"local_cpu"})
_UNDOCUMENTED_HEADROOM_PLATFORMS: frozenset[str] = frozenset({"colab_enterprise"})

# -- size primitives ----------------------------------------------------------------
#: The runner loads and stores every model in float32 (``plan_runner.FROZEN_DTYPE``).
FLOAT32_BYTES_PER_PARAMETER = 4

#: Bytes per parameter of the *upstream* repository a model is cached from. The pinned SmolLM2-135M
#: repository ships one 2-byte-per-parameter ``model.safetensors`` (269,060,552 B, the figure the
#: Tier-1 plan comments record); 134,515,008 declared parameters x 2 B = 269,030,016 B, a 0.01 %
#: difference. An estimate, not a rule the hub guarantees.
UPSTREAM_BYTES_PER_PARAMETER = 2

#: AdamW optimiser state per trainable parameter, in float32 tensors. Measured anchor:
#: ``plan_runner`` records "a full-QAT checkpoint carries the optimiser state for every linear weight
#: (~1.6 GB)" for the 135 M model, i.e. ~3 fp32 copies of the trainable weights (exp_avg, exp_avg_sq
#: and the step-side bookkeeping the checkpoint flattens into the same payload).
ADAMW_STATE_FLOAT32_TENSORS_PER_TRAINABLE_PARAM = 3

#: Arm kinds whose *entire* weight set is trainable (full-model QAT): the optimiser state covers
#: every parameter, so the checkpoint is model + 3 x model.
FULL_TRAINABLE_KINDS: frozenset[str] = frozenset({"r2_fullqat_4bit"})

#: Fraction of the model's parameters a *factor/adapter* arm trains, used for its optimiser state.
#: An estimate: the recorded Tier-1 method run's CP-SAT allocation chose rank 32 on 206 of 210 layers
#: of a 576-wide hidden size (2 x 32 x 576 x 206 = 7.6 M parameters = 5.6 % of 134,515,008). 10 % is
#: the conservative upper bound used here; the true figure is not knowable without solving the
#: allocation, and the term is small beside the full model state either way.
FACTOR_PARAMETER_FRACTION_ESTIMATE = 0.10

#: Measured anchors for the per-arm artifacts, from the collected Tier-1 bundle at
#: ``artifacts/runs/tier1_smollm2_135m-cloud/download/spectraquant-export/`` (seven arms of the
#: frozen plan, 2026-10-10) and from this workstation's own tiny-budget run:
#:
#: * plain eval arm manifest 10,168 B / 10,352 B / 10,576 B -> 10,400 B base;
#: * allocated arm manifest 485,925 B (210 layers of ``allocation.layer.*`` metrics) -> 486,000 B;
#: * the trained arm's manifest 513,471 B for 300 steps -> ~57 B per recorded optimiser step
#:   (losses, dev losses and the four regularizer term means) -> 60 B rounded;
#: * ``metrics.json`` is the manifest's own metrics block: 2,795-4,972 B plain, 468,117 B allocated
#:   -> 3,000 B base and the same allocation addition;
#: * ``run.log`` 19,713 B for seven arms -> 2,800 B per (arm, seed);
#: * the executed notebook 35,732 B and the exported run manifest 526,327 B.
MEASURED_MANIFEST_BASE_BYTES = 10_400
MEASURED_ALLOCATION_MANIFEST_BYTES = 486_000
MEASURED_MANIFEST_BYTES_PER_TRAINING_STEP = 60
MEASURED_METRICS_BASE_BYTES = 3_000
MEASURED_RUN_LOG_BYTES_PER_ARM = 2_800
MEASURED_EXECUTED_NOTEBOOK_BYTES = 35_732
MEASURED_EXPORT_MANIFEST_BYTES = 526_327
MEASURED_PLAN_AGGREGATE_METRICS_BYTES = 4_972

#: Measured anchor: the pinned WikiText-2 document-level revision is cached at 13,383,039 B on this
#: workstation (``~/.cache/huggingface/datasets``). The *training* corpus is streamed from the plan's
#: train-role dataset (``allenai/c4`` ``en`` train is hundreds of gigabytes and is never materialised
#: - see ``plan_runner._training_corpus``), so only a bounded shard buffer is resident; 512 MiB is a
#: documented estimate for that buffer, not a measurement.
MEASURED_WIKITEXT_CACHE_BYTES = 13_383_039
STREAMED_SHARD_BUFFER_ESTIMATE_BYTES = 512 * 1024**2

#: Estimate of the materialised pinned environment, dominated by the torch wheel. The plan's install
#: spec (``spec._default_plan_install_spec``) materialises the lock and then *replaces* torch with the
#: CUDA build for a GPU plan; the CPU build is far smaller. Both figures are round estimates: the
#: wheel sizes are not declared anywhere in the repository and the uv cache is not pruned by the run.
WHEELS_ESTIMATE_BYTES_BY_DEVICE: dict[str, int] = {
    "cuda": 3_500_000_000,
    "cpu": 700_000_000,
}

#: Reserve kept free on a measured volume (``local_cpu``), so the projection never plans the
#: workstation down to its last byte. A documented constant, not a measurement.
MEASURED_VOLUME_RESERVE_BYTES = 2_000_000_000


class WorkerDiskExceeded(ValueError):
    """Raised when a plan's projected peak footprint exceeds the worker's documented headroom."""


def _gb(value: int) -> str:
    """Return ``value`` bytes as a decimal GB string (a human-facing *estimate* rendering)."""
    return f"{value / 1e9:.2f} GB"


@dataclass(frozen=True)
class ProjectionTerm:
    """One itemised line of a :class:`DiskProjection`.

    Attributes:
        name: stable identifier of the term (``checkpoints:<arm>``, ``run_artifacts:<arm>``, ...).
        bytes_estimate: the projected size in bytes - always an :data:`DISK_ESTIMATE_LABEL`.
        basis: the arithmetic that produced the estimate, plus the anchor it rests on.
        counted_in_peak: whether the runner's retention policy can have this term resident at the
            peak; every term is counted in the unpruned total.
        label: measurement-class honesty label; always :data:`DISK_ESTIMATE_LABEL` here.
    """

    name: str
    bytes_estimate: int
    basis: str
    counted_in_peak: bool
    label: str = DISK_ESTIMATE_LABEL

    def to_json(self) -> dict[str, Any]:
        """Return the term as JSON-serializable data."""
        return {
            "name": self.name,
            "bytes_estimate": int(self.bytes_estimate),
            "gigabytes_estimate": round(self.bytes_estimate / 1e9, 3),
            "basis": self.basis,
            "counted_in_peak": bool(self.counted_in_peak),
            "label": self.label,
        }

    def render(self) -> str:
        """Return one printable line: size, name and the arithmetic behind it."""
        return (
            f"{_gb(self.bytes_estimate):>12}  {self.name}"
            f"{'' if self.counted_in_peak else '  (unpruned total only)'}\n"
            f"{'':>12}  {self.basis}"
        )


@dataclass(frozen=True)
class DiskProjection:
    """The itemised pre-flight disk projection of one resolved plan.

    Attributes:
        plan_name: the plan's ``name``.
        plan_file: the plan document the projection was computed from.
        platform: the worker platform whose headroom was used.
        seeds: the (arm, seed) pairs the projection assumes, in order.
        headroom_bytes: the limit compared against.
        headroom_source: where the limit came from (documented run book figure, measured volume, or
            an explicit override).
        terms: the itemised projection, in report order.
        peak_bytes: ``sum(term.bytes_estimate for term in terms if term.counted_in_peak)`` - the
            decision quantity.
        unpruned_total_bytes: every term summed, i.e. the footprint if the runner's per-(arm, seed)
            checkpoint pruning did not happen. Reported, never the decision quantity.
    """

    plan_name: str
    plan_file: str
    platform: str
    seeds: tuple[int, ...]
    headroom_bytes: int
    headroom_source: str
    terms: tuple[ProjectionTerm, ...]
    peak_bytes: int
    unpruned_total_bytes: int

    # -- derived --------------------------------------------------------------------
    @property
    def exceeds(self) -> bool:
        """True when the projected peak does not fit the headroom."""
        return self.peak_bytes > self.headroom_bytes

    @property
    def unpruned_exceeds(self) -> bool:
        """True when the unpruned total does not fit - the run then depends on its pruning policy."""
        return self.unpruned_total_bytes > self.headroom_bytes

    @property
    def largest_term(self) -> ProjectionTerm:
        """The largest term counted in the peak (the offender a failure must name)."""
        counted = [term for term in self.terms if term.counted_in_peak]
        return max(counted, key=lambda term: term.bytes_estimate)

    @property
    def remaining_bytes(self) -> int:
        """Headroom minus the projected peak (negative when the projection does not fit)."""
        return self.headroom_bytes - self.peak_bytes

    def summary(self) -> str:
        """Return the itemised, printable pre-flight report."""
        lines = [
            f"pre-flight disk projection (all figures {DISK_ESTIMATE_LABEL}s): plan "
            f"{self.plan_name!r} on {self.platform}",
            f"  headroom: {_gb(self.headroom_bytes)} ({self.headroom_bytes:,} B) - "
            f"{self.headroom_source}",
            f"  seeds: {list(self.seeds)}",
            "  terms:",
        ]
        lines.extend(f"    {term.render()}" for term in self.terms)
        if self.exceeds:
            lines.append(
                f"  projected peak: {_gb(self.peak_bytes)} ({self.peak_bytes:,} B) - OVER the "
                f"{_gb(self.headroom_bytes)} headroom by "
                f"{_gb(-self.remaining_bytes)}; largest term {self.largest_term.name}"
            )
        else:
            lines.append(
                f"  projected peak: {_gb(self.peak_bytes)} ({self.peak_bytes:,} B); "
                f"{_gb(self.remaining_bytes)} of the headroom left; largest term "
                f"{self.largest_term.name}"
            )
        lines.append(
            f"  unpruned total: {_gb(self.unpruned_total_bytes)} ({self.unpruned_total_bytes:,} B)"
            + (
                " - OVER the headroom: the run is collectable only because the runner prunes each "
                "(arm, seed)'s checkpoints as that arm finishes"
                if self.unpruned_exceeds
                else ""
            )
        )
        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        """Return the projection as JSON-serializable data."""
        return {
            "plan_name": self.plan_name,
            "plan_file": self.plan_file,
            "platform": self.platform,
            "seeds": list(self.seeds),
            "headroom_bytes": int(self.headroom_bytes),
            "headroom_source": self.headroom_source,
            "terms": [term.to_json() for term in self.terms],
            "peak_bytes": int(self.peak_bytes),
            "peak_gigabytes_estimate": round(self.peak_bytes / 1e9, 3),
            "unpruned_total_bytes": int(self.unpruned_total_bytes),
            "unpruned_total_gigabytes_estimate": round(self.unpruned_total_bytes / 1e9, 3),
            "exceeds": self.exceeds,
            "unpruned_exceeds": self.unpruned_exceeds,
            "largest_term": self.largest_term.name,
            "label": DISK_ESTIMATE_LABEL,
        }


# ------------------------------------------------------------------------------------
# Headroom
# ------------------------------------------------------------------------------------
def headroom_for_platform(
    platform: str,
    *,
    override: int | None = None,
    root: str | Path | None = None,
) -> tuple[int, str]:
    """Return ``(headroom_bytes, source)`` for ``platform``.

    Args:
        platform: one of :data:`spectraquant.cloud.spec.PLATFORMS`.
        override: explicit headroom in bytes; wins over everything else.
        root: volume to measure for ``local_cpu`` (defaults to the current directory).

    Returns:
        The limit and a human-readable statement of where it came from.

    Raises:
        ValueError: the platform is unknown, or its headroom is not documented and no override was
            given (this module never invents a limit).
    """
    if override is not None:
        if override <= 0:
            raise ValueError(f"headroom override must be positive, got {override}")
        return int(override), "explicit headroom override"

    if platform not in PLATFORMS:
        raise ValueError(f"unknown platform {platform!r}; expected one of {', '.join(PLATFORMS)}")

    documented = PLATFORM_DISK_HEADROOM_BYTES.get(platform)
    if documented is not None:
        return documented, (
            "documented free-tier disk headroom (cloud run book §3, 2026-10-08); an approximate "
            "operational limit, read in its decimal spelling"
        )

    if platform in _MEASURED_HEADROOM_PLATFORMS:
        target = Path(root) if root is not None else Path.cwd()
        usage = shutil.disk_usage(target)
        usable = int(usage.free) - MEASURED_VOLUME_RESERVE_BYTES
        if usable <= 0:
            raise ValueError(
                f"the volume holding {target} has {usage.free:,} B free, below the "
                f"{MEASURED_VOLUME_RESERVE_BYTES:,} B reserve: no room for a run"
            )
        return usable, (
            f"measured free space on the volume holding {target} ({usage.free:,} B) minus a "
            f"{MEASURED_VOLUME_RESERVE_BYTES:,} B reserve"
        )

    raise ValueError(
        f"the disk headroom of platform {platform!r} is not documented in the cloud run book "
        "§3, and this guard never invents a limit: pass headroom_bytes explicitly"
    )


# ------------------------------------------------------------------------------------
# Per-arm sizing
# ------------------------------------------------------------------------------------
def checkpoints_per_seed(plan: PlanConfig, arm_kind: str) -> int:
    """Return how many checkpoint files one (arm, seed) writes.

    From ``spectraquant.training.loop``: an intermediate checkpoint every ``checkpoint_every`` steps
    and one final ``final.pt``, and **no** checkpoints at all when ``checkpoint_every`` is 0 (the
    runner passes ``int(schedule.checkpoint_every) or None``, which disables checkpointing).
    """
    schedule = plan.training
    if schedule is None or int(schedule.checkpoint_every) <= 0:
        return 0
    return int(schedule.steps) // int(schedule.checkpoint_every) + 1


def _trainable_parameter_fraction(arm_kind: str) -> float:
    """Return the estimated fraction of parameters an arm's optimiser state covers."""
    return 1.0 if arm_kind in FULL_TRAINABLE_KINDS else FACTOR_PARAMETER_FRACTION_ESTIMATE


def _checkpoint_basis(plan: PlanConfig, parameters: int, arm_kind: str, count: int) -> str:
    """Return the arithmetic behind one (arm, seed)'s checkpoint estimate."""
    trainable = int(parameters * _trainable_parameter_fraction(arm_kind))
    return (
        f"{count} checkpoint file(s) per (arm, seed) x (model state {parameters:,} params x "
        f"{FLOAT32_BYTES_PER_PARAMETER} B + AdamW state {trainable:,} trainable params x "
        f"{FLOAT32_BYTES_PER_PARAMETER} B x {ADAMW_STATE_FLOAT32_TENSORS_PER_TRAINABLE_PARAM} "
        f"fp32 tensors); checkpoint_every={int(plan.training.checkpoint_every) if plan.training else 0}"
    )


def _checkpoint_bytes_per_seed(plan: PlanConfig, parameters: int, arm_kind: str) -> int:
    """Return the projected bytes of one (arm, seed)'s checkpoint set."""
    count = checkpoints_per_seed(plan, arm_kind)
    if count == 0:
        return 0
    trainable = int(parameters * _trainable_parameter_fraction(arm_kind))
    per_checkpoint = (
        parameters * FLOAT32_BYTES_PER_PARAMETER
        + trainable * FLOAT32_BYTES_PER_PARAMETER * ADAMW_STATE_FLOAT32_TENSORS_PER_TRAINABLE_PARAM
    )
    return count * per_checkpoint


def _arm_artifact_bytes(plan: PlanConfig, arm_kind: str, *, trainable: bool) -> tuple[int, str]:
    """Return ``(bytes, basis)`` for one (arm, seed)'s manifest, metrics and log."""
    schedule = plan.training
    steps = int(schedule.steps) if schedule is not None and trainable else 0
    allocation = arm_kind in ALLOCATION_ARM_KINDS
    manifest = MEASURED_MANIFEST_BASE_BYTES
    metrics = MEASURED_METRICS_BASE_BYTES
    parts = [
        f"manifest {MEASURED_MANIFEST_BASE_BYTES:,} B",
        f"metrics.json {MEASURED_METRICS_BASE_BYTES:,} B",
        f"log {MEASURED_RUN_LOG_BYTES_PER_ARM:,} B",
    ]
    if allocation:
        manifest += MEASURED_ALLOCATION_MANIFEST_BYTES
        metrics += MEASURED_ALLOCATION_MANIFEST_BYTES
        parts.append(
            f"allocation metrics {MEASURED_ALLOCATION_MANIFEST_BYTES:,} B x 2 (210 layers)"
        )
    if steps:
        training = steps * MEASURED_MANIFEST_BYTES_PER_TRAINING_STEP
        manifest += training
        metrics += training
        parts.append(
            f"training history {steps} steps x {MEASURED_MANIFEST_BYTES_PER_TRAINING_STEP} B"
        )
    return manifest + metrics, "; ".join(parts)


# ------------------------------------------------------------------------------------
# Projection
# ------------------------------------------------------------------------------------
def resolve_seeds(plan: PlanConfig, seeds: Sequence[int] | None) -> tuple[int, ...]:
    """Return the seeds the projection assumes (default: the plan's whole reported list)."""
    if seeds is None:
        return tuple(int(seed) for seed in plan.seeds.reported)
    resolved = tuple(int(seed) for seed in seeds)
    if not resolved:
        raise ValueError("seeds must name at least one seed")
    return resolved


def project_plan_disk(
    plan: PlanConfig | str | Path,
    *,
    platform: str,
    seeds: Sequence[int] | None = None,
    headroom_bytes: int | None = None,
    headroom_root: str | Path | None = None,
) -> DiskProjection:
    """Project the on-disk footprint of ``plan`` on ``platform``, itemised term by term.

    Every figure is an estimate with its arithmetic recorded in ``basis`` (``AGENTS.md`` §4.13). The
    decision quantity is ``peak_bytes``; ``unpruned_total_bytes`` is reported as the historical
    failure mode (see the module docstring).

    Args:
        plan: a loaded :class:`~spectraquant.experiment_plan.PlanConfig`, or a plan name/path.
        platform: the worker platform whose documented headroom applies.
        seeds: seeds to assume; defaults to the plan's whole ``seeds.reported`` list, which is what a
            :class:`~spectraquant.cloud.spec.RunSpec` derived from the plan records.
        headroom_bytes: explicit headroom override (required for a platform the run book does not
            document).
        headroom_root: volume to measure when the platform's headroom is a measurement.

    Returns:
        The itemised :class:`DiskProjection`.

    Raises:
        FileNotFoundError: the plan does not exist.
        ValueError: an unknown platform, an unknown seed, or a headroom that cannot be resolved.
    """
    config = plan if isinstance(plan, PlanConfig) else load_plan(plan)
    plan_file = plan_path(plan) if not isinstance(plan, PlanConfig) else None
    resolved_platform = cast_platform(platform)
    resolved_seeds = resolve_seeds(config, seeds)
    known = {int(seed) for seed in config.seeds.reported}
    unknown = [seed for seed in resolved_seeds if seed not in known]
    if unknown:
        raise ValueError(
            f"seed(s) {unknown} are not in plan {config.name!r}'s reported list "
            f"{sorted(known)}: the projection only covers the preregistered seeds"
        )
    limit, source = headroom_for_platform(
        resolved_platform, override=headroom_bytes, root=headroom_root
    )

    terms: list[ProjectionTerm] = []

    wheels = WHEELS_ESTIMATE_BYTES_BY_DEVICE.get(
        "cuda" if config.device == "cuda" else "cpu", WHEELS_ESTIMATE_BYTES_BY_DEVICE["cpu"]
    )
    terms.append(
        ProjectionTerm(
            name="environment",
            bytes_estimate=wheels,
            basis=(
                f"{wheels:,} B for the pinned environment materialised by the run's install spec "
                f"(device={config.device}); the torch wheel dominates and the uv cache is not pruned"
            ),
            counted_in_peak=True,
        )
    )

    for model in config.models:
        snapshot = int(model.parameters) * UPSTREAM_BYTES_PER_PARAMETER
        terms.append(
            ProjectionTerm(
                name=f"hub_model_snapshot:{model.id}",
                bytes_estimate=snapshot,
                basis=(
                    f"{int(model.parameters):,} declared parameters x "
                    f"{UPSTREAM_BYTES_PER_PARAMETER} B/parameter upstream storage = {snapshot:,} B; "
                    "the 2 B/parameter assumption cross-checks against the pinned Tier-1 model, "
                    "whose upstream model.safetensors measures 269,060,552 B on this workstation "
                    "(0.01 % apart). The loader additionally holds a float32 copy in RAM, which is "
                    "not on disk"
                ),
                counted_in_peak=True,
            )
        )

    dataset_term = MEASURED_WIKITEXT_CACHE_BYTES + STREAMED_SHARD_BUFFER_ESTIMATE_BYTES
    terms.append(
        ProjectionTerm(
            name="hub_dataset_cache",
            bytes_estimate=dataset_term,
            basis=(
                f"{MEASURED_WIKITEXT_CACHE_BYTES:,} B (the pinned WikiText-2 document-level revision, "
                f"measured in this workstation's cache) + "
                f"{STREAMED_SHARD_BUFFER_ESTIMATE_BYTES:,} B for the streamed train-corpus shard "
                "buffer (the train role streams; an estimate, not a measurement)"
            ),
            counted_in_peak=True,
        )
    )

    largest_parameters = max(int(model.parameters) for model in config.models)
    #: per (arm, seed) checkpoint bytes of every trainable arm, keyed by arm name.
    per_seed_checkpoints: dict[str, int] = {}
    for arm in config.arms:
        artifacts, artifacts_basis = _arm_artifact_bytes(config, arm.kind, trainable=arm.trainable)
        terms.append(
            ProjectionTerm(
                name=f"run_artifacts:{arm.name}",
                bytes_estimate=artifacts * len(resolved_seeds),
                basis=(
                    f"{artifacts:,} B per (arm, seed) x {len(resolved_seeds)} seed(s): "
                    f"{artifacts_basis}"
                ),
                counted_in_peak=True,
            )
        )
        if not arm.trainable:
            continue
        per_seed = _checkpoint_bytes_per_seed(config, largest_parameters, arm.kind)
        if per_seed == 0:
            continue
        per_seed_checkpoints[arm.name] = per_seed
        terms.append(
            ProjectionTerm(
                name=f"checkpoints:{arm.name}",
                bytes_estimate=per_seed * len(resolved_seeds),
                basis=(
                    f"{_checkpoint_basis(config, largest_parameters, arm.kind, checkpoints_per_seed(config, arm.kind))}"
                    f" x {len(resolved_seeds)} seed(s) = {per_seed * len(resolved_seeds):,} B; "
                    "unpruned total only - the resident set is the largest single (arm, seed) below"
                ),
                counted_in_peak=False,
            )
        )

    if per_seed_checkpoints:
        # One (arm, seed) checkpoint set is resident at a time: the runner prunes each arm's
        # checkpoint directory as the arm finishes (plan_runner._prune_checkpoints, recorded in every
        # manifest as training.checkpoints_pruned). The largest single set is therefore the one the
        # volume must hold; the per-arm terms above stay in the unpruned total.
        resident_arm, resident_bytes = max(per_seed_checkpoints.items(), key=lambda item: item[1])
        terms.append(
            ProjectionTerm(
                name="checkpoints_resident_peak",
                bytes_estimate=resident_bytes,
                basis=(
                    f"the largest single (arm, seed) checkpoint set ({resident_arm}) among the "
                    f"trainable arms above: the runner prunes an arm's checkpoints when it finishes, "
                    f"so one set is resident at the peak (plan_runner._prune_checkpoints)"
                ),
                counted_in_peak=True,
            )
        )

    terms.append(
        ProjectionTerm(
            name="executed_notebook",
            bytes_estimate=MEASURED_EXECUTED_NOTEBOOK_BYTES,
            basis=f"{MEASURED_EXECUTED_NOTEBOOK_BYTES:,} B (measured in the collected Tier-1 bundle)",
            counted_in_peak=True,
        )
    )
    terms.append(
        ProjectionTerm(
            name="export_manifest_and_aggregate",
            bytes_estimate=MEASURED_EXPORT_MANIFEST_BYTES + MEASURED_PLAN_AGGREGATE_METRICS_BYTES,
            basis=(
                f"{MEASURED_EXPORT_MANIFEST_BYTES:,} B exported run manifest + "
                f"{MEASURED_PLAN_AGGREGATE_METRICS_BYTES:,} B plan-level aggregate metrics.json "
                "(both measured in the collected Tier-1 bundle; the exported manifest embeds the "
                "aggregate plus every arm's metrics)"
            ),
            counted_in_peak=True,
        )
    )

    peak = sum(term.bytes_estimate for term in terms if term.counted_in_peak)
    unpruned = sum(term.bytes_estimate for term in terms)
    projection = DiskProjection(
        plan_name=config.name,
        plan_file=str(plan_file) if plan_file is not None else "<loaded PlanConfig>",
        platform=resolved_platform,
        seeds=resolved_seeds,
        headroom_bytes=limit,
        headroom_source=source,
        terms=tuple(terms),
        peak_bytes=peak,
        unpruned_total_bytes=unpruned,
    )
    if projection.unpruned_exceeds and not projection.exceeds:
        logger.warning(
            "pre-flight: plan %s needs %s unpruned (checkpoints per arm x arms) against a %s "
            "headroom: the run is collectable only because the runner prunes each arm's checkpoints "
            "as it finishes (analytical estimates)",
            config.name,
            _gb(unpruned),
            _gb(limit),
        )
    return projection


def assert_plan_fits(
    plan: PlanConfig | str | Path,
    *,
    platform: str,
    seeds: Sequence[int] | None = None,
    headroom_bytes: int | None = None,
    headroom_root: str | Path | None = None,
) -> DiskProjection:
    """Refuse a plan whose projected peak footprint exceeds the worker's documented headroom.

    Args:
        plan: a loaded :class:`~spectraquant.experiment_plan.PlanConfig`, or a plan name/path.
        platform: the worker platform the run would target.
        seeds: seeds to assume; defaults to the plan's reported list.
        headroom_bytes: explicit headroom override.
        headroom_root: volume to measure when the platform's headroom is a measurement.

    Returns:
        The :class:`DiskProjection` when it fits (never ``None``).

    Raises:
        WorkerDiskExceeded: the projection does not fit; the message names the projected size, the
            limit and the largest counted term, and lists every term.
        FileNotFoundError: the plan does not exist.
        ValueError: an unknown platform, seed or headroom that cannot be resolved.
    """
    projection = project_plan_disk(
        plan,
        platform=platform,
        seeds=seeds,
        headroom_bytes=headroom_bytes,
        headroom_root=headroom_root,
    )
    if not projection.exceeds:
        return projection
    offender = projection.largest_term
    raise WorkerDiskExceeded(
        f"pre-flight refused the submission: plan {projection.plan_name!r} projects a peak "
        f"footprint of {_gb(projection.peak_bytes)} ({projection.peak_bytes:,} B, "
        f"{DISK_ESTIMATE_LABEL}) against the {projection.platform} headroom of "
        f"{_gb(projection.headroom_bytes)} ({projection.headroom_bytes:,} B; "
        f"{projection.headroom_source}). The largest counted term is {offender.name!r} at "
        f"{_gb(offender.bytes_estimate)} ({offender.basis}). Reduce the footprint before submitting "
        f"(drop checkpoints by setting training.checkpoint_every=0, run fewer seeds, or select a "
        f"platform with more disk). Itemised projection:\n{projection.summary()}"
    )


def recorded_plan_file(spec: RunSpec, *, root: str | Path | None = None) -> Path | None:
    """Return the plan document a spec records, or ``None`` when it does not record a plan.

    A plan-derived spec records its plan in ``experiment_config`` (``plan_to_run_spec``); a
    Hydra-config spec records a composition config there instead, which this module cannot project.

    Args:
        spec: the frozen spec about to be submitted.
        root: repository root used to resolve a repo-relative recorded path when the path does not
            resolve from the current working directory.

    Returns:
        The plan path, or ``None`` when ``experiment_config`` is not a plan document (including when
        the file is not present in this checkout).
    """
    raw = Path(spec.experiment_config)
    candidates: list[Path] = [raw]
    if root is not None and not raw.is_absolute():
        candidates.append(Path(root) / raw)
    for candidate in candidates:
        if candidate.is_file() and is_plan_document(candidate):
            return candidate
    return None


def projection_for_spec(
    spec: RunSpec,
    *,
    root: str | Path | None = None,
    headroom_bytes: int | None = None,
) -> DiskProjection | None:
    """Project the disk footprint of the plan a spec records, or ``None`` when it records none.

    Args:
        spec: the frozen spec about to be submitted.
        root: repository root for resolving a repo-relative plan path.
        headroom_bytes: explicit headroom override.

    Returns:
        The itemised projection, or ``None`` when the spec's ``experiment_config`` is not a plan
        document (a Hydra composition config, or a plan absent from this checkout).
    """
    plan_file = recorded_plan_file(spec, root=root)
    if plan_file is None:
        logger.info(
            "pre-flight: spec %s records experiment_config=%r, which is not a cloud plan document: "
            "no disk projection was computed (the guard covers plans; a Hydra composition config "
            "carries no model or schedule to project)",
            spec.run_id,
            spec.experiment_config,
        )
        return None
    projection = project_plan_disk(
        plan_file,
        platform=spec.platform,
        seeds=spec.seeds,
        headroom_bytes=headroom_bytes,
        headroom_root=root,
    )
    return projection


def assert_spec_fits(
    spec: RunSpec,
    *,
    root: str | Path | None = None,
    headroom_bytes: int | None = None,
) -> DiskProjection | None:
    """:func:`assert_plan_fits` for a spec: refuse an over-budget plan, ignore a non-plan spec.

    Returns:
        The projection, or ``None`` when the spec records no plan document.

    Raises:
        WorkerDiskExceeded: the recorded plan's projection exceeds the headroom.
    """
    projection = projection_for_spec(spec, root=root, headroom_bytes=headroom_bytes)
    if projection is None or not projection.exceeds:
        return projection
    offender = projection.largest_term
    raise WorkerDiskExceeded(
        f"pre-flight refused the submission of spec {spec.run_id!r}: plan "
        f"{projection.plan_name!r} projects a peak footprint of {_gb(projection.peak_bytes)} "
        f"({projection.peak_bytes:,} B, {DISK_ESTIMATE_LABEL}) against the {projection.platform} "
        f"headroom of {_gb(projection.headroom_bytes)} ({projection.headroom_bytes:,} B; "
        f"{projection.headroom_source}). The largest counted term is {offender.name!r} at "
        f"{_gb(offender.bytes_estimate)} ({offender.basis}). Itemised projection:\n"
        f"{projection.summary()}"
    )
