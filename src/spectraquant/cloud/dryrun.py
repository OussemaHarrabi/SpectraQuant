"""Local dry-run of a whole plan on CPU with tiny budgets (pre-submission wiring check).

Why this exists (2026-10-11 cost post-mortem): two submissions of the same Tier-1 plan burned about
an hour of the GPU budget on defects that only surfaced on the GPU - a serialization device error and
a granularity argument error. Neither needed a GPU to find: both are visible the moment the plan's
arms are executed end to end with tiny budgets. This module executes **every arm the plan declares**,
one arm per runner invocation (the runner itself refuses to bind the allocated arms and the rank arms
in one command), on CPU, with a handful of optimiser steps and a tiny evaluation token budget, and
then asserts that each arm produced a *schema-valid manifest plus a metrics bundle*.

What a pass proves, and what it does not
----------------------------------------
Passing proves the plan's **wiring**: every declared arm is implemented and bindable on this plan,
the pinned model and datasets resolve, the compression/quantization path runs on the real model, the
manifest serializes and validates, and ``metrics.json`` is the manifest's own metrics block. It does
*not* measure the plan: the schedule, the token budgets and the corpus sizes are replaced with tiny
values (see :data:`TINY_TRAINING`), and the compression grid is narrowed to the points the arms are
bound to (see :func:`tiny_grid_document`), so a dry-run number is a path artifact and never a plan
measurement (``AGENTS.md`` §4.13, §5). The narrowed grid is what keeps the check inside a few
minutes: the allocated arms evaluate their proxy score table once per (layer x candidate), and the
frozen Tier-1 grid is 5 ranks x 4 bit widths.

Anything the runner records as skipped is reported as a failure of the dry-run: an arm a submission
would silently skip is exactly the defect class this check exists to catch.

Usage
-----
::

    spectraquant cloud dry-run --plan configs/tier1/smollm2_135m.yaml

The command is also the second half of the one-command pre-submission checklist,
``spectraquant cloud preflight``.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from spectraquant.cloud.adapters.base import CommandResult
from spectraquant.cloud.method_arms import ALLOCATION_ARM_KINDS
from spectraquant.cloud.plan_runner import (
    IMPLEMENTED_ARM_KINDS,
    M3_ARM_KINDS,
    arm_not_implemented_error,
)
from spectraquant.cloud.remote import RESULT_LINE_PREFIX
from spectraquant.experiment_plan import ArmSpec, PlanConfig, load_plan, plan_path
from spectraquant.reporting.manifests import validate_manifest_file

__all__ = [
    "DEFAULT_ARM_TIMEOUT_SECONDS",
    "DEFAULT_JOBS",
    "DEFAULT_THREADS",
    "TINY_MAX_DOCUMENTS",
    "TINY_MAX_TOKENS",
    "TINY_SEQ_LEN",
    "TINY_TRAINING",
    "ArmCommandRunner",
    "ArmOutcome",
    "DryRunReport",
    "arm_binding_args",
    "dry_run_plan",
    "run_arm_command",
    "select_grid_point",
    "tiny_grid_document",
    "tiny_plan_document",
    "write_tiny_plan",
]

logger = logging.getLogger("spectraquant.cloud.dryrun")

#: The tiny training schedule every dry-run arm runs. Rationale per field:
#:
#: * ``steps=2`` - two optimiser steps reach the checkpoint interval and the end of the schedule, so
#:   the training loop, its checkpoint writer and the runner's post-arm checkpoint pruning are all
#:   exercised (the first Tier-1 method run died at export because checkpoints were never pruned).
#: * ``checkpoint_every=1`` - writes a checkpoint (the run's largest artifact) and its pruning path.
#: * ``eval_every=1`` - records a dev measurement, so the loop's evaluation branch runs.
#: * ``warmup_steps=1`` - exercises the warmup scheduler without an empty or oversized warmup.
#: * ``batch_size=2`` / ``seq_len=32`` - the smallest shapes the loop accepts while still moving real
#:   tensors through the model (``seq_len`` is also the evaluation window passed on the command line).
#: * ``corpus_documents=2`` - the plan pins the train corpus size; the dry-run streams two documents.
#:
#: These are *path* values, not scientific ones: the dry-run is not a plan measurement.
TINY_TRAINING: dict[str, Any] = {
    "steps": 2,
    "batch_size": 2,
    "seq_len": 32,
    "warmup_steps": 1,
    "eval_every": 1,
    "checkpoint_every": 1,
    "corpus_documents": 2,
}

#: Evaluation window width (tokens) for the perplexity protocol.
TINY_SEQ_LEN = 32

#: Cap on scored tokens: a handful of windows, enough for the protocol to produce a number.
TINY_MAX_TOKENS = 64

#: Cap on documents read from the bounded (non-streamed) splits: the perplexity corpus and the dev
#: split.
TINY_MAX_DOCUMENTS = 2

#: Cap on the calibration documents the allocated arms' activation capture reads.
TINY_CALIBRATION_DOCUMENTS = 2

#: Per-arm wall-clock ceiling. A dry-run arm that has not finished in this long is killed and
#: reported as failed: an arm that hangs is a defect worth failing on, not a reason to spend GPU
#: hours discovering it later.
DEFAULT_ARM_TIMEOUT_SECONDS = 900.0

#: How many arms run at once. The arms are independent processes, so concurrency turns the check's
#: cost from the sum of the arms into the slowest arm (measured: the two allocated arms dominate at
#: ~3 min each because of the activation capture and the CP-SAT solve). The bound is a *memory*
#: bound, not a speed one: every arm holds the float32 model plus a working copy (~2 GiB each), and
#: this workstation has ~15 GiB usable (AGENTS.md §2), so four is the largest safe default.
DEFAULT_JOBS = 4

#: ``torch.set_num_threads`` for each arm. Four, not one: the dry-run's numbers are path artifacts
#: and never claims, and with :data:`DEFAULT_JOBS` arms running concurrently the run is not
#: bit-reproducible in any case - measured on this workstation, one thread multiplies the wall-clock
#: of the capture/solve arms by ~2-4x (the frozen plan's allocation arm: 372 s at four threads, and
#: the trained arm 806 s at one thread against 526 s at four). ``--threads 1 --jobs 1`` is the fully
#: sequential, bit-reproducible mode for debugging a single arm.
DEFAULT_THREADS = 4

#: Grid values the dry-run prefers when it has to bind an arm the plan leaves unbound: the point the
#: frozen Tier-1 plan's own ``rank_then_quant`` arm uses (``{bits: 4, rank: 8}``). When the plan's
#: grid does not contain the preferred value, :func:`select_grid_point` takes the closest one.
PREFERRED_RANK = 8
PREFERRED_BITS = 4

#: The grid point the dry-run binds the rank arms to when the plan's ``training`` block is absent.
_BITS_SUFFIX = re.compile(r"_(\d+)$")

#: A command runner for one arm: ``(argv, timeout_seconds) -> CommandResult``. Injected in tests so
#: the unit suite never spawns a subprocess or touches the network.
ArmCommandRunner = Callable[[Sequence[str], float], CommandResult]


def select_grid_point(values: Sequence[int], preferred: int) -> int:
    """Return the value of ``values`` closest to ``preferred`` (ties go to the smaller value).

    Raises:
        ValueError: ``values`` is empty.
    """
    candidates = [int(value) for value in values]
    if not candidates:
        raise ValueError("cannot select a grid point from an empty grid")
    return min(candidates, key=lambda value: (abs(value - preferred), value))


def named_bits(arm_name: str) -> int | None:
    """Return the ``_<bits>`` suffix of an arm name, or ``None`` when it carries none."""
    match = _BITS_SUFFIX.search(arm_name)
    return int(match.group(1)) if match is not None else None


def tiny_grid_document(plan: PlanConfig, *, rank: int, bits: int) -> dict[str, Any]:
    """Return the narrowed compression grid the dry-run executes.

    The dry-run keeps only the grid points its arms can actually be bound to: the point the caller
    asked for, every bit width an arm names in its own name, and every point the plan binds in an
    arm's ``point``. The rest of the grid is *candidate* space, and the allocated arms evaluate their
    proxy score table once per (layer x candidate): measured on this workstation (threads=4), the
    frozen Tier-1 grid (5 ranks x 4 bit widths) makes one allocation arm take 372 s while a two-point
    grid takes 84 s, for the same code path. The grid's breadth is a scientific choice the dry-run
    does not verify; what it verifies is that the allocation, the storage quantization and the
    manifest serialization run at all.
    """
    bit_values = {int(bits)}
    rank_values = {int(rank)}
    for arm in plan.arms:
        named = named_bits(arm.name)
        if named is not None:
            bit_values.add(int(named))
        if "bits" in arm.point:
            bit_values.add(int(arm.point["bits"]))
        if "rank" in arm.point:
            rank_values.add(int(arm.point["rank"]))
    return {
        "bits": sorted(bit_values),
        "ranks": sorted(rank_values),
        "group_sizes": sorted(int(value) for value in plan.grid.group_sizes),
        "budget_ladder_bytes": sorted(int(value) for value in plan.grid.budget_ladder_bytes),
    }


def tiny_plan_document(plan: PlanConfig, *, rank: int, bits: int) -> dict[str, Any]:
    """Return the tiny-budget copy of ``plan`` the dry-run executes.

    The copy keeps the plan's identity, models, datasets, seeds, arms, harness and cost envelope,
    replaces the training schedule with :data:`TINY_TRAINING` and narrows the compression grid to the
    points its arms can be bound to (see :func:`tiny_grid_document`). It is written to the dry-run's
    work directory: the runner records ``plan.path`` in every manifest, and that path must point at
    what actually ran.
    """
    document = plan.model_dump(mode="json")
    if document.get("training") is not None:
        tiny = dict(document["training"])
        tiny.update(TINY_TRAINING)
        # A declared learning-rate grid is kept: it is part of the plan's shape and its candidates are
        # cheap at one step each.
        document["training"] = tiny
    grid = tiny_grid_document(plan, rank=rank, bits=bits)
    document["grid"] = grid
    document["notes"] = (
        f"{document.get('notes', '')}\n\n[dry-run] schedule and evaluation budgets replaced with "
        f"tiny values ({TINY_TRAINING}) and the compression grid narrowed to {grid} by "
        "`spectraquant cloud dry-run`: this document is a wiring check, not the plan as frozen."
    ).strip()
    return document


def write_tiny_plan(plan: PlanConfig, dest_dir: str | Path, *, rank: int, bits: int) -> Path:
    """Write the tiny-budget copy of ``plan`` into ``dest_dir`` and return its path."""
    target = Path(dest_dir) / f"{plan.name}-dryrun.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(tiny_plan_document(plan, rank=rank, bits=bits), sort_keys=False),
        encoding="utf-8",
    )
    # The copy must still be a valid plan: reloading it here means a derivation bug is a local error
    # rather than a confusing remote one.
    load_plan(target)
    return target


def arm_binding_args(
    arm: ArmSpec,
    *,
    rank: int,
    bits: int,
    budget_bytes: int,
) -> list[str]:
    """Return the grid-point flags one arm needs, in the runner's own vocabulary.

    The runner refuses what an arm cannot use (``--bits``/``--rank`` on an allocated arm), so the
    dry-run must bind each arm the way the arm's own kind is bound:

    * allocated arms (``ALLOCATION_ARM_KINDS``) take the byte budget, never a rank or a bit width;
    * the frozen M3 arms carry their whole grid point in the plan (a plan that declares them binds
      them; the flags are omitted);
    * ``fp16_reference`` takes nothing, and ``ptq_uniform`` takes a bit width only when its own name
      does not already carry one (the arm name is authoritative);
    * every other implemented kind takes ``--rank`` and ``--bits``.
    """
    kind = arm.kind
    if kind in ALLOCATION_ARM_KINDS:
        return ["--budget-bytes", str(int(budget_bytes))]
    if kind in M3_ARM_KINDS or kind == "fp16_reference":
        return []
    if kind == "ptq_uniform":
        if named_bits(arm.name) is not None:
            return []
        return ["--bits", str(int(bits))]
    return ["--rank", str(int(rank)), "--bits", str(int(bits))]


def run_arm_command(argv: Sequence[str], timeout_seconds: float) -> CommandResult:
    """Run one arm's runner command in a subprocess, never raising for a non-zero exit.

    A timeout, a missing executable or a failure to start are reported as a non-zero
    :class:`~spectraquant.cloud.adapters.base.CommandResult` carrying the reason, so one arm's
    failure is a fact about that arm rather than an exception that aborts the report.
    """
    try:
        completed = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            argv=list(argv),
            returncode=-9,
            stdout=_as_text(exc.stdout),
            stderr=f"timed out after {timeout_seconds:g}s\n{_as_text(exc.stderr)}",
        )
    except OSError as exc:  # pragma: no cover - the interpreter always exists in practice
        return CommandResult(
            argv=list(argv),
            returncode=-1,
            stderr=f"could not start {argv[0]!r}: {exc}",
        )
    return CommandResult(
        argv=list(argv),
        returncode=int(completed.returncode),
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


def _as_text(value: Any) -> str:
    """Return ``value`` as text (``None`` becomes an empty string; bytes are decoded)."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


#: How much of a failed arm's output is kept in the report (characters, from the end).
OUTPUT_TAIL_CHARS = 2_000


def _output_tail(output: str, *, limit: int = OUTPUT_TAIL_CHARS) -> str:
    """Return the end of ``output``, bounded, so a failure carries its evidence."""
    text = _as_text(output).strip()
    if len(text) <= limit:
        return text
    return "..." + text[-limit:]


@dataclass(frozen=True)
class ArmOutcome:
    """The result of one arm's dry-run invocation.

    Attributes:
        arm: the arm's name.
        kind: the arm's kind.
        status: ``"passed"``, ``"failed"`` or ``"skipped"`` (skipped covers arms the runner cannot
            execute at all, e.g. a kind that is not implemented).
        detail: the failure reason, or a one-line statement of what was verified.
        seconds: wall-clock time of this arm's invocation.
        argv: the command that was run (recorded so a failure is reproducible by hand).
        manifest: path of the validated manifest, when one was produced.
        metrics: path of the metrics bundle, when one was produced.
        perplexity: the arm's perplexity, when the runner reported one.
        returncode: the runner's exit status (``None`` when the arm was never invoked).
        output_tail: the last part of the runner's (redacted) output, kept for a failed arm so the
            failure can be diagnosed without re-running it.
    """

    arm: str
    kind: str
    status: str
    detail: str
    seconds: float
    argv: tuple[str, ...] = ()
    manifest: str | None = None
    metrics: str | None = None
    perplexity: float | None = None
    returncode: int | None = None
    output_tail: str = ""

    @property
    def ok(self) -> bool:
        """True only for a passed arm: a skipped arm is not a pass."""
        return self.status == "passed"

    def to_json(self) -> dict[str, Any]:
        """Return the outcome as JSON-serializable data."""
        return {
            "arm": self.arm,
            "kind": self.kind,
            "status": self.status,
            "detail": self.detail,
            "seconds": round(self.seconds, 3),
            "argv": list(self.argv),
            "manifest": self.manifest,
            "metrics": self.metrics,
            "perplexity": self.perplexity,
            "returncode": self.returncode,
            "output_tail": self.output_tail,
        }

    def render(self) -> str:
        """Return one printable line."""
        return (
            f"{self.status:<7} {self.arm:<32} {self.seconds:6.1f}s  "
            f"{'' if self.perplexity is None else f'perplexity={self.perplexity:.4f} '}{self.detail}"
        )


@dataclass(frozen=True)
class DryRunReport:
    """The whole plan's dry-run report: one :class:`ArmOutcome` per declared arm.

    Attributes:
        plan_name: the plan's name.
        plan_file: the plan document the dry-run was asked to check.
        tiny_plan: the tiny-budget copy actually executed (recorded because the runner writes it
            into every manifest as ``plan.path``).
        work_dir: where the tiny plan and the per-arm outputs were written.
        seed: the single seed the dry-run ran.
        device: the device the arms ran on (``cpu``: this is the workstation).
        outcomes: per-arm results, in plan order.
        wall_time_s: total wall-clock time.
    """

    plan_name: str
    plan_file: str
    tiny_plan: str
    work_dir: str
    seed: int
    device: str
    outcomes: tuple[ArmOutcome, ...]
    wall_time_s: float
    settings: tuple[tuple[str, Any], ...] = ()

    @property
    def failures(self) -> tuple[ArmOutcome, ...]:
        """The arms that failed."""
        return tuple(outcome for outcome in self.outcomes if outcome.status == "failed")

    @property
    def skipped(self) -> tuple[ArmOutcome, ...]:
        """The arms that could not be executed at all."""
        return tuple(outcome for outcome in self.outcomes if outcome.status == "skipped")

    @property
    def passed(self) -> tuple[ArmOutcome, ...]:
        """The arms that passed."""
        return tuple(outcome for outcome in self.outcomes if outcome.status == "passed")

    @property
    def ok(self) -> bool:
        """True when every declared arm passed.

        A skipped arm is not a pass: a submission that would silently skip an arm is a defect this
        check exists to surface.
        """
        return not self.failures and not self.skipped and bool(self.outcomes)

    def summary(self) -> str:
        """Return the printable per-arm report."""
        lines = [
            f"cloud dry-run: plan {self.plan_name!r} on device={self.device} seed={self.seed} "
            f"({len(self.outcomes)} arm(s), {self.wall_time_s:.1f}s wall-clock)",
            f"  tiny plan: {self.tiny_plan}",
            f"  work dir: {self.work_dir}",
        ]
        if self.settings:
            rendered = ", ".join(f"{key}={value}" for key, value in self.settings)
            lines.append(f"  bindings: {rendered}")
        lines.extend(f"  {outcome.render()}" for outcome in self.outcomes)
        for outcome in self.failures:
            if outcome.output_tail:
                lines.append(f"  --- {outcome.arm} output tail ---")
                lines.extend(f"  | {line}" for line in outcome.output_tail.splitlines())
        verdict = (
            "PASSED: every declared arm produced a schema-valid manifest and metrics bundle"
            if self.ok
            else f"FAILED: {len(self.failures)} failed, {len(self.skipped)} skipped, "
            f"{len(self.passed)} passed"
        )
        lines.append(f"  {verdict}")
        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        """Return the report as JSON-serializable data."""
        return {
            "plan_name": self.plan_name,
            "plan_file": self.plan_file,
            "tiny_plan": self.tiny_plan,
            "work_dir": self.work_dir,
            "seed": self.seed,
            "device": self.device,
            "settings": dict(self.settings),
            "wall_time_s": round(self.wall_time_s, 3),
            "ok": self.ok,
            "outcomes": [outcome.to_json() for outcome in self.outcomes],
        }


def _parse_result_line(output: str) -> dict[str, Any] | None:
    """Return the parsed ``SPECTRAQUANT_RESULT_JSON`` payload in ``output``, or ``None``."""
    payload: dict[str, Any] | None = None
    for line in output.splitlines():
        if line.startswith(RESULT_LINE_PREFIX):
            # The last one wins: a runner prints exactly one, and taking the last is the safest
            # reading of a noisy stream.
            try:
                candidate = json.loads(line[len(RESULT_LINE_PREFIX) :])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                payload = candidate
    return payload


def _validate_bundle(
    arm: ArmSpec,
    arm_dir: Path,
    seed: int,
    payload: Mapping[str, Any] | None,
) -> tuple[str, str | None, str | None, float | None]:
    """Assert one arm produced a schema-valid manifest plus its metrics bundle.

    The runner writes ``run_manifest.json`` (schema-validated before it is written) and
    ``metrics.json`` (the manifest's own ``metrics`` block, which is what the collector and the
    analysis read) per (arm, seed), so both are checked *here*, independently of the runner's own
    claim: the manifest is re-validated against ``artifacts/schemas/run-manifest.schema.json``, and
    ``metrics.json`` must be JSON-equal to the manifest's metrics.

    Returns:
        ``(detail, manifest_path, metrics_path, perplexity)``.

    Raises:
        AssertionError: the bundle is missing, unparseable or inconsistent.
    """
    manifest_path: Path | None = None
    metrics_path: Path | None = None
    perplexity: float | None = None
    if payload is not None:
        for entry in payload.get("arms_run", ()):  # type: ignore[union-attr]
            if isinstance(entry, Mapping) and entry.get("arm") == arm.name:
                if entry.get("manifest") is not None:
                    manifest_path = Path(str(entry["manifest"]))
                if entry.get("metrics") is not None:
                    metrics_path = Path(str(entry["metrics"]))
                if entry.get("perplexity") is not None:
                    perplexity = float(entry["perplexity"])
    if manifest_path is None:
        found = sorted((arm_dir / arm.name / f"seed-{seed}").glob("run_manifest.json"))
        if not found:
            found = sorted(arm_dir.rglob("run_manifest.json"))
        manifest_path = found[0] if found else None
    if manifest_path is None or not manifest_path.is_file():
        raise AssertionError(
            f"arm {arm.name!r} exited zero but wrote no run_manifest.json under {arm_dir}"
        )

    document = validate_manifest_file(manifest_path)
    metrics = document.get("metrics")
    if not isinstance(metrics, Mapping) or not metrics:
        raise AssertionError(
            f"arm {arm.name!r} wrote a schema-valid manifest with an empty metrics block: "
            f"{manifest_path}"
        )

    if metrics_path is None:
        candidate = manifest_path.with_name("metrics.json")
        metrics_path = candidate if candidate.is_file() else None
    if metrics_path is None or not metrics_path.is_file():
        raise AssertionError(
            f"arm {arm.name!r} wrote no metrics.json beside {manifest_path}: the bundle is "
            "incomplete"
        )
    bundle_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if bundle_metrics != dict(metrics):
        raise AssertionError(
            f"arm {arm.name!r}'s metrics bundle {metrics_path} does not match the manifest's own "
            "metrics block: the two documents disagree about what was measured"
        )
    if perplexity is None and isinstance(metrics.get("perplexity"), (int, float)):
        perplexity = float(metrics["perplexity"])
    return (
        f"manifest schema-valid ({manifest_path.name}), metrics bundle matches",
        str(manifest_path),
        str(metrics_path),
        perplexity,
    )


def _arm_command(
    *,
    tiny_plan: Path,
    arm: ArmSpec,
    out_dir: Path,
    seed: int,
    device: str,
    threads: int,
    binding_flags: Sequence[str],
    runner_args: Sequence[str],
) -> list[str]:
    """Return the argv for one arm's tiny run."""
    return [
        sys.executable,
        "-m",
        "spectraquant",
        "run-plan",
        "--plan",
        str(tiny_plan),
        "--arms",
        arm.name,
        "--out",
        str(out_dir),
        "--seed",
        str(int(seed)),
        "--device",
        device,
        "--threads",
        str(int(threads)),
        "--seq-len",
        str(TINY_SEQ_LEN),
        "--max-tokens",
        str(TINY_MAX_TOKENS),
        "--max-documents",
        str(TINY_MAX_DOCUMENTS),
        "--calibration-documents",
        str(TINY_CALIBRATION_DOCUMENTS),
        *binding_flags,
        *runner_args,
    ]


def dry_run_plan(
    plan: PlanConfig | str | Path,
    *,
    work_dir: str | Path | None = None,
    keep_work_dir: bool = False,
    seed: int | None = None,
    rank: int | None = None,
    bits: int | None = None,
    budget_bytes: int | None = None,
    granularity: str = "per_group",
    group_size: int | None = 128,
    threads: int = DEFAULT_THREADS,
    jobs: int = DEFAULT_JOBS,
    device: str = "cpu",
    timeout_seconds: float = DEFAULT_ARM_TIMEOUT_SECONDS,
    runner_args: Sequence[str] = (),
    runner: ArmCommandRunner | None = None,
    progress: Callable[[str], None] | None = None,
) -> DryRunReport:
    """Execute every arm the plan declares on CPU with tiny budgets and report per arm.

    Args:
        plan: a loaded :class:`~spectraquant.experiment_plan.PlanConfig`, or a plan name/path.
        work_dir: where the tiny plan and the per-arm outputs go; defaults to a fresh temporary
            directory.
        keep_work_dir: keep a temporary work directory instead of removing it at the end (a supplied
            ``work_dir`` is never removed).
        seed: the seed to run; defaults to the plan's first reported seed (which is what the runner
            uses when no ``--seed`` is passed).
        rank: rank for the rank-bound arms; defaults to the plan's grid point closest to
            :data:`PREFERRED_RANK`.
        bits: bit width for the rank-bound arms; defaults to the grid point closest to
            :data:`PREFERRED_BITS`.
        budget_bytes: byte budget for the allocated arms; defaults to the plan's largest ladder rung
            (the most feasible rung, so the dry-run checks the plumbing rather than the solver's
            tightness).
        granularity: quantization granularity passed to every arm.
        group_size: group size for ``per_group``; must be a plan grid point (``None`` with
            ``granularity="per_channel"``).
        threads: ``torch.set_num_threads`` value (:data:`DEFAULT_THREADS`); ``1`` with
            ``jobs=1`` is the sequential, bit-reproducible mode.
        jobs: how many arms run at once (:data:`DEFAULT_JOBS`; ``1`` runs them in plan order, one at
            a time, which is the setting to use when debugging a single arm).
        device: the device the arms run on. ``cpu`` is the only honest local value (``AGENTS.md``
            §2); the runner refuses a CUDA request it cannot satisfy.
        timeout_seconds: per-arm wall-clock ceiling.
        runner_args: extra flags appended to every arm's invocation (each must start with ``--``).
        runner: command runner; injectable so tests exercise the flow without a subprocess.
        progress: callback for human-readable progress (defaults to a logger call).

    Returns:
        The :class:`DryRunReport`; ``report.ok`` is False when any arm failed or was skipped.

    Raises:
        FileNotFoundError: the plan does not exist.
        ValueError: an unknown group size, a malformed ``runner_args`` entry, or an empty plan.
    """
    say = progress if progress is not None else (lambda message: logger.info("%s", message))
    config = plan if isinstance(plan, PlanConfig) else load_plan(plan)
    plan_file = plan_path(plan) if not isinstance(plan, PlanConfig) else None
    if granularity == "per_group":
        if group_size is None:
            raise ValueError("granularity='per_group' requires group_size")
        if int(group_size) not in {int(value) for value in config.grid.group_sizes}:
            raise ValueError(
                f"group_size={group_size} is not a member of plan {config.name!r}'s grid "
                f"{config.grid.group_sizes}: the dry-run binds the same grid the runner validates"
            )
    for argument in runner_args:
        if not argument.startswith("--"):
            raise ValueError(
                f"runner_args entries must be flags starting with '--', got {argument!r}: the "
                "dry-run may not be redirected to an arbitrary program"
            )

    reported = [int(value) for value in config.seeds.reported]
    chosen_seed = reported[0] if seed is None else int(seed)
    if chosen_seed not in reported:
        raise ValueError(
            f"seed {chosen_seed} is not in plan {config.name!r}'s reported list {reported}: a "
            "dry-run may only use the seeds the preregistration fixed"
        )
    resolved_rank = (
        select_grid_point(config.grid.ranks, PREFERRED_RANK) if rank is None else int(rank)
    )
    resolved_bits = (
        select_grid_point(config.grid.bits, PREFERRED_BITS) if bits is None else int(bits)
    )
    resolved_budget = (
        max(int(rung) for rung in config.grid.budget_ladder_bytes)
        if budget_bytes is None
        else int(budget_bytes)
    )

    owned_dir: Path | None = None
    if work_dir is None:
        owned_dir = Path(tempfile.mkdtemp(prefix="spectraquant-dryrun-"))
        root = owned_dir
    else:
        root = Path(work_dir)
        root.mkdir(parents=True, exist_ok=True)
    run_command_fn = runner if runner is not None else run_arm_command

    started = time.perf_counter()
    try:
        tiny_plan = write_tiny_plan(config, root, rank=resolved_rank, bits=resolved_bits)
        say(
            f"dry-run: plan {config.name!r} -> tiny plan {tiny_plan} "
            f"(schedule {TINY_TRAINING}, grid {tiny_grid_document(config, rank=resolved_rank, bits=resolved_bits)}, "
            f"rank {resolved_rank}, bits {resolved_bits}, "
            f"budget {resolved_budget}, granularity {granularity}/{group_size}, device {device})"
        )
        outcomes: list[ArmOutcome] = []

        def execute_arm(arm: ArmSpec) -> ArmOutcome:
            """Run one arm's tiny invocation and check the bundle it produced."""
            out_dir = root / "runs" / arm.name
            if arm.kind not in IMPLEMENTED_ARM_KINDS:
                reason = str(arm_not_implemented_error(arm))
                say(f"dry-run: arm {arm.name!r} cannot run on this slice: {reason}")
                return ArmOutcome(
                    arm=arm.name,
                    kind=arm.kind,
                    status="skipped",
                    detail=f"not implemented in this slice: {reason}",
                    seconds=0.0,
                )
            bindings = arm_binding_args(
                arm,
                rank=resolved_rank,
                bits=resolved_bits,
                budget_bytes=resolved_budget,
            )
            argv = _arm_command(
                tiny_plan=tiny_plan,
                arm=arm,
                out_dir=out_dir,
                seed=chosen_seed,
                device=device,
                threads=threads,
                binding_flags=[
                    "--granularity",
                    granularity,
                    *([] if group_size is None else ["--group-size", str(int(group_size))]),
                    *bindings,
                ],
                runner_args=runner_args,
            )
            say(f"dry-run: arm {arm.name!r} ({arm.kind}) -> {' '.join(argv)}")
            arm_started = time.perf_counter()
            result = run_command_fn(argv, timeout_seconds)
            elapsed = time.perf_counter() - arm_started
            if not result.ok:
                tail = result.combined.strip().splitlines()
                return ArmOutcome(
                    arm=arm.name,
                    kind=arm.kind,
                    status="failed",
                    detail=f"runner exited {result.returncode}: {tail[-1] if tail else 'no output'}",
                    seconds=elapsed,
                    argv=tuple(argv),
                    returncode=int(result.returncode),
                    output_tail=_output_tail(result.combined),
                )
            try:
                detail, manifest, metrics, perplexity = _validate_bundle(
                    arm, out_dir, chosen_seed, _parse_result_line(result.stdout)
                )
            except (AssertionError, ValueError, OSError) as exc:
                return ArmOutcome(
                    arm=arm.name,
                    kind=arm.kind,
                    status="failed",
                    detail=f"bundle check failed: {type(exc).__name__}: {exc}",
                    seconds=elapsed,
                    argv=tuple(argv),
                    returncode=int(result.returncode),
                    output_tail=_output_tail(result.combined),
                )
            return ArmOutcome(
                arm=arm.name,
                kind=arm.kind,
                status="passed",
                detail=detail,
                seconds=elapsed,
                argv=tuple(argv),
                manifest=manifest,
                metrics=metrics,
                perplexity=perplexity,
                returncode=int(result.returncode),
            )

        if int(jobs) <= 1:
            outcomes = [execute_arm(arm) for arm in config.arms]
        else:
            # The arms are independent processes writing to their own directories, so the wall-clock
            # cost of the check is the slowest arm rather than the sum. The pool is bounded because
            # each arm holds the float32 model (and, for the allocation arms, a calibration capture)
            # in RAM, and this workstation has ~15 GiB usable (AGENTS.md §2).
            with ThreadPoolExecutor(max_workers=max(1, min(int(jobs), len(config.arms)))) as pool:
                futures = [
                    (index, pool.submit(execute_arm, arm)) for index, arm in enumerate(config.arms)
                ]
                collected: dict[int, ArmOutcome] = {}
                for index, future in futures:
                    collected[index] = future.result()
            outcomes = [collected[index] for index in range(len(config.arms))]
    finally:
        if owned_dir is not None and not keep_work_dir:
            shutil.rmtree(owned_dir, ignore_errors=True)

    return DryRunReport(
        plan_name=config.name,
        plan_file=str(plan_file) if plan_file is not None else "<loaded PlanConfig>",
        tiny_plan=str(tiny_plan),
        work_dir=str(root),
        seed=chosen_seed,
        device=device,
        outcomes=tuple(outcomes),
        wall_time_s=time.perf_counter() - started,
        settings=(
            ("rank", resolved_rank),
            ("bits", resolved_bits),
            ("budget_bytes", resolved_budget),
            ("granularity", granularity),
            ("group_size", group_size),
            ("threads", int(threads)),
            ("jobs", int(jobs)),
            ("timeout_seconds", float(timeout_seconds)),
            ("tiny_training", dict(TINY_TRAINING)),
        ),
    )
