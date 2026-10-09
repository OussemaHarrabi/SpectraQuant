"""Run-registry validation: a directory of run manifests -> a comparability cell matrix.

Milestone 7 is where validated cloud bundles stop being files and start being a research record.
This module is the gate in front of that record. It walks a runs directory, schema-validates every
manifest it finds through :mod:`spectraquant.reporting.manifests`, indexes the runs by the
**comparability key** the frozen evaluation protocol declares (``eval-protocol.md`` section 8: the
substrate, the hardware generation, the model id + revision, the dataset revisions, the split, the
arm and the compression settings), and then reports, per cell:

* **expected vs present seeds** and the **missing** seeds (the preregistration's seed floor and the
  plan's reported seed list, ``preregistration.md`` sections 6 and 13, recorded by
  ``spectraquant run-plan`` as ``plan.reported_seeds`` / ``plan.seed_floor``);
* **duplicate runs** for the same ``(cell, seed)`` - two *successful* records of one cell at one
  seed, with a note on whether their metric payloads are byte-identical (a replication) or disagree
  (a conflicting measurement);
* **interrupted / aborted** runs (``status`` in ``failed``/``aborted``) and orphan ``metrics.json``
  files left behind by a truncated bundle (the teardown marker of ``design-cloud-adapter.md``
  section 3);
* **schema-invalid or unparseable** files - every one is reported and the pass continues, so a single
  broken record cannot hide the rest of the registry;
* **incomparable** cells - cells of one family that differ in a frozen comparison field and therefore
  must never be pooled, averaged or differenced (``eval-protocol.md`` section 8: "Substrate is part
  of the key");
* the **equal-memory** verdict of every arm pair inside a comparison group, obtained from
  :func:`spectraquant.reporting.comparability.assert_equal_memory` (``AGENTS.md`` section 4.5). A
  refusal is *recorded*, never raised, so one pass lists every problem.

Three keys, deliberately distinct:

=================  ==========================================================================
comparability key  every field that must be identical for two runs to be one cell
cell               one arm at one compression grid point under one substrate/hardware
                   generation: the unit of the seed matrix (runs of a cell differ by seed)
comparison group   a cell key minus ``(arm, compression)``: the runs a comparison may draw its
                   arms from, and therefore the runs whose arms are gated at equal memory
family             a cell key minus the comparison *axes* (substrate, hardware, runtime backend,
                   model revision, dataset revisions/checksums): two cells of one family that
                   differ in an axis are reported incomparable
=================  ==========================================================================

**Fixtures.** A manifest is a *fixture* - not a run - when its own document declares it
(``resolved_config.fixture.not_a_real_run == true``, the marker the committed equal-memory fixtures
carry). A fixture is never a cell, never contributes a seed, a duplicate or an incomparable group,
and never fails the gate: it is listed, and its arm pairs are still gated so the mechanism is
exercised on committed data - at *warning* severity, because the preregistration says a fixture is
never a result (``preregistration.md`` section 8). A directory whose manifests are all fixtures is
reported as fixture-only rather than as a missing cell.

**Blocking vs warning.** Blocking (a non-zero CLI exit) is exactly the set the Milestone-7 contract
names: missing seeds for a declared cell, duplicate runs, schema-invalid or unparseable files, and an
unequal-memory arm pair - the latter meaning the gate *applied a tolerance and the byte gap exceeded
it*. A pair the gate refuses earlier (no declared byte source, no declared tolerance, or two
different byte sources or serializers) is a **warning**: such a pair is not a declared equal-memory
comparison at all, which is reported verbatim but is not evidence about memory. Everything else
(interrupted runs, incomparable cells, orphan metrics files, a declared-but-conflicting seed list, a
missing runs directory) is a warning too.

The JSON index is deterministic: every list is sorted on a stable key and every mapping is emitted
with sorted keys, so the same directory always produces byte-identical JSON.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from spectraquant.reporting.comparability import (
    UnequalMemoryComparison,
    assert_equal_memory,
)
from spectraquant.reporting.manifests import (
    CompressionBlock,
    HardwareBlock,
    ManifestValidationError,
    RunManifest,
    load_schema,
    validate_manifest_dict,
)

__all__ = [
    "AXES",
    "BLOCKING_CODES",
    "CLOUD_MANIFEST_FILENAME",
    "CODE_DUPLICATE_RUN",
    "CODE_INCOMPARABLE",
    "CODE_INTERRUPTED_RUN",
    "CODE_MEMORY_GATE_REFUSED",
    "CODE_MISSING_SEEDS",
    "CODE_ORPHAN_METRICS",
    "CODE_RUNS_DIR_MISSING",
    "CODE_SCHEMA_INVALID",
    "CODE_SEED_DECLARATION_CONFLICT",
    "CODE_UNEQUAL_MEMORY",
    "CODE_UNPARSEABLE",
    "MANIFEST_GLOBS",
    "MANIFEST_SUFFIX",
    "SEVERITY_BLOCKING",
    "SEVERITY_WARNING",
    "CellRecord",
    "ComparabilityKey",
    "CompressionKey",
    "DirectoryRecord",
    "DuplicateGroup",
    "FamilyRecord",
    "Finding",
    "HardwareKey",
    "RegistryReport",
    "RunRecord",
    "comparability_key",
    "validate_registry",
]

#: Revision of the JSON index this module emits.
SCHEMA_VERSION = 1

#: File-name suffix that identifies a run manifest: ``<run_id>.manifest.json`` (the committed sample
#: layout), ``smoke-manifest.json``, and ``run_manifest.json`` (the cloud artifact-bundle layout,
#: ``design-cloud-adapter.md`` section 5; the plan runner writes ``<out>/<arm>/seed-<n>/run_manifest.json``).
MANIFEST_SUFFIX = "manifest.json"

#: Shell-glob view of :data:`MANIFEST_SUFFIX`, for documentation and doctors.
MANIFEST_GLOBS: tuple[str, ...] = ("*manifest.json",)

#: Manifest file name inside a downloaded cloud bundle (``design-cloud-adapter.md`` section 5).
CLOUD_MANIFEST_FILENAME = "run_manifest.json"

#: Per-run metrics file the plan runner writes beside its manifest
#: (``design-cloud-adapter.md`` section 11.3).
METRICS_FILENAME = "metrics.json"

#: ``status`` value of a run that may contribute a result.
SUCCESS = "success"

#: ``status`` values that mean the run stopped without a result
#: (``preregistration.md`` section 10.4).
INTERRUPTED_STATES: frozenset[str] = frozenset({"failed", "aborted"})

SEVERITY_BLOCKING = "blocking"
SEVERITY_WARNING = "warning"
Severity = Literal["blocking", "warning"]

CODE_MISSING_SEEDS = "missing_seeds"
CODE_DUPLICATE_RUN = "duplicate_run"
CODE_SCHEMA_INVALID = "schema_invalid"
CODE_UNPARSEABLE = "unparseable_manifest"
CODE_UNEQUAL_MEMORY = "unequal_memory"
CODE_MEMORY_GATE_REFUSED = "memory_gate_refused"
CODE_INCOMPARABLE = "incomparable_cells"
CODE_INTERRUPTED_RUN = "interrupted_run"
CODE_ORPHAN_METRICS = "orphan_metrics"
CODE_SEED_DECLARATION_CONFLICT = "conflicting_seed_declaration"
CODE_RUNS_DIR_MISSING = "runs_dir_missing"

#: The finding codes that fail the gate (Milestone-7 contract).
BLOCKING_CODES: frozenset[str] = frozenset(
    {
        CODE_MISSING_SEEDS,
        CODE_DUPLICATE_RUN,
        CODE_SCHEMA_INVALID,
        CODE_UNPARSEABLE,
        CODE_UNEQUAL_MEMORY,
    }
)

#: Directories never descended into while scanning (a runs directory may sit inside a checkout).
_PRUNED_DIRECTORIES: frozenset[str] = frozenset(
    {
        ".git",
        ".hypothesis",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "node_modules",
        "venv",
    }
)

#: The frozen comparison axes: a difference in any of these between two cells of one family makes
#: them incomparable (``eval-protocol.md`` section 8).
AXES: tuple[str, ...] = (
    "substrate",
    "hardware",
    "runtime_backend",
    "model_revision",
    "datasets",
)


# --------------------------------------------------------------------------------------
# Canonical serialization
# --------------------------------------------------------------------------------------
def _canonical(value: Any) -> str:
    """Return the canonical JSON text of ``value`` (sorted keys, compact separators)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _digest(value: Any) -> str:
    """Return the sha256 hex digest of the canonical JSON text of ``value``."""
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------
# The comparability key
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class HardwareKey:
    """The recorded hardware generation a run executed on (``eval-protocol.md`` section 8).

    ``ram_total_mb`` and ``cpu_count_logical`` are deliberately excluded: they describe the *host
    instance*, not the hardware generation, and a cloud scheduler legitimately changes them between
    two arms. ``torch_threads`` is kept, because the protocol pins ``OMP_NUM_THREADS`` per substrate
    and records it in the manifest.
    """

    platform: str
    system: str
    release: str
    machine: str
    processor: str | None
    gpu_devices: tuple[str, ...]
    torch_device: str
    torch_threads: int

    @classmethod
    def from_block(cls, hardware: HardwareBlock) -> HardwareKey:
        """Build the key from a manifest's validated ``hardware`` block."""
        return cls(
            platform=hardware.platform,
            system=hardware.system,
            release=hardware.release,
            machine=hardware.machine,
            processor=hardware.processor,
            gpu_devices=tuple(sorted(str(device) for device in hardware.gpu_devices)),
            torch_device=hardware.torch_device,
            torch_threads=int(hardware.torch_threads),
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable document."""
        return {
            "platform": self.platform,
            "system": self.system,
            "release": self.release,
            "machine": self.machine,
            "processor": self.processor,
            "gpu_devices": list(self.gpu_devices),
            "torch_device": self.torch_device,
            "torch_threads": self.torch_threads,
        }


@dataclass(frozen=True)
class CompressionKey:
    """The compression *settings* a run applied - not the bytes it measured.

    The byte figures (``accounted_bytes``, ``measured_bytes``, ...) are outcomes, not settings: two
    runs of one arm at one grid point that report different stored bytes are the same cell with
    conflicting measurements (reported as a duplicate), while the equal-memory gate is what decides
    whether either may be compared with another arm.
    """

    method: str
    bits: int | None
    group_size: int | None
    ranks: tuple[int, ...] | None
    exclusions: tuple[str, ...]
    serializer: str | None
    bytes_source: str | None

    @classmethod
    def from_block(cls, compression: CompressionBlock) -> CompressionKey:
        """Build the key from a manifest's validated ``compression`` block."""
        ranks = (
            None if compression.ranks is None else tuple(int(rank) for rank in compression.ranks)
        )
        return cls(
            method=compression.method,
            bits=compression.bits,
            group_size=compression.group_size,
            ranks=ranks,
            exclusions=tuple(sorted(str(item) for item in compression.exclusions)),
            serializer=compression.serializer,
            bytes_source=compression.bytes_source,
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable document."""
        return {
            "method": self.method,
            "bits": self.bits,
            "group_size": self.group_size,
            "ranks": None if self.ranks is None else list(self.ranks),
            "exclusions": list(self.exclusions),
            "serializer": self.serializer,
            "bytes_source": self.bytes_source,
        }


@dataclass(frozen=True)
class ComparabilityKey:
    """Every frozen field that decides whether two runs are the same cell (``eval-protocol.md`` 8).

    Two runs with equal keys are the same cell and may differ only in their seed. The key is built
    from recorded values only: nothing is inferred, and a field the manifest does not record (the
    plan substrate of a non-plan run) is recorded as ``"undeclared"`` rather than filled with a
    guess.
    """

    substrate: str
    hardware: HardwareKey
    runtime_backend: str
    model_id: str | None
    model_revision: str | None
    dataset_ids: tuple[str, ...]
    dataset_revisions: tuple[str | None, ...]
    dataset_checksums: tuple[str, ...]
    split: str
    arm: str | None
    compression: CompressionKey

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable document."""
        return {
            "substrate": self.substrate,
            "hardware": self.hardware.to_json_dict(),
            "runtime_backend": self.runtime_backend,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "dataset_ids": list(self.dataset_ids),
            "dataset_revisions": list(self.dataset_revisions),
            "dataset_checksums": list(self.dataset_checksums),
            "split": self.split,
            "arm": self.arm,
            "compression": self.compression.to_json_dict(),
        }

    @property
    def digest(self) -> str:
        """Stable 16-hex identifier of this cell (the cell key)."""
        return _digest(self.to_json_dict())[:16]

    @property
    def family(self) -> tuple[str, ...]:
        """The estimand identity: everything except the frozen comparison axes."""
        return (
            self.model_id or "<local>",
            "|".join(self.dataset_ids),
            self.split,
            self.arm or "<no-arm>",
            _canonical(self.compression.to_json_dict()),
        )

    @property
    def family_digest(self) -> str:
        """Stable 16-hex identifier of the family this cell belongs to."""
        return _digest(list(self.family))[:16]

    @property
    def comparison_group(self) -> tuple[str, ...]:
        """The cell key minus ``(arm, compression)``: the runs a comparison draws arms from."""
        return (
            self.substrate,
            _canonical(self.hardware.to_json_dict()),
            self.runtime_backend,
            self.model_id or "<local>",
            self.model_revision or "<none>",
            "|".join(self.dataset_ids),
            _canonical(list(self.dataset_revisions)),
            _canonical(list(self.dataset_checksums)),
            self.split,
        )

    @property
    def group_digest(self) -> str:
        """Stable 16-hex identifier of the comparison group this cell belongs to."""
        return _digest(list(self.comparison_group))[:16]

    def axes(self) -> dict[str, str]:
        """Return the frozen comparison axes and the recorded value of each."""
        return {
            "substrate": self.substrate,
            "hardware": _canonical(self.hardware.to_json_dict()),
            "runtime_backend": self.runtime_backend,
            "model_revision": self.model_revision or "<none>",
            "datasets": _canonical([list(self.dataset_revisions), list(self.dataset_checksums)]),
        }

    def label(self) -> str:
        """Short human-readable label: ``<split>/<arm>/<model>@<rev>/<method>``."""
        model = self.model_id or "<local>"
        revision = (self.model_revision or "-")[:12]
        return f"{self.split}/{self.arm or '<no-arm>'}/{model}@{revision}/{self.compression.method}"

    def hardware_label(self) -> str:
        """Short human-readable substrate/hardware label for the table."""
        devices = ",".join(self.hardware.gpu_devices) or "no-gpu"
        return f"{self.substrate}/{self.hardware.torch_device}({devices})"


def comparability_key(manifest: RunManifest) -> ComparabilityKey:
    """Build the :class:`ComparabilityKey` of one validated manifest.

    Args:
        manifest: a schema-validated manifest (``reporting/manifests.py``).

    Returns:
        The cell key. The substrate is the plan substrate the manifest records
        (``metrics["plan.substrate"]``, written by ``spectraquant run-plan``); a manifest that
        records none is ``"undeclared"``, never guessed from the hardware. The arm is
        ``metrics["plan.arm"]`` when recorded. The expected seed set
        (``metrics["plan.reported_seeds"]``) and the seed floor (``metrics["plan.seed_floor"]``) are
        *not* part of the key: they are the cell's declaration, checked for agreement across its
        runs.
    """
    metrics = manifest.metrics
    declared_substrate = metrics.get("plan.substrate")
    declared_arm = metrics.get("plan.arm")
    substrate = declared_substrate if isinstance(declared_substrate, str) else ""
    arm = declared_arm if isinstance(declared_arm, str) else ""
    return ComparabilityKey(
        substrate=substrate or "undeclared",
        hardware=HardwareKey.from_block(manifest.hardware),
        runtime_backend=manifest.runtime_backend,
        model_id=manifest.model_id,
        model_revision=manifest.model_revision,
        dataset_ids=tuple(str(item) for item in manifest.dataset_ids),
        dataset_revisions=tuple(manifest.dataset_revisions),
        dataset_checksums=tuple(str(item) for item in manifest.dataset_checksums),
        split=manifest.split,
        arm=arm or None,
        compression=CompressionKey.from_block(manifest.compression),
    )


# --------------------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RunRecord:
    """One schema-valid manifest: what the registry knows about a single run."""

    run_id: str
    path: str
    seed: int
    status: str
    measurement_class: int | None
    expected_seeds: tuple[int, ...] | None
    seed_floor: int | None
    is_fixture: bool
    plan_measurement: bool | None
    failure_reason: str | None
    metrics_digest: str
    key: ComparabilityKey
    #: The validated manifest, kept for the equal-memory gate (never serialized).
    manifest: RunManifest = field(repr=False, compare=False)

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable document."""
        return {
            "run_id": self.run_id,
            "path": self.path,
            "seed": self.seed,
            "status": self.status,
            "measurement_class": self.measurement_class,
            "expected_seeds": None if self.expected_seeds is None else list(self.expected_seeds),
            "seed_floor": self.seed_floor,
            "fixture": self.is_fixture,
            "plan_measurement": self.plan_measurement,
            "failure_reason": self.failure_reason,
            "metrics_digest": self.metrics_digest,
            "cell_key": self.key.digest,
        }


@dataclass(frozen=True)
class DuplicateGroup:
    """Two or more *successful* runs of one ``(cell, seed)``."""

    seed: int
    run_ids: tuple[str, ...]
    identical_metrics: bool

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable document."""
        return {
            "seed": self.seed,
            "run_ids": list(self.run_ids),
            "identical_metrics": self.identical_metrics,
        }


@dataclass(frozen=True)
class Finding:
    """One problem the pass found. Never raised: collected, so one pass lists every problem."""

    code: str
    severity: Severity
    detail: str
    cell: str | None = None
    axis: str | None = None
    runs: tuple[str, ...] = ()
    fixture_only: bool = False

    @property
    def blocking(self) -> bool:
        """True when this finding must fail the gate."""
        return self.severity == SEVERITY_BLOCKING

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable document."""
        return {
            "code": self.code,
            "severity": self.severity,
            "detail": self.detail,
            "cell": self.cell,
            "axis": self.axis,
            "runs": list(self.runs),
            "fixture_only": self.fixture_only,
        }


@dataclass(frozen=True)
class CellRecord:
    """One cell: its runs, its seed matrix, its duplicates and its incomparable axes.

    The equal-memory verdicts are not stored here: a verdict is about a *pair* of cells, so it lives
    in :attr:`RegistryReport.equal_memory` (which names both cells) and in the findings, not on one
    of the two cells.
    """

    key: ComparabilityKey
    runs: tuple[RunRecord, ...]
    expected_seeds: tuple[int, ...] | None
    seed_floor: int | None
    present_seeds: tuple[int, ...]
    missing_seeds: tuple[int, ...]
    duplicates: tuple[DuplicateGroup, ...]
    incomparable_axes: tuple[str, ...]

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable document."""
        return {
            "cell_key": self.key.digest,
            "identity": self.key.to_json_dict(),
            "label": self.key.label(),
            "group_key": self.key.group_digest,
            "family_key": self.key.family_digest,
            "expected_seeds": None if self.expected_seeds is None else list(self.expected_seeds),
            "seed_floor": self.seed_floor,
            "present_seeds": list(self.present_seeds),
            "missing_seeds": list(self.missing_seeds),
            "duplicates": [group.to_json_dict() for group in self.duplicates],
            "incomparable_axes": list(self.incomparable_axes),
            "runs": [run.to_json_dict() for run in self.runs],
        }


@dataclass(frozen=True)
class FamilyRecord:
    """One family of cells and the axes on which its cells disagree."""

    family_key: str
    label: str
    cells: tuple[str, ...]
    divergent_axes: dict[str, list[str]]

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable document."""
        return {
            "family_key": self.family_key,
            "label": self.label,
            "cells": list(self.cells),
            "divergent_axes": {
                axis: list(values) for axis, values in sorted(self.divergent_axes.items())
            },
        }


@dataclass(frozen=True)
class DirectoryRecord:
    """What one directory under the runs root contributed."""

    path: str
    manifests: int
    fixtures: int
    cells: int
    fixture_only: bool

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable document."""
        return {
            "path": self.path,
            "manifests": self.manifests,
            "fixtures": self.fixtures,
            "cells": self.cells,
            "fixture_only": self.fixture_only,
        }


@dataclass(frozen=True)
class RegistryReport:
    """The whole pass: cells, fixtures, families, directories and findings."""

    runs_dir: str
    files: tuple[str, ...]
    runs: tuple[RunRecord, ...]
    fixtures: tuple[RunRecord, ...]
    cells: tuple[CellRecord, ...]
    families: tuple[FamilyRecord, ...]
    directories: tuple[DirectoryRecord, ...]
    equal_memory: tuple[dict[str, Any], ...]
    findings: tuple[Finding, ...]

    @property
    def blocking(self) -> tuple[Finding, ...]:
        """The findings that fail the gate."""
        return tuple(finding for finding in self.findings if finding.blocking)

    @property
    def warnings(self) -> tuple[Finding, ...]:
        """The findings that are reported without failing the gate."""
        return tuple(finding for finding in self.findings if not finding.blocking)

    @property
    def ok(self) -> bool:
        """True when the gate passes (warnings only)."""
        return not self.blocking

    def summary_line(self) -> str:
        """One-line human-readable verdict."""
        return (
            f"{len(self.cells)} cell(s), {len(self.runs)} run(s), {len(self.fixtures)} fixture(s): "
            f"{len(self.blocking)} blocking, {len(self.warnings)} warning(s)"
        )

    def cell_rows(self) -> list[list[str]]:
        """Table rows, one per cell: label, substrate, seeds present/expected, missing, runs, dupes."""
        counts: dict[str, int] = {}
        for finding in self.findings:
            if finding.cell is not None:
                counts[finding.cell] = counts.get(finding.cell, 0) + 1
        rows: list[list[str]] = []
        for cell in sorted(self.cells, key=lambda item: item.key.digest):
            expected = "?" if cell.expected_seeds is None else str(len(cell.expected_seeds))
            problems = counts.get(cell.key.digest, 0)
            rows.append(
                [
                    cell.key.label(),
                    cell.key.hardware_label(),
                    f"{len(cell.present_seeds)}/{expected}",
                    ",".join(str(seed) for seed in cell.missing_seeds) or "-",
                    str(len(cell.runs)),
                    str(len(cell.duplicates)),
                    str(problems) if problems else "-",
                ]
            )
        return rows

    def finding_rows(self) -> list[list[str]]:
        """Table rows, one per finding: severity, code, detail."""
        return [[finding.severity, finding.code, finding.detail] for finding in self.findings]

    def fixture_rows(self) -> list[list[str]]:
        """Table rows, one per declared fixture: path, run id, cell key."""
        return [
            [run.path, run.run_id, run.key.digest]
            for run in sorted(self.fixtures, key=lambda item: item.path)
        ]

    def to_json_dict(self) -> dict[str, Any]:
        """Return the deterministic JSON index (sorted keys make it byte-stable)."""
        return {
            "schema_version": SCHEMA_VERSION,
            "tool": "spectraquant registry-validate",
            "runs_dir": self.runs_dir,
            "summary": {
                "files": len(self.files),
                "runs": len(self.runs),
                "fixtures": len(self.fixtures),
                "cells": len(self.cells),
                "families": len(self.families),
                "blocking": len(self.blocking),
                "warnings": len(self.warnings),
                "ok": self.ok,
            },
            "files": list(self.files),
            "directories": [record.to_json_dict() for record in self.directories],
            "cells": [cell.to_json_dict() for cell in self.cells],
            "families": [family.to_json_dict() for family in self.families],
            "fixtures": [run.to_json_dict() for run in self.fixtures],
            "equal_memory": list(self.equal_memory),
            "findings": [finding.to_json_dict() for finding in self.findings],
            "blocking": not self.ok,
        }


# --------------------------------------------------------------------------------------
# Discovery and ingest
# --------------------------------------------------------------------------------------
def _discover(root: Path) -> tuple[list[Path], set[Path], set[Path]]:
    """Walk ``root``; return ``(manifests, dirs_with_metrics, dirs_with_manifests)``, sorted."""
    manifests: list[Path] = []
    metrics_dirs: set[Path] = set()
    manifest_dirs: set[Path] = set()
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            entries = sorted(directory.iterdir(), key=lambda entry: entry.name)
        except OSError:  # pragma: no cover - unreadable directory
            continue
        for entry in entries:
            if entry.is_dir():
                if entry.name not in _PRUNED_DIRECTORIES:
                    stack.append(entry)
                continue
            if not entry.is_file():
                continue
            if entry.name == METRICS_FILENAME:
                metrics_dirs.add(entry.parent)
            elif entry.name.endswith(MANIFEST_SUFFIX):
                manifests.append(entry)
                manifest_dirs.add(entry.parent)
    manifests.sort(key=lambda path: path.as_posix())
    return manifests, metrics_dirs, manifest_dirs


def _load_candidate(
    path: Path, schema: dict[str, Any]
) -> tuple[dict[str, Any] | None, RunManifest | None, str | None, bool]:
    """Read and validate one candidate file.

    Returns ``(document, manifest, error, unparseable)``. Exactly one of ``manifest`` / ``error`` is
    set; ``unparseable`` distinguishes "not readable JSON" from "JSON that violates the schema".
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return None, None, f"{type(exc).__name__}: {exc}", True
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, None, f"invalid JSON: {exc}", True
    if not isinstance(document, dict):
        return None, None, "manifest must be a JSON object", True
    try:
        validate_manifest_dict(document, schema=schema, source=str(path))
    except ManifestValidationError as exc:
        messages = exc.errors[:3]
        tail = "" if len(exc.errors) <= 3 else f" (+{len(exc.errors) - 3} more)"
        return None, None, "; ".join(messages) + tail, False
    try:
        manifest = RunManifest.model_validate(document)
    except ValidationError as exc:  # pragma: no cover - the schema already accepted the document
        return None, None, f"typed model rejected the document: {exc.errors()[:2]}", False
    return document, manifest, None, False


def _is_fixture(document: Mapping[str, Any]) -> bool:
    """True when the manifest declares itself a fixture rather than a run."""
    resolved = document.get("resolved_config")
    if not isinstance(resolved, Mapping):
        return False
    marker = resolved.get("fixture")
    if not isinstance(marker, Mapping):
        return False
    return bool(marker.get("not_a_real_run"))


def _int_tuple(value: Any) -> tuple[int, ...] | None:
    """Coerce a recorded seed list into a sorted tuple of ints (``None`` when unrecorded)."""
    if not isinstance(value, (list, tuple)) or not value:
        return None
    try:
        return tuple(sorted({int(item) for item in value}))
    except (TypeError, ValueError):
        return None


def _seed_declaration(metrics: Mapping[str, Any]) -> tuple[tuple[int, ...] | None, int | None]:
    """Return ``(expected_seeds, seed_floor)`` as the run's plan recorded them."""
    expected = _int_tuple(metrics.get("plan.reported_seeds"))
    raw_floor = metrics.get("plan.seed_floor")
    try:
        floor = int(raw_floor) if raw_floor is not None else None
    except (TypeError, ValueError):
        floor = None
    return expected, floor


def _run_record(manifest: RunManifest, *, path: str, is_fixture: bool) -> RunRecord:
    """Build the :class:`RunRecord` for one validated manifest."""
    expected, floor = _seed_declaration(manifest.metrics)
    plan_measurement = manifest.metrics.get("substitution.is_plan_measurement")
    return RunRecord(
        run_id=manifest.run_id,
        path=path,
        seed=int(manifest.seed),
        status=manifest.status,
        measurement_class=manifest.measurement_class,
        expected_seeds=expected,
        seed_floor=floor,
        is_fixture=is_fixture,
        plan_measurement=plan_measurement if isinstance(plan_measurement, bool) else None,
        failure_reason=manifest.failure_reason,
        metrics_digest=_digest(manifest.metrics),
        key=comparability_key(manifest),
        manifest=manifest,
    )


def _severity(code: str, runs: Sequence[RunRecord]) -> tuple[Severity, bool]:
    """Return ``(severity, fixture_only)`` for ``code`` over ``runs``.

    A blocking code over fixture-only runs is reported at warning severity: a fixture is never a
    result (``preregistration.md`` section 8), so it may not fail the Milestone-7 gate.
    """
    fixture_only = bool(runs) and all(run.is_fixture for run in runs)
    if code in BLOCKING_CODES:
        return (SEVERITY_WARNING if fixture_only else SEVERITY_BLOCKING), fixture_only
    return SEVERITY_WARNING, fixture_only


def _finding_sort_key(finding: Finding) -> tuple[int, str, str, str]:
    """Deterministic order: blocking first, then code, cell and detail."""
    return (0 if finding.blocking else 1, finding.code, finding.cell or "", finding.detail)


# --------------------------------------------------------------------------------------
# The pass
# --------------------------------------------------------------------------------------
def validate_registry(runs_dir: Path | str) -> RegistryReport:
    """Validate a directory of run manifests and return the cell matrix plus every problem found.

    Args:
        runs_dir: directory walked for ``*.manifest.json`` / ``run_manifest.json`` files. A missing
            directory is a warning, not an error: the gate must be runnable before the first cloud
            bundle lands (``design-cloud-adapter.md`` section 5).

    Returns:
        A :class:`RegistryReport`. Nothing raises for a bad manifest: unreadable, unparseable and
        schema-invalid files each become a finding and the pass continues, so one pass lists every
        problem.
    """
    root = Path(runs_dir)
    findings: list[Finding] = []

    def shown(path: Path) -> str:
        try:
            return path.relative_to(root).as_posix()
        except ValueError:  # pragma: no cover - every scanned path is under root
            return path.as_posix()

    if not root.exists():
        findings.append(
            Finding(
                code=CODE_RUNS_DIR_MISSING,
                severity=SEVERITY_WARNING,
                detail=(
                    f"no runs directory at {root.as_posix()}: nothing to validate. Point --runs at "
                    "the directory the cloud bundles are collected into "
                    "(spectraquant cloud fetch --dest <dir>)."
                ),
            )
        )
        return RegistryReport(
            runs_dir=root.as_posix(),
            files=(),
            runs=(),
            fixtures=(),
            cells=(),
            families=(),
            directories=(),
            equal_memory=(),
            findings=tuple(findings),
        )

    schema = load_schema()
    paths, metrics_dirs, manifest_dirs = _discover(root)
    files = tuple(shown(path) for path in paths)

    records: list[RunRecord] = []
    for path in paths:
        document, manifest, error, unparseable = _load_candidate(path, schema)
        if error is not None or document is None or manifest is None:
            findings.append(
                Finding(
                    code=CODE_UNPARSEABLE if unparseable else CODE_SCHEMA_INVALID,
                    severity=SEVERITY_BLOCKING,
                    detail=f"{shown(path)}: {error}",
                )
            )
            continue
        records.append(_run_record(manifest, path=shown(path), is_fixture=_is_fixture(document)))

    for directory in sorted(metrics_dirs - manifest_dirs, key=lambda item: item.as_posix()):
        if directory == root:
            continue
        findings.append(
            Finding(
                code=CODE_ORPHAN_METRICS,
                severity=SEVERITY_WARNING,
                detail=(
                    f"{shown(directory)}: {METRICS_FILENAME} present without a run manifest. An "
                    "interrupted or truncated bundle, or the invocation-level aggregate of a plan "
                    "run (design-cloud-adapter.md section 11.3) - check before treating it as a run."
                ),
                runs=(shown(directory),),
            )
        )

    for record in records:
        if record.status != SUCCESS:
            findings.append(
                Finding(
                    code=CODE_INTERRUPTED_RUN,
                    severity=SEVERITY_WARNING,
                    detail=(
                        f"{record.path}: run {record.run_id} (seed {record.seed}) ended with status "
                        f"{record.status!r}: "
                        f"{record.failure_reason or 'no failure reason recorded'}"
                    ),
                    cell=record.key.digest,
                    runs=(record.run_id,),
                )
            )

    fixtures = tuple(sorted((run for run in records if run.is_fixture), key=lambda r: r.path))
    runs = tuple(
        sorted((run for run in records if not run.is_fixture), key=lambda r: (r.seed, r.path))
    )

    cells, cell_findings, verdicts = _build_cells(runs)
    families, updated_cells, family_findings = _build_families(cells)
    findings.extend(cell_findings)
    findings.extend(family_findings)
    fixture_groups: dict[str, list[RunRecord]] = {}
    for fixture in fixtures:
        fixture_groups.setdefault(fixture.key.group_digest, []).append(fixture)
    fixture_verdicts: list[dict[str, Any]] = []
    for group_key in sorted(fixture_groups):
        group_findings, group_verdicts = _gate_group(
            fixture_groups[group_key], skip_same_cell=False, collapse_refusals=True
        )
        findings.extend(group_findings)
        fixture_verdicts.extend(group_verdicts)

    return RegistryReport(
        runs_dir=root.as_posix(),
        files=files,
        runs=runs,
        fixtures=fixtures,
        cells=updated_cells,
        families=families,
        directories=_directory_records(records, updated_cells),
        equal_memory=tuple(verdicts) + tuple(fixture_verdicts),
        findings=tuple(sorted(findings, key=_finding_sort_key)),
    )


def _build_cells(
    runs: Sequence[RunRecord],
) -> tuple[tuple[CellRecord, ...], list[Finding], list[dict[str, Any]]]:
    """Group runs into cells, check their seed matrices, and gate arm pairs at equal memory."""
    by_cell: dict[str, list[RunRecord]] = {}
    groups: dict[str, list[RunRecord]] = {}
    for run in runs:
        by_cell.setdefault(run.key.digest, []).append(run)
        groups.setdefault(run.key.group_digest, []).append(run)

    findings: list[Finding] = []
    cells: list[CellRecord] = []
    for cell_key in sorted(by_cell):
        cell_runs = tuple(sorted(by_cell[cell_key], key=lambda run: (run.seed, run.path)))
        key = cell_runs[0].key
        expected, expected_conflict = _agreeing(cell_runs, "expected_seeds")
        floor, floor_conflict = _agreeing(cell_runs, "seed_floor")
        present = tuple(sorted({run.seed for run in cell_runs if run.status == SUCCESS}))
        missing = (
            () if expected is None else tuple(seed for seed in expected if seed not in present)
        )

        if expected_conflict or floor_conflict:
            declared = sorted(
                {run.expected_seeds for run in cell_runs if run.expected_seeds is not None}
            )
            floors = sorted({run.seed_floor for run in cell_runs if run.seed_floor is not None})
            findings.append(
                Finding(
                    code=CODE_SEED_DECLARATION_CONFLICT,
                    severity=SEVERITY_WARNING,
                    detail=(
                        f"{key.label()}: the runs of this cell disagree about the declared seed set "
                        f"({declared} expected, floor {floors}); no missing-seed check is possible "
                        "until they agree"
                    ),
                    cell=cell_key,
                    runs=tuple(run.run_id for run in cell_runs),
                )
            )

        if missing or (expected is None and floor is not None and len(present) < floor):
            if missing:
                detail = (
                    f"{key.label()}: declared seeds {list(expected or ())} but only "
                    f"{list(present)} have a successful run; missing {list(missing)}"
                )
            else:
                detail = (
                    f"{key.label()}: the declared seed floor is {floor} but only {len(present)} "
                    f"successful seed(s) are present ({list(present)})"
                )
            severity, fixture_only = _severity(CODE_MISSING_SEEDS, cell_runs)
            findings.append(
                Finding(
                    code=CODE_MISSING_SEEDS,
                    severity=severity,
                    detail=detail,
                    cell=cell_key,
                    runs=tuple(run.run_id for run in cell_runs),
                    fixture_only=fixture_only,
                )
            )

        duplicates = _duplicate_groups(cell_runs)
        for duplicate in duplicates:
            severity, fixture_only = _severity(CODE_DUPLICATE_RUN, cell_runs)
            findings.append(
                Finding(
                    code=CODE_DUPLICATE_RUN,
                    severity=severity,
                    detail=(
                        f"{key.label()} seed {duplicate.seed}: {len(duplicate.run_ids)} successful "
                        f"runs of the same cell and seed ({', '.join(duplicate.run_ids)}); "
                        + (
                            "their metric payloads are byte-identical (a replication)"
                            if duplicate.identical_metrics
                            else "their metric payloads DIFFER (a conflicting measurement)"
                        )
                    ),
                    cell=cell_key,
                    runs=duplicate.run_ids,
                    fixture_only=fixture_only,
                )
            )

        cells.append(
            CellRecord(
                key=key,
                runs=cell_runs,
                expected_seeds=expected,
                seed_floor=floor,
                present_seeds=present,
                missing_seeds=missing,
                duplicates=duplicates,
                incomparable_axes=(),
            )
        )

    verdicts: list[dict[str, Any]] = []
    for group_key in sorted(groups):
        group_findings, group_verdicts = _gate_group(
            groups[group_key], skip_same_cell=True, collapse_refusals=True
        )
        verdicts.extend(group_verdicts)
        findings.extend(group_findings)

    return tuple(cells), findings, verdicts


def _agreeing(runs: Sequence[RunRecord], attribute: str) -> tuple[Any, bool]:
    """Return ``(value, conflicting)`` for a per-run declaration the cell must agree on."""
    values = {getattr(run, attribute) for run in runs if getattr(run, attribute) is not None}
    if not values:
        return None, False
    if len(values) > 1:
        return None, True
    return next(iter(values)), False


def _duplicate_groups(runs: Sequence[RunRecord]) -> tuple[DuplicateGroup, ...]:
    """Return one group per ``(cell, seed)`` with more than one successful run."""
    by_seed: dict[int, list[RunRecord]] = {}
    for run in runs:
        if run.status == SUCCESS:
            by_seed.setdefault(run.seed, []).append(run)
    groups: list[DuplicateGroup] = []
    for seed in sorted(by_seed):
        successful = sorted(by_seed[seed], key=lambda run: run.run_id)
        if len(successful) > 1:
            groups.append(
                DuplicateGroup(
                    seed=seed,
                    run_ids=tuple(run.run_id for run in successful),
                    identical_metrics=len({run.metrics_digest for run in successful}) == 1,
                )
            )
    return tuple(groups)


@dataclass(frozen=True)
class _MemoryOutcome:
    """What the equal-memory gate said about one pair (a refusal carries the figures)."""

    verdict: dict[str, Any] | None = None
    refusal: str | None = None
    blocking: bool = False
    reason: str | None = None
    bytes_a: int | None = None
    bytes_b: int | None = None
    difference_bytes: int | None = None
    tolerance_bytes: int | None = None
    bytes_source: str | None = None


def _memory_outcome(run_a: RunRecord, run_b: RunRecord) -> _MemoryOutcome:
    """Gate one pair and classify the outcome.

    ``blocking`` is true only when the gate resolved both byte figures, applied a tolerance and found
    the gap wider than it: that is the ``unequal_memory`` case. Any earlier refusal (undeclared byte
    source, no declared tolerance, different sources or serializers, or a malformed tolerance
    declaration) is reported verbatim and treated as a warning.
    """
    try:
        verdict = assert_equal_memory(run_a.manifest, run_b.manifest)
    except UnequalMemoryComparison as exc:
        return _MemoryOutcome(
            refusal=str(exc),
            blocking=exc.tolerance_bytes is not None,
            reason=exc.reason,
            bytes_a=exc.bytes_a,
            bytes_b=exc.bytes_b,
            difference_bytes=exc.difference_bytes,
            tolerance_bytes=exc.tolerance_bytes,
            bytes_source=exc.bytes_source,
        )
    except ValueError as exc:  # pragma: no cover - a malformed tolerance declaration
        return _MemoryOutcome(refusal=f"{type(exc).__name__}: {exc}")
    return _MemoryOutcome(verdict=verdict.to_json_dict())


def _gate_group(
    runs: Sequence[RunRecord], *, skip_same_cell: bool, collapse_refusals: bool
) -> tuple[list[Finding], list[dict[str, Any]]]:
    """Gate every matched-seed arm pair of one comparison group at equal memory.

    Args:
        runs: the runs of the group (already filtered to real runs or to fixtures).
        skip_same_cell: skip pairs that are two runs of one cell - for real runs those are duplicates,
            already reported; fixture groups are hand-authored gate inputs and gate every pair.
        collapse_refusals: fold the not-declared refusals into one warning per group.
    """
    findings: list[Finding] = []
    verdicts: list[dict[str, Any]] = []
    refused: dict[str, dict[str, Any]] = {}
    successes: dict[int, list[RunRecord]] = {}
    for run in runs:
        if run.status == SUCCESS:
            successes.setdefault(run.seed, []).append(run)

    for seed in sorted(successes):
        matched = sorted(successes[seed], key=lambda run: run.path)
        for run_a, run_b in combinations(matched, 2):
            if skip_same_cell and run_a.key == run_b.key:
                continue
            outcome = _memory_outcome(run_a, run_b)
            if outcome.refusal is not None:
                if outcome.blocking:
                    severity, fixture_only = _severity(CODE_UNEQUAL_MEMORY, (run_a, run_b))
                    findings.append(
                        Finding(
                            code=CODE_UNEQUAL_MEMORY,
                            severity=severity,
                            detail=(
                                f"seed {seed}: {run_a.path} vs {run_b.path}: {outcome.reason} "
                                f"(bytes_a={outcome.bytes_a}, bytes_b={outcome.bytes_b}, "
                                f"source={outcome.bytes_source!r})"
                            ),
                            cell=None,
                            runs=(run_a.run_id, run_b.run_id),
                            fixture_only=fixture_only,
                        )
                    )
                elif collapse_refusals:
                    # Collapse on the reason sentence: the pair context only names the two runs, so
                    # grouping on the full message would leave one entry per pair.
                    reason = outcome.refusal.split(" (", 1)[0]
                    entry = refused.setdefault(reason, {"count": 0, "example": (run_a, run_b)})
                    entry["count"] += 1
                else:
                    severity, fixture_only = _severity(CODE_MEMORY_GATE_REFUSED, (run_a, run_b))
                    findings.append(
                        Finding(
                            code=CODE_MEMORY_GATE_REFUSED,
                            severity=severity,
                            detail=f"seed {seed}: {run_a.path} vs {run_b.path} - {outcome.refusal}",
                            runs=(run_a.run_id, run_b.run_id),
                            fixture_only=fixture_only,
                        )
                    )
            elif outcome.verdict is not None:
                verdicts.append(
                    {
                        "seed": seed,
                        "run_a": run_a.run_id,
                        "run_b": run_b.run_id,
                        "cell_a": run_a.key.digest,
                        "cell_b": run_b.key.digest,
                        "fixture": run_a.is_fixture,
                        **outcome.verdict,
                    }
                )

    if refused:
        reasons = sorted(refused)
        total = sum(int(entry["count"]) for entry in refused.values())
        listed = "; ".join(
            f"{refused[reason]['count']}x {reason} "
            f"[e.g. {refused[reason]['example'][0].run_id} vs "
            f"{refused[reason]['example'][1].run_id}]"
            for reason in reasons[:3]
        )
        tail = "" if len(reasons) <= 3 else f" (+{len(reasons) - 3} more distinct refusals)"
        participants = tuple(
            sorted({run.run_id for entry in refused.values() for run in entry["example"]})
        )
        severity, fixture_only = _severity(CODE_MEMORY_GATE_REFUSED, tuple(runs))
        findings.append(
            Finding(
                code=CODE_MEMORY_GATE_REFUSED,
                severity=severity,
                detail=(
                    f"{total} arm pair(s) cannot be gated at equal memory "
                    f"(AGENTS.md section 4.5): {listed}{tail}"
                ),
                runs=participants,
                fixture_only=fixture_only,
            )
        )
    return findings, verdicts


def _build_families(
    cells: Sequence[CellRecord],
) -> tuple[tuple[FamilyRecord, ...], tuple[CellRecord, ...], list[Finding]]:
    """Group cells into families; report the axes on which a family's cells disagree."""
    by_family: dict[str, list[CellRecord]] = {}
    for cell in cells:
        by_family.setdefault(cell.key.family_digest, []).append(cell)

    findings: list[Finding] = []
    records: list[FamilyRecord] = []
    incomparable: dict[str, set[str]] = {}
    for family_key in sorted(by_family):
        family = sorted(by_family[family_key], key=lambda cell: cell.key.digest)
        divergent: dict[str, list[str]] = {}
        for axis in AXES:
            values = sorted({cell.key.axes()[axis] for cell in family})
            if len(values) > 1:
                divergent[axis] = values
        label = family[0].key.label()
        records.append(
            FamilyRecord(
                family_key=family_key,
                label=label,
                cells=tuple(cell.key.digest for cell in family),
                divergent_axes=divergent,
            )
        )
        for axis, values in sorted(divergent.items()):
            for cell in family:
                incomparable.setdefault(cell.key.digest, set()).add(axis)
            findings.append(
                Finding(
                    code=CODE_INCOMPARABLE,
                    severity=SEVERITY_WARNING,
                    detail=(
                        f"{label}: {len(family)} cells belong to one family but differ in {axis!r} "
                        f"({', '.join(values)}); they must not be pooled, averaged or differenced "
                        "(eval-protocol.md section 8)"
                    ),
                    cell=family[0].key.digest,
                    axis=axis,
                    runs=tuple(cell.key.digest for cell in family),
                )
            )

    updated = tuple(
        CellRecord(
            key=cell.key,
            runs=cell.runs,
            expected_seeds=cell.expected_seeds,
            seed_floor=cell.seed_floor,
            present_seeds=cell.present_seeds,
            missing_seeds=cell.missing_seeds,
            duplicates=cell.duplicates,
            incomparable_axes=tuple(sorted(incomparable.get(cell.key.digest, ()))),
        )
        for cell in cells
    )
    return tuple(records), updated, findings


def _directory_records(
    records: Sequence[RunRecord], cells: Sequence[CellRecord]
) -> tuple[DirectoryRecord, ...]:
    """Summarize what each directory contributed, and say when a directory is fixture-only."""
    real_cell_keys = {cell.key.digest for cell in cells}
    per_dir: dict[str, dict[str, Any]] = {}
    for run in records:
        directory = Path(run.path).parent.as_posix()
        entry = per_dir.setdefault(directory, {"manifests": 0, "fixtures": 0, "cells": set()})
        entry["manifests"] += 1
        if run.is_fixture:
            entry["fixtures"] += 1
        elif run.key.digest in real_cell_keys:
            entry["cells"].add(run.key.digest)

    result: list[DirectoryRecord] = []
    for directory in sorted(per_dir):
        entry = per_dir[directory]
        manifests = int(entry["manifests"])
        fixtures = int(entry["fixtures"])
        result.append(
            DirectoryRecord(
                path=directory,
                manifests=manifests,
                fixtures=fixtures,
                cells=len(entry["cells"]),
                fixture_only=fixtures == manifests,
            )
        )
    return tuple(result)
