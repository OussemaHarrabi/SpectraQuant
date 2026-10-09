"""Collection and validation of a downloaded artifact bundle (``design-cloud-adapter.md`` §5).

:func:`collect` is the gate between "something came back from the cloud" and "a result exists". It
verifies, in order:

1. ``run_manifest.json`` is present and validates against ``artifacts/schemas/run-manifest.schema.json``;
2. the manifest's ``run_id`` is the run being collected;
3. the manifest's ``git_commit`` matches the frozen spec (a dirty run must declare ``git_dirty``);
4. every ``expected_artifacts`` entry is present, at least ``min_bytes`` long, and — when a digest is
   declared — byte-identical to it;
5. the *executed* notebook is present (a notebook file with execution counts, tagged with this run id
   when it carries the generator's metadata).

Only a full success produces ``validated``; anything else produces ``rejected`` with the reasons. No
partial success exists, and metrics become visible to analysis only through the ``validated`` line.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from spectraquant.cloud.registry import Registry, default_registry
from spectraquant.cloud.remote import parse_result_line
from spectraquant.cloud.spec import RunSpec, checksum_hex
from spectraquant.reporting.manifests import validate_manifest_dict

__all__ = ["MANIFEST_FILENAME", "ValidationReport", "collect", "find_manifest"]

#: Name of the manifest inside a bundle.
MANIFEST_FILENAME = "run_manifest.json"


@dataclass(frozen=True)
class ValidationReport:
    """Outcome of collecting one run's bundle.

    Attributes:
        run_id: the run being collected.
        state: ``"validated"`` or ``"rejected"``.
        reasons: why collection failed (empty on success).
        source_dir: directory the bundle was read from.
        manifest: the validated manifest document, or ``None``.
        manifest_path: path of the manifest that was read.
        manifest_sha256: ``sha256:<hex>`` computed from the manifest's bytes.
        artifact_checksums: ``{name: sha256:<hex>}`` computed from the downloaded files.
        executed_notebook: path of the executed notebook, when found.
        checksum_verified: True only when every declared digest matched the downloaded bytes.
        platform: platform recorded for the run.
        remote_id: platform-side run id.
        git_commit: commit recorded in the manifest.
        spec_sha256: digest of the frozen spec.
        submitted_by: who submitted the run.
        result_line: the parsed ``SPECTRAQUANT_RESULT_JSON=`` payload, when present.
        teardown_observed: True/False when a log with markers was found, else ``None``.
    """

    run_id: str
    state: str
    source_dir: str
    reasons: list[str] = field(default_factory=list)
    manifest: dict[str, Any] | None = None
    manifest_path: str | None = None
    manifest_sha256: str | None = None
    artifact_checksums: dict[str, str] = field(default_factory=dict)
    executed_notebook: str | None = None
    checksum_verified: bool = False
    platform: str | None = None
    remote_id: str | None = None
    git_commit: str | None = None
    spec_sha256: str | None = None
    submitted_by: str | None = None
    result_line: dict[str, Any] | None = None
    teardown_observed: bool | None = None

    @property
    def ok(self) -> bool:
        """True when the bundle validated."""
        return self.state == "validated"

    def metrics(self) -> dict[str, Any]:
        """Return the manifest's measured metrics (only meaningful when validated)."""
        if not self.manifest:
            return {}
        metrics = self.manifest.get("metrics")
        return dict(metrics) if isinstance(metrics, Mapping) else {}

    def summary(self) -> str:
        """One-line human summary."""
        if self.ok:
            return f"{self.run_id}: validated ({len(self.artifact_checksums)} artifacts verified)"
        return f"{self.run_id}: rejected — " + "; ".join(self.reasons)

    def to_json(self) -> dict[str, Any]:
        """JSON-serializable report."""
        return {
            "run_id": self.run_id,
            "state": self.state,
            "source_dir": self.source_dir,
            "reasons": list(self.reasons),
            "manifest_path": self.manifest_path,
            "manifest_sha256": self.manifest_sha256,
            "artifact_checksums": dict(self.artifact_checksums),
            "executed_notebook": self.executed_notebook,
            "checksum_verified": self.checksum_verified,
            "platform": self.platform,
            "remote_id": self.remote_id,
            "git_commit": self.git_commit,
            "spec_sha256": self.spec_sha256,
            "submitted_by": self.submitted_by,
            "teardown_observed": self.teardown_observed,
            "status": None if not self.manifest else self.manifest.get("status"),
        }


def find_manifest(source_dir: str | Path) -> Path | None:
    """Return the bundle's own ``run_manifest.json`` under ``source_dir``.

    The bundle carries nested manifests too: every arm of a plan writes its own
    ``artifacts/runs/<plan>/<arm>/seed-N/run_manifest.json``. Those are the arms' records, not the
    run's, and they carry the arms' run ids - picking one makes the identity check reject a valid
    bundle. The run's manifest is the shallowest one in the tree (the export stage writes it at
    ``<bundle>/run_manifest.json``), so rank by depth before name.
    """
    root = Path(source_dir)
    if not root.is_dir():
        return None
    direct = root / MANIFEST_FILENAME
    if direct.is_file():
        return direct
    matches = [path for path in root.rglob(MANIFEST_FILENAME) if path.is_file()]
    if not matches:
        return None
    return min(matches, key=lambda path: (len(path.parts), str(path)))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _resolve_artifact(source_dir: Path, name: str) -> Path | None:
    """Resolve an expected artifact name inside ``source_dir`` (exact path, then by basename)."""
    direct = source_dir / name
    if direct.is_file():
        return direct
    matches = sorted(path for path in source_dir.rglob(Path(name).name) if path.is_file())
    return matches[0] if matches else None


def _find_executed_notebook(source_dir: Path, run_id: str) -> tuple[Path | None, list[str]]:
    """Return ``(path, reasons)`` for the executed notebook of ``run_id``.

    A notebook qualifies when it parses, is nbformat 4, and has at least one code cell with an
    ``execution_count`` (i.e. it actually ran). A file whose generator metadata names a different run
    id is never accepted.
    """
    reasons: list[str] = []
    candidates: list[Path] = []
    for path in sorted(source_dir.rglob("*.ipynb")):
        if not path.is_file():
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(document, Mapping) or int(document.get("nbformat", 0) or 0) != 4:
            continue
        cells = document.get("cells") or []
        executed = any(
            isinstance(cell, Mapping)
            and cell.get("cell_type") == "code"
            and cell.get("execution_count") is not None
            for cell in cells
        )
        if not executed:
            continue
        metadata = document.get("metadata") or {}
        generated = metadata.get("spectraquant") if isinstance(metadata, Mapping) else None
        if isinstance(generated, Mapping) and generated.get("run_id") not in (None, run_id):
            continue
        candidates.append(path)
    if not candidates:
        reasons.append(
            "executed notebook missing: no .ipynb with execution counts found under "
            f"{source_dir} (a template notebook is not evidence)"
        )
        return None, reasons
    return candidates[0], reasons


def _check_commit(spec: RunSpec, manifest: Mapping[str, Any]) -> list[str]:
    """Return the commit-mismatch reasons (empty when the manifest matches the spec)."""
    manifest_commit = manifest.get("git_commit")
    if not manifest_commit:
        return ["manifest records no git_commit: the run cannot be tied to a commit"]
    if manifest_commit == spec.git_commit:
        return []
    if spec.allow_dirty and manifest.get("git_dirty") is True:
        return []
    return [
        f"git commit mismatch: manifest {manifest_commit} != spec {spec.git_commit}"
        + ("" if spec.allow_dirty else " (the spec does not allow a dirty tree)")
    ]


def _check_artifacts(spec: RunSpec, source_dir: Path) -> tuple[dict[str, str], list[str]]:
    """Verify every expected artifact; return ``(checksums, reasons)``."""
    reasons: list[str] = []
    checksums: dict[str, str] = {}
    if not spec.expected_artifacts:
        reasons.append(
            "the spec declares no expected_artifacts: nothing can be verified, so nothing is "
            "validated (add the artifacts the run must produce)"
        )
        return checksums, reasons
    for expected in spec.expected_artifacts:
        path = _resolve_artifact(source_dir, expected.name)
        if path is None:
            reasons.append(f"expected artifact missing: {expected.name}")
            continue
        size = path.stat().st_size
        if size < expected.min_bytes:
            reasons.append(
                f"artifact {expected.name} is {size} B, below min_bytes={expected.min_bytes}"
            )
            continue
        digest = _sha256_file(path)
        if expected.sha256 is not None and checksum_hex(expected.sha256) != checksum_hex(digest):
            reasons.append(
                f"artifact {expected.name} sha256 mismatch: declared {expected.sha256}, "
                f"downloaded {digest}"
            )
            continue
        checksums[expected.name] = digest
    return checksums, reasons


def collect(
    run_id: str,
    source_dir: str | Path,
    *,
    spec: RunSpec | None = None,
    registry: Registry | None = None,
    record: bool = True,
) -> ValidationReport:
    """Validate a downloaded bundle and record the outcome.

    Args:
        run_id: the run being collected.
        source_dir: directory holding the exported bundle (manifest, artifacts, executed notebook).
        spec: the frozen spec; when omitted it is read from the registry's ``submitted`` line.
        registry: registry to read/write; defaults to the repository registry.
        record: when False, validate without touching the registry (used by dry-run checks).

    Returns:
        A :class:`ValidationReport`; ``validated`` only on full success.
    """
    registry = registry if registry is not None else default_registry()
    root = Path(source_dir)
    recorded = registry.latest(run_id) or {}
    if spec is None:
        document = registry.spec_document(run_id)
        spec = RunSpec.model_validate(document) if document else None

    report = ValidationReport(
        run_id=run_id,
        state="rejected",
        source_dir=str(root),
        platform=(spec.platform if spec else recorded.get("platform")),
        remote_id=registry.remote_id(run_id),
        spec_sha256=(spec.sha256() if spec else None),
        submitted_by=recorded.get("submitted_by"),
    )

    reasons: list[str] = []
    manifest_path = find_manifest(root)
    if manifest_path is None:
        reasons.append(f"no {MANIFEST_FILENAME} under {root}")
    if spec is None:
        reasons.append(
            "no RunSpec is recorded for this run: submission must persist the spec before "
            "collection can verify it"
        )

    if reasons:
        return _finish(report, reasons, registry=registry, record=record)

    assert manifest_path is not None and spec is not None  # reasons would be non-empty otherwise
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = validate_manifest_dict(document, source=str(manifest_path))
    except (OSError, ValueError) as exc:
        reasons.append(f"{MANIFEST_FILENAME} is not a valid run manifest: {exc}")
        return _finish(report, reasons, registry=registry, record=record)

    if manifest.get("run_id") != run_id:
        reasons.append(
            f"manifest run_id {manifest.get('run_id')!r} does not match the collected run "
            f"{run_id!r}"
        )
    reasons.extend(_check_commit(spec, manifest))

    checksums, artifact_reasons = _check_artifacts(spec, root)
    reasons.extend(artifact_reasons)

    notebook_path, notebook_reasons = _find_executed_notebook(root, run_id)
    reasons.extend(notebook_reasons)

    if manifest.get("status") != "success":
        reasons.append(
            f"the run reported status={manifest.get('status')!r}: "
            f"{manifest.get('failure_reason') or 'no failure reason recorded'}"
        )

    checksums[MANIFEST_FILENAME] = _sha256_file(manifest_path)
    if notebook_path is not None:
        checksums[notebook_path.relative_to(root).as_posix()] = _sha256_file(notebook_path)

    result_line, teardown = _scan_logs(root)
    report = replace(
        report,
        manifest=manifest,
        manifest_path=str(manifest_path),
        manifest_sha256=checksums[MANIFEST_FILENAME],
        artifact_checksums=checksums,
        executed_notebook=None if notebook_path is None else str(notebook_path),
        checksum_verified=not reasons,
        git_commit=None if manifest.get("git_commit") is None else str(manifest.get("git_commit")),
        result_line=result_line,
        teardown_observed=teardown,
    )
    return _finish(report, reasons, registry=registry, record=record)


def _scan_logs(source_dir: Path) -> tuple[dict[str, Any] | None, bool | None]:
    """Return ``(parsed result line, teardown marker observed)`` from any log in the bundle."""
    result: dict[str, Any] | None = None
    saw_marker = False
    saw_log = False
    for path in sorted(source_dir.rglob("*.log")):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:  # pragma: no cover - unreadable log
            continue
        saw_log = True
        parsed = parse_result_line(text)
        if parsed is not None:
            result = parsed
        if "SPECTRAQUANT_TEARDOWN_OK " in text:
            saw_marker = True
    if not saw_log:
        return result, None
    return result, saw_marker


def _finish(
    report: ValidationReport, reasons: Sequence[str], *, registry: Registry, record: bool
) -> ValidationReport:
    """Record ``collected`` + the outcome and return the final report.

    Raises:
        RegistryError: the registry already holds a conflicting terminal state for this run (the
            registry never rewrites history).
    """
    if not record:
        state = "rejected" if reasons else "validated"
        return replace(report, state=state, reasons=list(reasons))
    state = "rejected" if reasons else "validated"
    final = replace(report, state=state, reasons=list(reasons))
    current = registry.state(report.run_id)
    if current not in {"collected", "validated", "rejected"} and report.manifest_path is not None:
        registry.record_collected(
            report.run_id,
            platform=report.platform,
            remote_id=report.remote_id,
            detail="bundle downloaded",
        )
    if final.ok:
        registry.record_validated(final)
    else:
        registry.record_rejected(final)
    return final
