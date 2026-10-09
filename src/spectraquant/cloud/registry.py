"""Append-only run registry (``design-cloud-adapter.md`` §5).

One JSON line per state transition at ``artifacts/runs/registry.jsonl`` (git-ignored; the schema is
documented in ``scripts/cloud/README.md``). The registry is the research record of what the cloud
substrate actually produced, so it is deliberately hard to lie to:

* states follow a fixed transition graph — an impossible transition raises;
* ``validated`` is **unreachable** through the generic API: only
  :meth:`Registry.record_validated` can write it, and it refuses any report whose artifact checksums
  were not verified against the downloaded files;
* metrics exist on a line only when that line came from a validated manifest;
* every line is passed through the credential redactor before it is written.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spectraquant.cloud.secrets import redact
from spectraquant.paths import repo_root
from spectraquant.reporting.manifests import utc_timestamp

__all__ = [
    "ALLOWED_TRANSITIONS",
    "REGISTRY_FILENAME",
    "STATES",
    "Registry",
    "RegistryError",
    "Transition",
    "default_registry",
    "default_registry_path",
    "runs_dir",
]

#: File name inside the runs directory.
REGISTRY_FILENAME = "registry.jsonl"

#: Every state a run may be recorded in, in the order the design note lists them.
STATES: tuple[str, ...] = (
    "submitted",
    "resubmitted",
    "running",
    "finished",
    "failed",
    "collected",
    "validated",
    "rejected",
)

#: Allowed successor states. A same-state repeat is always allowed (append-only idempotency).
ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    "submitted": frozenset(
        {"resubmitted", "running", "finished", "failed", "collected", "rejected"}
    ),
    "resubmitted": frozenset({"running", "finished", "failed", "collected", "rejected"}),
    "running": frozenset({"resubmitted", "finished", "failed", "collected", "rejected"}),
    "finished": frozenset({"collected", "failed", "rejected"}),
    "failed": frozenset({"collected", "rejected"}),
    "collected": frozenset({"validated", "rejected"}),
    "validated": frozenset(),
    # A rejection is a verdict on one *collection attempt*, not on the run: collection re-validates
    # every artifact from scratch, so a bundle rejected because of a collector defect can be
    # re-collected once the defect is fixed. Observed: the collector picked a nested per-arm
    # manifest, rejected a valid bundle, and the repaired collector could not record its verdict.
    "rejected": frozenset({"validated"}),
}


class RegistryError(ValueError):
    """Raised for an invalid state, an impossible transition, or a fabricated validation."""


@dataclass(frozen=True)
class Transition:
    """One appended registry line.

    Attributes:
        state: one of :data:`STATES`.
        run_id: the spec's run id.
        timestamp_utc: when the transition was recorded (ISO 8601, ``Z``).
        platform: execution substrate, when known.
        remote_id: platform-side run id (``None`` for a notebook handoff).
        git_commit: the commit the run was supposed to execute.
        spec_sha256: digest of the frozen spec, when known.
        notebook_digest: digest of the generated notebook, when known.
        measurement_class: declared class (1..4), when known.
        artifact_checksums: ``{relative path: sha256:<hex>}`` verified at collection.
        manifest_sha256: digest of the collected ``run_manifest.json``.
        failure_reason: why the run failed/was rejected.
        submitted_by: ``"agent"``/``"human"`` — never claims more than happened.
        detail: free-text provenance note.
        spec: the full spec document, recorded on ``submitted`` so collection can resume.
        metrics: measured metrics, present only on a ``validated`` line.
        checksum_verified: internal gate proving checksums were verified against real files.
    """

    state: str
    run_id: str
    timestamp_utc: str = field(default_factory=utc_timestamp)
    platform: str | None = None
    remote_id: str | None = None
    git_commit: str | None = None
    spec_sha256: str | None = None
    notebook_digest: str | None = None
    measurement_class: int | None = None
    artifact_checksums: Mapping[str, str] = field(default_factory=dict)
    manifest_sha256: str | None = None
    failure_reason: str | None = None
    submitted_by: str | None = None
    detail: str | None = None
    spec: Mapping[str, Any] | None = None
    metrics: Mapping[str, Any] | None = None
    checksum_verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable line (dropping unset optional fields)."""
        document: dict[str, Any] = {"state": self.state, "run_id": self.run_id}
        optional: dict[str, Any] = {
            "timestamp_utc": self.timestamp_utc,
            "platform": self.platform,
            "remote_id": self.remote_id,
            "git_commit": self.git_commit,
            "spec_sha256": self.spec_sha256,
            "notebook_digest": self.notebook_digest,
            "measurement_class": self.measurement_class,
            "artifact_checksums": dict(self.artifact_checksums),
            "manifest_sha256": self.manifest_sha256,
            "failure_reason": self.failure_reason,
            "submitted_by": self.submitted_by,
            "detail": self.detail,
            "spec": None if self.spec is None else dict(self.spec),
            "metrics": None if self.metrics is None else dict(self.metrics),
            "checksum_verified": self.checksum_verified,
        }
        document.update(optional)
        return document


def runs_dir() -> Path:
    """Return the runs directory (``$SPECTRAQUANT_RUNS_DIR`` or ``<repo>/artifacts/runs``)."""
    override = os.environ.get("SPECTRAQUANT_RUNS_DIR")
    if override:
        return Path(override).expanduser()
    return repo_root() / "artifacts" / "runs"


def default_registry_path() -> Path:
    """Return the default registry path (``<runs>/registry.jsonl``)."""
    return runs_dir() / REGISTRY_FILENAME


class Registry:
    """Append-only JSONL registry of run state transitions.

    Args:
        path: registry file; defaults to :func:`default_registry_path`. Parents are created on first
            write.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path is not None else default_registry_path()

    @property
    def path(self) -> Path:
        """Registry file path."""
        return self._path

    # -- reading ------------------------------------------------------------------------
    def transitions(self, run_id: str | None = None) -> list[dict[str, Any]]:
        """Return every recorded transition, optionally filtered to one run id.

        A line that is not valid JSON is skipped rather than raising: the registry is append-only
        and a partially written trailing line must not make the record unreadable.
        """
        if not self._path.is_file():
            return []
        records: list[dict[str, Any]] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                continue
            if run_id is None or parsed.get("run_id") == run_id:
                records.append(parsed)
        return records

    def latest(self, run_id: str) -> dict[str, Any] | None:
        """Return the most recent transition for ``run_id``, or ``None``."""
        records = self.transitions(run_id)
        return records[-1] if records else None

    def state(self, run_id: str) -> str | None:
        """Return the current state of ``run_id``, or ``None`` when unrecorded."""
        latest = self.latest(run_id)
        return None if latest is None else str(latest.get("state"))

    def remote_id(self, run_id: str) -> str | None:
        """Return the persisted platform-side run id for ``run_id``, if any.

        The id is taken from the most recent transition that carries one, so an interruption between
        submit and poll cannot orphan a running job (``design-cloud-adapter.md`` §4).
        """
        for record in reversed(self.transitions(run_id)):
            remote_id = record.get("remote_id")
            if remote_id:
                return str(remote_id)
        return None

    def spec_document(self, run_id: str) -> dict[str, Any] | None:
        """Return the spec document recorded at submission, or ``None``."""
        for record in reversed(self.transitions(run_id)):
            document = record.get("spec")
            if isinstance(document, dict):
                return document
        return None

    def spec(self, run_id: str) -> Any:
        """Return the recorded :class:`RunSpec` for ``run_id``, or ``None``."""
        document = self.spec_document(run_id)
        if document is None:
            return None
        from spectraquant.cloud.spec import RunSpec

        return RunSpec.model_validate(document)

    def status_summary(self) -> list[dict[str, Any]]:
        """Return ``[{run_id, state, platform, remote_id, timestamp_utc, ...}]``, one per run."""
        summary: dict[str, dict[str, Any]] = {}
        for record in self.transitions():
            run_id = str(record.get("run_id"))
            entry = summary.setdefault(run_id, {"run_id": run_id, "transitions": 0})
            entry["transitions"] = int(entry["transitions"]) + 1
            entry["state"] = record.get("state")
            entry["timestamp_utc"] = record.get("timestamp_utc")
            entry["platform"] = record.get("platform")
            entry["remote_id"] = record.get("remote_id")
            entry["failure_reason"] = record.get("failure_reason")
        return [summary[key] for key in sorted(summary)]

    # -- writing ------------------------------------------------------------------------
    def append(self, transition: Transition) -> Transition:
        """Validate and append one transition.

        Args:
            transition: the transition to record.

        Returns:
            The recorded transition.

        Raises:
            RegistryError: unknown state, impossible transition, or a ``validated`` line without
                verified checksums.
        """
        if transition.state not in STATES:
            raise RegistryError(f"unknown state {transition.state!r}; expected one of {STATES}")
        current = self.state(transition.run_id)
        if current is not None and transition.state != current:
            allowed = ALLOWED_TRANSITIONS[current]
            if transition.state not in allowed:
                raise RegistryError(
                    f"illegal transition for {transition.run_id!r}: {current} -> "
                    f"{transition.state} (allowed: {sorted(allowed) or 'none'})"
                )
        if transition.state == "validated":
            if not transition.checksum_verified:
                raise RegistryError(
                    "refusing to record 'validated': no checksum-verified artifact set "
                    "(design-cloud-adapter.md §5 — the registry only records what was downloaded "
                    "and verified)"
                )
            if not transition.artifact_checksums:
                raise RegistryError(
                    "refusing to record 'validated': the artifact checksum set is empty"
                )
            if not transition.manifest_sha256:
                raise RegistryError(
                    "refusing to record 'validated': no validated run_manifest.json digest"
                )

        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(transition.to_dict(), sort_keys=True, default=str)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(redact(line) + "\n")
        return transition

    def record(self, state: str, run_id: str, **fields: Any) -> Transition:
        """Record a transition that does not require verified checksums.

        Raises:
            RegistryError: ``state == "validated"`` (use :meth:`record_validated`).
        """
        if state == "validated":
            raise RegistryError(
                "'validated' cannot be recorded through record(); it requires a ValidationReport "
                "with checksum-verified artifacts (use record_validated)"
            )
        return self.append(Transition(state=state, run_id=run_id, **fields))

    def record_submitted(
        self,
        spec: Any,
        *,
        remote_id: str | None,
        submitted_by: str,
        notebook_digest: str | None = None,
        detail: str | None = None,
    ) -> Transition:
        """Persist a submission (idempotently) **before** the caller returns.

        A repeated submission for the same run id with the same remote id is a no-op, so an
        interrupted submit→poll cycle re-uses the persisted id instead of re-submitting.
        """
        existing = self.latest(spec.run_id)
        if existing is not None:
            same_target = (
                existing.get("state") in {"submitted", "resubmitted", "running"}
                and existing.get("remote_id") == remote_id
                and existing.get("notebook_digest") == notebook_digest
            )
            if same_target:
                # A genuine repeat: same kernel, same notebook, still in flight.
                return _transition_from_dict(existing)
        # Anything else is a NEW submission for this run id - a different kernel, a different
        # notebook, or a previous attempt that ended. It is recorded as ``resubmitted`` so the
        # history keeps both attempts instead of silently overwriting the first one.
        state = "submitted" if existing is None else "resubmitted"
        return self.record(
            state,
            spec.run_id,
            platform=spec.platform,
            remote_id=remote_id,
            git_commit=spec.git_commit or None,
            spec_sha256=spec.sha256(),
            notebook_digest=notebook_digest,
            measurement_class=spec.measurement_class_expected,
            submitted_by=submitted_by,
            detail=detail,
            spec=spec.to_json(),
        )

    def record_collected(
        self, run_id: str, *, platform: str | None = None, **fields: Any
    ) -> Transition:
        """Record that a bundle was downloaded (evidence now exists locally)."""
        return self.record("collected", run_id, platform=platform, **fields)

    def record_rejected(self, report: Any) -> Transition:
        """Record a rejected collection, with its reasons."""
        return self.record(
            "rejected",
            report.run_id,
            platform=report.platform,
            remote_id=report.remote_id,
            git_commit=report.git_commit,
            spec_sha256=report.spec_sha256,
            artifact_checksums=report.artifact_checksums,
            manifest_sha256=report.manifest_sha256,
            failure_reason="; ".join(report.reasons) or "validation failed",
            detail="collection rejected",
        )

    def record_validated(self, report: Any) -> Transition:
        """Record a validated collection, exposing the manifest's metrics to analysis.

        Raises:
            RegistryError: the report is not a successful, checksum-verified validation.
        """
        if getattr(report, "state", None) != "validated":
            raise RegistryError(
                "record_validated requires a ValidationReport in state 'validated' "
                f"(got {getattr(report, 'state', None)!r})"
            )
        if not getattr(report, "checksum_verified", False):
            raise RegistryError(
                "record_validated requires checksum_verified=True (the checksums must have been "
                "computed from the downloaded files, not from the manifest)"
            )
        metrics = report.metrics() if callable(getattr(report, "metrics", None)) else None
        return self.append(
            Transition(
                state="validated",
                run_id=report.run_id,
                platform=report.platform,
                remote_id=report.remote_id,
                git_commit=report.git_commit,
                spec_sha256=report.spec_sha256,
                artifact_checksums=report.artifact_checksums,
                manifest_sha256=report.manifest_sha256,
                submitted_by=getattr(report, "submitted_by", None),
                detail="collection validated",
                metrics=metrics,
                checksum_verified=True,
            )
        )

    def record_failure(
        self, run_id: str, reason: str, *, platform: str | None = None, **fields: Any
    ) -> Transition:
        """Record a failure with its reason (failures are preserved, never retried into success)."""
        return self.record(
            "failed",
            run_id,
            platform=platform,
            failure_reason=reason,
            detail="run failed",
            **fields,
        )


def _transition_from_dict(document: Mapping[str, Any]) -> Transition:
    """Rebuild a :class:`Transition` from an already-recorded line."""
    return Transition(
        state=str(document.get("state")),
        run_id=str(document.get("run_id")),
        timestamp_utc=str(document.get("timestamp_utc") or utc_timestamp()),
        platform=_optional_str(document.get("platform")),
        remote_id=_optional_str(document.get("remote_id")),
        git_commit=_optional_str(document.get("git_commit")),
        spec_sha256=_optional_str(document.get("spec_sha256")),
        notebook_digest=_optional_str(document.get("notebook_digest")),
        measurement_class=document.get("measurement_class"),
        artifact_checksums=dict(document.get("artifact_checksums") or {}),
        manifest_sha256=_optional_str(document.get("manifest_sha256")),
        failure_reason=_optional_str(document.get("failure_reason")),
        submitted_by=_optional_str(document.get("submitted_by")),
        detail=_optional_str(document.get("detail")),
        spec=document.get("spec"),
        metrics=document.get("metrics"),
        checksum_verified=bool(document.get("checksum_verified", False)),
    )


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def default_registry() -> Registry:
    """Return a :class:`Registry` at the default path."""
    return Registry()


def record_transition(state: str, run_id: str, **fields: Any) -> Transition:
    """Record a transition in the default registry."""
    return default_registry().record(state, run_id, **fields)


def state_of(run_id: str) -> str | None:
    """Return the current state of ``run_id`` in the default registry."""
    return default_registry().state(run_id)


def validated_runs() -> Sequence[dict[str, Any]]:
    """Return the ``validated`` transitions of the default registry (metrics included)."""
    return [
        record for record in default_registry().transitions() if record.get("state") == "validated"
    ]
