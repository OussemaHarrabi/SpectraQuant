"""Shared builders for the ``test_cloud_*`` modules (not a test module itself).

Kept as a plain importable module rather than a ``conftest.py`` so it cannot collide with another
stream's fixture file. Nothing here touches the network or the real registry.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spectraquant.cloud.adapters.base import FetchReport, RunStatus
from spectraquant.cloud.registry import Registry
from spectraquant.cloud.spec import DatasetRef, ExpectedArtifact, RunSpec

#: A syntactically valid (but not real) commit SHA used by every unit fixture.
COMMIT = "0123456789abcdef0123456789abcdef01234567"
OTHER_COMMIT = "fedcba9876543210fedcba9876543210fedcba98"
SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64

ARTIFACT_NAME = "artifacts/result.json"
ARTIFACT_BYTES = b'{"ok": true, "val_loss": 0.5}\n'


def make_spec(**overrides: Any) -> RunSpec:
    """Return a valid :class:`RunSpec` with sensible defaults for the unit tests."""
    values: dict[str, Any] = {
        "run_id": "cloud-unit-run",
        "experiment_config": "configs/experiment/smoke.yaml",
        "overrides": [],
        "platform": "local_cpu",
        "repo_url": "https://github.com/example/SpectraQuant.git",
        "git_commit": COMMIT,
        "install_spec": "uv sync --frozen --extra cloud",
        "dataset_refs": [
            DatasetRef(
                name="synthetic-lcg-v1",
                revision="config-sha256:" + "0" * 64,
                split="train",
                checksum=SHA_A,
            )
        ],
        "seeds": [1234],
        "gpu_required": False,
        "timeout_minutes": 60,
        "expected_artifacts": [ExpectedArtifact(name=ARTIFACT_NAME, min_bytes=4)],
        "measurement_class_expected": 1,
    }
    values.update(overrides)
    return RunSpec(**values)


def make_manifest(
    spec: RunSpec,
    *,
    git_commit: str | None = None,
    status: str = "success",
    failure_reason: str | None = None,
    metrics: dict[str, Any] | None = None,
    run_id: str | None = None,
    artifact_paths: list[str] | None = None,
    timestamp_utc: str = "2026-10-08T00:00:00Z",
) -> dict[str, Any]:
    """Build a schema-valid run manifest document for ``spec`` (no network, no torch training)."""
    from spectraquant.reporting.environment import collect_hardware, collect_software
    from spectraquant.reporting.manifests import (
        CompressionBlock,
        HardwareBlock,
        RunManifest,
        TrainingBlock,
        validate_manifest_dict,
    )

    document = RunManifest(
        run_id=run_id if run_id is not None else spec.run_id,
        timestamp_utc=timestamp_utc,
        git_commit=git_commit if git_commit is not None else spec.git_commit,
        git_dirty=False,
        config_path=spec.experiment_config,
        resolved_config={"experiment": {"name": "cloud-unit"}},
        dataset_ids=[ref.name for ref in spec.dataset_refs],
        dataset_revisions=[ref.revision for ref in spec.dataset_refs],
        dataset_checksums=[ref.checksum for ref in spec.dataset_refs],
        split="train",
        seed=spec.seeds[0],
        hardware=HardwareBlock(**collect_hardware()),
        software=collect_software(),
        compression=CompressionBlock(method="none"),
        runtime_backend="torch-cpu-fp32",
        measurement_class=None,
        training=TrainingBlock(steps=10, tokens=1024, wall_time_s=1.5),
        metrics={"val_loss": 0.5} if metrics is None else metrics,
        status=status,
        failure_reason=failure_reason,
        log_path=None,
        artifact_paths=artifact_paths if artifact_paths is not None else [ARTIFACT_NAME],
    ).to_json_dict()
    # The published schema requires the five baseline keys (null is allowed for ranks/bits/
    # group_size) but does not accept `null` for the optional class-3 fields, so a model dump that
    # includes those as null is not a valid document. Keep the required keys, drop the null optionals.
    required_compression_keys = {"method", "ranks", "bits", "group_size", "exclusions"}
    document["compression"] = {
        key: value
        for key, value in document["compression"].items()
        if key in required_compression_keys or value is not None
    }
    validate_manifest_dict(document)
    return document


def executed_notebook(run_id: str, *, execution_count: int | None = 3) -> dict[str, Any]:
    """Return the JSON of an *executed* notebook (a code cell with an execution count)."""
    return {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": execution_count,
                "metadata": {},
                "outputs": [],
                "source": ["print('ran')"],
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "spectraquant": {"run_id": run_id, "generated": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def make_bundle(
    root: Path,
    spec: RunSpec,
    *,
    document: dict[str, Any] | None = None,
    artifact_bytes: bytes | None = ARTIFACT_BYTES,
    notebook: bool = True,
    notebook_executed: bool = True,
    log: bool = True,
    teardown: bool = True,
) -> Path:
    """Write a bundle that :func:`spectraquant.cloud.collect.collect` should accept."""
    root.mkdir(parents=True, exist_ok=True)
    manifest = document if document is not None else make_manifest(spec)
    (root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if artifact_bytes is not None:
        target = root / ARTIFACT_NAME
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(artifact_bytes)
    if notebook:
        payload = executed_notebook(spec.run_id, execution_count=3 if notebook_executed else None)
        (root / "notebook").mkdir(parents=True, exist_ok=True)
        (root / "notebook" / f"{spec.run_id}.ipynb").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    if log:
        lines = ["installed pinned environment"]
        lines.append(f'SPECTRAQUANT_RESULT_JSON={{"run_id": "{spec.run_id}", "status": "success"}}')
        if teardown:
            lines.append(f'SPECTRAQUANT_TEARDOWN_OK {{"run_id": "{spec.run_id}"}}')
        (root / "logs").mkdir(parents=True, exist_ok=True)
        (root / "logs" / "run.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


@dataclass
class FakeAdapter:
    """A platform adapter with no network: it records the submission and copies a bundle."""

    bundle_source: Path
    registry: Registry
    remote_id: str = "fake/platform-run-1"
    state: str = "finished"
    submitted: list[str] = field(default_factory=list)
    submit_calls: int = 0

    name = "fake"

    def submit(self, spec: RunSpec, notebook_path: str) -> str:
        """Record the submission (persisting the remote id before returning)."""
        self.submit_calls += 1
        self.submitted.append(notebook_path)
        self.registry.record_submitted(
            spec,
            remote_id=self.remote_id,
            submitted_by="agent",
            notebook_digest=None,
            detail="fake adapter (no network)",
        )
        return self.remote_id

    def status(self, run_id: str) -> RunStatus:
        """Return the configured state."""
        return RunStatus(run_id=run_id, state=self.state, remote_id=self.remote_id)

    def fetch(self, run_id: str, dest_dir: str) -> FetchReport:
        """Copy the prepared bundle into ``dest_dir``."""
        target = Path(dest_dir)
        shutil.copytree(self.bundle_source, target, dirs_exist_ok=True)
        files = sorted(
            path.relative_to(target).as_posix() for path in target.rglob("*") if path.is_file()
        )
        return FetchReport(
            run_id=run_id,
            dest_dir=str(target),
            files=files,
            executed_notebook=next((name for name in files if name.endswith(".ipynb")), None),
            log=next((name for name in files if name.endswith(".log")), None),
        )

    def can_resume(self, run_id: str) -> bool:
        """True when the remote id is persisted."""
        return self.registry.remote_id(run_id) is not None
