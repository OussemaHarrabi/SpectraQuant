"""Notebook stages: the repository-side work a cloud notebook delegates.

A generated notebook must be *thin* (``AGENTS.md`` §2b rule 1): it records the environment, checks out
the pinned commit, then calls repository code. On the notebook platforms that call cannot be an
in-process import, because the kernel interpreter is the *platform's* Python (observed: 3.13) while
the project pins 3.11 and a CPU torch build - installing the locked set into the kernel is impossible
and installing the platform's stack instead would silently replace the pinned environment.

So every stage runs as a subprocess through ``uv run --project <repo>``, i.e. in the pinned
environment, and communicates through a single JSON **stage context** file the notebook writes and
each stage extends:

    {"run_id": ..., "spec": {...}, "paths": {...}, "scalars": {...}, "datasets": [...], ...}

Stages, in order:

* ``datasets``  - materialise and checksum-verify the declared dataset splits.
* ``manifest``  - collect the run's own artifacts, build and validate ``run_manifest.json``.
* ``export``    - copy the bundle into the platform's exported directory and print the result line.

Each stage is idempotent and writes its result back into the context, so a stage can be re-run after
an interruption without repeating the earlier ones.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

__all__ = ["STAGES", "run_stage"]

#: The stages a notebook may request, in execution order.
STAGES: tuple[str, ...] = ("datasets", "manifest", "export")


def _load_context(path: str | Path) -> dict[str, Any]:
    """Read the stage context written by the notebook."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"stage context must be a JSON object, got {type(document).__name__}")
    for key in ("run_id", "spec", "paths"):
        if key not in document:
            raise ValueError(f"stage context is missing {key!r}")
    return document


def _save_context(path: str | Path, context: dict[str, Any]) -> None:
    """Write the stage context back, deterministically."""
    Path(path).write_text(json.dumps(context, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _stage_datasets(context: dict[str, Any]) -> str:
    """Materialise the declared datasets and record their checksums in the context."""
    from spectraquant.cloud.remote import materialise_datasets

    spec = context["spec"]
    paths = context["paths"]
    records = materialise_datasets(
        config_path=spec["experiment_config"],
        overrides=spec["overrides"],
        dataset_refs=spec["dataset_refs"],
        dest_dir=Path(paths["data"]),
        seed=int(spec["seeds"][0]),
    )
    context["datasets"] = records
    return f"datasets materialised: {len(records)} split(s)"


def _stage_manifest(context: dict[str, Any]) -> str:
    """Build, validate and write the run manifest, plus the cloud provenance sidecar."""
    from spectraquant.cloud.remote import (
        build_run_manifest,
        collect_artifact_checksums,
        find_run_manifest,
        write_run_manifest,
    )

    spec = context["spec"]
    paths = context["paths"]
    scalars = context.get("scalars", {})
    artifact_dir = Path(paths["artifact"])
    repo_dir = Path(paths["repo"])

    # The run's own outputs (its manifest, checkpoints, tables) are part of the bundle.
    repo_artifacts = repo_dir / "artifacts"
    if repo_artifacts.is_dir():
        shutil.copytree(repo_artifacts, artifact_dir / "artifacts", dirs_exist_ok=True)

    found = find_run_manifest(repo_dir, context["run_id"])
    run_manifest = json.loads(found.read_text()) if found else None
    run_rc = int(scalars.get("run_rc", 1))
    document = build_run_manifest(
        spec=spec,
        workdir=repo_dir,
        artifacts_dir=artifact_dir,
        status="success" if run_rc == 0 else "failed",
        started_utc=str(scalars["started_utc"]),
        finished_utc=str(scalars["finished_utc"]),
        environment=scalars.get("environment"),
        run_manifest=run_manifest,
        metrics={"cloud.wall_time_s": scalars.get("wall_time_s")},
        failure_reason=None
        if run_rc == 0
        else f"the declared runner exited {run_rc}: {str(scalars.get('run_out', ''))[-500:].strip()}",
        log_path=Path(paths["log"]) if paths.get("log") else None,
        remote_run_id=scalars.get("remote_run_id"),
        submitted_by=scalars.get("submitted_by"),
        notebook_digest=scalars.get("notebook_digest"),
        gpu_hours=scalars.get("gpu_hours"),
    )
    cloud_context = {
        "run_id": context["run_id"],
        "platform": spec["platform"],
        "spec_sha256": context.get("spec_sha256"),
        "notebook_digest": scalars.get("notebook_digest"),
        "notebook_template_version": scalars.get("notebook_template_version"),
        "git_commit": spec["git_commit"],
        "head_sha": scalars.get("head_sha"),
        "status": document.get("status"),
        "started_utc": scalars.get("started_utc"),
        "finished_utc": scalars.get("finished_utc"),
        "wall_time_s": scalars.get("wall_time_s"),
        "gpu_hours": scalars.get("gpu_hours"),
        "environment": scalars.get("environment"),
        "datasets": context.get("datasets", []),
        "artifact_checksums": collect_artifact_checksums(artifact_dir),
        "run_manifest_found": None if found is None else str(found),
        "install_spec": spec["install_spec"],
    }
    written = write_run_manifest(
        out_dir=artifact_dir, document=document, cloud_context=cloud_context
    )
    context["manifest_path"] = str(written)
    context["run_status"] = document.get("status")
    return f"manifest written: {written} (status {document.get('status')})"


def _stage_export(context: dict[str, Any]) -> str:
    """Copy the bundle into the exported directory and print the machine-readable result line."""
    from spectraquant.cloud.remote import collect_artifact_checksums

    paths = context["paths"]
    artifact_dir = Path(paths["artifact"])
    export_dir = Path(paths["export"])
    export_dir.mkdir(parents=True, exist_ok=True)

    for item in sorted(artifact_dir.rglob("*")):
        if item.is_file():
            relative = item.relative_to(artifact_dir)
            destination = export_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, destination)

    summary = {
        "run_id": context["run_id"],
        "status": context.get("run_status"),
        "manifest": context.get("manifest_path"),
        "artifact_checksums": collect_artifact_checksums(export_dir),
        "datasets": context.get("datasets", []),
    }
    (export_dir / "result.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("SPECTRAQUANT_RESULT_JSON=" + json.dumps(summary, sort_keys=True))
    return f"exported {len(summary['artifact_checksums'])} artifact(s) to {export_dir}"


_STAGE_FUNCTIONS = {
    "datasets": _stage_datasets,
    "manifest": _stage_manifest,
    "export": _stage_export,
}


def run_stage(stage: str, context_path: str | Path) -> str:
    """Run one stage against the context file and persist the context back.

    Args:
        stage: one of :data:`STAGES`.
        context_path: the JSON stage context the notebook wrote.

    Returns:
        A one-line human-readable summary (the notebook prints it).

    Raises:
        ValueError: the stage is unknown, or the context is malformed.
    """
    if stage not in _STAGE_FUNCTIONS:
        raise ValueError(f"unknown stage {stage!r}; known: {', '.join(STAGES)}")
    context = _load_context(context_path)
    summary = _STAGE_FUNCTIONS[stage](context)
    _save_context(context_path, context)
    return summary
