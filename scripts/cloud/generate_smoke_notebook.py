"""Regenerate the committed example cloud notebook for ``configs/experiment/smoke.yaml``.

This is the reproducible way to produce ``notebooks/generated/smoke-cloud-example.ipynb`` and its
sidecar ``.spec.json``: the generator is a pure function of ``(spec, template version)``, so running
this script on the same commit reproduces the same notebook digest.

    uv run python scripts/cloud/generate_smoke_notebook.py
    uv run python scripts/cloud/generate_smoke_notebook.py --platform kaggle --json

``--allow-dirty`` is required while the working tree has uncommitted changes: a cloud run must
otherwise execute an exact commit (``AGENTS.md`` §2b, ``design-cloud-adapter.md`` §2).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from spectraquant.cloud.notebook import cell_stages, write_notebook
from spectraquant.cloud.spec import save_spec, spec_from_config
from spectraquant.paths import repo_root

app = typer.Typer(add_completion=False, help=__doc__)

ConfigOption = Annotated[Path, typer.Option("--config", "-c", help="Experiment config to compose.")]
PlatformOption = Annotated[
    str, typer.Option("--platform", help="colab | kaggle | colab_enterprise | local_cpu.")
]
RunIdOption = Annotated[str, typer.Option("--run-id", help="Stable run id (slug).")]
OutOption = Annotated[
    Path, typer.Option("--out", help="Directory for the notebook and its RunSpec sidecar.")
]
AllowDirtyOption = Annotated[
    bool,
    typer.Option(
        "--allow-dirty/--require-clean",
        help="Permit a dirty tree (documented, deliberate) instead of refusing.",
    ),
]
JsonOption = Annotated[bool, typer.Option("--json", help="Emit the summary as JSON.")]


def _display(path: Path) -> str:
    """Return the path relative to the repository root when possible."""
    try:
        return path.resolve().relative_to(repo_root()).as_posix()
    except ValueError:
        return str(path)


@app.command()
def main(
    config: ConfigOption = Path("configs/experiment/smoke.yaml"),
    platform: PlatformOption = "colab",
    run_id: RunIdOption = "smoke-cloud-example",
    out: OutOption = Path("notebooks/generated"),
    allow_dirty: AllowDirtyOption = True,
    as_json: JsonOption = False,
) -> None:
    """Write the example notebook plus its RunSpec and print the digest and cell list."""
    from spectraquant.cloud.spec import ExpectedArtifact
    from spectraquant.config import load_experiment_config

    cfg = load_experiment_config(config)
    # The run's own manifest is the artifact the smoke run must produce; the cloud manifest is
    # written by the notebook itself. Collection verifies both.
    run_manifest_artifact = (
        f"{cfg.experiment.output.results_dir.strip('/')}/{cfg.experiment.output.filename}"
    )
    spec = spec_from_config(
        config,
        platform=platform,
        run_id=run_id,
        allow_dirty=allow_dirty or None,
        expected_artifacts=[
            ExpectedArtifact(name="run_manifest.json", min_bytes=64),
            ExpectedArtifact(name=run_manifest_artifact, min_bytes=64),
        ],
    )
    target = Path(out) / f"{run_id}.ipynb"
    digest = write_notebook(spec, target)
    spec_path = save_spec(spec, Path(f"{target}.spec.json"))
    stages = cell_stages(spec)
    summary = {
        "run_id": spec.run_id,
        "platform": spec.platform,
        "git_commit": spec.git_commit,
        "allow_dirty": spec.allow_dirty,
        "notebook": _display(target),
        "spec": _display(spec_path),
        "notebook_digest": digest,
        "spec_sha256": spec.sha256(),
        "cells": stages,
    }
    if as_json:
        typer.echo(json.dumps(summary, indent=2))
        return
    for key in (
        "run_id",
        "platform",
        "git_commit",
        "notebook",
        "spec",
        "notebook_digest",
        "spec_sha256",
    ):
        typer.echo(f"{key}: {summary[key]}")
    for cell in stages:
        typer.echo(
            f"  cell {cell['index']}: {cell['stage']} ({cell['cell_type']}, {cell['chars']} chars)"
        )


if __name__ == "__main__":
    app()
