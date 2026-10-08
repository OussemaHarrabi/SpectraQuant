"""SpectraQuant command line interface.

Subcommands:

* ``env`` — resolved device, hardware and software versions as JSON.
* ``smoke`` — the tiny deterministic end-to-end experiment plus a schema-valid run manifest.
* ``validate-manifest`` — JSON-Schema validation of any run manifest.
* ``compare-manifests`` — the equal-memory gate: two manifests are comparable at equal stored
  memory or the command exits non-zero (AGENTS.md section 4.5).
* ``cloud`` — the cloud execution substrate: notebook generation, submission, status, fetch,
  collection and the run registry (``docs/coordination/design-cloud-adapter.md`` §7).

The CLI is the only supported entry point for recorded runs: it is what writes manifests.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, NoReturn

import typer
from rich.console import Console
from rich.table import Table

from spectraquant import __version__
from spectraquant.cloud.adapters import adapter_for
from spectraquant.cloud.budget import assert_submission_allowed
from spectraquant.cloud.collect import collect
from spectraquant.cloud.notebook import cell_stages, write_notebook
from spectraquant.cloud.registry import Registry, default_registry_path, runs_dir
from spectraquant.cloud.secrets import install_redacting_handler
from spectraquant.cloud.spec import PLATFORMS, load_spec, save_spec, spec_from_config
from spectraquant.config import load_experiment_config
from spectraquant.reporting.comparability import (
    UnequalMemoryComparison,
    compare_manifest_files,
)
from spectraquant.reporting.environment import environment_report
from spectraquant.reporting.logging import configure_logging, get_logger, log_event
from spectraquant.reporting.manifests import (
    ManifestValidationError,
    validate_manifest_file,
)
from spectraquant.training.smoke import run_smoke_experiment

app = typer.Typer(
    name="spectraquant",
    help="Budget-aware joint low-rank and low-bit training for efficient transformers.",
    add_completion=False,
    no_args_is_help=True,
)
cloud_app = typer.Typer(
    name="cloud",
    help="Generate, submit, collect and validate cloud GPU runs (AGENTS.md section 2b).",
    add_completion=False,
    no_args_is_help=True,
)
app.add_typer(cloud_app, name="cloud")
console = Console()
error_console = Console(stderr=True)

DEFAULT_SMOKE_CONFIG = Path("configs/experiment/smoke.yaml")


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"spectraquant {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the package version and exit.",
    ),
) -> None:
    """SpectraQuant CLI."""


@app.command()
def env(
    as_json: bool = typer.Option(True, "--json/--no-json", help="Emit JSON (default)."),
) -> None:
    """Print the resolved environment: device, hardware, and software versions."""
    report = environment_report()
    if as_json:
        console.print_json(json.dumps(report))
    else:
        for section, values in report.items():
            console.print(f"[bold]{section}[/bold]")
            if isinstance(values, dict):
                for key, value in sorted(values.items()):
                    console.print(f"  {key}: {value}")
            else:
                console.print(f"  {values}")


@app.command()
def smoke(
    config: Path = typer.Option(
        DEFAULT_SMOKE_CONFIG,
        "--config",
        "-c",
        help="Experiment config to compose (relative paths resolve against <repo>/configs).",
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Directory for the run manifest; written as <run_id>.manifest.json. Default: the "
        "config's results_dir with the config's stable filename.",
    ),
    seed: int | None = typer.Option(None, "--seed", help="Override experiment.seed."),
    quiet: bool = typer.Option(False, "--quiet", help="Print only the manifest path."),
) -> None:
    """Run the tiny deterministic end-to-end experiment and write a validated manifest."""
    configure_logging()
    logger = get_logger("spectraquant.cli")
    overrides = [f"experiment.seed={seed}"] if seed is not None else []
    try:
        cfg = load_experiment_config(config, overrides)
        result = run_smoke_experiment(cfg, results_dir=out, verbose=not quiet)
    except (FileNotFoundError, ValueError, ManifestValidationError) as exc:
        log_event(logger, 40, "smoke_failed", fields={"error": f"{type(exc).__name__}: {exc}"})
        error_console.print(f"[red]smoke failed:[/red] {type(exc).__name__}: {exc}")
        raise typer.Exit(code=1) from exc

    if quiet:
        console.print(str(result.manifest_path), soft_wrap=True)
        return

    console.print(f"run_id: {result.run_id}")
    console.print(f"seed: {cfg.experiment.seed}  steps: {cfg.experiment.training.steps}")
    console.print(f"parameters: {result.parameter_count}")
    console.print(
        "loss sequence: " + " ".join(f"{loss:.6f}" for loss in result.loss_sequence),
        soft_wrap=True,
    )
    console.print(f"train loss: {result.initial_loss:.6f} -> {result.final_loss:.6f}")
    console.print(f"val loss: {result.val_loss:.6f}")
    console.print(f"tokens: {result.tokens_seen}  wall time: {result.wall_time_s:.3f}s")
    console.print(f"loss_sequence_sha256: {result.loss_sequence_sha256}", soft_wrap=True)
    if result.manifest_path is not None:
        console.print(f"manifest: {result.manifest_path}", soft_wrap=True)


@app.command("validate-manifest")
def validate_manifest(
    path: Path = typer.Argument(..., help="Path to a run manifest JSON file."),
    as_json: bool = typer.Option(False, "--json", help="Emit the result as JSON."),
) -> None:
    """Validate a run manifest against artifacts/schemas/run-manifest.schema.json."""
    try:
        document = validate_manifest_file(path)
    except ManifestValidationError as exc:
        if as_json:
            console.print_json(
                json.dumps({"valid": False, "path": str(path), "errors": exc.errors})
            )
        else:
            error_console.print(f"[red]INVALID[/red] {path}")
            for message in exc.errors:
                error_console.print(f"  - {message}")
        raise typer.Exit(code=1) from exc

    if as_json:
        console.print_json(
            json.dumps(
                {
                    "valid": True,
                    "path": str(path),
                    "run_id": document.get("run_id"),
                    "schema_version": document.get("schema_version"),
                }
            )
        )
    else:
        console.print(
            f"[green]OK[/green] {path} validates against run-manifest schema "
            f"v{document.get('schema_version')} (run_id={document.get('run_id')})"
        )


@app.command("compare-manifests")
def compare_manifests(
    run_a: Path = typer.Argument(..., help="Reference run manifest (JSON)."),
    run_b: Path = typer.Argument(..., help="Candidate run manifest (JSON)."),
    tolerance_bytes: int | None = typer.Option(
        None,
        "--tolerance-bytes",
        min=0,
        help="Maximum allowed |bytes_a - bytes_b| in bytes; overrides the tolerance declared in "
        "compression.measured_bytes_tolerance. When neither is given the comparison is refused.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit the verdict as JSON."),
) -> None:
    """Gate two run manifests at equal stored memory (AGENTS.md section 4.5)."""
    try:
        verdict = compare_manifest_files(run_a, run_b, tolerance_bytes)
    except (ManifestValidationError, UnequalMemoryComparison) as exc:
        if as_json:
            payload: dict[str, Any] = {
                "equal": False,
                "run_a": str(run_a),
                "run_b": str(run_b),
                "error": str(exc),
            }
            if isinstance(exc, UnequalMemoryComparison):
                payload.update(
                    {
                        "bytes_a": exc.bytes_a,
                        "bytes_b": exc.bytes_b,
                        "difference_bytes": exc.difference_bytes,
                        "tolerance_bytes": exc.tolerance_bytes,
                        "bytes_source": exc.bytes_source,
                    }
                )
            console.print_json(json.dumps(payload))
        else:
            error_console.print(f"[red]REJECTED[/red] {run_a} vs {run_b}")
            error_console.print(f"  {exc}", soft_wrap=True)
        raise typer.Exit(code=1) from exc

    if as_json:
        console.print_json(json.dumps(verdict.to_json_dict()))
    else:
        console.print(f"[green]OK[/green] {verdict.summary()}", soft_wrap=True)


# --------------------------------------------------------------------------------------
# `spectraquant cloud ...` — the cloud execution substrate
# --------------------------------------------------------------------------------------
def _cloud_fail(exc: BaseException) -> NoReturn:
    """Print a redacted failure and exit non-zero."""
    error_console.print(f"[red]cloud failed:[/red] {type(exc).__name__}: {exc}")
    raise typer.Exit(code=1) from exc


def _registry() -> Registry:
    install_redacting_handler()
    return Registry(default_registry_path())


@cloud_app.command("notebook")
def cloud_notebook(
    config: Path = typer.Option(
        ..., "--config", "-c", help="Experiment config under configs/experiment/."
    ),
    platform: str = typer.Option("colab", "--platform", help=f"One of: {', '.join(PLATFORMS)}."),
    out: Path | None = typer.Option(
        None, "--out", help="Notebook path; default notebooks/generated/<run_id>.ipynb."
    ),
    run_id: str | None = typer.Option(None, "--run-id", help="Explicit run id (slug)."),
    override: list[str] = typer.Option(
        None, "--override", "-o", help="Hydra override (repeatable)."
    ),
    gpu_required: bool | None = typer.Option(None, "--gpu/--no-gpu", help="Needs a CUDA device."),
    timeout_minutes: int | None = typer.Option(
        None, "--timeout-minutes", help="Wall-clock ceiling."
    ),
    max_cost: float | None = typer.Option(
        None, "--max-cost-usd", help="Authorized spend ceiling (required for paid platforms)."
    ),
    install_spec: str | None = typer.Option(None, "--install-spec", help="Pinned install command."),
    repo_url: str | None = typer.Option(None, "--repo-url", help="Clone URL of this repository."),
    allow_dirty: bool = typer.Option(
        False, "--allow-dirty", help="Permit a dirty tree (not reproducible; deliberate only)."
    ),
    spec_out: Path | None = typer.Option(
        None, "--spec-out", help="Where to write the RunSpec JSON (default beside the notebook)."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit the summary as JSON."),
) -> None:
    """Generate the execution-ready notebook (and its RunSpec) for an experiment config."""
    install_redacting_handler()
    try:
        spec = spec_from_config(
            config,
            platform=platform,
            run_id=run_id,
            overrides=list(override or []),
            gpu_required=gpu_required,
            timeout_minutes=timeout_minutes,
            max_cost_authorized_usd=max_cost,
            install_spec=install_spec,
            repo_url=repo_url,
            allow_dirty=allow_dirty or None,
        )
        target = out if out is not None else spec.notebook_path()
        digest = write_notebook(spec, target)
        spec_path = spec_out if spec_out is not None else Path(f"{target}.spec.json")
        save_spec(spec, spec_path)
    except Exception as exc:
        _cloud_fail(exc)

    stages = cell_stages(spec)
    if as_json:
        console.print_json(
            json.dumps(
                {
                    "run_id": spec.run_id,
                    "platform": spec.platform,
                    "notebook": str(target),
                    "spec": str(spec_path),
                    "digest": digest,
                    "spec_sha256": spec.sha256(),
                    "cells": stages,
                }
            )
        )
        return
    console.print(f"run_id: {spec.run_id}")
    console.print(f"platform: {spec.platform}  gpu_required: {spec.gpu_required}")
    console.print(f"notebook: {target}", soft_wrap=True)
    console.print(f"spec: {spec_path}", soft_wrap=True)
    console.print(f"notebook_digest: {digest}", soft_wrap=True)
    console.print(f"spec_sha256: {spec.sha256()}", soft_wrap=True)
    for cell in stages:
        console.print(
            f"  cell {cell['index']}: {cell['stage']} ({cell['cell_type']})", soft_wrap=True
        )


@cloud_app.command("submit")
def cloud_submit(
    spec_path: Path = typer.Option(
        ..., "--spec", help="RunSpec JSON produced by `cloud notebook`."
    ),
    notebook: Path | None = typer.Option(
        None, "--notebook", help="Notebook to submit; generated when missing."
    ),
) -> None:
    """Submit a spec through its platform adapter (Kaggle/Colab Enterprise/local_cpu)."""
    install_redacting_handler()
    try:
        spec = load_spec(spec_path)
        # Refuse an unauthorized paid submission *before* emitting anything.
        assert_submission_allowed(spec, os.environ)
        target = notebook if notebook is not None else spec.notebook_path()
        if not target.is_file():
            write_notebook(spec, target)
        adapter = adapter_for(spec.platform)
        remote_id = adapter.submit(spec, str(target))
    except Exception as exc:
        _cloud_fail(exc)
    console.print(f"submitted: {spec.run_id}")
    console.print(f"platform: {spec.platform}")
    console.print(f"remote_id: {remote_id}", soft_wrap=True)
    console.print(f"notebook: {target}", soft_wrap=True)


@cloud_app.command("status")
def cloud_status(
    run_id: str = typer.Option(..., "--run-id", help="Run id to query."),
    as_json: bool = typer.Option(False, "--json", help="Emit the status as JSON."),
) -> None:
    """Query the platform for a run's status (bounded; never blocks indefinitely)."""
    install_redacting_handler()
    registry = _registry()
    platform = (registry.latest(run_id) or {}).get("platform")
    if platform is None:
        _cloud_fail(ValueError(f"run {run_id!r} is not in the registry ({registry.path})"))
    try:
        status = adapter_for(str(platform)).status(run_id)
    except Exception as exc:
        _cloud_fail(exc)
    if as_json:
        console.print_json(
            json.dumps(
                {
                    "run_id": status.run_id,
                    "state": status.state,
                    "remote_id": status.remote_id,
                    "detail": status.detail,
                }
            )
        )
        return
    console.print(f"run_id: {status.run_id}")
    console.print(f"platform: {platform}")
    console.print(f"state: {status.state}")
    console.print(f"remote_id: {status.remote_id}")
    if status.detail:
        console.print(f"detail: {status.detail}", soft_wrap=True)


@cloud_app.command("fetch")
def cloud_fetch(
    run_id: str = typer.Option(..., "--run-id", help="Run id to download."),
    dest: Path = typer.Option(..., "--dest", help="Destination directory for the bundle."),
    as_json: bool = typer.Option(False, "--json", help="Emit the fetch report as JSON."),
) -> None:
    """Download logs, artifacts and the executed notebook from the platform."""
    install_redacting_handler()
    registry = _registry()
    platform = (registry.latest(run_id) or {}).get("platform")
    if platform is None:
        _cloud_fail(ValueError(f"run {run_id!r} is not in the registry ({registry.path})"))
    try:
        report = adapter_for(str(platform)).fetch(run_id, str(dest))
    except Exception as exc:
        _cloud_fail(exc)
    if as_json:
        console.print_json(json.dumps(report.to_json()))
    else:
        console.print(f"run_id: {report.run_id}  ok: {report.ok}")
        console.print(f"dest: {report.dest_dir}", soft_wrap=True)
        console.print(f"files: {len(report.files)}")
        console.print(f"executed_notebook: {report.executed_notebook}", soft_wrap=True)
        if report.detail:
            console.print(f"detail: {report.detail}", soft_wrap=True)
    if not report.ok:
        raise typer.Exit(code=1)


@cloud_app.command("collect")
def cloud_collect(
    run_id: str = typer.Option(..., "--run-id", help="Run id to collect."),
    source: Path = typer.Option(..., "--source", help="Directory holding the downloaded bundle."),
    spec_path: Path | None = typer.Option(
        None, "--spec", help="RunSpec JSON; defaults to the spec recorded at submission."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit the validation report as JSON."),
) -> None:
    """Validate a downloaded bundle; only full success records `validated`."""
    install_redacting_handler()
    try:
        spec = load_spec(spec_path) if spec_path is not None else None
        report = collect(run_id, source, spec=spec, registry=_registry())
    except Exception as exc:
        _cloud_fail(exc)
    if as_json:
        console.print_json(json.dumps(report.to_json()))
    else:
        style = "green" if report.ok else "red"
        console.print(f"[{style}]{report.summary()}[/{style}]", soft_wrap=True)
        if report.executed_notebook:
            console.print(f"executed notebook: {report.executed_notebook}", soft_wrap=True)
        console.print(f"manifest: {report.manifest_path}", soft_wrap=True)
    if not report.ok:
        raise typer.Exit(code=1)


@cloud_app.command("registry")
def cloud_registry(
    run_id: str | None = typer.Option(None, "--run-id", help="Show one run's transitions."),
    as_json: bool = typer.Option(False, "--json", help="Emit the registry as JSON."),
) -> None:
    """Show the append-only run registry (`--status` is the default view)."""
    install_redacting_handler()
    registry = _registry()
    records = registry.transitions(run_id) if run_id is not None else registry.status_summary()
    if as_json:
        console.print_json(json.dumps({"path": str(registry.path), "records": records}))
        return
    console.print(f"registry: {registry.path}  runs_dir: {runs_dir()}", soft_wrap=True)
    if not records:
        console.print("no runs recorded")
        return
    table = Table("run_id", "state", "platform", "remote_id", "when", "failure")
    for record in records:
        table.add_row(
            str(record.get("run_id")),
            str(record.get("state")),
            str(record.get("platform") or "-"),
            str(record.get("remote_id") or "-"),
            str(record.get("timestamp_utc") or "-"),
            str(record.get("failure_reason") or "-")[:60],
        )
    console.print(table)


if __name__ == "__main__":  # pragma: no cover - module entry point
    app()
