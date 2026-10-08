"""SpectraQuant command line interface.

Subcommands:

* ``env`` — resolved device, hardware and software versions as JSON.
* ``smoke`` — the tiny deterministic end-to-end experiment plus a schema-valid run manifest.
* ``validate-manifest`` — JSON-Schema validation of any run manifest.

The CLI is the only supported entry point for recorded runs: it is what writes manifests.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from spectraquant import __version__
from spectraquant.config import load_experiment_config
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


if __name__ == "__main__":  # pragma: no cover - module entry point
    app()
