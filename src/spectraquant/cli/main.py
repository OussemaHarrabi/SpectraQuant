"""SpectraQuant command line interface.

Subcommands:

* ``env`` — resolved device, hardware and software versions as JSON.
* ``smoke`` — the tiny deterministic end-to-end experiment plus a schema-valid run manifest.
* ``run-plan`` — execute a frozen cloud plan's implemented arms on the remote substrate and write
  schema-validated per-arm manifests plus ``metrics.json`` (``docs/coordination/design-cloud-adapter.md``
  §11).
* ``validate-manifest`` — JSON-Schema validation of any run manifest.
* ``compare-manifests`` — the equal-memory gate: two manifests are comparable at equal stored
  memory or the command exits non-zero (AGENTS.md section 4.5).
* ``registry-validate`` — the Milestone-7 result-registry gate: a directory of run manifests becomes
  a comparability cell matrix (expected vs present seeds, duplicates, interrupted runs, schema-invalid
  files, incomparable cells) with every arm pair gated at equal memory; exits non-zero on a blocking
  problem and zero for warnings only.
* ``claim-guard`` — the measurement-integrity gate over prose: scans documents for forbidden claim
  patterns and exits non-zero on an unmarked violation (AGENTS.md sections 4 and 5).
* ``cloud`` — the cloud execution substrate: notebook generation (from a Hydra config **or** a frozen
  plan), spec emission, submission, status, fetch, collection and the run registry
  (``docs/coordination/design-cloud-adapter.md`` §7).

The CLI is the only supported entry point for recorded runs: it is what writes manifests.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, NoReturn

import typer
from rich.console import Console
from rich.markup import escape as markup_escape
from rich.table import Table

from spectraquant import __version__
from spectraquant.cloud.adapters import adapter_for
from spectraquant.cloud.budget import assert_submission_allowed
from spectraquant.cloud.collect import collect
from spectraquant.cloud.notebook import cell_stages, write_notebook
from spectraquant.cloud.plan_runner import run_plan
from spectraquant.cloud.registry import Registry, default_registry_path, runs_dir
from spectraquant.cloud.secrets import install_redacting_handler
from spectraquant.cloud.spec import (
    PLATFORMS,
    load_spec,
    plan_to_run_spec,
    save_spec,
    spec_from_config,
)
from spectraquant.config import load_experiment_config
from spectraquant.experiment_plan import load_plan
from spectraquant.reporting.claim_guard import (
    default_document_paths,
    scan_paths,
)
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
from spectraquant.reporting.run_registry import validate_registry
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


@app.command("run-plan")
def run_plan_command(
    plan: Path = typer.Option(
        ..., "--plan", help="Frozen cloud plan under configs/{tier1,tier2,repro}/."
    ),
    seed: list[int] | None = typer.Option(
        None, "--seed", help="Plan seed to run (repeatable); default: the first reported seed."
    ),
    out: Path | None = typer.Option(
        None, "--out", help="Output root; default artifacts/runs/<plan>-plan."
    ),
    arms: str | None = typer.Option(
        None, "--arms", help="Comma-separated arm names; default: every implemented arm."
    ),
    bits: int | None = typer.Option(None, "--bits", help="Bit width for the quantized arms."),
    rank: int | None = typer.Option(None, "--rank", help="Rank for the low-rank arms."),
    granularity: str = typer.Option(
        "per_channel", "--granularity", help="per_tensor | per_channel | per_group."
    ),
    group_size: int | None = typer.Option(None, "--group-size", help="Required for per_group."),
    seq_len: int = typer.Option(2048, "--seq-len", help="Perplexity window length in tokens."),
    max_tokens: int | None = typer.Option(None, "--max-tokens", help="Cap scored tokens (smoke)."),
    max_documents: int | None = typer.Option(None, "--max-documents", help="Cap documents."),
    dataset_config: str | None = typer.Option(
        None, "--dataset-config", help="Dataset config name for the perplexity split."
    ),
    threads: int = typer.Option(1, "--threads", help="torch.set_num_threads value."),
    device: str = typer.Option("cpu", "--device", help="cpu (local) or cuda (cloud GPU)."),
    model_dir: Path | None = typer.Option(
        None, "--model-dir", help="Local stand-in model; requires --allow-local-substitution."
    ),
    perplexity_text: Path | None = typer.Option(
        None,
        "--perplexity-text",
        help="Local stand-in corpus; requires --allow-local-substitution.",
    ),
    allow_local_substitution: bool = typer.Option(
        False,
        "--allow-local-substitution",
        help="Acknowledge that --model-dir/--perplexity-text replace the plan's pinned assets, so "
        "this run is a path check and not a plan measurement.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit the run summary as JSON."),
) -> None:
    """Execute a frozen plan's implemented arms and write schema-valid manifests."""
    install_redacting_handler()
    chosen_arms = [item.strip() for item in arms.split(",") if item.strip()] if arms else None
    try:
        result = run_plan(
            plan,
            seeds=seed,
            out_dir=out,
            arms=chosen_arms,
            bits=bits,
            rank=rank,
            granularity=granularity,
            group_size=group_size,
            seq_len=seq_len,
            max_tokens=max_tokens,
            max_documents=max_documents,
            dataset_config=dataset_config,
            threads=threads,
            device=device,
            model_dir=model_dir,
            perplexity_text=perplexity_text,
            allow_local_substitution=allow_local_substitution,
            progress=lambda message: error_console.print(f"[dim]{message}[/dim]", soft_wrap=True),
        )
    except NotImplementedError as exc:
        error_console.print(f"[red]not implemented:[/red] {markup_escape(str(exc))}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        error_console.print(
            f"[red]run-plan failed:[/red] {markup_escape(f'{type(exc).__name__}: {exc}')}"
        )
        raise typer.Exit(code=1) from exc

    if as_json:
        console.print_json(json.dumps(result.metrics))
        return
    console.print(f"plan: {result.plan_name}  model: {result.model_id}@{result.model_revision}")
    console.print(f"out: {result.out_dir}", soft_wrap=True)
    table = Table("arm", "seed", "class", "perplexity", "accounted bytes", "manifest")
    for run in result.runs:
        table.add_row(
            run.arm,
            str(run.seed),
            "-" if run.measurement_class is None else str(run.measurement_class),
            "not measured" if run.perplexity is None else f"{run.perplexity:.6f}",
            str(run.accounted_bytes),
            str(run.manifest_path),
        )
    console.print(table)
    if result.skipped:
        console.print("[yellow]skipped arms:[/yellow]")
        for entry in result.skipped:
            console.print(f"  {entry['arm']} ({entry['kind']}): {entry['reason']}", soft_wrap=True)
    if result.aggregate_metrics_path is not None:
        console.print(f"metrics: {result.aggregate_metrics_path}", soft_wrap=True)


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


@app.command("registry-validate")
def registry_validate(
    runs: Path | None = typer.Option(
        None,
        "--runs",
        help="Directory of run manifests (*manifest.json). Default: the runs directory the cloud "
        "bundles are fetched into ($SPECTRAQUANT_RUNS_DIR or <repo>/artifacts/runs).",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit the deterministic JSON index."),
    out: Path | None = typer.Option(
        None, "--out", help="Also write the JSON index to this path (parent directories created)."
    ),
) -> None:
    """Turn a directory of run manifests into a comparability cell matrix (Milestone 7 gate).

    Reports, per cell: expected vs present seeds and the missing ones, duplicate (cell, seed) runs,
    interrupted/aborted runs, schema-invalid or unparseable files, cells that differ in a frozen
    comparability field (incomparable, so never pooled) and every arm pair the equal-memory gate
    refuses. Note what the exit code means: non-zero only for a blocking problem - a missing seed for
    a declared cell, a duplicate run, a schema-invalid/unparseable manifest, or an unequal-memory arm
    pair; a directory that is only a fixture, a warning, or an absent runs directory exits zero.
    """
    target = runs if runs is not None else runs_dir()
    report = validate_registry(target)

    if not as_json:
        style = "green" if report.ok else "red"
        console.print(f"registry: {report.runs_dir}", soft_wrap=True)
        console.print(f"[{style}]{report.summary_line()}[/{style}]", soft_wrap=True)

        if report.cells:
            table = Table(
                "cell",
                "substrate/hardware",
                "seeds p/e",
                "missing",
                "runs",
                "dupes",
                "problems",
            )
            for row in report.cell_rows():
                table.add_row(*(markup_escape(item) for item in row))
            console.print(table)
        else:
            console.print("no cells: no manifest in this directory is a run")

        if report.fixtures:
            console.print(
                "fixtures (declared non-runs; excluded from the cell matrix, their arm pairs are "
                "gated at warning severity):"
            )
            fixture_table = Table("fixture", "run_id", "cell key")
            for row in report.fixture_rows():
                fixture_table.add_row(*(markup_escape(item) for item in row))
            console.print(fixture_table)

        for directory in report.directories:
            if directory.fixture_only:
                console.print(
                    f"[dim]{markup_escape(directory.path)}: fixture-only "
                    f"({directory.fixtures} declared fixture(s), no run) - not a missing cell[/dim]",
                    soft_wrap=True,
                )

        if report.findings:
            findings_table = Table("severity", "code", "detail")
            for severity, code, detail in report.finding_rows():
                colour = "red" if severity == "blocking" else "yellow"
                findings_table.add_row(
                    f"[{colour}]{severity}[/{colour}]",
                    markup_escape(code),
                    markup_escape(detail),
                )
            console.print(findings_table)

    payload = json.dumps(report.to_json_dict(), indent=2, sort_keys=True) + "\n"
    if as_json:
        console.print_json(payload)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")
        if not as_json:
            console.print(f"json: {out}", soft_wrap=True)

    if not report.ok:
        raise typer.Exit(code=1)


@app.command("claim-guard")
def claim_guard(
    paths: list[Path] | None = typer.Argument(
        None,
        help="Files or directories to scan; directories expand to their *.md files. Default: "
        "README.md, AGENTS.md, docs/**/*.md, reports/**/*.md.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit the structured scan report as JSON."),
) -> None:
    """Scan prose for forbidden measurement claims (Milestone 9 integrity gate).

    Exits non-zero when any violation is found. Quoted rules (forbidden-claims tables, prohibition
    lists, attributed literature figures) are reported as quotes and do not fail the gate.
    """
    targets = [Path(p) for p in paths] if paths else list(default_document_paths())
    report = scan_paths(targets)

    if as_json:
        console.print_json(json.dumps(report.to_json_dict()))
    else:
        if report.findings:
            table = Table("document", "line", "pattern", "verdict", "context")
            for finding in report.findings:
                style = "red" if finding.verdict == "violation" else "yellow"
                table.add_row(
                    finding.document,
                    str(finding.line),
                    finding.pattern,
                    f"[{style}]{finding.verdict}[/{style}]",
                    markup_escape(finding.context[:160]),
                )
            console.print(table)
        if report.skipped:
            console.print("skipped (not violations):", soft_wrap=True)
            for reason, count in report.skipped.items():
                console.print(f"  {count:>4}  {reason}")
        style = "red" if report.violations else "green"
        console.print(f"[{style}]{report.summary()}[/{style}]", soft_wrap=True)

    if report.violations:
        raise typer.Exit(code=1)


# --------------------------------------------------------------------------------------
# `spectraquant cloud ...` — the cloud execution substrate
# --------------------------------------------------------------------------------------
def _cloud_fail(exc: BaseException) -> NoReturn:
    """Print a redacted failure and exit non-zero."""
    # Escape the message: rich would otherwise eat bracketed text such as 'spectraquant[cloud]'.
    error_console.print(f"[red]cloud failed:[/red] {markup_escape(f'{type(exc).__name__}: {exc}')}")
    raise typer.Exit(code=1) from exc


def _registry() -> Registry:
    install_redacting_handler()
    return Registry(default_registry_path())


@cloud_app.command("notebook")
def cloud_notebook(
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Experiment config under configs/experiment/ (exactly one of --config/--plan).",
    ),
    plan: Path | None = typer.Option(
        None,
        "--plan",
        help="Frozen cloud plan under configs/{tier1,tier2,repro}/ (exactly one of --config/--plan).",
    ),
    runner_arg: list[str] | None = typer.Option(
        None,
        "--runner-arg",
        help="Extra flag appended to the plan's runner command (repeatable), e.g. "
        "--runner-arg --max-tokens=20000. Each must start with '--'.",
    ),
    platform: str | None = typer.Option(
        None,
        "--platform",
        help=f"One of: {', '.join(PLATFORMS)}. Default: the plan's substrate, else colab.",
    ),
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
    """Generate the execution-ready notebook (and its RunSpec) for a config or a frozen plan."""
    install_redacting_handler()
    if (config is None) == (plan is None):
        _cloud_fail(
            ValueError(
                "exactly one of --config (a Hydra experiment config) or --plan (a frozen cloud "
                "plan) is required"
            )
        )
    try:
        if plan is not None:
            _reject_plan_fixed_options(
                run_id=run_id,
                overrides=list(override or []),
                gpu_required=gpu_required,
                timeout_minutes=timeout_minutes,
                max_cost=max_cost,
                install_spec=install_spec,
            )
            spec = plan_to_run_spec(
                plan,
                platform=platform or load_plan(plan).substrate,
                repo_url=repo_url,
                allow_dirty=allow_dirty or None,
                runner_args=tuple(runner_arg or ()),
            )
        else:
            assert config is not None  # narrowed by the XOR check above
            spec = spec_from_config(
                config,
                platform=platform or "colab",
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
    if spec.runner_command:
        console.print(f"runner: {spec.runner_command}", soft_wrap=True)
    console.print(f"notebook: {target}", soft_wrap=True)
    console.print(f"spec: {spec_path}", soft_wrap=True)
    console.print(f"notebook_digest: {digest}", soft_wrap=True)
    console.print(f"spec_sha256: {spec.sha256()}", soft_wrap=True)
    for cell in stages:
        console.print(
            f"  cell {cell['index']}: {cell['stage']} ({cell['cell_type']})", soft_wrap=True
        )


def _reject_plan_fixed_options(
    *,
    run_id: str | None,
    overrides: list[str],
    gpu_required: bool | None,
    timeout_minutes: int | None,
    max_cost: float | None,
    install_spec: str | None,
) -> None:
    """Refuse options a frozen plan already fixes (silently ignoring them would misdescribe the run)."""
    fixed = {
        "--run-id": run_id,
        "--override": overrides or None,
        "--gpu/--no-gpu": gpu_required,
        "--timeout-minutes": timeout_minutes,
        "--max-cost-usd": max_cost,
        "--install-spec": install_spec,
    }
    supplied = [name for name, value in fixed.items() if value is not None]
    if supplied:
        raise ValueError(
            f"{', '.join(supplied)} may not be used with --plan: the plan freezes the run id, the "
            "grid, the substrate's GPU requirement, the cost envelope and the pinned environment "
            "(AGENTS.md §2b rule 8). Edit the plan through a preregistration amendment instead."
        )


@cloud_app.command("plan-spec")
def cloud_plan_spec(
    plan: Path = typer.Option(..., "--plan", help="Frozen cloud plan to map into a RunSpec."),
    runner_arg: list[str] | None = typer.Option(
        None, "--runner-arg", help="Extra flag appended to the plan's runner command (repeatable)."
    ),
    platform: str | None = typer.Option(
        None, "--platform", help=f"One of: {', '.join(PLATFORMS)}. Default: the plan's substrate."
    ),
    out: Path | None = typer.Option(None, "--out", help="Also write the RunSpec JSON here."),
    repo_url: str | None = typer.Option(None, "--repo-url", help="Clone URL of this repository."),
    allow_dirty: bool = typer.Option(
        False, "--allow-dirty", help="Permit a dirty tree (not reproducible; deliberate only)."
    ),
    as_json: bool = typer.Option(True, "--json/--no-json", help="Emit the spec as JSON (default)."),
) -> None:
    """Emit the ``RunSpec`` a frozen plan maps to (the same document ``cloud notebook --plan`` writes)."""
    install_redacting_handler()
    try:
        resolved_platform = platform or load_plan(plan).substrate
        spec = plan_to_run_spec(
            plan,
            platform=resolved_platform,
            repo_url=repo_url,
            allow_dirty=allow_dirty or None,
            runner_args=tuple(runner_arg or ()),
        )
        if out is not None:
            save_spec(spec, out)
    except Exception as exc:
        _cloud_fail(exc)

    if as_json:
        typer.echo(json.dumps(spec.to_json(), indent=2, sort_keys=True))
        return
    console.print(f"run_id: {spec.run_id}")
    console.print(f"platform: {spec.platform}  gpu_required: {spec.gpu_required}")
    console.print(f"runner: {spec.runner_command}")
    console.print(f"timeout_minutes: {spec.timeout_minutes}")
    console.print(f"max_cost_authorized_usd: {spec.max_cost_authorized_usd}")
    console.print(f"measurement_class_expected: {spec.measurement_class_expected}")
    console.print(f"seeds: {spec.seeds}")
    for ref in spec.dataset_refs:
        console.print(f"  dataset {ref.name}@{ref.revision} ({ref.split})", soft_wrap=True)
    for artifact in spec.expected_artifacts:
        console.print(f"  expects {artifact.name} (min {artifact.min_bytes} B)", soft_wrap=True)
    if out is not None:
        console.print(f"spec: {out}", soft_wrap=True)


@cloud_app.command("notebook-stage")
def cloud_notebook_stage(
    stage: str = typer.Option(
        ...,
        "--stage",
        help="Stage to run: datasets | manifest | export.",
    ),
    context: Path = typer.Option(
        ...,
        "--context",
        help="JSON stage context written by the generated notebook (extended in place).",
    ),
) -> None:
    """Run one repository stage for a generated notebook, inside the pinned environment.

    A generated notebook never imports repository code into the platform's interpreter (which is a
    different Python than the project pins). Each stage runs as `uv run --project <repo> spectraquant
    cloud notebook-stage ...`, so the science uses the locked environment while the notebook stays
    thin (AGENTS.md section 2b rule 1).
    """
    from spectraquant.cloud.notebook_stages import run_stage

    try:
        summary = run_stage(stage, context)
    except Exception as exc:
        error_console.print(f"[red]stage failed[/red] {stage}: {exc}")
        raise typer.Exit(code=1) from exc
    console.print(summary)


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
