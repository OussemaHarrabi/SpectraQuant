"""Collect and validate existing A-0025 runs; never submit compute."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

from spectraquant.cloud.adapters.kaggle import parse_status_output
from spectraquant.cloud.collect import ValidationReport, collect
from spectraquant.cloud.registry import Registry
from spectraquant.cloud.spec import load_spec
from spectraquant.evaluation.phase4_report import (
    aggregate_transfer,
    verify_transfer,
    write_transfer_report,
)


def inspect_bundle(key: str):
    run = Path(f"artifacts/runs/sq-phase4-{key}-20261012")
    spec = load_spec(run / "spec.json")
    validation = collect(spec.run_id, run / "collected", spec=spec, record=False)
    if not validation.ok or validation.manifest is None:
        raise ValueError(validation.summary())
    paths = list((run / "collected").rglob(f"phase4-transfer-{key}/metrics.json"))
    if len(paths) != 1:
        raise ValueError("exactly one scientific report required")
    path = paths[0]
    report = json.loads(path.read_text(encoding="utf-8"))
    integrity = verify_transfer(report, path.parent, validation.manifest)
    if report["git"]["commit"] != spec.git_commit or report["model_key"] != key:
        raise ValueError("scientific source differs from frozen spec/model")
    return run, validation, report, integrity, path


def reject_collection(registry: Registry, run: Path, reason: str, remote: str | None) -> None:
    """Preserve copy/comparability errors without erasing historical acceptance."""
    rejected = ValidationReport(
        run_id=run.name,
        state="rejected",
        source_dir=str(run / "collected"),
        reasons=[reason],
        platform="kaggle",
        remote_id=remote,
    )
    run.mkdir(parents=True, exist_ok=True)
    (run / "validation-rejection.json").write_text(
        json.dumps(rejected.to_json(), indent=2) + "\n", encoding="utf-8"
    )
    if registry.state(run.name) != "validated":
        registry.record_rejected(rejected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("smol360", "pythia410"))
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--kaggle-cli", default="kaggle")
    args = parser.parse_args()
    registry = Registry("artifacts/runs/registry.jsonl")
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    if not args.aggregate:
        if not args.model:
            parser.error("--model required without --aggregate")
        run = Path(f"artifacts/runs/sq-phase4-{args.model}-20261012")
        submission = json.loads((run / "submission.json").read_text())
        remote = submission["remote_id"]
        result = subprocess.run(
            [args.kaggle_cli, "kernels", "status", remote], capture_output=True, text=True, env=env
        )
        state = parse_status_output(result.stdout + result.stderr)
        print(f"{remote}: {state}", flush=True)
        if result.returncode or state not in {"finished", "failed"}:
            return
        if args.fetch:
            dest = run / "collected"
            pattern = r"(run_manifest\.json$|metrics\.json$|scores\.json$|frozen\.json$|weights\.sqpack$|layout\.json$|unchanged\.pt$|\.ipynb$|\.log$)"
            subprocess.run(
                [
                    args.kaggle_cli,
                    "kernels",
                    "output",
                    remote,
                    "-p",
                    str(dest),
                    "--file-pattern",
                    pattern,
                ],
                check=True,
                env=env,
            )
            subprocess.run(
                [args.kaggle_cli, "kernels", "pull", remote, "-p", str(dest / "platform-notebook")],
                check=True,
                env=env,
            )
            expected = dest / "notebook" / f"{run.name}.ipynb"
            if not expected.exists():
                for path in dest.rglob("*.ipynb"):
                    document = json.loads(path.read_text(encoding="utf-8"))
                    if any(
                        c.get("cell_type") == "code" and c.get("execution_count") is not None
                        for c in document.get("cells", [])
                    ):
                        expected.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(path, expected)
                        break
        try:
            run, validation, report, integrity, _ = inspect_bundle(args.model)
        except (ValueError, OSError, KeyError) as exc:
            reject_collection(registry, run, str(exc), remote)
            raise
        if registry.state(run.name) == "validated":
            print(f"{run.name}: existing validated study preserved; local integrity rechecked")
            return
        if registry.state(run.name) not in {"collected", "rejected"}:
            registry.record_collected(
                run.name,
                platform="kaggle",
                remote_id=remote,
                detail="per-model integrity checked; cross-model comparison pending",
            )
        provisional = replace(
            validation,
            state="collected",
            reasons=["per-model integrity verified; cross-model fingerprints pending"],
        )
        (run / "validation.json").write_text(json.dumps(provisional.to_json(), indent=2) + "\n")
        print(
            json.dumps(
                {
                    "model": args.model,
                    "integrity": integrity,
                    "registry": "collected",
                    "transfer_result": "pending other model",
                },
                indent=2,
            )
        )
        return
    bundles = []
    for key in ("smol360", "pythia410"):
        try:
            bundles.append(inspect_bundle(key))
        except (ValueError, OSError, KeyError) as exc:
            run = Path(f"artifacts/runs/sq-phase4-{key}-20261012")
            reject_collection(registry, run, str(exc), registry.remote_id(run.name))
            raise
    reports = [bundle[2] for bundle in bundles]
    try:
        aggregate_transfer(reports)  # Identical fingerprints/config/scientific source.
    except ValueError as exc:
        for run, validation, *_ in bundles:
            reject_collection(registry, run, str(exc), validation.remote_id)
        raise
    for run, validation, report, integrity, path in bundles:
        if registry.state(run.name) not in {"collected", "validated", "rejected"}:
            registry.record_collected(run.name, platform="kaggle", remote_id=validation.remote_id)
        if registry.state(run.name) != "validated":
            registry.record_validated(validation)
        (run / "validation.json").write_text(json.dumps(validation.to_json(), indent=2) + "\n")
        target = Path("artifacts/sample-results/phase4-transfer") / report["model_key"]
        target.mkdir(parents=True, exist_ok=True)
        for name in ("metrics.json", "scores.json", "frozen.json"):
            shutil.copyfile(path.parent / name, target / name)
        (target / "collection.json").write_text(
            json.dumps(
                {
                    "integrity": integrity,
                    "manifest_sha256": validation.manifest_sha256,
                    "metrics_sha256": validation.manifest["metrics"]["phase4.full_report_sha256"],
                    "scores_sha256": integrity["authenticated_scores_sha256"],
                    "source_commit": validation.git_commit,
                    "spec_sha256": validation.spec_sha256,
                    "artifact_checksums_recorded": len(validation.artifact_checksums),
                    "declared_expected_artifacts": len(
                        load_spec(run / "spec.json").expected_artifacts
                    ),
                    "remote_id": validation.remote_id,
                    "cloud_timing": {
                        k: v
                        for k, v in validation.manifest["metrics"].items()
                        if k
                        in {
                            "cloud.started_utc",
                            "cloud.finished_utc",
                            "cloud.wall_time_s",
                            "cloud.gpu_hours",
                            "cloud.gpu_hours_semantics",
                        }
                    },
                    "scientific_software": validation.manifest["software"],
                    "scope": "extracted verified collection attestation, not original manifest or independent scientific rerun",
                },
                indent=2,
            )
            + "\n"
        )
    summary = write_transfer_report(reports, Path("docs/results/phase4-clipping-transfer.md"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
