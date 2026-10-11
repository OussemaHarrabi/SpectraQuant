"""Resume status/collection of an existing Phase2 run; never submit compute."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from spectraquant.cloud.adapters.kaggle import parse_status_output
from spectraquant.cloud.collect import collect
from spectraquant.cloud.registry import Registry
from spectraquant.cloud.spec import load_spec
from spectraquant.evaluation.phase2_report import (
    summarize,
    verify_collected_study,
    write_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--study", choices=("phase2", "phase3"), default="phase2")
    parser.add_argument("--kaggle-cli", default="kaggle")
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--analyze-local", action="store_true")
    args = parser.parse_args()
    args.run_id = args.run_id or (
        "sq-phase2-global-20261011" if args.study == "phase2" else "sq-phase3-controls-20261011"
    )
    run_root = Path("artifacts/runs") / args.run_id
    spec = load_spec(run_root / "spec.json")
    submission = json.loads((run_root / "submission.json").read_text(encoding="utf-8"))
    remote = submission["remote_id"]
    result = subprocess.run(
        [args.kaggle_cli, "kernels", "status", remote], capture_output=True, text=True, check=False
    )
    state = parse_status_output(result.stdout + result.stderr)
    print(f"{remote}: {state}")
    if result.returncode or state not in {"finished", "failed"}:
        print("No completed result is claimed; resume this command after the remote run finishes.")
        return
    if not args.fetch and not args.analyze_local:
        return
    dest = run_root / "collected"
    dest.mkdir(parents=True, exist_ok=True)
    pattern = r"(run_manifest\.json$|metrics\.json$|scores\.json$|selection\.json$|weights\.sqpack$|layout\.json$|unchanged\.pt$|\.ipynb$|\.log$|cloud_context\.json$|dependencies\.txt$)"
    child_env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    completed = (
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
            check=False,
            env=child_env,
        )
        if args.fetch
        else None
    )
    if completed is not None and completed.returncode:
        raise SystemExit(completed.returncode)
    # The platform's executed notebook may be available through pull even when
    # export could not copy its own execution snapshot. Never accept a template.
    if args.fetch:
        subprocess.run(
            [args.kaggle_cli, "kernels", "pull", remote, "-p", str(dest / "platform-notebook")],
            check=True,
            env=child_env,
        )
    expected_notebook = dest / "notebook" / f"{args.run_id}.ipynb"
    if not expected_notebook.exists():
        # Normalize an actual executed platform notebook's name only. Never
        # synthesize execution counts or turn a source template into evidence.
        for path in dest.rglob("*.ipynb"):
            document = json.loads(path.read_text(encoding="utf-8"))
            generated_id = document.get("metadata", {}).get("spectraquant", {}).get("run_id")
            executed = any(
                cell.get("cell_type") == "code" and cell.get("execution_count") is not None
                for cell in document.get("cells", [])
            )
            if document.get("nbformat") == 4 and executed and generated_id in (None, args.run_id):
                expected_notebook.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, expected_notebook)
                break
    registry = Registry("artifacts/runs/registry.jsonl")
    validation = collect(args.run_id, dest, spec=spec, registry=registry, record=False)
    study_name = "phase2-global-damage" if args.study == "phase2" else "phase3-controlled-followup"
    validation, report, integrity = verify_collected_study(
        validation, dest, study_name=study_name, study_key=args.study
    )
    if (
        registry.state(args.run_id) not in {"collected", "validated", "rejected"}
        and validation.manifest_path is not None
    ):
        registry.record_collected(
            args.run_id,
            platform=validation.platform,
            remote_id=validation.remote_id,
            detail="bundle downloaded",
        )
    if validation.ok:
        registry.record_validated(validation)
    else:
        registry.record_rejected(validation)
    (run_root / "validation.json").write_text(
        json.dumps(validation.to_json(), indent=2) + "\n", encoding="utf-8"
    )
    print(validation.summary())
    if not validation.ok:
        raise SystemExit(1)
    assert report is not None and integrity is not None
    summary = summarize(report)
    summary["integrity"] = integrity
    target = Path(f"docs/results/{study_name}.md")
    write_report(report, integrity, target)
    target.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
