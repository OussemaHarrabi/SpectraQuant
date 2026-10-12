"""Generate/submit one frozen private transfer run, never overwrite an attempt."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from spectraquant.cloud.notebook import notebook_digest, write_notebook
from spectraquant.cloud.registry import Registry
from spectraquant.cloud.spec import (
    DatasetRef,
    ExpectedArtifact,
    RunSpec,
    dataset_declaration_digest,
    plan_to_run_spec,
    save_spec,
)
from spectraquant.reporting.gitinfo import git_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("smol360", "pythia410"), required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--kaggle-cli", default="kaggle")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    run_id = args.run_id or f"sq-phase4-{args.model}-20261012"
    root = Path.cwd()
    run = root / "artifacts/runs" / run_id
    if (run / "submission.json").exists():
        raise RuntimeError(
            "existing submission attempt: resume collection, never push this id again"
        )
    info = git_info(root)
    if info.error or info.dirty or not info.commit:
        raise RuntimeError("a clean committed checkout is required")
    config_path = "configs/phase4/transfer.json"
    cfg = json.loads(Path(config_path).read_text())
    plan_path = cfg["models"][args.model]["plan"]
    from spectraquant.experiment_plan import load_plan

    plan = load_plan(plan_path)
    base = plan_to_run_spec(plan_path, platform="kaggle")
    refs = {d.name: d for d in plan.datasets}
    declarations = [
        ("allenai/c4", "train"),
        ("allenai/c4", "validation"),
        ("deepmind/pg19", "test"),
    ]
    prefix = f"artifacts/runs/phase4-transfer-{args.model}"
    arms = [
        "rtn_axis0",
        "mse_axis0",
        "rtn_axis1",
        "mse_axis1",
        *[f"act_axis1_seed{s}" for s in cfg["seeds"]],
    ]
    artifacts = [
        ExpectedArtifact(name="run_manifest.json", min_bytes=2),
        ExpectedArtifact(name=f"notebook/{run_id}.ipynb", min_bytes=64),
        *[
            ExpectedArtifact(name=f"{prefix}/{name}.json", min_bytes=2)
            for name in ("metrics", "scores", "frozen")
        ],
        *[
            ExpectedArtifact(name=f"{prefix}/checkpoints/{arm}/{name}", min_bytes=2)
            for arm in arms
            for name in ("weights.sqpack", "layout.json", "unchanged.pt")
        ],
    ]
    data = base.model_dump(mode="json")
    data.update(
        run_id=run_id,
        git_commit=info.commit,
        timeout_minutes=cfg["platform_timeout_seconds"] // 60,
        seeds=cfg["seeds"],
        measurement_class_expected=3,
        runner_command=f"spectraquant phase4 --config {config_path} --model {args.model}",
        install_spec='uv sync --frozen --extra cloud --extra models --extra alloc && uv pip install --python .venv --reinstall --index-url https://download.pytorch.org/whl/cu128 "torch==2.11.0"',
        dataset_refs=[
            DatasetRef(
                name=name,
                revision=refs[name].revision,
                split=split,
                checksum=dataset_declaration_digest(name, refs[name].revision, split),
            ).model_dump(mode="json")
            for name, split in declarations
        ],
        expected_artifacts=[a.model_dump(mode="json") for a in artifacts],
    )
    spec = RunSpec.model_validate(data)
    staging = run / "kaggle"
    staging.mkdir(parents=True, exist_ok=True)
    save_spec(spec, run / "spec.json")
    write_notebook(spec, staging / "notebook.ipynb")
    metadata = {
        "id": f"oussemaharrabi/{run_id}",
        "title": run_id,
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (staging / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"generated{staging}; scientific SHA{info.commit}")
    if not args.submit:
        return
    attempt = {
        "remote_id": metadata["id"],
        "git_commit": info.commit,
        "timeout_seconds": cfg["platform_timeout_seconds"],
        "maximum_visible_device_hours": cfg["platform_timeout_seconds"] / 3600 * 2,
        "status": "submission_requested",
    }
    path = run / "submission.json"
    path.write_text(json.dumps(attempt, indent=2) + "\n")
    result = subprocess.run(
        [
            args.kaggle_cli,
            "kernels",
            "push",
            "-p",
            str(staging),
            "-t",
            str(cfg["platform_timeout_seconds"]),
        ],
        check=False,
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
    )
    attempt.update(
        submission_exit_code=result.returncode,
        status="submitted" if result.returncode == 0 else "submission_failed",
    )
    path.write_text(json.dumps(attempt, indent=2) + "\n")
    if result.returncode:
        raise SystemExit(result.returncode)
    Registry(root / "artifacts/runs/registry.jsonl").record_submitted(
        spec,
        remote_id=metadata["id"],
        submitted_by="agent",
        notebook_digest=notebook_digest(spec),
        detail="approved frozen A-0025 private free transfer study, 90min cap",
    )


if __name__ == "__main__":
    main()
