"""Generate a reproducible phase2 RunSpec/notebook; submit only when requested.

The official Kaggle CLI reads credentials outside the repository. No credentials
are copied into the spec, notebook or logs. This script contains orchestration
only; scientific logic lives in spectraquant.cloud.phase2.
"""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--run-id", default="sq-phase2-global-20261011")
    parser.add_argument("--kaggle-cli", default="kaggle")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    info = git_info(root)
    if info.error or info.dirty or not info.commit:
        raise RuntimeError(f"phase2 requires a clean committed checkout: {info}")
    base = plan_to_run_spec("configs/tier1/smollm2_135m.yaml", platform="kaggle")
    data = base.model_dump(mode="json")
    wiki = next(ref for ref in base.dataset_refs if "wikitext" in ref.name)
    data["dataset_refs"].append(
        DatasetRef(
            name=wiki.name,
            revision=wiki.revision,
            split="validation",
            checksum=dataset_declaration_digest(wiki.name, wiki.revision, "validation"),
        ).model_dump(mode="json")
    )
    data.update(
        run_id=args.run_id,
        git_commit=info.commit,
        timeout_minutes=200,
        seeds=[0, 1, 2],
        runner_command="spectraquant phase2 --config configs/phase2/global_damage.json",
        measurement_class_expected=3,
        install_spec='uv sync --frozen --extra cloud --extra models --extra alloc && uv pip install --python .venv --reinstall --index-url https://download.pytorch.org/whl/cu128 "torch==2.11.0"',
        expected_artifacts=[
            ExpectedArtifact(name="run_manifest.json", min_bytes=2).model_dump(mode="json"),
            ExpectedArtifact(
                name="artifacts/runs/phase2-global-damage/metrics.json", min_bytes=2
            ).model_dump(mode="json"),
            ExpectedArtifact(
                name="artifacts/runs/phase2-global-damage/scores.json", min_bytes=2
            ).model_dump(mode="json"),
            ExpectedArtifact(name=f"notebook/{args.run_id}.ipynb", min_bytes=64).model_dump(
                mode="json"
            ),
        ],
    )
    spec = RunSpec.model_validate(data)
    artifacts = list(spec.expected_artifacts)
    for arm in [
        "uniform4",
        *[f"{family}_seed{seed}" for seed in (0, 1, 2) for family in ("bits", "residual")],
    ]:
        for filename in ("weights.sqpack", "layout.json", "unchanged.pt"):
            artifacts.append(
                ExpectedArtifact(
                    name=f"artifacts/runs/phase2-global-damage/checkpoints/{arm}/{filename}",
                    min_bytes=2,
                )
            )
    spec = RunSpec.model_validate(
        {
            **spec.model_dump(mode="json"),
            "expected_artifacts": [x.model_dump(mode="json") for x in artifacts],
        }
    )
    staging = root / "artifacts" / "runs" / args.run_id / "kaggle"
    staging.mkdir(parents=True, exist_ok=True)
    save_spec(spec, staging.parent / "spec.json")
    write_notebook(spec, staging / "notebook.ipynb")
    metadata = {
        "id": f"oussemaharrabi/{args.run_id}",
        "title": args.run_id,
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
    (staging / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"generated {staging}; immutable commit={info.commit}")
    if args.submit:
        if (staging.parent / "submission.json").exists():
            raise RuntimeError(
                "a submission attempt is already recorded; resume status/fetch instead of pushing again"
            )
        remote = metadata["id"]
        attempt = {
            "remote_id": remote,
            "git_commit": info.commit,
            "timeout_seconds": 12000,
            "gpu_device_hours_upper_bound": 6.667,
            "status": "submission_requested",
        }
        (staging.parent / "submission.json").write_text(
            json.dumps(attempt, indent=2) + "\n", encoding="utf-8"
        )
        completed = subprocess.run(
            [args.kaggle_cli, "kernels", "push", "-p", str(staging), "-t", "12000"], check=False
        )
        attempt["submission_exit_code"] = completed.returncode
        attempt["status"] = "submitted" if completed.returncode == 0 else "submission_failed"
        (staging.parent / "submission.json").write_text(
            json.dumps(attempt, indent=2) + "\n", encoding="utf-8"
        )
        if completed.returncode:
            raise SystemExit(completed.returncode)
        Registry(root / "artifacts" / "runs" / "registry.jsonl").record_submitted(
            spec,
            remote_id=remote,
            submitted_by="agent",
            notebook_digest=notebook_digest(spec),
            detail="approved A-0023 private free-tier run, 12000s platform limit",
        )


if __name__ == "__main__":
    main()
