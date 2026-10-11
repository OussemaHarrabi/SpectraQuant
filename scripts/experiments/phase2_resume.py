"""Resume status/collection of an existing Phase2 run; never submit compute."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from spectraquant.cloud.adapters.kaggle import parse_status_output
from spectraquant.cloud.collect import collect
from spectraquant.cloud.registry import Registry
from spectraquant.cloud.spec import load_spec
from spectraquant.evaluation.phase2_report import summarize, verify_checkpoints, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="sq-phase2-global-20261011")
    parser.add_argument("--kaggle-cli", default="kaggle")
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
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
    if not args.fetch:
        return
    dest = run_root / "collected"
    dest.mkdir(parents=True, exist_ok=True)
    pattern = r"(run_manifest\.json$|metrics\.json$|scores\.json$|weights\.sqpack$|layout\.json$|unchanged\.pt$|\.ipynb$|\.log$|cloud_context\.json$|dependencies\.txt$)"
    completed = subprocess.run(
        [args.kaggle_cli, "kernels", "output", remote, "-p", str(dest), "--file-pattern", pattern],
        check=False,
    )
    if completed.returncode:
        raise SystemExit(completed.returncode)
    # The platform's executed notebook may be available through pull even when
    # export could not copy its own execution snapshot. Never accept a template.
    subprocess.run(
        [args.kaggle_cli, "kernels", "pull", remote, "-p", str(dest / "platform-notebook")],
        check=True,
    )
    validation = collect(
        args.run_id, dest, spec=spec, registry=Registry("artifacts/runs/registry.jsonl")
    )
    (run_root / "validation.json").write_text(
        json.dumps(validation.to_json(), indent=2) + "\n", encoding="utf-8"
    )
    print(validation.summary())
    if not validation.ok:
        raise SystemExit(1)
    matches = list(dest.rglob("phase2-global-damage/metrics.json"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one Phase2 metrics file, found {len(matches)}")
    report = json.loads(matches[0].read_text(encoding="utf-8"))
    integrity = verify_checkpoints(report, matches[0].parent)
    summary = summarize(report)
    summary["integrity"] = integrity
    target = Path("docs/results/phase2-global-damage.md")
    write_report(report, integrity, target)
    target.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
