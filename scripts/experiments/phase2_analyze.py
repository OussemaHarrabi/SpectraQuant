"""Verify collected Phase2 checkpoints, summarize metrics and emit a research report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spectraquant.evaluation.phase2_report import summarize, verify_checkpoints, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("docs/results/phase2-global-damage.md"))
    args = parser.parse_args()
    report = json.loads((args.study_dir / "metrics.json").read_text(encoding="utf-8"))
    integrity = verify_checkpoints(report, args.study_dir)
    summary = summarize(report)
    summary["integrity"] = integrity
    write_report(report, integrity, args.out)
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
