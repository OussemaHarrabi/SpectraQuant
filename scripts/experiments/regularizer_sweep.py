"""Round-aware preparation coefficient sweep on the Tier-0 dev fixture (Milestone 5, H3).

Thin entry point: all scientific logic lives in :mod:`spectraquant.regularizers.sweep`
(``AGENTS.md`` section 2b rule 1). Writes

* ``artifacts/sample-results/regularizer/sweep.json`` — machine-readable document;
* ``docs/results/regularizer-report.md`` — the human-readable report rendered from the same document.

Everything here is `LOCAL-FIXTURE`, exploratory, measurement class 1-2. The confirmatory Tier-1 H3
cell (`CLOUD-COLAB`) is NOT RUN and is reported as such.

Usage:
    uv run python scripts/experiments/regularizer_sweep.py
        [--seeds 0 1 2 3 4] [--pretrain-steps 60] [--prepare-steps 150]
        [--out artifacts/sample-results/regularizer/sweep.json]
        [--report docs/results/regularizer-report.md]
"""

from __future__ import annotations

import argparse
import time

from spectraquant.regularizers.sweep import (
    CONFIG_PATH,
    DEFAULT_PLAN,
    SweepPlan,
    build_document,
    git_state,
    run_sweep,
    write_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_PLAN.seeds))
    parser.add_argument("--pretrain-steps", type=int, default=DEFAULT_PLAN.pretrain_steps)
    parser.add_argument("--prepare-steps", type=int, default=DEFAULT_PLAN.prepare_steps)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_PLAN.batch_size)
    parser.add_argument("--lr", type=float, default=DEFAULT_PLAN.lr)
    parser.add_argument("--config", default=CONFIG_PATH, help="Tier-0 experiment config")
    parser.add_argument(
        "--out", default="artifacts/sample-results/regularizer/sweep.json", help="JSON document"
    )
    parser.add_argument(
        "--report", default="docs/results/regularizer-report.md", help="markdown report"
    )
    args = parser.parse_args()

    plan = SweepPlan(
        seeds=tuple(args.seeds),
        pretrain_steps=args.pretrain_steps,
        prepare_steps=args.prepare_steps,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    started = time.perf_counter()
    records, results_by_cell, provenance = run_sweep(plan, config_path=args.config)
    wall_time_s = time.perf_counter() - started
    document = build_document(
        records, results_by_cell, provenance, plan, wall_time_s=wall_time_s, git=git_state()
    )
    artifact, report = write_artifacts(document, artifact_path=args.out, report_path=args.report)

    print(f"sweep wall time: {wall_time_s:.1f} s")
    print(f"wrote {artifact} and {report}")
    print("\n== cells (means over seeds) ==")
    header = f"{'cell':16s} {'group':15s} {'lam_round':>9s} {'lam_res':>8s} {'lam_spec':>8s}"
    metric_header = f"{'dev_nll_q':>10s} {'out_err':>10s} {'fac_resid':>10s} {'prod_resid':>11s}"
    print(header + " " + metric_header)
    for label in sorted(document["cells"]):
        summary = document["cells"][label]
        weights = summary["weights"]
        metrics = summary["metrics"]
        print(
            f"{label:16s} {summary['group']:15s} {weights['lambda_round']:9g} "
            f"{weights['lambda_residual']:8g} {weights['lambda_spectrum']:8g} "
            f"{metrics['dev_nll_quantized']['mean']:10.5f} "
            f"{metrics['output_error_vs_dense']['mean']:10.5f} "
            f"{metrics['factor_rounding_residual']['mean']:10.5f} "
            f"{metrics['product_rounding_residual']['mean']:11.5f}"
        )
    print("\n== direction checks ==")
    for name, check in document["direction_checks"].items():
        print(
            f"{name}: lambdas={check['lambdas']} values={[round(v, 6) for v in check['values']]} "
            f"strictly_decreasing={check['strictly_decreasing']} rel={check['relative_change']:+.2%}"
        )
    print(report.read_text(encoding="utf-8").splitlines()[0])


if __name__ == "__main__":
    main()
