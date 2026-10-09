"""Regenerate the allocator oracle comparison that ``docs/results/allocator-report.md`` cites.

Closes audit issue B7: the report's solver comparison existed only as prose, so a re-auditor could
not diff it. This script writes the machine-readable artifact the report now cites.

Usage:
    uv run python scripts/experiments/allocator_oracle_comparison.py [--out DIR]

CPU-only, deterministic (CP-SAT is seeded and the exhaustive oracle is exact); only the solver
wall-clock fields vary between runs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from spectraquant.allocation import (
    QuantCostModel,
    make_accounting_cost_fn,
    make_problem,
    ortools_available,
    solve_exhaustive,
    solve_greedy,
    solve_ortools,
    solve_uniform,
)

#: Layer shapes used by the report's comparison (small enough for exhaustive enumeration).
LAYER_SHAPES: dict[str, tuple[int, int]] = {
    "a": (64, 64),
    "b": (128, 64),
    "c": (64, 128),
    "d": (256, 128),
}
RANKS = (2, 4, 8, 16, 32)
BITS = (4, 8)
BUDGETS = (4000, 8000, 16000)
LAYER_SCALES = {"a": 1.0, "b": 2.0, "c": 0.5, "d": 3.0}


def analytic_error(layer: str, rank: int, bits: int) -> float:
    """The report's analytic per-layer error model (no proxy involved; this is a solver study)."""
    return LAYER_SCALES[layer] * (1.0 / max(rank, 1)) + 0.01 * bits


def git_commit() -> str:
    """Current commit SHA, or ``unknown`` outside a checkout."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/sample-results/allocator")
    args = parser.parse_args()

    model = QuantCostModel(granularity="per_group", group_size=32, symmetric=True)
    document: dict = {
        "git_commit": git_commit(),
        "schema_version": "spectraquant-allocator-oracle-v1",
        "measurement_class": 1,
        "substrate": "LOCAL-FIXTURE",
        "error_model": "analytic: layer_scale/rank + 0.01*bits",
        "layer_shapes": {k: list(v) for k, v in LAYER_SHAPES.items()},
        "ranks": list(RANKS),
        "bits": list(BITS),
        "ortools_available": ortools_available(),
        "budgets": [],
        "overhead_examples": [],
    }

    for budget in BUDGETS:
        problem = make_problem(
            layer_shapes=LAYER_SHAPES,
            ranks=RANKS,
            bits=BITS,
            budget_bytes=budget,
            cost_model=model,
            error_fn=analytic_error,
        )
        row: dict = {
            "budget_bytes": budget,
            "min_feasible_budget_bytes": problem.min_feasible_budget(),
            "solvers": {},
        }
        for name, solver in (
            ("uniform", solve_uniform),
            ("greedy", solve_greedy),
            ("exhaustive", solve_exhaustive),
        ):
            started = time.perf_counter()
            allocation = solver(problem)
            row["solvers"][name] = {
                "predicted_error": allocation.predicted_error,
                "accounted_bytes": allocation.accounted_bytes,
                "solver_wall_time_ms": (time.perf_counter() - started) * 1000.0,
            }
        if ortools_available():
            started = time.perf_counter()
            allocation = solve_ortools(problem, time_limit_s=10.0, seed=0)
            row["solvers"]["ortools"] = {
                "predicted_error": allocation.predicted_error,
                "accounted_bytes": allocation.accounted_bytes,
                "solver_wall_time_ms": (time.perf_counter() - started) * 1000.0,
            }
        oracle = row["solvers"]["exhaustive"]["predicted_error"]
        row["relative_excess_vs_oracle"] = {
            name: (values["predicted_error"] / oracle - 1.0)
            for name, values in row["solvers"].items()
        }
        row["oracle_matches_ortools"] = (
            abs(row["solvers"].get("ortools", {}).get("predicted_error", oracle) - oracle) < 1e-9
        )
        document["budgets"].append(row)

    cost_fn = make_accounting_cost_fn(LAYER_SHAPES, model)
    for rank, bits in ((8, 4), (32, 4), (64, 4), (8, 8)):
        accounted = cost_fn("b", rank, bits)
        nominal = (rank * 64 + 128 * rank) * bits // 8
        document["overhead_examples"].append(
            {
                "layer": "b",
                "shape": [128, 64],
                "rank": rank,
                "bits": bits,
                "accounted_bytes": accounted,
                "nominal_payload_bytes": nominal,
                "overhead_bytes": accounted - nominal,
                "overhead_fraction": (accounted - nominal) / nominal,
            }
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "oracle-comparison.json"
    path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {path}")
    for row in document["budgets"]:
        excess = row["relative_excess_vs_oracle"]
        print(
            f"  budget {row['budget_bytes']:6d} min {row['min_feasible_budget_bytes']:6d} | "
            + " | ".join(f"{k}: {v * 100:+.2f}%" for k, v in excess.items())
            + f" | oracle==ortools {row['oracle_matches_ortools']}"
        )


if __name__ == "__main__":
    main()
