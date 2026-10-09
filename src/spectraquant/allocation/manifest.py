"""Deterministic JSON manifests for allocations — emit, load, and replay.

Design note: ``docs/coordination/design-m2-interfaces.md`` section 4 (owner: allocation stream):
*"allocations are emitted as deterministic JSON manifests with the git SHA, the cost model version,
and the byte total, and must be reproducible from the manifest alone."*

A manifest is a **complete input record**: it stores the problem (layer shapes, allowed ranks/bits,
budget), the quantization cost-model descriptor, the solver and its parameters, plus the evaluated
``error_table``/``overhead_table`` the injected callables produced. Replaying it through
:func:`reproduce_allocation` re-solves the same problem and returns the identical allocation.

The git SHA is recorded via :func:`spectraquant.reporting.gitinfo.git_info` (full SHA plus dirty
flag, ``None`` outside a checkout). The cost-model version is compared on replay: a manifest built by
a different cost model is refused rather than silently re-interpreted (AGENTS.md section 9.6).

Honesty note (AGENTS.md section 4.13): ``accounted_bytes`` is a **class-1 analytical** figure. Real
serialized-byte validation arrives at M6; nothing in this manifest is a measured class-3 size.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from spectraquant.allocation.problem import (
    COST_MODEL_VERSION,
    Allocation,
    AllocationProblem,
    QuantCostModel,
    make_problem,
)
from spectraquant.allocation.solvers import (
    solve_exhaustive,
    solve_greedy,
    solve_ortools,
    solve_uniform,
)
from spectraquant.reporting.gitinfo import GitInfo, git_info

__all__ = [
    "MANIFEST_SCHEMA_ID",
    "CostModelMismatch",
    "build_manifest",
    "load_manifest",
    "manifest_from_json",
    "manifest_to_json",
    "problem_from_manifest",
    "recorded_allocation",
    "reproduce_allocation",
    "write_manifest",
]

#: Schema identifier carried by every allocation manifest.
MANIFEST_SCHEMA_ID = "spectraquant.allocation.manifest.v1"


class CostModelMismatch(ValueError):
    """Raised when a manifest was produced by a different cost-model version than the current code."""

    def __init__(self, recorded: str, current: str) -> None:
        self.recorded = recorded
        self.current = current
        super().__init__(
            f"manifest cost model {recorded!r} does not match the current {current!r}; "
            "re-solve the problem instead of replaying a foreign manifest"
        )


def _option_key(rank: int, bits: int) -> str:
    """Canonical ``"rank:bits"`` key used in the evaluated tables."""
    return f"{rank}:{bits}"


def _git_block(git: GitInfo | None) -> dict[str, Any]:
    """Serialize a :class:`GitInfo` into the manifest (all fields ``None``-safe)."""
    info = git if git is not None else git_info()
    return {
        "commit": info.commit,
        "short_commit": info.short_commit,
        "branch": info.branch,
        "describe": info.describe,
        "dirty": info.dirty,
        "error": info.error,
    }


def _evaluated_table(problem: AllocationProblem, fn: Any) -> dict[str, dict[str, float]]:
    """Evaluate ``fn(layer, rank, bits)`` over the whole option space into a JSON-ready table."""
    return {
        name: {_option_key(rank, bits): fn(name, rank, bits) for rank, bits in problem.options}
        for name in problem.layer_names
    }


def build_manifest(
    allocation: Allocation,
    problem: AllocationProblem,
    *,
    solver_params: Mapping[str, Any] | None = None,
    git: GitInfo | None = None,
) -> dict[str, Any]:
    """Build the deterministic manifest for one solved allocation.

    Args:
        allocation: The allocation to record.
        problem: The problem it solved; must carry a :class:`QuantCostModel` so the accounting cost
            can be rebuilt on replay.
        solver_params: Solver keyword arguments (e.g. ``{"time_limit_s": ..., "seed": ...}``).
        git: Git provenance; defaults to :func:`spectraquant.reporting.gitinfo.git_info`.

    Returns:
        A JSON-serializable manifest dict.

    Raises:
        ValueError: If ``problem.cost_model`` is ``None`` (a custom cost function is not replayable).
    """
    if problem.cost_model is None:
        raise ValueError(
            "cannot emit a manifest for a problem with cost_model=None: build the problem with "
            "make_problem(...) so the accounting cost model is recorded"
        )
    per_layer = allocation.as_assignment()
    return {
        "schema": MANIFEST_SCHEMA_ID,
        "cost_model_version": problem.cost_model_version,
        "git": _git_block(git),
        "solver": allocation.solver,
        "solver_params": dict(solver_params or {}),
        "budget_bytes": int(problem.budget_bytes),
        "layer_shapes": {
            name: [int(out), int(in_features)]
            for name, (out, in_features) in sorted(problem.layer_shapes.items())
        },
        "allowed_ranks": sorted(int(rank) for rank in problem.ranks),
        "allowed_bits": sorted(int(bits) for bits in problem.bits),
        "cost_model": problem.cost_model.to_dict(),
        "accounted_bytes": int(allocation.accounted_bytes),
        "predicted_error": float(allocation.predicted_error),
        "feasible": bool(allocation.feasible),
        "per_layer": {
            name: [int(rank), int(bits)] for name, (rank, bits) in sorted(per_layer.items())
        },
        "diagnostics": {key: float(value) for key, value in sorted(allocation.diagnostics.items())},
        "error_table": _evaluated_table(problem, problem.error_fn),
        "overhead_table": _evaluated_table(problem, problem.overhead_fn),
    }


def manifest_to_json(manifest: Mapping[str, Any]) -> str:
    """Serialize a manifest to canonical JSON (sorted keys, 2-space indent, trailing newline)."""
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def manifest_from_json(text: str) -> dict[str, Any]:
    """Parse manifest JSON text, validating its schema identifier."""
    document = json.loads(text)
    if not isinstance(document, dict):
        raise ValueError("allocation manifest must be a JSON object")
    if document.get("schema") != MANIFEST_SCHEMA_ID:
        raise ValueError(
            f"unexpected manifest schema {document.get('schema')!r}, expected {MANIFEST_SCHEMA_ID!r}"
        )
    return document


def write_manifest(manifest: Mapping[str, Any], path: str | Path) -> Path:
    """Write a manifest to ``path`` and return the path."""
    target = Path(path)
    target.write_text(manifest_to_json(manifest), encoding="utf-8")
    return target


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Read and validate a manifest file."""
    return manifest_from_json(Path(path).read_text(encoding="utf-8"))


def _table_lookup(table: Mapping[str, Any], layer: str, rank: int, bits: int) -> Any:
    """Look up one ``(layer, rank, bits)`` entry, failing loudly if the manifest is incomplete."""
    try:
        return table[layer][_option_key(rank, bits)]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"manifest has no recorded value for layer {layer!r} option ({rank}, {bits})"
        ) from exc


def problem_from_manifest(manifest: Mapping[str, Any]) -> AllocationProblem:
    """Rebuild the exact :class:`AllocationProblem` a manifest recorded.

    The cost function is reconstructed from the accounting cost model, and the injected error/overhead
    callables are reconstructed from the evaluated tables, so no science is hidden in the manifest.

    Args:
        manifest: A manifest produced by :func:`build_manifest`.

    Returns:
        The reconstructed problem.

    Raises:
        CostModelMismatch: If the manifest's cost-model version differs from the current code.
    """
    if manifest.get("schema") != MANIFEST_SCHEMA_ID:
        raise ValueError(f"unexpected manifest schema {manifest.get('schema')!r}")
    recorded = str(manifest["cost_model_version"])
    if recorded != COST_MODEL_VERSION:
        raise CostModelMismatch(recorded, COST_MODEL_VERSION)
    cost_model = QuantCostModel.from_dict(manifest["cost_model"])
    layer_shapes = {
        str(name): (int(dims[0]), int(dims[1])) for name, dims in manifest["layer_shapes"].items()
    }
    error_table = manifest["error_table"]
    overhead_table = manifest.get("overhead_table", {})

    def error_fn(layer: str, rank: int, bits: int) -> float:
        return float(_table_lookup(error_table, layer, rank, bits))

    def overhead_fn(layer: str, rank: int, bits: int) -> int:
        return int(_table_lookup(overhead_table, layer, rank, bits))

    return make_problem(
        layer_shapes=layer_shapes,
        ranks=[int(rank) for rank in manifest["allowed_ranks"]],
        bits=[int(bits) for bits in manifest["allowed_bits"]],
        budget_bytes=int(manifest["budget_bytes"]),
        cost_model=cost_model,
        error_fn=error_fn,
        overhead_fn=overhead_fn,
    )


def recorded_allocation(manifest: Mapping[str, Any]) -> Allocation:
    """Return the allocation a manifest recorded verbatim (without re-solving)."""
    per_layer = {
        str(name): (int(rank), int(bits)) for name, (rank, bits) in manifest["per_layer"].items()
    }
    return Allocation(
        per_layer=per_layer,
        accounted_bytes=int(manifest["accounted_bytes"]),
        predicted_error=float(manifest["predicted_error"]),
        solver=str(manifest["solver"]),
        feasible=bool(manifest["feasible"]),
        diagnostics={str(k): float(v) for k, v in manifest["diagnostics"].items()},
    )


_SOLVERS = {
    "uniform": solve_uniform,
    "greedy": solve_greedy,
    "exhaustive": solve_exhaustive,
    "ortools": solve_ortools,
}


def reproduce_allocation(manifest: Mapping[str, Any]) -> Allocation:
    """Re-solve the problem described by a manifest, returning the reproduced allocation.

    Args:
        manifest: A manifest produced by :func:`build_manifest`.

    Returns:
        The allocation a fresh solve of the reconstructed problem produces.

    Raises:
        ValueError: If the manifest names an unknown solver.
        CostModelMismatch: If the cost-model version differs.
    """
    problem = problem_from_manifest(manifest)
    solver = str(manifest["solver"])
    try:
        solve = _SOLVERS[solver]
    except KeyError as exc:
        raise ValueError(f"manifest names unknown solver {solver!r}") from exc
    params = dict(manifest.get("solver_params") or {})
    if solver == "ortools":
        return solve(
            problem,
            time_limit_s=float(params.get("time_limit_s", 10.0)),
            seed=int(params.get("seed", 0)),
        )
    return solve(problem)
