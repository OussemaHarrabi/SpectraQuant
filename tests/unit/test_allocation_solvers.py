"""Solver tests: budget compliance, the exhaustive-vs-CP-SAT objective match, and determinism.

The oracle (:func:`solve_exhaustive`) defines optimality; every other solver must be no better than
the oracle (``exhaustive_error <= solver_error``). CP-SAT is compared through the integer objective
(:func:`scaled_error`) so the comparison is exact.
"""

from __future__ import annotations

import pytest
from _allocation_fixtures import budget_range, build_tiny_problem

from spectraquant.allocation import (
    ERROR_SCALE,
    InfeasibleBudget,
    scaled_error,
    solve_exhaustive,
    solve_greedy,
    solve_ortools,
    solve_uniform,
)

# --------------------------------------------------------------------------------------
# tiny instances for the oracle-vs-CP-SAT comparison
# --------------------------------------------------------------------------------------
INSTANCES = [
    pytest.param(
        {"blocks.0.attn": (64, 32), "blocks.0.mlp": (32, 16)},
        {"blocks.0.attn": 4.0, "blocks.0.mlp": 1.0},
        (2, 4, 8),
        (4, 8),
        id="two-layers-three-ranks",
    ),
    pytest.param(
        {"a": (32, 32), "b": (16, 64)},
        {"a": 2.5, "b": 3.25},
        (2, 4, 8),
        (4, 8),
        id="two-layers-different-shapes",
    ),
    pytest.param(
        {"a": (48, 16), "b": (16, 48), "c": (32, 32)},
        {"a": 1.0, "b": 2.0, "c": 0.5},
        (2, 4, 8),
        (4, 8),
        id="three-layers",
    ),
]


def _budget_for(problem) -> int:
    minimum, maximum = budget_range(problem)
    return (minimum + maximum) // 2


def _problem(shapes, scales, ranks, bits):
    minimum, maximum = budget_range(
        build_tiny_problem(
            budget_bytes=0, layer_shapes=shapes, scales=scales, ranks=ranks, bits=bits
        )
    )
    return build_tiny_problem(
        budget_bytes=(minimum + maximum) // 2,
        layer_shapes=shapes,
        scales=scales,
        ranks=ranks,
        bits=bits,
    )


# --------------------------------------------------------------------------------------
# budget compliance
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("solver", [solve_uniform, solve_greedy, solve_exhaustive])
def test_every_solver_respects_the_budget(solver) -> None:
    problem = build_tiny_problem(budget_bytes=_budget_for(build_tiny_problem(budget_bytes=0)))
    allocation = solver(problem)
    assert allocation.feasible is True
    assert allocation.accounted_bytes <= problem.budget_bytes


def test_ortools_respects_the_budget() -> None:
    pytest.importorskip("ortools.sat.python.cp_model", reason="ortools (alloc extra) not installed")
    problem = build_tiny_problem(budget_bytes=1000)
    allocation = solve_ortools(problem, time_limit_s=10.0, seed=0)
    assert allocation.accounted_bytes <= problem.budget_bytes


@pytest.mark.parametrize("solver", [solve_uniform, solve_greedy, solve_exhaustive])
def test_solver_raises_infeasible_budget_with_the_minimum(solver) -> None:
    feasible = build_tiny_problem(budget_bytes=10_000)
    minimum = feasible.min_feasible_budget()
    starved = build_tiny_problem(budget_bytes=minimum - 1)
    with pytest.raises(InfeasibleBudget) as excinfo:
        solver(starved)
    assert excinfo.value.minimum_budget_bytes == minimum


# --------------------------------------------------------------------------------------
# oracle-vs-CP-SAT objective (design note §6 item 4)
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("shapes,scales,ranks,bits", INSTANCES)
def test_exhaustive_matches_ortools_objective(shapes, scales, ranks, bits) -> None:
    pytest.importorskip("ortools.sat.python.cp_model", reason="ortools (alloc extra) not installed")
    problem = _problem(shapes, scales, ranks, bits)
    oracle = solve_exhaustive(problem)
    cpsat = solve_ortools(problem, time_limit_s=10.0, seed=0)
    assert scaled_error(problem, oracle.per_layer) == scaled_error(problem, cpsat.per_layer)
    assert cpsat.diagnostics["ortools_objective_scaled"] == scaled_error(problem, cpsat.per_layer)
    tolerance = len(problem.layer_names) / ERROR_SCALE
    assert abs(oracle.predicted_error - cpsat.predicted_error) <= tolerance


@pytest.mark.parametrize("shapes,scales,ranks,bits", INSTANCES)
def test_exhaustive_never_worse_than_greedy(shapes, scales, ranks, bits) -> None:
    """Directional guarantee: greedy cannot beat the oracle (equality allowed)."""
    problem = _problem(shapes, scales, ranks, bits)
    oracle = solve_exhaustive(problem)
    greedy = solve_greedy(problem)
    assert oracle.predicted_error <= greedy.predicted_error + 1e-12


def test_uniform_produces_a_single_option_for_every_layer() -> None:
    problem = build_tiny_problem(budget_bytes=1000)
    allocation = solve_uniform(problem)
    assert len(set(allocation.per_layer.values())) == 1
    assert allocation.solver == "uniform"


# --------------------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("solver", [solve_uniform, solve_greedy, solve_exhaustive])
def test_solver_is_deterministic_across_runs(solver) -> None:
    problem = build_tiny_problem(budget_bytes=1000)
    first = solver(problem)
    second = solver(problem)
    assert first.per_layer == second.per_layer
    assert first.accounted_bytes == second.accounted_bytes
    assert first.predicted_error == second.predicted_error


def test_ortools_is_deterministic_for_a_fixed_seed() -> None:
    pytest.importorskip("ortools.sat.python.cp_model", reason="ortools (alloc extra) not installed")
    problem = build_tiny_problem(budget_bytes=1000)
    first = solve_ortools(problem, time_limit_s=10.0, seed=0)
    second = solve_ortools(problem, time_limit_s=10.0, seed=0)
    assert first.accounted_bytes == second.accounted_bytes
    assert first.predicted_error == second.predicted_error


# --------------------------------------------------------------------------------------
# guards
# --------------------------------------------------------------------------------------
def test_exhaustive_refuses_a_large_space() -> None:
    problem = build_tiny_problem(
        budget_bytes=10_000_000,
        layer_shapes={f"layer{i}": (16, 16) for i in range(12)},
        scales={f"layer{i}": 1.0 for i in range(12)},
        ranks=(1, 2, 4, 8),
        bits=(2, 4, 8),
    )
    assert len(problem.options) ** len(problem.layer_names) > 1_000_000
    with pytest.raises(ValueError, match="enumerate"):
        solve_exhaustive(problem)


def test_greedy_uses_only_allowed_options() -> None:
    problem = build_tiny_problem(budget_bytes=1000, ranks=(2, 8), bits=(4,))
    allocation = solve_greedy(problem)
    allowed = {(2, 4), (8, 4)}
    assert set(allocation.per_layer.values()) <= allowed


def test_diagnostics_are_numeric_and_include_headroom() -> None:
    problem = build_tiny_problem(budget_bytes=1000)
    allocation = solve_greedy(problem)
    assert (
        allocation.diagnostics["headroom_bytes"]
        == problem.budget_bytes - allocation.accounted_bytes
    )
    assert all(isinstance(value, float) for value in allocation.diagnostics.values())
    assert allocation.diagnostics["n_layers"] == float(len(problem.layer_names))
