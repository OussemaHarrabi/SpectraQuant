"""Allocation solvers: uniform, greedy, exhaustive (oracle) and CP-SAT.

Design note: ``docs/coordination/design-m2-interfaces.md`` section 4 (owner: allocation stream).
Every solver returns an :class:`~spectraquant.allocation.problem.Allocation` whose
``accounted_bytes`` is the same ``sum(cost_fn + overhead_fn)`` the budget constraint was applied to,
and raises :class:`~spectraquant.allocation.problem.InfeasibleBudget` — carrying the minimum feasible
budget — when no allowed allocation fits (review §6, B16; design note §7.6).

Determinism
-----------
All solvers are deterministic for a fixed problem:

* ``solve_uniform`` — ties broken by ``(predicted_error, accounted_bytes, bits, rank)`` ascending.
* ``solve_greedy`` — start from every layer's cheapest option; repeatedly apply the single-layer
  change with the best marginal error reduction per added byte; ties broken by
  ``(free-move?, ratio, improvement, delta_bytes, layer name, bits, rank)`` ascending. "Free" moves
  (``delta_bytes <= 0`` and a strict improvement) are ranked ahead of any ratio gain. Iteration stops
  when no strictly improving, budget-feasible move remains.
* ``solve_exhaustive`` — full enumeration; ties broken by ``(error, bytes, per-layer (bits, rank))``
  ascending, giving the unique lexicographically smallest optimal assignment.
* ``solve_ortools`` — one CP-SAT worker with a fixed seed; the error objective is rounded to integers
  at :data:`ERROR_SCALE`, so an optimal CP-SAT objective equals the exhaustive objective **in the
  scaled integer metric** (see :func:`scaled_error`), not necessarily in raw float.

The error objective is always minimized (lower error is better).
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from spectraquant.allocation.problem import (
    Allocation,
    AllocationProblem,
    InfeasibleBudget,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ortools.sat.python import cp_model

__all__ = [
    "ERROR_SCALE",
    "MAX_EXHAUSTIVE_COMBINATIONS",
    "ORToolsNotInstalledError",
    "ortools_available",
    "scaled_error",
    "solve_exhaustive",
    "solve_greedy",
    "solve_ortools",
    "solve_uniform",
]

#: Integer scale for the CP-SAT objective: ``round(error * ERROR_SCALE)``. The same scale is used by
#: :func:`scaled_error`, so the exhaustive and CP-SAT objectives are comparable exactly as integers.
ERROR_SCALE = 1_000_000

#: Refuse to enumerate more than this many full assignments in :func:`solve_exhaustive`.
MAX_EXHAUSTIVE_COMBINATIONS = 1_000_000


class ORToolsNotInstalledError(ImportError):
    """Raised when :func:`solve_ortools` is called without the optional ``alloc`` extra installed."""


def ortools_available() -> bool:
    """Return ``True`` when the optional ``ortools`` dependency can be imported."""
    try:  # pragma: no cover - exercised by import only
        import ortools.sat.python.cp_model  # noqa: F401
    except ImportError:
        return False
    return True


def scaled_error(problem: AllocationProblem, assignment: Mapping[str, tuple[int, int]]) -> int:
    """Return ``sum(round(error_fn(layer, *option) * ERROR_SCALE))`` over the assignment.

    This is the integer objective CP-SAT minimizes, exposed so a test can compare an exhaustive and
    a CP-SAT optimum exactly rather than through a float tolerance.
    """
    return sum(
        round(problem.error_fn(name, *assignment[name]) * ERROR_SCALE)
        for name in problem.layer_names
    )


def _diagnostics(
    problem: AllocationProblem, assignment: Mapping[str, tuple[int, int]]
) -> dict[str, float]:
    """Common numeric diagnostics for an assignment (mean rank/bits, budget headroom)."""
    names = problem.layer_names
    accounted = problem.assignment_cost(assignment)
    ranks = [assignment[name][0] for name in names]
    bits = [assignment[name][1] for name in names]
    return {
        "n_layers": float(len(names)),
        "budget_bytes": float(problem.budget_bytes),
        "headroom_bytes": float(problem.budget_bytes - accounted),
        "mean_rank": sum(ranks) / len(ranks),
        "mean_bits": sum(bits) / len(bits),
    }


def _finalize(
    problem: AllocationProblem,
    assignment: Mapping[str, tuple[int, int]],
    solver: str,
    extra: Mapping[str, float] | None = None,
) -> Allocation:
    """Build an :class:`Allocation` from a full assignment, recomputing the reported quantities."""
    # Guarantee the invariant at the single point of emission: the reported byte total is the same
    # sum the budget was checked against (design note §7.6).
    accounted = problem.assignment_cost(assignment)
    if accounted > problem.budget_bytes:
        raise InfeasibleBudget(problem.min_feasible_budget(), problem.budget_bytes)
    diagnostics = _diagnostics(problem, assignment)
    if extra:
        diagnostics.update({key: float(value) for key, value in extra.items()})
    return Allocation(
        per_layer={name: (int(rank), int(bits)) for name, (rank, bits) in assignment.items()},
        accounted_bytes=accounted,
        predicted_error=problem.assignment_error(assignment),
        solver=solver,
        feasible=True,
        diagnostics=diagnostics,
    )


def solve_uniform(problem: AllocationProblem) -> Allocation:
    """Assign one ``(rank, bits)`` pair to every layer, minimizing total error under the budget.

    Args:
        problem: The allocation problem.

    Returns:
        The cheapest uniform assignment that fits: among all uniform options it minimizes
        ``(predicted_error, accounted_bytes, bits, rank)`` ascending.

    Raises:
        InfeasibleBudget: If no uniform option fits (carries the minimum feasible budget, which is the
            per-layer cheapest sum — smaller than any uniform total).
    """
    problem.check_budget()
    best: tuple[tuple[float, int, int, int], tuple[int, int]] | None = None
    for option in problem.options:
        rank, bits = option
        assignment = dict.fromkeys(problem.layer_names, option)
        accounted = problem.assignment_cost(assignment)
        if accounted > problem.budget_bytes:
            continue
        error = problem.assignment_error(assignment)
        key = (error, accounted, bits, rank)
        if best is None or key < best[0]:
            best = (key, option)
    if best is None:
        raise InfeasibleBudget(problem.min_feasible_budget(), problem.budget_bytes)
    rank, bits = best[1]
    assignment = dict.fromkeys(problem.layer_names, (rank, bits))
    return _finalize(
        problem,
        assignment,
        "uniform",
        extra={"uniform_rank": float(rank), "uniform_bits": float(bits)},
    )


def solve_greedy(problem: AllocationProblem) -> Allocation:
    """Greedily spend the budget on the best marginal error reduction per added byte.

    Starts from every layer's cheapest option and repeatedly applies the single-layer option change
    with the greatest marginal error reduction per additional byte. Deterministic tie-breaking is
    documented in the module docstring; the move ranking is
    ``(free, ratio, improvement, delta_bytes, layer, bits, rank)`` ascending.

    Args:
        problem: The allocation problem.

    Returns:
        A feasible (possibly budget-underusing) allocation.

    Raises:
        InfeasibleBudget: If even the cheapest option for every layer exceeds the budget.
    """
    problem.check_budget()
    names = problem.layer_names
    options = problem.options
    assignment: dict[str, tuple[int, int]] = {name: problem.cheapest_option(name) for name in names}
    cost = problem.assignment_cost(assignment)
    errors = {name: float(problem.error_fn(name, *assignment[name])) for name in names}

    iteration_cap = len(names) * len(options) + 1
    moves = 0
    while moves < iteration_cap:
        best_key: tuple[Any, ...] | None = None
        best_move: tuple[str, tuple[int, int], float] | None = None
        for name in names:
            current_rank, current_bits = assignment[name]
            current_cost = problem.total_cost(name, current_rank, current_bits)
            for option in options:
                if option == (current_rank, current_bits):
                    continue
                error = float(problem.error_fn(name, *option))
                improvement = errors[name] - error
                if improvement <= 0.0:
                    continue
                option_cost = problem.total_cost(name, *option)
                delta = option_cost - current_cost
                if cost + delta > problem.budget_bytes:
                    continue
                rank, bits = option
                if delta <= 0:
                    key = (1, 0.0 if delta == 0 else 1.0, improvement, delta, name, bits, rank)
                else:
                    key = (0, improvement / delta, improvement, delta, name, bits, rank)
                if best_key is None or key < best_key:
                    best_key = key
                    best_move = (name, option, error)
        if best_move is None:
            break
        name, option, error = best_move
        assignment[name] = option
        errors[name] = error
        cost = problem.assignment_cost(assignment)
        moves += 1
    return _finalize(
        problem,
        assignment,
        "greedy",
        extra={"greedy_moves": float(moves), "greedy_iterations": float(moves)},
    )


def solve_exhaustive(problem: AllocationProblem) -> Allocation:
    """Enumerate every assignment and return the globally optimal one (the oracle).

    Tiny problems only: refuses more than :data:`MAX_EXHAUSTIVE_COMBINATIONS` full assignments. Ties
    are broken by ``(predicted_error, accounted_bytes, per-layer (bits, rank))`` ascending, yielding a
    unique deterministic optimum.

    Args:
        problem: The allocation problem.

    Returns:
        The minimum-error feasible allocation.

    Raises:
        InfeasibleBudget: If no assignment fits.
        ValueError: If the assignment space is too large to enumerate.
    """
    problem.check_budget()
    names = problem.layer_names
    options = problem.options
    combinations = len(options) ** len(names)
    if combinations > MAX_EXHAUSTIVE_COMBINATIONS:
        raise ValueError(
            f"solve_exhaustive would enumerate {combinations} assignments "
            f"(> {MAX_EXHAUSTIVE_COMBINATIONS}); use a smaller problem or solve_ortools"
        )
    costs = {
        name: {option: problem.total_cost(name, *option) for option in options} for name in names
    }
    errors = {
        name: {option: float(problem.error_fn(name, *option)) for option in options}
        for name in names
    }
    best_key: tuple[Any, ...] | None = None
    best_assignment: dict[str, tuple[int, int]] | None = None
    for combo in itertools.product(options, repeat=len(names)):
        accounted = sum(costs[name][option] for name, option in zip(names, combo, strict=True))
        if accounted > problem.budget_bytes:
            continue
        error = sum(errors[name][option] for name, option in zip(names, combo, strict=True))
        tie = tuple((bits, rank) for rank, bits in combo)
        key = (error, accounted, tie)
        if best_key is None or key < best_key:
            best_key = key
            best_assignment = dict(zip(names, combo, strict=True))
    if best_assignment is None:
        raise InfeasibleBudget(problem.min_feasible_budget(), problem.budget_bytes)
    return _finalize(
        problem,
        best_assignment,
        "exhaustive",
        extra={"enumerated_assignments": float(combinations)},
    )


def solve_ortools(
    problem: AllocationProblem, *, time_limit_s: float = 10.0, seed: int = 0
) -> Allocation:
    """Solve the allocation exactly with OR-Tools CP-SAT (optional ``alloc`` extra).

    The error objective is rounded to integers at :data:`ERROR_SCALE` for CP-SAT; the returned
    allocation's ``predicted_error`` and ``accounted_bytes`` are recomputed exactly from the problem,
    so only the *optimum* is affected by the rounding (AGENTS.md section 4.13: labelled approximate).

    Args:
        problem: The allocation problem.
        time_limit_s: Wall-clock solver limit, seconds.
        seed: CP-SAT random seed (single worker, so the run is reproducible).

    Returns:
        The optimal (in the scaled metric) feasible allocation.

    Raises:
        ORToolsNotInstalledError: If ``ortools`` is missing; the message names the ``alloc`` extra.
        InfeasibleBudget: If no assignment fits.
        RuntimeError: If CP-SAT exhausts ``time_limit_s`` without a solution.
    """
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ORToolsNotInstalledError(
            "solve_ortools requires the optional 'alloc' extra: run "
            "`uv sync --extra alloc` (or `pip install 'spectraquant[alloc]'`) to install ortools"
        ) from exc

    problem.check_budget()
    names = problem.layer_names
    options = problem.options
    # ortools ships no type stubs for the CP-SAT model object, so the builder is typed Any here
    # rather than silenced with a blanket ignore: the solver's behaviour is covered by tests.
    model: Any = cp_model.CpModel()
    variables: dict[tuple[str, int], cp_model.IntVar] = {}
    for name in names:
        layer_vars = []
        for index in range(len(options)):
            var = model.NewBoolVar(f"{name}__{index}")
            variables[(name, index)] = var
            layer_vars.append(var)
        model.AddExactlyOne(layer_vars)
    cost_terms = []
    error_terms = []
    for name in names:
        for index, option in enumerate(options):
            var = variables[(name, index)]
            cost = problem.total_cost(name, *option)
            if cost:
                cost_terms.append(var * int(cost))
            scaled = round(problem.error_fn(name, *option) * ERROR_SCALE)
            error_terms.append(var * scaled)
    if cost_terms:
        model.Add(sum(cost_terms) <= int(problem.budget_bytes))
    model.Minimize(sum(error_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = int(seed)
    status = solver.Solve(model)
    if status == cp_model.INFEASIBLE:
        raise InfeasibleBudget(problem.min_feasible_budget(), problem.budget_bytes)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(
            f"CP-SAT returned {solver.StatusName(status)} within {time_limit_s}s; "
            "increase time_limit_s"
        )
    assignment: dict[str, tuple[int, int]] = {}
    for name in names:
        for index, option in enumerate(options):
            if solver.Value(variables[(name, index)]) == 1:
                assignment[name] = option
                break
        else:  # pragma: no cover - AddExactlyOne guarantees a selection
            raise RuntimeError(f"CP-SAT selected no option for layer {name!r}")
    return _finalize(
        problem,
        assignment,
        "ortools",
        extra={
            "ortools_status_code": float(status),
            "ortools_wall_time_s": float(solver.WallTime()),
            "ortools_objective_scaled": float(solver.ObjectiveValue()),
        },
    )
