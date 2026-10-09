"""Allocation manifests: provenance, deterministic JSON, and replay from the manifest alone."""

from __future__ import annotations

import json

import pytest
from _allocation_fixtures import build_tiny_problem

from spectraquant.allocation import (
    COST_MODEL_VERSION,
    Allocation,
    CostModelMismatch,
    build_manifest,
    load_manifest,
    manifest_from_json,
    manifest_to_json,
    reproduce_allocation,
    solve_exhaustive,
    solve_greedy,
    solve_uniform,
    write_manifest,
)


def _cheap_problem():
    return build_tiny_problem(budget_bytes=1000)


def test_manifest_records_provenance_and_the_byte_total() -> None:
    problem = _cheap_problem()
    allocation = solve_greedy(problem)
    manifest = build_manifest(allocation, problem)

    assert manifest["schema"] == "spectraquant.allocation.manifest.v1"
    assert manifest["solver"] == "greedy"
    assert manifest["cost_model_version"] == COST_MODEL_VERSION
    assert manifest["accounted_bytes"] == allocation.accounted_bytes
    assert manifest["budget_bytes"] == problem.budget_bytes
    assert manifest["per_layer"]  # the full per-layer map
    # git SHA is recorded (may be None outside a checkout, but the block must exist).
    assert set(manifest["git"]) >= {"commit", "short_commit", "dirty"}
    assert "cost_model" in manifest


def test_manifest_json_is_canonical_and_stable() -> None:
    problem = _cheap_problem()
    allocation = solve_greedy(problem)
    manifest = build_manifest(allocation, problem)
    text_a = manifest_to_json(manifest)
    text_b = manifest_to_json(build_manifest(allocation, problem))
    assert text_a == text_b
    assert json.loads(text_a) == manifest
    # sort_keys=True means the sections appear in a fixed order.
    assert text_a.index('"accounted_bytes"') < text_a.index('"per_layer"')


@pytest.mark.parametrize("solver", [solve_uniform, solve_greedy, solve_exhaustive])
def test_replay_from_the_manifest_reproduces_the_identical_allocation(solver) -> None:
    problem = _cheap_problem()
    original = solver(problem)
    manifest = build_manifest(original, problem, solver_params={})

    # Round-trip through JSON text only — nothing but the manifest is passed to the replay.
    replayed = reproduce_allocation(manifest_from_json(manifest_to_json(manifest)))
    assert replayed.per_layer == original.per_layer
    assert replayed.accounted_bytes == original.accounted_bytes
    assert replayed.predicted_error == original.predicted_error
    assert replayed.solver == original.solver


def test_replay_of_the_ortools_manifest_reproduces_the_allocation() -> None:
    pytest.importorskip("ortools.sat.python.cp_model", reason="ortools (alloc extra) not installed")
    from spectraquant.allocation import solve_ortools

    problem = _cheap_problem()
    original = solve_ortools(problem, time_limit_s=10.0, seed=0)
    manifest = build_manifest(original, problem, solver_params={"time_limit_s": 10.0, "seed": 0})
    replayed = reproduce_allocation(manifest)
    assert replayed.accounted_bytes == original.accounted_bytes
    assert replayed.predicted_error == original.predicted_error


def test_replay_of_every_problem_default_solver_is_the_one_recorded() -> None:
    problem = _cheap_problem()
    allocation = solve_uniform(problem)
    manifest = build_manifest(allocation, problem)
    assert reproduce_allocation(manifest).solver == "uniform"


def test_manifest_round_trips_through_a_file(tmp_path) -> None:
    problem = _cheap_problem()
    allocation = solve_greedy(problem)
    path = write_manifest(build_manifest(allocation, problem), tmp_path / "alloc.json")
    loaded = load_manifest(path)
    assert reproduce_allocation(loaded).per_layer == allocation.per_layer


def test_foreign_cost_model_version_is_refused() -> None:
    problem = _cheap_problem()
    manifest = build_manifest(solve_greedy(problem), problem)
    manifest["cost_model_version"] = "spectraquant.allocation.cost.v0"
    with pytest.raises(CostModelMismatch):
        reproduce_allocation(manifest)


def test_unknown_solver_name_is_refused() -> None:
    problem = _cheap_problem()
    manifest = build_manifest(solve_greedy(problem), problem)
    manifest["solver"] = "annealing"
    with pytest.raises(ValueError, match="unknown solver"):
        reproduce_allocation(manifest)


def test_recorded_allocation_matches_the_solved_one() -> None:
    problem = _cheap_problem()
    allocation = solve_greedy(problem)
    manifest = build_manifest(allocation, problem)
    from spectraquant.allocation import recorded_allocation

    recorded = recorded_allocation(manifest)
    assert isinstance(recorded, Allocation)
    assert recorded.per_layer == allocation.per_layer
    assert recorded.accounted_bytes == allocation.accounted_bytes


def test_manifest_requires_an_accounting_cost_model() -> None:
    from _allocation_fixtures import TINY_SHAPES, per_group_cost_model, power_law_error

    from spectraquant.allocation import AllocationProblem, make_accounting_cost_fn

    problem = AllocationProblem(
        layer_shapes=TINY_SHAPES,
        ranks=(2, 4),
        bits=(4,),
        budget_bytes=1000,
        cost_fn=make_accounting_cost_fn(TINY_SHAPES, per_group_cost_model()),
        error_fn=power_law_error({"blocks.0.attn": 4.0, "blocks.0.mlp": 1.0}),
    )
    allocation = solve_greedy(problem)
    with pytest.raises(ValueError, match="cost_model=None"):
        build_manifest(allocation, problem)


def test_manifest_rejects_an_unexpected_schema() -> None:
    with pytest.raises(ValueError, match="schema"):
        manifest_from_json('{"schema": "something-else"}')


def test_incomplete_error_table_is_refused_on_replay() -> None:
    problem = _cheap_problem()
    manifest = build_manifest(solve_greedy(problem), problem)
    manifest["error_table"].pop("blocks.0.mlp")
    with pytest.raises(ValueError, match="no recorded value"):
        reproduce_allocation(manifest)
