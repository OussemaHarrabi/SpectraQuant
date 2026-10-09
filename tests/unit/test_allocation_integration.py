"""Milestone-6 integration: the proxy adapter, byte reconciliation, and the equal-memory gate.

These tests cover the pieces the frontier experiment is built from:

* the proxy -> allocator adapter (a proxy's per-layer score becomes the allocator's ``error_fn``);
* class-3 measured vs class-1 accounted byte reconciliation and its tolerance assertion;
* the equal-memory gate on allocation manifests, including the required negative case;
* deterministic manifest emission and replay;
* replay of the committed frontier manifests.

The proxy used here is a deterministic in-test stub, so the adapter is checked independently of any
real variant; the real candidate is exercised end-to-end by
``scripts/experiments/allocation_frontier.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from spectraquant.allocation import (
    BYTE_PARITY_TOLERANCE,
    QuantCostModel,
    allocation_factor_state,
    assert_byte_parity,
    build_manifest,
    equal_memory_manifest,
    gate_equal_memory,
    make_accounting_cost_fn,
    make_proxy_problem,
    manifest_to_json,
    proxy_error_fn,
    reconcile_allocation_bytes,
    recorded_allocation,
    reproduce_allocation,
    solve_greedy,
    solve_ortools,
)
from spectraquant.proxies.base import LayerInputs, ProxyResult
from spectraquant.reporting.comparability import UnequalMemoryComparison

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTIER_DIR = REPO_ROOT / "artifacts" / "sample-results" / "allocation-frontier"

SHAPES: dict[str, tuple[int, int]] = {"l0": (8, 4), "l1": (6, 6)}
RANKS: tuple[int, ...] = (2, 4)
BITS: tuple[int, ...] = (4, 8)


def cost_model() -> QuantCostModel:
    return QuantCostModel(granularity="per_group", group_size=4, symmetric=True, axis=1)


def layer_inputs(out: int, in_features: int, gain: float = 1.0) -> LayerInputs:
    weight = torch.randn(out, in_features)
    activations = torch.randn(5, in_features)
    return LayerInputs(weight=weight, activations=activations, downstream_gain=gain)


class StubProxy:
    """A deterministic stand-in for a real proxy variant (implements the frozen protocol)."""

    name = "stub"

    def __init__(self, table: dict[tuple[str, int, int], float]) -> None:
        self.table = table

    def score_layer(self, w: torch.Tensor, x: torch.Tensor, rank: int, spec: Any) -> ProxyResult:
        value = float(rank + spec.bits)
        return ProxyResult(
            value=value,
            exact=True,
            measurement_class=2,
            per_layer={"layer": value},
            diagnostics={},
        )

    def score_model(self, layers: Any, ranks: Any, specs: Any) -> ProxyResult:
        per_layer = {
            name: float(self.table[(name, int(ranks[name]), int(specs[name].bits))])
            for name in layers
        }
        return ProxyResult(
            value=float(sum(per_layer.values())),
            exact=True,
            measurement_class=2,
            per_layer=per_layer,
            diagnostics={},
        )


def stub_context() -> tuple[StubProxy, dict[str, LayerInputs]]:
    table = {
        ("l0", 2, 4): 5.0,
        ("l0", 2, 8): 4.0,
        ("l0", 4, 4): 3.0,
        ("l0", 4, 8): 2.0,
        ("l1", 2, 4): 9.0,
        ("l1", 2, 8): 8.0,
        ("l1", 4, 4): 7.0,
        ("l1", 4, 8): 6.0,
    }
    proxy = StubProxy(table)
    inputs = {"l0": layer_inputs(8, 4), "l1": layer_inputs(6, 6)}
    return proxy, inputs


# ---------------------------------------------------------------- adapter


def test_proxy_error_fn_matches_the_proxy_output() -> None:
    proxy, inputs = stub_context()
    error_fn = proxy_error_fn(proxy, inputs, cost_model())
    assert error_fn("l0", 4, 8) == 2.0
    assert error_fn("l1", 2, 4) == 9.0
    # the adapter asks the proxy for the spec of the requested bit width
    assert (
        error_fn("l0", 2, 4)
        == proxy.score_model(
            {"l0": inputs["l0"]}, {"l0": 2}, {"l0": cost_model().spec_for_bits(4)}
        ).value
    )


def test_proxy_error_fn_supports_option_specific_context() -> None:
    proxy, inputs = stub_context()
    calls: list[tuple[str, int, int]] = []

    def context(layer: str, rank: int, bits: int) -> LayerInputs:
        calls.append((layer, rank, bits))
        return inputs[layer]

    error_fn = proxy_error_fn(proxy, context, cost_model())
    assert error_fn("l1", 4, 4) == 7.0
    assert calls == [("l1", 4, 4)]


def test_proxy_error_fn_rejects_an_unknown_layer() -> None:
    proxy, inputs = stub_context()
    error_fn = proxy_error_fn(proxy, inputs, cost_model())
    with pytest.raises(KeyError):
        error_fn("nope", 4, 4)


def test_make_proxy_problem_wires_the_accounting_cost() -> None:
    proxy, inputs = stub_context()
    model = cost_model()
    problem = make_proxy_problem(
        proxy=proxy,
        layer_inputs=inputs,
        layer_shapes=SHAPES,
        ranks=RANKS,
        bits=BITS,
        budget_bytes=100_000,
        cost_model=model,
    )
    reference_cost = make_accounting_cost_fn(SHAPES, model)
    for name in problem.layer_names:
        for rank in RANKS:
            for bits in BITS:
                assert problem.cost_fn(name, rank, bits) == reference_cost(name, rank, bits)
    assert problem.error_fn("l0", 4, 8) == 2.0


# ---------------------------------------------------------------- byte reconciliation


def test_reconcile_bytes_matches_the_class_one_accounting() -> None:
    weights = {"l0": torch.randn(8, 4), "l1": torch.randn(6, 6)}
    per_layer = {"l0": (2, 4), "l1": (4, 8)}
    recon = reconcile_allocation_bytes(
        weights=weights, per_layer=per_layer, cost_model=cost_model()
    )
    assert recon["within_tolerance"] is True
    assert recon["accounted_bytes"] == recon["measured_bytes"]
    assert recon["relative_difference"] == 0.0
    assert recon["container_overhead_bytes"] == 0
    assert recon["serializer"] == "spectraquant-sqpack-v1"
    assert_byte_parity(recon)


def test_assert_byte_parity_rejects_a_violation() -> None:
    doctored = {
        "accounted_bytes": 100,
        "measured_bytes": 103,
        "difference_bytes": 3,
        "container_overhead_bytes": 3,
        "relative_difference": 0.03,
        "tolerance": BYTE_PARITY_TOLERANCE,
        "within_tolerance": False,
    }
    with pytest.raises(AssertionError, match="differ from class-1"):
        assert_byte_parity(doctored)


def test_allocation_factor_state_rejects_a_clamped_rank() -> None:
    weights = {"l0": torch.randn(4, 8)}
    with pytest.raises(ValueError, match="effective rank"):
        allocation_factor_state(weights, {"l0": (8, 4)}, cost_model())


# ---------------------------------------------------------------- equal-memory gate


def test_equal_memory_gate_accepts_parity_and_rejects_a_gap() -> None:
    reference = equal_memory_manifest("ref", measured_bytes=1000, accounted_bytes=1000)
    near = equal_memory_manifest("near", measured_bytes=1004, accounted_bytes=1004)
    far = equal_memory_manifest("far", measured_bytes=1050, accounted_bytes=1050)

    verdict = gate_equal_memory(reference, near)
    assert verdict.equal is True
    assert verdict.measurement_class == 3

    with pytest.raises(UnequalMemoryComparison, match="tolerance"):
        gate_equal_memory(reference, far)
    # the same pair is accepted once the tolerance is raised above the gap: the rejection is caused
    # by the tolerance check, not by some unrelated error path
    raised = gate_equal_memory(reference, far, tolerance=0.10)
    assert raised.equal is True


# ---------------------------------------------------------------- manifest emission and replay


def _proxy_problem() -> Any:
    proxy, inputs = stub_context()
    return make_proxy_problem(
        proxy=proxy,
        layer_inputs=inputs,
        layer_shapes=SHAPES,
        ranks=RANKS,
        bits=BITS,
        budget_bytes=100_000,
        cost_model=cost_model(),
    )


def test_manifest_replay_reproduces_a_proxy_allocation() -> None:
    problem = _proxy_problem()
    allocation = solve_greedy(problem)
    manifest = build_manifest(allocation, problem)
    replayed = reproduce_allocation(manifest)
    assert replayed.per_layer == allocation.per_layer
    assert replayed.accounted_bytes == allocation.accounted_bytes
    assert replayed.predicted_error == allocation.predicted_error
    assert recorded_allocation(manifest).per_layer == allocation.per_layer


def test_two_solves_emit_identical_manifests() -> None:
    problem = _proxy_problem()
    first = manifest_to_json(build_manifest(solve_greedy(problem), problem))
    second = manifest_to_json(build_manifest(solve_greedy(problem), problem))
    assert first == second


def test_two_cp_sat_solves_emit_identical_manifests() -> None:
    pytest.importorskip("ortools.sat.python.cp_model", reason="ortools (alloc extra) not installed")
    problem = _proxy_problem()
    first = manifest_to_json(build_manifest(solve_ortools(problem), problem))
    second = manifest_to_json(build_manifest(solve_ortools(problem), problem))
    # only the wall-clock diagnostic differs; the allocation and its tables are identical
    first_doc = json.loads(first)
    second_doc = json.loads(second)
    first_doc["diagnostics"].pop("ortools_wall_time_s", None)
    second_doc["diagnostics"].pop("ortools_wall_time_s", None)
    assert first_doc == second_doc


# ---------------------------------------------------------------- committed artifact


def test_committed_frontier_manifests_replay() -> None:
    manifest_paths = sorted((FRONTIER_DIR / "manifests").glob("*.json"))
    assert manifest_paths, f"no frontier manifests under {FRONTIER_DIR}"
    for path in manifest_paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        replayed = reproduce_allocation(document)
        recorded = recorded_allocation(document)
        assert replayed.per_layer == recorded.per_layer, path.name
        assert replayed.accounted_bytes == recorded.accounted_bytes, path.name


def test_committed_frontier_artifact_reconciles_its_byte_claims() -> None:
    document = json.loads((FRONTIER_DIR / "frontier.json").read_text(encoding="utf-8"))
    reconciliation = document["byte_reconciliation"]
    assert reconciliation["all_within_tolerance"] is True
    assert reconciliation["max_relative_difference"] <= BYTE_PARITY_TOLERANCE
    for point in document["points"]:
        for arm in point["arms"].values():
            assert arm["byte_parity"]["accounted_bytes"] == arm["accounted_bytes"]
            assert arm["byte_parity"]["measured_bytes"] == arm["measured_bytes"]
            assert arm["byte_parity"]["within_tolerance"] is True
