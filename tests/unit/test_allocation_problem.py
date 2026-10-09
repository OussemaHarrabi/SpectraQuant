"""``AllocationProblem``: accounting-backed cost, validation, and the equal-memory invariant.

The central property under test is design-note §7.6 / review §6 (B16): the quantity the budget
constraint is applied to and the quantity an :class:`Allocation` reports must be **the same number**,
and both must move when ``overhead_fn`` moves.
"""

from __future__ import annotations

import pytest
from _allocation_fixtures import (
    TINY_SHAPES,
    build_tiny_problem,
    per_group_cost_model,
    power_law_error,
)

from spectraquant.allocation import (
    InfeasibleBudget,
    QuantCostModel,
    make_accounting_cost_fn,
    make_problem,
)
from spectraquant.quantization import QuantSpec, accounted_bytes, byte_breakdown

GROUP = 16


def test_cost_fn_calls_the_single_accounting_source() -> None:
    """``cost_fn`` is exactly ``accounted_bytes`` on the two factor tensors — no re-derivation."""
    model = per_group_cost_model(GROUP)
    cost_fn = make_accounting_cost_fn(TINY_SHAPES, model)
    out, in_features = TINY_SHAPES["blocks.0.attn"]
    for rank, bits in ((2, 4), (4, 8), (8, 8)):
        spec = model.spec_for_bits(bits)
        expected = accounted_bytes((rank, in_features), spec) + accounted_bytes((out, rank), spec)
        assert cost_fn("blocks.0.attn", rank, bits) == expected


def test_default_axis_reproduces_memory_accounting_section4_example() -> None:
    """The int4 r=64, group=128, 4096x4096 factor budget of ``memory-accounting.md`` §7.3 = 274 432 B."""
    model = QuantCostModel(granularity="per_group", group_size=128, symmetric=True, axis=1)
    cost_fn = make_accounting_cost_fn({"proj": (4096, 4096)}, model)
    assert cost_fn("proj", 64, 4) == 274_432


def test_accounted_bytes_is_the_constrained_quantity() -> None:
    """A solved allocation's ``accounted_bytes`` equals the sum the budget was checked against."""
    problem = build_tiny_problem(budget_bytes=1000)
    from spectraquant.allocation import solve_greedy

    allocation = solve_greedy(problem)
    recomputed = sum(
        problem.cost_fn(name, *allocation.per_layer[name])
        + problem.overhead_fn(name, *allocation.per_layer[name])
        for name in problem.layer_names
    )
    assert allocation.accounted_bytes == recomputed
    assert allocation.accounted_bytes <= problem.budget_bytes


def test_mutating_overhead_fn_changes_both_the_constraint_and_the_report() -> None:
    """The review's B16 requirement: ``overhead_fn`` must move the constrained and reported number.

    With a per-layer constant overhead the minimum feasible budget and the reported bytes both rise by
    ``overhead * n_layers``; a budget that is feasible at zero overhead becomes infeasible.
    """
    base = build_tiny_problem(budget_bytes=1000)
    baseline_min = base.min_feasible_budget()
    overhead_per_call = 37

    def overcharged(layer: str, rank: int, bits: int) -> int:
        del rank, bits
        return overhead_per_call if layer in base.layer_shapes else 0

    charged = make_problem(
        layer_shapes=TINY_SHAPES,
        ranks=(2, 4, 8),
        bits=(4, 8),
        budget_bytes=1000,
        cost_model=per_group_cost_model(GROUP),
        error_fn=power_law_error({"blocks.0.attn": 4.0, "blocks.0.mlp": 1.0}),
        overhead_fn=overcharged,
    )
    n_layers = len(base.layer_names)
    assert charged.min_feasible_budget() == baseline_min + overhead_per_call * n_layers

    # Same per-layer choice, but the reported number must now include the overhead.
    from spectraquant.allocation import solve_greedy

    plain = solve_greedy(base)
    costly = solve_greedy(charged)
    assert costly.accounted_bytes > plain.accounted_bytes
    assert costly.accounted_bytes <= 1000

    # Shrink the budget to the overhead-free minimum: the charged problem can no longer fit.
    infeasible = make_problem(
        layer_shapes=TINY_SHAPES,
        ranks=(2, 4, 8),
        bits=(4, 8),
        budget_bytes=baseline_min,
        cost_model=per_group_cost_model(GROUP),
        error_fn=power_law_error({"blocks.0.attn": 4.0, "blocks.0.mlp": 1.0}),
        overhead_fn=overcharged,
    )
    with pytest.raises(InfeasibleBudget) as excinfo:
        solve_greedy(infeasible)
    assert excinfo.value.minimum_budget_bytes == baseline_min + overhead_per_call * n_layers
    assert excinfo.value.budget_bytes == baseline_min


def test_infeasible_budget_carries_the_minimum() -> None:
    problem = build_tiny_problem(budget_bytes=1000)
    minimum = problem.min_feasible_budget()
    assert minimum > 0

    starved = build_tiny_problem(budget_bytes=minimum - 1)
    with pytest.raises(InfeasibleBudget) as excinfo:
        starved.check_budget()
    assert excinfo.value.minimum_budget_bytes == minimum
    assert excinfo.value.budget_bytes == minimum - 1


def test_min_feasible_budget_uses_cheapest_options() -> None:
    problem = build_tiny_problem(budget_bytes=10_000)
    expected = sum(
        problem.total_cost(name, *problem.cheapest_option(name)) for name in problem.layer_names
    )
    assert problem.min_feasible_budget() == expected


def test_options_are_ordered_deterministically_by_bits_then_rank() -> None:
    problem = build_tiny_problem(budget_bytes=10_000)
    assert problem.options == ((2, 4), (4, 4), (8, 4), (2, 8), (4, 8), (8, 8))


def test_byte_breakdown_consistency_of_the_factor_cost() -> None:
    """The cost is the sum of per-factor accounted bytes, each with counted group metadata."""
    model = per_group_cost_model(GROUP)
    cost_fn = make_accounting_cost_fn(TINY_SHAPES, model)
    spec = model.spec_for_bits(4)
    out, in_features = TINY_SHAPES["blocks.0.mlp"]
    rank = 4
    breakdown_a = byte_breakdown((rank, in_features), spec)
    breakdown_b = byte_breakdown((out, rank), spec)
    assert cost_fn("blocks.0.mlp", rank, 4) == breakdown_a.total_bytes + breakdown_b.total_bytes
    assert breakdown_a.group_metadata_bytes > 0
    assert breakdown_b.group_metadata_bytes > 0


def test_cost_model_rejects_unsupported_bit_width() -> None:
    model = per_group_cost_model(GROUP)
    with pytest.raises(ValueError):
        model.spec_for_bits(5)


def test_cost_model_round_trips_through_dict() -> None:
    model = QuantCostModel(granularity="per_channel", group_size=None, symmetric=False, axis=1)
    assert QuantCostModel.from_dict(model.to_dict()) == model


def test_problem_validation_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        build_tiny_problem(budget_bytes=-1)
    with pytest.raises(ValueError):
        make_problem(
            layer_shapes={},
            ranks=(2,),
            bits=(4,),
            budget_bytes=100,
            cost_model=per_group_cost_model(GROUP),
            error_fn=power_law_error({}),
        )
    with pytest.raises(ValueError):
        make_problem(
            layer_shapes={"x": (8, 8)},
            ranks=(),
            bits=(4,),
            budget_bytes=100,
            cost_model=per_group_cost_model(GROUP),
            error_fn=power_law_error({"x": 1.0}),
        )
    with pytest.raises(ValueError):
        make_problem(
            layer_shapes={"x": (8, 8, 8)},  # not a 2-D weight
            ranks=(2,),
            bits=(4,),
            budget_bytes=100,
            cost_model=per_group_cost_model(GROUP),
            error_fn=power_law_error({"x": 1.0}),
        )


def test_symmetric_and_asymmetric_cost_models_differ() -> None:
    symmetric = make_accounting_cost_fn(TINY_SHAPES, per_group_cost_model(GROUP, symmetric=True))
    asymmetric = make_accounting_cost_fn(TINY_SHAPES, per_group_cost_model(GROUP, symmetric=False))
    assert asymmetric("blocks.0.attn", 4, 4) > symmetric("blocks.0.attn", 4, 4)


def test_quant_spec_is_still_the_single_source_for_bit_semantics() -> None:
    """A guard that nothing here silently redefines supported bit widths."""
    model = per_group_cost_model(GROUP)
    assert isinstance(model.spec_for_bits(8), QuantSpec)
