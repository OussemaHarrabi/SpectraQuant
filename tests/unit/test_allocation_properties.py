"""Property tests over random tiny allocation problems (Hypothesis).

The oracle (:func:`solve_exhaustive`) is the global minimiser of the predicted error, so it is a lower
bound on every other solver's objective. These tests also assert that no solver ever exceeds the byte
budget, for randomly generated feasibility-ranges.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from spectraquant.allocation import (
    QuantCostModel,
    make_problem,
    solve_exhaustive,
    solve_greedy,
    solve_uniform,
)


def _error_model(scales: dict[str, float]):
    def error_fn(layer: str, rank: int, bits: int) -> float:
        return scales[layer] * (1.0 / rank + 2.0 ** (-bits))

    return error_fn


@settings(max_examples=75, deadline=None)
@given(data=st.data())
def test_oracle_is_a_lower_bound_on_greedy_and_uniform(data) -> None:
    n_layers = data.draw(st.integers(min_value=1, max_value=3), label="n_layers")
    names = [f"l{i}" for i in range(n_layers)]
    shapes = {
        name: (
            data.draw(st.integers(min_value=8, max_value=32), label=f"{name}_out"),
            data.draw(st.integers(min_value=8, max_value=32), label=f"{name}_in"),
        )
        for name in names
    }
    scales = {
        name: data.draw(st.floats(min_value=0.5, max_value=8.0), label=f"{name}_scale")
        for name in names
    }
    ranks = tuple(
        data.draw(
            st.lists(st.sampled_from([1, 2, 4, 8]), min_size=1, max_size=2, unique=True),
            label="ranks",
        )
    )
    bits = tuple(
        data.draw(
            st.lists(st.sampled_from([4, 8]), min_size=1, max_size=2, unique=True), label="bits"
        )
    )
    cost_model = QuantCostModel(granularity="per_group", group_size=8, symmetric=True, axis=1)

    def build(budget: int):
        return make_problem(
            layer_shapes=shapes,
            ranks=ranks,
            bits=bits,
            budget_bytes=budget,
            cost_model=cost_model,
            error_fn=_error_model(scales),
        )

    problem = build(budget=0)
    minimum = problem.min_feasible_budget()
    maximum = sum(
        max(problem.total_cost(name, *option) for option in problem.options) for name in names
    )
    budget = data.draw(st.integers(min_value=minimum, max_value=maximum), label="budget")
    problem = build(budget)

    oracle = solve_exhaustive(problem)
    greedy = solve_greedy(problem)
    uniform = solve_uniform(problem)

    for allocation in (oracle, greedy, uniform):
        assert allocation.accounted_bytes <= budget
        # The reported number is the constrained number: the same sum the solver checked.
        assert allocation.accounted_bytes == sum(
            problem.total_cost(name, *allocation.per_layer[name]) for name in names
        )

    assert oracle.predicted_error <= greedy.predicted_error + 1e-12
    assert oracle.predicted_error <= uniform.predicted_error + 1e-12
