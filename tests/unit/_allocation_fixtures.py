"""Shared fixtures for the allocation tests (owner: allocation stream).

The error callables here are **analytic** functions defined in the test suite, not the real proxy:
the proxy-backed ``error_fn`` is wired at integration (M6). Keeping them analytic makes the solvers
independently checkable — the oracle's optimum can be reasoned about by hand.
"""

from __future__ import annotations

from collections.abc import Mapping

from spectraquant.allocation import (
    AllocationProblem,
    QuantCostModel,
    make_problem,
    zero_overhead,
)

#: Two small layers used across solver/manifest tests.
TINY_SHAPES: dict[str, tuple[int, int]] = {"blocks.0.attn": (64, 32), "blocks.0.mlp": (32, 16)}

#: Per-layer error scales for the default analytic error model.
TINY_SCALES: dict[str, float] = {"blocks.0.attn": 4.0, "blocks.0.mlp": 1.0}


def per_group_cost_model(group_size: int = 16, *, symmetric: bool = True) -> QuantCostModel:
    """The default class-1 cost model used by the tests (per-group, fp16 scales, factor axis 1)."""
    return QuantCostModel(
        granularity="per_group", group_size=group_size, symmetric=symmetric, axis=1
    )


def power_law_error(scales: Mapping[str, float]):
    """Analytic per-layer error model ``scale * (1/rank + 2**-bits)`` (decreasing in rank and bits)."""

    def error_fn(layer: str, rank: int, bits: int) -> float:
        return float(scales[layer]) * (1.0 / rank + 2.0 ** (-bits))

    return error_fn


def build_tiny_problem(
    *,
    budget_bytes: int,
    layer_shapes: Mapping[str, tuple[int, int]] | None = None,
    scales: Mapping[str, float] | None = None,
    ranks: tuple[int, ...] = (2, 4, 8),
    bits: tuple[int, ...] = (4, 8),
    cost_model: QuantCostModel | None = None,
) -> AllocationProblem:
    """Build a small accounting-backed problem with the analytic error model."""
    shapes = dict(layer_shapes or TINY_SHAPES)
    error_scales = dict(scales or TINY_SCALES)
    return make_problem(
        layer_shapes=shapes,
        ranks=ranks,
        bits=bits,
        budget_bytes=budget_bytes,
        cost_model=cost_model or per_group_cost_model(),
        error_fn=power_law_error(error_scales),
        overhead_fn=zero_overhead,
    )


def budget_range(problem: AllocationProblem) -> tuple[int, int]:
    """Return ``(minimum, maximum)`` total cost over the whole option space."""
    minimum = problem.min_feasible_budget()
    maximum = sum(
        max(problem.total_cost(name, *option) for option in problem.options)
        for name in problem.layer_names
    )
    return minimum, maximum
