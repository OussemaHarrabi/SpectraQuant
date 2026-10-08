"""Budget allocation of layer-wise ranks and bit widths.

Milestone 4 (owner: allocation stream). The allocator consumes a sensitivity proxy plus a memory
budget and returns per-layer ranks and bit widths; it must be feasible (never exceed the budget)
and its output must be checked against the analytical memory model. Not implemented in the
bootstrap scaffold.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

__all__ = ["AllocationPlan", "allocate_ranks_and_bits", "budget_feasible"]


@dataclass(frozen=True)
class AllocationPlan:
    """Per-layer compression plan produced by the allocator.

    Attributes:
        ranks: ``layer -> rank`` for factorized layers.
        bits: ``layer -> bit width`` for quantized layers.
        predicted_bits_per_parameter: class-1 analytical estimate implied by the plan.
        budget_bytes: the budget the plan was required to respect.
    """

    ranks: Mapping[str, int]
    bits: Mapping[str, int]
    predicted_bits_per_parameter: float
    budget_bytes: int


def allocate_ranks_and_bits(
    sensitivity: Mapping[str, float],
    *,
    budget_bytes: int,
    layer_shapes: Mapping[str, Sequence[int]],
    rank_choices: Sequence[int] = (8, 16, 32, 64),
    bit_choices: Sequence[int] = (4, 8),
) -> AllocationPlan:
    """Choose per-layer ranks and bit widths under an explicit memory budget.

    Raises:
        NotImplementedError: Milestone 4 has not landed (allocation stream).
    """
    raise NotImplementedError(
        "allocate_ranks_and_bits is a Milestone 4 deliverable (allocation stream); the bootstrap "
        "scaffold ships no allocator implementation"
    )


def budget_feasible(plan: AllocationPlan, *, budget_bytes: int) -> bool:
    """Check that a plan respects the budget under the analytical (class 1) memory model.

    Raises:
        NotImplementedError: Milestone 4 has not landed (allocation stream).
    """
    raise NotImplementedError("budget_feasible is a Milestone 4 deliverable (allocation stream)")
