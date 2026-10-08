"""Allocation package (Milestone 4 — not implemented).

Planned responsibilities: layer-wise rank and bit-width allocation under an explicit memory budget,
feasibility checking against the analytical memory model (class 1), and the equal-memory baseline
that every compression comparison requires (``AGENTS.md`` section 4.5).

Every entry point raises ``NotImplementedError``; see :mod:`spectraquant.allocation.allocator`.
"""

from __future__ import annotations

from spectraquant.allocation.allocator import (
    AllocationPlan,
    allocate_ranks_and_bits,
    budget_feasible,
)

__all__ = ["AllocationPlan", "allocate_ranks_and_bits", "budget_feasible"]
