"""Budget-aware layer-wise rank/bit allocation (Milestone 2, owner: allocation stream).

Public surface, frozen by ``docs/coordination/design-m2-interfaces.md`` section 4:

* :mod:`spectraquant.allocation.problem` — :class:`AllocationProblem`, :class:`Allocation`,
  :class:`InfeasibleBudget`, and the accounting-backed :class:`QuantCostModel`. The cost comes from
  :mod:`spectraquant.quantization.accounting` (the one byte-accounting source of truth); the
  constrained quantity and the reported ``accounted_bytes`` are the same ``cost_fn + overhead_fn``
  sum (design note §7.6, review §6 / B16).
* :mod:`spectraquant.allocation.solvers` — :func:`solve_uniform`, :func:`solve_greedy`,
  :func:`solve_exhaustive` (oracle), :func:`solve_ortools` (CP-SAT behind the optional ``alloc``
  extra). Deterministic tie-breaking is documented on each.
* :mod:`spectraquant.allocation.manifest` — deterministic JSON emission and replay
  (:func:`build_manifest`, :func:`reproduce_allocation`) carrying the git SHA, cost-model version,
  byte total and the full per-layer map.

The injected ``error_fn`` is the proxy hook: the real output-aware proxy is wired at integration
(M6). Until then the allocator optimizes whatever deterministic error model it is given, and every
byte figure is class 1 (analytical) — never a measured stored size.
"""

from __future__ import annotations

from spectraquant.allocation.manifest import (
    MANIFEST_SCHEMA_ID,
    CostModelMismatch,
    build_manifest,
    load_manifest,
    manifest_from_json,
    manifest_to_json,
    problem_from_manifest,
    recorded_allocation,
    reproduce_allocation,
    write_manifest,
)
from spectraquant.allocation.problem import (
    COST_MODEL_VERSION,
    Allocation,
    AllocationProblem,
    InfeasibleBudget,
    QuantCostModel,
    make_accounting_cost_fn,
    make_problem,
    zero_overhead,
)
from spectraquant.allocation.proxy_adapter import make_proxy_problem, proxy_error_fn
from spectraquant.allocation.solvers import (
    ERROR_SCALE,
    ORToolsNotInstalledError,
    ortools_available,
    scaled_error,
    solve_exhaustive,
    solve_greedy,
    solve_ortools,
    solve_uniform,
)
from spectraquant.allocation.validation import (
    BYTE_PARITY_TOLERANCE,
    accounted_allocation_bytes,
    allocation_factor_state,
    assert_byte_parity,
    equal_memory_manifest,
    gate_equal_memory,
    reconcile_allocation_bytes,
    reconciliation_of_allocation,
)

__all__ = [
    "BYTE_PARITY_TOLERANCE",
    "COST_MODEL_VERSION",
    "ERROR_SCALE",
    "MANIFEST_SCHEMA_ID",
    "Allocation",
    "AllocationProblem",
    "CostModelMismatch",
    "InfeasibleBudget",
    "ORToolsNotInstalledError",
    "QuantCostModel",
    "accounted_allocation_bytes",
    "allocation_factor_state",
    "assert_byte_parity",
    "build_manifest",
    "equal_memory_manifest",
    "gate_equal_memory",
    "load_manifest",
    "make_accounting_cost_fn",
    "make_problem",
    "make_proxy_problem",
    "manifest_from_json",
    "manifest_to_json",
    "ortools_available",
    "problem_from_manifest",
    "proxy_error_fn",
    "reconcile_allocation_bytes",
    "reconciliation_of_allocation",
    "recorded_allocation",
    "reproduce_allocation",
    "scaled_error",
    "solve_exhaustive",
    "solve_greedy",
    "solve_ortools",
    "solve_uniform",
    "write_manifest",
    "zero_overhead",
]
