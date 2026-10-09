"""The allocation problem, its solution record, and the accounting-backed cost model.

Design note: ``docs/coordination/design-m2-interfaces.md`` section 4 (owner: allocation stream).
This module is the **only** place that turns a layer shape plus a ``(rank, bits)`` choice into a
byte count; it does so by calling :func:`spectraquant.quantization.accounted_bytes` — the one
byte-accounting source of truth (invariant 1) — on the two factor tensors of the project's single
factorization convention (invariant 2):

* ``W (out, in) ≈ B @ A`` with ``A: (rank, in)`` and ``B: (out, rank)``.

The **constrained** quantity and the **reported** quantity are the same number. For every layer the
allocator pays

    total_cost(layer, rank, bits) = cost_fn(layer, rank, bits) + overhead_fn(layer, rank, bits)

where ``cost_fn`` is accounting-backed and ``overhead_fn`` carries the residual/outlier/alignment
extras. :attr:`Allocation.accounted_bytes` is exactly ``sum(total_cost(...))`` over the chosen
per-layer options, so a plan that passes the budget check cannot report more bytes than the budget
(review §6, blocking issue B16; design note §7.6).

Approximation labelling (AGENTS.md section 4.13)
------------------------------------------------
``accounted_bytes`` is **class 1 (analytical)** — it is the memory *model* the packer must match,
never a measured class-3 size. Real serialized-byte validation arrives at M6. Every number produced
here must carry the class-1 label.

Deterministic tie-breaking is documented on each solver in :mod:`spectraquant.allocation.solvers`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from spectraquant.quantization import SUPPORTED_BITS, QuantSpec, accounted_bytes

__all__ = [
    "COST_MODEL_VERSION",
    "Allocation",
    "AllocationProblem",
    "InfeasibleBudget",
    "QuantCostModel",
    "make_accounting_cost_fn",
    "make_problem",
    "zero_overhead",
]

#: Version of the cost model implemented by :func:`make_accounting_cost_fn`. A manifest records it
#: and a replay refuses a mismatch. Any change to the byte formula (which shapes are accounted, which
#: accounting function is called, or the factor convention) MUST bump this string.
COST_MODEL_VERSION = "spectraquant.allocation.cost.v1"

#: Type of the per-layer cost/overhead callables: ``(layer, rank, bits) -> bytes``.
CostFn = Callable[[str, int, int], int]
#: Type of the per-layer error callable: ``(layer, rank, bits) -> error`` (lower is better).
ErrorFn = Callable[[str, int, int], float]


def zero_overhead(layer: str, rank: int, bits: int) -> int:
    """Default ``overhead_fn``: no residual/outlier/alignment extras (returns 0 bytes).

    Args:
        layer: Layer name (ignored).
        rank: Factorization rank (ignored).
        bits: Weight bit width (ignored).

    Returns:
        ``0`` — the pure accounting cost is the whole cost.
    """
    del layer, rank, bits
    return 0


class InfeasibleBudget(ValueError):
    """Raised when no allocation in the allowed space fits the byte budget.

    Attributes:
        minimum_budget_bytes: The cheapest achievable total over all layers (each layer independently
            at its minimum-cost option), i.e. the smallest budget for which a feasible allocation
            exists. This is the number the caller must raise the budget to.
        budget_bytes: The budget that was rejected.
    """

    def __init__(self, minimum_budget_bytes: int, budget_bytes: int) -> None:
        self.minimum_budget_bytes = int(minimum_budget_bytes)
        self.budget_bytes = int(budget_bytes)
        super().__init__(
            f"infeasible budget: {budget_bytes} B < minimum feasible {minimum_budget_bytes} B"
        )


@dataclass(frozen=True)
class QuantCostModel:
    """The quantization configuration the accounting cost is evaluated under.

    A pure, serializable descriptor: it records the granularity/group/axis convention so a manifest
    can rebuild the identical :func:`accounted_bytes` calls without storing a callable.

    Args:
        granularity: ``"per_tensor"``, ``"per_channel"`` or ``"per_group"`` (``QuantSpec``).
        group_size: Elements sharing one scale; required for ``per_group`` and ``None`` otherwise.
        symmetric: Symmetric (signed, zero zero-point) versus asymmetric quantization.
        axis: Quantized axis of each factor tensor. Defaults to ``1`` (the factor's *input/rank*
            axis), which reproduces ``blocks_A = rank·ceil(in/g)`` and ``blocks_B = out·ceil(rank/g)``
            of ``docs/protocols/memory-accounting.md`` section 4. ``axis=0`` would place the blocks on
            the other factor axis and is not the project convention.
        round_mode: ``"nearest"`` only in M2.

    Raises:
        ValueError: On an invalid combination, via :class:`~spectraquant.quantization.QuantSpec`.
    """

    granularity: str = "per_group"
    group_size: int | None = 128
    symmetric: bool = True
    axis: int = 1
    round_mode: str = "nearest"

    def __post_init__(self) -> None:
        # Validate through QuantSpec itself so the cost model cannot encode a configuration the
        # quantizer would reject.
        self.spec_for_bits(SUPPORTED_BITS[0])

    def spec_for_bits(self, bits: int) -> QuantSpec:
        """Return the :class:`QuantSpec` this model uses for a given payload bit width.

        Args:
            bits: Payload bit width; one of :data:`spectraquant.quantization.SUPPORTED_BITS`.

        Returns:
            A validated :class:`QuantSpec`.

        Raises:
            ValueError: If ``bits`` is not a supported width.
        """
        return QuantSpec(
            bits=bits,
            granularity=self.granularity,
            group_size=self.group_size,
            symmetric=self.symmetric,
            axis=self.axis,
            round_mode=self.round_mode,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable description of this cost model."""
        return {
            "granularity": self.granularity,
            "group_size": self.group_size,
            "symmetric": self.symmetric,
            "axis": self.axis,
            "round_mode": self.round_mode,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> QuantCostModel:
        """Rebuild a cost model from :meth:`to_dict` output."""
        return cls(
            granularity=str(data["granularity"]),
            group_size=None if data.get("group_size") is None else int(data["group_size"]),
            symmetric=bool(data["symmetric"]),
            axis=int(data.get("axis", 1)),
            round_mode=str(data.get("round_mode", "nearest")),
        )


def _normalize_layer_shapes(
    layer_shapes: Mapping[str, Sequence[int]],
) -> dict[str, tuple[int, int]]:
    """Validate and normalize ``layer -> (out, in)`` shapes to a plain dict of 2-tuples."""
    if not layer_shapes:
        raise ValueError("layer_shapes must not be empty")
    normalized: dict[str, tuple[int, int]] = {}
    for name, shape in layer_shapes.items():
        dims = tuple(int(d) for d in shape)
        if len(dims) != 2:
            raise ValueError(f"layer {name!r}: expected an (out, in) 2-tuple, got {dims!r}")
        out, in_features = dims
        if out <= 0 or in_features <= 0:
            raise ValueError(f"layer {name!r}: dimensions must be positive, got {dims!r}")
        normalized[str(name)] = (out, in_features)
    return normalized


def _normalize_choices(values: Sequence[int], *, what: str) -> tuple[int, ...]:
    """Validate a non-empty sequence of positive ints and return it as a tuple."""
    items = tuple(values)
    if not items:
        raise ValueError(f"{what} must not be empty")
    for value in items:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{what} must be positive ints, got {value!r}")
    return items


def make_accounting_cost_fn(
    layer_shapes: Mapping[str, Sequence[int]], cost_model: QuantCostModel
) -> CostFn:
    """Build the accounting-backed ``cost_fn`` for a set of layer shapes.

    For each layer the two factor tensors ``A: (rank, in)`` and ``B: (out, rank)`` are quantized
    under ``cost_model.spec_for_bits(bits)`` and their class-1 accounted bytes are summed — including
    padding, per-block scales and (when asymmetric) zero-points. No byte cost is re-derived here.

    Args:
        layer_shapes: ``layer -> (out, in)`` weight-matrix shapes (the factorization convention).
        cost_model: The quantization configuration to account under.

    Returns:
        A callable ``(layer, rank, bits) -> int`` of class-1 accounted bytes for the factorized layer.

    Raises:
        KeyError: If asked for a layer not present in ``layer_shapes``.
        ValueError: On a non-positive ``rank`` or an unsupported ``bits`` (via ``QuantSpec``).
    """
    shapes = _normalize_layer_shapes(layer_shapes)

    def cost_fn(layer: str, rank: int, bits: int) -> int:
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise ValueError(f"rank must be a positive int, got {rank!r}")
        out_features, in_features = shapes[layer]
        spec = cost_model.spec_for_bits(bits)
        factors_a = (rank, in_features)  # A: (rank, in_features)
        factors_b = (out_features, rank)  # B: (out_features, rank)
        return accounted_bytes(factors_a, spec) + accounted_bytes(factors_b, spec)

    return cost_fn


@dataclass(frozen=True)
class AllocationProblem:
    """Layer-wise rank/bit allocation problem (design note section 4).

    Args:
        layer_shapes: ``layer -> (out, in)`` for every compressible weight matrix.
        ranks: Allowed factorization ranks (positive ints).
        bits: Allowed payload bit widths.
        budget_bytes: Byte budget for the **whole** model. Asserted against the same
            ``cost_fn + overhead_fn`` total that :attr:`Allocation.accounted_bytes` reports.
        cost_fn: Per-layer accounting-backed cost ``(layer, rank, bits) -> bytes``. Build it with
            :func:`make_accounting_cost_fn` (or :func:`make_problem`) so it comes from
            :mod:`spectraquant.quantization.accounting`.
        error_fn: Injected per-layer error model ``(layer, rank, bits) -> float`` (lower is better).
            The real proxy is wired at M6; tests inject analytic error functions so the solvers are
            independently checkable.
        overhead_fn: Residual/outlier/alignment extras ``(layer, rank, bits) -> bytes``. Defaults to
            :func:`zero_overhead`. It is added to ``cost_fn`` for both the constraint and the report.
        cost_model: The :class:`QuantCostModel` ``cost_fn`` was built from, when known. Required for
            manifest emission/replay; ``None`` for arbitrary custom cost functions.
        cost_model_version: Version tag of the cost model (see :data:`COST_MODEL_VERSION`).

    Raises:
        ValueError: On malformed shapes, empty/negative choices, or a negative budget.
    """

    layer_shapes: Mapping[str, tuple[int, int]]
    ranks: Sequence[int]
    bits: Sequence[int]
    budget_bytes: int
    cost_fn: CostFn
    error_fn: ErrorFn
    overhead_fn: CostFn = zero_overhead
    cost_model: QuantCostModel | None = None
    cost_model_version: str = COST_MODEL_VERSION

    def __post_init__(self) -> None:
        normalized = _normalize_layer_shapes(self.layer_shapes)
        object.__setattr__(self, "layer_shapes", normalized)
        object.__setattr__(self, "ranks", _normalize_choices(self.ranks, what="ranks"))
        object.__setattr__(self, "bits", _normalize_choices(self.bits, what="bits"))
        budget = self.budget_bytes
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
            raise ValueError(f"budget_bytes must be a non-negative int, got {budget!r}")
        for rank in self.ranks:
            for bits in self.bits:
                for name in normalized:
                    self.total_cost(name, rank, bits)

    @property
    def layer_names(self) -> tuple[str, ...]:
        """Layer names in a deterministic (sorted) order."""
        return tuple(sorted(self.layer_shapes))

    @property
    def options(self) -> tuple[tuple[int, int], ...]:
        """All allowed ``(rank, bits)`` options in a deterministic order.

        Ordered by ``(bits, rank)`` ascending, so ties in a solver are broken identically on every
        run and platform.
        """
        return tuple(
            sorted(
                ((rank, bits) for rank in self.ranks for bits in self.bits),
                key=lambda option: (option[1], option[0]),
            )
        )

    def options_for(self, layer: str) -> tuple[tuple[int, int], ...]:
        """The allowed ``(rank, bits)`` options for one layer (validating the layer name)."""
        if layer not in self.layer_shapes:
            raise KeyError(f"unknown layer {layer!r}")
        return self.options

    def total_cost(self, layer: str, rank: int, bits: int) -> int:
        """``cost_fn + overhead_fn`` for one layer and one option — the quantity that is constrained.

        Invariant: this is the **only** function used both to check the budget and to report
        :attr:`Allocation.accounted_bytes` (design note §7.6).
        """
        return int(self.cost_fn(layer, rank, bits)) + int(self.overhead_fn(layer, rank, bits))

    def assignment_cost(self, assignment: Mapping[str, tuple[int, int]]) -> int:
        """Total accounted bytes of a full ``layer -> (rank, bits)`` assignment."""
        return sum(self.total_cost(name, *assignment[name]) for name in self.layer_names)

    def assignment_error(self, assignment: Mapping[str, tuple[int, int]]) -> float:
        """Total predicted error of a full assignment (sum of per-layer errors)."""
        return float(sum(self.error_fn(name, *assignment[name]) for name in self.layer_names))

    def cheapest_option(self, layer: str) -> tuple[int, int]:
        """The minimum-cost option for one layer, tie-broken by ``(bits, rank)`` ascending."""
        options = self.options
        return min(
            options,
            key=lambda option: (self.total_cost(layer, *option), option[1], option[0]),
        )

    def min_feasible_budget(self) -> int:
        """The smallest budget for which any allocation exists.

        Because the budget is a sum of independent per-layer costs, this is the sum of each layer's
        cheapest option — the exact ``minimum_feasible_budget`` an :class:`InfeasibleBudget` carries.
        """
        return sum(self.total_cost(name, *self.cheapest_option(name)) for name in self.layer_names)

    def check_budget(self) -> None:
        """Raise :class:`InfeasibleBudget` when the budget cannot be met at all."""
        minimum = self.min_feasible_budget()
        if minimum > self.budget_bytes:
            raise InfeasibleBudget(minimum, self.budget_bytes)


def make_problem(
    *,
    layer_shapes: Mapping[str, Sequence[int]],
    ranks: Sequence[int],
    bits: Sequence[int],
    budget_bytes: int,
    cost_model: QuantCostModel,
    error_fn: ErrorFn,
    overhead_fn: CostFn = zero_overhead,
) -> AllocationProblem:
    """Build an :class:`AllocationProblem` whose ``cost_fn`` is accounting-backed.

    This is the supported constructor: it derives ``cost_fn`` from ``cost_model`` via
    :func:`make_accounting_cost_fn`, so the class-1 byte cost always comes from
    :mod:`spectraquant.quantization.accounting` and the problem is emittable to a replayable manifest.

    Args:
        layer_shapes: ``layer -> (out, in)`` weight shapes.
        ranks: Allowed ranks.
        bits: Allowed bit widths.
        budget_bytes: Total byte budget.
        cost_model: Quantization configuration to account under.
        error_fn: Injected per-layer error model.
        overhead_fn: Extra per-layer bytes; defaults to none.

    Returns:
        A fully wired :class:`AllocationProblem` with ``cost_model`` set.
    """
    normalized_shapes = _normalize_layer_shapes(layer_shapes)
    return AllocationProblem(
        layer_shapes=normalized_shapes,
        ranks=ranks,
        bits=bits,
        budget_bytes=budget_bytes,
        cost_fn=make_accounting_cost_fn(normalized_shapes, cost_model),
        error_fn=error_fn,
        overhead_fn=overhead_fn,
        cost_model=cost_model,
    )


@dataclass(frozen=True)
class Allocation:
    """The result of solving an :class:`AllocationProblem` (design note section 4).

    Args:
        per_layer: ``layer -> (rank, bits)`` chosen for every layer in the problem.
        accounted_bytes: **The constrained quantity**, ``sum(cost_fn + overhead_fn)`` over
            ``per_layer``. Class 1 (analytical); never a measured size.
        predicted_error: ``sum(error_fn)`` over ``per_layer`` (lower is better).
        solver: Name of the solver that produced this allocation.
        feasible: Whether ``accounted_bytes <= budget_bytes`` (always ``True`` for a returned
            allocation; solvers raise :class:`InfeasibleBudget` otherwise). Retained as a field so a
            manifest records the claim explicitly.
        diagnostics: Numeric solver diagnostics (headroom, option counts, status codes, timings).
    """

    per_layer: Mapping[str, tuple[int, int]]
    accounted_bytes: int
    predicted_error: float
    solver: str
    feasible: bool
    diagnostics: Mapping[str, float]

    def as_assignment(self) -> dict[str, tuple[int, int]]:
        """Return ``per_layer`` as a plain mutable dict of ``(rank, bits)`` tuples."""
        return {name: (int(rank), int(bits)) for name, (rank, bits) in self.per_layer.items()}

    def to_dict(self) -> dict[str, Any]:
        """Return the allocation record (without the problem/manifest context) as JSON-ready data."""
        return {
            "solver": self.solver,
            "feasible": self.feasible,
            "accounted_bytes": int(self.accounted_bytes),
            "predicted_error": float(self.predicted_error),
            "per_layer": {
                name: [int(rank), int(bits)]
                for name, (rank, bits) in sorted(self.per_layer.items())
            },
            "diagnostics": {key: float(value) for key, value in sorted(self.diagnostics.items())},
        }
