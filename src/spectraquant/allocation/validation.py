"""Measured-byte reconciliation and equal-memory gating for allocations (Milestone 6).

The allocator constrains and reports **class-1** accounted bytes
(:mod:`spectraquant.quantization.accounting`). This module closes the loop by actually *serializing*
the quantized factors of an allocated configuration and comparing the resulting **class-3** measured
container size against the class-1 figure, under the preregistered byte-parity tolerance
(``preregistration.md`` section 5: ``|a - b| / b <= 0.5 %``).

Two honest outcomes are possible and both are representable:

* a **headerless SpectraQuant container** stores exactly the accounted terms (padded payload +
  per-block scales + zero-points), so the measured size equals the accounted size and the relative
  difference is ``0.0``; the "container overhead" is then exactly zero bytes and is reported as such;
* a container that adds its own overhead (a third-party format, an ONNX invocation, ...) yields a
  positive difference, which is returned in full (``difference_bytes``, ``relative_difference``)
  rather than hidden, and :func:`assert_byte_parity` refuses the comparison when it exceeds the
  tolerance.

The equal-memory side builds the minimal :class:`~spectraquant.reporting.manifests.RunManifest` pair
that :func:`spectraquant.reporting.comparability.assert_equal_memory` requires, so a frontier
comparison is gated through the same mechanism every other compression comparison uses
(``AGENTS.md`` section 4.5; design note section 7.6).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from spectraquant.allocation.problem import (
    Allocation,
    CostFn,
    QuantCostModel,
    make_accounting_cost_fn,
)
from spectraquant.factorization import truncated_svd
from spectraquant.quantization import (
    SQ_CONTAINER_FORMAT_ID,
    QuantSpec,
    measure_serialized_bytes,
)
from spectraquant.reporting.comparability import ComparisonVerdict, assert_equal_memory
from spectraquant.reporting.manifests import (
    CompressionBlock,
    HardwareBlock,
    RunManifest,
    TrainingBlock,
    utc_timestamp,
)

__all__ = [
    "BYTE_PARITY_TOLERANCE",
    "accounted_allocation_bytes",
    "allocation_factor_state",
    "assert_byte_parity",
    "equal_memory_manifest",
    "gate_equal_memory",
    "reconcile_allocation_bytes",
    "reconciliation_of_allocation",
]

#: Preregistered byte-parity tolerance (``preregistration.md`` section 5): a relative byte
#: difference above ``0.5 %`` is not an equal-memory comparison.
BYTE_PARITY_TOLERANCE: float = 0.005


def allocation_factor_state(
    weights: Mapping[str, torch.Tensor],
    per_layer: Mapping[str, tuple[int, int]],
    cost_model: QuantCostModel,
) -> tuple[dict[str, torch.Tensor], dict[str, QuantSpec]]:
    """Truncate and describe every allocated layer as a serializable factor state.

    Args:
        weights: ``layer -> (out, in)`` uncompressed weight (CPU float tensor).
        per_layer: ``layer -> (rank, bits)`` allocation.
        cost_model: The quantization configuration (its ``spec_for_bits`` supplies the spec).

    Returns:
        ``(state, specs)`` where ``state[f"{layer}.A"]`` / ``state[f"{layer}.B"]`` are the two factor
        tensors (``A: (rank, in)``, ``B: (out, rank)``) and ``specs`` maps the same keys to specs.

    Raises:
        ValueError: If a requested rank is not the effective rank the factorization stores (a rank
            larger than the layer's intrinsic rank), because the class-1 accounting is then evaluated
            on a shape the serialized factors do not have and any byte parity would be vacuous.
        KeyError: If ``per_layer`` names a layer absent from ``weights``.
    """
    state: dict[str, torch.Tensor] = {}
    specs: dict[str, QuantSpec] = {}
    for layer in sorted(per_layer):
        rank, bits = int(per_layer[layer][0]), int(per_layer[layer][1])
        weight = weights[layer]
        spec = cost_model.spec_for_bits(bits)
        factors = truncated_svd(weight, rank)
        if factors.rank != rank:
            raise ValueError(
                f"layer {layer!r}: requested rank {rank} is not the stored effective rank "
                f"{factors.rank}; the class-1 cost would be accounted on a shape the serialized "
                "factors do not have (restrict the allowed ranks to <= min(out, in))"
            )
        state[f"{layer}.A"] = factors.A
        state[f"{layer}.B"] = factors.B
        specs[f"{layer}.A"] = spec
        specs[f"{layer}.B"] = spec
    return state, specs


def accounted_allocation_bytes(
    layer_shapes: Mapping[str, tuple[int, int]],
    per_layer: Mapping[str, tuple[int, int]],
    cost_model: QuantCostModel,
) -> int:
    """Class-1 accounted bytes of one allocation, from the accounting cost function itself."""
    cost_fn: CostFn = make_accounting_cost_fn(layer_shapes, cost_model)
    return sum(cost_fn(layer, int(rank), int(bits)) for layer, (rank, bits) in per_layer.items())


def reconcile_allocation_bytes(
    *,
    weights: Mapping[str, torch.Tensor],
    per_layer: Mapping[str, tuple[int, int]],
    cost_model: QuantCostModel,
    tolerance: float = BYTE_PARITY_TOLERANCE,
) -> dict[str, object]:
    """Serialize an allocated configuration and compare class-3 measured against class-1 bytes.

    Args:
        weights: ``layer -> (out, in)`` uncompressed CPU weight.
        per_layer: ``layer -> (rank, bits)`` allocation.
        cost_model: The quantization configuration.
        tolerance: Relative byte-parity tolerance (default: preregistered ``0.5 %``).

    Returns:
        A JSON-ready dict with ``accounted_bytes`` (class 1), ``measured_bytes`` (class 3),
        ``difference_bytes`` (measured - accounted), ``relative_difference``,
        ``container_overhead_bytes`` (identical to ``difference_bytes`` for this pairing: the whole
        gap is container overhead), ``tolerance``, ``within_tolerance``, ``serializer`` and
        ``n_layer_files``.
    """
    layer_shapes = {name: (int(w.shape[0]), int(w.shape[1])) for name, w in weights.items()}
    accounted = accounted_allocation_bytes(layer_shapes, per_layer, cost_model)
    state, specs = allocation_factor_state(weights, per_layer, cost_model)
    measured = int(measure_serialized_bytes(state, specs))
    difference = measured - accounted
    relative = abs(difference) / accounted if accounted else float("inf")
    return {
        "accounted_bytes": int(accounted),
        "measured_bytes": int(measured),
        "difference_bytes": int(difference),
        "container_overhead_bytes": int(difference),
        "relative_difference": float(relative),
        "tolerance": float(tolerance),
        "within_tolerance": bool(relative <= tolerance),
        "serializer": SQ_CONTAINER_FORMAT_ID,
        "n_layer_files": len(state),
    }


def assert_byte_parity(reconciliation: Mapping[str, Any]) -> None:
    """Raise :class:`AssertionError` when a reconciliation exceeds the byte-parity tolerance."""
    if not bool(reconciliation["within_tolerance"]):
        raise AssertionError(
            "class-3 measured bytes differ from class-1 accounted bytes by "
            f"{reconciliation['relative_difference']:.6%} "
            f"(accounted {reconciliation['accounted_bytes']} B, measured "
            f"{reconciliation['measured_bytes']} B, container overhead "
            f"{reconciliation['container_overhead_bytes']} B), above the "
            f"{float(reconciliation['tolerance']):.4%} tolerance"
        )


def equal_memory_manifest(
    run_id: str,
    *,
    measured_bytes: int,
    accounted_bytes: int | None = None,
    tolerance: float = BYTE_PARITY_TOLERANCE,
    serializer: str = SQ_CONTAINER_FORMAT_ID,
    seed: int = 0,
    method: str = "rtn",
) -> RunManifest:
    """Build a minimal measured-bytes :class:`RunManifest` for the equal-memory gate.

    The manifest declares ``bytes_source='measured'`` with the frozen serializer id, which is what
    :func:`~spectraquant.reporting.comparability.assert_equal_memory` requires to compare two arms at
    byte parity.
    """
    return RunManifest(
        run_id=run_id,
        timestamp_utc=utc_timestamp(),
        git_commit=None,
        git_dirty=None,
        config_path="scripts/experiments/allocation_frontier.py",
        resolved_config={"substrate": "LOCAL-FIXTURE"},
        split="fixture",
        seed=seed,
        hardware=HardwareBlock(
            platform="local-fixture",
            system="fixture",
            release="0",
            machine="cpu",
            torch_device="cpu",
        ),
        compression=CompressionBlock(
            method=method,
            accounted_bytes=accounted_bytes,
            measured_bytes=int(measured_bytes),
            measured_bytes_tolerance=float(tolerance),
            serializer=serializer,
            bytes_source="measured",
        ),
        measurement_class=3,
        training=TrainingBlock(steps=0, tokens=0, wall_time_s=0.0),
    )


def gate_equal_memory(
    reference: RunManifest,
    candidate: RunManifest,
    *,
    tolerance: float | None = None,
) -> ComparisonVerdict:
    """Gate two allocation manifest arms through :func:`assert_equal_memory`.

    By default the tolerance the manifests declare (``compression.measured_bytes_tolerance``, set to
    the preregistered ``0.5 %`` by :func:`equal_memory_manifest`) is used and converted to an absolute
    byte budget by the comparability module itself. An explicit ``tolerance`` fraction overrides it.

    Raises:
        UnequalMemoryComparison: If the arms are not at measured byte parity.
    """
    tolerance_bytes: int | None = None
    if tolerance is not None:
        reference_bytes = int(reference.compression.measured_bytes or 0)
        candidate_bytes = int(candidate.compression.measured_bytes or 0)
        tolerance_bytes = max(1, math.ceil(tolerance * max(reference_bytes, candidate_bytes)))
    return assert_equal_memory(reference, candidate, tolerance_bytes=tolerance_bytes)


def reconciliation_of_allocation(
    allocation: Allocation,
    weights: Mapping[str, torch.Tensor],
    cost_model: QuantCostModel,
    *,
    tolerance: float = BYTE_PARITY_TOLERANCE,
) -> dict[str, object]:
    """Convenience wrapper: reconcile an :class:`Allocation` against its own layer map."""
    return reconcile_allocation_bytes(
        weights=weights, per_layer=allocation.per_layer, cost_model=cost_model, tolerance=tolerance
    )
