"""Proxy -> allocator adapter (Milestone 6 integration).

The allocator's ``error_fn`` is an injected callable ``(layer, rank, bits) -> error``
(:mod:`spectraquant.allocation.problem`). Milestone 2 left it analytic; this module is the adapter
that wires the **real** sensitivity proxy into it, without touching the solvers.

It converts measured per-layer :class:`~spectraquant.proxies.base.LayerInputs` plus a declared
:class:`~spectraquant.proxies.base.Proxy` variant into exactly that callable, and assembles an
accounting-backed :class:`~spectraquant.allocation.problem.AllocationProblem` around it. The cost
side is unchanged: :func:`~spectraquant.allocation.problem.make_problem` derives ``cost_fn`` from
:mod:`spectraquant.quantization.accounting`, so the class-1 byte figures still come from the one
accounting source of truth.

Measurement classes (``AGENTS.md`` section 5): the returned ``error_fn`` is a **class-2**
fake-quantization quality estimate (the proxy unit is the plug-in mean squared layer-output error),
never a byte or latency figure. The ``cost_fn`` it is paired with is class 1.

Context convention
------------------
A proxy variant may need context (the following normalisation's Jacobian, a measured residual error,
or an estimated downstream gain) that itself depends on the ``(rank, bits)`` being scored. The
adapter therefore accepts either

* a fixed ``name -> LayerInputs`` mapping (the same context for every option), or
* a callable ``(layer, rank, bits) -> LayerInputs`` returning the option-specific context.

Using the option-specific form is the faithful choice when the context was measured at each option;
the fixed form is an explicit approximation (the context is held at one reference compression) and
the caller is responsible for labelling it as such.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TypeAlias

from spectraquant.allocation.problem import (
    AllocationProblem,
    CostFn,
    ErrorFn,
    QuantCostModel,
    make_problem,
    zero_overhead,
)
from spectraquant.proxies.base import LayerInputs, Proxy

__all__ = ["ProxyContext", "make_proxy_problem", "proxy_error_fn"]

#: A context source: one fixed :class:`LayerInputs` per layer, or a callable returning the inputs
#: for a specific ``(layer, rank, bits)`` option.
ProxyContext: TypeAlias = Mapping[str, LayerInputs] | Callable[[str, int, int], LayerInputs]


def _resolve_context(
    context: ProxyContext,
    layer: str,
    rank: int,
    bits: int,
) -> LayerInputs:
    """Return the :class:`LayerInputs` for one ``(layer, rank, bits)`` option."""
    inputs = context(layer, rank, bits) if callable(context) else context.get(layer)
    if inputs is None:
        raise KeyError(
            f"proxy context has no LayerInputs for layer {layer!r} "
            f"(option rank={rank}, bits={bits})"
        )
    if not isinstance(inputs, LayerInputs):  # pragma: no cover - defensive type guard
        raise TypeError(
            f"proxy context for layer {layer!r} must be a LayerInputs, got {type(inputs).__name__}"
        )
    return inputs


def proxy_error_fn(
    proxy: Proxy,
    context: ProxyContext,
    cost_model: QuantCostModel,
) -> ErrorFn:
    """Build the allocator ``error_fn`` from a proxy variant and per-layer context.

    Args:
        proxy: A proxy variant (e.g. :class:`~spectraquant.proxies.variants.GainAwareComposedProxy`,
            the declared candidate). Scored through the frozen public ``score_model`` entry point
            with a single-layer mapping, so every variant's context handling is honoured.
        context: Either ``name -> LayerInputs`` or ``(layer, rank, bits) -> LayerInputs``; see the
            module docstring for which to use.
        cost_model: The quantization cost model, used only to obtain the :class:`QuantSpec` for a bit
            width via :meth:`QuantCostModel.spec_for_bits` (so the proxy is scored under the exact
            spec the byte cost is accounted with).

    Returns:
        A callable ``(layer, rank, bits) -> float`` of class-2 predicted error (lower is better).

    Raises:
        ValueError: On a non-positive/non-int ``rank`` or an unsupported ``bits`` (via ``QuantSpec``).
        KeyError: If the context has no inputs for the requested layer.
    """

    if not isinstance(proxy, Proxy):  # pragma: no cover - defensive protocol check
        raise TypeError(f"proxy must implement the Proxy protocol, got {type(proxy).__name__}")

    def error_fn(layer: str, rank: int, bits: int) -> float:
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise ValueError(f"rank must be a positive int, got {rank!r}")
        spec = cost_model.spec_for_bits(bits)  # validates the bit width
        inputs = _resolve_context(context, layer, rank, bits)
        result = proxy.score_model({layer: inputs}, {layer: int(rank)}, {layer: spec})
        return float(result.value)

    return error_fn


def make_proxy_problem(
    *,
    proxy: Proxy,
    layer_inputs: ProxyContext,
    layer_shapes: Mapping[str, Sequence[int]],
    ranks: Sequence[int],
    bits: Sequence[int],
    budget_bytes: int,
    cost_model: QuantCostModel,
    overhead_fn: CostFn = zero_overhead,
) -> AllocationProblem:
    """Assemble an accounting-backed :class:`AllocationProblem` whose ``error_fn`` is the proxy.

    This is the supported M6 wiring: the proxy supplies the quality axis, the accounting module the
    byte axis. No solver is reimplemented here.

    Args:
        proxy: The proxy variant to score with.
        layer_inputs: The per-layer context (fixed mapping or option-specific callable).
        layer_shapes: ``layer -> (out, in)`` weight shapes (the factorization convention).
        ranks: Allowed ranks.
        bits: Allowed bit widths.
        budget_bytes: Total byte budget for the whole model.
        cost_model: Quantization configuration the class-1 cost is accounted under.
        overhead_fn: Optional residual/outlier/alignment extras (defaults to none).

    Returns:
        A fully wired problem with the accounting ``cost_fn`` and the proxy ``error_fn``.
    """
    return make_problem(
        layer_shapes=layer_shapes,
        ranks=ranks,
        bits=bits,
        budget_bytes=budget_bytes,
        cost_model=cost_model,
        error_fn=proxy_error_fn(proxy, layer_inputs, cost_model),
        overhead_fn=overhead_fn,
    )
