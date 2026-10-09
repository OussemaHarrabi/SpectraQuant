"""Downstream-gain estimation for the composition-aware proxy (``design-m2-interfaces.md`` section 7.2).

The measured problem: for a residual stack ``e_{l+1} = J_l e_l + delta_l`` the model-output error is
``sum_l (prod_{j>l} J_j) delta_l``, so a layer's contribution is its own error multiplied by the
**downstream gain product**. Summing per-layer proxies sets that product to 1, which the adversarial
review measured to be wrong by 3-4 orders of magnitude (``joint / sum(proxy) = 1e-4 .. 4.9e-2``) and
non-uniformly across layers (gain spread 33-110x).

This module estimates the gain **empirically** rather than analytically: inject the layer's own error
into the model, run the rest of the network, and take the RMS ratio of the output change to the
injected error. That is a measurement (class 2), not a derivation, and its cost is one extra forward
pass per layer per calibration batch — reported in the diagnostics so the budget is visible.

Shapes:
    ``perturbations[name]``: ``(n_samples, d_residual)`` error injected at that layer's residual
    contribution. ``forward(name, perturbation)`` must return the model-output tensor
    ``(n_samples, d_out)`` with the perturbation applied at ``name`` (and the unperturbed output when
    ``perturbation`` is all zeros).

Numerical limitations:
    The gain is a first-order RMS ratio measured on one calibration batch, so it absorbs the batch's
    activation statistics; it is an estimate (``exact=False`` in the variants that use it), never a
    proof. A gain of 0 is reported as 0 (the layer's error is annihilated downstream), not as an
    error.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import torch

__all__ = ["GainEstimate", "estimate_downstream_gains", "rms"]


def rms(tensor: torch.Tensor) -> float:
    """Root-mean-square of ``tensor`` in float64; ``0.0`` for an empty tensor."""
    if tensor.numel() == 0:
        return 0.0
    t = tensor.detach().to(torch.float64)
    return float(torch.sqrt(torch.mean(t * t)))


@dataclass(frozen=True)
class GainEstimate:
    """Per-layer downstream gains plus the cost/provenance of the estimate.

    Attributes:
        per_layer: ``name -> g`` where ``g = rms(output change) / rms(injected error)``.
        diagnostics: ``name -> {injected_rms, output_rms, forward_calls}`` flattened as
            ``"<name>/<key>"``, so the estimate's inputs are auditable.
        forward_calls: total number of model forwards used (one per layer plus one baseline).
    """

    per_layer: dict[str, float]
    diagnostics: dict[str, float] = field(default_factory=dict)
    forward_calls: int = 0


def estimate_downstream_gains(
    forward: Callable[[str, torch.Tensor], torch.Tensor],
    perturbations: Mapping[str, torch.Tensor],
    *,
    baseline: torch.Tensor | None = None,
) -> GainEstimate:
    """Estimate ``g_l`` for every layer by injecting that layer's error and measuring the output change.

    Args:
        forward: ``forward(name, perturbation) -> (n_samples, d_out)``. Must be deterministic and
            must apply ``perturbation`` at ``name``'s residual contribution. Calling it with a zero
            perturbation yields the baseline output.
        perturbations: ``name -> (n_samples, d_residual)`` measured layer error (e.g. the change in
            the layer's sub-layer output caused by compressing that layer alone).
        baseline: precomputed baseline output; recomputed when ``None``.

    Returns:
        :class:`GainEstimate`. Layers whose injected error has zero RMS get gain ``0.0`` and a
        diagnostic saying so, because the ratio is undefined.

    Cost:
        ``len(perturbations) + 1`` forward passes, recorded in ``forward_calls``.
    """
    if not perturbations:
        return GainEstimate(per_layer={}, diagnostics={}, forward_calls=0)
    if baseline is None:
        first = next(iter(perturbations.values()))
        baseline = forward(next(iter(perturbations)), torch.zeros_like(first))
    calls = 1
    per_layer: dict[str, float] = {}
    diagnostics: dict[str, float] = {}
    for name, perturbation in perturbations.items():
        injected = rms(perturbation)
        if injected == 0.0:
            per_layer[name] = 0.0
            diagnostics[f"{name}/injected_rms"] = 0.0
            diagnostics[f"{name}/output_rms"] = 0.0
            diagnostics[f"{name}/undefined_zero_injection"] = 1.0
            continue
        perturbed = forward(name, perturbation)
        calls += 1
        change = (perturbed - baseline).to(torch.float64)
        output_rms = rms(change)
        per_layer[name] = output_rms / injected
        diagnostics[f"{name}/injected_rms"] = injected
        diagnostics[f"{name}/output_rms"] = output_rms
        diagnostics[f"{name}/undefined_zero_injection"] = 0.0
    return GainEstimate(per_layer=per_layer, diagnostics=diagnostics, forward_calls=calls)
