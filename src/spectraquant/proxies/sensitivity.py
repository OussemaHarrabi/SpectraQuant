"""Output-aware sensitivity proxies.

Milestone 3 (owner: proxy stream). The proxy must rank layers by the damage that compressing them
would cause to the *model output*, and that ranking must be validated against exact toy computations
before it is trusted (``AGENTS.md`` section 7, required coverage). Not implemented in the bootstrap
scaffold.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn

__all__ = ["output_aware_sensitivity", "rank_layers", "sensitivity_report"]


def output_aware_sensitivity(
    model: nn.Module,
    inputs: torch.Tensor,
    *,
    targets: torch.Tensor | None = None,
    method: str = "output-aware",
) -> dict[str, float]:
    """Per-layer sensitivity of the model output to perturbing that layer's weights.

    Args:
        model: module under study (CPU, float32).
        inputs: ``(B, T)`` int64 token ids (or ``(B, in_features)`` for linear models).
        targets: optional ``(B, T)`` targets when the proxy uses a task loss rather than output
            distance.
        method: proxy family, e.g. ``"output-aware"``, ``"fisher"``, ``"weight-magnitude"``.

    Returns:
        Mapping ``parameter_name -> sensitivity`` (higher means more damage when compressed).

    Raises:
        NotImplementedError: Milestone 3 has not landed (proxy stream).
    """
    raise NotImplementedError(
        "output_aware_sensitivity is a Milestone 3 deliverable (proxy stream); the bootstrap "
        "scaffold ships no proxy implementation"
    )


def rank_layers(sensitivity: Mapping[str, float], *, descending: bool = True) -> list[str]:
    """Order layer names by sensitivity.

    Raises:
        NotImplementedError: Milestone 3 has not landed (proxy stream).
    """
    raise NotImplementedError("rank_layers is a Milestone 3 deliverable (proxy stream)")


def sensitivity_report(
    sensitivity: Mapping[str, float],
    *,
    reference: Mapping[str, float] | None = None,
) -> Sequence[dict[str, object]]:
    """Tabular report comparing a proxy ranking against a reference ranking.

    Raises:
        NotImplementedError: Milestone 3 has not landed (proxy stream).
    """
    raise NotImplementedError("sensitivity_report is a Milestone 3 deliverable (proxy stream)")
