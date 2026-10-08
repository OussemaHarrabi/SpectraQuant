"""General training loop for joint low-rank / low-bit training.

Milestone 5 (owner: training stream). The local CI smoke fixture lives in
:mod:`spectraquant.training.smoke` and is fully implemented; the general loop below is the Tier-2+
entry point and is deliberately unimplemented, because every run it would launch needs an external
GPU that is not yet secured (ADR-0002).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from torch import nn

__all__ = ["evaluate_language_model", "train_language_model"]


def train_language_model(
    model: nn.Module,
    *,
    method: str,
    steps: int,
    output_dir: Path | str,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train a model with the selected joint compression method (Tier 2+).

    Raises:
        NotImplementedError: Milestone 5 has not landed, and every Tier-2+ run requires an
            external GPU that is not available on this workstation.
    """
    raise NotImplementedError(
        "train_language_model is a Milestone 5 deliverable and requires a GPU for Tier-2+ runs; "
        "see ADR-0002. Locally executable training lives in spectraquant.training.smoke"
    )


def evaluate_language_model(
    model: nn.Module,
    *,
    dataset_id: str,
    split: str = "validation",
) -> dict[str, float]:
    """Evaluate a model on a language-modelling split (Tier 2+).

    Raises:
        NotImplementedError: Milestone 6 has not landed (evaluation stream).
    """
    raise NotImplementedError(
        "evaluate_language_model is a Milestone 6 deliverable (evaluation stream)"
    )
