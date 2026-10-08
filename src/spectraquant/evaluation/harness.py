"""Evaluation harness API.

Milestone 6 (owner: evaluation stream). Planned responsibilities: LM Evaluation Harness integration
for the Tier-2+ models, train/eval overlap (contamination) detection, and downstream quality metrics
recorded as measurement class 2 (fake quantization) or class 4 (kernel-backed) — never mixed.

Not implemented in the bootstrap scaffold; also note that no evaluation question set may be trained
on (``AGENTS.md`` section 4.7).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from torch import nn

__all__ = ["detect_train_eval_overlap", "evaluate_checkpoint", "run_lm_eval_harness"]


def evaluate_checkpoint(
    checkpoint: str,
    *,
    tasks: Sequence[str],
    limit: int | None = None,
    backend: str = "torch-cpu-fp32",
) -> Mapping[str, Any]:
    """Evaluate a compressed checkpoint and return per-task metrics.

    Raises:
        NotImplementedError: Milestone 6 has not landed (evaluation stream).
    """
    raise NotImplementedError(
        "evaluate_checkpoint is a Milestone 6 deliverable (evaluation stream); the bootstrap "
        "scaffold ships no evaluation implementation"
    )


def run_lm_eval_harness(
    model: nn.Module,
    *,
    tasks: Sequence[str],
    harness_config: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Thin wrapper over EleutherAI ``lm-evaluation-harness`` (Tier 2+, GPU-gated).

    Raises:
        NotImplementedError: Milestone 6 has not landed (evaluation stream).
    """
    raise NotImplementedError("run_lm_eval_harness is a Milestone 6 deliverable")


def detect_train_eval_overlap(
    train_texts: Sequence[str],
    eval_texts: Sequence[str],
    *,
    ngram: int = 13,
) -> Mapping[str, Any]:
    """Detect n-gram overlap between training data and evaluation sets.

    Raises:
        NotImplementedError: Milestone 6 has not landed (evaluation stream).
    """
    raise NotImplementedError("detect_train_eval_overlap is a Milestone 6 deliverable")
