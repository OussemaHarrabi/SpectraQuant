"""Evaluation package (Milestone 6 — not implemented).

Planned responsibilities: downstream quality evaluation of compressed checkpoints, LM Evaluation
Harness integration for Tier-2+ models, and train/eval overlap detection.

Every entry point raises ``NotImplementedError``; see :mod:`spectraquant.evaluation.harness`.
"""

from __future__ import annotations

from spectraquant.evaluation.harness import (
    detect_train_eval_overlap,
    evaluate_checkpoint,
    run_lm_eval_harness,
)

__all__ = ["detect_train_eval_overlap", "evaluate_checkpoint", "run_lm_eval_harness"]
