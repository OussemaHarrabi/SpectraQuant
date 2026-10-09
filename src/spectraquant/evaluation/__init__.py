"""Evaluation package.

Milestone 2 ships the Tier-0 toy fixtures and exact ground truth here
(:mod:`spectraquant.evaluation.toy`); downstream quality evaluation and LM Evaluation Harness
integration for Tier-2+ models remain to be implemented in :mod:`spectraquant.evaluation.harness`
and raise ``NotImplementedError``.
"""

from __future__ import annotations

from spectraquant.evaluation.harness import (
    detect_train_eval_overlap,
    evaluate_checkpoint,
    run_lm_eval_harness,
)
from spectraquant.evaluation.toy import (
    TinyConfig,
    TinyTransformer,
    exact_layer_output_error,
    layer_inputs,
    layer_names,
    network_joint_damage,
    network_single_layer_damage,
    norm_preimages,
    residual_contribution_key,
    train_tiny,
)

__all__ = [
    "TinyConfig",
    "TinyTransformer",
    "detect_train_eval_overlap",
    "evaluate_checkpoint",
    "exact_layer_output_error",
    "layer_inputs",
    "layer_names",
    "network_joint_damage",
    "network_single_layer_damage",
    "norm_preimages",
    "residual_contribution_key",
    "run_lm_eval_harness",
    "train_tiny",
]
