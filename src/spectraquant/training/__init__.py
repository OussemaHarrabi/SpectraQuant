"""Training package.

Implemented: deterministic seeding (:mod:`spectraquant.training.seeding`), the tiny smoke-fixture
transformer (:mod:`spectraquant.training.tiny_lm`), the end-to-end smoke experiment
(:mod:`spectraquant.training.smoke`), the factorized linear carrier of low-rank preparation
(:mod:`spectraquant.training.low_rank`) and the general preparation loop
(:mod:`spectraquant.training.loop`) with checkpoint/resume and per-term regularizer diagnostics.

Tier-1+ training (WikiText-2, pretrained LMs, QAT/LoRA arms) is **not** started locally: it runs on
the cloud notebook substrate (``AGENTS.md`` sections 2.3 and 2b). ``evaluate_language_model`` stays a
loud ``NotImplementedError`` until the Milestone-6 corpus loaders exist.
"""

from __future__ import annotations

from spectraquant.training.loop import (
    CHECKPOINT_FORMAT,
    LoopConfig,
    SequenceData,
    TrainingResult,
    evaluate_language_model,
    load_checkpoint,
    save_checkpoint,
    train_language_model,
)
from spectraquant.training.low_rank import LowRankLinear, factorize_linears
from spectraquant.training.seeding import SeedState, seed_everything, seed_state
from spectraquant.training.smoke import SmokeResult, run_smoke_experiment
from spectraquant.training.tiny_lm import TinyCharTransformer, count_parameters

__all__ = [
    "CHECKPOINT_FORMAT",
    "LoopConfig",
    "LowRankLinear",
    "SeedState",
    "SequenceData",
    "SmokeResult",
    "TinyCharTransformer",
    "TrainingResult",
    "count_parameters",
    "evaluate_language_model",
    "factorize_linears",
    "load_checkpoint",
    "run_smoke_experiment",
    "save_checkpoint",
    "seed_everything",
    "seed_state",
    "train_language_model",
]
