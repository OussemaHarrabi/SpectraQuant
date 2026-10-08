"""Training package.

Implemented: deterministic seeding (:mod:`spectraquant.training.seeding`), the tiny smoke-fixture
transformer (:mod:`spectraquant.training.tiny_lm`) and the end-to-end smoke experiment
(:mod:`spectraquant.training.smoke`).

Not implemented: the general joint low-rank/low-bit training loop and Tier-2+ evaluation
(:mod:`spectraquant.training.loop`), which are GPU-gated.
"""

from __future__ import annotations

from spectraquant.training.loop import evaluate_language_model, train_language_model
from spectraquant.training.seeding import SeedState, seed_everything, seed_state
from spectraquant.training.smoke import SmokeResult, run_smoke_experiment
from spectraquant.training.tiny_lm import TinyCharTransformer, count_parameters

__all__ = [
    "SeedState",
    "SmokeResult",
    "TinyCharTransformer",
    "count_parameters",
    "evaluate_language_model",
    "run_smoke_experiment",
    "seed_everything",
    "seed_state",
    "train_language_model",
]
