"""Config-composition tests for the H3 regularizer arms (Milestone 5).

Each arm config under ``configs/method/regularizer_*.yaml`` must compose through Hydra and validate
through Pydantic, must declare a distinct objective, and must share one compression plan so that the
arms are comparable at equal stored bytes.
"""

from __future__ import annotations

import pytest

from spectraquant.config import load_experiment_config

CONFIG = "configs/experiment/regularizer_tier0.yaml"

EXPECTED_ARMS = {
    "none": (0.0, 0.0, 0.0, 0.0, None),
    "factorization": (1.0, 0.0, 0.0, 0.0, None),
    "rounding": (1.0, 1.0, 0.0, 0.0, None),
    "spectral": (1.0, 0.0, 0.0, 1.0, 4),
    "full": (1.0, 1.0, 0.0, 1.0, 4),
}


@pytest.mark.parametrize("arm", sorted(EXPECTED_ARMS))
def test_arm_config_composes_and_validates(arm: str) -> None:
    cfg = load_experiment_config(CONFIG, [f"method=regularizer_{arm}"])
    regularizer = cfg.method.regularizer
    expected = EXPECTED_ARMS[arm]
    assert regularizer.name == arm
    assert (
        regularizer.lambda_factor,
        regularizer.lambda_round,
        regularizer.lambda_residual,
        regularizer.lambda_spectrum,
        regularizer.tail_rank,
    ) == expected
    assert cfg.method.measurement_class == 2
    assert cfg.method.compression.ranks == [8]
    assert cfg.method.compression.bits == 4
    assert cfg.method.compression.group_size == 16


def test_arms_share_one_compression_plan_and_differ_only_in_the_objective() -> None:
    plans = set()
    objectives = set()
    for arm in EXPECTED_ARMS:
        cfg = load_experiment_config(CONFIG, [f"method=regularizer_{arm}"])
        plans.add(cfg.method.compression.model_dump_json())
        regularizer = cfg.method.regularizer
        objectives.add(
            (
                regularizer.lambda_factor,
                regularizer.lambda_round,
                regularizer.lambda_residual,
                regularizer.lambda_spectrum,
            )
        )
    assert len(plans) == 1, "every arm must declare the identical (equal-bytes) compression plan"
    assert len(objectives) == len(EXPECTED_ARMS), "each arm must be a distinct objective"


def test_regularizer_config_rejects_inconsistent_tail_rank() -> None:
    with pytest.raises(ValueError):
        load_experiment_config(
            CONFIG, ["method=regularizer_full", "method.regularizer.tail_rank=null"]
        )
    with pytest.raises(ValueError):
        load_experiment_config(
            CONFIG, ["method=regularizer_rounding", "method.regularizer.tail_rank=4"]
        )
    with pytest.raises(ValueError):
        load_experiment_config(CONFIG, ["method.regularizer.lambda_round=-1.0"])
    # Hydra refuses an unknown key outright; the same key force-added must reach Pydantic, whose
    # ``extra="forbid"`` rejects it (so a typo can never be silently ignored).
    from hydra.errors import ConfigCompositionException

    with pytest.raises(ConfigCompositionException):
        load_experiment_config(CONFIG, ["method.regularizer.unknown_knob=1"])
    with pytest.raises(ValueError):
        load_experiment_config(CONFIG, ["+method.regularizer.unknown_knob=1"])


def test_default_regularizer_block_is_a_noop() -> None:
    cfg = load_experiment_config("configs/experiment/smoke.yaml")
    assert cfg.method.regularizer.is_noop()
    assert cfg.method.regularizer.name == "none"
