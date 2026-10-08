"""Budget guard: free tier passes, paid requires explicit authorization (design note §6)."""

from __future__ import annotations

import logging

import pytest
from _cloud_fixtures import make_spec

from spectraquant.cloud.budget import (
    FREE_PLATFORMS,
    PAID_PLATFORM_CEILINGS_USD,
    PLATFORM_RATES_USD_PER_GPU_HOUR,
    CostCeilingExceeded,
    PaidSubmissionNotAuthorized,
    assert_submission_allowed,
    budget_summary,
    estimated_cost_usd,
    estimated_gpu_hours,
)


@pytest.mark.parametrize("platform", sorted(FREE_PLATFORMS))
def test_free_tiers_always_pass(platform: str) -> None:
    spec = make_spec(platform=platform, gpu_required=platform != "local_cpu")

    assert_submission_allowed(spec, {})  # must not raise


def test_paid_platform_without_the_flag_raises() -> None:
    spec = make_spec(
        platform="colab_enterprise",
        gpu_required=True,
        timeout_minutes=60,
        max_cost_authorized_usd=5.0,
    )

    with pytest.raises(PaidSubmissionNotAuthorized, match="SPECTRAQUANT_ALLOW_PAID"):
        assert_submission_allowed(spec, {})


def test_paid_platform_with_the_flag_passes() -> None:
    spec = make_spec(
        platform="colab_enterprise",
        gpu_required=True,
        timeout_minutes=60,
        max_cost_authorized_usd=5.0,
    )

    assert_submission_allowed(spec, {"SPECTRAQUANT_ALLOW_PAID": "1"})  # must not raise


def test_paid_platform_over_the_documented_ceiling_raises() -> None:
    ceiling = PAID_PLATFORM_CEILINGS_USD["colab_enterprise"]
    spec = make_spec(
        platform="colab_enterprise",
        gpu_required=True,
        timeout_minutes=60,
        max_cost_authorized_usd=ceiling + 1.0,
    )

    with pytest.raises(CostCeilingExceeded, match="exceeds the documented"):
        assert_submission_allowed(spec, {"SPECTRAQUANT_ALLOW_PAID": "1"})


def test_estimate_above_the_authorized_ceiling_raises() -> None:
    rate = PLATFORM_RATES_USD_PER_GPU_HOUR["colab_enterprise"]
    assert rate > 0
    spec = make_spec(
        platform="colab_enterprise",
        gpu_required=True,
        timeout_minutes=600,  # 10 h * rate >> 1 USD
        max_cost_authorized_usd=1.0,
    )

    with pytest.raises(CostCeilingExceeded, match="estimated cost"):
        assert_submission_allowed(spec, {"SPECTRAQUANT_ALLOW_PAID": "1"})


def test_the_flag_must_be_exactly_one() -> None:
    spec = make_spec(
        platform="colab_enterprise",
        gpu_required=True,
        timeout_minutes=60,
        max_cost_authorized_usd=1.0,
    )

    with pytest.raises(PaidSubmissionNotAuthorized):
        assert_submission_allowed(spec, {"SPECTRAQUANT_ALLOW_PAID": "true"})


def test_gpu_hours_estimate_is_zero_without_a_gpu() -> None:
    spec = make_spec(platform="local_cpu", gpu_required=False, timeout_minutes=120)

    assert estimated_gpu_hours(spec) == 0.0
    assert estimated_cost_usd(spec) == 0.0


def test_gpu_hours_estimate_is_labelled_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    spec = make_spec(platform="colab", gpu_required=True, timeout_minutes=120)

    with caplog.at_level(logging.INFO, logger="spectraquant.cloud.budget"):
        assert_submission_allowed(spec, {})

    summary = budget_summary(spec, {})
    assert summary["estimate_label"] == "analytical_estimate"
    assert summary["estimated_gpu_hours"] == 2.0
    assert summary["free_tier"] is True
    assert "estimated_gpu_hours=2.0" in caplog.text


def test_paid_authorization_is_logged_at_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    spec = make_spec(
        platform="colab_enterprise",
        gpu_required=True,
        timeout_minutes=60,
        max_cost_authorized_usd=5.0,
    )

    with caplog.at_level(logging.WARNING, logger="spectraquant.cloud.budget"):
        assert_submission_allowed(spec, {"SPECTRAQUANT_ALLOW_PAID": "1"})

    assert "PAID submission authorized" in caplog.text
