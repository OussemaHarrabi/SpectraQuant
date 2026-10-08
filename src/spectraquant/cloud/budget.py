"""Submission budget guard (``design-cloud-adapter.md`` §6, ``AGENTS.md`` §2b rule 8).

Free tiers are the default. A paid platform is refused unless the user has explicitly authorized it
(``SPECTRAQUANT_ALLOW_PAID=1``) **and** the spec carries a spend ceiling no larger than the
platform's documented ceiling. The estimated cost is always logged, and it is labelled an
*estimate*: it is class-1 arithmetic over the declared timeout and a documented hourly rate, never a
measured bill.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any

from spectraquant.cloud.spec import PAID_PLATFORMS, RunSpec

__all__ = [
    "FREE_PLATFORMS",
    "PAID_PLATFORM_CEILINGS_USD",
    "PLATFORM_RATES_USD_PER_GPU_HOUR",
    "BudgetError",
    "CostCeilingExceeded",
    "PaidSubmissionNotAuthorized",
    "assert_submission_allowed",
    "budget_summary",
    "estimated_cost_usd",
    "estimated_gpu_hours",
]

logger = logging.getLogger("spectraquant.cloud.budget")

#: Platforms that cost nothing (documented free tiers).
FREE_PLATFORMS: frozenset[str] = frozenset({"colab", "kaggle", "local_cpu"})

#: Documented hourly rate per GPU-hour used for the *estimate* (USD). Colab Enterprise A100-class
#: on-demand pricing; the run book states the source and the date.
PLATFORM_RATES_USD_PER_GPU_HOUR: dict[str, float] = {"colab_enterprise": 3.67}

#: Hard ceiling on the spend a single spec may authorize (USD), per platform.
PAID_PLATFORM_CEILINGS_USD: dict[str, float] = {"colab_enterprise": 50.0}

#: Environment variable that authorizes paid resources.
ALLOW_PAID_ENV = "SPECTRAQUANT_ALLOW_PAID"


class BudgetError(ValueError):
    """Raised when a submission would start an unauthorized or over-budget resource."""


class PaidSubmissionNotAuthorized(BudgetError):
    """Raised when a paid platform is requested without explicit user authorization."""


class CostCeilingExceeded(BudgetError):
    """Raised when the authorized ceiling exceeds the platform's documented maximum."""


def estimated_gpu_hours(spec: RunSpec) -> float:
    """Return the *estimated* GPU-hours of the run (class-1 estimate from the declared timeout).

    The estimate assumes the whole ``timeout_minutes`` budget is consumed on a GPU; it is an upper
    bound, labelled approximate, and never presented as a measured duration.
    """
    if not spec.gpu_required:
        return 0.0
    return round(spec.timeout_minutes / 60.0, 4)


def estimated_cost_usd(spec: RunSpec) -> float:
    """Return the *estimated* cost in USD (estimated GPU-hours × the platform's documented rate)."""
    rate = PLATFORM_RATES_USD_PER_GPU_HOUR.get(spec.platform, 0.0)
    return round(estimated_gpu_hours(spec) * rate, 4)


def budget_summary(spec: RunSpec, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Return the logged budget block (all monetary values are estimates)."""
    source = os.environ if env is None else env
    return {
        "run_id": spec.run_id,
        "platform": spec.platform,
        "free_tier": spec.platform in FREE_PLATFORMS,
        "paid_authorized": source.get(ALLOW_PAID_ENV) == "1",
        "estimated_gpu_hours": estimated_gpu_hours(spec),
        "estimated_cost_usd": estimated_cost_usd(spec),
        "max_cost_authorized_usd": spec.max_cost_authorized_usd,
        "estimate_label": "analytical_estimate",
    }


def assert_submission_allowed(spec: RunSpec, env: Mapping[str, str] | None = None) -> None:
    """Refuse a submission that would start an unauthorized or over-budget paid resource.

    Free-tier platforms always pass (their estimated GPU-hours are still logged).

    Args:
        spec: the frozen spec about to be submitted.
        env: environment mapping (defaults to ``os.environ``); ``SPECTRAQUANT_ALLOW_PAID`` must be
            exactly ``"1"`` for a paid platform.

    Raises:
        PaidSubmissionNotAuthorized: a paid platform without ``SPECTRAQUANT_ALLOW_PAID=1``.
        CostCeilingExceeded: no authorized ceiling, a ceiling above the documented maximum, or an
            estimated cost above the authorized ceiling.
    """
    source = os.environ if env is None else env
    summary = budget_summary(spec, source)

    if spec.platform not in PAID_PLATFORMS:
        logger.info(
            "budget: free tier platform=%s estimated_gpu_hours=%s (analytical estimate)",
            spec.platform,
            summary["estimated_gpu_hours"],
        )
        return

    if source.get(ALLOW_PAID_ENV) != "1":
        raise PaidSubmissionNotAuthorized(
            f"platform {spec.platform!r} is paid and {ALLOW_PAID_ENV} != '1': no paid cloud resource "
            "may be started without the user's prior authorization (AGENTS.md §2b rule 8)"
        )

    if spec.max_cost_authorized_usd is None:
        raise CostCeilingExceeded(
            f"platform {spec.platform!r} requires max_cost_authorized_usd before submission"
        )

    ceiling = PAID_PLATFORM_CEILINGS_USD.get(spec.platform, 0.0)
    if spec.max_cost_authorized_usd > ceiling:
        raise CostCeilingExceeded(
            f"max_cost_authorized_usd={spec.max_cost_authorized_usd} exceeds the documented "
            f"{spec.platform} ceiling of {ceiling} USD; reduce the run budget or amend the ceiling "
            "deliberately"
        )

    estimate = estimated_cost_usd(spec)
    if estimate > spec.max_cost_authorized_usd:
        raise CostCeilingExceeded(
            f"estimated cost {estimate} USD (analytical estimate: {summary['estimated_gpu_hours']} "
            f"GPU-hours × {PLATFORM_RATES_USD_PER_GPU_HOUR.get(spec.platform)} USD/h) exceeds the "
            f"authorized {spec.max_cost_authorized_usd} USD"
        )

    logger.warning(
        "budget: PAID submission authorized platform=%s estimated_gpu_hours=%s "
        "estimated_cost_usd=%s (analytical estimate) authorized_usd=%s",
        spec.platform,
        summary["estimated_gpu_hours"],
        estimate,
        spec.max_cost_authorized_usd,
    )
