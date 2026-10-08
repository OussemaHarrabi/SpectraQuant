"""Regression guard for the committed equal-memory fixtures.

``artifacts/sample-results/comparability/*.manifest.json`` are hand-authored, schema-valid
manifests that the CI gate runs ``spectraquant compare-manifests`` on. They carry synthetic byte
figures (``resolved_config.fixture.not_a_real_run``) and no result may be cited from them; what
this test guards is that the gate still *accepts* the within-tolerance pair and still *rejects*
the oversized one -- i.e. that the fixtures and the gate did not drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spectraquant.reporting.comparability import (
    UnequalMemoryComparison,
    compare_manifest_files,
)

pytestmark = pytest.mark.regression

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "sample-results" / "comparability"
ARM_A = FIXTURE_DIR / "int4-arm-a.manifest.json"
ARM_B = FIXTURE_DIR / "int4-arm-b.manifest.json"
ARM_OVERSIZED = FIXTURE_DIR / "int4-arm-oversized.manifest.json"
ACCOUNTED_A = FIXTURE_DIR / "accounted-arm-a.manifest.json"
ACCOUNTED_B = FIXTURE_DIR / "accounted-arm-b.manifest.json"


@pytest.mark.parametrize("path", [ARM_A, ARM_B, ARM_OVERSIZED, ACCOUNTED_A, ACCOUNTED_B])
def test_fixture_is_committed_and_labelled_synthetic(path: Path) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))

    assert document["resolved_config"]["fixture"]["not_a_real_run"] is True
    assert document["compression"]["bytes_source"] in {"accounted", "measured"}


def test_within_tolerance_pair_is_accepted() -> None:
    verdict = compare_manifest_files(ARM_A, ARM_B)

    assert verdict.equal
    assert verdict.bytes_source == "measured"
    assert verdict.measurement_class == 3
    assert verdict.difference_bytes == 100


def test_oversized_arm_is_rejected_at_a_zero_tolerance() -> None:
    with pytest.raises(UnequalMemoryComparison) as excinfo:
        compare_manifest_files(ARM_A, ARM_OVERSIZED, 0)

    assert excinfo.value.difference_bytes == 10_012
    assert excinfo.value.bytes_a == 1_001_200
    assert excinfo.value.bytes_b == 1_011_212


def test_oversized_arm_is_rejected_against_its_own_declared_tolerance() -> None:
    with pytest.raises(UnequalMemoryComparison):
        compare_manifest_files(ARM_A, ARM_OVERSIZED)


def test_accounted_only_pair_is_labelled_class_1() -> None:
    verdict = compare_manifest_files(ACCOUNTED_A, ACCOUNTED_B)

    assert verdict.bytes_source == "accounted"
    assert verdict.measurement_class == 1


def test_measured_and_accounted_arms_are_not_comparable() -> None:
    with pytest.raises(UnequalMemoryComparison, match="different byte sources"):
        compare_manifest_files(ARM_A, ACCOUNTED_A)
