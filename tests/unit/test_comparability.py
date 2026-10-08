"""Equal-memory comparability gate: what must be accepted, and what must be rejected.

The negative case is the point of the mechanism (AGENTS.md section 4.5): a pair of arms that
differ by more than the predeclared tolerance must be REJECTED, and the rejection must be caused
by the tolerance check -- ``test_byte_gap_beyond_tolerance_is_rejected`` also asserts that the very
same pair is accepted once the tolerance is raised above the gap, so the test cannot pass through
some unrelated error path (deleting the check in
:func:`spectraquant.reporting.comparability.assert_equal_memory` makes it fail).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spectraquant.reporting.comparability import (
    CLASS_OF_SOURCE,
    ComparisonVerdict,
    UnequalMemoryComparison,
    assert_equal_memory,
    compare_manifest_files,
    load_comparable_manifests,
    tolerance_to_bytes,
)
from spectraquant.reporting.manifests import (
    CompressionBlock,
    HardwareBlock,
    ManifestValidationError,
    RunManifest,
    TrainingBlock,
    utc_timestamp,
)

SERIALIZER = "spectraquant.container.int4.v1+rtn(group=128,axis=0)"
OTHER_SERIALIZER = "spectraquant.container.int4.v1+rtn(group=64,axis=0)"


def build_manifest(
    run_id: str,
    *,
    accounted_bytes: int | None = 1_000_000,
    measured_bytes: int | None = 1_001_200,
    tolerance: int | float | None = 200,
    serializer: str | None = SERIALIZER,
    source: str | None = "measured",
    measurement_class: int | None = 3,
    measured_bits_per_param: float | None = 4.0048,
) -> RunManifest:
    """An int4 arm manifest with the byte figures of the equal-memory comparison."""
    return RunManifest(
        run_id=run_id,
        timestamp_utc=utc_timestamp(),
        git_commit="0" * 40,
        git_dirty=False,
        config_path="configs/experiment/smoke.yaml",
        resolved_config={"model": {"name": "tiny-char-transformer"}},
        model_id=None,
        model_revision=None,
        dataset_ids=["synthetic-lcg-v1"],
        dataset_revisions=[None],
        dataset_checksums=["sha256:" + "a" * 64],
        split="test",
        seed=1234,
        hardware=HardwareBlock(platform="test", system="test", release="1", machine="x86_64"),
        software={"python": "3.11.16", "torch": "2.14.1+cpu"},
        compression=CompressionBlock(
            method="rtn",
            bits=4,
            group_size=128,
            accounted_bytes=accounted_bytes,
            measured_bytes=measured_bytes,
            measured_bytes_tolerance=tolerance,
            serializer=serializer,
            nominal_bits_per_param=4.0,
            measured_bits_per_param=measured_bits_per_param,
            bytes_source=source,  # type: ignore[arg-type]  # exercised with invalid values
        ),
        theoretical_bits=4.0,
        packed_bytes=measured_bytes,
        runtime_backend="not-applicable",
        measurement_class=measurement_class,
        training=TrainingBlock(steps=0, tokens=0, wall_time_s=0.0, peak_mem_mb=None),
        metrics={"parameter_count": 2_000_000},
        status="success",
        failure_reason=None,
        log_path=None,
        artifact_paths=[],
    )


def accounted_manifest(run_id: str, *, accounted_bytes: int = 900_000) -> RunManifest:
    """A class-1-only arm: no serialized container exists, only the analytical figure."""
    return build_manifest(
        run_id,
        accounted_bytes=accounted_bytes,
        measured_bytes=None,
        tolerance=0,
        serializer=None,
        source="accounted",
        measurement_class=1,
        measured_bits_per_param=None,
    )


# --------------------------------------------------------------------------------------
# accepted comparisons
# --------------------------------------------------------------------------------------
def test_pair_within_tolerance_is_accepted() -> None:
    run_a = build_manifest("arm-a")
    run_b = build_manifest("arm-b", measured_bytes=1_001_300)

    verdict = assert_equal_memory(run_a, run_b, 200)

    assert isinstance(verdict, ComparisonVerdict)
    assert verdict.equal
    assert (verdict.bytes_a, verdict.bytes_b) == (1_001_200, 1_001_300)
    assert verdict.difference_bytes == 100
    assert verdict.tolerance_bytes == 200
    assert verdict.bytes_source == "measured"
    assert verdict.measurement_class == CLASS_OF_SOURCE["measured"] == 3


def test_pair_at_the_tolerance_boundary_is_accepted() -> None:
    run_a = build_manifest("arm-a")
    run_b = build_manifest("arm-b", measured_bytes=1_001_300)

    verdict = assert_equal_memory(run_a, run_b, 100)

    assert verdict.difference_bytes == verdict.tolerance_bytes == 100
    assert verdict.equal


def test_identical_arms_are_accepted_with_zero_tolerance() -> None:
    verdict = assert_equal_memory(build_manifest("arm-a"), build_manifest("arm-a"), 0)

    assert verdict.difference_bytes == 0
    assert verdict.equal


def test_tolerance_declared_in_the_manifests_is_used_when_no_flag_is_given() -> None:
    verdict = assert_equal_memory(
        build_manifest("arm-a", tolerance=200),
        build_manifest("arm-b", measured_bytes=1_001_300, tolerance=200),
    )

    assert verdict.tolerance == 200
    assert verdict.tolerance_bytes == 200


def test_explicit_tolerance_overrides_the_declared_one() -> None:
    with pytest.raises(UnequalMemoryComparison):
        assert_equal_memory(
            build_manifest("arm-a", tolerance=0),
            build_manifest("arm-b", measured_bytes=1_001_300, tolerance=0),
            None,
        )

    verdict = assert_equal_memory(
        build_manifest("arm-a", tolerance=0),
        build_manifest("arm-b", measured_bytes=1_001_300, tolerance=0),
        500,
    )

    assert verdict.tolerance == 500


def test_fraction_tolerance_is_taken_of_the_larger_total() -> None:
    run_a = build_manifest("arm-a")

    inside = assert_equal_memory(run_a, build_manifest("arm-b", measured_bytes=1_001_300), None)
    assert inside.tolerance == 200

    # 1e-3 of 1 001 300 B = 1 002 B >= 100 B difference
    verdict = assert_equal_memory(
        build_manifest("arm-a", tolerance=0.001),
        build_manifest("arm-b", measured_bytes=1_001_300, tolerance=0.001),
        None,
    )
    assert verdict.tolerance == 0.001
    assert verdict.tolerance_bytes == 1002

    with pytest.raises(UnequalMemoryComparison):
        assert_equal_memory(
            build_manifest("arm-a", measured_bytes=1_000_000, tolerance=0.001),
            build_manifest("arm-b", measured_bytes=1_010_000, tolerance=0.001),
            None,
        )


# --------------------------------------------------------------------------------------
# rejected comparisons
# --------------------------------------------------------------------------------------
def test_byte_gap_beyond_tolerance_is_rejected() -> None:
    run_a = build_manifest("arm-a")  # 1 001 200 B
    run_c = build_manifest("arm-oversized", measured_bytes=1_011_212)  # +1.0 %

    with pytest.raises(UnequalMemoryComparison) as excinfo:
        assert_equal_memory(run_a, run_c, 200)

    error = excinfo.value
    assert error.bytes_a == 1_001_200
    assert error.bytes_b == 1_011_212
    assert error.difference_bytes == 10_012
    assert error.tolerance_bytes == 200
    assert error.bytes_source == "measured"
    message = str(error)
    assert "1001200" in message and "1011212" in message and "10012" in message
    assert "unequal-memory comparison rejected" in message

    # the same pair, same figures, passes as soon as the tolerance covers the gap: the rejection
    # above is caused by the tolerance check and by nothing else.
    verdict = assert_equal_memory(run_a, run_c, 10_012)
    assert verdict.equal
    assert verdict.difference_bytes == 10_012


def test_class_3_comparison_requires_the_same_frozen_serializer() -> None:
    run_a = build_manifest("arm-a")
    run_b = build_manifest("arm-b", serializer=OTHER_SERIALIZER)

    with pytest.raises(UnequalMemoryComparison) as excinfo:
        assert_equal_memory(run_a, run_b, 200)

    assert "serializer" in str(excinfo.value)
    assert excinfo.value.difference_bytes == 0  # the bytes agree, the invocation does not


def test_class_3_comparison_requires_a_serializer_on_both_runs() -> None:
    run_a = build_manifest("arm-a")
    run_b = build_manifest("arm-b", serializer=None)

    with pytest.raises(UnequalMemoryComparison, match="frozen invocation"):
        assert_equal_memory(run_a, run_b, 200)


def test_no_tolerance_anywhere_is_refused() -> None:
    run_a = build_manifest("arm-a", tolerance=None)
    run_b = build_manifest("arm-b", tolerance=None)

    with pytest.raises(UnequalMemoryComparison, match="no tolerance"):
        assert_equal_memory(run_a, run_b)


def test_conflicting_declared_tolerances_are_refused() -> None:
    run_a = build_manifest("arm-a", tolerance=200)
    run_b = build_manifest("arm-b", tolerance=500)

    with pytest.raises(UnequalMemoryComparison, match="different tolerances"):
        assert_equal_memory(run_a, run_b)


# --------------------------------------------------------------------------------------
# a manifest without measured bytes must not silently pass
# --------------------------------------------------------------------------------------
def test_accounted_pair_is_labelled_class_1_not_silently_class_3() -> None:
    verdict = assert_equal_memory(accounted_manifest("acc-a"), accounted_manifest("acc-b"), None)

    assert verdict.bytes_source == "accounted"
    assert verdict.measurement_class == 1
    assert verdict.to_json_dict()["measurement_class"] == 1
    assert "class 1" in verdict.summary()
    assert verdict.bytes_a == verdict.bytes_b == 900_000


def test_accounted_arm_is_never_compared_against_a_measured_arm() -> None:
    run_a = build_manifest("arm-a")  # class 3
    run_b = accounted_manifest("acc-b")  # class 1, measured_bytes is null

    with pytest.raises(UnequalMemoryComparison) as excinfo:
        assert_equal_memory(run_a, run_b, 200)

    message = str(excinfo.value)
    assert "different byte sources" in message
    assert "'measured'" in message and "'accounted'" in message
    assert excinfo.value.bytes_a == 1_001_200
    assert excinfo.value.bytes_b == 900_000


def test_undeclared_byte_source_is_refused() -> None:
    run_a = build_manifest("arm-a", source=None)
    run_b = build_manifest("arm-b")

    with pytest.raises(UnequalMemoryComparison, match="bytes_source is not declared"):
        assert_equal_memory(run_a, run_b, 200)


def test_manifest_without_any_byte_figure_is_refused() -> None:
    run_a = build_manifest("arm-a", accounted_bytes=None, measured_bytes=None)
    run_b = build_manifest("arm-b")

    with pytest.raises(UnequalMemoryComparison, match="measured_bytes is null"):
        assert_equal_memory(run_a, run_b, 200)


# --------------------------------------------------------------------------------------
# tolerance conversion
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("value", "reference", "expected"),
    [(0, 1_000, 0), (200, 1_000, 200), (0.001, 1_001_300, 1002), (0.5, 1001, 501)],
)
def test_tolerance_to_bytes(value: int | float, reference: int, expected: int) -> None:
    assert tolerance_to_bytes(value, reference) == expected


@pytest.mark.parametrize("value", [-1, 1.5, float("nan"), -0.5])
def test_invalid_tolerance_is_refused(value: float) -> None:
    with pytest.raises(ValueError):
        tolerance_to_bytes(value, 1_000)


def test_tolerance_rejects_non_numbers() -> None:
    with pytest.raises(ValueError, match="number"):
        tolerance_to_bytes("200", 1_000)  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# loading from disk
# --------------------------------------------------------------------------------------
def test_committed_fixture_pair_is_accepted_without_a_flag(
    comparability_fixtures_dir: Path,
) -> None:
    verdict = compare_manifest_files(
        comparability_fixtures_dir / "int4-arm-a.manifest.json",
        comparability_fixtures_dir / "int4-arm-b.manifest.json",
    )

    assert verdict.equal
    assert verdict.bytes_a == 1_001_200
    assert verdict.bytes_b == 1_001_300


def test_committed_oversized_fixture_is_rejected(comparability_fixtures_dir: Path) -> None:
    with pytest.raises(UnequalMemoryComparison):
        compare_manifest_files(
            comparability_fixtures_dir / "int4-arm-a.manifest.json",
            comparability_fixtures_dir / "int4-arm-oversized.manifest.json",
        )


def test_load_comparable_manifests_validates_against_the_schema(
    tmp_path: Path, comparability_fixtures_dir: Path
) -> None:
    good = comparability_fixtures_dir / "int4-arm-a.manifest.json"
    broken = tmp_path / "broken.manifest.json"
    document = json.loads(good.read_text(encoding="utf-8"))
    document["compression"]["bytes_source"] = "accounted"  # with measured_bytes still set
    broken.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ManifestValidationError, match="bytes_source"):
        load_comparable_manifests(good, broken)


def test_compare_manifest_files_reports_a_missing_file(
    tmp_path: Path, comparability_fixtures_dir: Path
) -> None:
    good = comparability_fixtures_dir / "int4-arm-a.manifest.json"

    with pytest.raises(ManifestValidationError, match="file not found"):
        compare_manifest_files(tmp_path / "nope.manifest.json", good)


def test_compare_manifest_files_reports_invalid_json(tmp_path: Path) -> None:
    broken = tmp_path / "broken.manifest.json"
    broken.write_text("{not json", encoding="utf-8")

    with pytest.raises(ManifestValidationError, match="invalid JSON"):
        compare_manifest_files(broken, broken)
