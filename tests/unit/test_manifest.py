"""Run-manifest schema tests: what must be accepted, and what must be rejected."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from spectraquant.reporting.manifests import (
    CompressionBlock,
    HardwareBlock,
    ManifestValidationError,
    RunManifest,
    TrainingBlock,
    load_schema,
    utc_timestamp,
    validate_manifest_dict,
    validate_manifest_file,
    write_manifest,
)

CHECKSUM = "sha256:" + "a" * 64


def valid_manifest() -> dict[str, Any]:
    """A minimal, valid manifest document (uncompressed smoke-shaped run)."""
    manifest = RunManifest(
        run_id="unit-test-20260101T000000Z-000000000000-abcdef",
        timestamp_utc=utc_timestamp(),
        git_commit="0" * 40,
        git_dirty=False,
        config_path="configs/experiment/smoke.yaml",
        resolved_config={"model": {"name": "tiny-char-transformer"}},
        model_id=None,
        model_revision=None,
        dataset_ids=["synthetic-lcg-v1"],
        dataset_revisions=[None],
        dataset_checksums=[CHECKSUM],
        split="train",
        seed=1234,
        hardware=HardwareBlock(platform="test", system="test", release="1", machine="x86_64"),
        software={"python": "3.11.16", "torch": "2.14.1+cpu"},
        compression=CompressionBlock(method="none"),
        theoretical_bits=None,
        packed_bytes=None,
        runtime_backend="torch-cpu-fp32",
        measurement_class=None,
        training=TrainingBlock(steps=50, tokens=25600, wall_time_s=1.5, peak_mem_mb=100.0),
        metrics={"val_loss": 0.1, "loss_sequence": [1.0, 0.5]},
        status="success",
        failure_reason=None,
        log_path=None,
        artifact_paths=["artifacts/sample-results/smoke-manifest.json"],
    )
    return manifest.to_json_dict()


def test_valid_manifest_is_accepted() -> None:
    document = valid_manifest()

    assert validate_manifest_dict(document) == document


def test_compressed_manifest_with_class_is_accepted() -> None:
    document = valid_manifest()
    document["compression"] = {
        "method": "spectraquant",
        "ranks": [8, 16],
        "bits": 4,
        "group_size": 64,
        "exclusions": ["lm_head"],
    }
    document["theoretical_bits"] = 4.5
    document["packed_bytes"] = 123456
    document["measurement_class"] = 3

    validate_manifest_dict(document)


@pytest.mark.parametrize("measurement_class", [1, 2, 3, 4, 5])
def test_measurement_class_in_range_is_accepted(measurement_class: int) -> None:
    document = valid_manifest()
    document["compression"]["method"] = "rtn"
    document["compression"]["bits"] = 4
    document["theoretical_bits"] = 4.0
    document["measurement_class"] = measurement_class

    validate_manifest_dict(document)


@pytest.mark.parametrize("measurement_class", [0, 6, -1, 100])
def test_measurement_class_out_of_bounds_is_rejected(measurement_class: int) -> None:
    document = valid_manifest()
    document["measurement_class"] = measurement_class

    with pytest.raises(ManifestValidationError) as excinfo:
        validate_manifest_dict(document)

    assert any("measurement_class" in message for message in excinfo.value.errors)


@pytest.mark.parametrize("measurement_class", [2.5, "2", True, [2]])
def test_measurement_class_wrong_type_is_rejected(measurement_class: object) -> None:
    document = valid_manifest()
    document["measurement_class"] = measurement_class

    with pytest.raises(ManifestValidationError):
        validate_manifest_dict(document)


def test_null_class_requires_uncompressed_method() -> None:
    document = valid_manifest()
    document["compression"]["method"] = "gptq"
    document["compression"]["bits"] = 4
    document["measurement_class"] = None

    with pytest.raises(ManifestValidationError, match="measurement_class must be 1-5"):
        validate_manifest_dict(document)


def test_missing_required_field_is_rejected() -> None:
    document = valid_manifest()
    del document["run_id"]

    with pytest.raises(ManifestValidationError) as excinfo:
        validate_manifest_dict(document)

    assert any("run_id" in message for message in excinfo.value.errors)


def test_unknown_field_is_rejected() -> None:
    document = valid_manifest()
    document["sneaky_extra"] = 1

    with pytest.raises(ManifestValidationError, match="sneaky_extra"):
        validate_manifest_dict(document)


@pytest.mark.parametrize("bits", [5, 1, 64])
def test_invalid_bit_width_is_rejected(bits: int) -> None:
    document = valid_manifest()
    document["compression"]["bits"] = bits

    with pytest.raises(ManifestValidationError):
        validate_manifest_dict(document)


def test_unknown_status_is_rejected() -> None:
    document = valid_manifest()
    document["status"] = "planned"

    with pytest.raises(ManifestValidationError):
        validate_manifest_dict(document)


def test_failure_reason_must_match_status() -> None:
    document = valid_manifest()
    document["failure_reason"] = "oom"
    with pytest.raises(ManifestValidationError, match="failure_reason must be null"):
        validate_manifest_dict(document)

    failed = valid_manifest()
    failed["status"] = "failed"
    with pytest.raises(ManifestValidationError, match="failure_reason is required"):
        validate_manifest_dict(failed)

    failed["failure_reason"] = "oom"
    validate_manifest_dict(failed)


@pytest.mark.parametrize(
    "timestamp",
    ["2026-01-01T00:00:00+01:00", "not-a-timestamp", "2026-01-01T00:00:00", ""],
)
def test_bad_timestamp_is_rejected(timestamp: str) -> None:
    document = valid_manifest()
    document["timestamp_utc"] = timestamp

    with pytest.raises(ManifestValidationError):
        validate_manifest_dict(document)


@pytest.mark.parametrize("commit", ["abc", "Z" * 40, "0" * 39])
def test_bad_git_commit_is_rejected(commit: str) -> None:
    document = valid_manifest()
    document["git_commit"] = commit

    with pytest.raises(ManifestValidationError):
        validate_manifest_dict(document)


def test_null_git_commit_is_allowed_outside_a_checkout() -> None:
    document = valid_manifest()
    document["git_commit"] = None
    document["git_dirty"] = None

    validate_manifest_dict(document)


def test_bad_dataset_checksum_is_rejected() -> None:
    document = valid_manifest()
    document["dataset_checksums"] = ["deadbeef"]

    with pytest.raises(ManifestValidationError):
        validate_manifest_dict(document)


def test_negative_training_counters_are_rejected() -> None:
    document = valid_manifest()
    document["training"]["tokens"] = -1

    with pytest.raises(ManifestValidationError):
        validate_manifest_dict(document)


def test_unknown_runtime_backend_is_rejected() -> None:
    document = valid_manifest()
    document["runtime_backend"] = "torch-tpu-fp32"

    with pytest.raises(ManifestValidationError):
        validate_manifest_dict(document)


def test_write_and_read_round_trip(tmp_path: Path) -> None:
    document = valid_manifest()
    target = tmp_path / "nested" / "run.manifest.json"

    written = write_manifest(document, target)

    assert written == target
    assert validate_manifest_file(target) == document
    assert json.loads(target.read_text(encoding="utf-8")) == document


def test_write_refuses_invalid_document(tmp_path: Path) -> None:
    document = valid_manifest()
    document["seed"] = -5

    with pytest.raises(ManifestValidationError):
        write_manifest(document, tmp_path / "bad.manifest.json")

    assert not (tmp_path / "bad.manifest.json").exists()


def test_validate_manifest_file_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ManifestValidationError, match="file not found"):
        validate_manifest_file(tmp_path / "nope.json")


def test_validate_manifest_file_reports_invalid_json(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")

    with pytest.raises(ManifestValidationError, match="invalid JSON"):
        validate_manifest_file(broken)


def test_schema_declares_the_documented_fields(schema_path: Path) -> None:
    schema = load_schema(schema_path)

    assert schema["$schema"].endswith("2020-12/schema")
    required = set(schema["required"])
    assert {
        "run_id",
        "timestamp_utc",
        "git_commit",
        "git_dirty",
        "config_path",
        "resolved_config",
        "model_id",
        "model_revision",
        "dataset_ids",
        "dataset_revisions",
        "dataset_checksums",
        "split",
        "seed",
        "hardware",
        "software",
        "compression",
        "theoretical_bits",
        "packed_bytes",
        "runtime_backend",
        "measurement_class",
        "training",
        "metrics",
        "status",
        "failure_reason",
        "log_path",
        "artifact_paths",
    } <= required
    assert set(schema["properties"]["compression"]["required"]) == {
        "method",
        "ranks",
        "bits",
        "group_size",
        "exclusions",
    }


def test_schema_declares_the_equal_memory_byte_fields(schema_path: Path) -> None:
    compression = load_schema(schema_path)["properties"]["compression"]["properties"]

    assert {
        "accounted_bytes",
        "measured_bytes",
        "measured_bytes_tolerance",
        "serializer",
        "nominal_bits_per_param",
        "measured_bits_per_param",
        "bytes_source",
    } <= set(compression)
    # the tolerance is either an absolute byte count, a fraction of the compared totals, or null
    tolerance = compression["measured_bytes_tolerance"]
    assert tolerance["oneOf"][0] == {"type": "integer", "minimum": 0}
    assert tolerance["oneOf"][1]["exclusiveMaximum"] == 1.0
    assert {"type": "null"} in tolerance["oneOf"]
    assert compression["bytes_source"]["enum"] == ["accounted", "measured", None]


def test_compression_byte_fields_round_trip_through_the_model() -> None:
    document = valid_manifest()
    document["compression"] = {
        **document["compression"],
        "method": "rtn",
        "bits": 4,
        "group_size": 128,
        "accounted_bytes": 1_000_000,
        "measured_bytes": 1_001_200,
        "measured_bytes_tolerance": 200,
        "serializer": "spectraquant.container.int4.v1+rtn(group=128,axis=0)",
        "nominal_bits_per_param": 4.0,
        "measured_bits_per_param": 4.0048,
        "bytes_source": "measured",
    }
    document["measurement_class"] = 3
    document["packed_bytes"] = 1_001_200

    validate_manifest_dict(document)

    manifest = RunManifest.model_validate(document)
    assert manifest.compression.accounted_bytes == 1_000_000
    assert manifest.compression.measured_bytes == 1_001_200
    assert manifest.compression.measured_bytes_tolerance == 200
    assert manifest.compression.serializer is not None
    assert manifest.compression.bytes_source == "measured"


def test_accounted_source_may_not_hide_a_measured_figure() -> None:
    document = valid_manifest()
    document["compression"] = {
        **document["compression"],
        "method": "rtn",
        "bits": 4,
        "accounted_bytes": 1_000_000,
        "measured_bytes": 1_001_200,
        "bytes_source": "accounted",
        "serializer": "spectraquant.container.int4.v1+rtn(group=128,axis=0)",
    }
    document["measurement_class"] = 3

    with pytest.raises(ManifestValidationError, match="may not be downgraded"):
        validate_manifest_dict(document)


def test_measured_source_requires_a_figure_and_a_serializer() -> None:
    document = valid_manifest()
    document["compression"] = {
        **document["compression"],
        "method": "rtn",
        "bits": 4,
        "accounted_bytes": 1_000_000,
        "measured_bytes": None,
        "bytes_source": "measured",
        "serializer": "spectraquant.container.int4.v1+rtn(group=128,axis=0)",
    }
    document["measurement_class"] = 3

    with pytest.raises(ManifestValidationError, match="measured_bytes is null"):
        validate_manifest_dict(document)


def test_measured_bytes_without_a_serializer_is_rejected() -> None:
    document = valid_manifest()
    document["compression"] = {
        **document["compression"],
        "method": "rtn",
        "bits": 4,
        "accounted_bytes": 1_000_000,
        "measured_bytes": 1_001_200,
        "bytes_source": "measured",
    }
    document["measurement_class"] = 3

    with pytest.raises(ManifestValidationError, match="serializer"):
        validate_manifest_dict(document)


def test_accounted_source_requires_the_analytical_figure() -> None:
    document = valid_manifest()
    document["compression"] = {
        **document["compression"],
        "method": "rtn",
        "bits": 4,
        "accounted_bytes": None,
        "measured_bytes": None,
        "bytes_source": "accounted",
    }
    document["measurement_class"] = 1

    with pytest.raises(ManifestValidationError, match="accounted_bytes is null"):
        validate_manifest_dict(document)


@pytest.mark.parametrize("tolerance", [1.5, -1, 2.5, "200"])
def test_invalid_tolerance_is_rejected_by_the_schema(tolerance: object) -> None:
    document = valid_manifest()
    document["compression"] = {
        **document["compression"],
        "method": "rtn",
        "bits": 4,
        "accounted_bytes": 1_000_000,
        "measured_bytes": 1_001_200,
        "measured_bytes_tolerance": tolerance,
        "serializer": "spectraquant.container.int4.v1+rtn(group=128,axis=0)",
        "bytes_source": "measured",
    }
    document["measurement_class"] = 3

    with pytest.raises(ManifestValidationError, match="measured_bytes_tolerance"):
        validate_manifest_dict(document)


def test_sample_manifest_still_validates_without_the_new_fields(
    sample_manifest_path: Path,
) -> None:
    """The byte fields are optional: the committed sample manifest predates them."""
    document = validate_manifest_file(sample_manifest_path)

    assert "bytes_source" not in document["compression"]
    assert "accounted_bytes" not in document["compression"]


def test_valid_manifest_is_not_mutated() -> None:
    document = valid_manifest()
    snapshot = copy.deepcopy(document)

    validate_manifest_dict(document)

    assert document == snapshot
