"""``collect``: the gate between a downloaded bundle and a recorded result (design note §5)."""

from __future__ import annotations

import json
from pathlib import Path

from _cloud_fixtures import (
    ARTIFACT_BYTES,
    ARTIFACT_NAME,
    COMMIT,
    OTHER_COMMIT,
    SHA_B,
    executed_notebook,
    make_bundle,
    make_manifest,
    make_spec,
)

from spectraquant.cloud.collect import ValidationReport, collect, find_manifest
from spectraquant.cloud.registry import REGISTRY_FILENAME, Registry
from spectraquant.cloud.spec import ExpectedArtifact


def _registry(tmp_path: Path) -> Registry:
    return Registry(tmp_path / "runs" / REGISTRY_FILENAME)


def _submitted(tmp_path: Path, spec, registry: Registry | None = None) -> Registry:
    reg = registry if registry is not None else _registry(tmp_path)
    reg.record_submitted(spec, remote_id="fake/run-1", submitted_by="agent", notebook_digest=None)
    return reg


def test_a_complete_bundle_validates(tmp_path: Path) -> None:
    spec = make_spec()
    registry = _submitted(tmp_path, spec)
    bundle = make_bundle(tmp_path / "bundle", spec)

    report = collect(spec.run_id, bundle, registry=registry)

    assert report.ok is True, report.reasons
    assert report.checksum_verified is True
    assert report.manifest is not None
    assert report.manifest["run_id"] == spec.run_id
    assert report.executed_notebook is not None
    assert registry.state(spec.run_id) == "validated"
    assert report.metrics()["val_loss"] == 0.5


def test_validated_entry_records_the_artifact_checksums(tmp_path: Path) -> None:
    spec = make_spec()
    registry = _submitted(tmp_path, spec)
    bundle = make_bundle(tmp_path / "bundle", spec)

    collect(spec.run_id, bundle, registry=registry)

    line = registry.latest(spec.run_id)
    assert line is not None
    assert line["state"] == "validated"
    assert ARTIFACT_NAME in line["artifact_checksums"]
    assert line["manifest_sha256"].startswith("sha256:")
    assert line["checksum_verified"] is True


def test_collect_rejects_a_commit_mismatch(tmp_path: Path) -> None:
    spec = make_spec(git_commit=COMMIT)
    registry = _submitted(tmp_path, spec)
    bundle = make_bundle(
        tmp_path / "bundle", spec, document=make_manifest(spec, git_commit=OTHER_COMMIT)
    )

    report = collect(spec.run_id, bundle, registry=registry)

    assert report.ok is False
    assert any("git commit mismatch" in reason for reason in report.reasons)
    assert registry.state(spec.run_id) == "rejected"


def test_collect_rejects_a_missing_artifact(tmp_path: Path) -> None:
    spec = make_spec()
    registry = _submitted(tmp_path, spec)
    bundle = make_bundle(tmp_path / "bundle", spec, artifact_bytes=None)

    report = collect(spec.run_id, bundle, registry=registry)

    assert report.ok is False
    assert any("expected artifact missing" in reason for reason in report.reasons)


def test_collect_rejects_a_checksum_mismatch(tmp_path: Path) -> None:
    spec = make_spec(
        expected_artifacts=[ExpectedArtifact(name=ARTIFACT_NAME, sha256=SHA_B, min_bytes=4)]
    )
    registry = _submitted(tmp_path, spec)
    bundle = make_bundle(tmp_path / "bundle", spec)

    report = collect(spec.run_id, bundle, registry=registry)

    assert report.ok is False
    assert any("sha256 mismatch" in reason for reason in report.reasons)
    assert registry.state(spec.run_id) == "rejected"


def test_collect_accepts_a_matching_declared_checksum(tmp_path: Path) -> None:
    import hashlib

    digest = "sha256:" + hashlib.sha256(ARTIFACT_BYTES).hexdigest()
    spec = make_spec(
        expected_artifacts=[ExpectedArtifact(name=ARTIFACT_NAME, sha256=digest, min_bytes=4)]
    )
    registry = _submitted(tmp_path, spec)
    bundle = make_bundle(tmp_path / "bundle", spec)

    report = collect(spec.run_id, bundle, registry=registry)

    assert report.ok is True, report.reasons
    assert report.artifact_checksums[ARTIFACT_NAME] == digest


def test_collect_rejects_a_truncated_artifact(tmp_path: Path) -> None:
    spec = make_spec(expected_artifacts=[ExpectedArtifact(name=ARTIFACT_NAME, min_bytes=10_000)])
    registry = _submitted(tmp_path, spec)
    bundle = make_bundle(tmp_path / "bundle", spec)

    report = collect(spec.run_id, bundle, registry=registry)

    assert report.ok is False
    assert any("below min_bytes" in reason for reason in report.reasons)


def test_collect_rejects_a_missing_executed_notebook(tmp_path: Path) -> None:
    spec = make_spec()
    registry = _submitted(tmp_path, spec)
    bundle = make_bundle(tmp_path / "bundle", spec, notebook=False)

    report = collect(spec.run_id, bundle, registry=registry)

    assert report.ok is False
    assert any("executed notebook missing" in reason for reason in report.reasons)


def test_collect_rejects_a_template_notebook_without_execution_counts(tmp_path: Path) -> None:
    spec = make_spec()
    registry = _submitted(tmp_path, spec)
    bundle = make_bundle(tmp_path / "bundle", spec, notebook_executed=False)

    report = collect(spec.run_id, bundle, registry=registry)

    assert report.ok is False
    assert any("executed notebook missing" in reason for reason in report.reasons)


def test_collect_ignores_a_notebook_from_another_run(tmp_path: Path) -> None:
    spec = make_spec()
    registry = _submitted(tmp_path, spec)
    bundle = make_bundle(tmp_path / "bundle", spec, notebook=False)
    (bundle / "notebook").mkdir(parents=True, exist_ok=True)
    (bundle / "notebook" / "other.ipynb").write_text(
        json.dumps(executed_notebook("some-other-run")), encoding="utf-8"
    )

    report = collect(spec.run_id, bundle, registry=registry)

    assert report.ok is False
    assert any("executed notebook missing" in reason for reason in report.reasons)


def test_collect_rejects_a_failed_run(tmp_path: Path) -> None:
    spec = make_spec()
    registry = _submitted(tmp_path, spec)
    bundle = make_bundle(
        tmp_path / "bundle",
        spec,
        document=make_manifest(spec, status="failed", failure_reason="CUDA OOM"),
    )

    report = collect(spec.run_id, bundle, registry=registry)

    assert report.ok is False
    assert any("status='failed'" in reason for reason in report.reasons)
    assert registry.state(spec.run_id) == "rejected"


def test_collect_rejects_an_unknown_run(tmp_path: Path) -> None:
    spec = make_spec()
    registry = _registry(tmp_path)
    bundle = make_bundle(tmp_path / "bundle", spec)

    report = collect(spec.run_id, bundle, registry=registry)

    assert report.ok is False
    assert any("no RunSpec is recorded" in reason for reason in report.reasons)


def test_collect_rejects_a_missing_bundle(tmp_path: Path) -> None:
    spec = make_spec()
    registry = _submitted(tmp_path, spec)

    report = collect(spec.run_id, tmp_path / "does-not-exist", registry=registry)

    assert report.ok is False
    assert any("no run_manifest.json" in reason for reason in report.reasons)
    assert registry.state(spec.run_id) == "rejected"


def test_collect_rejects_an_invalid_manifest(tmp_path: Path) -> None:
    spec = make_spec()
    registry = _submitted(tmp_path, spec)
    bundle = make_bundle(tmp_path / "bundle", spec)
    (bundle / "run_manifest.json").write_text('{"run_id": "x"}', encoding="utf-8")

    report = collect(spec.run_id, bundle, registry=registry)

    assert report.ok is False
    assert any("not a valid run manifest" in reason for reason in report.reasons)


def test_collect_rejects_a_manifest_for_another_run(tmp_path: Path) -> None:
    spec = make_spec()
    registry = _submitted(tmp_path, spec)
    bundle = make_bundle(
        tmp_path / "bundle", spec, document=make_manifest(spec, run_id="some-other-run")
    )

    report = collect(spec.run_id, bundle, registry=registry)

    assert report.ok is False
    assert any("does not match the collected run" in reason for reason in report.reasons)


def test_collect_rejects_a_spec_without_expected_artifacts(tmp_path: Path) -> None:
    spec = make_spec(expected_artifacts=[])
    registry = _submitted(tmp_path, spec)
    bundle = make_bundle(tmp_path / "bundle", spec)

    report = collect(spec.run_id, bundle, registry=registry)

    assert report.ok is False
    assert any("no expected_artifacts" in reason for reason in report.reasons)


def test_collect_accepts_an_explicit_spec_without_a_registry_entry(tmp_path: Path) -> None:
    spec = make_spec()
    bundle = make_bundle(tmp_path / "bundle", spec)
    registry = _registry(tmp_path)

    report = collect(spec.run_id, bundle, spec=spec, registry=registry)

    assert report.ok is True, report.reasons


def test_collect_can_validate_without_recording(tmp_path: Path) -> None:
    spec = make_spec()
    registry = _submitted(tmp_path, spec)
    bundle = make_bundle(tmp_path / "bundle", spec)

    report = collect(spec.run_id, bundle, registry=registry, record=False)

    assert report.ok is True
    assert registry.state(spec.run_id) == "submitted"


def test_collect_records_collected_then_validated(tmp_path: Path) -> None:
    spec = make_spec()
    registry = _submitted(tmp_path, spec)
    bundle = make_bundle(tmp_path / "bundle", spec)

    collect(spec.run_id, bundle, registry=registry)

    states = [record["state"] for record in registry.transitions(spec.run_id)]
    assert states == ["submitted", "collected", "validated"]


def test_collect_reads_the_result_line_and_teardown_marker(tmp_path: Path) -> None:
    spec = make_spec()
    registry = _submitted(tmp_path, spec)
    bundle = make_bundle(tmp_path / "bundle", spec)

    report = collect(spec.run_id, bundle, registry=registry)

    assert report.result_line == {"run_id": spec.run_id, "status": "success"}
    assert report.teardown_observed is True


def test_collect_detects_a_truncated_run(tmp_path: Path) -> None:
    spec = make_spec()
    registry = _submitted(tmp_path, spec)
    bundle = make_bundle(tmp_path / "bundle", spec, teardown=False)

    report = collect(spec.run_id, bundle, registry=registry)

    assert report.ok is True  # the teardown marker is informational for collection
    assert report.teardown_observed is False


def test_collect_accepts_a_dirty_manifest_when_the_spec_allows_it(tmp_path: Path) -> None:
    spec = make_spec(git_commit="", allow_dirty=True)
    registry = _submitted(tmp_path, spec)
    document = make_manifest(spec, git_commit=COMMIT)
    document["git_dirty"] = True
    bundle = make_bundle(tmp_path / "bundle", spec, document=document)

    report = collect(spec.run_id, bundle, registry=registry)

    assert report.ok is True, report.reasons


def test_report_summary_and_json_are_serializable(tmp_path: Path) -> None:
    spec = make_spec()
    registry = _submitted(tmp_path, spec)
    bundle = make_bundle(tmp_path / "bundle", spec)

    report = collect(spec.run_id, bundle, registry=registry)

    assert "validated" in report.summary()
    assert json.loads(json.dumps(report.to_json()))["state"] == "validated"


def test_find_manifest_searches_one_level_down(tmp_path: Path) -> None:
    spec = make_spec()
    bundle = make_bundle(tmp_path / "bundle" / "nested", spec)

    assert find_manifest(bundle.parent) is not None
    assert find_manifest(tmp_path / "nothing") is None


def test_a_rejected_report_is_a_validation_report() -> None:
    report = ValidationReport(run_id="r", state="rejected", source_dir=".", reasons=["x"])

    assert report.ok is False
    assert report.metrics() == {}
