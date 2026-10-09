"""Registry: append-only transitions, and ``validated`` is unreachable without verification."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from _cloud_fixtures import SHA_A, make_spec

from spectraquant.cloud.collect import ValidationReport
from spectraquant.cloud.registry import (
    ALLOWED_TRANSITIONS,
    REGISTRY_FILENAME,
    STATES,
    Registry,
    RegistryError,
    default_registry_path,
    runs_dir,
)


def _registry(tmp_path: Path) -> Registry:
    return Registry(tmp_path / "runs" / REGISTRY_FILENAME)


def _report(run_id: str, state: str = "validated", **overrides: object) -> ValidationReport:
    values: dict[str, object] = {
        "run_id": run_id,
        "state": state,
        "source_dir": "/tmp/bundle",
        "manifest": {"metrics": {"val_loss": 0.25}, "status": "success"},
        "manifest_path": "/tmp/bundle/run_manifest.json",
        "manifest_sha256": SHA_A,
        "artifact_checksums": {"artifacts/result.json": SHA_A},
        "checksum_verified": state == "validated",
        "platform": "kaggle",
        "remote_id": "someone/run-1",
        "git_commit": "0" * 40,
    }
    values.update(overrides)
    return ValidationReport(**values)  # type: ignore[arg-type]


def test_a_new_notebook_for_the_same_remote_id_is_recorded_as_resubmitted() -> None:
    """Same kernel, different notebook = a NEW submission, not an idempotent repeat.

    Regression for the defect the first real Kaggle submission exposed: the idempotency check
    compared only the remote id, so a genuine re-push (new commit, new notebook digest) was dropped
    and the registry kept pointing at the superseded attempt.
    """
    registry = Registry(Path(tempfile.mkdtemp()) / "registry.jsonl")
    spec = make_spec(platform="kaggle", gpu_required=True)
    first = registry.record_submitted(
        spec, remote_id="owner/run", submitted_by="agent", notebook_digest="sha256:aaa"
    )
    repeat = registry.record_submitted(
        spec, remote_id="owner/run", submitted_by="agent", notebook_digest="sha256:aaa"
    )
    assert repeat.timestamp_utc == first.timestamp_utc  # a true repeat is a no-op
    assert len(registry.transitions(spec.run_id)) == 1

    resubmitted = registry.record_submitted(
        spec, remote_id="owner/run", submitted_by="agent", notebook_digest="sha256:bbb"
    )
    assert resubmitted.state == "resubmitted"
    assert registry.state(spec.run_id) == "resubmitted"
    assert len(registry.transitions(spec.run_id)) == 2


def test_states_and_transitions_are_documented() -> None:
    assert STATES == (
        "submitted",
        "resubmitted",
        "running",
        "finished",
        "failed",
        "collected",
        "validated",
        "rejected",
    )
    assert ALLOWED_TRANSITIONS["collected"] == frozenset({"validated", "rejected"})
    # A validated run may be re-run on the same kernel: the record is append-only, so the previous
    # attempt's validation stays visible and the next collection re-validates the new bundle.
    assert ALLOWED_TRANSITIONS["validated"] == frozenset({"resubmitted"})


def test_default_path_is_under_artifacts_runs() -> None:
    assert default_registry_path().name == REGISTRY_FILENAME
    assert runs_dir().name == "runs"


def test_records_append_one_json_line_per_transition(tmp_path: Path) -> None:
    registry = _registry(tmp_path)

    registry.record("submitted", "run-1", platform="kaggle", remote_id="someone/run-1")
    registry.record("running", "run-1", platform="kaggle")
    registry.record("finished", "run-1", platform="kaggle")

    lines = registry.path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    assert all(json.loads(line)["run_id"] == "run-1" for line in lines)
    assert registry.state("run-1") == "finished"


def test_unknown_state_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(RegistryError, match="unknown state"):
        _registry(tmp_path).record("launched", "run-1")


def test_illegal_transition_is_rejected(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.record("submitted", "run-1")
    registry.record("rejected", "run-1", failure_reason="bad bundle")

    with pytest.raises(RegistryError, match="illegal transition"):
        registry.record("running", "run-1")


def test_validated_cannot_be_recorded_through_the_generic_api(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.record("submitted", "run-1")
    registry.record_collected("run-1")

    with pytest.raises(RegistryError, match="cannot be recorded through record"):
        registry.record(
            "validated",
            "run-1",
            artifact_checksums={"a": SHA_A},
            manifest_sha256=SHA_A,
            checksum_verified=True,
        )


def test_validated_is_unreachable_without_a_checksum_verified_report(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.record("submitted", "run-1")
    registry.record_collected("run-1")

    with pytest.raises(RegistryError, match="checksum_verified=True"):
        registry.record_validated(_report("run-1", checksum_verified=False))

    assert registry.state("run-1") == "collected"


def test_validated_is_unreachable_without_artifact_checksums(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.record("submitted", "run-1")
    registry.record_collected("run-1")

    with pytest.raises(RegistryError, match="artifact checksum set is empty"):
        registry.append(_transition_with(run_id="run-1", state="validated", checksum_verified=True))


def test_validated_requires_a_manifest_digest(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.record("submitted", "run-1")
    registry.record_collected("run-1")

    with pytest.raises(RegistryError, match=r"run_manifest\.json digest"):
        registry.record_validated(_report("run-1", manifest_sha256=None))


def _transition_with(**fields: object):  # type: ignore[no-untyped-def]
    from spectraquant.cloud.registry import Transition

    return Transition(**fields)  # type: ignore[arg-type]


def test_a_rejected_report_never_exposes_metrics(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.record("submitted", "run-1")
    registry.record_collected("run-1")

    transition = registry.record_rejected(
        _report("run-1", state="rejected", checksum_verified=False)
    )

    assert transition.state == "rejected"
    assert transition.metrics is None
    assert "manifest" not in json.loads(registry.path.read_text(encoding="utf-8").splitlines()[-1])


def test_validated_report_exposes_the_manifest_metrics(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.record("submitted", "run-1")
    registry.record_collected("run-1")

    transition = registry.record_validated(_report("run-1"))

    assert transition.state == "validated"
    assert transition.metrics == {"val_loss": 0.25}
    assert registry.state("run-1") == "validated"


def test_persisting_a_remote_id_is_idempotent(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    spec = make_spec()

    first = registry.record_submitted(spec, remote_id="someone/run-1", submitted_by="agent")
    second = registry.record_submitted(spec, remote_id="someone/run-1", submitted_by="agent")

    assert first.state == "submitted"
    assert second.state == "submitted"
    assert len(registry.transitions(spec.run_id)) == 1
    assert registry.remote_id(spec.run_id) == "someone/run-1"


def test_the_submitted_line_carries_the_full_spec(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    spec = make_spec()

    registry.record_submitted(spec, remote_id="someone/run-1", submitted_by="human")

    restored = registry.spec(spec.run_id)
    assert restored == spec
    assert registry.spec_document(spec.run_id)["run_id"] == spec.run_id


def test_remote_id_survives_a_later_transition(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.record("submitted", "run-1", remote_id="someone/run-1")
    registry.record("running", "run-1")

    assert registry.remote_id("run-1") == "someone/run-1"


def test_secrets_are_redacted_before_they_reach_the_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_registry_secret_value")
    registry = _registry(tmp_path)

    registry.record("submitted", "run-1", detail="token=hf_registry_secret_value")

    text = registry.path.read_text(encoding="utf-8")
    assert "hf_registry_secret_value" not in text
    assert "[REDACTED:HF_TOKEN]" in text


def test_status_summary_reports_the_last_state_per_run(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.record("submitted", "run-1", platform="kaggle")
    registry.record("running", "run-1", platform="kaggle")
    registry.record("submitted", "run-2", platform="colab")

    summary = registry.status_summary()

    assert [entry["run_id"] for entry in summary] == ["run-1", "run-2"]
    assert summary[0]["state"] == "running"
    assert summary[0]["transitions"] == 2


def test_a_malformed_line_is_skipped_not_fatal(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.record("submitted", "run-1")
    with registry.path.open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")

    assert registry.state("run-1") == "submitted"


def test_a_validated_run_may_be_resubmitted(tmp_path: Path) -> None:
    """The kernel is the same; the artifacts are new. Both attempts stay in the append-only record."""
    registry = _registry(tmp_path)
    registry.record("submitted", "run-1")
    registry.record_collected("run-1")
    registry.record_validated(_report("run-1"))
    registry.record("resubmitted", "run-1")

    assert registry.state("run-1") == "resubmitted"
    assert [record["state"] for record in registry.transitions("run-1")][-2:] == [
        "validated",
        "resubmitted",
    ]
