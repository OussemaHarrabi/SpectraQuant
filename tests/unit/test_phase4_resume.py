"""Collection rejection must preserve the reason and append-only history."""

import json

from _cloud_fixtures import SHA_A, make_spec

from spectraquant.cloud.collect import ValidationReport
from spectraquant.cloud.registry import Registry


def test_recollection_corruption_is_rejected_from_collected_state(tmp_path):
    from scripts.experiments.phase4_resume import reject_collection

    registry = Registry(tmp_path / "registry.jsonl")
    spec = make_spec(platform="kaggle")
    run = tmp_path / spec.run_id
    registry.record_submitted(spec, remote_id="owner/run", submitted_by="agent")
    registry.record_collected(spec.run_id)
    reject_collection(registry, run, "missing packed checkpoint", "owner/run")
    assert registry.state(spec.run_id) == "rejected"
    assert "missing packed checkpoint" in registry.transitions(spec.run_id)[-1]["failure_reason"]
    assert json.loads((run / "validation-rejection.json").read_text())["state"] == "rejected"


def test_corrupt_local_copy_does_not_erase_historical_validated_result(tmp_path):
    from scripts.experiments.phase4_resume import reject_collection

    registry = Registry(tmp_path / "registry.jsonl")
    spec = make_spec(platform="kaggle")
    run = tmp_path / spec.run_id
    registry.record_submitted(spec, remote_id="owner/run", submitted_by="agent")
    registry.record_collected(spec.run_id)
    registry.record_validated(
        ValidationReport(
            run_id=spec.run_id,
            state="validated",
            source_dir=str(run),
            checksum_verified=True,
            artifact_checksums={"metrics.json": SHA_A},
            manifest={"metrics": {}},
            manifest_sha256=SHA_A,
        )
    )
    reject_collection(registry, run, "local copy changed after validation", "owner/run")
    assert registry.state(spec.run_id) == "validated"
    assert (
        "local copy changed"
        in json.loads((run / "validation-rejection.json").read_text())["reasons"][0]
    )
