"""Synthetic end-to-end: ``submit -> fetch -> collect`` with no network (design note §9)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _cloud_fixtures import (
    ARTIFACT_NAME,
    FakeAdapter,
    make_bundle,
    make_spec,
)

from spectraquant.cloud.collect import collect
from spectraquant.cloud.registry import REGISTRY_FILENAME, Registry
from spectraquant.cloud.spec import ExpectedArtifact, spec_from_config

RESULT_NAME = "artifacts/result.json"


def test_submit_fetch_collect_produces_a_validated_entry(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "runs" / REGISTRY_FILENAME)
    spec = make_spec(platform="kaggle", gpu_required=True, run_id="e2e-run")
    bundle = make_bundle(tmp_path / "cloud-bundle", spec)
    adapter = FakeAdapter(bundle_source=bundle, registry=registry)
    notebook = tmp_path / "generated.ipynb"
    notebook.write_text("{}", encoding="utf-8")

    remote_id = adapter.submit(spec, str(notebook))
    status = adapter.status(spec.run_id)
    registry.record("running", spec.run_id, platform=spec.platform, remote_id=remote_id)
    registry.record("finished", spec.run_id, platform=spec.platform, remote_id=remote_id)
    fetched = adapter.fetch(spec.run_id, str(tmp_path / "fetched"))
    report = collect(spec.run_id, str(tmp_path / "fetched"), registry=registry)

    assert status.state == "finished"
    assert fetched.ok is True
    assert report.ok is True, report.reasons
    assert registry.state(spec.run_id) == "validated"

    states = [record["state"] for record in registry.transitions(spec.run_id)]
    assert states == ["submitted", "running", "finished", "collected", "validated"]

    validated = registry.latest(spec.run_id)
    assert validated is not None
    assert validated["remote_id"] == remote_id
    assert validated["metrics"]["val_loss"] == 0.5
    assert validated["artifact_checksums"][ARTIFACT_NAME].startswith("sha256:")
    assert validated["checksum_verified"] is True

    manifest = json.loads((tmp_path / "fetched" / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == spec.run_id
    assert manifest["git_commit"] == spec.git_commit
    assert manifest["status"] == "success"
    assert manifest["dataset_ids"] == ["synthetic-lcg-v1"]


def test_the_same_flow_rejects_a_tampered_artifact(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "runs" / REGISTRY_FILENAME)
    spec = make_spec(platform="kaggle", gpu_required=True, run_id="e2e-tampered")
    bundle = make_bundle(tmp_path / "cloud-bundle", spec)
    adapter = FakeAdapter(bundle_source=bundle, registry=registry)
    notebook = tmp_path / "generated.ipynb"
    notebook.write_text("{}", encoding="utf-8")
    adapter.submit(spec, str(notebook))
    adapter.fetch(spec.run_id, str(tmp_path / "fetched"))
    # Tamper with the downloaded artifact after the fact.
    (tmp_path / "fetched" / ARTIFACT_NAME).write_bytes(b"tampered\n")

    spec_with_digest = make_spec(
        platform="kaggle",
        gpu_required=True,
        run_id="e2e-tampered",
        expected_artifacts=[
            ExpectedArtifact(
                name=RESULT_NAME,
                sha256="sha256:" + "c" * 64,
                min_bytes=1,
            )
        ],
    )
    report = collect(
        spec.run_id, str(tmp_path / "fetched"), spec=spec_with_digest, registry=registry
    )

    assert report.ok is False
    assert any("sha256 mismatch" in reason for reason in report.reasons)
    assert registry.state(spec.run_id) == "rejected"


def test_a_resumed_collection_does_not_resubmit(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "runs" / REGISTRY_FILENAME)
    spec = make_spec(platform="kaggle", gpu_required=True, run_id="e2e-resume")
    bundle = make_bundle(tmp_path / "cloud-bundle", spec)
    adapter = FakeAdapter(bundle_source=bundle, registry=registry)
    notebook = tmp_path / "generated.ipynb"
    notebook.write_text("{}", encoding="utf-8")

    first = adapter.submit(spec, str(notebook))
    # Simulate an interruption: the collection step is re-run and finds the persisted id.
    persisted = registry.remote_id(spec.run_id)
    second = adapter.submit(spec, str(notebook))

    assert persisted == first
    assert second == first
    assert adapter.submit_calls == 2  # the fake adapter re-submits; the real ones short-circuit
    assert len(registry.transitions(spec.run_id)) == 1


@pytest.mark.parametrize("platform", ["kaggle", "colab_enterprise", "local_cpu"])
def test_collect_uses_the_recorded_spec_for_every_platform(tmp_path: Path, platform: str) -> None:
    registry = Registry(tmp_path / "runs" / REGISTRY_FILENAME)
    extra = {"max_cost_authorized_usd": 5.0} if platform == "colab_enterprise" else {}
    spec = make_spec(
        platform=platform,
        gpu_required=platform != "local_cpu",
        run_id=f"e2e-{platform}",
        **extra,
    )
    registry.record_submitted(spec, remote_id="fake/1", submitted_by="agent", notebook_digest=None)
    bundle = make_bundle(tmp_path / "bundle", spec)

    report = collect(spec.run_id, bundle, registry=registry)

    assert report.ok is True, report.reasons
    assert report.platform == platform


def test_the_spec_checksum_matches_the_remote_materialisation(tmp_path: Path) -> None:
    """The notebook's data cell must reproduce exactly the checksum the spec declares."""
    from spectraquant.cloud.remote import materialise_datasets

    spec = spec_from_config("configs/experiment/smoke.yaml", platform="colab", allow_dirty=True)

    records = materialise_datasets(
        config_path=spec.experiment_config,
        overrides=spec.overrides,
        dataset_refs=[ref.to_json() for ref in spec.dataset_refs],
        dest_dir=tmp_path / "data",
        seed=spec.seeds[0],
    )

    assert len(records) == len(spec.dataset_refs)
    assert records[0]["checksum"] == spec.dataset_refs[0].checksum
    assert Path(records[0]["path"]).is_file()


def test_a_tampered_dataset_is_refused_by_the_remote_verifier(tmp_path: Path) -> None:
    from spectraquant.cloud.remote import ChecksumMismatch, materialise_datasets

    spec = spec_from_config("configs/experiment/smoke.yaml", platform="colab", allow_dirty=True)
    refs = [ref.to_json() for ref in spec.dataset_refs]
    refs[0]["checksum"] = "sha256:" + "d" * 64

    with pytest.raises(ChecksumMismatch, match="checksum mismatch"):
        materialise_datasets(
            config_path=spec.experiment_config,
            overrides=spec.overrides,
            dataset_refs=refs,
            dest_dir=tmp_path / "data",
            seed=spec.seeds[0],
        )


def test_the_manifest_cell_produces_a_schema_valid_manifest(tmp_path: Path) -> None:
    """Exercise the notebook's manifest cell against the real config and the published schema."""
    from spectraquant.cloud.remote import build_run_manifest, write_run_manifest

    spec = spec_from_config("configs/experiment/smoke.yaml", platform="colab", allow_dirty=True)
    artifacts = tmp_path / "artifacts"
    (artifacts / "artifacts").mkdir(parents=True)
    (artifacts / "artifacts" / "smoke-manifest.json").write_text(
        '{"run_id": "x"}', encoding="utf-8"
    )
    document = build_run_manifest(
        spec=spec.to_json(),
        workdir=tmp_path,
        artifacts_dir=artifacts,
        status="success",
        started_utc="2026-10-08T00:00:00Z",
        finished_utc="2026-10-08T00:05:00Z",
        notebook_digest="sha256:" + "e" * 64,
        gpu_hours=0.5,
        remote_run_id="fake/1",
        submitted_by="human",
    )

    path = write_run_manifest(
        out_dir=artifacts, document=document, cloud_context={"run_id": spec.run_id}
    )

    assert path.name == "run_manifest.json"
    assert (artifacts / "cloud_context.json").is_file()
    assert document["run_id"] == spec.run_id
    assert document["compression"]["method"] == "none"
    assert document["measurement_class"] is None  # method=none records no class
    assert document["metrics"]["cloud.gpu_hours"] == 0.5
    assert document["metrics"]["cloud.notebook_digest"].startswith("sha256:")
    assert document["dataset_checksums"] == [ref.checksum for ref in spec.dataset_refs]
