"""Tests for the notebook stage bridge (``spectraquant.cloud.notebook_stages``)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spectraquant.cloud.notebook_stages import STAGES, run_stage


def _context(tmp_path: Path, **overrides: object) -> Path:
    work = tmp_path / "work"
    (work / "artifacts" / "run").mkdir(parents=True, exist_ok=True)
    (work / "repo").mkdir(parents=True, exist_ok=True)
    (work / "logs").mkdir(parents=True, exist_ok=True)
    document = {
        "run_id": "unit-run",
        "spec": {
            "run_id": "unit-run",
            "platform": "kaggle",
            "experiment_config": "configs/tier1/smollm2_135m.yaml",
            "overrides": [],
            "dataset_refs": [],
            "seeds": [0],
            "git_commit": "0" * 40,
            "install_spec": "uv sync --all-extras",
            "measurement_class_expected": 2,
            "gpu_required": False,
        },
        "paths": {
            "repo": str(work / "repo"),
            "work": str(work),
            "artifact": str(work / "artifacts" / "run"),
            "log": str(work / "logs" / "run.log"),
            "data": str(work / "data"),
            "export": str(work / "export"),
        },
        "scalars": {
            "started_utc": "2026-10-09T00:00:00Z",
            "finished_utc": "2026-10-09T00:05:00Z",
            "run_rc": 0,
            "run_out": "ok",
            "wall_time_s": 12.5,
            "environment": {"python": "3.11"},
            "notebook_digest": "sha256:" + "a" * 64,
            "notebook_template_version": "2.0.0",
        },
    }
    document.update(overrides)
    path = work / "stage-context.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_stage_list_is_the_documented_order() -> None:
    assert STAGES == ("datasets", "manifest", "export")


def test_unknown_stage_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown stage"):
        run_stage("nonsense", _context(tmp_path))


def test_missing_context_keys_are_rejected(tmp_path: Path) -> None:
    path = _context(tmp_path)
    document = json.loads(path.read_text())
    del document["paths"]
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="paths"):
        run_stage("export", path)


def test_manifest_stage_writes_a_validated_manifest_and_extends_the_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The manifest stage builds the document through repository code and records its path."""
    path = _context(tmp_path)

    def fake_write(out_dir, document, cloud_context=None):  # type: ignore[no-untyped-def]
        target = Path(out_dir) / "run_manifest.json"
        target.write_text(json.dumps(document), encoding="utf-8")
        (Path(out_dir) / "cloud_context.json").write_text(
            json.dumps(cloud_context), encoding="utf-8"
        )
        return target

    monkeypatch.setattr("spectraquant.cloud.remote.write_run_manifest", fake_write)
    summary = run_stage("manifest", path)

    document = json.loads(path.read_text())
    assert "manifest_path" in document
    assert document["run_status"] == "success"
    assert "manifest written" in summary
    assert (Path(document["paths"]["artifact"]) / "run_manifest.json").is_file()


def test_manifest_stage_records_a_failed_run_as_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-zero runner exit must land in the manifest as a failure, never as a success."""
    path = _context(tmp_path)
    document = json.loads(path.read_text())
    document["scalars"]["run_rc"] = 3
    document["scalars"]["run_out"] = "boom"
    path.write_text(json.dumps(document), encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_write(out_dir, document, cloud_context=None):  # type: ignore[no-untyped-def]
        captured["status"] = document.get("status")
        captured["failure_reason"] = document.get("failure_reason")
        target = Path(out_dir) / "run_manifest.json"
        target.write_text(json.dumps(document), encoding="utf-8")
        return target

    monkeypatch.setattr("spectraquant.cloud.remote.write_run_manifest", fake_write)
    run_stage("manifest", path)

    assert captured["status"] == "failed"
    assert "exited 3" in str(captured["failure_reason"])


def test_export_stage_copies_the_bundle_and_prints_the_result_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _context(tmp_path)
    document = json.loads(path.read_text())
    artifact = Path(document["paths"]["artifact"])
    (artifact / "run_manifest.json").write_text("{}", encoding="utf-8")
    (artifact / "metrics.json").write_text("{}", encoding="utf-8")

    summary = run_stage("export", path)

    export = Path(document["paths"]["export"])
    assert (export / "run_manifest.json").is_file()
    assert (export / "metrics.json").is_file()
    assert (export / "result.json").is_file()
    printed = capsys.readouterr().out
    assert "SPECTRAQUANT_RESULT_JSON=" in printed
    assert "2 artifact" in summary


def test_context_round_trips_deterministically(tmp_path: Path) -> None:
    path = _context(tmp_path)
    document = json.loads(path.read_text())
    artifact = Path(document["paths"]["artifact"])
    (artifact / "run_manifest.json").write_text("{}", encoding="utf-8")
    run_stage("export", path)
    first = path.read_text(encoding="utf-8")
    run_stage("export", path)
    second = path.read_text(encoding="utf-8")
    assert first == second
