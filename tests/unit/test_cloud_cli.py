"""The ``spectraquant cloud`` CLI surface (design note §7)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _cloud_fixtures import make_bundle, make_spec
from typer.testing import CliRunner

from spectraquant.cli.main import app
from spectraquant.cloud.registry import REGISTRY_FILENAME, Registry
from spectraquant.cloud.spec import load_spec

runner = CliRunner()


def _combined(result: object) -> str:
    return str(getattr(result, "output", "")) + str(getattr(result, "stderr", "") or "")


@pytest.fixture
def isolated_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Registry:
    """Point the default registry and the notebook directory at throwaway paths."""
    runs = tmp_path / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    notebooks = tmp_path / "generated"
    notebooks.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SPECTRAQUANT_RUNS_DIR", str(runs))
    monkeypatch.setenv("SPECTRAQUANT_NOTEBOOK_DIR", str(notebooks))
    return Registry(runs / REGISTRY_FILENAME)


def test_help_lists_the_cloud_subcommands() -> None:
    result = runner.invoke(app, ["cloud", "--help"])

    assert result.exit_code == 0
    for command in ("notebook", "submit", "status", "fetch", "collect", "registry"):
        assert command in _combined(result)


def test_root_help_still_lists_the_original_subcommands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("env", "smoke", "validate-manifest", "cloud"):
        assert command in _combined(result)


def test_notebook_command_writes_a_notebook_and_a_spec(
    tmp_path: Path, smoke_config_path: Path
) -> None:
    pytest.importorskip("nbformat")
    out = tmp_path / "smoke.ipynb"

    result = runner.invoke(
        app,
        [
            "cloud",
            "notebook",
            "--config",
            str(smoke_config_path),
            "--platform",
            "colab",
            "--out",
            str(out),
            "--run-id",
            "cli-notebook-run",
            "--allow-dirty",
        ],
    )

    assert result.exit_code == 0, _combined(result)
    assert out.is_file()
    text = _combined(result)
    assert "notebook_digest: sha256:" in text
    assert "cell 0: header" in text
    assert "cell 8: teardown" in text
    spec = load_spec(Path(f"{out}.spec.json"))
    assert spec.run_id == "cli-notebook-run"
    assert spec.platform == "colab"


def test_notebook_command_emits_json(tmp_path: Path, smoke_config_path: Path) -> None:
    pytest.importorskip("nbformat")
    out = tmp_path / "smoke.ipynb"

    result = runner.invoke(
        app,
        [
            "cloud",
            "notebook",
            "--config",
            str(smoke_config_path),
            "--out",
            str(out),
            "--run-id",
            "cli-json-run",
            "--allow-dirty",
            "--json",
        ],
    )

    assert result.exit_code == 0, _combined(result)
    payload = json.loads(result.output)
    assert payload["digest"].startswith("sha256:")
    assert [cell["stage"] for cell in payload["cells"]][1] == "environment"


def test_notebook_command_reports_a_bad_platform(tmp_path: Path, smoke_config_path: Path) -> None:
    result = runner.invoke(
        app,
        ["cloud", "notebook", "--config", str(smoke_config_path), "--platform", "on_prem"],
    )

    assert result.exit_code == 1
    assert "cloud failed" in _combined(result)


def test_submit_refuses_a_paid_platform_without_authorization(
    tmp_path: Path, isolated_registry: Registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A GCP project is authorized, but the paid flag is not: the budget guard must refuse.
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-project")
    monkeypatch.delenv("SPECTRAQUANT_ALLOW_PAID", raising=False)
    spec = make_spec(
        platform="colab_enterprise",
        gpu_required=True,
        max_cost_authorized_usd=5.0,
        run_id="cli-paid-run",
    )
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(spec.to_json_text(), encoding="utf-8")

    result = runner.invoke(app, ["cloud", "submit", "--spec", str(spec_path)])

    assert result.exit_code == 1
    assert "PaidSubmissionNotAuthorized" in _combined(result)
    # The guard runs before anything is emitted.
    assert list((tmp_path / "generated").iterdir()) == []


def test_submit_refuses_a_paid_platform_without_a_project(
    tmp_path: Path, isolated_registry: Registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Paid spend is authorized, but no GCP project is configured: the project guard must refuse.
    monkeypatch.setenv("SPECTRAQUANT_ALLOW_PAID", "1")
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GCLOUD_PROJECT", raising=False)
    spec = make_spec(
        platform="colab_enterprise",
        gpu_required=True,
        max_cost_authorized_usd=5.0,
        run_id="cli-no-project-run",
    )
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(spec.to_json_text(), encoding="utf-8")

    result = runner.invoke(app, ["cloud", "submit", "--spec", str(spec_path)])

    assert result.exit_code == 1
    assert "MissingGcpProject" in _combined(result)


def test_registry_command_lists_recorded_runs(tmp_path: Path, isolated_registry: Registry) -> None:
    isolated_registry.record("submitted", "cli-run-1", platform="kaggle", remote_id="someone/x")

    result = runner.invoke(app, ["cloud", "registry"])

    assert result.exit_code == 0, _combined(result)
    assert "cli-run-1" in _combined(result)
    assert "submitted" in _combined(result)


def test_registry_command_emits_json(tmp_path: Path, isolated_registry: Registry) -> None:
    isolated_registry.record("submitted", "cli-run-1", platform="kaggle")

    result = runner.invoke(app, ["cloud", "registry", "--json"])

    payload = json.loads(result.output)
    assert payload["records"][0]["run_id"] == "cli-run-1"
    assert payload["path"].endswith(REGISTRY_FILENAME)


def test_collect_command_validates_a_bundle(tmp_path: Path, isolated_registry: Registry) -> None:
    spec = make_spec(run_id="cli-collect-run")
    isolated_registry.record_submitted(
        spec, remote_id="fake/1", submitted_by="agent", notebook_digest=None
    )
    bundle = make_bundle(tmp_path / "bundle", spec)

    result = runner.invoke(
        app,
        ["cloud", "collect", "--run-id", spec.run_id, "--source", str(bundle)],
    )

    assert result.exit_code == 0, _combined(result)
    assert "validated" in _combined(result)
    assert isolated_registry.state(spec.run_id) == "validated"


def test_collect_command_exits_one_on_rejection(
    tmp_path: Path, isolated_registry: Registry
) -> None:
    spec = make_spec(run_id="cli-reject-run")
    isolated_registry.record_submitted(
        spec, remote_id="fake/1", submitted_by="agent", notebook_digest=None
    )
    bundle = make_bundle(tmp_path / "bundle", spec, artifact_bytes=None)

    result = runner.invoke(
        app, ["cloud", "collect", "--run-id", spec.run_id, "--source", str(bundle), "--json"]
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["state"] == "rejected"
    assert any("expected artifact missing" in reason for reason in payload["reasons"])


def test_status_command_reports_the_recorded_platform(
    tmp_path: Path, isolated_registry: Registry
) -> None:
    isolated_registry.record("submitted", "cli-run-2", platform="colab", remote_id=None)

    result = runner.invoke(app, ["cloud", "status", "--run-id", "cli-run-2", "--json"])

    assert result.exit_code == 0, _combined(result)
    payload = json.loads(result.output)
    assert payload["state"] == "unknown"
    assert "no submission API" in payload["detail"]


def test_status_command_fails_for_an_unknown_run(
    tmp_path: Path, isolated_registry: Registry
) -> None:
    result = runner.invoke(app, ["cloud", "status", "--run-id", "never-recorded"])

    assert result.exit_code == 1
    assert "not in the registry" in _combined(result)


def test_fetch_command_refuses_for_a_colab_handoff(
    tmp_path: Path, isolated_registry: Registry
) -> None:
    isolated_registry.record("submitted", "cli-run-3", platform="colab", remote_id=None)

    result = runner.invoke(
        app, ["cloud", "fetch", "--run-id", "cli-run-3", "--dest", str(tmp_path / "dest")]
    )

    assert result.exit_code == 1
    assert "no artifact-download API" in _combined(result)


def test_submit_command_uses_the_local_adapter(
    tmp_path: Path, isolated_registry: Registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = make_spec(run_id="cli-local-run")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(spec.to_json_text(), encoding="utf-8")
    notebook = tmp_path / "run.ipynb"
    notebook.write_text("{}", encoding="utf-8")

    # The local adapter would run the (not yet implemented) `spectraquant run`; the failure is
    # recorded rather than raised, which is exactly what the CLI must report.
    result = runner.invoke(
        app, ["cloud", "submit", "--spec", str(spec_path), "--notebook", str(notebook)]
    )

    assert result.exit_code == 0, _combined(result)
    assert "submitted: cli-local-run" in _combined(result)
    assert isolated_registry.state(spec.run_id) in {"finished", "failed"}
