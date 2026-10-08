"""CLI tests: ``env``, ``smoke`` and ``validate-manifest`` through the real Typer app."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from spectraquant.cli.main import app
from spectraquant.reporting.manifests import validate_manifest_file

runner = CliRunner()


def _combined(result: Any) -> str:
    """stdout + stderr, whichever stream the click version routes output to."""
    return result.output + (getattr(result, "stderr", "") or "")


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert "spectraquant 0.1.0.dev0" in _combined(result)


def test_env_prints_json() -> None:
    result = runner.invoke(app, ["env"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["device"] == "cpu"
    assert payload["software"]["python"].startswith("3.11")
    classes = payload["measurement_classes"]
    assert classes["locally_producible"] == [1, 2, 3]
    assert classes["locally_producible_with_conditions"] == ["4-CPU"]
    assert classes["not_producible_locally"] == ["4-GPU", 5]
    assert classes["class_4_cpu_implemented"] is False


def test_validate_manifest_accepts_committed_sample(sample_manifest_path: Path) -> None:
    result = runner.invoke(app, ["validate-manifest", str(sample_manifest_path)])

    assert result.exit_code == 0, _combined(result)
    assert "OK" in _combined(result)


def test_validate_manifest_json_output(sample_manifest_path: Path) -> None:
    result = runner.invoke(app, ["validate-manifest", str(sample_manifest_path), "--json"])

    assert result.exit_code == 0
    assert json.loads(result.output)["valid"] is True


def test_validate_manifest_rejects_invalid_document(tmp_path: Path) -> None:
    broken = tmp_path / "broken.manifest.json"
    broken.write_text('{"run_id": "x"}', encoding="utf-8")

    result = runner.invoke(app, ["validate-manifest", str(broken)])

    assert result.exit_code == 1
    assert "INVALID" in _combined(result)


def test_smoke_writes_a_valid_manifest(tmp_path: Path, smoke_config_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "smoke",
            "--config",
            str(smoke_config_path),
            "--out",
            str(tmp_path),
            "--quiet",
        ],
    )

    assert result.exit_code == 0, _combined(result)
    manifest_path = Path(result.output.strip().splitlines()[-1])
    assert manifest_path.is_file()
    assert manifest_path.parent == tmp_path
    validate_manifest_file(manifest_path)


def test_smoke_reports_the_loss_sequence(tmp_path: Path, smoke_config_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "smoke",
            "--config",
            str(smoke_config_path),
            "--out",
            str(tmp_path),
            "--seed",
            "7",
        ],
    )

    assert result.exit_code == 0, _combined(result)
    output = _combined(result)
    assert "loss sequence:" in output
    assert "loss_sequence_sha256: sha256:" in output
    assert "seed: 7" in output


def test_smoke_with_missing_config_fails_loudly(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["smoke", "--config", str(tmp_path / "configs" / "experiment" / "nope.yaml")]
    )

    assert result.exit_code == 1
    assert "smoke failed" in _combined(result)


def test_help_lists_the_subcommands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("env", "smoke", "validate-manifest", "compare-manifests"):
        assert command in _combined(result)


def test_compare_manifests_accepts_an_equal_memory_pair(
    comparability_fixtures_dir: Path,
) -> None:
    result = runner.invoke(
        app,
        [
            "compare-manifests",
            str(comparability_fixtures_dir / "int4-arm-a.manifest.json"),
            str(comparability_fixtures_dir / "int4-arm-b.manifest.json"),
            "--tolerance-bytes",
            "200",
        ],
    )

    assert result.exit_code == 0, _combined(result)
    output = _combined(result)
    assert "OK" in output and "EQUAL MEMORY" in output
    assert "1001200" in output and "1001300" in output


def test_compare_manifests_json_verdict(comparability_fixtures_dir: Path) -> None:
    result = runner.invoke(
        app,
        [
            "compare-manifests",
            str(comparability_fixtures_dir / "int4-arm-a.manifest.json"),
            str(comparability_fixtures_dir / "int4-arm-b.manifest.json"),
            "--json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, _combined(result)
    payload = json.loads(result.output)
    assert payload["equal"] is True
    assert payload["bytes_source"] == "measured"
    assert payload["measurement_class"] == 3
    assert payload["difference_bytes"] == 100


def test_compare_manifests_rejects_a_byte_unequal_pair(
    comparability_fixtures_dir: Path,
) -> None:
    result = runner.invoke(
        app,
        [
            "compare-manifests",
            str(comparability_fixtures_dir / "int4-arm-a.manifest.json"),
            str(comparability_fixtures_dir / "int4-arm-oversized.manifest.json"),
            "--tolerance-bytes",
            "0",
        ],
    )

    assert result.exit_code == 1
    output = _combined(result)
    assert "REJECTED" in output
    assert "10012" in output


def test_compare_manifests_rejection_json_reports_the_figures(
    comparability_fixtures_dir: Path,
) -> None:
    result = runner.invoke(
        app,
        [
            "compare-manifests",
            str(comparability_fixtures_dir / "int4-arm-a.manifest.json"),
            str(comparability_fixtures_dir / "int4-arm-oversized.manifest.json"),
            "--tolerance-bytes",
            "0",
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["equal"] is False
    assert payload["bytes_a"] == 1_001_200
    assert payload["bytes_b"] == 1_011_212
    assert payload["difference_bytes"] == 10_012


def test_compare_manifests_rejects_an_unvalidatable_manifest(tmp_path: Path) -> None:
    broken = tmp_path / "broken.manifest.json"
    broken.write_text("{not json", encoding="utf-8")

    result = runner.invoke(app, ["compare-manifests", str(broken), str(broken)])

    assert result.exit_code == 1
    assert "REJECTED" in _combined(result)
