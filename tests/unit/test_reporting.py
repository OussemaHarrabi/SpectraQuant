"""Reporting utilities: git provenance, structured logging, environment capture."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import pytest
import torch

from spectraquant.reporting import (
    collect_hardware,
    collect_software,
    environment_report,
    git_info,
    process_peak_rss_mb,
    resolve_device,
)
from spectraquant.reporting.logging import configure_logging, get_logger, log_event, reset_logging
from spectraquant.reporting.manifests import (
    CompressionBlock,
    RunManifest,
    TrainingBlock,
    utc_timestamp,
    validate_manifest_dict,
)

SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


@pytest.fixture(autouse=True)
def _clean_logging() -> None:
    reset_logging()
    yield
    reset_logging()


# --------------------------------------------------------------------------------------
# git provenance
# --------------------------------------------------------------------------------------
def test_git_info_reports_a_commit_inside_the_repository(repo_root: Path) -> None:
    info = git_info(repo_root)

    assert info.error is None
    assert info.commit is not None
    assert SHA_PATTERN.match(info.commit)
    assert info.short_commit == info.commit[:12]
    assert info.dirty in (True, False)


def test_git_info_does_not_fail_outside_a_checkout(tmp_path: Path) -> None:
    info = git_info(tmp_path)

    assert info.commit is None
    assert info.short_commit is None
    assert info.dirty is None
    assert info.error is not None
    assert not info.is_available


def test_git_info_survives_a_missing_git_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))

    info = git_info(tmp_path)

    assert info.commit is None
    assert info.error is not None


# --------------------------------------------------------------------------------------
# structured logging
# --------------------------------------------------------------------------------------
def test_level_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPECTRAQUANT_LOG_LEVEL", "debug")

    assert configure_logging(force=True).level == logging.DEBUG


def test_invalid_level_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPECTRAQUANT_LOG_LEVEL", "chatty")

    with pytest.raises(ValueError, match="SPECTRAQUANT_LOG_LEVEL"):
        configure_logging(force=True)


def test_explicit_level_overrides_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPECTRAQUANT_LOG_LEVEL", "DEBUG")

    assert configure_logging("WARNING", force=True).level == logging.WARNING


def test_text_event_renders_fields(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("SPECTRAQUANT_LOG_JSON", raising=False)
    logger = configure_logging("INFO", force=True)

    log_event(logger, logging.INFO, "step", fields={"step": 3, "loss": 0.5})
    output = capsys.readouterr().err

    assert "event=step" in output
    assert "step=3" in output
    assert "loss=0.5" in output


def test_json_mode_emits_one_object_per_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SPECTRAQUANT_LOG_JSON", "1")
    logger = configure_logging("INFO", force=True)

    log_event(logger, logging.INFO, "manifest_written", fields={"path": "a/b.json"})
    line = capsys.readouterr().err.strip()

    payload = json.loads(line)
    assert payload["event"] == "manifest_written"
    assert payload["path"] == "a/b.json"
    assert payload["level"] == "INFO"


def test_reserved_log_fields_are_rejected() -> None:
    logger = configure_logging("INFO", force=True)

    with pytest.raises(ValueError, match="reserved"):
        log_event(logger, logging.INFO, "event", fields={"message": "clash"})


def test_get_logger_namespaces_children() -> None:
    assert get_logger("training").name == "spectraquant.training"
    assert get_logger("spectraquant.cli").name == "spectraquant.cli"


# --------------------------------------------------------------------------------------
# environment capture
# --------------------------------------------------------------------------------------
def test_device_is_cpu_without_cuda() -> None:
    if not torch.cuda.is_available():
        assert resolve_device() == "cpu"


def test_hardware_block_matches_the_schema() -> None:
    hardware = collect_hardware()

    assert hardware["gpu_available"] == torch.cuda.is_available()
    assert hardware["gpu_devices"] == []
    assert hardware["torch_threads"] >= 1
    assert hardware["cpu_count_logical"] == os.cpu_count()

    document = _manifest_with_captured_environment()
    validate_manifest_dict(document)


def test_software_block_records_versions() -> None:
    software = collect_software()

    assert software["python"].startswith("3.11")
    assert software["torch"]
    assert software["spectraquant"] == "0.1.0.dev0"
    assert software["torch_cuda_build"] is None  # CPU wheel on the workstation of record


def test_environment_report_states_class_availability() -> None:
    report = environment_report()
    classes = report["measurement_classes"]

    assert classes["locally_producible"] == [1, 2, 3]
    assert classes["locally_producible_with_conditions"] == ["4-CPU"]
    assert classes["not_producible_locally"] == ["4-GPU", 5]
    assert classes["class_4_cpu_available_in_principle"] is True
    assert classes["class_4_cpu_implemented"] is False
    assert "hardware" in report and "software" in report


def test_peak_rss_is_measured_or_explicitly_unknown() -> None:
    value = process_peak_rss_mb()

    assert value is None or value > 0.0


def _manifest_with_captured_environment() -> dict[str, object]:
    """Build a manifest whose hardware/software blocks come from the live capture."""
    manifest = RunManifest(
        run_id="env-capture-test",
        timestamp_utc=utc_timestamp(),
        config_path="configs/experiment/smoke.yaml",
        resolved_config={},
        seed=0,
        hardware=collect_hardware(),
        software=collect_software(),
        compression=CompressionBlock(method="none"),
        measurement_class=None,
        training=TrainingBlock(steps=0, tokens=0, wall_time_s=0.0),
    )
    return manifest.to_json_dict()
