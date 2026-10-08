"""Regression guard for the committed sample manifest.

The repository ships ``artifacts/sample-results/smoke-manifest.json`` as evidence that the harness
works end to end. This test fails if that artifact stops validating against the published schema or
stops being internally consistent (recorded digest vs recorded loss sequence, recorded dataset
checksum vs the recorded data config and seed).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from spectraquant.config import DataConfig
from spectraquant.data.synthetic import build_corpus
from spectraquant.reporting.manifests import validate_manifest_file
from spectraquant.training.smoke import loss_sequence_sha256

pytestmark = pytest.mark.regression

RUN_ID_PATTERN = re.compile(r"^smoke-\d{8}T\d{6}Z-[0-9a-z]{12}-[0-9a-z]{6}$")


@pytest.fixture(scope="module")
def sample_manifest(sample_manifest_path: Path) -> dict[str, Any]:
    if not sample_manifest_path.is_file():
        pytest.fail(
            f"committed sample manifest is missing: {sample_manifest_path}; "
            "regenerate it with `uv run spectraquant smoke --config configs/experiment/smoke.yaml`"
        )
    return json.loads(sample_manifest_path.read_text(encoding="utf-8"))


def test_sample_manifest_validates_against_schema(sample_manifest_path: Path) -> None:
    document = validate_manifest_file(sample_manifest_path)

    assert document["schema_version"] == 1


def test_sample_manifest_identity_and_status(sample_manifest: dict[str, Any]) -> None:
    assert RUN_ID_PATTERN.match(sample_manifest["run_id"])
    assert sample_manifest["status"] == "success"
    assert sample_manifest["failure_reason"] is None
    assert sample_manifest["compression"]["method"] == "none"
    assert sample_manifest["measurement_class"] is None
    assert sample_manifest["runtime_backend"] == "torch-cpu-fp32"
    assert sample_manifest["git_dirty"] in (True, False, None)


def test_recorded_loss_digest_matches_recorded_losses(sample_manifest: dict[str, Any]) -> None:
    losses = sample_manifest["metrics"]["loss_sequence"]

    assert len(losses) == sample_manifest["training"]["steps"]
    assert loss_sequence_sha256(losses) == sample_manifest["metrics"]["loss_sequence_sha256"]


def test_recorded_dataset_checksum_matches_recorded_config(
    sample_manifest: dict[str, Any],
) -> None:
    data_cfg = DataConfig.model_validate(sample_manifest["resolved_config"]["data"])

    corpus = build_corpus(data_cfg, seed=sample_manifest["seed"])

    assert corpus.checksum == sample_manifest["dataset_checksums"][0]
    assert corpus.seq_len == sample_manifest["metrics"]["seq_len"]
    assert 0 < sample_manifest["metrics"]["parameter_count"] <= 200_000
    assert sample_manifest["dataset_revisions"] == [None]


def test_sample_manifest_is_self_consistent(sample_manifest: dict[str, Any]) -> None:
    training = sample_manifest["training"]
    metrics = sample_manifest["metrics"]

    assert training["tokens"] == training["steps"] * metrics["batch_size"] * metrics["seq_len"]
    assert metrics["train_loss_final"] < metrics["train_loss_initial"]
    assert training["wall_time_s"] > 0.0
    assert training["peak_mem_mb"] is None or training["peak_mem_mb"] > 0.0
    assert sample_manifest["hardware"]["torch_threads"] == 1
    assert sample_manifest["model_revision"].startswith("sha256:")
    assert sample_manifest["config_path"].endswith("smoke.yaml")
