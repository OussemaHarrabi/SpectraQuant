"""End-to-end integration test: the committed smoke experiment, twice, must be identical.

This is the harness-level guarantee the whole project rests on: same config + same seed =>
bit-for-bit identical loss sequence, plus a manifest that validates against the published schema.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from spectraquant.config import load_experiment_config
from spectraquant.reporting.manifests import validate_manifest_file
from spectraquant.training.smoke import run_smoke_experiment
from spectraquant.training.tiny_lm import MAX_SMOKE_PARAMS

pytestmark = pytest.mark.integration

WALL_TIME_BUDGET_S = 120.0


def test_smoke_is_deterministic_and_writes_valid_manifests(
    tmp_path: Path, smoke_config_path: Path
) -> None:
    cfg = load_experiment_config(smoke_config_path)

    first = run_smoke_experiment(cfg, results_dir=tmp_path / "run-a")
    second = run_smoke_experiment(cfg, results_dir=tmp_path / "run-b")

    # 1. determinism: identical loss sequences and identical digests
    assert first.loss_sequence == second.loss_sequence
    assert first.loss_sequence_sha256 == second.loss_sequence_sha256
    assert first.val_loss == second.val_loss

    # 2. budget: CPU, fast, tiny
    assert first.wall_time_s < WALL_TIME_BUDGET_S
    assert first.parameter_count <= MAX_SMOKE_PARAMS
    assert len(first.loss_sequence) == cfg.experiment.training.steps

    # 3. the experiment actually learns: below the uniform-prior entropy, and improving
    uniform_entropy = math.log(cfg.data.vocab_size)
    assert first.initial_loss > uniform_entropy * 0.9
    assert first.final_loss < first.initial_loss
    assert first.val_loss < uniform_entropy

    # 4. manifests exist, validate, and describe what happened
    assert first.manifest_path is not None and second.manifest_path is not None
    for path in (first.manifest_path, second.manifest_path):
        document = validate_manifest_file(path)
        assert document["schema_version"] == 1
        assert document["status"] == "success"
        assert document["failure_reason"] is None
        assert document["compression"]["method"] == "none"
        assert document["measurement_class"] is None
        assert document["runtime_backend"] == "torch-cpu-fp32"
        assert document["hardware"]["torch_device"] == "cpu"
        assert document["seed"] == cfg.experiment.seed
        assert document["dataset_ids"] == [cfg.data.name]
        assert document["dataset_checksums"][0] == first.corpus.checksum
        assert document["training"]["steps"] == cfg.experiment.training.steps
        assert document["training"]["tokens"] == (
            cfg.experiment.training.steps
            * cfg.experiment.training.batch_size
            * first.corpus.seq_len
        )
        assert document["training"]["peak_mem_mb"] is not None
        assert document["metrics"]["loss_sequence"] == [
            round(loss, 9) for loss in first.loss_sequence
        ]


def test_write_disabled_creates_no_files(tmp_path: Path, smoke_config_path: Path) -> None:
    cfg = load_experiment_config(
        smoke_config_path, ["experiment.training.steps=2", "experiment.training.batch_size=2"]
    )
    results_dir = tmp_path / "not-created"

    result = run_smoke_experiment(cfg, results_dir=results_dir, write=False)

    assert result.manifest_path is None
    assert not results_dir.exists()


def test_seed_override_changes_the_run(tmp_path: Path, smoke_config_path: Path) -> None:
    overrides = ["experiment.training.steps=5"]
    base = load_experiment_config(smoke_config_path, overrides)
    other = load_experiment_config(smoke_config_path, [*overrides, "experiment.seed=99"])

    a = run_smoke_experiment(base, results_dir=tmp_path / "a", write=False)
    b = run_smoke_experiment(other, results_dir=tmp_path / "b", write=False)

    assert a.loss_sequence != b.loss_sequence
    assert a.corpus.checksum != b.corpus.checksum


def test_corpus_splits_are_disjoint(smoke_config_path: Path) -> None:
    cfg = load_experiment_config(smoke_config_path)

    result = run_smoke_experiment(cfg, write=False)

    train_starts = set(result.corpus.train[:, 0].tolist())
    val_starts = set(result.corpus.val[:, 0].tolist())
    assert not train_starts & val_starts
