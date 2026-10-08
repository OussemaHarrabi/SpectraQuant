"""Shared pytest fixtures for the SpectraQuant test suite.

All tests are CPU-only and deterministic: nothing here downloads data, touches a GPU, or depends on
wall-clock time. Temporary config trees are created under a directory literally named ``configs`` so
that Hydra composition behaves exactly as it does in the repository.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent

_MODEL_YAML = """\
name: tiny-char-transformer
vocab_size: 32
d_model: 64
n_heads: 4
n_layers: 2
max_seq_len: 32
dropout: 0.0
"""

_DATA_YAML = """\
name: synthetic-lcg-v1
kind: synthetic
vocab_size: 32
seq_len: 32
n_train_sequences: 24
n_val_sequences: 8
generator: lcg
lcg_a: 3
lcg_b: 7
train_start: 0
val_start: 24
"""

_METHOD_YAML = """\
name: none
kind: none
compression:
  method: none
  ranks: null
  bits: null
  group_size: null
  exclusions: []
measurement_class: null
"""

_EXPERIMENT_YAML = """\
# @package _global_
defaults:
  - /model: tiny
  - /data: synthetic
  - /method: none
  - _self_

experiment:
  name: smoke
  seed: 1234
  training:
    steps: 3
    batch_size: 4
    lr: 0.003
    weight_decay: 0.0
    grad_clip: 1.0
    log_every: 1
  output:
    results_dir: artifacts/sample-results
    filename: smoke-manifest.json
"""


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path of the repository root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def smoke_config_path(repo_root: Path) -> Path:
    """The committed smoke experiment config."""
    return repo_root / "configs" / "experiment" / "smoke.yaml"


@pytest.fixture(scope="session")
def sample_manifest_path(repo_root: Path) -> Path:
    """The committed sample run manifest (produced by ``spectraquant smoke``)."""
    return repo_root / "artifacts" / "sample-results" / "smoke-manifest.json"


@pytest.fixture(scope="session")
def schema_path(repo_root: Path) -> Path:
    """The published run-manifest JSON Schema."""
    return repo_root / "artifacts" / "schemas" / "run-manifest.schema.json"


@pytest.fixture
def config_tree(tmp_path: Path) -> Path:
    """Create a throwaway ``configs/`` tree that composes like the real one.

    Returns the path of the ``configs`` directory; the experiment file is
    ``<configs>/experiment/smoke.yaml``. Tests may rewrite individual files to exercise
    validation failures.
    """
    configs = tmp_path / "configs"
    (configs / "model").mkdir(parents=True)
    (configs / "data").mkdir(parents=True)
    (configs / "method").mkdir(parents=True)
    (configs / "experiment").mkdir(parents=True)
    (configs / "model" / "tiny.yaml").write_text(_MODEL_YAML, encoding="utf-8")
    (configs / "data" / "synthetic.yaml").write_text(_DATA_YAML, encoding="utf-8")
    (configs / "method" / "none.yaml").write_text(_METHOD_YAML, encoding="utf-8")
    (configs / "experiment" / "smoke.yaml").write_text(_EXPERIMENT_YAML, encoding="utf-8")
    return configs
