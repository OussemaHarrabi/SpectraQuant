"""Config composition and validation tests (Hydra + Pydantic)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from spectraquant.config import ExperimentConfig, compose_raw_config, load_experiment_config


def test_smoke_config_composes(smoke_config_path: Path) -> None:
    cfg = load_experiment_config(smoke_config_path)

    assert isinstance(cfg, ExperimentConfig)
    assert cfg.experiment.name == "smoke"
    assert cfg.experiment.training.steps == 50
    assert cfg.model.vocab_size == cfg.data.vocab_size == 32
    assert cfg.model.d_model % cfg.model.n_heads == 0
    assert cfg.method.compression.method == "none"
    assert cfg.method.measurement_class is None
    assert cfg.is_uncompressed
    assert cfg.config_path == str(smoke_config_path.resolve())


def test_compose_raw_config_is_plain_python(smoke_config_path: Path) -> None:
    raw, resolved = compose_raw_config(smoke_config_path)

    assert set(raw) >= {"model", "data", "method", "experiment"}
    assert isinstance(raw["model"], dict)
    assert raw["experiment"]["seed"] == 1234
    assert resolved == smoke_config_path.resolve()


def test_relative_config_path_resolves_from_any_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    cfg = load_experiment_config("configs/experiment/smoke.yaml")

    assert cfg.experiment.seed == 1234


def test_override_applied(smoke_config_path: Path) -> None:
    cfg = load_experiment_config(smoke_config_path, ["experiment.seed=7", "model.n_layers=1"])

    assert cfg.experiment.seed == 7
    assert cfg.model.n_layers == 1


def test_missing_config_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        load_experiment_config("configs/experiment/does-not-exist.yaml")


def test_unknown_key_is_rejected(config_tree: Path) -> None:
    experiment = config_tree / "experiment" / "smoke.yaml"
    experiment.write_text(
        experiment.read_text(encoding="utf-8").replace(
            "  name: smoke", "  name: smoke\n  typo_key: 1"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError) as excinfo:
        load_experiment_config(experiment)

    assert "typo_key" in str(excinfo.value)


def test_model_vocab_must_match_data_vocab(smoke_config_path: Path) -> None:
    with pytest.raises(ValidationError, match="must equal"):
        load_experiment_config(smoke_config_path, ["model.vocab_size=16"])


def test_seq_len_must_fit_model(smoke_config_path: Path) -> None:
    with pytest.raises(ValidationError, match="must be >="):
        load_experiment_config(smoke_config_path, ["model.max_seq_len=16"])


def test_heads_must_divide_d_model(smoke_config_path: Path) -> None:
    with pytest.raises(ValidationError, match="divisible"):
        load_experiment_config(smoke_config_path, ["model.n_heads=3"])


def test_measurement_class_bounds_enforced(smoke_config_path: Path) -> None:
    assert (
        load_experiment_config(
            smoke_config_path, ["method.measurement_class=2"]
        ).method.measurement_class
        == 2
    )

    with pytest.raises(ValidationError):
        load_experiment_config(smoke_config_path, ["method.measurement_class=6"])
    with pytest.raises(ValidationError):
        load_experiment_config(smoke_config_path, ["method.measurement_class=0"])


def test_negative_seed_rejected(smoke_config_path: Path) -> None:
    with pytest.raises(ValidationError):
        load_experiment_config(smoke_config_path, ["experiment.seed=-1"])


def test_unknown_group_entry_fails_composition(smoke_config_path: Path) -> None:
    with pytest.raises(Exception, match=r"Could not find|In 'smoke'"):
        load_experiment_config(smoke_config_path, ["method=does-not-exist"])


def test_config_layer_permits_bits_with_method_none(smoke_config_path: Path) -> None:
    """The config layer describes a plan; the manifest validator is what enforces class rules."""
    cfg = load_experiment_config(
        smoke_config_path, ["method.compression.method=none", "method.compression.bits=4"]
    )

    assert cfg.method.compression.bits == 4
    assert cfg.is_uncompressed
