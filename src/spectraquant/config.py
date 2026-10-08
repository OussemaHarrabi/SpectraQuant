"""Typed experiment configuration (Hydra composition + Pydantic validation).

Hydra composes ``configs/experiment/*.yaml`` with the ``model``/``data``/``method`` config groups;
Pydantic then validates the *resolved* tree so that a malformed experiment fails before any tensor
is allocated. ``extra="forbid"`` is deliberate: a typo in a YAML key is an error, never a silently
ignored override.

Usage::

    cfg = load_experiment_config("configs/experiment/smoke.yaml")
    cfg.experiment.seed
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, model_validator

from spectraquant.paths import configs_dir

__all__ = [
    "CompressionSpec",
    "DataConfig",
    "ExperimentConfig",
    "ExperimentMeta",
    "MethodConfig",
    "ModelConfig",
    "OutputConfig",
    "TrainingConfig",
    "compose_raw_config",
    "load_experiment_config",
    "resolve_config_target",
]


class StrictModel(BaseModel):
    """Base model: unknown keys are errors, instances are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(StrictModel):
    """Tiny char-level transformer geometry (CI smoke fixture, Tier 0)."""

    name: str = Field(min_length=1)
    vocab_size: int = Field(ge=2)
    d_model: int = Field(ge=8)
    n_heads: int = Field(ge=1)
    n_layers: int = Field(ge=1)
    max_seq_len: int = Field(ge=2)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def _heads_divide_model(self) -> ModelConfig:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        return self


class DataConfig(StrictModel):
    """Deterministic synthetic sequence data (no external download, no license surface)."""

    name: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    vocab_size: int = Field(ge=2)
    seq_len: int = Field(ge=2)
    n_train_sequences: int = Field(ge=1)
    n_val_sequences: int = Field(ge=1)
    generator: str = Field(min_length=1)
    lcg_a: int = Field(ge=1)
    lcg_b: int = Field(ge=0)
    train_start: int = Field(default=0, ge=0)
    val_start: int = Field(default=97, ge=0)


class CompressionSpec(StrictModel):
    """What compression (if any) the run applies. ``method="none"`` means uncompressed baseline."""

    method: str = Field(min_length=1)
    ranks: list[int] | None = None
    bits: int | None = Field(default=None, ge=2, le=32)
    group_size: int | None = Field(default=None, ge=1)
    exclusions: list[str] = Field(default_factory=list)


class MethodConfig(StrictModel):
    """Compression method plus the measurement class its numbers belong to."""

    name: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    compression: CompressionSpec
    measurement_class: int | None = Field(default=None, ge=1, le=5)


class TrainingConfig(StrictModel):
    """Optimisation budget for the smoke run."""

    steps: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    lr: float = Field(gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    grad_clip: float | None = Field(default=None, gt=0.0)
    log_every: int = Field(default=10, ge=1)


class OutputConfig(StrictModel):
    """Where the run manifest is written."""

    results_dir: str = Field(min_length=1)
    filename: str = Field(default="smoke-manifest.json", min_length=1)


class ExperimentMeta(StrictModel):
    """Experiment-level knobs: identity, seed, optimiser budget, outputs."""

    name: str = Field(min_length=1)
    seed: int = Field(ge=0)
    training: TrainingConfig
    output: OutputConfig


class ExperimentConfig(StrictModel):
    """The fully resolved experiment configuration."""

    model: ModelConfig
    data: DataConfig
    method: MethodConfig
    experiment: ExperimentMeta
    config_path: str | None = None

    @model_validator(mode="after")
    def _vocab_sizes_agree(self) -> ExperimentConfig:
        if self.model.vocab_size != self.data.vocab_size:
            raise ValueError(
                "model.vocab_size "
                f"({self.model.vocab_size}) must equal data.vocab_size ({self.data.vocab_size})"
            )
        if self.model.max_seq_len < self.data.seq_len:
            raise ValueError(
                f"model.max_seq_len ({self.model.max_seq_len}) must be >= data.seq_len "
                f"({self.data.seq_len})"
            )
        return self

    @property
    def is_uncompressed(self) -> bool:
        """True when the method declares no compression (``method="none"``)."""
        return self.method.compression.method == "none"


def resolve_config_target(config_path: str | Path) -> tuple[Path, str]:
    """Split a config file path into ``(config_dir, config_name)`` for Hydra.

    The last ``configs`` path segment is used as the composition root, so
    ``configs/experiment/smoke.yaml`` becomes ``(<repo>/configs, "experiment/smoke")``.
    """
    path = Path(config_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"experiment config not found: {path}")
    resolved = path.resolve()
    stem_parts = resolved.with_suffix("").parts
    segments = [i for i, part in enumerate(stem_parts) if part == "configs"]
    if not segments:
        return resolved.parent, resolved.stem
    split_at = segments[-1]
    return Path(*stem_parts[: split_at + 1]), "/".join(stem_parts[split_at + 1 :])


@contextmanager
def _hydra_context(config_dir: Path) -> Iterator[None]:
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        yield


def _candidate_config_paths(raw: Path) -> list[Path]:
    """Candidate locations for a config path: as given, then relative to ``<repo>/configs``."""
    if raw.is_absolute() or raw.is_file():
        return [raw]
    candidates = [raw]
    parts = raw.parts
    if parts and parts[0] == "configs":
        candidates.append(configs_dir().joinpath(*parts[1:]))
    else:
        candidates.append(configs_dir() / raw)
    return candidates


def compose_raw_config(
    config_path: str | Path,
    overrides: list[str] | None = None,
) -> tuple[dict[str, Any], Path]:
    """Compose the Hydra config for ``config_path`` and return plain-Python data.

    Returns the resolved config mapping (primitives only) and the resolved absolute path of the
    experiment file. A relative ``config_path`` that does not exist in the current directory is
    resolved against the repository ``configs`` directory, so the documented commands work from any
    working directory.
    """
    raw_path = Path(config_path).expanduser()
    for candidate in _candidate_config_paths(raw_path):
        if candidate.is_file():
            config_dir, config_name = resolve_config_target(candidate)
            break
    else:
        tried = ", ".join(str(candidate) for candidate in _candidate_config_paths(raw_path))
        raise FileNotFoundError(f"experiment config not found: {raw_path} (tried: {tried})")

    with _hydra_context(config_dir):
        composed: DictConfig = compose(
            config_name=config_name,
            overrides=list(overrides or []),
            return_hydra_config=False,
        )
    raw = OmegaConf.to_container(composed, resolve=True, throw_on_missing=True)
    if not isinstance(raw, dict):  # pragma: no cover - Hydra always returns a mapping here
        raise TypeError(f"composed config is not a mapping: {type(raw)!r}")
    return cast("dict[str, Any]", raw), candidate.resolve()


def load_experiment_config(
    config_path: str | Path,
    overrides: list[str] | None = None,
) -> ExperimentConfig:
    """Compose and validate an experiment config.

    Raises:
        FileNotFoundError: the experiment YAML does not exist.
        hydra.errors.OverrideParseException / ConfigCompositionException: bad overrides or a
            missing config group entry.
        pydantic.ValidationError: the resolved tree violates the typed schema.
    """
    raw, resolved_path = compose_raw_config(config_path, overrides)
    return ExperimentConfig.model_validate({**raw, "config_path": str(resolved_path)})
