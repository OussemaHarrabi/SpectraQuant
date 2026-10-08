"""The tiny end-to-end smoke experiment: the local CI smoke fixture (Tier 0).

Purpose: prove that the whole harness — config composition, deterministic seeding, data, training,
metric capture, manifest construction, schema validation — works end to end on CPU, fast, and
bit-for-bit reproducibly. It is *not* a research result: the model is a ~0.1 M-parameter toy
trained on a synthetic recurrence, and the manifest therefore carries ``measurement_class: null``
(no compression is applied, so no class 1-5 number is claimed).

Determinism contract: two runs with the same seed and config produce identical loss sequences.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor, nn

from spectraquant.config import ExperimentConfig
from spectraquant.data.synthetic import SyntheticCorpus, build_corpus
from spectraquant.paths import repo_root
from spectraquant.reporting.environment import (
    collect_hardware,
    collect_software,
    process_peak_rss_mb,
)
from spectraquant.reporting.gitinfo import git_info
from spectraquant.reporting.logging import configure_logging, get_logger, log_event
from spectraquant.reporting.manifests import (
    CompressionBlock,
    HardwareBlock,
    RunManifest,
    TrainingBlock,
    build_run_id,
    manifest_path,
    utc_timestamp,
    write_manifest,
)
from spectraquant.training.seeding import seed_everything
from spectraquant.training.tiny_lm import (
    MAX_SMOKE_PARAMS,
    TinyCharTransformer,
    count_parameters,
    initial_state_checksum,
)

__all__ = ["SmokeResult", "cross_entropy_loss", "loss_sequence_sha256", "run_smoke_experiment"]

_LOGGER_NAME = "spectraquant.training.smoke"


def cross_entropy_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Mean token-level cross-entropy.

    Args:
        logits: ``(B, T, V)`` float tensor of unnormalised scores.
        targets: ``(B, T)`` int64 tensor of target token ids.

    Returns:
        Scalar loss tensor (natural log, mean over ``B*T`` positions).
    """
    vocab = logits.shape[-1]
    return nn.functional.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1))


@dataclass(frozen=True)
class SmokeResult:
    """Everything a smoke run produced, including the manifest that was written."""

    run_id: str
    loss_sequence: list[float]
    val_loss: float
    parameter_count: int
    tokens_seen: int
    wall_time_s: float
    peak_mem_mb: float | None
    corpus: SyntheticCorpus
    manifest: RunManifest
    manifest_path: Path | None = None
    log_path: Path | None = None
    metrics: dict[str, object] = field(default_factory=dict)

    @property
    def initial_loss(self) -> float:
        """Loss at the first optimiser step."""
        return self.loss_sequence[0]

    @property
    def final_loss(self) -> float:
        """Loss at the last optimiser step."""
        return self.loss_sequence[-1]

    @property
    def loss_sequence_sha256(self) -> str:
        """Hash of the rounded loss sequence; the cheap determinism witness."""
        return loss_sequence_sha256(self.loss_sequence)


def loss_sequence_sha256(losses: list[float]) -> str:
    digest = hashlib.sha256()
    for loss in losses:
        digest.update(f"{loss:.9f}\n".encode("ascii"))
    return f"sha256:{digest.hexdigest()}"


def _runtime_backend(model: nn.Module) -> str:
    """Backend string for the manifest, derived from the actual device and parameter dtype."""
    device = next(model.parameters()).device.type
    dtype = next(model.parameters()).dtype
    precision = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}.get(
        dtype, str(dtype).removeprefix("torch.")
    )
    return f"torch-{device}-{precision}"


def run_smoke_experiment(
    cfg: ExperimentConfig,
    *,
    results_dir: Path | str | None = None,
    filename: str | None = None,
    write: bool = True,
    verbose: bool = False,
) -> SmokeResult:
    """Run the tiny end-to-end experiment described by ``cfg``.

    Args:
        cfg: validated experiment config.
        results_dir: where the manifest is written. When given, the file is named
            ``<run_id>.manifest.json`` (a real run record); when omitted, the stable sample file
            ``<repo>/<experiment.output.results_dir>/<experiment.output.filename>`` is written.
        filename: explicit manifest file name, overriding the default naming described above.
        write: when ``False`` the manifest is validated but not written to disk.
        verbose: emit per-step structured log events.

    Returns:
        A :class:`SmokeResult` carrying the loss sequence, metrics and the manifest.

    Raises:
        ValueError: the model exceeds :data:`MAX_SMOKE_PARAMS` or the data config is inconsistent.
    """
    if verbose:
        configure_logging()
    logger = get_logger(_LOGGER_NAME)

    seed = cfg.experiment.seed
    seed_state = seed_everything(seed, deterministic=True, threads=1)
    log_event(
        logger,
        20,
        "seed",
        fields={
            "seed": seed_state.seed,
            "threads": seed_state.torch_threads,
            "deterministic_algorithms": seed_state.deterministic_algorithms,
        },
    )

    corpus = build_corpus(cfg.data, seed=seed)
    model = TinyCharTransformer(cfg.model)
    parameter_count = count_parameters(model)
    if parameter_count > MAX_SMOKE_PARAMS:
        raise ValueError(
            f"model has {parameter_count} parameters, above the smoke-fixture cap "
            f"{MAX_SMOKE_PARAMS}; shrink configs/model/*.yaml, or run a real experiment on the cloud "
            "substrate (AGENTS.md section 2b)"
        )
    model_revision = initial_state_checksum(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.experiment.training.lr,
        weight_decay=cfg.experiment.training.weight_decay,
    )

    training = cfg.experiment.training
    n_train = int(corpus.train.shape[0])
    loss_sequence: list[float] = []

    started = time.perf_counter()
    model.train()
    for step in range(training.steps):
        batch_idx = torch.randint(0, n_train, (training.batch_size,))
        window = corpus.train[batch_idx]
        inputs, targets = window[:, :-1], window[:, 1:]

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = cross_entropy_loss(logits, targets)
        loss.backward()
        if training.grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), training.grad_clip)
        optimizer.step()

        step_loss = float(loss.detach())
        loss_sequence.append(step_loss)
        if verbose and (step + 1) % training.log_every == 0:
            log_event(
                logger,
                20,
                "step",
                fields={"step": step + 1, "loss": round(step_loss, 6)},
            )

    val_loss = _evaluate(model, corpus.val, batch_size=training.batch_size)
    wall_time_s = time.perf_counter() - started
    peak_mem_mb = process_peak_rss_mb()
    tokens_seen = training.steps * training.batch_size * corpus.seq_len

    metrics: dict[str, object] = {
        "train_loss_initial": round(loss_sequence[0], 9),
        "train_loss_final": round(loss_sequence[-1], 9),
        "train_loss_reduction": round(loss_sequence[0] - loss_sequence[-1], 9),
        "val_loss": round(val_loss, 9),
        "uniform_prior_entropy": round(
            float(torch.log(torch.tensor(float(cfg.data.vocab_size)))), 9
        ),
        "loss_sequence": [round(loss, 9) for loss in loss_sequence],
        "loss_sequence_sha256": loss_sequence_sha256(loss_sequence),
        "parameter_count": parameter_count,
        "batch_size": training.batch_size,
        "learning_rate": training.lr,
        "steps": training.steps,
        "seq_len": corpus.seq_len,
    }

    git = git_info(repo_root())
    manifest = RunManifest(
        run_id=build_run_id(cfg.experiment.name, git_commit=git.commit),
        timestamp_utc=utc_timestamp(),
        git_commit=git.commit,
        git_dirty=git.dirty,
        config_path=str(cfg.config_path or ""),
        resolved_config=_resolved_config(cfg),
        model_id=None,
        model_revision=model_revision,
        dataset_ids=[cfg.data.name],
        dataset_revisions=[None],
        dataset_checksums=[corpus.checksum],
        split="train",
        seed=seed,
        hardware=HardwareBlock(**collect_hardware()),
        software=collect_software(),
        compression=CompressionBlock(
            method=cfg.method.compression.method,
            ranks=cfg.method.compression.ranks,
            bits=cfg.method.compression.bits,
            group_size=cfg.method.compression.group_size,
            exclusions=list(cfg.method.compression.exclusions),
        ),
        theoretical_bits=None,
        packed_bytes=None,
        runtime_backend=_runtime_backend(model),
        measurement_class=cfg.method.measurement_class,
        training=TrainingBlock(
            steps=training.steps,
            tokens=tokens_seen,
            wall_time_s=round(wall_time_s, 6),
            peak_mem_mb=None if peak_mem_mb is None else round(peak_mem_mb, 3),
        ),
        metrics=metrics,
        status="success",
        failure_reason=None,
        log_path=None,
        artifact_paths=[],
    )

    manifest_file: Path | None = None
    if write:
        if results_dir is not None:
            target_dir = Path(results_dir)
            target = (
                Path(filename)
                if filename is not None
                else manifest_path(manifest.run_id, target_dir)
            )
        else:
            target_dir = repo_root() / cfg.experiment.output.results_dir
            target = target_dir / (filename or cfg.experiment.output.filename)
        manifest_file = write_manifest(manifest, target)
        log_event(logger, 20, "manifest_written", fields={"path": str(manifest_file)})

    log_event(
        logger,
        20,
        "smoke_complete",
        fields={
            "run_id": manifest.run_id,
            "steps": training.steps,
            "val_loss": round(val_loss, 6),
            "wall_time_s": round(wall_time_s, 3),
        },
    )

    return SmokeResult(
        run_id=manifest.run_id,
        loss_sequence=loss_sequence,
        val_loss=val_loss,
        parameter_count=parameter_count,
        tokens_seen=tokens_seen,
        wall_time_s=wall_time_s,
        peak_mem_mb=peak_mem_mb,
        corpus=corpus,
        manifest=manifest,
        manifest_path=manifest_file,
        metrics=metrics,
    )


def _evaluate(model: nn.Module, sequences: Tensor, *, batch_size: int) -> float:
    """Mean cross-entropy over ``sequences`` in eval mode (no grad, deterministic)."""
    model.eval()
    total_loss = 0.0
    total_positions = 0
    with torch.no_grad():
        for start in range(0, int(sequences.shape[0]), batch_size):
            window = sequences[start : start + batch_size]
            inputs, targets = window[:, :-1], window[:, 1:]
            logits = model(inputs)
            loss = cross_entropy_loss(logits, targets)
            total_loss += float(loss) * int(targets.numel())
            total_positions += int(targets.numel())
    model.train()
    if total_positions == 0:  # pragma: no cover - configs forbid empty splits
        raise ValueError("validation split is empty")
    return total_loss / total_positions


def _resolved_config(cfg: ExperimentConfig) -> dict[str, object]:
    """Resolved config as stored in the manifest (config_path excluded: it is a separate field)."""
    dumped = cfg.model_dump(mode="json")
    dumped.pop("config_path", None)
    return dict(dumped)
