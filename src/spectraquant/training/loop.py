"""The training loop for joint low-rank / low-bit preparation (Milestone 5, hypothesis H3).

This module replaces the bootstrap scaffold's ``NotImplementedError`` stub with the loop the
Milestone-5 ticket asks for: deterministic seeding, a real optimiser path, optional checkpoint/resume,
per-term loss diagnostics for the preparation objective, and a schema-validated run manifest.

Scope and substrate (``AGENTS.md`` sections 2.3 and 6)
------------------------------------------------------
The loop is *substrate-agnostic* — it takes any ``nn.Module``, any token-tensor split and any task
loss — but the only thing that may be **run locally** is the Tier-0 fixture: a tiny synthetic
character transformer on the deterministic LCG corpus. Tier 1+ training (WikiText-2, a pretrained LM,
QAT/LoRA arms) runs on the cloud notebook substrate and is not started by this module; nothing here
downloads a model or a dataset, and nothing here requires CUDA (a CUDA device would make the local
paths below produce different numerics, so they are refused at the boundary by
:mod:`spectraquant.training.low_rank` and by the CPU-only seeding contract).

Everything this loop reports is **measurement class 2** (``AGENTS.md`` section 5): the metrics are
float-execution numbers; the byte figure in the manifest is explicitly the class-1 *analytical*
estimate (``compression.bytes_source == "accounted"``), never a class-3 measured storage number.

Determinism contract
--------------------
``training/seeding.py`` seeds Python, NumPy and PyTorch and pins the CPU thread count to 1. This loop
additionally draws every training batch from its **own** :class:`torch.Generator` whose state is part
of the checkpoint, so (a) two runs at the same seed produce identical loss sequences and (b) a run
resumed from a checkpoint reproduces the remaining loss sequence *bit for bit*.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from spectraquant.config import ExperimentConfig
from spectraquant.paths import repo_root
from spectraquant.quantization import QuantSpec
from spectraquant.regularizers.spectral import PreparationObjective
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
from spectraquant.training.low_rank import (
    LowRankLinear,
    factor_storage_bytes,
    gather_factor_pairs,
    parameter_counts,
)
from spectraquant.training.seeding import SeedState, seed_everything
from spectraquant.training.smoke import cross_entropy_loss, loss_sequence_sha256

__all__ = [
    "CHECKPOINT_FORMAT",
    "LoopConfig",
    "SequenceData",
    "TrainingResult",
    "default_task_loss",
    "evaluate_language_model",
    "evaluate_sequences",
    "load_checkpoint",
    "save_checkpoint",
    "train_language_model",
]

_LOGGER_NAME = "spectraquant.training.loop"

#: Checkpoint payload revision; loaded checkpoints must carry exactly this string.
CHECKPOINT_FORMAT = "spectraquant-training-checkpoint-v1"

#: Fields of :class:`LoopConfig` that change the *numbers* a run produces. A resumed run must match
#: them; presentation-only fields (``log_every``, ``eval_every``, ``checkpoint_every``) may differ.
_NUMERIC_LOOP_FIELDS: tuple[str, ...] = (
    "batch_size",
    "lr",
    "weight_decay",
    "grad_clip",
    "seed",
    "threads",
)


# --------------------------------------------------------------------------------------
# Inputs, configuration, results
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SequenceData:
    """Token-tensor splits for the Tier-0 loop.

    Attributes:
        train: ``(n_train, seq_len + 1)`` int64; ``[:, :-1]`` are inputs, ``[:, 1:]`` targets.
        val: ``(n_val, seq_len + 1)`` int64, same width as ``train`` — the *dev* split. The Tier-0
            synthetic corpus has no separate test split (``docs/research/preregistration.md``
            section 3 defines one for Tier 1+), so a local fixture sweep observes train + dev only
            and never a test set.
        checksum: content checksum recorded in the manifest (``sha256:`` prefixed).
    """

    train: Tensor
    val: Tensor
    checksum: str = ""

    def __post_init__(self) -> None:
        for label, split in (("train", self.train), ("val", self.val)):
            if not isinstance(split, Tensor):
                raise TypeError(f"{label} must be a torch.Tensor, got {type(split).__name__}")
            if split.ndim != 2:
                raise ValueError(f"{label} must be 2-D (sequences, seq_len + 1)")
            if split.shape[1] < 2:
                raise ValueError(f"{label} needs at least two tokens per sequence")
        if self.train.shape[1] != self.val.shape[1]:
            raise ValueError(
                f"train/val widths differ: {self.train.shape[1]} vs {self.val.shape[1]}"
            )

    @property
    def seq_len(self) -> int:
        """Number of predicted positions per sequence."""
        return int(self.train.shape[1]) - 1

    @classmethod
    def from_corpus(cls, corpus: Any) -> SequenceData:
        """Build from a :class:`spectraquant.data.synthetic.SyntheticCorpus` (duck-typed)."""
        return cls(train=corpus.train, val=corpus.val, checksum=str(corpus.checksum))


@dataclass(frozen=True)
class LoopConfig:
    """Optimisation knobs and reporting cadence for :func:`train_language_model`.

    Args:
        batch_size: sequences per step.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        grad_clip: global-norm clip (``None`` disables it).
        log_every: structured-log cadence in steps.
        seed: master seed; seeds every RNG and the batch generator.
        threads: ``torch.set_num_threads`` value (1 keeps float reductions bit-reproducible).
        checkpoint_every: write an intermediate checkpoint every N steps (``None`` disables
            checkpointing); a final ``final.pt`` is written whenever this is not ``None``.
        eval_every: evaluate the dev split every N steps (``None`` disables intermediate evaluation).
        warmup_steps: linear learning-rate warmup from ``lr / (warmup_steps + 1)`` to ``lr`` over the
            first ``warmup_steps`` steps (0 disables it). Applied through a ``LambdaLR`` so the
            checkpoint's optimiser state carries it and a resume continues the same schedule.

    Raises:
        ValueError: on a non-positive step/batch/seed combination or a negative rate.
    """

    batch_size: int = 16
    lr: float = 3e-3
    weight_decay: float = 0.0
    grad_clip: float | None = 1.0
    log_every: int = 20
    seed: int = 0
    threads: int = 1
    checkpoint_every: int | None = None
    eval_every: int | None = None
    warmup_steps: int = 0

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if not math.isfinite(self.lr) or self.lr <= 0.0:
            raise ValueError(f"lr must be finite and > 0, got {self.lr!r}")
        if self.weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if self.grad_clip is not None and self.grad_clip <= 0.0:
            raise ValueError(f"grad_clip must be > 0 or None, got {self.grad_clip!r}")
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")
        if self.threads < 1:
            raise ValueError(f"threads must be >= 1, got {self.threads}")
        for label in ("log_every", "checkpoint_every", "eval_every"):
            value = getattr(self, label)
            if value is not None and value < 1:
                raise ValueError(f"{label} must be >= 1 or None, got {value!r}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")

    @classmethod
    def from_experiment(cls, cfg: ExperimentConfig, **overrides: Any) -> LoopConfig:
        """Build from a validated experiment config's ``experiment.training`` block and seed."""
        training = cfg.experiment.training
        payload: dict[str, Any] = {
            "batch_size": training.batch_size,
            "lr": training.lr,
            "weight_decay": training.weight_decay,
            "grad_clip": training.grad_clip,
            "log_every": training.log_every,
            "seed": cfg.experiment.seed,
        }
        payload.update(overrides)
        return cls(**payload)

    def numeric_fields(self) -> dict[str, Any]:
        """The subset of fields that changes the loss sequence (checkpoint compatibility)."""
        return {name: getattr(self, name) for name in _NUMERIC_LOOP_FIELDS}


@dataclass(frozen=True)
class TrainingResult:
    """Everything a loop run produced.

    Attributes:
        losses: the optimised (total = task + penalty) loss per completed step.
        terms: per-step component diagnostics, including ``task`` and every objective term.
        val_losses: ``(step, dev cross-entropy)`` pairs, in evaluation order.
        checkpoint_paths: checkpoint files written by this run, in order.
        manifest: the manifest that was built (and written when ``write`` is true).
        manifest_path: the file the manifest was written to, or ``None``.
        steps_completed: total optimiser steps the run has executed from its start.
        wall_time_s: wall-clock seconds spent inside the training loop and the final evaluation.
        seed_state: the seeding configuration that was applied.
        metadata: caller-supplied provenance attached to the checkpoint and the manifest.
    """

    losses: tuple[float, ...]
    terms: tuple[dict[str, float], ...]
    val_losses: tuple[tuple[int, float], ...]
    checkpoint_paths: tuple[str, ...]
    manifest: RunManifest
    manifest_path: str | None
    steps_completed: int
    wall_time_s: float
    seed_state: SeedState
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def loss_sequence_sha256(self) -> str:
        """Digest of the loss sequence, as in the smoke fixture (bit-for-bit comparability)."""
        return loss_sequence_sha256(list(self.losses))

    def term_means(self) -> dict[str, float]:
        """Mean of every recorded diagnostic across steps (empty when no step ran)."""
        return _term_means(self.terms)


#: Signature of a task-loss function: ``(model, inputs, targets) -> scalar``.
TaskLoss = Callable[[nn.Module, Tensor, Tensor], Tensor]


def logits_of(output: Any) -> Tensor:
    """Return the logits tensor from a model output.

    Two shapes are supported: a model that returns the logits directly (the Tier-0 fixture) and a
    Hugging Face causal LM, which returns an output object carrying ``.logits``. Both reach the same
    loss, so the loop does not need two task losses.
    """
    logits = getattr(output, "logits", output)
    if not isinstance(logits, Tensor):
        raise TypeError(
            f"model returned {type(output).__name__} with no `.logits` tensor: the task loss cannot "
            "be computed"
        )
    return logits


def default_task_loss(model: nn.Module, inputs: Tensor, targets: Tensor) -> Tensor:
    """Next-token cross-entropy on a ``(batch, seq_len)`` token batch (the Tier-0 task)."""
    return cross_entropy_loss(logits_of(model(inputs)), targets)


# --------------------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------------------
def save_checkpoint(
    path: Path | str,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    step: int,
    loop: LoopConfig,
    losses: Sequence[float],
    terms: Sequence[Mapping[str, float]],
    val_losses: Sequence[tuple[int, float]],
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write a resumable checkpoint (model, optimiser, batch-generator state, history, config).

    The checkpoint is self-describing: :func:`load_checkpoint` refuses a payload from another format
    revision, another seed, or a numerically incompatible :class:`LoopConfig`.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "step": int(step),
        "seed": int(loop.seed),
        "loop": asdict(loop),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "generator": generator.get_state(),
        "losses": [float(value) for value in losses],
        "terms": [dict(record) for record in terms],
        "val_losses": [[int(step_i), float(value)] for step_i, value in val_losses],
        "metadata": json.loads(json.dumps(dict(metadata or {}), default=str)),
    }
    torch.save(payload, target)
    return target


def load_checkpoint(
    path: Path | str,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    loop: LoopConfig,
) -> dict[str, Any]:
    """Restore ``model``/``optimizer``/``generator`` in place and return the checkpoint payload.

    Args:
        path: a checkpoint file written by :func:`save_checkpoint`.
        model: the same architecture that was checkpointed (a state-dict mismatch raises).
        optimizer: the same optimiser type and parameter order.
        generator: the batch generator whose state is restored (bit-identical batch order).
        loop: the loop configuration of the resumed run.

    Returns:
        The checkpoint payload (``step``, histories, metadata).

    Raises:
        ValueError: on an unknown format revision, a seed mismatch, or a numerically incompatible
            loop configuration — each of which would silently break the resume-reproducibility claim.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"{path} is not a {CHECKPOINT_FORMAT} checkpoint (got {payload!r:.80})"
            if isinstance(payload, dict)
            else f"{path} is not a checkpoint mapping"
        )
    if int(payload["seed"]) != loop.seed:
        raise ValueError(
            f"checkpoint seed {payload['seed']} != loop seed {loop.seed}; a resumed run must reuse "
            "the seed of the run it continues"
        )
    stored = payload.get("loop", {})
    mismatch = {
        name: (stored.get(name), getattr(loop, name))
        for name in _NUMERIC_LOOP_FIELDS
        if stored.get(name) != getattr(loop, name)
    }
    if mismatch:
        raise ValueError(f"checkpoint loop configuration differs from this run: {mismatch}")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    generator.set_state(payload["generator"])
    return payload


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------
def evaluate_sequences(model: nn.Module, sequences: Tensor, *, batch_size: int) -> float:
    """Mean next-token cross-entropy over ``sequences`` in eval mode (no grad)."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_positions = 0
    with torch.no_grad():
        for start in range(0, int(sequences.shape[0]), batch_size):
            window = sequences[start : start + batch_size]
            inputs, targets = window[:, :-1], window[:, 1:]
            loss = cross_entropy_loss(logits_of(model(inputs)), targets)
            positions = int(targets.numel())
            total_loss += float(loss) * positions
            total_positions += positions
    model.train(was_training)
    return total_loss / max(total_positions, 1)


def evaluate_language_model(
    model: nn.Module,
    *,
    dataset_id: str,
    split: str = "validation",
) -> dict[str, float]:
    """Evaluate a model on a language-modelling split (Tier 2+, Milestone 6).

    Raises:
        NotImplementedError: always. Loading and tokenising an evaluation corpus is the Milestone-6
            evaluation slice (``configs/tier2``, ``docs/protocols/eval-protocol.md``) and needs
            dataset access this module deliberately does not have; the Tier-0 loop evaluates through
            its own held-out token tensors in :meth:`LoopConfig.eval_every` /
            :func:`train_language_model`. This entry point exists so the interface is stable, and it
            fails loudly rather than returning an unmeasured number.
    """
    raise NotImplementedError(
        "evaluate_language_model is a Milestone 6 (evaluation stream) deliverable: it needs the "
        f"Tier-2+ corpus loaders for dataset {dataset_id!r} split {split!r}, which do not exist "
        "locally. The Tier-0 loop evaluates its own dev token tensors."
    )


# --------------------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------------------
def _backend_string(model: nn.Module) -> str:
    """Manifest backend string derived from the actual device and parameter dtype."""
    parameter = next(model.parameters())
    precision = {
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
    }.get(parameter.dtype, str(parameter.dtype).removeprefix("torch."))
    return f"torch-{parameter.device.type}-{precision}"


def _ranks_of(model: nn.Module) -> list[int] | None:
    """Distinct ranks of the model's factorized layers, or ``None`` when it has none."""
    ranks = sorted({module.rank for module in model.modules() if isinstance(module, LowRankLinear)})
    return ranks or None


def train_language_model(
    model: nn.Module,
    *,
    method: str,
    steps: int,
    output_dir: Path | str,
    config: Mapping[str, Any] | None = None,
    data: SequenceData | None = None,
    objective: PreparationObjective | None = None,
    references: Mapping[str, Tensor] | None = None,
    task_loss: TaskLoss | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    loop: LoopConfig | None = None,
    spec: QuantSpec | None = None,
    measurement_class: int | None = 2,
    exclusions: Sequence[str] = (),
    resume_from: Path | str | None = None,
    write: bool = True,
    extra_metrics: Mapping[str, Any] | None = None,
    verbose: bool = False,
) -> TrainingResult:
    """Train ``model`` on ``data`` for ``steps`` optimiser steps, with the preparation objective.

    The optimised loss is ``task_loss(model, inputs, targets) + objective(factors)``; the objective
    is optional, so the same loop runs the regularizer-enabled and regularizer-disabled arms.

    Args:
        model: the (usually factorized) model. It is trained in place.
        method: compression method recorded in the manifest (e.g. ``"spectraquant"``, ``"none"``).
        steps: total optimiser steps the run should reach (counting any resumed steps).
        output_dir: directory for checkpoints and the run manifest.
        config: the resolved experiment config, stored verbatim in the manifest's
            ``resolved_config`` (provenance only; this loop reads no field from it).
        data: train/dev token tensors; required.
        objective: the preparation objective; ``None`` disables every penalty term.
        references: ``name -> W_ref`` dense reference weights for the objective's factorization term.
        task_loss: ``(model, inputs, targets) -> scalar``; defaults to next-token cross-entropy.
        optimizer: pre-built optimiser; defaults to AdamW over ``model.parameters()``.
        loop: optimisation knobs; defaults to :class:`LoopConfig` with ``seed=0``.
        spec: the target quantizer, used for the class-1 factor byte accounting in the manifest.
            Defaults to ``objective.spec`` when an objective is given.
        measurement_class: class of the reported quality numbers (``2`` = fake-quantization-quality
            execution, the Tier-0 default); ``None`` is only legal with ``method == "none"``.
        exclusions: layer-name patterns excluded from compression (manifest provenance).
        resume_from: checkpoint written by :func:`save_checkpoint`; the run continues from it.
        write: validate and write the manifest (``False`` keeps it in memory only).
        extra_metrics: additional JSON-scalar metrics merged into the manifest.
        verbose: emit per-step structured log events.

    Returns:
        A :class:`TrainingResult` with the loss sequences, per-term diagnostics and the manifest.

    Raises:
        ValueError: on missing data, a non-positive ``steps``, a resume checkpoint inconsistent with
            this run, or a ``measurement_class``/``method`` combination the manifest rejects.
    """
    if verbose:
        configure_logging()
    logger = get_logger(_LOGGER_NAME)
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError(f"steps must be an int >= 1, got {steps!r}")
    if data is None:
        raise ValueError(
            "data is required: the loop needs train/dev token tensors (see SequenceData); no "
            "dataset is loaded implicitly, because the Tier-1+ corpora belong to the cloud substrate"
        )
    if objective is not None and spec is None:
        spec = objective.spec
    if measurement_class is None and method != "none":
        raise ValueError(
            f"measurement_class must be 1-5 when method={method!r} (AGENTS.md section 5); only "
            "method='none' may omit it"
        )

    active_loop = loop or LoopConfig()
    seed_state = seed_everything(active_loop.seed, deterministic=True, threads=active_loop.threads)
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

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_path / "checkpoints"

    # The batch generator is separate from the global RNG so that its state is exactly restorable.
    generator = torch.Generator().manual_seed(active_loop.seed + 1)
    active_task_loss: TaskLoss = task_loss or default_task_loss
    active_optimizer = optimizer or torch.optim.AdamW(
        model.parameters(), lr=active_loop.lr, weight_decay=active_loop.weight_decay
    )

    losses: list[float] = []
    terms: list[dict[str, float]] = []
    val_losses: list[tuple[int, float]] = []
    checkpoint_paths: list[Path] = []
    metadata = dict(extra_metrics or {})
    start_step = 0

    if resume_from is not None:
        payload = load_checkpoint(
            resume_from,
            model=model,
            optimizer=active_optimizer,
            generator=generator,
            loop=active_loop,
        )
        start_step = int(payload["step"])
        losses = [float(value) for value in payload.get("losses", [])]
        terms = [dict(record) for record in payload.get("terms", [])]
        val_losses = [(int(step), float(value)) for step, value in payload.get("val_losses", [])]
        metadata.update(dict(payload.get("metadata", {})))
        log_event(logger, 20, "resumed", fields={"from": str(resume_from), "step": start_step})
        if start_step >= steps:
            raise ValueError(
                f"checkpoint is already at step {start_step}, which is >= the requested {steps}; "
                "nothing to resume"
            )

    # Warmup is a pure function of the step index, so a resume reproduces the same schedule by
    # starting the scheduler at the resumed step - no scheduler state needs to be checkpointed.
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None
    if active_loop.warmup_steps > 0:
        warmup = int(active_loop.warmup_steps)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            active_optimizer,
            lr_lambda=lambda step: min(1.0, (step + 1) / (warmup + 1)),
            last_epoch=start_step - 1,
        )

    n_train = int(data.train.shape[0])
    started = time.perf_counter()
    model.train()
    for step in range(start_step, steps):
        batch_idx = torch.randint(0, n_train, (active_loop.batch_size,), generator=generator)
        window = data.train[batch_idx]
        inputs, targets = window[:, :-1], window[:, 1:]

        active_optimizer.zero_grad(set_to_none=True)
        task = active_task_loss(model, inputs, targets)
        record: dict[str, float] = {}
        if objective is not None:
            evaluated = objective.terms(gather_factor_pairs(model), references=references)
            record = evaluated.to_dict()
            total = task + evaluated.total
        else:
            record = {"factorization": 0.0, "rounding": 0.0, "residual": 0.0, "spectrum": 0.0}
            total = task
        record["task"] = float(task.detach())
        record["total"] = float(total.detach())
        total.backward()
        if active_loop.grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), active_loop.grad_clip)
        active_optimizer.step()
        if scheduler is not None:
            scheduler.step()

        losses.append(record["total"])
        terms.append(record)
        if verbose and (step + 1) % active_loop.log_every == 0:
            log_event(
                logger,
                20,
                "step",
                fields={"step": step + 1, "loss": round(record["total"], 6)},
            )
        if active_loop.eval_every is not None and (step + 1) % active_loop.eval_every == 0:
            val_losses.append(
                (step + 1, evaluate_sequences(model, data.val, batch_size=active_loop.batch_size))
            )
        if (
            active_loop.checkpoint_every is not None
            and (step + 1) % active_loop.checkpoint_every == 0
        ):
            checkpoint_paths.append(
                save_checkpoint(
                    checkpoint_dir / f"step-{step + 1:07d}.pt",
                    model=model,
                    optimizer=active_optimizer,
                    generator=generator,
                    step=step + 1,
                    loop=active_loop,
                    losses=losses,
                    terms=terms,
                    val_losses=val_losses,
                    metadata=metadata,
                )
            )

    final_val_loss = evaluate_sequences(model, data.val, batch_size=active_loop.batch_size)
    if not val_losses or val_losses[-1][0] != steps:
        val_losses.append((steps, final_val_loss))
    wall_time_s = time.perf_counter() - started
    if active_loop.checkpoint_every is not None:
        checkpoint_paths.append(
            save_checkpoint(
                checkpoint_dir / "final.pt",
                model=model,
                optimizer=active_optimizer,
                generator=generator,
                step=steps,
                loop=active_loop,
                losses=losses,
                terms=terms,
                val_losses=val_losses,
                metadata=metadata,
            )
        )

    peak_mem_mb = process_peak_rss_mb()
    counts = parameter_counts(model)
    accounted = None
    if spec is not None and counts["factorized"] > 0:
        accounted = factor_storage_bytes(model, spec)

    term_means = _term_means(terms)
    metric_payload: dict[str, Any] = {
        "loss_initial": round(losses[start_step], 9),
        "loss_final": round(losses[-1], 9),
        "loss_reduction": round(losses[start_step] - losses[-1], 9),
        "loss_sequence_sha256": loss_sequence_sha256(losses),
        "val_loss_final": round(final_val_loss, 9),
        "val_loss_evaluations": [[int(step), round(value, 9)] for step, value in val_losses],
        "steps": steps,
        "resumed_from_step": start_step,
        "batch_size": active_loop.batch_size,
        "learning_rate": active_loop.lr,
        "seq_len": data.seq_len,
        "parameter_count": counts["total"],
        "factorized_parameter_count": counts["factorized"],
        "accounted_bytes": accounted,
        "regularizer_name": objective.name if objective is not None else "none",
        "regularizer_is_noop": bool(objective.is_noop()) if objective is not None else True,
        "loss_sequence": [round(value, 9) for value in losses],
    }
    for key, value in term_means.items():
        metric_payload[f"term_mean_{key}"] = round(value, 12)
    if objective is not None:
        for key, value in objective.weights().items():
            metric_payload[key] = value
    for key, value in metadata.items():
        if isinstance(value, (str, bool, int, float)) or value is None:
            metric_payload[key] = value
        else:  # pragma: no cover - callers pass JSON scalars
            metric_payload[key] = str(value)

    resolved = dict(config or {})
    data_block = resolved.get("data")
    dataset_name = (
        str(data_block.get("name")) if isinstance(data_block, Mapping) else "synthetic-lcg"
    )
    git = git_info(repo_root())
    manifest = RunManifest(
        run_id=build_run_id(f"train-{method}", git_commit=git.commit),
        timestamp_utc=utc_timestamp(),
        git_commit=git.commit,
        git_dirty=git.dirty,
        config_path=str(resolved.get("config_path") or "<in-memory>"),
        resolved_config=resolved,
        model_id=None,
        model_revision=None,
        dataset_ids=[dataset_name],
        dataset_revisions=[None],
        dataset_checksums=[data.checksum] if data.checksum else [],
        split="train",
        seed=active_loop.seed,
        hardware=HardwareBlock(**collect_hardware()),
        software=collect_software(),
        compression=CompressionBlock(
            method=method,
            ranks=_ranks_of(model),
            bits=None if spec is None else spec.bits,
            group_size=None if spec is None else spec.group_size,
            exclusions=list(exclusions),
            accounted_bytes=accounted,
            bytes_source="accounted" if accounted is not None else None,
        ),
        theoretical_bits=None,
        packed_bytes=None,
        runtime_backend=_backend_string(model),
        measurement_class=measurement_class,
        training=TrainingBlock(
            steps=steps,
            tokens=steps * active_loop.batch_size * data.seq_len,
            wall_time_s=round(wall_time_s, 6),
            peak_mem_mb=None if peak_mem_mb is None else round(peak_mem_mb, 3),
        ),
        metrics=metric_payload,
        status="success",
        failure_reason=None,
        log_path=None,
        artifact_paths=[str(path) for path in checkpoint_paths],
    )

    manifest_file: Path | None = None
    if write:
        manifest_file = write_manifest(manifest, manifest_path(manifest.run_id, output_path))
        log_event(logger, 20, "manifest_written", fields={"path": str(manifest_file)})

    return TrainingResult(
        losses=tuple(losses),
        terms=tuple(terms),
        val_losses=tuple(val_losses),
        checkpoint_paths=tuple(str(path) for path in checkpoint_paths),
        manifest=manifest,
        manifest_path=str(manifest_file) if manifest_file is not None else None,
        steps_completed=steps,
        wall_time_s=wall_time_s,
        seed_state=seed_state,
        metadata=metadata,
    )


def _term_means(terms: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Mean of every recorded diagnostic across steps (empty when no step ran)."""
    if not terms:
        return {}
    keys = sorted({key for record in terms for key in record})
    return {key: sum(float(record.get(key, 0.0)) for record in terms) / len(terms) for key in keys}
