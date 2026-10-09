"""``spectraquant run-plan`` — execute a frozen cloud plan's arms remotely.

A plan (``configs/{tier1,tier2,repro}/*.yaml``) is the frozen definition of a remote experiment: a
pinned model, pinned datasets, a compression grid, a method matrix and a cost envelope. This module
is the *remote entry point* that turns a plan into recorded measurements. It is deliberately the
opposite of a notebook: everything scientific lives here, and the notebook only invokes it
(``AGENTS.md`` §2b rule 1).

Honest scope of this slice
--------------------------
* **Implemented** (non-trainable, on a real pretrained model, float execution — class 2 for quality):

  - ``fp16_reference`` — the uncompressed upper bound (no compression; class-1 byte count).
  - ``ptq_uniform`` — fake-quantized linear weights at the arm's bit width
    (:func:`spectraquant.quantization.fake_quantize`), packed through the real container
    (:func:`spectraquant.quantization.measure_serialized_bytes`, class 3).
  - ``low_rank_only`` — truncated SVD at the arm's rank
    (:func:`spectraquant.factorization.truncated_svd`; class-1 factor bytes).
  - ``rank_then_quant`` — SVD then fake-quantization of **both** factors (class 3 measured bytes).

* **Not implemented here**: every other kind, including all ``trainable`` arms. Those raise
  :class:`NotImplementedError` naming the milestone that owns them; no number is ever emitted for
  them (``AGENTS.md`` §4.5, §4.13). A default (arm-unrestricted) invocation runs the implemented
  arms and *records* the rest as skipped (with the reason) in the summary — it never silently narrows the
  plan's matrix, and an explicitly requested unimplemented arm is a hard error.

Nothing here may fabricate a number: the model revision actually loaded is recorded and compared
against the plan's pin, a floating revision is refused, and every byte figure comes from
:mod:`spectraquant.quantization.accounting` (the one byte-accounting source of truth,
``design-m2-interfaces.md`` §0.1).

Perplexity protocol (this slice, documented, not the frozen harness cell)
------------------------------------------------------------------------
Perplexity is measured on the plan's ``test_perplexity`` dataset split, pinned by revision:
per document, the token stream is cut into **non-overlapping full-context windows** (a trailing
shorter window is dropped), and ``perplexity = exp(sum(NLL) / sum(tokens))`` over all documents, with
the model in float32 (``eval-protocol.md`` §6.3 freezes fp32 for cross-arm byte-identity) and no
sampling. The protocol id is recorded in every manifest, together with the fact that the *frozen*
P1 metric is the ``lm-evaluation-harness`` ``wikitext`` task (``eval-protocol.md`` §6.1), which this
slice does not yet wire — a run of this runner is a real measurement of a real model, not the Tier-1
published number.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from spectraquant.cloud.plan_data import (
    load_plan_texts,
    plan_dataset_ref,
    require_transformers,
    verify_pinned_revision,
)
from spectraquant.cloud.remote import RESULT_LINE_PREFIX, sha256_of_files
from spectraquant.experiment_plan import ArmSpec, PlanConfig, load_plan, plan_path
from spectraquant.factorization import factor_bytes, reconstruction_error, truncated_svd
from spectraquant.quantization import (
    SQ_CONTAINER_FORMAT_ID,
    QuantSpec,
    accounted_bytes,
    fake_quantize,
    measure_serialized_bytes,
    quantization_error,
    theoretical_bits,
)
from spectraquant.reporting.environment import (
    collect_hardware,
    collect_software,
    process_peak_rss_mb,
)
from spectraquant.reporting.gitinfo import git_info
from spectraquant.reporting.manifests import (
    CompressionBlock,
    HardwareBlock,
    RunManifest,
    TrainingBlock,
    build_run_id,
    utc_timestamp,
    write_manifest,
)
from spectraquant.training.loop import SequenceData
from spectraquant.training.seeding import seed_everything

__all__ = [
    "FROZEN_DTYPE",
    "IMPLEMENTED_ARM_KINDS",
    "PERPLEXITY_PROTOCOL",
    "ArmCompression",
    "ArmRun",
    "ModelRevisionMismatch",
    "PlanRunResult",
    "apply_arm",
    "arm_not_implemented_error",
    "compressible_linear_names",
    "load_model",
    "resolve_arm_compression",
    "revision_of",
    "run_plan",
    "token_level_perplexity",
]

#: Arm kinds this slice can execute honestly (no training loop involved).
#: Arms that are *trained* after their initialisation. They take the trainable path through the loop:
#: the target layers are replaced by :class:`QuantizedPlusLowRankLinear` (frozen quantized base plus
#: trainable factors), the loop optimises the factors, and the deployed weight is ``Wq + B @ A``.
TRAINABLE_ARM_KINDS: frozenset[str] = frozenset({"loftq", "lr_qat"})

IMPLEMENTED_ARM_KINDS: frozenset[str] = frozenset(
    {"fp16_reference", "ptq_uniform", "low_rank_only", "rank_then_quant"}
) | TRAINABLE_ARM_KINDS

#: Identifier of the perplexity protocol implemented here (see the module docstring).
PERPLEXITY_PROTOCOL = "non-overlapping-window-token-ce-v1"

#: Perplexity above which the model is not degraded but broken (it assigns ~zero probability to the
#: held-out text). Such a number is recorded with ``perplexity_degenerate: true`` so no analysis can
#: treat it as a quality measurement: the observed rank-8 truncation arm scored 1.4e16.
DEGENERATE_PERPLEXITY = 1.0e6

#: The frozen evaluation dtype: ``eval-protocol.md`` §6.3 freezes float32 for cross-arm identity.
FROZEN_DTYPE = "float32"

#: ``compression.method`` per arm kind, restricted to the manifest schema's enum.
_METHOD_BY_KIND: dict[str, str] = {
    "fp16_reference": "none",
    "ptq_uniform": "rtn",
    "low_rank_only": "svd",
    "rank_then_quant": "rtn",
    "loftq": "svd",
    "lr_qat": "rtn",
}

#: Milestone that owns each arm kind this slice does not implement, with what it will do.
_ARM_OWNER: dict[str, tuple[str, str]] = {
    "quant_then_residual": ("M4", "the quantize-then-residual allocation arm"),
    "proxy_allocated": ("M4", "the proxy-allocated layer-wise rank/bit arm"),
    "proxy_allocated_regularized": ("M5", "the regularized joint training arm (the H3 arm)"),
    "qlora": ("M5", "the QLoRA reproduction arm"),
    "spectraquant": ("M5", "the SpectraQuant joint low-rank/low-bit arm"),
}

_BITS_SUFFIX = re.compile(r"_(\d+)$")

#: Alternating steps of the LoftQ initialisation (arXiv 2310.08659). One step is *not* LoftQ - it is
#: the first residual decomposition - and the paper's ablations use T = 1, 3, 5 with the published
#: quality claims made for T >= 3, so the default is 3. Recorded in the manifest as
#: ``compression.loftq_iterations``.
LOFTQ_ITERATIONS = 3

#: Marker recorded when the perplexity corpus is a substituted local stand-in file.
_LOCAL_TEXT_STAND_IN = "local-perplexity-text"


class ModelRevisionMismatch(RuntimeError):
    """Raised when the revision actually loaded is not the plan's pin.

    The model revision is the identity of the measured weights; a run that cannot prove it is not
    evidence (``AGENTS.md`` §2b rule 3, §4.11).
    """


def arm_not_implemented_error(arm: ArmSpec) -> NotImplementedError:
    """Return the error an unimplemented arm must raise (never a placeholder number).

    The message names the milestone that owns the arm *and* why this slice cannot run it: a trainable
    arm needs the M5 training loop wired to a pretrained model, and everything else needs its own
    slice. Naming the owner is what makes the refusal actionable instead of a bare failure.
    """
    owner, description = _ARM_OWNER.get(arm.kind, ("a later milestone", f"the {arm.kind!r} arm"))
    if arm.trainable:
        why = ["it is a trainable arm"]
        if owner == "M5":
            why.append("the M5 training loop exists but is not yet wired to a pretrained model")
    else:
        why = [
            "this slice implements only the non-trainable arms "
            + ", ".join(sorted(IMPLEMENTED_ARM_KINDS))
        ]
    return NotImplementedError(
        f"arm {arm.name!r} (kind {arm.kind!r}) is not implemented in this slice: "
        f"{'; '.join(why)}; {owner} owns {description}. No number may be produced for it "
        "(AGENTS.md §4.5)."
    )


@dataclass(frozen=True)
class ArmCompression:
    """The grid point one arm run uses, validated against the plan's predeclared grid."""

    bits: int | None
    rank: int | None
    granularity: str
    group_size: int | None
    symmetric: bool
    axis: int

    def quant_spec(self, bits: int) -> QuantSpec:
        """Return the :class:`QuantSpec` for ``bits`` at this arm's granularity."""
        return QuantSpec(
            bits=bits,
            granularity=self.granularity,
            group_size=self.group_size,
            symmetric=self.symmetric,
            axis=self.axis,
        )


@dataclass(frozen=True)
class ArmRun:
    """One executed (arm, seed) pair and where its records were written."""

    arm: str
    kind: str
    seed: int
    run_id: str
    manifest_path: Path
    metrics_path: Path
    measurement_class: int | None
    perplexity: float
    accounted_bytes: int


@dataclass(frozen=True)
class PlanRunResult:
    """Everything one ``run-plan`` invocation produced."""

    plan_name: str
    plan_path: Path
    out_dir: Path
    model_id: str
    model_revision: str
    model_source: str
    runs: tuple[ArmRun, ...] = ()
    skipped: tuple[dict[str, str], ...] = ()
    aggregate_metrics_path: Path | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------------------
# Arm selection and grid resolution
# --------------------------------------------------------------------------------------
def _grid_point(
    value: int, allowed: Sequence[int], *, what: str, arm: str, plan: PlanConfig
) -> int:
    if value not in set(allowed):
        raise ValueError(
            f"{what}={value} for arm {arm!r} is not in plan {plan.name!r}'s predeclared grid "
            f"{list(allowed)}: a run may only use grid points the plan authorised"
        )
    return int(value)


def _bits_from_name(name: str) -> int | None:
    match = _BITS_SUFFIX.search(name)
    return int(match.group(1)) if match else None


def _arm_measurement_class(kind: str) -> int | None:
    """Return the measurement class of one arm's quality number.

    Class 2 is *fake-quantization quality*: float execution that simulates quantization numerics. The
    fp16 reference measures no compression at all and carries no class. A pure low-rank truncation
    executes an approximate artifact in float, which the frozen taxonomy has no separate class for;
    it is reported as class 2 with the compression block recording that no quantization was applied,
    and the mismatch between the label and the arm is an open question for the taxonomy's owner.
    """
    if kind == "fp16_reference":
        return None
    return 2


def resolve_arm_compression(
    plan: PlanConfig,
    arm: ArmSpec,
    *,
    bits: int | None = None,
    rank: int | None = None,
    granularity: str = "per_channel",
    group_size: int | None = None,
    symmetric: bool = True,
    axis: int = 0,
) -> ArmCompression:
    """Resolve the grid point an arm runs at, refusing anything the plan did not predeclare.

    The frozen plans declare the grid (bit widths, ranks, group sizes) but not a per-arm point, so
    the point is selected explicitly: ``bits`` may come from the arm name (``ptq_uniform_8``) or
    from ``bits``; ``rank`` must be given for the rank arms. Both are validated against
    ``plan.grid``, so a run cannot explore a point the preregistration never authorised.

    Args:
        plan: the loaded plan.
        arm: the arm to resolve.
        bits: explicit bit width; defaults to the arm name's ``_<bits>`` suffix when present.
        rank: explicit rank; required by ``low_rank_only`` / ``rank_then_quant``.
        granularity: ``per_tensor`` | ``per_channel`` | ``per_group`` (default ``per_channel``).
        group_size: required iff ``granularity == "per_group"``; validated against the grid.
        symmetric: symmetric codes (default) or asymmetric.
        axis: channel axis for per-channel/per-group blocks.

    Returns:
        The validated :class:`ArmCompression` (``bits``/``rank`` are ``None`` where not applicable).

    Raises:
        ValueError: the arm's point is missing, or lies outside the plan's grid.
        NotImplementedError: the arm kind is not implemented in this slice.
    """
    kind = arm.kind
    if kind not in IMPLEMENTED_ARM_KINDS:
        raise arm_not_implemented_error(arm)

    if granularity not in {"per_tensor", "per_channel", "per_group"}:
        raise ValueError(
            f"granularity must be per_tensor/per_channel/per_group, got {granularity!r}"
        )
    if granularity == "per_group":
        if group_size is None:
            raise ValueError(
                f"granularity='per_group' for arm {arm.name!r} requires group_size "
                f"(plan grid: {plan.grid.group_sizes})"
            )
        group_size = _grid_point(
            group_size, plan.grid.group_sizes, what="group_size", arm=arm.name, plan=plan
        )
    elif group_size is not None:
        raise ValueError(
            f"group_size={group_size} is meaningless for granularity={granularity!r}: pass "
            "--granularity per_group, or drop it"
        )

    if kind == "fp16_reference":
        return ArmCompression(
            bits=None,
            rank=None,
            granularity=granularity,
            group_size=group_size,
            symmetric=symmetric,
            axis=axis,
        )

    # The plan's own binding is the base point; an explicit CLI value overrides it, and an arm that
    # names its width in its own name still wins over both (see below).
    bound = dict(getattr(arm, "point", {}) or {})
    if rank is None and "rank" in bound:
        rank = int(bound["rank"])
    if bits is None and "bits" in bound:
        bits = int(bound["bits"])
    if group_size is None and "group_size" in bound:
        group_size = int(bound["group_size"])

    named_bits = _bits_from_name(arm.name)
    if named_bits is not None:
        if bits is not None and int(bits) != named_bits:
            # A silent override is the worst option: it produced a run whose "ptq_uniform_8" arm was
            # 4-bit, with the same perplexity and the same byte count as the 4-bit arm, and nothing in
            # the record said so. The arm's name is part of its identity, so it wins, and a conflict is
            # reported instead of resolved.
            raise ValueError(
                f"--bits={bits} conflicts with arm {arm.name!r}, which names {named_bits}-bit; "
                "the arm name is authoritative - drop --bits, or run a differently named arm"
            )
        resolved_bits = named_bits
    else:
        resolved_bits = bits
    resolved_rank = rank

    if kind == "ptq_uniform":
        if resolved_bits is None:
            raise ValueError(
                f"arm {arm.name!r} does not name a bit width and --bits was not given "
                f"(plan grid: {plan.grid.bits})"
            )
        resolved_bits = _grid_point(
            resolved_bits, plan.grid.bits, what="bits", arm=arm.name, plan=plan
        )
    elif kind == "low_rank_only":
        if resolved_rank is None:
            raise ValueError(
                f"arm {arm.name!r} needs an explicit rank: the plan declares a grid "
                f"({plan.grid.ranks}) but does not bind this arm to a point; pass --rank"
            )
        resolved_rank = _grid_point(
            resolved_rank, plan.grid.ranks, what="rank", arm=arm.name, plan=plan
        )
    elif kind in ("rank_then_quant", *TRAINABLE_ARM_KINDS):
        if resolved_rank is None:
            raise ValueError(
                f"arm {arm.name!r} needs an explicit rank: the plan declares a grid "
                f"({plan.grid.ranks}) but does not bind this arm to a point; pass --rank"
            )
        resolved_rank = _grid_point(
            resolved_rank, plan.grid.ranks, what="rank", arm=arm.name, plan=plan
        )
        if resolved_bits is None:
            raise ValueError(
                f"arm {arm.name!r} does not name a bit width and --bits was not given "
                f"(plan grid: {plan.grid.bits})"
            )
        resolved_bits = _grid_point(
            resolved_bits, plan.grid.bits, what="bits", arm=arm.name, plan=plan
        )

    return ArmCompression(
        bits=resolved_bits,
        rank=resolved_rank,
        granularity=granularity,
        group_size=group_size,
        symmetric=symmetric,
        axis=axis,
    )


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
def revision_of(model: Any, *, requested: str, source: str, model_dir: Path | None = None) -> str:
    """Return the revision that was *actually* loaded, and prove it is the requested one.

    Args:
        model: the loaded model (its ``config`` carries Hugging Face's ``_commit_hash``).
        requested: the revision the plan pins.
        source: ``"hub"`` or ``"local_dir"``.
        model_dir: the local directory, when ``source == "local_dir"``.

    Returns:
        The revision string recorded in the manifest.

    Raises:
        ModelRevisionMismatch: the loaded commit differs from the pin, or the hub load did not
            record a commit at all (an unprovable revision is a failure, not a warning).
    """
    if source == "local_dir":
        if model_dir is None:  # pragma: no cover - defensive; callers always pass it
            raise ModelRevisionMismatch("local model source without a directory")
        files = sorted(
            path
            for path in model_dir.rglob("*")
            if path.is_file() and path.suffix in {".json", ".safetensors", ".bin"}
        )
        return f"local-dir:{sha256_of_files(files, root=model_dir)}"

    actual = getattr(getattr(model, "config", None), "_commit_hash", None)
    if not actual:
        raise ModelRevisionMismatch(
            f"the model loaded from {requested!r} recorded no commit hash: the run cannot prove "
            "which weights it measured, so no number may be recorded (AGENTS.md §2b rule 3)"
        )
    if str(actual) != requested:
        raise ModelRevisionMismatch(
            f"loaded model revision {actual} != plan pin {requested!r}: the run would measure "
            "different weights than the plan froze"
        )
    return str(actual)


def load_model(
    plan: PlanConfig,
    *,
    model_dir: str | Path | None = None,
    dtype: str = FROZEN_DTYPE,
) -> tuple[Any, Any, str, str, str]:
    """Load the plan's pretrained model (or an explicit local stand-in) and its tokenizer.

    Args:
        plan: the loaded plan.
        model_dir: local stand-in model directory (a path check, not a plan measurement).
        dtype: torch dtype *name*; the frozen protocol dtype by default.

    Returns:
        ``(model, tokenizer, model_id, model_revision, model_source)``.

    Raises:
        PinnedRevisionError: the plan's model revision is not an immutable commit.
        ModelRevisionMismatch: the loaded commit is not the pin.
        ImportError: the ``models`` extra is missing.
        FileNotFoundError: ``model_dir`` does not exist.
    """
    transformers = require_transformers()
    model_ref = plan.models[0]
    torch_dtype = getattr(torch, dtype)

    def _from_pretrained(loader: Any, ident: str, **kwargs: Any) -> Any:
        try:
            return loader.from_pretrained(ident, dtype=torch_dtype, **kwargs)
        except TypeError:  # pragma: no cover - transformers < 4.56 spells this `torch_dtype`
            return loader.from_pretrained(ident, torch_dtype=torch_dtype, **kwargs)

    if model_dir is not None:
        directory = Path(model_dir)
        if not directory.is_dir():
            raise FileNotFoundError(f"--model-dir {directory} is not a directory")
        model = _from_pretrained(transformers.AutoModelForCausalLM, str(directory))
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(directory))
        revision = revision_of(model, requested="", source="local_dir", model_dir=directory)
        return _finish(model, tokenizer, str(directory), revision, "local_dir")

    verify_pinned_revision(model_ref.id, model_ref.revision, repo_type="model")
    model = _from_pretrained(
        transformers.AutoModelForCausalLM, model_ref.id, revision=model_ref.revision
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_ref.id, revision=model_ref.revision
    )
    revision = revision_of(model, requested=model_ref.revision, source="hub")
    return _finish(model, tokenizer, model_ref.id, revision, "hub")


def _finish(
    model: Any, tokenizer: Any, model_id: str, revision: str, source: str
) -> tuple[Any, Any, str, str, str]:
    model.eval()
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, model_id, revision, source


def compressible_linear_names(model: nn.Module) -> tuple[list[str], list[str]]:
    """Return ``(compressed, excluded)`` **parameter** names of the compressible linear weights.

    The names are ``state_dict`` keys (``"<module>.weight"``), not module names: a
    ``load_state_dict`` mapping keyed by module name silently matches nothing.

    Embeddings and layer norms are never compressed, and a linear weight *tied* to an embedding (a
    tied ``lm_head``, as in SmolLM2/GPT-2) is excluded too: writing a quantized copy in place would
    also mutate the embedding it shares storage with. The excluded names are recorded in the
    manifest's ``compression.exclusions`` so the reader can see exactly what was left in fp16.
    """
    tied = {
        module.weight.data_ptr()
        for module in model.modules()
        if isinstance(module, nn.Embedding) and module.weight is not None
    }
    compressed: list[str] = []
    excluded: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or module.weight.ndim != 2:
            continue
        parameter = f"{name}.weight"
        if module.weight.data_ptr() in tied:
            excluded.append(f"{parameter} (tied embedding weight)")
        else:
            compressed.append(parameter)
    return compressed, excluded


# --------------------------------------------------------------------------------------
# Perplexity
# --------------------------------------------------------------------------------------
def token_level_perplexity(
    model: Any,
    tokenizer: Any,
    documents: Sequence[str],
    *,
    seq_len: int = 2048,
    max_tokens: int | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Measure token-level perplexity on ``documents`` under the documented protocol.

    Per document the token stream is cut into non-overlapping windows of ``seq_len`` tokens (clamped
    to the model's context length), a trailing shorter window is dropped, the cross-entropy of every
    next token is summed, and ``perplexity = exp(total_nll / total_tokens)``. The model is in eval
    mode under ``torch.no_grad``; no sampling happens, so the number is deterministic given the
    weights.

    Args:
        model: a causal LM in eval mode on ``device``.
        tokenizer: its tokenizer.
        documents: document texts (document boundaries are preserved: no window spans two documents).
        seq_len: window length in tokens.
        max_tokens: optional cap on scored tokens (deterministic head truncation), for smoke runs.
        device: ``"cpu"`` or ``"cuda"`` (the latter only on the cloud substrate).

    Returns:
        ``{"perplexity", "perplexity_degenerate", "mean_ce_loss", "nll_sum", "n_tokens",
        "n_windows", "n_documents", "seq_len", "protocol"}``.

    Raises:
        ValueError: no document produced a scorable window, or the loss was non-finite (a failed
            measurement is reported as a failure, never as a number).
    """
    configured = getattr(getattr(model, "config", None), "max_position_embeddings", seq_len)
    max_positions = int(configured or seq_len)
    window = max(2, min(int(seq_len), max_positions))
    total_nll = 0.0
    total_tokens = 0
    windows = 0
    scored_documents = 0

    with torch.no_grad():
        for document in documents:
            if not document.strip():
                continue
            ids = tokenizer(document, return_tensors="pt").input_ids.to(device)
            length = int(ids.shape[1])
            if length < 2:
                continue
            scored_documents += 1
            for start in range(0, length - 1, window):
                chunk = ids[:, start : start + window]
                if int(chunk.shape[1]) < 2:
                    break  # trailing short window: dropped, as documented
                logits = model(chunk).logits[:, :-1]
                targets = chunk[:, 1:]
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]).float(),
                    targets.reshape(-1),
                    reduction="sum",
                )
                total_nll += float(nll)
                total_tokens += int(targets.numel())
                windows += 1
                if max_tokens is not None and total_tokens >= max_tokens:
                    break
            if max_tokens is not None and total_tokens >= max_tokens:
                break

    if total_tokens == 0:
        raise ValueError(
            "no scorable window: every document was shorter than two tokens (check the tokenizer "
            "and the corpus)"
        )
    if not math.isfinite(total_nll):
        # A non-finite loss is a failed measurement, not a number: report it as one and never let it
        # enter a comparison (a NaN perplexity silently sorted as "best" is the classic failure mode).
        raise ValueError(
            f"the model produced a non-finite cross-entropy over {total_tokens} tokens: this is a "
            "failed measurement, not a perplexity - check the compressed artifact"
        )
    mean_ce = total_nll / total_tokens
    perplexity = float(math.exp(mean_ce)) if mean_ce < 700.0 else float("inf")
    degenerate = perplexity > DEGENERATE_PERPLEXITY
    return {
        "perplexity": perplexity,
        "perplexity_degenerate": bool(degenerate),
        "mean_ce_loss": float(mean_ce),
        "nll_sum": float(total_nll),
        "n_tokens": int(total_tokens),
        "n_windows": int(windows),
        "n_documents": int(scored_documents),
        "seq_len": int(window),
        "protocol": PERPLEXITY_PROTOCOL,
    }


# --------------------------------------------------------------------------------------
# Arm application
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class _TrainableInit:
    """The initial state of a trainable arm: the frozen base and the starting factors, per layer.

    A trainable arm does not "load a state" into the model: it *replaces* the target layers with
    :class:`~spectraquant.training.low_rank.QuantizedPlusLowRankLinear` and then optimises the
    factors, so its initialisation is a structural change, not a ``load_state_dict`` payload.
    """

    bases: dict[str, Tensor]
    factors: dict[str, tuple[Tensor, Tensor]]
    iterations: int | None = None


@dataclass(frozen=True)
class _Applied:
    """Result of compressing one model's target weights for one arm."""

    state: dict[str, Tensor]
    storage: dict[str, Tensor]
    storage_specs: dict[str, QuantSpec]
    compression: dict[str, Any]
    accounted: int
    measured: int | None
    theoretical_bits: float | None
    metrics: dict[str, Any]
    trainable_init: _TrainableInit | None = None


def _module_of(weight_name: str) -> str:
    """Return the module path of a ``state_dict``-style weight name (``a.b.weight`` -> ``a.b``)."""
    suffix = ".weight"
    return weight_name[: -len(suffix)] if weight_name.endswith(suffix) else weight_name


def _hf_task_loss(model: nn.Module, inputs: Tensor, targets: Tensor) -> Tensor:
    """Next-token cross-entropy for a Hugging Face causal LM.

    The loop's default task loss expects a model that returns a logits *tensor* (the Tier-0 fixture).
    A pretrained causal LM returns an output object with ``.logits``, so the runner supplies this
    instead of letting the loop call ``model(inputs)`` and treat the object as a tensor. The logits
    are upcast to float32 before the reduction, exactly as the perplexity protocol does, so a
    half-precision forward cannot change the loss.
    """
    output = model(inputs)
    logits = getattr(output, "logits", output)
    if not isinstance(logits, Tensor):
        raise TypeError(
            f"model returned {type(output).__name__} with no `.logits` tensor: the task loss cannot "
            "be computed"
        )
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1), reduction="mean"
    )


def _train_arm(
    model: nn.Module,
    *,
    arm: ArmSpec,
    compression: ArmCompression,
    applied: _Applied,
    config: PlanConfig,
    data: SequenceData,
    seed: int,
    threads: int,
    out_dir: Path,
    verbose: bool,
) -> tuple[dict[str, Any], Any]:
    """Train one arm's factors and return the training provenance and the loop's result.

    The target layers are replaced by :class:`QuantizedPlusLowRankLinear` (frozen quantized base plus
    trainable factors) before the first step, so the optimiser sees exactly the artifact the arm
    stores. The dev split drives any intermediate evaluation; the test split is not read here.

    Args:
        model: the loaded model, modified in place.
        arm: the arm being trained.
        compression: its resolved grid point.
        applied: its initialisation (``trainable_init`` must be set).
        config: the plan, whose ``training`` block is the pinned schedule.
        data: the loop's train/dev tensors.
        seed: the reported seed for this run.
        threads: ``torch.set_num_threads`` value.
        out_dir: where the loop writes checkpoints.
        verbose: forward the loop's structured log.

    Returns:
        ``(provenance, result)`` - the ``training.*`` manifest metrics and the loop's result object.

    Raises:
        ValueError: the plan has no training schedule, or the arm carries no initialisation.
    """
    from spectraquant.training.loop import LoopConfig, train_language_model
    from spectraquant.training.low_rank import replace_linears_quantized

    schedule = config.training
    if schedule is None:
        raise ValueError(
            f"plan {config.name!r} declares trainable arm {arm.name!r} but no `training` block"
        )
    init = applied.trainable_init
    if init is None:
        raise ValueError(f"arm {arm.name!r} is trainable but carries no initialisation")
    assert compression.bits is not None


    # `init.bases` is keyed by the *weight* name the accounting uses (`...q_proj.weight`), while the
    # replacement addresses the *module* (`...q_proj`). The two are the same set of layers under two
    # naming conventions, and the conversion is asserted rather than assumed.
    bases = {_module_of(name): tensor for name, tensor in init.bases.items()}
    factors = {_module_of(name): pair for name, pair in init.factors.items()}
    if len(bases) != len(init.bases):
        raise ValueError(
            "two target weights map to the same module, so the replacement would silently drop one: "
            f"{sorted(init.bases)}"
        )
    replace_linears_quantized(model, bases=bases, factors=factors)
    out_dir.mkdir(parents=True, exist_ok=True)
    loop = LoopConfig(
        batch_size=int(schedule.batch_size),
        lr=float(schedule.learning_rate),
        grad_clip=float(schedule.grad_clip),
        seed=int(seed),
        threads=int(threads),
        warmup_steps=int(schedule.warmup_steps),
        checkpoint_every=int(schedule.checkpoint_every) or None,
        eval_every=int(schedule.eval_every) or None,
    )
    result = train_language_model(
        model,
        method=arm.kind,
        steps=int(schedule.steps),
        output_dir=out_dir,
        data=data,
        loop=loop,
        task_loss=_hf_task_loss,
        spec=compression.quant_spec(int(compression.bits)),
        measurement_class=2,
        # The loop must not write its own manifest: the runner's manifest is the one the collector
        # validates, and two manifests in one directory would make the identity check ambiguous.
        write=False,
        extra_metrics={
            "training.arm": arm.name,
            "training.seed": int(seed),
            "training.steps_requested": int(schedule.steps),
            "training.initialisation": (
                f"{arm.kind}: frozen quantized base + factors"
                + (f" from {init.iterations} alternating LoftQ steps" if init.iterations else "")
            ),
        },
        verbose=verbose,
    )
    provenance: dict[str, Any] = {
        "training.method": arm.kind,
        "training.steps_requested": int(schedule.steps),
        "training.steps_completed": int(result.steps_completed),
        "training.learning_rate": float(schedule.learning_rate),
        "training.batch_size": int(schedule.batch_size),
        # The window width actually built, and the pinned value beside it: a substituted corpus (a
        # fixture, or a smoke run) must not be recorded as if it used the plan's width.
        "training.seq_len": int(data.train.shape[1]) - 1,
        "training.seq_len_pinned": int(schedule.seq_len),
        "training.warmup_steps": int(schedule.warmup_steps),
        "training.grad_clip": float(schedule.grad_clip),
        "training.optimizer": str(schedule.optimizer),
        "training.wall_time_s": float(result.wall_time_s),
        "training.final_task_loss": (float(result.losses[-1]) if result.losses else None),
        "training.dev_losses": [[int(step), float(value)] for step, value in result.val_losses],
        "training.corpus_checksum": data.checksum,
        "training.checkpoints": [str(path) for path in result.checkpoint_paths],
        "training.initialisation": (
            f"{arm.kind}: frozen quantized base + factors"
            + (f" from {init.iterations} alternating LoftQ steps" if init.iterations else "")
        ),
    }
    if int(result.steps_completed) != int(schedule.steps):
        raise ValueError(
            f"arm {arm.name!r} completed {result.steps_completed} of {schedule.steps} requested "
            "steps: a short run must not be recorded as a complete one"
        )
    return provenance, result


def _apply_trainable_arm(
    weights: dict[str, Tensor],
    names: Sequence[str],
    arm: ArmSpec,
    compression: ArmCompression,
    metrics: dict[str, Any],
    total_numel: int,
) -> _Applied:
    """Initialise one trainable arm: a frozen quantized base plus the arm's starting factors.

    Both reproduction arms share the deployed structure - a quantized main weight with a low-rank
    correction - and differ only in how the factors are *initialised*:

    * ``lr_qat`` (arXiv 2406.06385): the factors start at the truncated SVD of the residual left by
      the quantized base, then training compensates the quantization error.
    * ``loftq`` (arXiv 2310.08659): the factors start from the **alternating** schedule, which
      re-quantizes the residual and re-decomposes it, so the initialisation already absorbs the
      rounding grid.

    Args:
        weights: ``{module name: 2-D linear weight}``, the same set for every arm.
        names: the sorted layer names.
        arm: the arm being run.
        compression: its resolved grid point (rank and bits are both required).
        metrics: the manifest metrics block to extend in place.
        total_numel: total element count of the target tensors.

    Returns:
        An :class:`_Applied` whose ``trainable_init`` carries the bases and factors, with class-1
        accounting for the *stored* object: the quantized base plus the two factors at fp16.
    """
    from spectraquant.factorization.loftq import loftq_initialise
    from spectraquant.factorization.lr_qat import lr_qat_pair

    assert compression.rank is not None and compression.bits is not None
    rank = int(compression.rank)
    spec = compression.quant_spec(int(compression.bits))

    bases: dict[str, Tensor] = {}
    factors: dict[str, tuple[Tensor, Tensor]] = {}
    residual_errors: list[float] = []
    base_errors: list[float] = []
    for name in names:
        weight = weights[name]
        if arm.kind == "lr_qat":
            base, pair = lr_qat_pair(weight, rank=rank, spec=spec)
        else:
            # The LoftQ primitive returns the low-rank half only; its companion base is the *next*
            # quantization half-step of the same alternation, recomputed from the returned factors as
            # `fake_quantize(w - B @ A)`. Quantizing the full weight here instead would pair a base
            # with factors that do not belong to it, and the reconstruction would be far worse than
            # the naive baseline (measured: 75% relative error versus 7.3%).
            pair = loftq_initialise(weight, rank=rank, spec=spec, iterations=LOFTQ_ITERATIONS)
            base = fake_quantize(weight - pair[1] @ pair[0], spec)
        bases[name] = base
        factors[name] = pair
        base_errors.append(_relative_fro(weight, base))
        residual_errors.append(_relative_fro(weight, base + pair[1] @ pair[0]))

    metrics["compression.relative_fro_mean"] = (
        float(sum(residual_errors) / len(residual_errors)) if residual_errors else None
    )
    metrics["compression.relative_fro_max"] = max(residual_errors) if residual_errors else None
    metrics["compression.relative_fro_worst_layer"] = (
        names[residual_errors.index(max(residual_errors))] if residual_errors else None
    )
    # The error the *initialisation* leaves is not the error the trained arm leaves; recording both
    # keeps the two apart, so a later improvement can be attributed to training rather than to init.
    metrics["compression.initial_base_relative_fro_mean"] = (
        float(sum(base_errors) / len(base_errors)) if base_errors else None
    )
    metrics["compression.initial_residual_relative_fro_mean"] = metrics[
        "compression.relative_fro_mean"
    ]
    metrics["compression.effective_ranks"] = sorted({int(factors[n][0].shape[0]) for n in names})
    if arm.kind == "loftq":
        metrics["compression.loftq_iterations"] = int(LOFTQ_ITERATIONS)

    # Stored object: the quantized base's codes+scales, plus the two factors at fp16. The factors are
    # the *trained* artifact, so they are counted at storage precision, not at their float32 size.
    # Account from the factors' *actual* shapes: the SVD clamps the rank on a narrow layer, and
    # charging the requested rank there would overstate the stored bytes.
    accounted = 0
    for name in names:
        weight = weights[name]
        out_features, in_features = int(weight.shape[0]), int(weight.shape[1])
        a, b = factors[name]
        accounted += accounted_bytes((out_features, in_features), spec)
        accounted += 2 * int(a.numel() + b.numel())

    return _Applied(
        state={},
        storage={},
        storage_specs={},
        compression={
            "method": _METHOD_BY_KIND[arm.kind],
            "ranks": [rank],
            "bits": int(compression.bits),
            "group_size": compression.group_size,
            "bytes_source": "accounted",
            "accounted_bytes": accounted,
            "measured_bytes": None,
            "nominal_bits_per_param": None,
            "measured_bits_per_param": None,
        },
        accounted=accounted,
        measured=None,
        theoretical_bits=None,
        metrics=metrics,
        trainable_init=_TrainableInit(
            bases=bases,
            factors=factors,
            iterations=LOFTQ_ITERATIONS if arm.kind == "loftq" else None,
        ),
    )


def _relative_fro(reference: Tensor, reconstructed: Tensor) -> float:
    return quantization_error(reference, reconstructed)["relative_fro"]


#: Which split of a dataset each protocol role reads. The role names the *purpose*; the split is the
#: dataset's own vocabulary, and the two differ (``development`` reads WikiText-2's ``validation``).
ROLE_SPLITS: dict[str, str] = {
    "train": "train",
    "calibration": "train",
    "development": "validation",
    "test_perplexity": "test",
    "test_downstream": "test",
}


def _token_windows(texts: Sequence[str], tokenizer: Any, seq_len: int) -> Tensor:
    """Cut ``texts`` into non-overlapping ``seq_len + 1`` token windows.

    Args:
        texts: document texts; no window spans two documents (document boundaries are preserved).
        tokenizer: a Hugging Face tokenizer; the pinned revision is the caller's responsibility.
        seq_len: window length in tokens; each row is ``seq_len + 1`` wide so inputs and targets are
            ``[:, :-1]`` and ``[:, 1:]``.

    Returns:
        An int64 tensor of shape ``(n_windows, seq_len + 1)`` on CPU. A trailing short window is
        dropped, as in the perplexity protocol.
    """
    width = int(seq_len) + 1
    rows: list[Tensor] = []
    for document in texts:
        if not document.strip():
            continue
        ids = tokenizer(document, return_tensors="pt").input_ids.reshape(-1)
        length = int(ids.numel())
        for start in range(0, length - width + 1, width):
            rows.append(ids[start : start + width])
    if not rows:
        raise ValueError(
            f"no training window of {width} tokens could be built: the corpus documents are too "
            "short for this seq_len"
        )
    return torch.stack(rows).to(torch.int64)


def _training_corpus(
    config: PlanConfig,
    tokenizer: Any,
    *,
    seq_len: int,
    max_documents: int | None,
    context: int | None = None,
) -> tuple[SequenceData, dict[str, Any]]:
    """Build the loop's train/dev tensors from the plan's own role-pinned corpora.

    Train reads the plan's ``train`` role; dev reads its ``development`` role. They are different
    datasets in the frozen plans (C4 and WikiText-2), so the dev split can never leak into training,
    and the *test* split is never read here at all - it is measured once, after training.

    Args:
        config: the plan.
        tokenizer: the pinned tokenizer.
        seq_len: training window length.
        max_documents: optional cap per split (deterministic head), for smoke runs.
        context: the model's maximum position count. A window wider than the context cannot be
            forwarded, and clamping would run a different experiment than the plan declares, so the
            mismatch is refused rather than absorbed.

    Returns:
        ``(SequenceData, provenance)`` where provenance names both datasets, their configs, the
        splits read and the resolved revisions - recorded in the manifest.

    Raises:
        KeyError: the plan declares no dataset for the ``train`` or ``development`` role.
        ImportError: the ``models`` extra is missing.
    """
    if context and int(seq_len) > int(context):
        raise ValueError(
            f"plan {config.name!r} pins training.seq_len={seq_len} but the model's context is "
            f"{context}: the schedule and the model disagree, and clamping would run a different "
            "experiment than the plan declares"
        )

    provenance: dict[str, Any] = {}
    tensors: dict[str, Tensor] = {}
    for role, key in (("train", "train"), ("development", "val")):
        ref = plan_dataset_ref(config, role)
        texts = load_plan_texts(
            ref,
            split=ROLE_SPLITS[role],
            dataset_config=ref.config,
            max_documents=max_documents,
        )
        tensors[key] = _token_windows(texts, tokenizer, seq_len)
        provenance[f"training.{key}_dataset"] = ref.name
        provenance[f"training.{key}_config"] = ref.config
        provenance[f"training.{key}_split"] = ROLE_SPLITS[role]
        provenance[f"training.{key}_revision"] = ref.revision
        provenance[f"training.{key}_documents"] = len(texts)
        provenance[f"training.{key}_windows"] = int(tensors[key].shape[0])

    digest = hashlib.sha256()
    for key in ("train", "val"):
        digest.update(tensors[key].numpy().tobytes())
    data = SequenceData(
        train=tensors["train"],
        val=tensors["val"],
        checksum=f"sha256:{digest.hexdigest()}",
    )
    provenance["training.corpus_checksum"] = data.checksum
    provenance["training.seq_len"] = int(seq_len)
    provenance["training.split_rule"] = (
        "train = plan role 'train' split 'train'; dev = plan role 'development' split 'validation'; "
        "the test split is never read during training"
    )
    return data, provenance


def apply_arm(
    weights: dict[str, Tensor],
    arm: ArmSpec,
    compression: ArmCompression,
) -> _Applied:
    """Compress ``weights`` for one arm, reusing the frozen quantization/factorization primitives.

    Args:
        weights: ``{module name: 2-D linear weight}`` — the same tensor set for every arm, so the
            arms' byte figures are comparable at equal memory (``AGENTS.md`` §4.5).
        arm: the arm being run.
        compression: its resolved grid point.

    Returns:
        The compressed ``state`` (what to load back into the model), the ``storage`` tensors and
        ``storage_specs`` that define the arm's stored object, the manifest ``compression`` block,
        and the class-1/class-3 byte figures.

    Raises:
        NotImplementedError: the arm kind is not implemented in this slice.
    """
    kind = arm.kind
    if kind not in IMPLEMENTED_ARM_KINDS:
        raise arm_not_implemented_error(arm)

    names = sorted(weights)
    total_numel = sum(int(weights[name].numel()) for name in names)
    metrics: dict[str, Any] = {
        "compression.n_tensors": len(names),
        "compression.n_parameters": total_numel,
        "compression.granularity": compression.granularity,
        "compression.symmetric": bool(compression.symmetric),
        "compression.axis": int(compression.axis),
        "compression.group_size": compression.group_size,
    }

    if kind == "fp16_reference":
        # No compression: the reference stores the same tensors at 16 bits (class-1 analytical).
        accounted = 2 * total_numel
        metrics["compression.relative_fro_mean"] = 0.0
        return _Applied(
            state={},
            storage={},
            storage_specs={},
            compression={
                "method": _METHOD_BY_KIND[kind],
                "ranks": None,
                "bits": None,
                "group_size": None,
                "bytes_source": "accounted",
                "accounted_bytes": accounted,
                "measured_bytes": None,
                "nominal_bits_per_param": None,
                "measured_bits_per_param": None,
            },
            accounted=accounted,
            measured=None,
            theoretical_bits=None,
            metrics=metrics,
        )

    state: dict[str, Tensor] = {}
    storage: dict[str, Tensor] = {}
    storage_specs: dict[str, QuantSpec] = {}
    relative_errors: list[float] = []
    svd_relative_errors: list[float] = []

    if kind in TRAINABLE_ARM_KINDS:
        return _apply_trainable_arm(weights, names, arm, compression, metrics, total_numel)

    if kind == "ptq_uniform":
        assert compression.bits is not None  # guaranteed by resolve_arm_compression
        spec = compression.quant_spec(compression.bits)
        for name in names:
            weight = weights[name]
            quantized = fake_quantize(weight, spec)
            state[name] = quantized
            storage[name] = quantized
            storage_specs[name] = spec
            relative_errors.append(_relative_fro(weight, quantized))
    else:
        assert compression.rank is not None  # guaranteed by resolve_arm_compression
        for name in names:
            weight = weights[name]
            factors = truncated_svd(weight, compression.rank)
            svd_relative_errors.append(
                float(reconstruction_error(weight, factors, norm="relative_fro"))
            )
            if kind == "low_rank_only":
                state[name] = factors.B @ factors.A
            else:
                assert compression.bits is not None
                spec = compression.quant_spec(compression.bits)
                quantized_a = fake_quantize(factors.A, spec)
                quantized_b = fake_quantize(factors.B, spec)
                state[name] = quantized_b @ quantized_a
                # The stored object is the two quantized factors, so those are what we serialize.
                storage[f"{name}.A"] = quantized_a
                storage[f"{name}.B"] = quantized_b
                storage_specs[f"{name}.A"] = spec
                storage_specs[f"{name}.B"] = spec
            relative_errors.append(_relative_fro(weight, state[name]))

    metrics["compression.relative_fro_mean"] = (
        float(sum(relative_errors) / len(relative_errors)) if relative_errors else None
    )
    metrics["compression.relative_fro_max"] = max(relative_errors) if relative_errors else None
    metrics["compression.relative_fro_worst_layer"] = (
        names[relative_errors.index(max(relative_errors))] if relative_errors else None
    )
    if svd_relative_errors:
        metrics["compression.svd_relative_fro_mean"] = float(
            sum(svd_relative_errors) / len(svd_relative_errors)
        )
        metrics["compression.svd_relative_fro_max"] = max(svd_relative_errors)

    if kind == "low_rank_only":
        # Float factors at fp16 storage: class-1 analytical only (our container packs quantized
        # codes, so nothing about this arm is serialized into our own format).
        accounted = sum(
            factor_bytes(
                int(weights[name].shape[1]),
                int(weights[name].shape[0]),
                int(compression.rank or 0),
            )
            for name in names
        )
        metrics["compression.accounted_bytes_class"] = 1
        return _Applied(
            state=state,
            storage={},
            storage_specs={},
            compression={
                "method": _METHOD_BY_KIND[kind],
                "ranks": [int(compression.rank)] if compression.rank is not None else None,
                "bits": None,
                "group_size": None,
                "bytes_source": "accounted",
                "accounted_bytes": accounted,
                "measured_bytes": None,
                "nominal_bits_per_param": None,
                "measured_bits_per_param": None,
            },
            accounted=accounted,
            measured=None,
            theoretical_bits=None,
            metrics=metrics,
        )

    # Quantized arms: the stored object is the packed container, so bytes are *measured* (class 3).
    accounted = sum(
        accounted_bytes(tuple(tensor.shape), storage_specs[key]) for key, tensor in storage.items()
    )
    payload_bits = sum(
        theoretical_bits(tuple(tensor.shape), storage_specs[key]) * int(tensor.numel())
        for key, tensor in storage.items()
    )
    packed_numel = sum(int(tensor.numel()) for tensor in storage.values())
    measured = measure_serialized_bytes(storage, storage_specs)
    theoretical = payload_bits / packed_numel if packed_numel else None
    bits = compression.bits
    metrics["compression.accounted_bytes_class"] = 1
    metrics["compression.measured_bytes_class"] = 3
    metrics["compression.serializer"] = SQ_CONTAINER_FORMAT_ID
    metrics["compression.stored_n_tensors"] = len(storage)
    metrics["compression.stored_n_parameters"] = packed_numel
    return _Applied(
        state=state,
        storage=storage,
        storage_specs=storage_specs,
        compression={
            "method": _METHOD_BY_KIND[kind],
            "ranks": [int(compression.rank)] if compression.rank is not None else None,
            "bits": bits,
            "group_size": compression.group_size,
            "bytes_source": "measured",
            "accounted_bytes": accounted,
            "measured_bytes": measured,
            "serializer": SQ_CONTAINER_FORMAT_ID,
            "nominal_bits_per_param": float(bits) if bits is not None else None,
            "measured_bits_per_param": 8.0 * measured / packed_numel if packed_numel else None,
        },
        accounted=accounted,
        measured=measured,
        theoretical_bits=theoretical,
        metrics=metrics,
    )


# --------------------------------------------------------------------------------------
# The runner
# --------------------------------------------------------------------------------------
def _load_corpus_documents(
    plan: PlanConfig,
    *,
    split: str,
    dataset_config: str | None,
    local_text: Path | None,
    max_documents: int | None,
) -> tuple[list[str], dict[str, Any]]:
    """Return ``(documents, provenance)`` for the plan's perplexity corpus."""
    if local_text is not None:
        text = Path(local_text).read_text(encoding="utf-8")
        documents = [chunk for chunk in text.split("\n\n") if chunk.strip()] or [text]
        provenance: dict[str, Any] = {
            "name": _LOCAL_TEXT_STAND_IN,
            "source": str(local_text),
            "sha256": sha256_of_files([local_text]),
            "revision_requested": None,
            "revision_resolved": None,
            "role": "test_perplexity (local stand-in)",
            "split": split,
            "config": None,
        }
        return documents, provenance

    ref = plan_dataset_ref(plan, "test_perplexity")
    documents = load_plan_texts(
        ref, split=split, dataset_config=dataset_config, max_documents=max_documents
    )
    # `load_plan_texts` verifies the pin before loading, so the resolved commit *is* the pin.
    return documents, {
        "name": ref.name,
        "source": "huggingface",
        "sha256": None,
        "revision_requested": ref.revision,
        "revision_resolved": ref.revision,
        "role": "test_perplexity",
        "split": split,
        "config": dataset_config,
    }


def _target_weights(model: nn.Module) -> tuple[dict[str, Tensor], list[str]]:
    """Return ``({parameter name: weight clone}, exclusions)`` for the compressible linear weights."""
    state = model.state_dict()
    names, excluded = compressible_linear_names(model)
    missing = [name for name in names if name not in state]
    if missing:  # pragma: no cover - defensive: a Linear weight is always in the state dict
        raise KeyError(f"linear weights missing from the state dict: {missing}")
    weights = {name: state[name].detach().clone() for name in names}
    return weights, excluded


def _manifest_for(
    *,
    plan: PlanConfig,
    plan_file: Path,
    arm: ArmSpec,
    seed: int,
    applied: _Applied,
    exclusions: list[str],
    perplexity: dict[str, Any],
    model_id: str,
    model_revision: str,
    model_source: str,
    device: str,
    threads: int,
    corpus: dict[str, Any],
    training: dict[str, Any] | None,
    wall_time_s: float,
    run_id: str,
    substitution: dict[str, Any] | None = None,
) -> RunManifest:
    """Build the schema-validated manifest for one (arm, seed) run."""
    git = git_info()
    kind = arm.kind
    measurement_class = _arm_measurement_class(kind)
    compression = dict(applied.compression)
    compression["exclusions"] = list(exclusions)
    peak_mb = process_peak_rss_mb()
    metrics: dict[str, Any] = {
        **applied.metrics,
        **{f"dataset.{key}": value for key, value in corpus.items()},
        **(training or {}),
        **perplexity,
        "plan.name": plan.name,
        "plan.tier": int(plan.tier),
        "plan.substrate": plan.substrate,
        "plan.arm": arm.name,
        "plan.arm_kind": kind,
        "plan.arm_trainable": bool(arm.trainable),
        "plan.master_seed": int(plan.seeds.master),
        **{f"plan.derived_seeds.{key}": int(value) for key, value in plan.seeds.derived.items()},
        "plan.reported_seeds": [int(value) for value in plan.seeds.reported],
        "plan.seed_floor": int(plan.seeds.floor),
        "plan.measurement_classes_authorised": [int(value) for value in plan.measurement_classes],
        "plan.grid.bits": [int(value) for value in plan.grid.bits],
        "plan.grid.ranks": [int(value) for value in plan.grid.ranks],
        "plan.grid.group_sizes": [int(value) for value in plan.grid.group_sizes],
        "plan.budget_ladder_bytes": [int(value) for value in plan.grid.budget_ladder_bytes],
        "model.source": model_source,
        "model.dtype": FROZEN_DTYPE,
        "model.device": device,
        "model.threads": int(threads),
        "model.revision_pinned": plan.models[0].revision,
        "model.revision_loaded": model_revision,
        "model.parameters_pinned": int(plan.models[0].parameters),
        "perplexity.method": PERPLEXITY_PROTOCOL,
        "perplexity.value_class": 2,
        "quality.measurement_class": measurement_class,
        "bytes.accounted_class": 1 if applied.accounted else None,
        "bytes.measured_class": 3 if applied.measured is not None else None,
        "harness.expected_cell": str(plan.harness.get("perplexity_split", "")),
        "harness.commit": str(plan.harness.get("commit", "")),
        "harness.release": str(plan.harness.get("release", "")),
        "harness.note": (
            "this slice measures token-level perplexity directly; the frozen P1 metric is the "
            "lm-evaluation-harness wikitext task (eval-protocol.md 6.1) and is not yet wired"
        ),
    }
    if substitution is not None:
        # The manifest schema allows only scalar/array metrics, so the substitution is flattened.
        for key, value in substitution.items():
            metrics[f"substitution.{key}"] = value
        metrics["substitution.is_plan_measurement"] = False
    return RunManifest(
        run_id=run_id,
        timestamp_utc=utc_timestamp(),
        git_commit=git.commit,
        git_dirty=git.dirty,
        config_path=plan_file.as_posix(),
        resolved_config=plan.model_dump(mode="json"),
        model_id=model_id,
        model_revision=model_revision,
        dataset_ids=[str(corpus["name"])],
        dataset_revisions=[corpus["revision_resolved"]],
        dataset_checksums=[],
        split=str(corpus["split"]),
        seed=int(seed),
        hardware=HardwareBlock(**collect_hardware()),
        software=collect_software(),
        compression=CompressionBlock(**compression),
        theoretical_bits=applied.theoretical_bits,
        packed_bytes=applied.measured,
        runtime_backend="torch-cuda-fp16" if device == "cuda" else "torch-cpu-fp32",
        measurement_class=measurement_class,
        training=TrainingBlock(
            steps=0,
            tokens=int(perplexity["n_tokens"]),
            wall_time_s=round(float(wall_time_s), 6),
            peak_mem_mb=None if peak_mb is None else round(float(peak_mb), 3),
        ),
        metrics=metrics,
        status="success",
        failure_reason=None,
        log_path=None,
        artifact_paths=["metrics.json"],
    )


def run_plan(
    plan: str | Path,
    *,
    seeds: Sequence[int] | None = None,
    out_dir: str | Path | None = None,
    arms: Sequence[str] | None = None,
    bits: int | None = None,
    rank: int | None = None,
    granularity: str = "per_channel",
    group_size: int | None = None,
    symmetric: bool = True,
    axis: int = 0,
    seq_len: int = 2048,
    max_tokens: int | None = None,
    max_documents: int | None = None,
    dataset_config: str | None = None,
    threads: int = 1,
    device: str = "cpu",
    model_dir: str | Path | None = None,
    perplexity_text: str | Path | None = None,
    allow_local_substitution: bool = False,
    progress: Callable[[str], None] | None = None,
) -> PlanRunResult:
    """Execute the implemented arms of a frozen plan and record schema-valid manifests.

    Args:
        plan: plan name or path.
        seeds: plan seeds to run; each must be in ``plan.seeds.reported``. Defaults to the first
            reported seed — the arms implemented here train nothing and are seed-invariant.
        out_dir: output root; defaults to ``artifacts/runs/<plan-name>-plan`` in the repository.
        arms: arm names to run. Defaults to every arm this invocation can run — implemented **and**
            bound to a grid point by the arguments given — while every other arm is recorded as
            skipped with its reason. Naming an unimplemented arm is a hard error.
        bits: bit width for the quantized arms (default: the arm name's ``_<bits>`` suffix).
        rank: rank for the rank arms (required by ``low_rank_only`` / ``rank_then_quant``).
        granularity: quantization granularity (default ``per_channel``).
        group_size: required iff ``granularity == "per_group"``.
        symmetric: symmetric quantization codes (default).
        axis: channel axis for per-channel/per-group blocks.
        seq_len: perplexity window length in tokens (clamped to the model's context length).
        max_tokens: optional cap on scored tokens (smoke runs); recorded in the manifest.
        max_documents: optional cap on the number of documents.
        dataset_config: dataset config name for the perplexity split, when the repository exposes
            more than one (the runner never guesses).
        threads: ``torch.set_num_threads`` value; 1 keeps reductions bit-reproducible.
        device: ``cpu`` or ``cuda`` (the latter only on the cloud substrate).
        model_dir: **local stand-in** model directory; requires ``allow_local_substitution``.
        perplexity_text: **local stand-in** corpus (blank-line separated documents); requires
            ``allow_local_substitution``.
        allow_local_substitution: acknowledge that this run substitutes local assets for the plan's
            pinned ones, so it is a path check and not a plan measurement. Recorded in the manifest.
        progress: callback for human-readable progress (defaults to ``print``).

    Returns:
        A :class:`PlanRunResult` describing every run and where it was written.

    Raises:
        FileNotFoundError: the plan does not exist.
        ValueError: an unknown arm, a seed outside the plan, a missing rank/bit width, or a
            substitution that was not acknowledged.
        NotImplementedError: an explicitly requested arm is not implemented in this slice.
        PinnedRevisionError / ModelRevisionMismatch: an asset pin cannot be honoured.
        ImportError: the ``models`` extra is missing.
    """
    say = progress if progress is not None else print
    plan_file = plan_path(plan)
    config = load_plan(plan_file)

    substituted = model_dir is not None or perplexity_text is not None
    if substituted and not allow_local_substitution:
        raise ValueError(
            "--model-dir/--perplexity-text replace the plan's pinned assets with local stand-ins: "
            "pass --allow-local-substitution to record deliberately that this run is a path check "
            "and not a plan measurement"
        )

    declared = {arm.name: arm for arm in config.arms}

    def _skip_reason(arm: ArmSpec) -> str | None:
        """Why the default run cannot execute ``arm``, or ``None`` when it can."""
        if arm.kind not in IMPLEMENTED_ARM_KINDS:
            return str(arm_not_implemented_error(arm))
        try:
            resolve_arm_compression(
                config,
                arm,
                bits=bits,
                rank=rank,
                granularity=granularity,
                group_size=group_size,
                symmetric=symmetric,
                axis=axis,
            )
        except ValueError as exc:
            return str(exc)
        return None

    if arms is None:
        selected = [arm for arm in config.arms if _skip_reason(arm) is None]
        skipped: tuple[dict[str, str], ...] = tuple(
            {"arm": arm.name, "kind": arm.kind, "reason": reason}
            for arm in config.arms
            if (reason := _skip_reason(arm)) is not None
        )
    else:
        unknown = [name for name in arms if name not in declared]
        if unknown:
            raise ValueError(
                f"unknown arm(s) {unknown}; plan {config.name!r} declares {sorted(declared)}"
            )
        selected = []
        for name in arms:
            arm = declared[name]
            if arm.kind not in IMPLEMENTED_ARM_KINDS:
                raise arm_not_implemented_error(arm)
            selected.append(arm)
        skipped = ()
    if not selected:
        raise ValueError(
            f"plan {config.name!r} declares no arm this invocation can run: every arm needs a "
            "parameter this slice does not implement, a grid point the plan does not bind, or the "
            "M5 training loop (see the skip reasons printed above)"
        )

    # Resolve every grid point *before* loading a model: a missing --rank must not cost a download.
    resolutions: list[tuple[ArmSpec, ArmCompression]] = [
        (
            arm,
            resolve_arm_compression(
                config,
                arm,
                bits=bits,
                rank=rank,
                granularity=granularity,
                group_size=group_size,
                symmetric=symmetric,
                axis=axis,
            ),
        )
        for arm in selected
    ]

    reported = [int(seed) for seed in config.seeds.reported]
    if seeds is None:
        chosen_seeds = [reported[0]]
    else:
        chosen_seeds = [int(seed) for seed in seeds]
        outside = [seed for seed in chosen_seeds if seed not in reported]
        if outside:
            raise ValueError(
                f"seed(s) {outside} are not in plan {config.name!r}'s reported seed list "
                f"{reported}: a run may only use the seeds the preregistration fixed"
            )
    if not chosen_seeds:
        raise ValueError("at least one seed is required")

    root = (
        Path(out_dir) if out_dir is not None else Path("artifacts") / "runs" / f"{config.name}-plan"
    )
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    model, tokenizer, model_id, model_revision, model_source = load_model(
        config, model_dir=model_dir
    )
    say(
        f"model: {model_id} revision={model_revision} source={model_source} "
        f"dtype={FROZEN_DTYPE} device={device}"
    )
    # The dataset *config* is part of the plan's pin: a repository exposing several configs cannot be
    # loaded without one (the first real cloud run failed here, after the platform had already built
    # the environment). An explicit --dataset-config still wins, for an ad-hoc run.
    resolved_dataset_config = dataset_config
    if resolved_dataset_config is None and perplexity_text is None:
        resolved_dataset_config = plan_dataset_ref(config, "test_perplexity").config
    documents, corpus = _load_corpus_documents(
        config,
        split="test",
        dataset_config=resolved_dataset_config,
        local_text=Path(perplexity_text) if perplexity_text is not None else None,
        max_documents=max_documents,
    )
    say(f"corpus: {corpus['name']} ({len(documents)} documents, split={corpus['split']})")

    weights, exclusions = _target_weights(model)
    if not weights:
        raise ValueError("no compressible linear weight found in the model: nothing to measure")
    pristine = {name: tensor.clone() for name, tensor in weights.items()}
    say(f"compressible linear weights: {len(weights)} (excluded: {len(exclusions)})")

    # A trainable arm *replaces* layers, which no state-dict restore can undo, so it runs on a fresh
    # copy of the loaded model. The non-trainable arms keep sharing one instance.
    pristine_model = copy.deepcopy(model)
    training_data: SequenceData | None = None
    training_corpus_provenance: dict[str, Any] = {}

    runs: list[ArmRun] = []
    for arm, compression in resolutions:
        applied = apply_arm(weights, arm, compression)
        for seed in chosen_seeds:
            seed_everything(seed, deterministic=True, threads=threads)
            arm_training: dict[str, Any] = {}
            if applied.trainable_init is not None:
                model = copy.deepcopy(pristine_model)
                target = root / arm.name / f"seed-{seed}"
                if training_data is None:
                    training_data, training_corpus_provenance = _training_corpus(
                        config,
                        tokenizer,
                        seq_len=int(config.training.seq_len) if config.training else seq_len,
                        max_documents=max_documents,
                        context=int(
                            getattr(
                                getattr(model, "config", None), "max_position_embeddings", 0
                            )
                            or 0
                        ),
                    )
                    say(
                        "training corpus: "
                        f"{training_data.train.shape[0]} train / {training_data.val.shape[0]} dev "
                        f"windows of {training_data.train.shape[1] - 1} tokens "
                        f"({training_data.checksum})"
                    )
                arm_started = time.perf_counter()
                arm_training, _ = _train_arm(
                    model,
                    arm=arm,
                    compression=compression,
                    applied=applied,
                    config=config,
                    data=training_data,
                    seed=seed,
                    threads=threads,
                    out_dir=target,
                    verbose=True,
                )
                arm_training = {**training_corpus_provenance, **arm_training}
            else:
                # Restore the pristine weights, then load the arm's compressed ones in place.
                model.load_state_dict(pristine, strict=False)
                if applied.state:
                    model.load_state_dict(applied.state, strict=False)
                arm_started = time.perf_counter()
            perplexity = token_level_perplexity(
                model,
                tokenizer,
                documents,
                seq_len=seq_len,
                max_tokens=max_tokens,
                device=device,
            )
            wall_time_s = time.perf_counter() - arm_started
            run_id = build_run_id(
                f"{config.name}-{arm.name}-seed{seed}",
                git_commit=git_info().commit,
                suffix="plan",
            )
            target = root / arm.name / f"seed-{seed}"
            target.mkdir(parents=True, exist_ok=True)
            manifest = _manifest_for(
                plan=config,
                plan_file=plan_file,
                arm=arm,
                seed=seed,
                applied=applied,
                exclusions=exclusions,
                perplexity=perplexity,
                model_id=model_id,
                model_revision=model_revision,
                model_source=model_source,
                device=device,
                threads=threads,
                corpus=corpus,
                training=arm_training or None,
                wall_time_s=wall_time_s,
                run_id=run_id,
                substitution=(
                    {
                        "model_dir": None if model_dir is None else str(model_dir),
                        "perplexity_text": None
                        if perplexity_text is None
                        else str(perplexity_text),
                    }
                    if substituted
                    else None
                ),
            )
            manifest_path = write_manifest(manifest, target / "run_manifest.json")
            metrics_path = target / "metrics.json"
            metrics_path.write_text(
                json.dumps(manifest.to_json_dict()["metrics"], indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            measurement_class = _arm_measurement_class(arm.kind)
            runs.append(
                ArmRun(
                    arm=arm.name,
                    kind=arm.kind,
                    seed=seed,
                    run_id=run_id,
                    manifest_path=manifest_path,
                    metrics_path=metrics_path,
                    measurement_class=measurement_class,
                    perplexity=float(perplexity["perplexity"]),
                    accounted_bytes=int(applied.accounted),
                )
            )
            say(
                f"arm {arm.name} seed {seed}: perplexity={perplexity['perplexity']:.6f} "
                f"tokens={perplexity['n_tokens']} accounted_bytes={applied.accounted} "
                f"measured_bytes={applied.measured} -> {manifest_path}"
            )

    model.load_state_dict(pristine, strict=False)

    aggregate: dict[str, Any] = {
        "plan.name": config.name,
        "plan.path": plan_file.as_posix(),
        "out_dir": root.as_posix(),
        "model.id": model_id,
        "model.revision_loaded": model_revision,
        "model.source": model_source,
        "model.dtype": FROZEN_DTYPE,
        "device": device,
        "threads": int(threads),
        "wall_time_s": round(time.perf_counter() - started, 6),
        "perplexity.method": PERPLEXITY_PROTOCOL,
        "perplexity.seq_len": int(seq_len),
        "perplexity.max_tokens": None if max_tokens is None else int(max_tokens),
        "perplexity.max_documents": None if max_documents is None else int(max_documents),
        "dataset": corpus,
        "arms_run": [
            {
                "arm": run.arm,
                "kind": run.kind,
                "seed": run.seed,
                "run_id": run.run_id,
                "manifest": run.manifest_path.as_posix(),
                "metrics": run.metrics_path.as_posix(),
                "measurement_class": run.measurement_class,
                "perplexity": run.perplexity,
                "accounted_bytes": run.accounted_bytes,
            }
            for run in runs
        ],
        "arms_skipped": list(skipped),
        "local_substitution": {
            "model_dir": None if model_dir is None else str(model_dir),
            "perplexity_text": None if perplexity_text is None else str(perplexity_text),
            "is_plan_measurement": not substituted,
        },
    }
    aggregate_path = root / "metrics.json"
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    result_line = {
        "run_id": runs[0].run_id if runs else None,
        "plan": config.name,
        "status": "success",
        "model_id": model_id,
        "model_revision": model_revision,
        "out_dir": root.as_posix(),
        "aggregate_metrics": aggregate_path.as_posix(),
        "arms_run": [
            {
                "arm": run.arm,
                "seed": run.seed,
                "manifest": run.manifest_path.as_posix(),
                "metrics": run.metrics_path.as_posix(),
                "measurement_class": run.measurement_class,
                "perplexity": run.perplexity,
            }
            for run in runs
        ],
        "arms_skipped": [entry["arm"] for entry in skipped],
        "is_plan_measurement": not substituted,
    }
    print(RESULT_LINE_PREFIX + json.dumps(result_line, sort_keys=True, separators=(",", ":")))

    return PlanRunResult(
        plan_name=config.name,
        plan_path=plan_file,
        out_dir=root,
        model_id=model_id,
        model_revision=model_revision,
        model_source=model_source,
        runs=tuple(runs),
        skipped=skipped,
        aggregate_metrics_path=aggregate_path,
        metrics=aggregate,
    )
