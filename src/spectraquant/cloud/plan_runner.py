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

* **Implemented trainable arms** — the layer-replacement arms the loop optimises:

  - the project's comparator arms ``loftq`` and ``lr_qat``;
  - the **eight frozen M3 arms** of ``configs/m3/tier1_smollm2_135m.yaml``
    (``docs/decisions/design-m3-arms.md``): ``r1_fp16_lora``, ``r1_std_2bit``, ``r1_loftq_2bit``,
    ``r1_loftq_2bit_t1``, ``r2_fp16``, ``r2_rtn_4bit``, ``r2_lrqat_4bit``, ``r2_fullqat_4bit``.

  Trainable arms run a recorded learning-rate search on the **development** split when the plan
  declares ``training.learning_rate_grid``, then the full schedule at the selected rate; the test
  split is never read by the search (``AGENTS.md`` §4.6). A run that violates the predeclared
  divergence rule (perplexity > 1000, non-finite, or a loss that has not decreased after 100 steps)
  is recorded ``training.divergent: true`` with its trigger rather than as a quality number.

* **Not implemented here**: ``quant_then_residual``, ``proxy_allocated``,
  ``proxy_allocated_regularized``, ``qlora`` and ``spectraquant``. Those raise
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
    CodebookSpec,
    QuantSpec,
    accounted_bytes,
    codebook_accounted_bytes,
    fake_quantize,
    fake_quantize_codebook,
    measure_serialized_bytes,
    quant_params,
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
    "M3_ARM_KINDS",
    "PERPLEXITY_PROTOCOL",
    "TRAINABLE_ARM_KINDS",
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
#: trainable factors), or by :class:`StraightThroughQuantizedLinear` for the full-QAT arm, the loop
#: optimises them, and the deployed weight is the arm's own representation.
TRAINABLE_ARM_KINDS: frozenset[str] = frozenset(
    {
        "loftq",
        "lr_qat",
        # The frozen M3 arm set (docs/decisions/design-m3-arms.md).
        "r1_fp16_lora",
        "r1_std_2bit",
        "r1_loftq_2bit",
        "r1_loftq_2bit_t1",
        "r2_lrqat_4bit",
        "r2_fullqat_4bit",
    }
)

#: The frozen M3 arm set (``configs/m3/tier1_smollm2_135m.yaml``; reproduction-plan.md section 3).
M3_ARM_KINDS: frozenset[str] = frozenset(
    {
        "r1_fp16_lora",
        "r1_std_2bit",
        "r1_loftq_2bit",
        "r1_loftq_2bit_t1",
        "r2_fp16",
        "r2_rtn_4bit",
        "r2_lrqat_4bit",
        "r2_fullqat_4bit",
    }
)

#: The M3 arms whose base weight is the 2-bit NF codebook rather than the uniform quantizer.
R1_CODEBOOK_ARM_KINDS: frozenset[str] = frozenset(
    {"r1_std_2bit", "r1_loftq_2bit", "r1_loftq_2bit_t1"}
)

IMPLEMENTED_ARM_KINDS: frozenset[str] = (
    frozenset(
        {
            "fp16_reference",
            "ptq_uniform",
            "low_rank_only",
            "rank_then_quant",
            # The two eval-only M3 reference arms (no training, no adapter).
            "r2_fp16",
            "r2_rtn_4bit",
        }
    )
    | TRAINABLE_ARM_KINDS
)

#: Identifier of the perplexity protocol implemented here (see the module docstring).
PERPLEXITY_PROTOCOL = "non-overlapping-window-token-ce-v1"

#: Perplexity above which the model is not degraded but broken (it assigns ~zero probability to the
#: held-out text). Such a number is recorded with ``perplexity_degenerate: true`` so no analysis can
#: treat it as a quality measurement: the observed rank-8 truncation arm scored 1.4e16.
DEGENERATE_PERPLEXITY = 1.0e6

#: The frozen evaluation dtype: ``eval-protocol.md`` §6.3 freezes float32 for cross-arm identity.
FROZEN_DTYPE = "float32"

#: Divergence rule (reproduction-plan.md §5.4, predeclared): a run whose perplexity exceeds this, is
#: non-finite, or whose loss has not decreased after :data:`DIVERGENCE_LOSS_WINDOW` steps is recorded
#: as **divergent** (``training.divergent: true`` with the trigger) rather than as a quality number.
DIVERGENCE_PERPLEXITY = 1000.0

#: Step count after which "the loss has not decreased" is judged (reproduction-plan.md §5.4: "100
#: steps"). A shorter run is never judged divergent on the loss trajectory alone.
DIVERGENCE_LOSS_WINDOW = 100

#: Length of the recorded learning-rate-search prefix, as a fraction of the plan's ``steps``. The
#: search is a *selection* step on the development split, not a result: it reuses the same schedule
#: shape at a fraction of its cost (a documented constant, ``design-m3-arms.md`` §3.4). The full run
#: always uses the plan's own ``steps``.
LR_SEARCH_PREFIX_FRACTION = 1.0 / 50.0

#: Lower/upper clamp of the search prefix in steps: at least one optimiser step (so a candidate
#: produces a measurable dev loss) and at most this many (so the search stays cheap at any ``steps``).
LR_SEARCH_MIN_STEPS = 1
LR_SEARCH_MAX_STEPS = 20

#: ``compression.method`` per arm kind, restricted to the manifest schema's enum.
#:
#: The M3 arm kinds have no one-to-one enum entry, so each is mapped to the closest legal value and
#: the arm's own identity is preserved in ``metrics["plan.arm_kind"]`` (which carries the arm kind
#: verbatim). The choices, and why:
#:
#: * ``r1_fp16_lora`` -> ``none``: nothing is quantized; it is the upper reference.
#: * ``r1_std_2bit`` -> ``rtn``: the base is a round-to-nearest 2-bit codebook grid with the plain
#:   QLoRA/fixup adapter; the enum has no ``qlora`` and this arm must not be labelled ``loftq``
#:   (its initialisation is not LoftQ).
#: * ``r1_loftq_2bit`` / ``r1_loftq_2bit_t1`` -> ``loftq``.
#: * ``r2_fp16`` -> ``none``: nothing is quantized; eval-only reference.
#: * ``r2_rtn_4bit`` -> ``rtn``.
#: * ``r2_lrqat_4bit`` -> ``lr-qat``.
#: * ``r2_fullqat_4bit`` -> ``lr-qat``: full-model QAT is the LR-QAT paper's own upper reference
#:   arm, trained through the same 4-bit g128 quantizer; the enum has no plain ``qat`` value.
_METHOD_BY_KIND: dict[str, str] = {
    "fp16_reference": "none",
    "ptq_uniform": "rtn",
    "low_rank_only": "svd",
    "rank_then_quant": "rtn",
    "loftq": "svd",
    "lr_qat": "rtn",
    "r1_fp16_lora": "none",
    "r1_std_2bit": "rtn",
    "r1_loftq_2bit": "loftq",
    "r1_loftq_2bit_t1": "loftq",
    "r2_fp16": "none",
    "r2_rtn_4bit": "rtn",
    "r2_lrqat_4bit": "lr-qat",
    "r2_fullqat_4bit": "lr-qat",
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
    """The grid point one arm run uses, validated against the plan's predeclared grid.

    Attributes:
        bits: target bit width, or ``None`` when the arm quantizes nothing.
        rank: low-rank budget, or ``None`` when the arm has no adapter.
        granularity: ``per_tensor`` | ``per_channel`` | ``per_group`` for the uniform quantizer.
        group_size: quantization group size for ``per_group``.
        symmetric: whether the uniform quantizer's codes are symmetric.
        axis: channel axis for the uniform quantizer's blocks.
        quantizer: ``"uniform"`` (the uniform affine quantizer) or ``"codebook"`` (the 2-bit
            NF-style codebook of the R1 arms).
        block_size: the codebook block size (elements sharing one absmax scale); unused by the
            uniform quantizer.
        loftq_iterations: alternating steps of the LoftQ initialisation, or ``None`` when the arm
            does not use the schedule.
        step_size_lr: the learning rate of the LR-QAT step size ``s`` (``0.0`` means **frozen** —
            a real grid point), or ``None`` when the arm has no step size at all. The manifest must
            keep ``0.0`` and ``None`` apart.
    """

    bits: int | None
    rank: int | None
    granularity: str
    group_size: int | None
    symmetric: bool
    axis: int
    quantizer: str = "uniform"
    block_size: int = 64
    loftq_iterations: int | None = None
    step_size_lr: float | None = None

    def quant_spec(self, bits: int) -> QuantSpec:
        """Return the :class:`QuantSpec` for ``bits`` at this arm's granularity."""
        return QuantSpec(
            bits=bits,
            granularity=self.granularity,
            group_size=self.group_size,
            symmetric=self.symmetric,
            axis=self.axis,
        )

    def uniform_quant_spec(self) -> QuantSpec | None:
        """Return this arm's :class:`QuantSpec`, or ``None`` when it is not a uniform arm."""
        if self.quantizer != "uniform" or self.bits is None:
            return None
        return self.quant_spec(int(self.bits))

    def codebook_spec(self) -> CodebookSpec:
        """Return the 2-bit NF codebook spec of an R1 arm.

        Raises:
            ValueError: the arm is not a codebook arm, or its bit width is not the frozen 2 bits.
        """
        if self.quantizer != "codebook" or self.bits is None:
            raise ValueError(
                "codebook_spec() is only defined for a 2-bit codebook arm "
                f"(quantizer={self.quantizer!r}, bits={self.bits!r})"
            )
        return CodebookSpec(bits=int(self.bits), kind="nf", block_size=int(self.block_size))


@dataclass(frozen=True)
class ArmRun:
    """One executed (arm, seed) pair and where its records were written.

    Attributes:
        perplexity: the measured perplexity, or ``None`` when the protocol could not produce a finite
            number (the manifest then records the non-finite measurement and, under the frozen
            divergence rule, ``training.divergent``).
        divergent: whether the frozen divergence rule fired for this arm (reproduction-plan.md §5.4).
    """

    arm: str
    kind: str
    seed: int
    run_id: str
    manifest_path: Path
    metrics_path: Path
    measurement_class: int | None
    perplexity: float | None
    accounted_bytes: int
    divergent: bool = False


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
    fp16 references (``fp16_reference``, ``r2_fp16``) and the unquantized LoRA reference
    (``r1_fp16_lora``) measure no compression at all and carry no class. A pure low-rank truncation
    executes an approximate artifact in float, which the frozen taxonomy has no separate class for;
    it is reported as class 2 with the compression block recording that no quantization was applied,
    and the mismatch between the label and the arm is an open question for the taxonomy's owner.
    """
    if kind in {"fp16_reference", "r2_fp16", "r1_fp16_lora"}:
        return None
    return 2


def _positive_point(
    bound: dict[str, int | float], key: str, *, arm: str, default: int | None = None
) -> int:
    """Return a positive integer point value, or ``default`` when the plan does not bind it.

    The plan schema validates the values it declares; this function covers the case where a *caller*
    passes the value through the CLI instead, where no schema ran.
    """
    value = bound.get(key, default)
    if value is None:
        raise ValueError(f"arm {arm!r} does not bind {key!r} and no default is defined for it")
    if isinstance(value, bool) or int(value) <= 0:
        raise ValueError(f"arm {arm!r} binds {key}={value!r}, which is not a positive integer")
    return int(value)


def _resolve_m3_arm(
    plan: PlanConfig,
    arm: ArmSpec,
    *,
    bits: int | None,
    rank: int | None,
    granularity: str,
    group_size: int | None,
    symmetric: bool,
    axis: int,
) -> ArmCompression:
    """Resolve one frozen M3 arm's grid point from the plan's own binding.

    The eight M3 arms are fully specified by ``configs/m3/tier1_smollm2_135m.yaml``: each binds the
    point it runs at, and the plan's ``grid`` carries the values that may be used. A CLI ``--bits`` /
    ``--rank`` / ``--group-size`` may still be given (a differently parametrised ad-hoc run, as for
    the older arms), but it must agree with the plan's binding - the frozen plan is the declaration
    and an override that changes it silently is exactly the defect this runner has a regression test
    for.

    Granularity is *not* taken from the CLI for these arms: an R1 arm pins the 2-bit NF codebook and
    an R2 arm the uniform 4-bit per-group g128 quantizer, both of which are part of the arm's frozen
    identity (``docs/decisions/design-m3-arms.md`` §2). A contradicting ``--granularity`` is refused.

    Raises:
        ValueError: a required point is missing, or a value is outside the plan's grid.
    """
    kind = arm.kind
    bound = dict(getattr(arm, "point", {}) or {})

    def _pinned(name: str, cli: Any) -> Any:
        """Reconcile a plan binding with an explicit CLI value of the same name."""
        bound_value = bound.get(name)
        if cli is not None and bound_value is not None and int(cli) != int(bound_value):
            raise ValueError(
                f"arm {arm.name!r} binds {name}={bound_value} but {name}={cli} was passed: the "
                "plan's frozen point is authoritative for an M3 arm"
            )
        return bound_value if bound_value is not None else cli

    resolved_bits = _pinned("bits", bits)
    resolved_rank = _pinned("rank", rank)

    if granularity != "per_channel":
        expected = "codebook" if kind in R1_CODEBOOK_ARM_KINDS else "per_group"
        if kind in R1_CODEBOOK_ARM_KINDS:
            raise ValueError(
                f"arm {arm.name!r} uses the 2-bit NF codebook, not granularity={granularity!r}: "
                "its quantizer is part of the frozen arm definition"
            )
        if granularity != expected:
            raise ValueError(
                f"arm {arm.name!r} pins granularity={expected!r}, not {granularity!r}: its "
                "quantizer is part of the frozen arm definition"
            )
    if granularity == "per_group" and group_size is not None:
        pinned_group = bound.get("group_size")
        if pinned_group is not None and int(group_size) != int(pinned_group):
            raise ValueError(
                f"arm {arm.name!r} binds group_size={pinned_group} but group_size={group_size} was "
                "passed: the plan's frozen point is authoritative for an M3 arm"
            )

    # ---- R1: the LoftQ design (rank 16 model-level arms) -------------------------------------
    if kind in R1_CODEBOOK_ARM_KINDS or kind == "r1_fp16_lora":
        if resolved_rank is None:
            raise ValueError(
                f"arm {arm.name!r} is an R1 model-level arm and needs a rank; the plan binds none "
                f"and none was passed (plan grid: {plan.grid.ranks})"
            )
        resolved_rank = _grid_point(
            resolved_rank, plan.grid.ranks, what="rank", arm=arm.name, plan=plan
        )
        if kind == "r1_fp16_lora":
            # No quantization at all: the base is the fp32 weight itself.
            return ArmCompression(
                bits=None,
                rank=resolved_rank,
                granularity=granularity,
                group_size=None,
                symmetric=symmetric,
                axis=axis,
            )
        if resolved_bits is None:
            raise ValueError(
                f"arm {arm.name!r} is a 2-bit codebook arm and binds no bit width; the frozen arm "
                f"pins bits=2 (plan grid: {plan.grid.bits})"
            )
        resolved_bits = _grid_point(
            resolved_bits, plan.grid.bits, what="bits", arm=arm.name, plan=plan
        )
        if int(resolved_bits) != 2:
            raise ValueError(
                f"arm {arm.name!r} binds bits={resolved_bits}, but the R1 codebook is defined at 2 "
                "bits only (docs/decisions/design-m3-arms.md §3.1)"
            )
        block_size = _positive_point(bound, "block_size", arm=arm.name, default=64)
        iterations = None
        if kind in {"r1_loftq_2bit", "r1_loftq_2bit_t1"}:
            iterations = _positive_point(bound, "loftq_iterations", arm=arm.name, default=5)
        return ArmCompression(
            bits=int(resolved_bits),
            rank=resolved_rank,
            granularity="codebook",
            group_size=block_size,
            symmetric=True,
            # The codebook's own blocking moves the last axis to the row-major flatten
            # (`CodebookSpec.axis = -1`), which is the reference's behaviour; `axis` here records
            # that layout rather than the uniform quantizer's channel axis.
            axis=-1,
            quantizer="codebook",
            block_size=block_size,
            loftq_iterations=iterations,
        )

    # ---- R2: the LR-QAT design (4-bit symmetric, group 128) -----------------------------------
    if kind == "r2_fp16":
        if granularity == "per_group":
            raise ValueError(
                f"arm {arm.name!r} quantizes nothing, so --granularity per_group is not applicable"
            )
        return ArmCompression(
            bits=None,
            rank=None,
            granularity="per_channel",
            group_size=None,
            symmetric=symmetric,
            axis=axis,
        )

    if resolved_bits is None:
        raise ValueError(
            f"arm {arm.name!r} is an R2 arm and binds no bit width; the frozen design pins bits=4 "
            f"(plan grid: {plan.grid.bits})"
        )
    resolved_bits = _grid_point(resolved_bits, plan.grid.bits, what="bits", arm=arm.name, plan=plan)
    if int(resolved_bits) != 4:
        raise ValueError(
            f"arm {arm.name!r} binds bits={resolved_bits}, but the R2 uniform arms are defined at 4 "
            "bits only (docs/decisions/design-m3-arms.md §3.1)"
        )
    bound_group = bound.get("group_size")
    if bound_group is None:
        raise ValueError(
            f"arm {arm.name!r} is an R2 arm and binds no group_size; the frozen design pins 128 "
            f"(plan grid: {plan.grid.group_sizes})"
        )
    resolved_group = _grid_point(
        int(bound_group), plan.grid.group_sizes, what="group_size", arm=arm.name, plan=plan
    )
    if int(resolved_group) != 128:
        raise ValueError(
            f"arm {arm.name!r} binds group_size={resolved_group}, but the R2 uniform arms are "
            "defined at group size 128 only (docs/decisions/design-m3-arms.md §3.1)"
        )

    resolved_step_size_lr: float | None = None
    if kind == "r2_lrqat_4bit":
        if resolved_rank is None:
            raise ValueError(
                f"arm {arm.name!r} is the LR-QAT arm and needs a rank; the plan binds none and "
                f"none was passed (plan grid: {plan.grid.ranks})"
            )
        resolved_rank = _grid_point(
            resolved_rank, plan.grid.ranks, what="rank", arm=arm.name, plan=plan
        )
        if "step_size_lr" not in bound:
            # `0.0` (frozen) and "absent" are different grid points and the manifest must keep them
            # apart, so the arm cannot leave it unbound: a missing value would be recorded as
            # "frozen" and could never be told from "not applicable".
            raise ValueError(
                f"arm {arm.name!r} is the LR-QAT arm and binds no step_size_lr; its learned step "
                "size must declare a learning rate (0 = frozen), because the manifest keeps "
                "'frozen' and 'not applicable' apart"
            )
        step_size_lr = float(bound["step_size_lr"])
        if not math.isfinite(step_size_lr) or step_size_lr < 0.0:
            raise ValueError(
                f"arm {arm.name!r} binds step_size_lr={step_size_lr!r}, which is not a finite "
                "non-negative rate (0 = frozen)"
            )
        resolved_step_size_lr = step_size_lr
    elif "step_size_lr" in bound:
        raise ValueError(
            f"arm {arm.name!r} binds step_size_lr={bound['step_size_lr']!r} but has no learned step "
            "size; only r2_lrqat_4bit uses it"
        )
    elif resolved_rank is not None:
        raise ValueError(
            f"arm {arm.name!r} does not use a rank; --rank={resolved_rank} is not applicable to it"
        )

    return ArmCompression(
        bits=int(resolved_bits),
        rank=resolved_rank,
        granularity="per_group",
        group_size=int(resolved_group),
        symmetric=True,
        axis=axis,
        step_size_lr=resolved_step_size_lr,
    )


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

    if kind in M3_ARM_KINDS:
        # The frozen M3 arms carry their whole grid point in the plan (bits, rank, block size, group
        # size, LoftQ steps, step-size lr), so they are resolved by their own rules: an R2 arm pins
        # per-group g128, and an R1 arm pins the 2-bit NF codebook.
        return _resolve_m3_arm(
            plan,
            arm,
            bits=bits,
            rank=rank,
            granularity=granularity,
            group_size=group_size,
            symmetric=symmetric,
            axis=axis,
        )

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
    device: str = "cpu",
) -> tuple[Any, Any, str, str, str]:
    """Load the plan's pretrained model (or an explicit local stand-in) and its tokenizer.

    The model is moved to ``device`` here rather than at each call site: the evaluation moves the
    *inputs* to the device, so a model left on CPU while the inputs go to CUDA fails inside the first
    embedding lookup ("Expected all tensors to be on the same device"), which is exactly what the
    first GPU attempt did.

    Args:
        plan: the loaded plan.
        model_dir: local stand-in model directory (a path check, not a plan measurement).
        dtype: torch dtype *name*; the frozen protocol dtype by default.
        device: ``"cpu"`` or ``"cuda"``; a CUDA request is refused when the build has no CUDA rather
            than silently computing on CPU (``AGENTS.md`` section 2.2).

    Returns:
        ``(model, tokenizer, model_id, model_revision, model_source)``.

    Raises:
        PinnedRevisionError: the plan's model revision is not an immutable commit.
        ModelRevisionMismatch: the loaded commit is not the pin.
        ImportError: the ``models`` extra is missing.
        FileNotFoundError: ``model_dir`` does not exist.
        RuntimeError: ``device`` is CUDA but this torch build has no CUDA support.
    """
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "device=cuda was requested but this torch build has no CUDA support "
            f"(torch {torch.__version__}): refusing to compute on CPU while the manifest would "
            "claim a GPU"
        )
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
        return _finish(model, tokenizer, str(directory), revision, "local_dir", device=device)

    verify_pinned_revision(model_ref.id, model_ref.revision, repo_type="model")
    model = _from_pretrained(
        transformers.AutoModelForCausalLM, model_ref.id, revision=model_ref.revision
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_ref.id, revision=model_ref.revision
    )
    revision = revision_of(model, requested=model_ref.revision, source="hub")
    return _finish(model, tokenizer, model_ref.id, revision, "hub", device=device)


def _finish(
    model: Any, tokenizer: Any, model_id: str, revision: str, source: str, *, device: str = "cpu"
) -> tuple[Any, Any, str, str, str]:
    model.eval()
    model.to(device)
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
    allow_non_finite: bool = False,
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
        allow_non_finite: when ``True``, a non-finite cross-entropy is *reported*
            (``perplexity: None``, ``perplexity_non_finite: True``) instead of raising. The frozen
            divergence rule (reproduction-plan.md §5.4) must be able to record a broken artifact as
            divergent, and it cannot do that if the measurement raises before the record is written.
            The default keeps the raising behaviour for every other caller.

    Returns:
        ``{"perplexity", "perplexity_degenerate", "mean_ce_loss", "nll_sum", "n_tokens",
        "n_windows", "n_documents", "seq_len", "protocol"}``; with ``allow_non_finite=True`` a
        non-finite loss adds ``"perplexity_non_finite": True`` and ``perplexity`` becomes ``None``.

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
        if allow_non_finite:
            # The frozen divergence rule must be able to *record* a non-finite measurement as
            # divergent, so the number is reported as absent rather than raising before the write.
            return {
                "perplexity": None,
                "perplexity_non_finite": True,
                "perplexity_degenerate": False,
                "mean_ce_loss": None,
                "nll_sum": None,
                "n_tokens": int(total_tokens),
                "n_windows": int(windows),
                "n_documents": int(scored_documents),
                "seq_len": int(window),
                "protocol": PERPLEXITY_PROTOCOL,
            }
        raise ValueError(
            f"the model produced a non-finite cross-entropy over {total_tokens} tokens: this is a "
            "failed measurement, not a perplexity - check the compressed artifact"
        )
    mean_ce = total_nll / total_tokens
    perplexity = float(math.exp(mean_ce)) if mean_ce < 700.0 else float("inf")
    if allow_non_finite and not math.isfinite(perplexity):
        # `exp` overflows to `inf` long before the JSON encoder would choke; the record keeps the
        # cross-entropy (a finite number) and marks the perplexity as not measured, so no file in the
        # run carries a non-standard `Infinity` literal.
        return {
            "perplexity": None,
            "perplexity_non_finite": True,
            "perplexity_degenerate": False,
            "mean_ce_loss": float(mean_ce),
            "nll_sum": float(total_nll),
            "n_tokens": int(total_tokens),
            "n_windows": int(windows),
            "n_documents": int(scored_documents),
            "seq_len": int(window),
            "protocol": PERPLEXITY_PROTOCOL,
        }
    degenerate = perplexity > DEGENERATE_PERPLEXITY
    return {
        "perplexity": perplexity,
        "perplexity_non_finite": not math.isfinite(perplexity),
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
    :class:`~spectraquant.training.low_rank.QuantizedPlusLowRankLinear` (or, for the full-QAT arm,
    :class:`~spectraquant.training.low_rank.StraightThroughQuantizedLinear`) and then optimises
    them, so its initialisation is a structural change, not a ``load_state_dict`` payload.

    Attributes:
        bases: ``{weight name: frozen quantized main weight}`` (empty for the full-QAT style).
        factors: ``{weight name: (A, B)}`` starting factors (empty for the full-QAT style).
        iterations: alternating LoftQ steps, or ``None`` when the arm does not use the schedule.
        style: ``"quantized_plus_low_rank"`` or ``"straight_through"`` - which layer class the arm's
            target set is replaced with.
        spec: the quantizer the installed layers use (a :class:`QuantSpec`); the full-QAT arm passes
            it to its straight-through layers and the LR-QAT arm to nothing (its base is already
            quantized).
        step_sizes: ``{weight name: initial step size}`` for the LR-QAT arm; empty otherwise.
        dense_weights: ``{weight name: starting dense weight}`` for the full-QAT style; empty
            otherwise.
    """

    bases: dict[str, Tensor]
    factors: dict[str, tuple[Tensor, Tensor]]
    iterations: int | None = None
    style: str = "quantized_plus_low_rank"
    spec: QuantSpec | None = None
    step_sizes: dict[str, float] = field(default_factory=dict)
    dense_weights: dict[str, Tensor] = field(default_factory=dict)


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


def _initialisation_description(arm: ArmSpec, init: _TrainableInit) -> str:
    """Human-readable description of what the arm applied *before* step 0 (manifest provenance)."""
    if init.style == "straight_through":
        return f"{arm.kind}: every target weight trainable through the quantizer (straight-through)"
    suffix = f" from {init.iterations} alternating LoftQ steps" if init.iterations else ""
    step = "" if not init.step_sizes else " + trainable step size s"
    return f"{arm.kind}: frozen quantized base + factors{suffix}{step}"


def _install_trainable_layers(model: nn.Module, init: _TrainableInit, *, device: str) -> None:
    """Replace the arm's target layers with the class its ``init.style`` names, and move to device.

    ``init.bases`` is keyed by the *weight* name the accounting uses (``...q_proj.weight``), while the
    replacement addresses the *module* (``...q_proj``). The two are the same set of layers under two
    naming conventions, and the conversion is asserted rather than assumed.

    Every parameter **outside** the target modules is frozen (``requires_grad_(False)``): the frozen
    recipe trains only the arm's own parameters and keeps embeddings, the LM head and the norms at
    their pretrained values (reproduction-plan.md §3.2). Leaving them trainable would also have made
    the optimiser's parameter count depend on the model rather than on the arm.

    Raises:
        ValueError: two target weights map to the same module (the replacement would drop one).
    """
    from spectraquant.training.low_rank import (
        replace_linears_quantized,
        replace_linears_straight_through,
    )

    if init.style == "straight_through":
        if init.spec is None:
            raise ValueError("a straight-through initialisation must carry its QuantSpec")
        weights = {_module_of(name): tensor for name, tensor in init.dense_weights.items()}
        if len(weights) != len(init.dense_weights):
            raise ValueError(
                "two target weights map to the same module, so the replacement would silently "
                f"drop one: {sorted(init.dense_weights)}"
            )
        replace_linears_straight_through(model, spec=init.spec, weights=weights)
    else:
        bases = {_module_of(name): tensor for name, tensor in init.bases.items()}
        factors = {_module_of(name): pair for name, pair in init.factors.items()}
        step_sizes = {_module_of(name): value for name, value in init.step_sizes.items()}
        if len(bases) != len(init.bases):
            raise ValueError(
                "two target weights map to the same module, so the replacement would silently "
                f"drop one: {sorted(init.bases)}"
            )
        replace_linears_quantized(
            model, bases=bases, factors=factors, step_sizes=step_sizes or None
        )
        weights = bases

    targets = set(weights)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(
            any(name == module or name.startswith(f"{module}.") for module in targets)
        )
    # The layers were built from CPU tensors (the primitives are CPU-only), so they must be moved
    # onto the model's device before the loop batches data through them.
    model.to(device)


def _lr_qat_optimizer(
    model: nn.Module, *, lr_ab: float, lr_s: float | None
) -> torch.optim.Optimizer:
    """The LR-QAT AdamW: ``beta = (0.9, 0.95)``, weight decay 0, separate lr for the step size.

    ``Table B1`` of the LR-QAT paper (reproduction-plan.md §3.2) fixes ``beta = (0.9, 0.95)`` and
    weight decay ``0`` for ``A``, ``B`` and ``s``. The step size gets its own parameter group so
    ``lr_s = 0`` means **frozen** rather than "the same rate as the factors". It is used by the two
    R2 arms whose frozen recipe pins it - ``r2_lrqat_4bit`` (which also needs the separate ``lr_s``
    group) and ``r2_fullqat_4bit``; the R1 arms keep the library's AdamW defaults, because the LoftQ
    recipe (reproduction-plan.md §3.1) pins no ``beta``.
    """
    step_params: list[nn.Parameter] = []
    other_params: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith("step_size"):
            step_params.append(parameter)
        else:
            other_params.append(parameter)
    if not other_params:
        raise ValueError("the model has no trainable parameter to optimise")
    groups: list[dict[str, Any]] = [{"params": other_params, "lr": float(lr_ab)}]
    if step_params:
        groups.append({"params": step_params, "lr": float(lr_ab if lr_s is None else lr_s)})
    return torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)


def _step_size_values(model: nn.Module) -> list[float]:
    """The current value of every LR-QAT step size in ``model``, in parameter-name order."""
    return [
        float(parameter.detach())
        for name, parameter in model.named_parameters()
        if name.endswith("step_size")
    ]


def _divergence_trigger(losses: Sequence[float], perplexity: float | None) -> str | None:
    """Apply the frozen divergence rule (reproduction-plan.md §5.4).

    An arm is **divergent** when its perplexity exceeds :data:`DIVERGENCE_PERPLEXITY`, is NaN or
    non-finite, or its loss has not decreased after :data:`DIVERGENCE_LOSS_WINDOW` steps. The rule
    is evaluated here so the manifest can record the *trigger* rather than the number.

    Returns:
        The trigger as a human-readable string, or ``None`` when the arm is not divergent.
    """
    if perplexity is None:
        return "the perplexity measurement was non-finite (NaN/inf over the scored tokens)"
    if not math.isfinite(float(perplexity)):
        return f"perplexity is not a finite number ({perplexity!r})"
    if float(perplexity) > DIVERGENCE_PERPLEXITY:
        return (
            f"perplexity {float(perplexity):.6g} exceeds the predeclared threshold "
            f"{DIVERGENCE_PERPLEXITY:g}"
        )
    if losses:
        if any(not math.isfinite(float(value)) for value in losses):
            return "the training loss trajectory contains a non-finite value"
        if len(losses) > DIVERGENCE_LOSS_WINDOW and float(losses[-1]) >= float(losses[0]):
            return (
                f"the loss did not decrease after {DIVERGENCE_LOSS_WINDOW} steps "
                f"({float(losses[0]):.6g} -> {float(losses[-1]):.6g})"
            )
    return None


def _search_prefix_steps(steps: int) -> int:
    """Length of the learning-rate-search prefix: a small, documented fraction of the schedule."""
    return max(
        LR_SEARCH_MIN_STEPS,
        min(LR_SEARCH_MAX_STEPS, round(int(steps) * LR_SEARCH_PREFIX_FRACTION)),
    )


def _search_learning_rate(
    pristine_model: nn.Module,
    init: _TrainableInit,
    *,
    config: PlanConfig,
    arm: ArmSpec,
    data: SequenceData,
    seed: int,
    threads: int,
    device: str,
    out_dir: Path,
    spec: QuantSpec | None,
    verbose: bool,
) -> tuple[list[dict[str, Any]], float]:
    """Search the plan's learning-rate grid on the **development** split only.

    Each candidate is trained for :func:`_search_prefix_steps` steps from a fresh copy of the pristine
    model and the arm's own initialisation, then scored by the dev cross-entropy. The full schedule
    is not run here, the test split is never read (the corpus argument is the loop's train/dev
    tensor pair, and the runner's perplexity documents are not passed in), and every candidate's dev
    perplexity is returned so the manifest can record the whole search, not just the winner.

    The prefix scales its warmup proportionally to the full schedule; without that, a 100-step warmup
    would leave every candidate at nearly the same effective rate and the search would be uninformative.

    Returns:
        ``(records, selected_rate)``. ``records`` is one ``{candidate, steps, warmup_steps,
        dev_loss, dev_perplexity}`` mapping per grid point; ``selected_rate`` is the candidate with
        the lowest dev perplexity, or the plan's pinned ``learning_rate`` when no candidate produced
        a finite one.
    """
    from spectraquant.training.loop import LoopConfig, evaluate_sequences, train_language_model

    schedule = config.training
    assert schedule is not None
    prefix = _search_prefix_steps(int(schedule.steps))
    full_steps = max(1, int(schedule.steps))
    warmup_prefix = round(int(schedule.warmup_steps) * prefix / full_steps)
    records: list[dict[str, Any]] = []
    for index, candidate in enumerate(schedule.learning_rate_grid):
        search_model = copy.deepcopy(pristine_model)
        _install_trainable_layers(search_model, init, device=device)
        loop = LoopConfig(
            batch_size=int(schedule.batch_size),
            lr=float(candidate),
            grad_clip=float(schedule.grad_clip),
            seed=int(seed),
            threads=int(threads),
            warmup_steps=warmup_prefix,
            checkpoint_every=None,
            eval_every=None,
        )
        train_language_model(
            search_model,
            method=arm.kind,
            steps=prefix,
            output_dir=out_dir / "lr-search" / f"candidate-{index:02d}",
            data=data,
            loop=loop,
            task_loss=_hf_task_loss,
            spec=spec,
            measurement_class=2,
            write=False,
            verbose=False,
        )
        dev_loss = float(
            evaluate_sequences(search_model, data.val, batch_size=int(schedule.batch_size))
        )
        finite = math.isfinite(dev_loss)
        records.append(
            {
                "candidate": float(candidate),
                "steps": prefix,
                "warmup_steps": warmup_prefix,
                "dev_loss": dev_loss if finite else None,
                "dev_perplexity": float(math.exp(dev_loss))
                if finite and dev_loss < 700.0
                else None,
            }
        )
        if verbose:
            print(
                f"lr search {arm.name} seed {seed}: candidate={candidate:g} "
                f"prefix={prefix} dev_loss={dev_loss:.6f}"
            )
    usable = [record for record in records if record["dev_perplexity"] is not None]
    if not usable:
        # Every candidate diverged on the prefix: no grid point can be *selected* from the data, so
        # the plan's declared fallback rate is used and the search record shows why.
        return records, float(schedule.learning_rate)
    best = min(usable, key=lambda record: (record["dev_perplexity"], record["candidate"]))
    return records, float(best["candidate"])


def _train_arm(
    model: nn.Module,
    *,
    device: str,
    arm: ArmSpec,
    compression: ArmCompression,
    applied: _Applied,
    config: PlanConfig,
    data: SequenceData,
    seed: int,
    threads: int,
    out_dir: Path,
    verbose: bool,
    documents: Sequence[str],
    tokenizer: Any,
    eval_seq_len: int,
    max_tokens: int | None,
    learning_rate: float,
    lr_search: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], Any]:
    """Train one arm and return the training provenance and the loop's result.

    The target layers are replaced by :class:`QuantizedPlusLowRankLinear` (frozen quantized base plus
    trainable factors) or :class:`StraightThroughQuantizedLinear` (all weights trainable through the
    quantizer) before the first step, so the optimiser sees exactly the artifact the arm stores. The
    dev split drives any intermediate evaluation; the test split is not read here.

    Args:
        model: the loaded model, modified in place.
        device: the device the model lives on; the installed layers are moved there.
        arm: the arm being trained.
        compression: its resolved grid point.
        applied: its initialisation (``trainable_init`` must be set).
        config: the plan, whose ``training`` block is the pinned schedule.
        data: the loop's train/dev tensors.
        seed: the reported seed for this run.
        threads: ``torch.set_num_threads`` value.
        out_dir: where the loop writes checkpoints.
        verbose: forward the loop's structured log.
        documents: the test corpus, for the pre-training perplexity diagnostic.
        tokenizer: the pinned tokenizer.
        eval_seq_len: the evaluation window length (the same one the post-training eval uses).
        max_tokens: optional cap on scored tokens, as in the post-training eval.
        learning_rate: the rate the full schedule runs at (the search's selection, or the plan's).
        lr_search: the recorded learning-rate search, when the plan declared a grid.

    Returns:
        ``(provenance, result)`` - the ``training.*`` manifest metrics and the loop's result object.

    Raises:
        ValueError: the plan has no training schedule, or the arm carries no initialisation.
    """
    from spectraquant.training.loop import LoopConfig, train_language_model

    schedule = config.training
    if schedule is None:
        raise ValueError(
            f"plan {config.name!r} declares trainable arm {arm.name!r} but no `training` block"
        )
    init = applied.trainable_init
    if init is None:
        raise ValueError(f"arm {arm.name!r} is trainable but carries no initialisation")
    spec = compression.uniform_quant_spec()

    _install_trainable_layers(model, init, device=device)
    step_sizes_initial = _step_size_values(model)
    trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
    frozen_count = sum(1 for p in model.parameters() if not p.requires_grad)
    # Measure the arm *at its initialisation*, before a single optimiser step. Without this the
    # post-training number cannot be attributed: a trained arm that scores worse than its own
    # initialisation is a statement about the schedule, and the record has to make that visible.
    at_init = token_level_perplexity(
        model, tokenizer, documents, seq_len=eval_seq_len, max_tokens=max_tokens, device=device
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    loop = LoopConfig(
        batch_size=int(schedule.batch_size),
        lr=float(learning_rate),
        grad_clip=float(schedule.grad_clip),
        seed=int(seed),
        threads=int(threads),
        warmup_steps=int(schedule.warmup_steps),
        checkpoint_every=int(schedule.checkpoint_every) or None,
        eval_every=int(schedule.eval_every) or None,
    )
    # Only the two R2 arms whose frozen recipe pins `beta = (0.9, 0.95)` get the explicit
    # two-group optimiser; the R1 arms keep the loop's AdamW defaults (see _lr_qat_optimizer).
    optimizer = (
        _lr_qat_optimizer(model, lr_ab=float(learning_rate), lr_s=compression.step_size_lr)
        if (init.step_sizes or arm.kind == "r2_fullqat_4bit")
        else None
    )
    result = train_language_model(
        model,
        method=arm.kind,
        steps=int(schedule.steps),
        output_dir=out_dir,
        data=data,
        loop=loop,
        optimizer=optimizer,
        task_loss=_hf_task_loss,
        spec=spec,
        measurement_class=2,
        # The loop must not write its own manifest: the runner's manifest is the one the collector
        # validates, and two manifests in one directory would make the identity check ambiguous.
        write=False,
        extra_metrics={
            "training.arm": arm.name,
            "training.seed": int(seed),
            "training.steps_requested": int(schedule.steps),
            "training.initialisation": _initialisation_description(arm, init),
        },
        verbose=verbose,
    )
    provenance: dict[str, Any] = {
        "training.method": arm.kind,
        "training.steps_requested": int(schedule.steps),
        "training.steps_completed": int(result.steps_completed),
        "training.learning_rate": float(learning_rate),
        "training.learning_rate_declared": float(schedule.learning_rate),
        "training.lr_search": lr_search,
        "training.lr_search_split": ("development" if lr_search is not None else None),
        "training.lr_search_test_split_read": False if lr_search is not None else None,
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
        "training.losses": [float(value) for value in result.losses],
        "training.dev_losses": [[int(step), float(value)] for step, value in result.val_losses],
        "training.corpus_checksum": data.checksum,
        "training.perplexity_at_init": (
            None if at_init["perplexity"] is None else float(at_init["perplexity"])
        ),
        "training.perplexity_at_init_degenerate": bool(at_init["perplexity_degenerate"]),
        "training.checkpoints": [str(path) for path in result.checkpoint_paths],
        "training.initialisation": _initialisation_description(arm, init),
        "training.step_size_lr": compression.step_size_lr,
        "training.step_size_frozen": (
            None if compression.step_size_lr is None else bool(compression.step_size_lr == 0.0)
        ),
        "training.step_sizes_initial": step_sizes_initial,
        "training.step_sizes_final": _step_size_values(model),
        "training.n_trainable_parameters": trainable_count,
        "training.n_frozen_parameters": frozen_count,
    }
    if int(result.steps_completed) != int(schedule.steps):
        raise ValueError(
            f"arm {arm.name!r} completed {result.steps_completed} of {schedule.steps} requested "
            "steps: a short run must not be recorded as a complete one"
        )
    return provenance, result


def _standard_lora_pair(
    weight: Tensor, rank: int, *, generator: torch.Generator
) -> tuple[Tensor, Tensor]:
    """The standard LoRA initialisation: Kaiming-uniform ``A``, zero ``B`` (the QLoRA/fixup scheme).

    ``A`` is drawn uniformly from ``[-1/sqrt(in_features), 1/sqrt(in_features)]`` - the Kaiming-uniform
    bound of the LoRA default (``a = sqrt(5)``) - and ``B`` is exactly zero, so the arm starts from
    the base weight alone and the adapter's first gradient is ``x``-driven.

    Shapes / dtypes / device:
        ``A: (rank, in_features)``, ``B: (out_features, rank)``; ``weight``'s dtype; CPU.

    Determinism:
        The caller supplies the generator, seeded from the run's reported seed, so the initialisation
        is part of the seed's reproducibility claim (reproduction-plan.md section 3).

    Assumptions / limitations:
        This is the *standard* initialisation the LoftQ arm is compared against; it is not the
        frozen ``LowRankLinear.reset_parameters`` (whose ``B`` is non-zero and whose generator seed
        is fixed), which is a different, module-level convention.
    """
    bound = 1.0 / math.sqrt(int(weight.shape[1]))
    a = torch.empty(rank, int(weight.shape[1]), dtype=weight.dtype).uniform_(
        -bound, bound, generator=generator
    )
    b = torch.zeros(int(weight.shape[0]), rank, dtype=weight.dtype)
    return a, b


def _codebook_loftq_pair(
    weight: Tensor,
    *,
    rank: int,
    spec: CodebookSpec,
    iterations: int,
) -> tuple[Tensor, Tensor]:
    """The LoftQ alternating schedule driven by the 2-bit NF codebook (paper Algorithm 1).

    The frozen :func:`~spectraquant.factorization.loftq.loftq_initialise` takes a *uniform*
    :class:`QuantSpec` (its ``fake_quantize`` has no codebook branch), while the R1 arms quantize
    with :func:`~spectraquant.quantization.codebook.fake_quantize_codebook`. The schedule is
    therefore driven here from the same two frozen primitives and the same zero start, exactly as
    the T2-a exactness gate drives it (``tests/unit/test_m3_exactness.py``):

    .. code-block:: text

        A_0, B_0 <- 0
        for t in 1..T:
            Q_t     = q_NF2(W - B_{t-1} @ A_{t-1})
            A_t, B_t <- SVD_rank(W - Q_t)

    Returns:
        ``(A_T, B_T)``. The schedule's own base is ``Q_T = q_NF2(W - B_T @ A_T)``; the runner pairs
        these factors with the **shared** ``Q = q_NF2(W)`` instead (so the three R1 2-bit arms differ
        only in their adapter), and records the discarded companion's reconstruction error beside it.
        The alternative pairing is the same rule :func:`loftq_initialise` documents, and the same one
        the runner's regression test guards against breaking for the legacy ``loftq`` arm.

    Determinism:
        ``q_NF2`` and the dense CPU SVD are deterministic, so a fixed ``(W, rank, spec, T)`` is
        bit-reproducible.

    Raises:
        ValueError: ``iterations < 1`` (a single iteration is the first residual step, not LoftQ).
    """
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")
    a = torch.zeros(rank, int(weight.shape[1]), dtype=weight.dtype)
    b = torch.zeros(int(weight.shape[0]), rank, dtype=weight.dtype)
    for _ in range(int(iterations)):
        quantized = fake_quantize_codebook(weight - b @ a, spec)
        factors = truncated_svd(weight - quantized, rank)
        a, b = factors.A, factors.B
    return a, b


def _rtn_step_size(weight: Tensor, spec: QuantSpec) -> float:
    """The layer's RTN scale, used to initialise the LR-QAT step size ``s``.

    A per-group RTN quantization has one scale per group, while the layer's ``s`` is a single scalar
    (``design-m3-arms.md`` section 3.3), so the scalar is the **mean** group scale of the layer -
    a declared simplification, since ``merge_lr_qat`` documents the same per-group limitation.

    Raises:
        ValueError: the derived scale is not finite and positive (an empty or degenerate weight).
    """
    scale = float(quant_params(weight, spec).scales.mean())
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            f"the RTN scale of a {tuple(weight.shape)} weight is {scale!r}; the LR-QAT step size "
            "must start from a finite positive scale"
        )
    return scale


def _apply_trainable_arm(
    weights: dict[str, Tensor],
    names: Sequence[str],
    arm: ArmSpec,
    compression: ArmCompression,
    metrics: dict[str, Any],
    *,
    seed: int = 0,
) -> _Applied:
    """Initialise one trainable arm: a frozen quantized base plus the arm's starting factors.

    The arms share the deployed structure - a quantized main weight with a low-rank correction, or
    all weights trained through the quantizer - and differ in how they are *initialised*:

    * ``lr_qat`` (arXiv 2406.06385): the factors start at the truncated SVD of the residual left by
      the quantized base, then training compensates the quantization error.
    * ``loftq`` (arXiv 2310.08659): the factors start from the **alternating** schedule, which
      re-quantizes the residual and re-decomposes it, so the initialisation already absorbs the
      rounding grid.
    * ``r1_fp16_lora``: nothing is quantized; the standard adapter sits on the fp32 weight.
    * ``r1_std_2bit``: the 2-bit NF codebook base ``Q = q_NF2(W)`` with the standard adapter.
    * ``r1_loftq_2bit`` / ``r1_loftq_2bit_t1``: the LoftQ schedule (T = 5 / T = 1) over the *same*
      codebook ``Q`` as ``r1_std_2bit`` (the shared-base control of ``design-m3-arms.md`` §4.1).
    * ``r2_lrqat_4bit``: the 4-bit g128 base, rank-32 factors and a trainable scalar step size
      initialised from the base's RTN scale.
    * ``r2_fullqat_4bit``: every target weight stays trainable through the quantizer.

    Args:
        weights: ``{module name: 2-D linear weight}``, the same set for every arm.
        names: the sorted layer names.
        arm: the arm being run.
        compression: its resolved grid point.
        metrics: the manifest metrics block to extend in place.
        seed: the reported seed; it selects the standard adapter initialisation (``r1_*``), which is
            the one initialisation the frozen plan lets the seed control.

    Returns:
        An :class:`_Applied` whose ``trainable_init`` carries the bases, factors (and step sizes),
        with class-1 accounting for the *stored* object.
    """
    from spectraquant.factorization.loftq import loftq_initialise
    from spectraquant.factorization.lr_qat import lr_qat_pair

    kind = arm.kind
    rank = None if compression.rank is None else int(compression.rank)
    codebook = compression.codebook_spec() if compression.quantizer == "codebook" else None
    uniform_spec = compression.uniform_quant_spec()

    bases: dict[str, Tensor] = {}
    factors: dict[str, tuple[Tensor, Tensor]] = {}
    step_sizes: dict[str, float] = {}
    dense_weights: dict[str, Tensor] = {}
    residual_errors: list[float] = []
    base_errors: list[float] = []
    companion_errors: list[float] = []

    if kind == "r2_fullqat_4bit":
        # Every target weight stays trainable through the quantizer: the "initialisation" is the
        # dense weight itself, and the arm's stored object is its quantized form.
        assert uniform_spec is not None
        for name in names:
            weight = weights[name]
            dense_weights[name] = weight
            quantized = fake_quantize(weight, uniform_spec)
            base_errors.append(_relative_fro(weight, quantized))
            residual_errors.append(_relative_fro(weight, quantized))
        accounted = sum(accounted_bytes(tuple(weights[name].shape), uniform_spec) for name in names)
        _trainable_metrics(metrics, names, residual_errors, base_errors, factors)
        metrics["compression.quantizer"] = compression.quantizer
        metrics["compression.step_size_lr"] = None
        metrics["compression.step_size_frozen"] = None
        metrics["compression.trainable"] = "all_linear_weights"
        return _Applied(
            state={},
            storage={},
            storage_specs={},
            compression={
                "method": _METHOD_BY_KIND[kind],
                "ranks": None,
                "bits": int(compression.bits) if compression.bits is not None else None,
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
                bases={},
                factors={},
                style="straight_through",
                spec=uniform_spec,
                dense_weights=dense_weights,
            ),
        )

    generator = torch.Generator().manual_seed(int(seed))
    if rank is None:  # pragma: no cover - defensive: every remaining kind needs a rank
        raise ValueError(f"trainable arm kind {kind!r} carries no rank in its resolved point")
    for name in names:
        weight = weights[name]
        if kind == "r1_fp16_lora":
            base = weight.clone()
            pair = _standard_lora_pair(weight, int(rank), generator=generator)
        elif kind == "r1_std_2bit":
            assert codebook is not None
            base = fake_quantize_codebook(weight, codebook)
            pair = _standard_lora_pair(weight, int(rank), generator=generator)
        elif kind in {"r1_loftq_2bit", "r1_loftq_2bit_t1"}:
            assert codebook is not None and compression.loftq_iterations is not None
            # The base is the *same* Q as the std arm's, bit for bit (design-m3-arms.md §2/§4.1:
            # "the same quantized matrix Q", "the base must be identical across them"), so the three
            # 2-bit arms differ only in the adapter initialisation - which is the trend T2 tests.
            # The alternation's own base Q_T = q(W - B_{t-1} @ A_{t-1}) is internal to the schedule;
            # its companion base is recorded as a metric below rather than stored, because storing it
            # would break the shared-base control the frozen design requires.
            base = fake_quantize_codebook(weight, codebook)
            pair = _codebook_loftq_pair(
                weight,
                rank=int(rank),
                spec=codebook,
                iterations=int(compression.loftq_iterations),
            )
            companion = fake_quantize_codebook(weight - pair[1] @ pair[0], codebook)
            companion_errors.append(_relative_fro(weight, companion + pair[1] @ pair[0]))
        elif kind == "r2_lrqat_4bit":
            assert uniform_spec is not None
            base, pair = lr_qat_pair(weight, rank=int(rank), spec=uniform_spec)
            step_sizes[name] = _rtn_step_size(weight, uniform_spec)
        elif kind == "lr_qat":
            assert uniform_spec is not None
            base, pair = lr_qat_pair(weight, rank=int(rank), spec=uniform_spec)
        elif kind == "loftq":
            assert uniform_spec is not None
            # The LoftQ primitive returns the low-rank half only; its companion base is the *next*
            # quantization half-step of the same alternation, recomputed from the returned factors as
            # `fake_quantize(w - B @ A)`. Quantizing the full weight here instead would pair a base
            # with factors that do not belong to it, and the reconstruction would be far worse than
            # the naive baseline (measured: 75% relative error versus 7.3%).
            pair = loftq_initialise(
                weight, rank=int(rank), spec=uniform_spec, iterations=LOFTQ_ITERATIONS
            )
            base = fake_quantize(weight - pair[1] @ pair[0], uniform_spec)
        else:  # pragma: no cover - defensive: the kind set is closed by _METHOD_BY_KIND
            raise NotImplementedError(f"trainable arm kind {kind!r} has no initialisation")
        bases[name] = base
        factors[name] = pair
        base_errors.append(_relative_fro(weight, base))
        residual_errors.append(_relative_fro(weight, base + pair[1] @ pair[0]))

    _trainable_metrics(metrics, names, residual_errors, base_errors, factors)
    # The quantizer actually used, or `None` for the arm that quantizes nothing (`r1_fp16_lora`).
    metrics["compression.quantizer"] = (
        compression.quantizer if (codebook is not None or uniform_spec is not None) else None
    )
    if codebook is not None:
        metrics["compression.codebook_kind"] = codebook.kind
        metrics["compression.block_size"] = int(codebook.block_size)
    if step_sizes:
        metrics["compression.step_size_lr"] = compression.step_size_lr
        # `0.0` is the *frozen* grid point, not a missing value: the two are kept apart here, in
        # `compression.step_size_frozen`, so no reader can conflate them.
        metrics["compression.step_size_frozen"] = bool(compression.step_size_lr == 0.0)
        metrics["compression.step_size_initial_mean"] = float(
            sum(step_sizes.values()) / len(step_sizes)
        )
    else:
        metrics["compression.step_size_lr"] = None
        metrics["compression.step_size_frozen"] = None
    if arm.kind in {"loftq", "r1_loftq_2bit", "r1_loftq_2bit_t1"}:
        metrics["compression.loftq_iterations"] = int(
            LOFTQ_ITERATIONS if kind == "loftq" else compression.loftq_iterations or 0
        )
    if kind == "r1_std_2bit":
        # The std arm's base is the codebook quantization of the *dense* weight, which is also the
        # LoftQ schedule's own first base (A_0 = B_0 = 0). Recording that identity makes the shared-Q
        # invariant readable from the manifest rather than only from the test.
        metrics["compression.shared_base_with"] = ["r1_loftq_2bit", "r1_loftq_2bit_t1"]
        metrics["compression.shared_base_rule"] = "fake_quantize_codebook(W, spec)"
    if kind in {"r1_loftq_2bit", "r1_loftq_2bit_t1"}:
        metrics["compression.shared_base_with"] = ["r1_std_2bit"]
        metrics["compression.shared_base_rule"] = (
            "fake_quantize_codebook(W, spec) - the same Q as r1_std_2bit, bit for bit"
        )
        # The schedule's *own* base Q_T = q(W - B @ A) is not stored (it would break the shared-base
        # control); its reconstruction error is recorded so the reader can see what the companion
        # base would have given.
        metrics["compression.companion_base_relative_fro_mean"] = (
            float(sum(companion_errors) / len(companion_errors)) if companion_errors else None
        )

    # Stored object: the quantized base's codes+scales, plus the two factors at fp16, plus (for
    # LR-QAT) one fp32 step size per layer. The factors are the *trained* artifact, so they are
    # counted at storage precision, not at their float32 size. Account from the factors' *actual*
    # shapes: the SVD clamps the rank on a narrow layer, and charging the requested rank there would
    # overstate the stored bytes.
    accounted = 0
    for name in names:
        weight = weights[name]
        out_features, in_features = int(weight.shape[0]), int(weight.shape[1])
        if uniform_spec is not None:
            accounted += accounted_bytes((out_features, in_features), uniform_spec)
        elif codebook is not None:
            accounted += codebook_accounted_bytes((out_features, in_features), codebook)
        else:  # r1_fp16_lora: the fp16 base itself
            accounted += 2 * int(weight.numel())
        a, b = factors[name]
        accounted += 2 * int(a.numel() + b.numel())
        if name in step_sizes:
            accounted += 4  # one fp32 scalar per layer

    return _Applied(
        state={},
        storage={},
        storage_specs={},
        compression={
            "method": _METHOD_BY_KIND[arm.kind],
            "ranks": [int(rank)] if rank is not None else None,
            "bits": None if compression.bits is None else int(compression.bits),
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
            iterations=(
                LOFTQ_ITERATIONS
                if kind == "loftq"
                else compression.loftq_iterations
                if kind in {"r1_loftq_2bit", "r1_loftq_2bit_t1"}
                else None
            ),
            style="quantized_plus_low_rank",
            spec=uniform_spec,
            step_sizes=step_sizes,
        ),
    )


def _trainable_metrics(
    metrics: dict[str, Any],
    names: Sequence[str],
    residual_errors: list[float],
    base_errors: list[float],
    factors: dict[str, tuple[Tensor, Tensor]],
) -> dict[str, Any]:
    """The initialisation-error metrics every trainable arm records.

    The error the *initialisation* leaves is not the error the trained arm leaves; recording both
    keeps the two apart, so a later improvement can be attributed to training rather than to init.
    """
    metrics["compression.relative_fro_mean"] = (
        float(sum(residual_errors) / len(residual_errors)) if residual_errors else None
    )
    metrics["compression.relative_fro_max"] = max(residual_errors) if residual_errors else None
    metrics["compression.relative_fro_worst_layer"] = (
        names[residual_errors.index(max(residual_errors))] if residual_errors else None
    )
    metrics["compression.initial_base_relative_fro_mean"] = (
        float(sum(base_errors) / len(base_errors)) if base_errors else None
    )
    metrics["compression.initial_residual_relative_fro_mean"] = metrics[
        "compression.relative_fro_mean"
    ]
    if factors:
        metrics["compression.effective_ranks"] = sorted(
            {int(factors[name][0].shape[0]) for name in names}
        )
    else:
        metrics["compression.effective_ranks"] = None
    return metrics


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
    device: str = "cpu",
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
        device: where the loop will index these tensors. The loop batches ``data.train`` directly, so
            the tensors must live on the model's device; the checksum is computed before the move
            because a CUDA tensor has no ``numpy()`` view.

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
    # The *plan* pins the train corpus size; the invocation's cap applies to the bounded dev split.
    # Leaving the train size to the command line is what produced the first M3 cell: 200 documents
    # (99 windows) against 500 steps at batch 8, so the factors memorised the corpus.
    train_cap = int(config.training.corpus_documents) if config.training else max_documents
    for role, key in (("train", "train"), ("development", "val")):
        ref = plan_dataset_ref(config, role)
        cap = train_cap if role == "train" else max_documents
        # The train-role corpus is streamed: the frozen plans pin `allenai/c4`, whose `en` train split
        # is hundreds of gigabytes, and a non-streaming read materialises it. The dev corpus is a
        # bounded held-out split (WikiText-2 validation), so it is read normally.
        texts = load_plan_texts(
            ref,
            split=ROLE_SPLITS[role],
            dataset_config=ref.config,
            max_documents=cap,
            stream=role == "train",
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
    checksum = f"sha256:{digest.hexdigest()}"
    data = SequenceData(
        train=tensors["train"].to(device),
        val=tensors["val"].to(device),
        checksum=checksum,
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
    *,
    seed: int = 0,
) -> _Applied:
    """Compress ``weights`` for one arm, reusing the frozen quantization/factorization primitives.

    Args:
        weights: ``{module name: 2-D linear weight}`` — the same tensor set for every arm, so the
            arms' byte figures are comparable at equal memory (``AGENTS.md`` §4.5).
        arm: the arm being run.
        compression: its resolved grid point.
        seed: the reported seed; only the standard adapter initialisation depends on it.

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

    if kind in {"fp16_reference", "r2_fp16"}:
        # No compression: the reference stores the same tensors at 16 bits (class-1 analytical).
        # `r2_fp16` reports the identical figure as `fp16_reference` deliberately, so the two
        # uncompressed references of the two designs are byte-comparable.
        accounted = 2 * total_numel
        metrics["compression.relative_fro_mean"] = 0.0
        metrics["compression.reference_dtype"] = "fp32 execution; fp16 stored-size accounting"
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
        return _apply_trainable_arm(weights, names, arm, compression, metrics, seed=seed)

    if kind in {"ptq_uniform", "r2_rtn_4bit"}:
        # `r2_rtn_4bit` is the frozen R2 PTQ baseline (``R2-RTN-4bit-g128``) and is byte-for-byte
        # the same path at its own resolved point: uniform 4-bit symmetric per-group g128, RTN
        # scales, no training. Its stored object is the packed container, so bytes are measured.
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
    divergence_trigger: str | None = None,
) -> RunManifest:
    """Build the schema-validated manifest for one (arm, seed) run."""
    git = git_info()
    kind = arm.kind
    measurement_class = _arm_measurement_class(kind)
    compression = dict(applied.compression)
    compression["exclusions"] = list(exclusions)
    peak_mb = process_peak_rss_mb()
    divergent = divergence_trigger is not None
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
    # The divergence rule (reproduction-plan.md §5.4, predeclared): a divergent arm is *recorded* as
    # divergent, and its perplexity is marked as not a quality measurement, so no analysis can read
    # it as one. The number itself stays in the record (labelled), never in a comparison table.
    if training is not None or divergent:
        metrics["training.divergent"] = divergent
        metrics["training.divergence_trigger"] = divergence_trigger
    metrics["perplexity.is_quality_measurement"] = not divergent
    metrics["quality.is_measurement"] = not divergent
    if divergent:
        metrics["quality.divergence_note"] = (
            "the frozen divergence rule fired: this arm's perplexity is a failure indicator, not a "
            "quality measurement (reproduction-plan.md 5.4)"
        )
    if substitution is not None:
        # The manifest schema allows only scalar/array metrics, so the substitution is flattened.
        for key, value in substitution.items():
            metrics[f"substitution.{key}"] = value
        metrics["substitution.is_plan_measurement"] = False
    completed_steps = 0
    if training is not None:
        completed_steps = int(training.get("training.steps_completed", 0) or 0)
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
            steps=completed_steps,
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
            reported seed. The non-trainable arms are seed-invariant, but a trainable arm's standard
            adapter draw is seed-controlled, so a full M3 run passes the plan's whole seed list.
        out_dir: output root; defaults to ``artifacts/runs/<plan-name>-plan`` in the repository.
        arms: arm names to run. Defaults to every arm this invocation can run - implemented **and**
            bound to a grid point by the plan or the arguments given - while every other arm is
            recorded as skipped with its reason. Naming an unimplemented arm is a hard error.
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
        config, model_dir=model_dir, device=device
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
    # The quantization and factorization primitives are CPU-only by design (they are the exact,
    # reproducible reference implementations, and `torch.linalg.svd` has no half-precision CPU
    # kernel). The compression math therefore runs on CPU copies and the results are moved back onto
    # the model's device - either by `load_state_dict`, which copies across devices, or by `model.to`
    # after the trainable layers are installed. A GPU run must not push CUDA tensors into them.
    weights = {name: tensor.detach().to("cpu").clone() for name, tensor in weights.items()}
    say(f"compressible linear weights: {len(weights)} (excluded: {len(exclusions)})")

    # A trainable arm *replaces* layers, which no state-dict restore can undo, so it runs on a fresh
    # copy of the loaded model. The non-trainable arms keep sharing one instance.
    pristine_model = copy.deepcopy(model)
    training_data: SequenceData | None = None
    training_corpus_provenance: dict[str, Any] = {}

    runs: list[ArmRun] = []
    for arm, compression in resolutions:
        for seed in chosen_seeds:
            seed_everything(seed, deterministic=True, threads=threads)
            arm_training: dict[str, Any] = {}
            # Every arm gets its own copy of the loaded model, *and* its own initialisation: a
            # trainable arm replaces layers, no `load_state_dict` can undo that (reusing one
            # instance let `rank_then_quant` report the trained LoftQ arm's perplexity exactly), and
            # the standard adapter's draw is seed-controlled, so the initialisation is per (arm,
            # seed). The frozen quantization primitives are deterministic, so the shared-Q invariant
            # the R1 arms rely on is unaffected by the seed.
            applied = apply_arm(weights, arm, compression, seed=seed)
            model = copy.deepcopy(pristine_model)
            if applied.trainable_init is not None:
                target = root / arm.name / f"seed-{seed}"
                if training_data is None:
                    training_data, training_corpus_provenance = _training_corpus(
                        config,
                        tokenizer,
                        seq_len=int(config.training.seq_len) if config.training else seq_len,
                        max_documents=max_documents,
                        context=int(
                            getattr(getattr(model, "config", None), "max_position_embeddings", 0)
                            or 0
                        ),
                        device=device,
                    )
                    say(
                        "training corpus: "
                        f"{training_data.train.shape[0]} train / {training_data.val.shape[0]} dev "
                        f"windows of {training_data.train.shape[1] - 1} tokens "
                        f"({training_data.checksum})"
                    )
                # The learning-rate search: a recorded prefix per candidate on the *development*
                # split only, then the full schedule at the selected rate. It is skipped entirely
                # when the plan declares no grid.
                schedule = config.training
                assert schedule is not None
                selected_rate = float(schedule.learning_rate)
                lr_search: list[dict[str, Any]] | None = None
                arm_started = time.perf_counter()
                if schedule.learning_rate_grid:
                    lr_search, selected_rate = _search_learning_rate(
                        pristine_model,
                        applied.trainable_init,
                        config=config,
                        arm=arm,
                        data=training_data,
                        seed=seed,
                        threads=threads,
                        device=device,
                        out_dir=target,
                        spec=compression.uniform_quant_spec(),
                        verbose=True,
                    )
                    say(
                        f"arm {arm.name} seed {seed}: lr search selected {selected_rate:g} "
                        f"from {len(lr_search)} candidate(s) on the development split"
                    )
                arm_training, _ = _train_arm(
                    model,
                    device=device,
                    arm=arm,
                    compression=compression,
                    applied=applied,
                    config=config,
                    data=training_data,
                    seed=seed,
                    threads=threads,
                    out_dir=target,
                    verbose=True,
                    documents=documents,
                    tokenizer=tokenizer,
                    eval_seq_len=seq_len,
                    max_tokens=max_tokens,
                    learning_rate=selected_rate,
                    lr_search=lr_search,
                )
                arm_training = {**training_corpus_provenance, **arm_training}
            else:
                # The copy is pristine by construction; load the arm's compressed weights into it.
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
                # The frozen divergence rule must be able to *record* a non-finite measurement as
                # divergent instead of dying inside the protocol.
                allow_non_finite=True,
            )
            # §5.4: perplexity > 1000, NaN/non-finite, or a loss that has not decreased after 100
            # steps marks the arm divergent, and the number is then recorded as a failure indicator
            # rather than as a quality measurement.
            divergence_trigger = _divergence_trigger(
                arm_training.get("training.losses", ()) if arm_training else (),
                perplexity["perplexity"],
            )
            if divergence_trigger is not None:
                arm_training["training.divergent"] = True
                arm_training["training.divergence_trigger"] = divergence_trigger
                say(f"arm {arm.name} seed {seed}: DIVERGENT - {divergence_trigger}")
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
                divergence_trigger=divergence_trigger,
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
            measured_perplexity = perplexity["perplexity"]
            runs.append(
                ArmRun(
                    arm=arm.name,
                    kind=arm.kind,
                    seed=seed,
                    run_id=run_id,
                    manifest_path=manifest_path,
                    metrics_path=metrics_path,
                    measurement_class=measurement_class,
                    perplexity=(
                        None if measured_perplexity is None else float(measured_perplexity)
                    ),
                    accounted_bytes=int(applied.accounted),
                    divergent=bool(divergence_trigger is not None),
                )
            )
            shown = (
                "not measured"
                if measured_perplexity is None
                else f"{float(measured_perplexity):.6f}"
            )
            say(
                f"arm {arm.name} seed {seed}: perplexity={shown} "
                f"tokens={perplexity['n_tokens']} accounted_bytes={applied.accounted} "
                f"measured_bytes={applied.measured} -> {manifest_path}"
            )

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
                "divergent": run.divergent,
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
                "divergent": run.divergent,
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
