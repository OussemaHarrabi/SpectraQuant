"""The allocated arms: activation capture, proxy scoring, CP-SAT allocation, artifact build.

This module is the *scientific* half of the three arms the plan runner refuses to guess about
(``docs/decisions/ADR-0004-spectraquant-method.md`` is their contract):

* ``proxy_allocated`` (H4) — capture activations on the plan's **calibration** corpus, score every
  ``(layer, rank, bits)`` candidate the plan allows with the declared proxy, solve the layer-wise
  allocation with CP-SAT under the plan's byte budget, apply it, and evaluate. No training.
* ``proxy_allocated_regularized`` (H3, **the method**) — the same allocation, then the low-rank
  factors are trained with the four-term :class:`~spectraquant.regularizers.spectral.PreparationObjective`
  and the stored artifact is evaluated.
* ``quant_then_residual`` (M4 baseline) — quantize the weight, factorize the **residual** into a
  low-rank correction stored beside the quantized base, evaluate. No training, no allocation.

Stored artifact (one definition, used by the allocator, the report and the container)
-------------------------------------------------------------------------------------
Per targeted linear layer, with ``spec = cost_model.spec_for_bits(bits)``:

.. code-block:: text

    Q      = fake_quantize(W, spec)            # the dense quantized base
    A, B   = truncated_svd(W - Q, rank)        # the low-rank correction of the residual
    stored = (Q, Q(A), Q(B))                   # the artifact: all three quantized at `bits`
    W_eff  = Q + Q(B) @ Q(A)                   # what the forward pass computes

and the class-1 byte model is exactly the allocator's constrained quantity

.. code-block:: text

    bytes(layer) = accounted_bytes((out, in), spec)          # base
                 + accounted_bytes((rank, in), spec)         # A
                 + accounted_bytes((out, rank), spec)        # B

so the constrained quantity and the reported quantity are the same number
(``spectraquant/allocation/problem.py`` module docstring, invariant B16). Every term comes from
:mod:`spectraquant.quantization.accounting` — the one byte source of truth — and the container
written by :func:`~spectraquant.quantization.accounting.measure_serialized_bytes` has no header and
no inter-region padding, so the class-3 measured figure equals the class-1 accounted figure
(recorded beside it, never instead of it).

**Documented deviation from ADR-0004 §7.** The ADR's byte formula counts the factors at fp16
(``2 * (numel(A) + numel(B))``) while its §6 fixes the allocator's ``cost_fn`` to
:func:`~spectraquant.allocation.problem.make_accounting_cost_fn`, which counts *quantized* factors.
The two cannot both hold. This module implements the allocator's model — the factors are stored
quantized at the allocated bit width — because (i) the constrained quantity must equal the reported
quantity, (ii) ``PreparationObjective``'s ``rounding_grid_penalty`` is defined on the factors'
quantization grid, so the factors being quantized is the method's own premise, and (iii) the frozen
proxy unit (``design-m2-interfaces.md`` §0.3) is itself ``E_X ||X W^T - X (Q(B) Q(A))^T||^2``, i.e.
quantized factors. The ADR-§7 fp16-factor counterpart of every reported figure is recorded beside it
(``allocation.accounted_bytes_adr7_fp16_factors``). The question is flagged for the ADR owner.

Measurement classes (``AGENTS.md`` §5): byte figures are class 1 (analytical) with the class-3
serialized figure beside them; every quality number is class 2 (float execution that simulates the
quantization numerics). Nothing here is class 4 or 5.

CPU only (``AGENTS.md`` §2): the frozen quantizer and factorizer have no CUDA kernels, so captured
activations are moved to CPU before scoring and the allocation math runs on CPU.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from spectraquant.allocation import (
    COST_MODEL_VERSION,
    Allocation,
    AllocationProblem,
    QuantCostModel,
    make_accounting_cost_fn,
    solve_ortools,
)
from spectraquant.allocation.problem import CostFn, ErrorFn
from spectraquant.factorization import truncated_svd
from spectraquant.proxies.base import LayerInputs, Proxy
from spectraquant.proxies.variants import PROXY_VARIANTS, get_proxy
from spectraquant.quantization import (
    SQ_CONTAINER_FORMAT_ID,
    QuantSpec,
    accounted_bytes,
    fake_quantize,
    measure_serialized_bytes,
    theoretical_bits,
)
from spectraquant.regularizers.spectral import PreparationObjective
from spectraquant.training.low_rank import QuantizedPlusLowRankLinear

__all__ = [
    "ALLOCATION_ARM_KINDS",
    "CAPTURE_BATCH_WINDOWS",
    "CAPTURE_MAX_DOCUMENTS",
    "CAPTURE_MAX_SAMPLES_PER_LAYER",
    "CAPTURE_MAX_WINDOWS",
    "DEFAULT_ALLOC_AXIS",
    "DEFAULT_ALLOC_GROUP_SIZE",
    "DEFAULT_PROXY_VARIANT",
    "DEFAULT_SOLVER_TIME_LIMIT_S",
    "REGULARIZER_COEFFICIENT_KEYS",
    "ActivationCapture",
    "InfeasibleAllocationBudget",
    "PenaltyEvaluation",
    "ProxyGrid",
    "RegularizerCoefficients",
    "RegularizerRun",
    "ResidualArtifact",
    "artifact_storage",
    "build_residual_artifact",
    "capture_activations",
    "effective_cost_fn",
    "effective_rank_of",
    "frozen_parameter_bytes",
    "minimum_achievable_bytes",
    "module_name_of",
    "option_key",
    "preparation_penalty",
    "proxy_score_grid",
    "proxy_score_key",
    "quantize_factors",
    "quantize_installed_factors",
    "residual_factor_pairs",
    "resolve_proxy_variant",
    "resolve_regularizer_coefficients",
    "solve_allocation",
    "stored_artifact_bytes",
]

#: The two arms whose ``(rank, bits)`` come from an allocation rather than from a single grid point.
ALLOCATION_ARM_KINDS: frozenset[str] = frozenset({"proxy_allocated", "proxy_allocated_regularized"})

#: The declared candidate proxy (``ADR-0004`` §6). ``combined`` and every other registered variant are
#: selectable as an explicit override and the choice is recorded in the manifest.
DEFAULT_PROXY_VARIANT = "gain_aware_composed"

#: Activation-capture caps. The capture must be bounded (``AGENTS.md`` §2: 16 GB workstation): the
#: number of *documents* read, the number of *forward batches* (windows) run and the number of
#: activation rows *retained per layer* are all capped, and each cap is recorded in the manifest.
#: Retention is ``max_samples_per_layer * sum(in_features) * 4`` bytes — for SmolLM2-135M's 210
#: targeted layers that is ≈190 MB at the default, and the capture stops as soon as every layer is
#: full. The *test* split is never read: only the plan's ``calibration`` role is loaded.
CAPTURE_MAX_DOCUMENTS = 16
CAPTURE_MAX_WINDOWS = 16
CAPTURE_MAX_SAMPLES_PER_LAYER = 256
CAPTURE_BATCH_WINDOWS = 4

#: The allocation's quantization cost model: the group size the allocator uses when the caller does
#: not name one (validated against ``plan.grid.group_sizes``) and the axis that reproduces the factor
#: blocking of ``docs/protocols/memory-accounting.md`` §4 (``QuantCostModel``'s own default). The
#: granularity is the caller's declared ``--granularity``, so it is not pinned here.
DEFAULT_ALLOC_GROUP_SIZE = 128
DEFAULT_ALLOC_AXIS = 1

#: CP-SAT wall-clock limit (seconds) for one allocation solve. Recorded in the manifest; the solver's
#: own diagnostics carry the observed status code and objective.
DEFAULT_SOLVER_TIME_LIMIT_S = 10.0

#: The four :class:`PreparationObjective` coefficients plus the spectral tail rank. Every coefficient
#: defaults to ``0.0`` (the objective's own default: an omitted weight must be visible as a zero
#: coefficient, never silently applied).
REGULARIZER_COEFFICIENT_KEYS: tuple[str, ...] = (
    "lambda_factor",
    "lambda_round",
    "lambda_residual",
    "lambda_spectrum",
)


class InfeasibleAllocationBudget(ValueError):
    """Raised when no allocation inside the plan's grid fits the byte budget.

    The message names the budget *and* the minimum achievable bytes, because a run whose allocation
    exceeds the budget must fail loudly rather than silently shrink the allocation
    (``ADR-0004`` §9, ``AGENTS.md`` §4.13).

    Attributes:
        budget_bytes: the budget that was rejected.
        minimum_bytes: the cheapest achievable total over the plan's allowed ``(rank, bits)`` options.
        arm: the arm being allocated.
    """

    def __init__(self, *, arm: str, budget_bytes: int, minimum_bytes: int) -> None:
        self.arm = str(arm)
        self.budget_bytes = int(budget_bytes)
        self.minimum_bytes = int(minimum_bytes)
        super().__init__(
            f"the proxy allocation for arm {self.arm!r} is infeasible: the byte budget is "
            f"{self.budget_bytes} B but the minimum achievable allocation over the plan's allowed "
            f"(rank, bits) options is {self.minimum_bytes} B. Raise the budget (or widen the grid) "
            "deliberately; the allocation is never silently shrunk to fit."
        )


# --------------------------------------------------------------------------------------
# Names and keys
# --------------------------------------------------------------------------------------
def module_name_of(weight_name: str) -> str:
    """Return the module path of a ``state_dict``-style weight name (``a.b.weight`` -> ``a.b``)."""
    suffix = ".weight"
    return weight_name[: -len(suffix)] if weight_name.endswith(suffix) else weight_name


def option_key(rank: int, bits: int) -> str:
    """Canonical ``"r<rank>b<bits>"`` key of one ``(rank, bits)`` candidate."""
    return f"r{int(rank)}b{int(bits)}"


def proxy_score_key(layer: str, rank: int, bits: int) -> str:
    """Manifest metric key recording one layer/candidate proxy score.

    The value stored under this key is exactly the ``ProxyResult.value`` the frozen
    ``Proxy.score_model`` entry point returned for that layer, rank and bit width (see
    :func:`proxy_score_grid`), never a re-derivation.
    """
    return f"allocation.proxy_score.{layer}.{option_key(rank, bits)}"


# --------------------------------------------------------------------------------------
# Activation capture (calibration role only)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ActivationCapture:
    """Per-layer activation context captured on the plan's calibration corpus.

    Attributes:
        layers: ``{weight name: LayerInputs}`` — the pristine weight and the activations the
            uncompressed forward pass fed to it.
        provenance: flat, JSON-scalar manifest metrics: the dataset identity, the split read, the
            window/sample counts, the caps, the batch checksum and the retention in bytes.
        batch_checksum: ``"sha256:<hex>"`` over the token ids of the windows actually used.
    """

    layers: Mapping[str, LayerInputs]
    provenance: Mapping[str, Any] = field(default_factory=dict)
    batch_checksum: str = ""

    @property
    def n_samples(self) -> int:
        """Number of retained activation rows per layer (0 when nothing was captured)."""
        return len(next(iter(self.layers.values())).activations) if self.layers else 0

    def metrics(self) -> dict[str, Any]:
        """The provenance block, flat and JSON-scalar (the manifest's ``allocation.calibration.*``)."""
        return dict(self.provenance)


def _capture_windows(
    documents: Sequence[str],
    tokenizer: Any,
    *,
    seq_len: int,
    max_windows: int,
    context: int | None,
) -> list[Tensor]:
    """Cut ``documents`` into non-overlapping ``seq_len``-token windows (trailing short one dropped).

    The rule is the perplexity protocol's own (``plan_runner.token_level_perplexity``): document
    boundaries are preserved, the window is clamped to the model's context length, and a trailing
    shorter window is dropped so every captured activation comes from a full-width window. At most
    ``max_windows`` windows are returned — the capture's compute cap.
    """
    width = max(2, min(int(seq_len), int(context or seq_len)))
    windows: list[Tensor] = []
    for document in documents:
        if not document.strip():
            continue
        ids = tokenizer(document, return_tensors="pt").input_ids.reshape(-1)
        length = int(ids.numel())
        for start in range(0, length - width + 1, width):
            windows.append(ids[start : start + width].to(torch.int64))
            if len(windows) >= max_windows:
                return windows
    return windows


def capture_activations(
    model: nn.Module,
    tokenizer: Any,
    documents: Sequence[str],
    *,
    weight_names: Sequence[str],
    seq_len: int,
    device: str = "cpu",
    max_documents: int = CAPTURE_MAX_DOCUMENTS,
    max_windows: int = CAPTURE_MAX_WINDOWS,
    max_samples_per_layer: int = CAPTURE_MAX_SAMPLES_PER_LAYER,
    batch_windows: int = CAPTURE_BATCH_WINDOWS,
    dataset: Mapping[str, Any] | None = None,
    role: str = "calibration",
) -> ActivationCapture:
    """Capture the calibration activations of the targeted linear layers, with forward hooks.

    For every window of the calibration corpus the model is run under ``torch.no_grad()`` in eval
    mode and a **forward pre-hook** on each targeted :class:`torch.nn.Linear` records the tensor it
    consumed, flattened to ``(tokens, in_features)``. The rows are accumulated per layer up to
    ``max_samples_per_layer`` and the capture stops as soon as every layer is full or the window
    budget is exhausted, so both compute and retained memory are bounded by declared caps.

    Args:
        model: the (pristine, uncompressed) model; it is restored to its previous mode afterwards.
        tokenizer: the pinned tokenizer.
        documents: the plan's ``calibration``-role documents — **never** the test split.
        weight_names: ``state_dict``-style names of the targeted linear weights.
        seq_len: window length in tokens (clamped to the model's context).
        device: the model's device; captured tensors are moved to CPU because the frozen quantizer
            and factorizer are CPU-only.
        max_documents: cap on the documents read.
        max_windows: cap on the forward passes (one per window batch).
        max_samples_per_layer: cap on the retained rows per layer.
        batch_windows: windows per forward pass (compute efficiency only).
        dataset: the dataset provenance (``name``, ``config``, ``revision``, ``split``) recorded in
            the manifest; supplied by the caller so the identity comes from the plan's own pin.
        role: the protocol role the documents were loaded under (``"calibration"``); recorded.

    Returns:
        An :class:`ActivationCapture`; its :meth:`ActivationCapture.metrics` carries the caps, the
        counts, the batch checksum and ``allocation.calibration.test_split_read: false``.

    Raises:
        ValueError: a cap is not positive, a named weight has no linear module, a hook sees a
            non-tensor input, or no window could be built (a silent empty capture would make every
            proxy score undefined).
    """
    for label, value in (
        ("max_documents", max_documents),
        ("max_windows", max_windows),
        ("max_samples_per_layer", max_samples_per_layer),
        ("batch_windows", batch_windows),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label} must be a positive int, got {value!r}")

    names = list(weight_names)
    if not names:
        raise ValueError("weight_names must not be empty: there is nothing to capture")
    modules = dict(model.named_modules())
    targets: dict[str, str] = {}
    for weight_name in names:
        module_name = module_name_of(weight_name)
        module = modules.get(module_name)
        if not isinstance(module, nn.Linear):
            raise ValueError(
                f"layer {weight_name!r} is not a linear module of this model "
                f"(got {type(module).__name__}); the capture can only hook `nn.Linear` layers"
            )
        if module_name in targets.values():  # pragma: no cover - two weights, one module
            raise ValueError(f"two weights map to module {module_name!r}")
        targets[weight_name] = module_name

    windows = _capture_windows(
        list(documents)[: int(max_documents)],
        tokenizer,
        seq_len=seq_len,
        max_windows=int(max_windows),
        context=int(getattr(getattr(model, "config", None), "max_position_embeddings", 0) or 0),
    )
    if not windows:
        raise ValueError(
            f"no calibration window of {seq_len} tokens could be built from "
            f"{min(len(documents), max_documents)} document(s): an empty capture would make every "
            "proxy score undefined"
        )

    buffers: dict[str, list[Tensor]] = {name: [] for name in names}
    retained: dict[str, int] = dict.fromkeys(names, 0)
    hooks: list[Any] = []

    def make_hook(weight_name: str) -> Any:
        def hook(module: nn.Module, args: tuple[Any, ...]) -> None:
            if not args or not isinstance(args[0], Tensor):
                raise ValueError(
                    f"the forward pre-hook of {weight_name!r} expected a tensor input, got "
                    f"{type(args[0]).__name__ if args else 'no arguments'}"
                )
            if retained[weight_name] >= int(max_samples_per_layer):
                return None
            rows = args[0].detach().reshape(-1, int(args[0].shape[-1])).to("cpu", torch.float32)
            room = int(max_samples_per_layer) - retained[weight_name]
            if int(rows.shape[0]) > room:
                rows = rows[:room]
            buffers[weight_name].append(rows)
            retained[weight_name] += int(rows.shape[0])
            return None

        return hook

    was_training = model.training
    model.eval()
    used_windows: list[Tensor] = []
    n_windows_forwarded = 0
    try:
        for weight_name, module_name in targets.items():
            hooks.append(modules[module_name].register_forward_pre_hook(make_hook(weight_name)))
        with torch.no_grad():
            step = max(1, int(batch_windows))
            for start in range(0, len(windows), step):
                batch = torch.stack(windows[start : start + step]).to(device)
                model(batch)
                used_windows.append(batch.detach().to("cpu"))
                n_windows_forwarded += int(batch.shape[0])
                if all(retained[name] >= int(max_samples_per_layer) for name in names):
                    break
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)

    layers: dict[str, LayerInputs] = {}
    for weight_name in names:
        rows = buffers[weight_name]
        if not rows:
            raise ValueError(
                f"no activation reached layer {weight_name!r}: the hook never fired, so the layer's "
                "proxy score would be undefined"
            )
        module = modules[targets[weight_name]]
        assert isinstance(module, nn.Linear)
        layers[weight_name] = LayerInputs(
            weight=module.weight.detach().to("cpu", torch.float32).clone(),
            activations=torch.cat(rows, dim=0),
        )

    document_hashes = hashlib.sha256()
    for document in list(documents)[: int(max_documents)]:
        document_hashes.update(document.encode("utf-8"))
    batch_hashes = hashlib.sha256()
    for window in used_windows:
        batch_hashes.update(window.numpy().tobytes())
    activation_bytes = sum(
        int(inputs.activations.numel()) * inputs.activations.element_size()
        for inputs in layers.values()
    )
    provenance: dict[str, Any] = {
        "allocation.calibration.role": str(role),
        "allocation.calibration.dataset": str((dataset or {}).get("name", "unknown")),
        "allocation.calibration.dataset_config": (dataset or {}).get("config"),
        "allocation.calibration.dataset_revision": (dataset or {}).get("revision"),
        "allocation.calibration.split": str((dataset or {}).get("split", "unknown")),
        "allocation.calibration.documents": int(min(len(documents), int(max_documents))),
        "allocation.calibration.windows": int(n_windows_forwarded),
        "allocation.calibration.forward_batches": len(used_windows),
        "allocation.calibration.window_tokens": int(windows[0].numel()),
        "allocation.calibration.max_documents": int(max_documents),
        "allocation.calibration.max_windows": int(max_windows),
        "allocation.calibration.max_samples_per_layer": int(max_samples_per_layer),
        "allocation.calibration.samples_per_layer": int(min(retained.values())),
        "allocation.calibration.batch_checksum": f"sha256:{batch_hashes.hexdigest()}",
        "allocation.calibration.documents_sha256": f"sha256:{document_hashes.hexdigest()}",
        "allocation.calibration.activation_bytes": int(activation_bytes),
        "allocation.calibration.retention_rule": (
            "samples_per_layer <= allocation.calibration.max_samples_per_layer; the capture stops "
            "when every targeted layer is full, so retained activations are bounded"
        ),
        "allocation.calibration.forward_calls": len(used_windows),
        "allocation.calibration.pool": (
            "one forward pass per window batch on the uncompressed model in eval mode, "
            "torch.no_grad()"
        ),
        "allocation.calibration.test_split_read": False,
        "allocation.calibration.notice": (
            "activations were captured on the plan's 'calibration' role only; the test split is "
            "read once, after training, by the perplexity protocol"
        ),
    }
    return ActivationCapture(
        layers=layers,
        provenance=provenance,
        batch_checksum=str(provenance["allocation.calibration.batch_checksum"]),
    )


# --------------------------------------------------------------------------------------
# Proxy scoring over the candidate grid
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ProxyGrid:
    """Every candidate's proxy score, produced once and used by both the record and the solver.

    The same object is the allocator's ``error_fn`` (:meth:`ProxyGrid.error_fn`) and the manifest's
    score table (:meth:`ProxyGrid.metrics`), so the recorded number *is* the number the solver
    optimised — the frozen ``score_model`` entry point is called exactly once per ``(layer, rank,
    bits)`` candidate.

    Attributes:
        per_layer: ``{layer: {"r<rank>b<bits>": value}}`` in the common proxy unit.
        exact: whether every scored layer's result was exact for its definition.
        measurement_class: the result's class (1 analytical or 2 fake-quantization quality).
        fallback_layers: layers whose variant degraded to its context-free form (the frozen variants
            document this, e.g. ``gain_aware_composed`` without a downstream gain).
        proxy_name: the variant name that was scored (recorded in the manifest).
        n_samples: activation rows behind each score (the same for every layer).
    """

    per_layer: Mapping[str, Mapping[str, float]]
    exact: bool
    measurement_class: int
    fallback_layers: tuple[str, ...]
    proxy_name: str
    n_samples: int = 0

    @property
    def n_scores(self) -> int:
        """Number of scored ``(layer, rank, bits)`` candidates."""
        return sum(len(scores) for scores in self.per_layer.values())

    def value(self, layer: str, rank: int, bits: int) -> float:
        """The recorded proxy score of one candidate.

        Raises:
            KeyError: the layer or candidate was never scored (the grid is fixed up front, so this
                is a programming error rather than a data condition).
        """
        try:
            return float(self.per_layer[layer][option_key(rank, bits)])
        except KeyError as exc:  # pragma: no cover - the grid is built from the same options
            raise KeyError(
                f"no proxy score for layer {layer!r} at rank={rank} bits={bits}; the grid holds "
                f"{sorted(self.per_layer)} with options {sorted(next(iter(self.per_layer.values())))}"
            ) from exc

    def error_fn(self, layer: str, rank: int, bits: int) -> float:
        """The allocator's ``error_fn``: lower is better (the proxy's own value)."""
        return self.value(layer, rank, bits)

    def metrics(self) -> dict[str, float]:
        """The flat manifest block: one key per ``(layer, rank, bits)`` candidate, plus fallbacks."""
        out: dict[str, float] = {}
        for layer, scores in self.per_layer.items():
            for key, value in scores.items():
                out[f"allocation.proxy_score.{layer}.{key}"] = float(value)
        for layer in self.fallback_layers:
            out[f"allocation.proxy_fallback.{layer}"] = 1.0
        return out


def proxy_score_grid(
    proxy: Proxy,
    layer_inputs: Mapping[str, LayerInputs],
    *,
    ranks: Sequence[int],
    bits: Sequence[int],
    cost_model: QuantCostModel,
) -> ProxyGrid:
    """Score every ``(layer, rank, bits)`` candidate through the frozen ``score_model`` entry point.

    One call per candidate, with a single-layer mapping (exactly what
    :func:`spectraquant.allocation.proxy_adapter.proxy_error_fn` does), so the recorded value is the
    proxy's own output for those inputs and no look-alike re-derivation can drift from it.

    Args:
        proxy: the declared candidate (default ``gain_aware_composed``) or an explicitly overridden
            variant from :data:`~spectraquant.proxies.variants.PROXY_VARIANTS`.
        layer_inputs: ``{layer: LayerInputs}`` from :func:`capture_activations`.
        ranks: the plan's allowed ranks.
        bits: the plan's allowed payload bit widths.
        cost_model: supplies the :class:`QuantSpec` of each bit width, so the proxy is scored under
            the exact spec the byte cost is accounted with.

    Returns:
        A :class:`ProxyGrid` holding every score, the variant's identity and which layers fell back
        to its context-free form. The variants' own ``diagnostics`` are not re-stored here: they are
        per-candidate quantities, and the *selected* candidate's are recoverable from the manifest's
        score table and the recorded calibration block (rows, caps, checksum).

    Raises:
        ValueError: an empty grid, or a layer whose ``LayerInputs`` is missing.
        KeyError: a bit width the cost model does not support.
    """
    if not layer_inputs:
        raise ValueError("layer_inputs must not be empty: there is nothing to score")
    per_layer: dict[str, dict[str, float]] = {}
    fallback: list[str] = []
    exact = True
    classes: set[int] = set()
    for layer in sorted(layer_inputs):
        scores: dict[str, float] = {}
        for rank in ranks:
            for width in bits:
                spec = cost_model.spec_for_bits(int(width))
                result = proxy.score_model(
                    {layer: layer_inputs[layer]}, {layer: int(rank)}, {layer: spec}
                )
                scores[option_key(int(rank), int(width))] = float(result.value)
                exact = exact and bool(result.exact)
                classes.add(int(result.measurement_class))
                if float(result.diagnostics.get(f"{layer}/fallback", 0.0)) > 0.0:
                    fallback.append(layer)
        per_layer[layer] = scores
    return ProxyGrid(
        per_layer=per_layer,
        exact=exact,
        measurement_class=max(classes) if classes else 2,
        fallback_layers=tuple(sorted(set(fallback))),
        proxy_name=str(getattr(proxy, "name", "unknown")),
        n_samples=int(next(iter(layer_inputs.values())).activations.shape[0]),
    )


def resolve_proxy_variant(name: str | None = None) -> Proxy:
    """Instantiate the proxy variant to score with, refusing an unknown name.

    The default is the declared candidate :data:`DEFAULT_PROXY_VARIANT`; an explicit override must be
    a key of :data:`PROXY_VARIANTS` (``combined`` accepts no weights here — its documented defaults
    are used, and the choice is recorded).

    Raises:
        ValueError: the name is not a registered variant (naming the known ones).
    """
    variant = DEFAULT_PROXY_VARIANT if name is None else str(name)
    if variant not in PROXY_VARIANTS:
        raise ValueError(
            f"unknown proxy {variant!r}; known variants: {sorted(PROXY_VARIANTS)} "
            f"(the declared candidate is {DEFAULT_PROXY_VARIANT!r})"
        )
    return get_proxy(variant)  # type: ignore[return-value]


# --------------------------------------------------------------------------------------
# The byte model the allocator constrains
# --------------------------------------------------------------------------------------
def effective_rank_of(shape: Sequence[int], rank: int) -> int:
    """The rank the frozen factorizer actually stores for a ``(out, in)`` shape.

    ``truncated_svd`` clamps a requested rank to ``min(shape)`` and reports the effective rank
    (``ADR-0004`` §7: "the rank is the **effective** rank (the SVD clamps it on a narrow layer)"), so
    the byte model must use the clamped value or the class-1 figure would describe a shape the stored
    factors do not have.
    """
    return int(min(int(rank), min(int(dim) for dim in shape)))


def effective_cost_fn(
    layer_shapes: Mapping[str, Sequence[int]], cost_model: QuantCostModel
) -> CostFn:
    """The accounting ``cost_fn`` with each requested rank mapped to its effective rank.

    The returned callable is :func:`~spectraquant.allocation.problem.make_accounting_cost_fn` — the
    one supported accounting cost function, so no byte cost is re-derived here — composed with
    :func:`effective_rank_of`.
    """
    base = make_accounting_cost_fn(layer_shapes, cost_model)
    shapes = {name: tuple(int(dim) for dim in shape) for name, shape in layer_shapes.items()}

    def cost_fn(layer: str, rank: int, bits: int) -> int:
        return base(layer, effective_rank_of(shapes[layer], rank), bits)

    return cost_fn


def base_bytes_fn(layer_shapes: Mapping[str, Sequence[int]], cost_model: QuantCostModel) -> CostFn:
    """The allocator's ``overhead_fn``: the **quantized dense base**'s class-1 bytes.

    The artifact stores the dense quantized base beside the quantized factors (see the module
    docstring), so its bytes belong in the constrained quantity. They come from
    :func:`~spectraquant.quantization.accounting.accounted_bytes` at the same ``(layer, bits)`` — the
    base's cost does not depend on the rank.
    """
    shapes = {name: tuple(int(dim) for dim in shape) for name, shape in layer_shapes.items()}

    def overhead_fn(layer: str, rank: int, bits: int) -> int:
        del rank  # the dense base's payload is rank-independent
        return accounted_bytes(shapes[layer], cost_model.spec_for_bits(int(bits)))

    return overhead_fn


def _finite_or_none(value: Any) -> float | None:
    """Return ``value`` as a finite float, or ``None``.

    Solver diagnostics are optional: a solver that records none must leave a JSON ``null`` rather
    than a ``NaN`` literal, because a non-finite number in a manifest is neither valid JSON nor a
    measurement.
    """
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def minimum_achievable_bytes(
    layer_shapes: Mapping[str, Sequence[int]],
    *,
    ranks: Sequence[int],
    bits: Sequence[int],
    cost_model: QuantCostModel,
) -> int:
    """Cheapest total over all layers (each at its minimum-cost option) — the feasibility threshold.

    This is the number an :class:`InfeasibleAllocationBudget` reports: every budget below it admits
    no allocation at all.
    """
    cost_fn = effective_cost_fn(layer_shapes, cost_model)
    overhead_fn = base_bytes_fn(layer_shapes, cost_model)
    return sum(
        min(
            int(cost_fn(name, int(rank), int(width)))
            + int(overhead_fn(name, int(rank), int(width)))
            for rank in ranks
            for width in bits
        )
        for name in layer_shapes
    )


# --------------------------------------------------------------------------------------
# CP-SAT allocation
# --------------------------------------------------------------------------------------
def solve_allocation(
    *,
    arm: str,
    layer_shapes: Mapping[str, Sequence[int]],
    ranks: Sequence[int],
    bits: Sequence[int],
    budget_bytes: int,
    cost_model: QuantCostModel,
    error_fn: ErrorFn,
    seed: int = 0,
    time_limit_s: float = DEFAULT_SOLVER_TIME_LIMIT_S,
) -> Allocation:
    """Allocate one ``(rank, bits)`` pair per layer with CP-SAT under the byte budget.

    The problem is exactly the supported wiring: the accounting cost function (with effective ranks)
    as ``cost_fn``, the quantized dense base as ``overhead_fn``, and the frozen proxy as ``error_fn``
    (``ADR-0004`` §6). The returned :class:`Allocation`'s ``accounted_bytes`` is the same
    ``cost_fn + overhead_fn`` total the budget was checked against, so an allocation that returns
    cannot exceed the budget it reports.

    Args:
        arm: the arm being allocated (named in the failure message).
        layer_shapes: ``{layer: (out, in)}`` for the targeted weights.
        ranks: allowed ranks (the plan's grid).
        bits: allowed bit widths (the plan's grid).
        budget_bytes: the byte budget for the **compressed** weights.
        cost_model: the quantization configuration the class-1 cost is accounted under.
        error_fn: the proxy's error model (see :class:`ProxyGrid`).
        seed: CP-SAT random seed — the run's reported seed, so the same seed reproduces the same
            allocation and a different seed may legitimately differ.
        time_limit_s: CP-SAT wall-clock limit.

    Returns:
        The optimal feasible :class:`Allocation` (its ``diagnostics`` carry the CP-SAT status code,
        wall time and scaled objective).

    Raises:
        InfeasibleAllocationBudget: the budget is below the minimum achievable bytes; the message
            names both numbers.
        ORToolsNotInstalledError: the optional ``alloc`` extra is missing.
        RuntimeError: CP-SAT exhausted ``time_limit_s`` without a solution.
    """
    shapes: dict[str, tuple[int, int]] = {
        str(name): (int(dims[0]), int(dims[1])) for name, dims in layer_shapes.items()
    }
    problem = AllocationProblem(
        layer_shapes=shapes,
        ranks=ranks,
        bits=bits,
        budget_bytes=int(budget_bytes),
        cost_fn=effective_cost_fn(shapes, cost_model),
        error_fn=error_fn,
        overhead_fn=base_bytes_fn(shapes, cost_model),
        cost_model=cost_model,
    )
    minimum = problem.min_feasible_budget()
    if minimum > int(budget_bytes):
        raise InfeasibleAllocationBudget(
            arm=arm, budget_bytes=int(budget_bytes), minimum_bytes=int(minimum)
        )
    allocation = solve_ortools(problem, time_limit_s=float(time_limit_s), seed=int(seed))
    if int(allocation.accounted_bytes) > int(budget_bytes):  # pragma: no cover - solver guarantee
        raise InfeasibleAllocationBudget(
            arm=arm, budget_bytes=int(budget_bytes), minimum_bytes=int(minimum)
        )
    return allocation


# --------------------------------------------------------------------------------------
# The stored artifact
# --------------------------------------------------------------------------------------
def quantize_factors(
    factors: Mapping[str, tuple[Tensor, Tensor]],
    per_layer: Mapping[str, tuple[int, int]],
    cost_model: QuantCostModel,
) -> tuple[dict[str, tuple[Tensor, Tensor]], dict[str, QuantSpec]]:
    """Quantize both factors of every layer to its allocated bit width (the stored representation).

    Returns:
        ``(stored, specs)`` where ``stored[layer] = (Q(A), Q(B))`` and ``specs[layer]`` is the
        :class:`QuantSpec` of the layer's allocated bit width.
    """
    stored: dict[str, tuple[Tensor, Tensor]] = {}
    specs: dict[str, QuantSpec] = {}
    for layer, pair in factors.items():
        spec = cost_model.spec_for_bits(int(per_layer[layer][1]))
        stored[layer] = (fake_quantize(pair[0], spec), fake_quantize(pair[1], spec))
        specs[layer] = spec
    return stored, specs


def artifact_storage(
    bases: Mapping[str, Tensor],
    stored_factors: Mapping[str, tuple[Tensor, Tensor]],
    per_layer: Mapping[str, tuple[int, int]],
    cost_model: QuantCostModel,
) -> tuple[dict[str, Tensor], dict[str, QuantSpec]]:
    """The serializable state of the artifact: the base plus the two quantized factors per layer.

    The base is passed already dequantized (it is on the quantization grid, so re-quantizing it in the
    container is idempotent) and the factors are the *stored* ones, so the container's measured bytes
    describe exactly the artifact the forward pass computes.
    """
    storage: dict[str, Tensor] = {}
    specs: dict[str, QuantSpec] = {}
    for layer in sorted(bases):
        spec = cost_model.spec_for_bits(int(per_layer[layer][1]))
        storage[layer] = bases[layer]
        specs[layer] = spec
        stored_a, stored_b = stored_factors[layer]
        storage[f"{layer}.A"] = stored_a
        storage[f"{layer}.B"] = stored_b
        specs[f"{layer}.A"] = spec
        specs[f"{layer}.B"] = spec
    return storage, specs


def _storage_specs_for(
    per_layer: Mapping[str, tuple[int, int]], cost_model: QuantCostModel
) -> dict[str, QuantSpec]:
    return {
        layer: cost_model.spec_for_bits(int(per_layer[layer][1])) for layer in sorted(per_layer)
    }


def stored_artifact_bytes(
    storage: Mapping[str, Tensor], specs: Mapping[str, QuantSpec]
) -> tuple[int, int, float | None]:
    """``(class-1 accounted bytes, class-3 measured bytes, payload bits per stored parameter)``.

    The accounted figure is the sum over the container's tensors of
    :func:`~spectraquant.quantization.accounting.accounted_bytes`; the measured figure is
    :func:`~spectraquant.quantization.accounting.measure_serialized_bytes`, which actually writes the
    headerless SpectraQuant container into a temporary directory and measures it. Both are reported:
    the class-1 number is what the allocator constrained, the class-3 number is what we serialized.
    """
    accounted = sum(accounted_bytes(tuple(t.shape), specs[name]) for name, t in storage.items())
    measured = int(measure_serialized_bytes(storage, specs))
    payload_bits = sum(
        theoretical_bits(tuple(t.shape), specs[name]) * int(t.numel())
        for name, t in storage.items()
    )
    packed_numel = sum(int(t.numel()) for t in storage.values())
    theoretical = payload_bits / packed_numel if packed_numel else None
    return int(accounted), measured, theoretical


@dataclass(frozen=True)
class ResidualArtifact:
    """The built artifact of one residual (quantize-then-factorize) application.

    Attributes:
        per_layer: ``{layer: (requested rank, bits)}`` as applied.
        effective_ranks: ``{layer: effective rank}`` actually stored (the SVD clamps narrow layers).
        bases: ``{layer: dequantized dense base}`` — the frozen part of the artifact.
        train_factors: ``{layer: (A, B)}`` at their float training precision (the trainable
            representation; equals the stored factors for the eval-only arms).
        stored_factors: ``{layer: (Q(A), Q(B))}`` — what the container holds.
        storage: the serializable container state.
        storage_specs: one :class:`QuantSpec` per stored tensor.
        accounted: class-1 bytes of the artifact (the allocator's constrained quantity).
        measured: class-3 bytes of the container written from ``storage``.
        theoretical_bits: payload bits per stored parameter (class 1, no metadata/padding).
        state: ``{layer: W_eff}`` — the effective dense weight, loadable into the uncompressed
            model's linear layers (the eval-only arms' path).
        metrics: class-1/2 diagnostics for the manifest.
    """

    per_layer: Mapping[str, tuple[int, int]]
    effective_ranks: Mapping[str, int]
    bases: dict[str, Tensor]
    train_factors: dict[str, tuple[Tensor, Tensor]]
    stored_factors: dict[str, tuple[Tensor, Tensor]]
    storage: dict[str, Tensor]
    storage_specs: dict[str, QuantSpec]
    accounted: int
    measured: int
    theoretical_bits: float | None
    state: dict[str, Tensor]
    metrics: dict[str, Any] = field(default_factory=dict)


def build_residual_artifact(
    weights: Mapping[str, Tensor],
    per_layer: Mapping[str, tuple[int, int]],
    cost_model: QuantCostModel,
) -> ResidualArtifact:
    """Quantize each weight, factorize the residual, and measure the stored artifact.

    Per layer: ``Q = fake_quantize(W, spec)`` (the dense quantized base), ``(A, B) =
    truncated_svd(W - Q, rank)`` (the low-rank correction of the residual), and the stored object is
    ``(Q, Q(A), Q(B))`` at the layer's bit width. The forward pass computes ``Q + Q(B) @ Q(A)``.

    Args:
        weights: ``{layer: (out, in) float weight}`` — the pristine weights.
        per_layer: ``{layer: (rank, bits)}`` — the applied allocation or the arm's single grid point.
        cost_model: the quantization configuration (its ``spec_for_bits`` supplies each spec).

    Returns:
        The :class:`ResidualArtifact`; its ``metrics`` carry the per-layer allocation, the
        class-1/class-3 bytes and the relative reconstruction error.

    Raises:
        KeyError: a layer named in ``per_layer`` is absent from ``weights``.
        ValueError: an empty input.
    """
    if not per_layer:
        raise ValueError("per_layer must not be empty: there is nothing to build")
    missing = sorted(set(per_layer) - set(weights))
    if missing:
        raise KeyError(f"weights is missing the allocated layer(s) {missing}")

    bases: dict[str, Tensor] = {}
    factors: dict[str, tuple[Tensor, Tensor]] = {}
    effective: dict[str, int] = {}
    errors: dict[str, float] = {}
    for layer in sorted(per_layer):
        weight = weights[layer]
        rank, bits = int(per_layer[layer][0]), int(per_layer[layer][1])
        spec = cost_model.spec_for_bits(bits)
        base = fake_quantize(weight, spec)
        pair = truncated_svd(weight - base, rank)
        bases[layer] = base
        factors[layer] = (pair.A, pair.B)
        effective[layer] = int(pair.rank)
        denominator = float(weight.detach().to(torch.float64).pow(2).sum())
        residual = float((weight - base).detach().to(torch.float64).pow(2).sum())
        errors[layer] = math.sqrt(residual / denominator) if denominator > 0.0 else 0.0

    stored_factors, _ = quantize_factors(factors, per_layer, cost_model)
    storage, storage_specs = artifact_storage(bases, stored_factors, per_layer, cost_model)
    accounted, measured, theoretical = stored_artifact_bytes(storage, storage_specs)
    state: dict[str, Tensor] = {}
    relative: list[float] = []
    for layer in sorted(per_layer):
        stored_a, stored_b = stored_factors[layer]
        effective_weight = bases[layer] + stored_b @ stored_a
        state[layer] = effective_weight
        reference = weights[layer].detach().to(torch.float64)
        norm = float(reference.pow(2).sum())
        delta = (weights[layer] - effective_weight).detach().to(torch.float64)
        relative.append(math.sqrt(float(delta.pow(2).sum()) / norm) if norm > 0.0 else 0.0)

    metrics: dict[str, Any] = {
        "compression.n_tensors": len(per_layer),
        "compression.n_parameters": sum(int(weights[layer].numel()) for layer in per_layer),
        "compression.effective_ranks": sorted(set(effective.values())),
        "compression.requested_ranks": sorted({int(per_layer[layer][0]) for layer in per_layer}),
        "compression.relative_fro_mean": sum(relative) / len(relative),
        "compression.relative_fro_max": max(relative),
        "compression.relative_fro_worst_layer": sorted(per_layer)[relative.index(max(relative))],
        "compression.initial_residual_relative_fro_mean": sum(errors.values()) / len(errors),
        "compression.bytes_source": "measured",
        "compression.accounted_bytes_class": 1,
        "compression.measured_bytes_class": 3,
        "compression.serializer": SQ_CONTAINER_FORMAT_ID,
        "compression.stored_n_tensors": len(storage),
        "compression.stored_n_parameters": sum(int(t.numel()) for t in storage.values()),
        "allocation.bytes_formula": (
            "sum over layers of accounted_bytes((out,in),spec) + accounted_bytes((rank,in),spec) + "
            "accounted_bytes((out,rank),spec) with spec = cost_model.spec_for_bits(bits)"
        ),
    }
    return ResidualArtifact(
        per_layer={
            layer: (int(per_layer[layer][0]), int(per_layer[layer][1])) for layer in per_layer
        },
        effective_ranks=effective,
        bases=bases,
        train_factors=factors,
        stored_factors=stored_factors,
        storage=storage,
        storage_specs=storage_specs,
        accounted=accounted,
        measured=measured,
        theoretical_bits=theoretical,
        state=state,
        metrics=metrics,
    )


def quantize_installed_factors(
    model: nn.Module,
    per_layer: Mapping[str, tuple[int, int]],
    cost_model: QuantCostModel,
) -> dict[str, Any]:
    """Quantize the *trained* factors of the installed layers in place, and report what moved.

    The method's trained representation is float; the stored artifact quantizes the factors at the
    layer's allocated bit width (see the module docstring). This is what makes the evaluated model the
    artifact whose bytes are reported, instead of a float surrogate of it.

    Returns:
        ``{"training.factors_quantized_for_eval": True, "training.factor_quantization_bits": {...}}``
        — the per-layer bit width actually applied — for the manifest.

    Raises:
        ValueError: no target layer carries factors to quantize (a silent no-op would let a trained
            arm report float numbers under a quantized-artifact byte claim).
    """
    specs = {
        module_name_of(name): spec
        for name, spec in _storage_specs_for(per_layer, cost_model).items()
    }
    moved = 0
    bits: dict[str, Any] = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, QuantizedPlusLowRankLinear) or module_name not in specs:
            continue
        spec = specs[module_name]
        with torch.no_grad():
            module.A.data.copy_(fake_quantize(module.A.data, spec))
            module.B.data.copy_(fake_quantize(module.B.data, spec))
        bits[f"training.factor_quantization_bits.{module_name}"] = int(spec.bits)
        moved += 1
    if moved == 0:
        raise ValueError(
            "no QuantizedPlusLowRankLinear layer matched the allocation: the trained factors could "
            "not be quantized into the stored artifact"
        )
    return {"training.factors_quantized_for_eval": True, **bits}


# --------------------------------------------------------------------------------------
# The regularizer
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RegularizerCoefficients:
    """The four :class:`PreparationObjective` coefficients actually used (never inferred).

    Every coefficient defaults to ``0.0`` — the objective's own default — so an omitted weight is a
    recorded zero rather than a silently applied constant. ``tail_rank`` is required when
    ``lambda_spectrum > 0`` and forbidden otherwise (the objective's own rule).

    Raises:
        TypeError / ValueError: a coefficient is not a finite non-negative number, or the
            ``tail_rank``/``lambda_spectrum`` combination cannot be applied.
    """

    lambda_factor: float = 0.0
    lambda_round: float = 0.0
    lambda_residual: float = 0.0
    lambda_spectrum: float = 0.0
    tail_rank: int | None = None

    def __post_init__(self) -> None:
        for key in REGULARIZER_COEFFICIENT_KEYS:
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{key} must be a number, got {type(value).__name__}")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{key} must be finite and >= 0, got {value!r}")
        if self.tail_rank is not None and (
            isinstance(self.tail_rank, bool)
            or not isinstance(self.tail_rank, int)
            or int(self.tail_rank) < 1
        ):
            raise ValueError(f"tail_rank must be an int >= 1 or None, got {self.tail_rank!r}")
        if self.lambda_spectrum > 0.0 and self.tail_rank is None:
            raise ValueError("tail_rank is required when lambda_spectrum > 0")
        if self.lambda_spectrum == 0.0 and self.tail_rank is not None:
            raise ValueError("tail_rank must be None when lambda_spectrum == 0")

    def weights(self) -> dict[str, float]:
        """The four coefficients, keyed as :class:`PreparationObjective` records them."""
        return {key: float(getattr(self, key)) for key in REGULARIZER_COEFFICIENT_KEYS}

    def to_dict(self) -> dict[str, Any]:
        """The coefficient block for the manifest (floats plus the spectral tail rank)."""
        return {**self.weights(), "tail_rank": self.tail_rank}

    def is_noop(self) -> bool:
        """True when every coefficient is zero: the penalty contributes exactly nothing.

        A no-op objective is *skipped* rather than evaluated (its four terms would each be multiplied
        by zero), and the skip is recorded, so an unconfigured arm cannot look like a configured one.
        """
        return all(value == 0.0 for value in self.weights().values())


def resolve_regularizer_coefficients(
    plan: Any, override: Mapping[str, Any] | None = None
) -> tuple[RegularizerCoefficients, str]:
    """Resolve the coefficients from the plan (when it declares any) and an explicit override.

    The frozen plan schema has no ``regularizer`` block yet, so in practice the values come from the
    caller's explicit flags and the default is the objective's all-zero setting (a real declaration: it
    turns the penalty off, and the manifest says so). When the plan schema gains the block this
    function picks it up without a code change, and a caller-supplied value always wins.

    Args:
        plan: the loaded plan (duck-typed: a ``regularizer`` mapping or model is read when present).
        override: explicit per-coefficient values (``lambda_*``/``tail_rank``); unknown keys are
            refused rather than ignored.

    Returns:
        ``(coefficients, source)`` where ``source`` is ``"plan"``, ``"cli"``, ``"plan+cli"`` or
        ``"defaults"``; it is recorded in the manifest.

    Raises:
        ValueError: an unknown override key, or a value the objective would reject.
    """
    declared = getattr(plan, "regularizer", None)
    if declared is not None and hasattr(declared, "model_dump"):
        declared = declared.model_dump()
    payload: dict[str, Any] = {}
    sources: list[str] = []
    allowed = {*REGULARIZER_COEFFICIENT_KEYS, "tail_rank"}
    if isinstance(declared, Mapping):
        payload.update({str(key): value for key, value in declared.items() if str(key) in allowed})
        sources.append("plan")
    if override:
        unknown = sorted({str(key) for key in override} - allowed)
        if unknown:
            raise ValueError(
                f"unknown regularizer coefficient(s) {unknown}; allowed: {sorted(allowed)}"
            )
        payload.update({str(key): value for key, value in override.items()})
        sources.append("cli")
    return RegularizerCoefficients(**payload), ("+".join(sources) or "defaults")


@dataclass(frozen=True)
class PenaltyEvaluation:
    """One step's weighted penalty and its per-term diagnostics.

    Attributes:
        total: the 0-dim differentiable sum the loop adds to the task loss.
        terms: the four components' pooled values (the objective's own terms, unweighted).
        weighted: each component times its coefficient (what the total sums).
        groups: ``bits -> {term: value}`` for the layers sharing that bit width.
        is_noop: whether every coefficient was zero (no term was evaluated).
    """

    total: Tensor
    terms: dict[str, float]
    weighted: dict[str, float]
    groups: dict[str, dict[str, float]]
    is_noop: bool


@dataclass(frozen=True)
class RegularizerRun:
    """Everything the regularized arm needs to apply the objective and record it.

    Attributes:
        coefficients: the coefficients actually used.
        coefficients_source: where they came from (plan / cli / defaults).
        specs: ``{module name: QuantSpec}`` — the layer's own target quantizer (the allocation may
            pick a different bit width per layer, so the objective is applied per bit group).
        references: ``{module name: pristine weight}`` for the objective's factorization term; without
            it that term would be identically zero and a declared ``lambda_factor`` would be applied
            to nothing.
        name: label recorded in diagnostics.
    """

    coefficients: RegularizerCoefficients
    coefficients_source: str
    specs: Mapping[str, QuantSpec]
    references: Mapping[str, Tensor]
    name: str = "spectraquant_regularized"

    def metrics(self) -> dict[str, Any]:
        """The manifest block: every coefficient, the source, and whether the penalty was a no-op."""
        return {
            "regularizer.coefficients_source": self.coefficients_source,
            "regularizer.is_noop": self.coefficients.is_noop(),
            "regularizer.name": self.name,
            "regularizer.references_supplied": bool(self.references),
            "regularizer.pooling": "pooled within each bit-width group and summed across groups",
            **{
                f"regularizer.{key}": value
                for key, value in self.coefficients.to_dict().items()
                if value is not None
            },
        }

    def penalty(self, factors: Mapping[str, tuple[Tensor, Tensor]]) -> PenaltyEvaluation:
        """Evaluate the four-term objective on ``factors`` (``{module name: (A, B)}``)."""
        return preparation_penalty(
            factors,
            self.specs,
            self.coefficients,
            references=self.references,
            name=self.name,
        )


def preparation_penalty(
    factors: Mapping[str, tuple[Tensor, Tensor]],
    specs: Mapping[str, QuantSpec],
    coefficients: RegularizerCoefficients,
    *,
    references: Mapping[str, Tensor] | None = None,
    name: str = "spectraquant_regularized",
) -> PenaltyEvaluation:
    """Evaluate :class:`PreparationObjective` over the model's factors, one objective per bit group.

    The allocation gives each layer its own bit width, and the objective's ``rounding``/``residual``
    terms are defined on the *target quantizer*, so the layers are grouped by bit width and each group
    is evaluated by its own :class:`PreparationObjective` (the primitive pools within its group, its
    documented convention: "a ratio of total energies, not an average of per-layer ratios"). The four
    term values are summed across groups and the totals are added; the grouping is recorded.

    When every coefficient is zero the objective is a no-op and no term is evaluated: adding an exact
    zero would only pay for four terms per layer per step, and the skip is recorded
    (``is_noop: True``) rather than hidden.

    Args:
        factors: ``{module name: (A, B)}`` — the trainable pairs.
        specs: ``{module name: QuantSpec}`` — one per layer (must share the bit-width-independent
            fields, since they all come from one cost model).
        coefficients: the four coefficients plus the spectral tail rank.
        references: ``{module name: pristine weight}`` for the factorization term.
        name: the objective's label (the config arm).

    Returns:
        A :class:`PenaltyEvaluation`.

    Raises:
        ValueError: empty factors, a missing spec, or specs that disagree beyond their bit width.
    """
    if not factors:
        raise ValueError("preparation_penalty needs at least one factor pair")
    missing = sorted(set(factors) - set(specs))
    if missing:
        raise ValueError(f"no QuantSpec for layer(s) {missing}")
    dtype = next(iter(factors.values()))[0].dtype
    device = next(iter(factors.values()))[0].device
    if coefficients.is_noop():
        return PenaltyEvaluation(
            total=torch.zeros((), dtype=dtype, device=device),
            terms={"factorization": 0.0, "rounding": 0.0, "residual": 0.0, "spectrum": 0.0},
            weighted={"factorization": 0.0, "rounding": 0.0, "residual": 0.0, "spectrum": 0.0},
            groups={},
            is_noop=True,
        )

    groups: dict[int, list[str]] = {}
    for layer in sorted(factors):
        groups.setdefault(int(specs[layer].bits), []).append(layer)
    reference_fields: tuple[Any, ...] | None = None
    totals: dict[str, float] = {
        "factorization": 0.0,
        "rounding": 0.0,
        "residual": 0.0,
        "spectrum": 0.0,
    }
    per_group: dict[str, dict[str, float]] = {}
    total: Tensor | None = None
    for width in sorted(groups):
        members = groups[width]
        spec = specs[members[0]]
        fields = (
            spec.granularity,
            spec.group_size,
            spec.symmetric,
            spec.axis,
            spec.round_mode,
        )
        if reference_fields is None:
            reference_fields = fields
        elif fields != reference_fields:
            raise ValueError(
                "the layers' specs disagree beyond their bit width, so one objective per bit group "
                f"cannot represent them: {fields} vs {reference_fields}"
            )
        objective = PreparationObjective(
            spec=spec,
            lambda_factor=coefficients.lambda_factor,
            lambda_round=coefficients.lambda_round,
            lambda_residual=coefficients.lambda_residual,
            lambda_spectrum=coefficients.lambda_spectrum,
            tail_rank=coefficients.tail_rank,
            name=name,
        )
        group_factors = {layer: factors[layer] for layer in members}
        group_references = (
            None
            if not references
            else {layer: references[layer] for layer in members if layer in references}
        )
        evaluated = objective.terms(group_factors, references=group_references)
        total = evaluated.total if total is None else total + evaluated.total
        record = evaluated.to_dict()
        per_group[str(width)] = {
            key: float(record[key])
            for key in ("factorization", "rounding", "residual", "spectrum", "total")
        }
        for key in totals:
            totals[key] += float(record[key])
    assert total is not None  # guaranteed by the non-empty factor mapping
    weights = coefficients.weights()
    return PenaltyEvaluation(
        total=total,
        terms=totals,
        weighted={
            "factorization": weights["lambda_factor"] * totals["factorization"],
            "rounding": weights["lambda_round"] * totals["rounding"],
            "residual": weights["lambda_residual"] * totals["residual"],
            "spectrum": weights["lambda_spectrum"] * totals["spectrum"],
        },
        groups=per_group,
        is_noop=False,
    )


def residual_factor_pairs(model: nn.Module) -> dict[str, tuple[Tensor, Tensor]]:
    """``{module name: (A, B)}`` for every installed :class:`QuantizedPlusLowRankLinear`.

    This is the objective's input for the method's own layer class. (The training loop's own
    ``gather_factor_pairs`` collects :class:`~spectraquant.training.low_rank.LowRankLinear` only, so
    the regularized arm evaluates the objective through its own task-loss wrapper — see
    ``plan_runner._regularized_task_loss``.)

    Raises:
        ValueError: the model carries no such layer (a silent empty mapping would make every term a
            no-op).
    """
    pairs: dict[str, tuple[Tensor, Tensor]] = {}
    for name, module in model.named_modules():
        if isinstance(module, QuantizedPlusLowRankLinear) and name:
            pairs[name] = (module.A, module.B)
    if not pairs:
        raise ValueError(
            "no QuantizedPlusLowRankLinear layer found: the regularized arm must install its "
            "allocated layers before the objective can be evaluated"
        )
    return pairs


# --------------------------------------------------------------------------------------
# Whole-model bookkeeping
# --------------------------------------------------------------------------------------
def frozen_parameter_bytes(model: nn.Module, targeted_weight_names: Sequence[str]) -> int:
    """Class-1 fp16 bytes of every parameter the arm does **not** compress (unique storages).

    Recorded beside the allocation so a reader can build the whole-model stored-byte figure the plan's
    budget ladder is expressed in (its rungs are fractions of the fp16 checkpoint). Tied parameters
    (a tied ``lm_head`` sharing the embedding's storage) are counted once, by ``data_ptr``.
    """
    targeted = set(targeted_weight_names)
    seen: set[int] = set()
    total = 0
    for name, parameter in model.named_parameters():
        pointer = int(parameter.data_ptr())
        if pointer in seen:
            continue
        seen.add(pointer)
        if name in targeted:
            continue
        total += 2 * int(parameter.numel())
    return total


def allocation_metrics(
    *,
    grid: ResidualArtifact,
    allocation: Allocation,
    budget_bytes: int,
    budget_source: str,
    budget_rung_bytes: int | None,
    frozen_bytes: int,
    proxy_grid: ProxyGrid,
    cost_model: QuantCostModel,
    solver_time_limit_s: float,
    seed: int,
) -> dict[str, Any]:
    """The manifest block for one allocated arm: the allocation, the solver, the budget and the bytes.

    Every number is labelled: the allocation and the bytes are class 1
    (``allocation.bytes_class``), the container figure is class 3 (``allocation.measured_bytes_class``),
    and the proxy scores are the class-2 quality estimates (``allocation.proxy_value_class``).
    """
    per_layer_rank = {name: int(pair[0]) for name, pair in allocation.per_layer.items()}
    per_layer_bits = {name: int(pair[1]) for name, pair in allocation.per_layer.items()}
    cost_fn = effective_cost_fn(
        {name: tuple(int(dim) for dim in grid.bases[name].shape) for name in per_layer_rank},
        cost_model,
    )
    overhead_fn = base_bytes_fn(
        {name: tuple(int(dim) for dim in grid.bases[name].shape) for name in per_layer_rank},
        cost_model,
    )
    metrics: dict[str, Any] = {
        "allocation.budget_bytes": int(budget_bytes),
        "allocation.budget_source": budget_source,
        "allocation.budget_ladder_rung_bytes": budget_rung_bytes,
        "allocation.frozen_parameter_bytes": int(frozen_bytes),
        "allocation.model_total_bytes_estimate": int(allocation.accounted_bytes)
        + int(frozen_bytes),
        "allocation.frozen_bytes_exceed_rung": (
            None if budget_rung_bytes is None else bool(frozen_bytes > int(budget_rung_bytes))
        ),
        "allocation.frozen_bytes_note": (
            "the ladder rungs are fractions of the fp16 checkpoint (the whole model), so the "
            "untargeted remainder is recorded here; the allocator's budget constrains the targeted "
            "linear weights only"
        ),
        "allocation.solver": allocation.solver,
        "allocation.solver_time_limit_s": float(solver_time_limit_s),
        "allocation.solver_seed": int(seed),
        "allocation.feasible": bool(allocation.feasible),
        "allocation.predicted_error": float(allocation.predicted_error),
        "allocation.objective_scaled": _finite_or_none(
            allocation.diagnostics.get("ortools_objective_scaled")
        ),
        "allocation.solver_status_code": _finite_or_none(
            allocation.diagnostics.get("ortools_status_code")
        ),
        "allocation.solver_wall_time_s": _finite_or_none(
            allocation.diagnostics.get("ortools_wall_time_s")
        ),
        "allocation.achieved_bytes": int(allocation.accounted_bytes),
        "allocation.bytes_class": 1,
        "allocation.measured_bytes_class": 3,
        "allocation.measured_bytes": int(grid.measured),
        "allocation.serializer": SQ_CONTAINER_FORMAT_ID,
        "allocation.bytes_model": (
            "sum over layers of cost_fn (the quantized factors) + overhead_fn (the quantized dense "
            "base), both from spectraquant.quantization.accounting; class 1"
        ),
        "allocation.cost_model_version": str(COST_MODEL_VERSION),
        "allocation.cost_model_granularity": cost_model.granularity,
        "allocation.cost_model_group_size": cost_model.group_size,
        "allocation.cost_model_axis": int(cost_model.axis),
        "allocation.cost_model_symmetric": bool(cost_model.symmetric),
        "allocation.n_tensors": len(per_layer_rank),
        "allocation.distinct_ranks": sorted(set(per_layer_rank.values())),
        "allocation.distinct_bits": sorted(set(per_layer_bits.values())),
        "allocation.n_candidates": int(proxy_grid.n_scores),
        "allocation.proxy_calibration_rows": int(proxy_grid.n_samples),
        "allocation.proxy_variant": proxy_grid.proxy_name,
        "allocation.proxy_exact": bool(proxy_grid.exact),
        "allocation.proxy_value_class": int(proxy_grid.measurement_class),
        "allocation.proxy_fallback_layers": len(proxy_grid.fallback_layers),
        "allocation.proxy_context": "activations_only",
        "allocation.proxy_context_note": (
            "the proxy was scored through the frozen score_model entry point with the calibration "
            "activations; the variants' optional context (a following-normalisation Jacobian, a "
            "measured residual error, an estimated downstream gain) is not supplied by this slice, so "
            "gain_aware_composed records its documented context-free fallback (fallback=1) per layer"
        ),
        "allocation.budget_headroom_bytes": int(budget_bytes) - int(allocation.accounted_bytes),
        "allocation.accounted_bytes_adr7_fp16_factors": int(
            sum(
                int(overhead_fn(layer, rank, bits))
                + 2
                * int(
                    grid.effective_ranks[layer] * int(grid.bases[layer].shape[1])
                    + int(grid.bases[layer].shape[0]) * grid.effective_ranks[layer]
                )
                for layer, (rank, bits) in allocation.per_layer.items()
            )
        ),
        "allocation.adr7_note": (
            "ADR-0004 section 7 counts the factors at fp16 (2 bytes per element) while its section 6 "
            "fixes the allocator's cost_fn to make_accounting_cost_fn (quantized factors). The "
            "constrained and reported figure is the allocator's (quantized factors); the ADR-7 "
            "fp16-factor counterpart of the same allocation is recorded above"
        ),
        "allocation.determinism_note": (
            "the candidate table is fixed up front and CP-SAT runs single-worker with the run's "
            "reported seed, so the same (calibration capture, seed) reproduces the same allocation"
        ),
    }
    for layer in sorted(per_layer_rank):
        rank, bits = per_layer_rank[layer], per_layer_bits[layer]
        metrics[f"allocation.layer.{layer}.requested_rank"] = int(rank)
        metrics[f"allocation.layer.{layer}.bits"] = int(bits)
        metrics[f"allocation.layer.{layer}.effective_rank"] = int(grid.effective_ranks[layer])
        metrics[f"allocation.layer.{layer}.bytes"] = int(cost_fn(layer, rank, bits)) + int(
            overhead_fn(layer, rank, bits)
        )
    return metrics
