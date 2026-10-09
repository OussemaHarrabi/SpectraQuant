"""Factorized linear modules: the trainable carrier of low-rank *preparation* (Milestone 5).

A :class:`LowRankLinear` stores a linear map as two factors instead of a dense weight, following the
frozen convention ``W ~= B @ A`` with ``A: (rank, in_features)`` and ``B: (out_features, rank)``
(``docs/coordination/design-m2-interfaces.md`` section 0.2). :func:`factorize_linears` converts an
existing model in place, initialising each factor pair from the truncated SVD of the dense weight it
replaces, and returns the dense weights as a *reference* snapshot for the preparation objective.

Everything is CPU-only float32 and measurement class 2 (``AGENTS.md`` section 5): the factors are
float tensors; the low-bit representation only appears when the caller fake-quantizes them.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn

from spectraquant.factorization import truncated_svd
from spectraquant.quantization import QuantSpec, accounted_bytes, fake_quantize

__all__ = [
    "LowRankLinear",
    "check_against_dense",
    "clone_with_weights",
    "deployed_weights",
    "factor_storage_bytes",
    "factorize_linears",
    "gather_factor_pairs",
    "low_rank_layer_names",
    "parameter_counts",
    "sequence_shape_report",
]

#: A quantizer usable for the deployment-side measurement (class 2).
_DEPLOY_DTYPE = torch.float64


class LowRankLinear(nn.Module):
    """``y = x @ (B @ A)^T + bias`` with ``A: (rank, in)`` and ``B: (out, rank)`` trainable.

    Shapes:
        Input ``(*, in_features)``; output ``(*, out_features)``. ``A`` and ``B`` are
        :class:`torch.nn.Parameter` names ``A`` and ``B``; the bias (if present) keeps the name and
        shape of the linear layer it replaces.

    Dtypes / device:
        CPU, float32 by default (any float dtype may be requested). ``torch.linalg.svd`` has no CPU
        kernel for half precision, so the SVD-based initialiser rejects anything but float32/float64.

    Gradients:
        Both factors are leaves of the graph, so a task loss and every regularizer term in
        :mod:`spectraquant.regularizers.spectral` backpropagate into ``A`` and ``B``.

    Assumptions / limitations:
        The effective weight ``B @ A`` is materialised on every forward pass; this is a Tier-0
        fixture component, not a memory-optimised kernel path. No CUDA support.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        *,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        for label, value in (
            ("in_features", in_features),
            ("out_features", out_features),
            ("rank", rank),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be an int >= 1, got {value!r}")
        if rank > min(in_features, out_features):
            raise ValueError(
                f"rank {rank} exceeds min(in_features={in_features}, out_features={out_features})"
            )
        if device != "cpu" and torch.device(device).type != "cpu":
            raise ValueError(f"only CPU execution is supported, got device {device!r}")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.A = nn.Parameter(torch.empty(rank, in_features, dtype=dtype))
        self.B = nn.Parameter(torch.empty(out_features, rank, dtype=dtype))
        if bias:
            self.bias: nn.Parameter | None = nn.Parameter(torch.zeros(out_features, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Deterministic small-magnitude initialisation (overwritten by the SVD initialiser)."""
        bound = 1.0 / math.sqrt(self.in_features)
        generator = torch.Generator().manual_seed(0)
        with torch.no_grad():
            self.A.uniform_(-bound, bound, generator=generator)
            self.B.uniform_(-bound, bound, generator=generator)
            if self.bias is not None:
                self.bias.zero_()

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int) -> LowRankLinear:
        """Build a factorized layer initialised from the truncated SVD of ``linear.weight``.

        Args:
            linear: the dense layer to replace (CPU, float32/float64).
            rank: retained rank; must be ``<= min(out_features, in_features)``.

        Returns:
            A :class:`LowRankLinear` whose ``B @ A`` is the rank-``rank`` truncated SVD of the dense
            weight (Eckart-Young optimal), with the dense bias copied verbatim.
        """
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"linear must be an nn.Linear, got {type(linear).__name__}")
        weight = linear.weight.detach()
        factors = truncated_svd(weight, rank)
        module = cls(
            linear.in_features,
            linear.out_features,
            factors.rank,
            bias=linear.bias is not None,
            dtype=weight.dtype,
        )
        with torch.no_grad():
            module.A.copy_(factors.A)
            module.B.copy_(factors.B)
            if module.bias is not None and linear.bias is not None:
                module.bias.copy_(linear.bias.detach())
        return module

    # ------------------------------------------------------------------ shapes
    def effective_weight(self) -> Tensor:
        """The dense weight this layer is equivalent to: ``B @ A`` (differentiable)."""
        return self.B @ self.A

    def factor_pair(self) -> tuple[Tensor, Tensor]:
        """``(A, B)`` — the pair consumed by :mod:`spectraquant.regularizers.spectral`."""
        return self.A, self.B

    def forward(self, x: Tensor) -> Tensor:
        """``x @ (B @ A)^T + bias``; numerically equal to the dense linear with weight ``B @ A``."""
        return torch.nn.functional.linear(x, self.effective_weight(), self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, "
            f"bias={self.bias is not None}"
        )


def low_rank_layer_names(model: nn.Module) -> list[str]:
    """Names of every :class:`LowRankLinear` in ``model``, in ``named_modules`` order."""
    return [
        name for name, module in model.named_modules() if isinstance(module, LowRankLinear) and name
    ]


def gather_factor_pairs(model: nn.Module) -> dict[str, tuple[Tensor, Tensor]]:
    """``name -> (A, B)`` for every :class:`LowRankLinear`; the objective's input.

    Raises:
        ValueError: if the model contains no :class:`LowRankLinear` (a silent empty mapping would
            make every regularizer term a no-op).
    """
    pairs = {
        name: module.factor_pair()
        for name, module in model.named_modules()
        if isinstance(module, LowRankLinear) and name
    }
    if not pairs:
        raise ValueError(
            "model has no LowRankLinear layers; factorize it before applying a penalty"
        )
    return pairs


def factorize_linears(
    model: nn.Module,
    *,
    rank: int,
    ranks: Mapping[str, int] | None = None,
    exclude: Iterable[str] = (),
) -> dict[str, Tensor]:
    """Replace every ``nn.Linear`` of ``model`` with an SVD-initialised :class:`LowRankLinear`.

    Args:
        model: the module to modify **in place** (the loop owns it).
        rank: default retained rank for every matched layer.
        ranks: optional per-layer overrides, keyed by the layer's dotted name.
        exclude: name prefixes to leave dense (e.g. a classification head kept at full precision).

    Returns:
        ``name -> dense weight snapshot (detached clone)`` for every replaced layer, i.e. the
        preparation *reference* passed to the objective's factorization term.

    Raises:
        ValueError / TypeError: if a requested rank exceeds a layer's ``min(out, in)``, or if a
            ``ranks`` key matches no layer (a typo would otherwise silently keep a layer dense).
    """
    layers = {
        name: module for name, module in model.named_modules() if isinstance(module, nn.Linear)
    }
    if not layers:
        raise ValueError("model has no nn.Linear layers to factorize")
    overrides = dict(ranks or {})
    unknown = set(overrides) - set(layers)
    if unknown:
        raise ValueError(f"ranks refers to unknown layers: {sorted(unknown)}")
    prefixes = tuple(exclude)
    references: dict[str, Tensor] = {}
    for name, linear in layers.items():
        if any(name.startswith(prefix) for prefix in prefixes):
            continue
        layer_rank = overrides.get(name, rank)
        if layer_rank > min(linear.in_features, linear.out_features):
            raise ValueError(
                f"rank {layer_rank} exceeds min(out, in)={min(linear.in_features, linear.out_features)} "
                f"for layer {name!r}"
            )
        references[name] = linear.weight.detach().clone()
        parent, _, attribute = name.rpartition(".")
        container = model.get_submodule(parent) if parent else model
        setattr(container, attribute, LowRankLinear.from_linear(linear, layer_rank))
    if not references:
        raise ValueError("no layer was factorized; check the `exclude` prefixes")
    return references


def factor_storage_bytes(model: nn.Module, spec: QuantSpec) -> int:
    """Class-1 analytical stored bytes of the factorized weights under ``spec``.

    Uses :func:`spectraquant.quantization.accounted_bytes` — the single byte-accounting source of
    truth (``docs/coordination/design-m2-interfaces.md`` invariant 1) — once per factor, so scales,
    zero-points and group metadata are included exactly as the accounting module defines them.

    Shapes:
        Sums over every :class:`LowRankLinear`; returns an int (bytes). This is a **class-1**
        analytical estimate, never a measured (class-3) storage figure.
    """
    total = 0
    for module in model.modules():
        if isinstance(module, LowRankLinear):
            total += accounted_bytes(tuple(module.A.shape), spec)
            total += accounted_bytes(tuple(module.B.shape), spec)
    if total == 0:
        raise ValueError("model has no LowRankLinear layers; nothing to account for")
    return total


def deployed_weights(
    model: nn.Module,
    spec: QuantSpec,
    *,
    quantize: bool,
) -> dict[str, Tensor]:
    """Effective weight per factorized layer, optionally with both factors fake-quantized.

    Args:
        model: a model containing :class:`LowRankLinear` layers.
        spec: the target quantizer for both factors.
        quantize: ``False`` returns ``B @ A`` (the prepared map); ``True`` returns
            ``Q(B) @ Q(A)`` — the frozen rank-then-quantize deployment of
            ``docs/coordination/design-m2-interfaces.md`` section 7 and
            :func:`spectraquant.evaluation.toy`'s ``_compressed_weight``.

    Returns:
        ``name -> float64 tensor (out, in)``, detached. The float64 dtype is deliberate: these are
        measurement inputs, and the deployed evaluation must not introduce a second rounding source.
    """
    weights: dict[str, Tensor] = {}
    for name, module in model.named_modules():
        if not isinstance(module, LowRankLinear):
            continue
        a = module.A.detach().to(_DEPLOY_DTYPE)
        b = module.B.detach().to(_DEPLOY_DTYPE)
        weights[name] = (fake_quantize(b, spec) @ fake_quantize(a, spec)) if quantize else (b @ a)
    if not weights:
        raise ValueError("model has no LowRankLinear layers; nothing to deploy")
    return weights


def clone_with_weights(model: nn.Module, weights: Mapping[str, Tensor]) -> nn.Module:
    """Deep-copy ``model`` and overwrite the dense weights named in ``weights``.

    Used to evaluate a *compressed* model inside the identical network structure: the factorized
    forward is mathematically the dense forward with ``W = B @ A``, so writing the effective (and
    optionally fake-quantized) weights back into a dense clone isolates the compression damage from
    any implementation difference.

    Raises:
        ValueError: if a named target is not a linear layer, or a weight has the wrong shape.
    """
    clone = copy.deepcopy(model)
    for name, weight in weights.items():
        try:
            module = clone.get_submodule(name)
        except AttributeError as exc:  # pragma: no cover - defensive
            raise ValueError(f"model has no submodule {name!r}") from exc
        if not isinstance(module, nn.Linear):
            raise ValueError(f"{name!r} is {type(module).__name__}, expected nn.Linear")
        if tuple(module.weight.shape) != tuple(weight.shape):
            raise ValueError(
                f"{name!r}: weight shape {tuple(weight.shape)} does not match the layer's "
                f"{tuple(module.weight.shape)}"
            )
        with torch.no_grad():
            module.weight.copy_(weight.to(module.weight.dtype))
    return clone


def check_against_dense(
    model: nn.Module,
    reference: nn.Module,
    inputs: Tensor,
    *,
    atol: float = 1e-5,
) -> float:
    """Max absolute logits difference between a factorized model and its dense reference.

    A cheap correctness guard used by the tests and by the sweep: at the SVD initialisation the two
    models agree to float32 tolerance.

    Returns:
        ``max |logits_fact - logits_dense|`` as a float.
    """
    with torch.no_grad():
        difference = (model(inputs) - reference(inputs)).abs().max()
    value = float(difference)
    if not math.isfinite(value) or value > atol:
        raise AssertionError(
            f"factorized forward differs from the dense reference by {value:g} (> atol={atol:g})"
        )
    return value


def parameter_counts(model: nn.Module) -> dict[str, int]:
    """Parameter counts of a possibly factorized model (for manifests and reports)."""
    total = sum(p.numel() for p in model.parameters())
    factorized = sum(
        module.A.numel() + module.B.numel() + (0 if module.bias is None else module.bias.numel())
        for module in model.modules()
        if isinstance(module, LowRankLinear)
    )
    return {"total": int(total), "factorized": int(factorized)}


def sequence_shape_report(model: nn.Module) -> Sequence[tuple[str, int, int, int]]:
    """``(name, out_features, in_features, rank)`` per factorized layer, sorted (report table)."""
    rows = [
        (name, module.out_features, module.in_features, module.rank)
        for name, module in model.named_modules()
        if isinstance(module, LowRankLinear) and name
    ]
    return sorted(rows)
