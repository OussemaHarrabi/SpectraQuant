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
from spectraquant.factorization.lr_qat import fixed_point8_downcast, fixed_point8_upcast
from spectraquant.quantization import QuantSpec, accounted_bytes, fake_quantize

__all__ = [
    "LowRankLinear",
    "QuantizedPlusLowRankLinear",
    "StraightThroughQuantizedLinear",
    "check_against_dense",
    "clone_with_weights",
    "deployed_weights",
    "factor_storage_bytes",
    "factorize_linears",
    "gather_factor_pairs",
    "low_rank_layer_names",
    "parameter_counts",
    "replace_linears_quantized",
    "replace_linears_straight_through",
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


class QuantizedPlusLowRankLinear(nn.Module):
    """``y = x @ (Wq + B @ A)^T + bias`` with a **frozen** dequantized base ``Wq`` and trainable factors.

    This is the structure the reproduction arms need: LR-QAT trains a low-rank auxiliary weight on
    top of a quantized main weight, and LoftQ initialises the factors at the quantized residual of the
    alternating schedule. ``Wq`` is a buffer, so the base cannot drift; only ``A`` and ``B`` are
    optimised.

    Shapes:
        Input ``(*, in_features)``; output ``(*, out_features)``. ``Wq`` is ``(out, in)``,
        ``A`` is ``(rank, in)``, ``B`` is ``(out, rank)``.

    Dtypes / device:
        CPU, float32. ``Wq`` is float32 (the *dequantized* values), which is what the forward pass
        computes in; the quantized codes and scales are the stored artifact, accounted separately.

    Gradients:
        ``A`` and ``B`` are leaves; ``Wq`` is not (it is a buffer), so the base is frozen by
        construction rather than by ``requires_grad=False`` bookkeeping.

    Assumptions / limitations:
        ``Wq + B @ A`` is materialised every forward pass: this is the training representation, not
        a deployed kernel path. The *stored* object is ``(quantized Wq, A, B)`` and its byte count is
        the sum of those three, which is what the equal-memory comparison must use.

    The LR-QAT step size (``step_size``)
        When ``step_size`` is given, the forward is the LR-QAT φ-forward instead of ``Wq + B @ A``:
        the base is frozen into the paper's 8-bit fixed-point container
        (``phi0_codes = fixed_point8_downcast(base / step_size)``, "Q4.4" by default), the factors
        are added **inside** the rounding, the rounding is straight-through, and the result is
        scaled by the *trainable* scalar ``step_size``:

        .. code-block:: text

            z = upcast(phi0_codes) + (alpha / rank) * (B @ A)
            W = step_size * round_ste(z)          # round_ste: d round / d z = 1

        ``step_size`` is a third trainable parameter, so an optimiser can give it its own learning
        rate (LR-QAT's ``lr_s``) while ``A``/``B`` keep theirs. With ``step_size=None`` (the
        default) the class behaves exactly as before and its parameter set is exactly ``{A, B}``.
        The φ-forward uses a *single* scalar ``s`` per layer, while an RTN base has one scale per
        group; folding per-group scales into this layer is not implemented (the same limitation
        :func:`spectraquant.factorization.lr_qat.merge_lr_qat` documents).
    """

    #: Declared for the type checker: ``register_buffer`` assigns through ``nn.Module.__setattr__``.
    base: Tensor
    #: Present only when the layer carries the LR-QAT step size (``step_size`` is not ``None``).
    phi0_codes: Tensor
    bias: nn.Parameter | None
    step_size: nn.Parameter | None

    def __init__(
        self,
        base: Tensor,
        rank: int,
        *,
        bias: Tensor | None = None,
        dtype: torch.dtype = torch.float32,
        step_size: float | None = None,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        if base.ndim != 2:
            raise ValueError(f"base must be 2-D (out, in), got shape {tuple(base.shape)}")
        out_features, in_features = int(base.shape[0]), int(base.shape[1])
        for label, value in (("rank", rank), ("in_features", in_features)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be an int >= 1, got {value!r}")
        if rank > min(out_features, in_features):
            raise ValueError(
                f"rank {rank} exceeds min(out, in)={min(out_features, in_features)} "
                f"for a {out_features}x{in_features} base"
            )
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise ValueError(f"alpha must be a real scalar, got {alpha!r}")
        if not math.isfinite(float(alpha)):
            raise ValueError(f"alpha must be finite, got {alpha!r}")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.register_buffer("base", base.detach().to(dtype).clone())
        self.A = nn.Parameter(torch.zeros(rank, in_features, dtype=dtype))
        self.B = nn.Parameter(torch.zeros(out_features, rank, dtype=dtype))
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias.detach().to(dtype).clone())
        if step_size is None:
            # `None` is not a trainable parameter, so the parameter set stays exactly `{A, B}`.
            self.register_parameter("step_size", None)
        else:
            if (
                isinstance(step_size, bool)
                or not isinstance(step_size, (int, float))
                or not math.isfinite(float(step_size))
                or float(step_size) <= 0.0
            ):
                raise ValueError(
                    f"step_size must be a finite scalar > 0 or None, got {step_size!r}"
                )
            self.step_size = nn.Parameter(torch.tensor(float(step_size), dtype=dtype))
            # Frozen φ₀ container: the base divided by its initial scale, rounded into the paper's
            # 8-bit fixed point. It is a buffer, so training moves `s` and the factors, never it.
            with torch.no_grad():
                codes = fixed_point8_downcast(self.base / float(step_size))
            self.register_buffer("phi0_codes", codes)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int,
        *,
        base: Tensor | None = None,
        factors: tuple[Tensor, Tensor] | None = None,
        step_size: float | None = None,
        alpha: float = 1.0,
    ) -> QuantizedPlusLowRankLinear:
        """Build the layer for one arm.

        Args:
            linear: the dense layer being replaced (CPU, float32).
            rank: retained rank.
            base: the dequantized main weight ``Wq``; when omitted, the dense weight itself is used
                unquantized, which is only meaningful for an arm that quantizes nothing.
            factors: the ``(A, B)`` initialisation; when omitted the residual ``w - Wq`` is
                decomposed by truncated SVD, which is the plain "quantize then compensate" start.
                An arm with a specific schedule (LoftQ) passes its own factors.
            step_size: optional LR-QAT step size ``s`` (see the class docstring); ``None`` keeps the
                plain ``Wq + B @ A`` forward.
            alpha: LoRA scaling numerator in ``(alpha / rank) * B @ A`` (the paper fixes ``1.0``).

        Returns:
            The layer, with ``A``/``B`` set to ``factors`` or to the SVD of ``w - Wq``.
        """
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"linear must be an nn.Linear, got {type(linear).__name__}")
        dense = linear.weight.detach()
        wq = dense.clone() if base is None else base.detach()
        if tuple(wq.shape) != tuple(dense.shape):
            raise ValueError(
                f"base shape {tuple(wq.shape)} does not match the layer weight {tuple(dense.shape)}"
            )
        module = cls(
            wq,
            rank,
            bias=linear.bias,
            dtype=dense.dtype,
            step_size=step_size,
            alpha=alpha,
        )
        if factors is None:
            residual = dense - wq
            svd = truncated_svd(residual, rank)
            a, b = svd.A, svd.B
        else:
            a, b = factors
        if tuple(a.shape) != (rank, int(dense.shape[1])) or tuple(b.shape) != (
            int(dense.shape[0]),
            rank,
        ):
            raise ValueError(
                f"factors have shapes {tuple(a.shape)} and {tuple(b.shape)}; expected "
                f"({rank}, {int(dense.shape[1])}) and ({int(dense.shape[0])}, {rank})"
            )
        with torch.no_grad():
            module.A.copy_(a.detach().to(dense.dtype))
            module.B.copy_(b.detach().to(dense.dtype))
        return module

    @property
    def has_step_size(self) -> bool:
        """Whether this layer carries the trainable LR-QAT step size."""
        return self.step_size is not None

    def effective_weight(self) -> Tensor:
        """Return the weight the forward pass uses.

        ``Wq + B @ A`` without a step size; the LR-QAT φ-forward
        ``step_size * round_ste(upcast(phi0_codes) + (alpha / rank) * (B @ A))`` with one. Both are
        differentiable in ``A``/``B`` (and in ``step_size`` for the latter).
        """
        if self.step_size is None:
            return self.base + self.B @ self.A
        phi = fixed_point8_upcast(self.phi0_codes).to(dtype=self.base.dtype)
        z = phi + (self.alpha / self.rank) * (self.B @ self.A)
        # Straight-through rounding: the forward value is `round(z)`, the backward pass is `d/dz`.
        z = z + (torch.round(z) - z).detach()
        return self.step_size.to(dtype=z.dtype) * z

    def merged_weight(self) -> tuple[Tensor, Tensor]:
        """Fold the trained φ-adapter into the stored integer weight (LR-QAT's merge, gate T1-a).

        Returns:
            ``(codes, weight)`` from :func:`spectraquant.factorization.lr_qat.merge_lr_qat`:
            ``codes`` is the ``int8`` merged weight (what a packed container stores) and ``weight``
            is ``step_size * codes`` (the dequantized merged weight).

        Raises:
            ValueError: the layer carries no step size (nothing to fold).
        """
        from spectraquant.factorization.lr_qat import merge_lr_qat

        if self.step_size is None:
            raise ValueError(
                "this layer carries no LR-QAT step size; the merge is only defined for the "
                "φ-forward variant"
            )
        merged = merge_lr_qat(
            self.phi0_codes,
            self.A.detach(),
            self.B.detach(),
            step_size=float(self.step_size.detach()),
            alpha=self.alpha,
            rank=self.rank,
        )
        return merged.codes, merged.weight

    def forward(self, x: Tensor) -> Tensor:
        """Apply ``x @ effective_weight()^T + bias``."""
        return torch.nn.functional.linear(x, self.effective_weight(), self.bias)

    def extra_repr(self) -> str:
        step = "" if self.step_size is None else f", step_size={float(self.step_size):.6g}"
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, bias={self.bias is not None}{step}"
        )


class StraightThroughQuantizedLinear(nn.Module):
    """``y = x @ fake_quantize(W)^T + bias`` with **all** of ``W`` trainable (straight-through).

    This is the full-model-QAT layer (the R2 ``r2_fullqat_4bit`` arm): the dense linear's weight is
    a trainable parameter, and every forward pass quantizes it with ``spec``, dequantizes it and
    uses the dequantized value. It is deliberately a *separate* class from
    :class:`QuantizedPlusLowRankLinear`, because the trainable parameter sets differ: this one has a
    single dense ``weight`` (``out * in`` values), while the low-rank class has a frozen base plus
    two factors.

    Shapes:
        Input ``(*, in_features)``; output ``(*, out_features)``. ``weight`` is ``(out, in)`` (the
        name and shape of the linear layer it replaces), ``bias`` ``(out,)`` or absent.

    Dtypes / device:
        CPU; float32 or float64. ``fake_quantize`` works in float32 for float32 input, and the
        quantized value is cast back to the parameter's dtype, so the forward preserves dtype.
        No CUDA: the quantizer is CPU-only by design.

    Gradients:
        Straight-through: :func:`spectraquant.quantization.fake_quantize` returns
        ``dequantize(codes) + (w - w.detach())``, so ``d out / d W`` is the identity and the
        gradient that reaches ``W`` is exactly the gradient of the *dequantized* weight. The
        rounding's own Jacobian (zero almost everywhere) is deliberately replaced; this is the
        standard QAT estimator, not the true derivative.

    Assumptions / limitations:
        * **Straight-through is an approximation.** The reported gradient is not the derivative of
          the quantizer; it is the derivative through the dequantized value. No convergence or
          quality guarantee follows from it.
        * The forward materialises ``quant_params`` (and therefore a tensor of ``W``'s size) on every
          pass: this is the training representation, not a memory-optimised kernel path.
        * The *stored* artifact is the quantized weight (codes + scales), not this float tensor
          (measurement class 2 while training, class 3 only when a container is serialized).
    """

    weight: nn.Parameter
    bias: nn.Parameter | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        spec: QuantSpec,
        *,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
        weight: Tensor | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(spec, QuantSpec):
            raise TypeError(f"spec must be a QuantSpec, got {type(spec).__name__}")
        for label, value in (("in_features", in_features), ("out_features", out_features)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be an int >= 1, got {value!r}")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.spec = spec
        initial = (
            torch.zeros(out_features, in_features, dtype=dtype)
            if weight is None
            else weight.detach().to(dtype).clone()
        )
        if tuple(initial.shape) != (self.out_features, self.in_features):
            raise ValueError(
                f"weight shape {tuple(initial.shape)} does not match "
                f"({self.out_features}, {self.in_features})"
            )
        self.weight = nn.Parameter(initial)
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_features, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        spec: QuantSpec,
        *,
        weight: Tensor | None = None,
    ) -> StraightThroughQuantizedLinear:
        """Build the layer from a dense ``nn.Linear`` (its weight is the QAT starting point)."""
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"linear must be an nn.Linear, got {type(linear).__name__}")
        dense = linear.weight.detach()
        module = cls(
            int(linear.in_features),
            int(linear.out_features),
            spec,
            bias=linear.bias is not None,
            dtype=dense.dtype,
            weight=dense if weight is None else weight,
        )
        if linear.bias is not None and module.bias is not None:
            with torch.no_grad():
                module.bias.copy_(linear.bias.detach().to(dense.dtype))
        return module

    def effective_weight(self) -> Tensor:
        """The fake-quantized (dequantized) weight the forward pass multiplies by (differentiable)."""
        return fake_quantize(self.weight, self.spec)

    def forward(self, x: Tensor) -> Tensor:
        """Apply ``x @ fake_quantize(W)^T + bias``."""
        return torch.nn.functional.linear(x, self.effective_weight(), self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"spec=QuantSpec(bits={self.spec.bits}, granularity={self.spec.granularity!r}, "
            f"group_size={self.spec.group_size!r}), bias={self.bias is not None}"
        )


def replace_linears_straight_through(
    model: nn.Module,
    *,
    spec: QuantSpec,
    weights: Mapping[str, Tensor] | None = None,
) -> None:
    """Replace the named ``nn.Linear`` layers with :class:`StraightThroughQuantizedLinear`.

    Args:
        model: the module to modify **in place** (the loop owns it).
        spec: the quantizer every forward pass applies to the layer's trainable weight.
        weights: optional ``{layer name: starting weight}``; a named layer must exist and the
            mapping must not name an unknown layer (a typo would silently leave a layer dense, which
            changes both the arm and its byte accounting).

    Raises:
        TypeError: a named module is not an ``nn.Linear``.
        ValueError: a requested name matches no layer, or a starting weight has the wrong shape.
    """
    requested = dict(weights or {})
    layers = {
        name: module for name, module in model.named_modules() if isinstance(module, nn.Linear)
    }
    unknown = set(requested) - set(layers)
    if unknown:
        raise ValueError(f"weights refers to unknown linear layers: {sorted(unknown)}")
    if not requested:
        raise ValueError("no layer was requested for straight-through quantization")
    for name in sorted(requested):
        linear = layers[name]
        parent, _, attribute = name.rpartition(".")
        container = model.get_submodule(parent) if parent else model
        setattr(
            container,
            attribute,
            StraightThroughQuantizedLinear.from_linear(linear, spec, weight=requested[name]),
        )


def replace_linears_quantized(
    model: nn.Module,
    *,
    bases: Mapping[str, Tensor],
    factors: Mapping[str, tuple[Tensor, Tensor]],
    step_sizes: Mapping[str, float] | None = None,
) -> None:
    """Replace the named ``nn.Linear`` layers with :class:`QuantizedPlusLowRankLinear`, in place.

    This is the trainable arms' structural change: the frozen quantized base comes from the arm's
    initialisation (``bases``) and the factors from its schedule (``factors``). The replacement is
    keyed by layer name, so a layer the arm did not prepare keeps its dense weight - and the caller
    must have decided that deliberately, because an unprepared layer changes the byte accounting.

    Args:
        model: the module to modify **in place** (the loop owns it).
        bases: ``{layer name: dequantized main weight}`` for every layer to replace.
        factors: ``{layer name: (A, B)}``, the same key set as ``bases``.
        step_sizes: optional ``{layer name: initial step size}`` (LR-QAT). Keys must be a subset of
            ``bases``; an unlisted layer gets no step size and keeps the plain forward.

    Raises:
        KeyError: the two mappings do not describe the same layers, or ``step_sizes`` names an
            unknown layer.
        TypeError: a named module is not an ``nn.Linear``.
        ValueError: a rank exceeds the layer's ``min(out, in)`` or a factor shape is wrong.
    """
    if set(bases) != set(factors):
        missing = sorted(set(bases) ^ set(factors))
        raise KeyError(f"bases and factors describe different layers: {missing}")
    sizes = dict(step_sizes or {})
    unknown = set(sizes) - set(bases)
    if unknown:
        raise KeyError(
            f"step_sizes refers to layers that are not being replaced: {sorted(unknown)}"
        )
    for name in sorted(bases):
        parent, _, attribute = name.rpartition(".")
        container = model.get_submodule(parent) if parent else model
        linear = getattr(container, attribute, None)
        if not isinstance(linear, nn.Linear):
            raise TypeError(
                f"layer {name!r} is {type(linear).__name__}, not an nn.Linear: the arm's target set "
                "must match the model's actual linear layers"
            )
        rank = int(factors[name][0].shape[0])
        setattr(
            container,
            attribute,
            QuantizedPlusLowRankLinear.from_linear(
                linear,
                rank,
                base=bases[name],
                factors=factors[name],
                step_size=None if name not in sizes else float(sizes[name]),
            ),
        )


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
