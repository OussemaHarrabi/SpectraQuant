"""Rounding-aware spectral preparation objective (Milestone 5, hypothesis H3).

This module is the coupling between low-rank *preparation* and low-bit *training*: it shapes the
spectrum of the weights that quantization will subsequently see. Everything here is a
**measurement-class-2** object (``AGENTS.md`` section 5): float arithmetic that simulates the
numerics of quantize-to-grid, never low-bit storage (class 3) and never accelerated inference
(class 4/5).

The preregistration's objective is
``task_loss + lambda_factor * factorization_proxy + lambda_round * quantization_proxy
+ lambda_spectrum * spectral_regularizer`` (the task loss belongs to the caller). The terms are
exposed separately by :class:`PreparationObjective` so each can be ablated by a config arm.

Which terms are actually needed
-------------------------------
The predeclared first candidate was the exact rounding-residual ratio
``||Q(B)Q(A) - B A||_F^2 / ||B A||_F^2``, made differentiable by the straight-through estimator
(STE) that :func:`spectraquant.quantization.fake_quantize` already implements. It is implemented here
as :func:`rounding_residual_ratio` and monitored by the objective's ``residual`` term, but it is
**not** the term the direction test uses: its STE gradient is degenerate. With
``fq(X) = deq(X) + (X - X.detach())`` the Jacobian of ``fq`` is the identity, so the only surviving
gradient of ``||fq(B)fq(A) - B A||^2`` is a product of *two* rounding errors
(``2 (Q(B) - B)^T M`` with ``M = Q(B)(Q(A) - A) + (Q(B) - B) A``), a poor descent direction that
empirically does not lower the measured residual (measured; see ``docs/results/regularizer-report.md``).

The term that does the work is :func:`rounding_grid_penalty`, a smooth, scale-weighted surrogate of
the *squared* rounding error

    ``P(W) = sum_i (s_i^2 / 4) sin^2(pi w_i / s_i) / sum_i w_i^2``

where ``s_i`` is the per-element grid step of the target quantizer (per-tensor / per-channel /
per-group, from :func:`spectraquant.quantization.quant_params`, used as a *detached* calibration
constant). Writing ``delta_i = w_i - round(w_i / s_i) s_i``, the element term ``(s^2/4) sin^2(pi
delta / s)`` equals the true squared rounding error ``delta^2`` at the cell boundary (both are
``s^2/4``) and is ``(pi^2/4) delta^2`` near it: same zero set (the code grid), same cell amplitude,
same scale weighting. Its gradient ``(pi s / 4) sin(2 pi w / s)`` is genuinely non-zero, and the
penalty is tied to the rounding grid at the target bit width — which is the H3 claim itself.

The spectral term (:func:`spectral_tail_energy`) is the fraction of the effective weight's squared
singular-value energy beyond ``tail_rank``; it contains no quantization grid and is exactly the
plain-spectral **control arm** of the H3 falsifier. It is identically zero when ``tail_rank`` is at
least ``min(rows, cols)``, so the objective keeps it off by default and the report says why.

Conventions (frozen, ``docs/coordination/design-m2-interfaces.md`` section 0.2)
------------------------------------------------------------------------------
``W ~= B @ A`` with ``A: (rank, in_features)`` and ``B: (out_features, rank)``.

Dtypes / device
---------------
CPU only (``AGENTS.md`` section 2). float32 is the working dtype; float64 factors are honoured. No
function here moves a tensor to CUDA and no CUDA-only package is imported.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from spectraquant.quantization import QuantSpec, axis_view, fake_quantize, quant_params

if TYPE_CHECKING:  # pragma: no cover - import-cycle guard for type checkers only
    from spectraquant.config import RegularizerConfig

__all__ = [
    "FactorPair",
    "PreparationObjective",
    "PreparationTerms",
    "effective_rank",
    "factorization_proxy",
    "measured_factor_rounding_residual",
    "measured_product_rounding_residual",
    "orthogonality_penalty",
    "rounding_grid_penalty",
    "rounding_residual_ratio",
    "spectral_penalty",
    "spectral_tail_energy",
]

#: A low-rank factorization of one layer: ``(A, B)`` with ``A: (rank, in)``, ``B: (out, rank)``.
FactorPair = tuple[Tensor, Tensor]

#: Denominator guard for ratios whose factors could be exactly zero.
_TINY = 1e-30


# --------------------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------------------
def _check_factor_pair(name: str, pair: FactorPair) -> tuple[Tensor, Tensor]:
    """Validate one ``(A, B)`` pair and return it as ``(A, B)``."""
    if not isinstance(pair, tuple) or len(pair) != 2:
        raise TypeError(f"{name!r}: a factor pair must be a (A, B) tuple, got {pair!r}")
    a, b = pair
    for label, tensor in (("A", a), ("B", b)):
        if not isinstance(tensor, Tensor):
            raise TypeError(f"{name!r}.{label} must be a torch.Tensor, got {type(tensor).__name__}")
        if tensor.ndim != 2:
            raise ValueError(f"{name!r}.{label} must be 2-D, got shape {tuple(tensor.shape)}")
        if not tensor.is_floating_point():
            raise TypeError(f"{name!r}.{label} must be a float tensor, got {tensor.dtype}")
        if tensor.device.type != "cpu":  # pragma: no cover - no CUDA on the workstation of record
            raise ValueError(
                f"{name!r}.{label} is on device {tensor.device}; only CPU is supported"
            )
    if a.shape[0] != b.shape[1]:
        raise ValueError(
            f"{name!r}: rank mismatch between A {tuple(a.shape)} and B {tuple(b.shape)}; expected "
            "A (rank, in) and B (out, rank)"
        )
    return a, b


def _require_non_empty(factors: Mapping[str, FactorPair]) -> None:
    if not factors:
        raise ValueError("the objective needs at least one (name -> (A, B)) factor pair")


def _common_dtype(factors: Mapping[str, FactorPair]) -> torch.dtype:
    dtypes = {pair[0].dtype for pair in factors.values()} | {
        pair[1].dtype for pair in factors.values()
    }
    if len(dtypes) != 1:
        raise ValueError(f"all factors must share one dtype, got {sorted(map(str, dtypes))}")
    return next(iter(dtypes))


def _common_device(factors: Mapping[str, FactorPair]) -> torch.device:
    devices = {pair[0].device for pair in factors.values()} | {
        pair[1].device for pair in factors.values()
    }
    if len(devices) != 1:
        raise ValueError("all factors must live on one device")
    return next(iter(devices))


def _validated(factors: Mapping[str, FactorPair]) -> dict[str, tuple[Tensor, Tensor]]:
    """Validate every pair up front so a malformed factor fails with the specific error."""
    return {name: _check_factor_pair(name, pair) for name, pair in factors.items()}


def _zero(factors: Mapping[str, FactorPair]) -> Tensor:
    return torch.zeros((), dtype=_common_dtype(factors), device=_common_device(factors))


def _grid_step(tensor: Tensor, spec: QuantSpec) -> tuple[Tensor, Tensor]:
    """Return the block ``view`` of ``tensor`` and its per-element grid step for ``spec``.

    Mirrors the frozen block convention of :mod:`spectraquant.quantization.fake_quant` (row-major
    ``reshape(outer, axis_len, inner)`` with blocks running along ``spec.axis``). The duplication of
    that expansion rule is pinned by
    ``tests/unit/test_regularizer_objective.py::test_grid_step_matches_the_quantizer_expansion``,
    which asserts equality with the step recovered through the public
    :func:`spectraquant.quantization.fake_dequantize`.

    The step is returned detached: it is a calibration statistic, exactly as ``fake_quantize`` treats
    its own parameters.
    """
    outer, axis_len, inner = axis_view(tuple(tensor.shape), spec.axis)
    view = tensor.reshape(outer, axis_len, inner)
    params = quant_params(tensor, spec)
    if spec.granularity == "per_group":
        assert spec.group_size is not None  # guaranteed by QuantSpec.__post_init__
        index = torch.arange(axis_len, device=tensor.device) // spec.group_size
        step = params.scales.index_select(1, index)
    else:
        step = params.scales.expand(outer, axis_len, inner)
    return view, step.detach().to(view.dtype)


def _reference_for(name: str, references: Mapping[str, Tensor] | None, effective: Tensor) -> Tensor:
    if references is None:
        return effective
    if name not in references:
        raise KeyError(f"references is missing an entry for layer {name!r}")
    reference = references[name]
    if not isinstance(reference, Tensor):
        raise TypeError(f"references[{name!r}] must be a torch.Tensor")
    if reference.shape != effective.shape:
        raise ValueError(
            f"references[{name!r}] has shape {tuple(reference.shape)}, expected "
            f"{tuple(effective.shape)}"
        )
    return reference


# --------------------------------------------------------------------------------------
# Penalty components
# --------------------------------------------------------------------------------------
def factorization_proxy(
    factors: Mapping[str, FactorPair],
    references: Mapping[str, Tensor] | None = None,
) -> Tensor:
    """Pooled relative reconstruction error ``||W_ref - B A||_F^2 / ||W_ref||_F^2``.

    Shapes:
        ``factors``: ``name -> (A, B)``; ``references``: ``name -> W_ref`` with
        ``W_ref.shape == (B.shape[0], A.shape[1])``. When ``references`` is ``None`` each layer's
        reference is its own ``B @ A``, which makes the term identically zero; that keeps the
        factorization-ablated arm on one code path and reports a zero term instead of skipping it.

    Returns:
        A 0-dim tensor. Numerator and denominator are summed over layers before the ratio, so the
        term is a ratio of total energies, not an average of per-layer ratios.

    Dtypes / device:
        Same dtype/device as the factors. Measurement class 1 (a reconstruction quantity), never a
        quality claim.
    """
    _require_non_empty(factors)
    checked = _validated(factors)
    numerator = _zero(checked)
    denominator = _zero(checked)
    for name, (a, b) in checked.items():
        effective = b @ a
        reference = _reference_for(name, references, effective)
        numerator = numerator + (reference - effective).pow(2).sum()
        denominator = denominator + reference.pow(2).sum()
    return numerator / denominator.clamp_min(_TINY)


def rounding_grid_penalty(factors: Mapping[str, FactorPair], spec: QuantSpec) -> Tensor:
    """Scale-weighted smooth surrogate of the factors' squared rounding error on the code grid.

    Value, pooled over layers and over both factors::

        sum_l sum_{X in (A_l, B_l)} sum_i (s_i^2 / 4) sin^2(pi x_i / s_i)
        -----------------------------------------------------------------
                     sum_l sum_{X in (A_l, B_l)} ||X||_F^2

    with ``s_i`` the detached per-element grid step implied by ``spec``.

    Returns:
        A 0-dim tensor in roughly ``[0, pi^2/16]`` per element, dimensionless and invariant to a
        global rescaling of the factors.

    Dtypes / device:
        Same dtype/device as the factors; the step is cast to the factor dtype and carries no
        gradient (calibration statistics).

    Limitations:
        * The surrogate does **not** model the clip at ``+/- qmax * s``; it is valid only while the
          weights stay inside the calibrated range, which constrains ``lambda_round``.
        * The grid step is recomputed from the *current* weights on every call, as
          :func:`fake_quantize` does, so the penalty chases a moving grid; a frozen calibration is the
          caller's business and is not implemented here.
        * ``sin^2`` is periodic, so a very large step can move an element to a *different* grid cell
          without lowering the true residual as much as the surrogate suggests. The direction test in
          the unit suite and the coefficient sweep both bound the usable range empirically.
    """
    _require_non_empty(factors)
    checked = _validated(factors)
    numerator = _zero(checked)
    denominator = _zero(checked)
    for pair in checked.values():
        for tensor in pair:
            view, step = _grid_step(tensor, spec)
            # The periodic argument is evaluated in float64 on purpose: in float32 the argument
            # ``pi * w / s`` reaches ~22 for int4 weights, and its own rounding error (~1e-6) would
            # show up as a spurious ~1e-12 penalty value at a grid point. float64 keeps the value at
            # a grid point below 1e-24, i.e. numerically zero, while the gradient stays exact.
            view = view.to(torch.float64)
            step = step.to(torch.float64)
            numerator = (
                numerator + ((step.pow(2) / 4.0) * torch.sin(math.pi * view / step).pow(2)).sum()
            )
            denominator = denominator + view.pow(2).sum()
    return (numerator / denominator.clamp_min(_TINY)).to(_common_dtype(checked))


def rounding_residual_ratio(factors: Mapping[str, FactorPair], spec: QuantSpec) -> Tensor:
    """Exact pooled rounding-residual ratio ``||Q(B)Q(A) - B A||_F^2 / ||B A||_F^2``.

    The prescribed first candidate: the factors are fake-quantized with
    :func:`spectraquant.quantization.fake_quantize` (straight-through estimator), so the expression
    is differentiable. Its STE gradient is degenerate (see the module docstring), so the objective
    carries it as a *monitored* term (``residual``, coefficient ``lambda_residual``, default ``0.0``))
    rather than as the term the direction test uses.

    Returns:
        A 0-dim tensor pooled over layers (numerator and denominator summed before the ratio). Its
        forward value is the square of :func:`measured_product_rounding_residual`.

    Dtypes / device:
        Same dtype/device as the factors.
    """
    _require_non_empty(factors)
    checked = _validated(factors)
    numerator = _zero(checked)
    denominator = _zero(checked)
    for a, b in checked.values():
        effective = b @ a
        quantized = fake_quantize(b, spec) @ fake_quantize(a, spec)
        numerator = numerator + (quantized - effective).pow(2).sum()
        denominator = denominator + effective.pow(2).sum()
    return numerator / denominator.clamp_min(_TINY)


def spectral_tail_energy(factors: Mapping[str, FactorPair], *, tail_rank: int) -> Tensor:
    """Pooled fraction of squared singular-value energy beyond ``tail_rank``.

    Value: ``sum_l sum_{i > k} sigma_i(B_l A_l)^2 / sum_l sum_i sigma_i(B_l A_l)^2`` with
    ``k = tail_rank``. This is a **plain spectral penalty** (no quantization grid appears in it) and
    is the control arm of the H3 falsifier ("the improvement vanishes when the rounding-grid term is
    replaced by a plain spectral penalty").

    Returns:
        A 0-dim tensor in ``[0, 1]``, **numerically zero** (~1e-15, float32 SVD noise) when
        ``tail_rank >= min(out, in)`` for every layer, because a rank-``r`` product carries no energy
        beyond rank ``r``. The objective reports the zero rather than hiding it, and the report states
        the consequence.

    Dtypes / device:
        Same dtype/device as the factors.

    Gradients / limitations:
        Differentiable through ``torch.linalg.svdvals``. The gradient of a *repeated* singular value
        is not defined; PyTorch returns a finite (generally valid subgradient) result on the
        rank-deficient products used here, but a caller landing on exactly repeating non-zero
        singular values should not trust the direction (tested finite, not proven optimal).
    """
    if isinstance(tail_rank, bool) or not isinstance(tail_rank, int):
        raise TypeError(f"tail_rank must be an int, got {type(tail_rank).__name__}")
    if tail_rank < 1:
        raise ValueError(f"tail_rank must be >= 1, got {tail_rank}")
    _require_non_empty(factors)
    checked = _validated(factors)
    tail = _zero(checked)
    total = _zero(checked)
    for a, b in checked.values():
        energy = torch.linalg.svdvals(b @ a).pow(2)
        tail = tail + energy[tail_rank:].sum()
        total = total + energy.sum()
    return tail / total.clamp_min(_TINY)


# --------------------------------------------------------------------------------------
# Measured (no-grad) diagnostics — class 2
# --------------------------------------------------------------------------------------
def measured_factor_rounding_residual(factors: Mapping[str, FactorPair], spec: QuantSpec) -> float:
    """Measured factor rounding residual ``sqrt(sum ||Q(X) - X||_F^2 / sum ||X||_F^2)``.

    The hard-rounding quantity the ``rounding`` term claims to control (its factor projection),
    computed with no gradient in float64. Measurement class 2.
    """
    _require_non_empty(factors)
    checked = _validated(factors)
    numerator = 0.0
    denominator = 0.0
    for pair in checked.values():
        for tensor in pair:
            work = tensor.detach().to(torch.float64)
            residual = fake_quantize(work, spec, quant_params(work, spec)) - work
            numerator += float(residual.pow(2).sum())
            denominator += float(work.pow(2).sum())
    if denominator <= 0.0:
        return 0.0
    return math.sqrt(numerator / denominator)


def measured_product_rounding_residual(factors: Mapping[str, FactorPair], spec: QuantSpec) -> float:
    """Measured relative Frobenius residual of the product ``||Q(B)Q(A) - B A||_F / ||B A||_F``.

    The measured counterpart of :func:`rounding_residual_ratio` (the prescribed H3 quantity) and the
    quantity the deployed rank-then-quantize pipeline actually incurs. No gradient, float64,
    measurement class 2.
    """
    _require_non_empty(factors)
    checked = _validated(factors)
    numerator = 0.0
    denominator = 0.0
    for a, b in checked.values():
        a64 = a.detach().to(torch.float64)
        b64 = b.detach().to(torch.float64)
        effective = b64 @ a64
        quantized = fake_quantize(b64, spec, quant_params(b64, spec)) @ fake_quantize(
            a64, spec, quant_params(a64, spec)
        )
        numerator += float((quantized - effective).pow(2).sum())
        denominator += float(effective.pow(2).sum())
    if denominator <= 0.0:
        return 0.0
    return math.sqrt(numerator / denominator)


# --------------------------------------------------------------------------------------
# The objective
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PreparationTerms:
    """The objective's components for one set of factors.

    Every tensor field is 0-dim and connected to the autograd graph; ``total`` is the weighted sum
    the caller adds to the task loss, and ``weights`` records the coefficients actually applied so a
    manifest or a report can state the arm from the result alone.
    """

    factorization: Tensor
    rounding: Tensor
    residual: Tensor
    spectrum: Tensor
    total: Tensor
    weights: Mapping[str, float]

    def to_dict(self) -> dict[str, float]:
        """Detached floats for per-term diagnostics."""
        values = {
            "factorization": self.factorization,
            "rounding": self.rounding,
            "residual": self.residual,
            "spectrum": self.spectrum,
            "total": self.total,
        }
        payload = {name: float(value.detach()) for name, value in values.items()}
        payload.update({name: float(weight) for name, weight in self.weights.items()})
        return payload


class PreparationObjective(nn.Module):
    """Rounding-aware preparation objective with every preregistration term ablatable.

    ``total = lambda_factor * factorization + lambda_round * rounding
    + lambda_residual * residual + lambda_spectrum * spectrum``

    The task loss is **not** included: the training loop adds it, because only the loop knows the
    data. Every coefficient defaults to ``0.0``, so the default objective is an exact differentiable
    zero — an omitted weight shows up in the diagnostics instead of being silently applied.

    Args:
        spec: target quantizer for the data-free terms (``rounding`` and ``residual``).
        lambda_factor / lambda_round / lambda_residual / lambda_spectrum: non-negative weights.
        tail_rank: largest singular direction kept by :func:`spectral_tail_energy`; required when
            ``lambda_spectrum > 0`` and forbidden otherwise.
        name: label recorded in diagnostics and manifests (the config arm).

    Shapes / dtypes / device:
        Holds no tensors: a :class:`QuantSpec` (ints/strings), floats and a name. Every tensor it
        produces lives on the factors' device (CPU by construction) with the factors' dtype.

    Cost:
        One :meth:`terms` call computes all four components (they are cheap and the diagnostics are
        always wanted). Per layer that is one ``quant_params`` + one ``fake_quantize`` pair per
        factor plus one ``svdvals`` of the effective weight, all on CPU in the factors' dtype; this
        is a Tier-0 fixture cost model, not a memory-optimised production path (the reference
        quantizer materialises tensors of the factors' size — see
        :mod:`spectraquant.quantization.fake_quant`).

    Raises:
        TypeError / ValueError: on a non-:class:`QuantSpec` spec, a non-finite/negative coefficient,
            an empty name, or a ``tail_rank``/``lambda_spectrum`` combination that cannot be applied.
    """

    def __init__(
        self,
        *,
        spec: QuantSpec,
        lambda_factor: float = 0.0,
        lambda_round: float = 0.0,
        lambda_residual: float = 0.0,
        lambda_spectrum: float = 0.0,
        tail_rank: int | None = None,
        name: str = "regularizer",
    ) -> None:
        super().__init__()
        if not isinstance(spec, QuantSpec):
            raise TypeError(f"spec must be a QuantSpec, got {type(spec).__name__}")
        for label, value in (
            ("lambda_factor", lambda_factor),
            ("lambda_round", lambda_round),
            ("lambda_residual", lambda_residual),
            ("lambda_spectrum", lambda_spectrum),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{label} must be a number, got {type(value).__name__}")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{label} must be finite and >= 0, got {value!r}")
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")
        if lambda_spectrum > 0.0:
            if isinstance(tail_rank, bool) or not isinstance(tail_rank, int) or tail_rank < 1:
                raise ValueError(
                    f"tail_rank must be an int >= 1 when lambda_spectrum > 0, got {tail_rank!r}"
                )
        elif tail_rank is not None:
            raise ValueError("tail_rank must be None when lambda_spectrum == 0")

        self.spec = spec
        self.name = name
        self.lambda_factor = float(lambda_factor)
        self.lambda_round = float(lambda_round)
        self.lambda_residual = float(lambda_residual)
        self.lambda_spectrum = float(lambda_spectrum)
        self.tail_rank = tail_rank

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_config(cls, cfg: RegularizerConfig, *, spec: QuantSpec) -> PreparationObjective:
        """Build the objective from a validated :class:`spectraquant.config.RegularizerConfig`."""
        return cls(
            spec=spec,
            lambda_factor=cfg.lambda_factor,
            lambda_round=cfg.lambda_round,
            lambda_residual=cfg.lambda_residual,
            lambda_spectrum=cfg.lambda_spectrum,
            tail_rank=cfg.tail_rank,
            name=cfg.name,
        )

    def weights(self) -> dict[str, float]:
        """The coefficient applied to each component (for manifests and reports)."""
        return {
            "lambda_factor": self.lambda_factor,
            "lambda_round": self.lambda_round,
            "lambda_residual": self.lambda_residual,
            "lambda_spectrum": self.lambda_spectrum,
        }

    def is_noop(self) -> bool:
        """True when every coefficient is zero (the objective contributes nothing)."""
        return all(value == 0.0 for value in self.weights().values())

    # ------------------------------------------------------------------ forward
    def terms(
        self,
        factors: Mapping[str, FactorPair],
        *,
        references: Mapping[str, Tensor] | None = None,
    ) -> PreparationTerms:
        """Evaluate every component on ``factors`` and combine them with the configured weights.

        Args:
            factors: ``name -> (A, B)`` for the prepared layers. Gradients flow to both ``A`` and
                ``B``.
            references: optional ``name -> W_ref`` dense reference for the factorization term.

        Returns:
            The :class:`PreparationTerms` bundle; ``total`` is the differentiable scalar to add to
            the task loss.

        Raises:
            ValueError / TypeError / KeyError: on empty or malformed factors, or a reference whose
                shape does not match the layer it names.
        """
        _require_non_empty(factors)
        factorization = factorization_proxy(factors, references)
        rounding = rounding_grid_penalty(factors, self.spec)
        residual = rounding_residual_ratio(factors, self.spec)
        if self.lambda_spectrum > 0.0:
            assert self.tail_rank is not None  # guaranteed by __init__
            spectrum = spectral_tail_energy(factors, tail_rank=self.tail_rank)
        else:
            spectrum = _zero(factors)
        total = (
            self.lambda_factor * factorization
            + self.lambda_round * rounding
            + self.lambda_residual * residual
            + self.lambda_spectrum * spectrum
        )
        return PreparationTerms(
            factorization=factorization,
            rounding=rounding,
            residual=residual,
            spectrum=spectrum,
            total=total,
            weights=self.weights(),
        )

    def forward(
        self,
        factors: Mapping[str, FactorPair],
        *,
        references: Mapping[str, Tensor] | None = None,
    ) -> Tensor:
        """The weighted penalty scalar (see :meth:`terms`)."""
        return self.terms(factors, references=references).total

    def extra_repr(self) -> str:
        weights = ", ".join(f"{key}={value:g}" for key, value in self.weights().items())
        return f"name={self.name!r}, tail_rank={self.tail_rank}, {weights}"


# --------------------------------------------------------------------------------------
# Retained model-level API (Milestone-5 contract, now implemented)
# --------------------------------------------------------------------------------------
def spectral_penalty(
    model: nn.Module,
    *,
    target_rank: int,
    strength: float = 1.0,
    exclude: Iterable[str] = (),
) -> Tensor:
    """Plain spectral penalty over a model's 2-D weights: energy beyond ``target_rank``.

    For every named parameter that is 2-D and whose name does not start with an ``exclude`` prefix,
    the term is ``sum_{i > target_rank} sigma_i(W)^2 / sum_i sigma_i(W)^2``; the returned scalar is
    the sum over matched parameters times ``strength``. This is the "plain spectral penalty" of the
    H3 falsifier: it shapes the spectrum and knows nothing about the quantization grid.

    Shapes / dtypes / device:
        ``model`` on CPU; matched weights keep their dtype. Returns a 0-dim tensor.

    Raises:
        ValueError: if ``target_rank < 1``, ``strength`` is not finite and non-negative, or no 2-D
            parameter matches (a silent no-op would hide a typo in the exclusion list).
    """
    if isinstance(target_rank, bool) or not isinstance(target_rank, int) or target_rank < 1:
        raise ValueError(f"target_rank must be an int >= 1, got {target_rank!r}")
    if not math.isfinite(strength) or strength < 0.0:
        raise ValueError(f"strength must be finite and >= 0, got {strength!r}")
    prefixes = tuple(exclude)
    total: Tensor | None = None
    for name, parameter in model.named_parameters():
        if parameter.ndim != 2 or any(name.startswith(prefix) for prefix in prefixes):
            continue
        energy = torch.linalg.svdvals(parameter).pow(2)
        layer = energy[target_rank:].sum() / energy.sum().clamp_min(_TINY)
        total = layer if total is None else total + layer
    if total is None:
        raise ValueError(
            "spectral_penalty matched no 2-D parameter; check the model and the `exclude` prefixes"
        )
    return strength * total


def orthogonality_penalty(factors: FactorPair, *, strength: float = 1.0) -> Tensor:
    """Scale-invariant orthonormality penalty on one ``(A, B)`` factor pair.

    Each factor is normalised to ``X_hat = X sqrt(rank) / ||X||_F`` (so ``||X_hat||_F^2 == rank``) and
    the penalty is ``(||X_hat X_hat^T - I||_F^2 + ||Y_hat^T Y_hat - I||_F^2) / (2 rank)``.
    Normalising keeps the term from being trivially satisfied by shrinking the factors.

    Shapes:
        ``A``: ``(rank, in)``, ``B``: ``(out, rank)``; returns a 0-dim tensor.

    Assumptions:
        Retained Milestone-5 API; the H3 objective does not use it and the training loop never adds
        it silently.
    """
    a, b = _check_factor_pair("factors", factors)
    if not math.isfinite(strength) or strength < 0.0:
        raise ValueError(f"strength must be finite and >= 0, got {strength!r}")
    rank = a.shape[0]
    eye = torch.eye(rank, dtype=a.dtype, device=a.device)
    scaled_a = a * math.sqrt(rank) / a.norm().clamp_min(_TINY)
    scaled_b = b * math.sqrt(rank) / b.norm().clamp_min(_TINY)
    gram_a = scaled_a @ scaled_a.transpose(0, 1)
    gram_b = scaled_b.transpose(0, 1) @ scaled_b
    return strength * ((gram_a - eye).pow(2).sum() + (gram_b - eye).pow(2).sum()) / (2.0 * rank)


def effective_rank(weight: Tensor, *, tolerance: float = 1e-3) -> float:
    """Entropy-based effective rank of a 2-D weight matrix.

    Definition: with ``sigma`` the singular values in float64, drop those below
    ``tolerance * max(sigma)``, set ``p_i = sigma_i^2 / sum(sigma^2)`` and return
    ``exp(-sum_i p_i log p_i)`` (the exponential of the Shannon entropy of the energy distribution);
    it equals the rank for an exactly rank-``r`` matrix and degrades smoothly otherwise.

    Shapes / dtype / device:
        ``weight``: 2-D CPU float tensor; returns a Python float (``0.0`` for the zero matrix).

    Raises:
        TypeError: if ``weight`` is not a tensor.
        ValueError: if ``weight`` is not 2-D, is empty, or ``tolerance <= 0``.
    """
    if not isinstance(weight, Tensor):
        raise TypeError(f"weight must be a torch.Tensor, got {type(weight).__name__}")
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2-D, got shape {tuple(weight.shape)}")
    if weight.numel() == 0:
        raise ValueError("weight must not be empty")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError(f"tolerance must be finite and > 0, got {tolerance!r}")
    singular = torch.linalg.svdvals(weight.detach().to(torch.float64))
    if singular.numel() == 0 or float(singular.max()) <= 0.0:
        return 0.0
    kept = singular[singular >= tolerance * singular.max()]
    energy = kept.pow(2)
    total = float(energy.sum())
    if total <= 0.0:
        return 0.0
    probabilities = energy / total
    entropy = float(-(probabilities * probabilities.clamp_min(_TINY).log()).sum())
    return math.exp(entropy)
