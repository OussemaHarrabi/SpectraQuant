"""Uniform (affine) fake quantization and quantization parameters — measurement class 2.

This module implements the reference quantizer used by every other slice: it turns a float tensor
into integer codes and back **in float arithmetic** (``AGENTS.md`` section 5, class 2). A tensor
produced here is still a float tensor: it is never low-bit *storage* (class 3) and never accelerated
inference (class 4). The byte cost of a configuration is computed separately, from shapes only, in
:mod:`spectraquant.quantization.accounting` — the one byte-accounting source of truth
(``docs/coordination/design-m2-interfaces.md`` invariant 1).

Conventions (frozen by ``docs/coordination/design-m2-interfaces.md`` section 1)
---------------------------------------------------------------------------
* Codes are uniformly spaced and integer-valued:
  ``q = clamp(round(w / scale) + zero_point)`` and ``w_hat = (q - zero_point) * scale``.
* ``symmetric=True`` uses signed ranges: ``qmin = -2**(bits-1)``, ``qmax = 2**(bits-1) - 1``,
  ``zero_point = 0``, ``scale = max|w| / qmax`` per block.
* ``symmetric=False`` uses unsigned ranges: ``qmin = 0``, ``qmax = 2**bits - 1``. The block range is
  first widened to include zero (``lo = min(min w, 0)``, ``hi = max(max w, 0)`` — the ONNX Runtime
  ``DynamicQuantizeLinear`` convention), then ``scale = (hi - lo) / (qmax - qmin)`` and
  ``zero_point = clip(qmin - round(lo / scale), qmin, qmax)``. Zero is therefore represented exactly
  and the clip stays inactive for every block that is not all-zero.
* Blocks are formed **along** ``spec.axis``; with ``W`` viewed as ``(outer, axis_len, inner)``:
  - ``per_tensor``: one block covering every element — ``1`` block;
  - ``per_channel``: one block per index along ``axis``, each spanning all other axes —
    ``axis_len`` blocks;
  - ``per_group``: ``ceil(axis_len / group_size)`` consecutive blocks per ``(outer, inner)`` index
    pair — ``outer * inner * ceil(axis_len / group_size)`` blocks.
  This reproduces the ``n_blocks`` arithmetic of ``docs/protocols/memory-accounting.md`` section 2
  (``per_group`` along the input axis of ``W (out, in)`` gives ``out * ceil(in / g)``).
* Ties round **half to even** (``torch.round``), the convention of the numpy/ONNX reference
  kernels. ``round_mode`` accepts only ``"nearest"`` in M2.

Numerical limitations
---------------------
* Statistics accumulate in float32 (float64 for float64 input), so an extremely narrow dynamic range
  can yield a subnormal scale — or, for a block of subnormal values, make ``spread / qmax`` underflow
  to zero. The scale is never zero or NaN: a zero-range *or underflowing* block falls back to
  ``scale = 1.0`` with ``zero_point = 0`` when asymmetric, which rounds such values to code 0 (a
  bounded error no larger than the block's own magnitude). See
  :func:`_block_scale_and_zero_point` and
  ``tests/unit/test_quant_fake_quantize.py::test_subnormal_values_do_not_underflow_the_scale_into_nan``.
* The half-step bound ``|w - w_hat| <= scale / 2`` holds per element only when no clipping occurs;
  a clipped outlier contributes a systematic error of up to ``absmax - qmax * scale``.
* The reference implementation materialises a tensor the size of ``w`` (float32) while building
  per-block statistics, and expands per-block parameters to element resolution during
  quantize/dequantize. This is deliberate (clarity, no hidden broadcasts) and adequate for the
  Tier-0 fixtures in scope; it is not a memory-optimised production kernel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

__all__ = [
    "GRANULARITIES",
    "ROUND_MODES",
    "SUPPORTED_BITS",
    "QuantParams",
    "QuantSpec",
    "axis_view",
    "exact_round_trip_ok",
    "fake_dequantize",
    "fake_quantize",
    "params_shape",
    "quant_params",
    "quantization_error",
    "quantize_to_codes",
]

#: Bit widths the M2 quantizer accepts (frozen interface).
SUPPORTED_BITS: tuple[int, ...] = (2, 3, 4, 8)
#: Granularity strings the M2 quantizer accepts (frozen interface).
GRANULARITIES: tuple[str, ...] = ("per_tensor", "per_channel", "per_group")
#: Rounding modes the M2 quantizer accepts (frozen interface: nearest only).
ROUND_MODES: tuple[str, ...] = ("nearest",)


@dataclass(frozen=True)
class QuantSpec:
    """A validated uniform-quantization configuration.

    Args:
        bits: Target code width; one of :data:`SUPPORTED_BITS`.
        granularity: ``"per_tensor"``, ``"per_channel"`` or ``"per_group"``.
        group_size: Elements sharing one scale/zero-point along ``axis``. **Required** when
            ``granularity == "per_group"`` and **forbidden** otherwise. The frozen interface gives
            this field no default, so it is always passed explicitly.
        symmetric: ``True`` for signed codes with ``zero_point == 0``; ``False`` for unsigned codes
            with a computed zero-point.
        axis: Axis along which blocks are formed (negative values count from the end). Ignored by
            ``per_tensor``.
        round_mode: ``"nearest"`` only in M2 (ties-to-even).

    Raises:
        ValueError: On an invalid combination: unsupported bit width, unknown granularity or round
            mode, missing ``group_size`` for ``per_group``, ``group_size`` present for another
            granularity, non-positive ``group_size``, non-integer ``axis``, non-bool ``symmetric``.

    Dtypes / device:
        A pure, immutable configuration: it holds ints/strs only, is hashable (usable as a dict
        key), touches no tensor and has no device. Every function in this module and in
        :mod:`spectraquant.quantization.accounting` takes it by value.
    """

    bits: int
    granularity: str
    group_size: int | None
    symmetric: bool
    axis: int = 0
    round_mode: str = "nearest"

    def __post_init__(self) -> None:
        if isinstance(self.bits, bool) or not isinstance(self.bits, int):
            raise ValueError(f"bits must be an int, got {type(self.bits).__name__}")
        if self.bits not in SUPPORTED_BITS:
            raise ValueError(f"bits must be one of {SUPPORTED_BITS}, got {self.bits!r}")
        if self.granularity not in GRANULARITIES:
            raise ValueError(
                f"granularity must be one of {GRANULARITIES}, got {self.granularity!r}"
            )
        if self.round_mode not in ROUND_MODES:
            raise ValueError(f"round_mode must be one of {ROUND_MODES}, got {self.round_mode!r}")
        if not isinstance(self.symmetric, bool):
            raise ValueError(f"symmetric must be a bool, got {type(self.symmetric).__name__}")
        if isinstance(self.axis, bool) or not isinstance(self.axis, int):
            raise ValueError(f"axis must be an int, got {type(self.axis).__name__}")
        if self.granularity == "per_group":
            if isinstance(self.group_size, bool) or not isinstance(self.group_size, int):
                raise ValueError(
                    "group_size is required for per_group quantization and must be an int"
                )
            if self.group_size < 1:
                raise ValueError(f"group_size must be >= 1, got {self.group_size}")
        elif self.group_size is not None:
            raise ValueError(
                f"group_size must be None for granularity={self.granularity!r}, "
                f"got {self.group_size!r}"
            )

    @property
    def qmin(self) -> int:
        """Lowest representable code (inclusive)."""
        return -(1 << (self.bits - 1)) if self.symmetric else 0

    @property
    def qmax(self) -> int:
        """Highest representable code (inclusive)."""
        return (1 << (self.bits - 1)) - 1 if self.symmetric else (1 << self.bits) - 1

    @property
    def n_levels(self) -> int:
        """Number of distinct code values (``qmax - qmin + 1``)."""
        return self.qmax - self.qmin + 1

    def scale_dtype_bytes(self) -> int:
        """Bytes per stored scale: fp16 for ``per_group`` (2), fp32 otherwise (4).

        This is the project's accounting/packing convention, not a law of quantization: the
        declared width must accompany every claim (``memory-accounting.md`` section 3.2).
        """
        return 2 if self.granularity == "per_group" else 4

    def zero_point_bytes(self) -> int:
        """Bytes per stored zero-point: 0 when symmetric, 1 when asymmetric.

        Asymmetric zero-points are unsigned codes in ``[0, 2**bits - 1]`` stored as one unsigned
        byte — the int8-width convention of ``memory-accounting.md`` section 3.2; the declared width
        must accompany any claim.
        """
        return 0 if self.symmetric else 1

    def n_blocks_along_axis(self, axis_len: int) -> int:
        """Blocks along the quantized axis for a tensor whose ``axis`` length is ``axis_len``."""
        if self.granularity == "per_tensor":
            return 1
        if self.granularity == "per_channel":
            return axis_len
        assert self.group_size is not None  # guaranteed by __post_init__
        return math.ceil(axis_len / self.group_size)


@dataclass(frozen=True)
class QuantParams:
    """Per-block quantization parameters produced by :func:`quant_params`.

    Shapes:
        With the quantized tensor viewed as ``(outer, axis_len, inner)`` (the axis is the one the
        blocks run along), ``scales`` and ``zero_points`` have the *block* shape that broadcasts
        against that view after :func:`_expand_blocks`:

        * ``per_tensor``: ``(1, 1, 1)``;
        * ``per_channel``: ``(1, axis_len, 1)`` — one scale per channel index, shared across every
          other axis;
        * ``per_group``: ``(outer, ceil(axis_len / group_size), inner)`` — one scale per group per
          ``(outer, inner)`` index pair, which is the ``out * ceil(in / g)`` block count of
          ``memory-accounting.md`` section 2.

        ``zero_points`` has the same shape; its values are integer-valued floats in ``[qmin, qmax]``
        (all zeros when symmetric).

    Dtypes / device:
        Both tensors are float32 (float64 for float64 input) on the *same device* as the tensor
        they were computed from. Nothing here moves a tensor to CUDA.

    Fields:
        ``qmin`` / ``qmax`` are carried so that :func:`fake_dequantize` and the packer can work
        without re-deriving the code range from the spec.
    """

    scales: torch.Tensor
    zero_points: torch.Tensor
    qmin: int
    qmax: int


def _work_dtype(dtype: torch.dtype) -> torch.dtype:
    """Accumulate in float64 only when the input is float64; float32 otherwise."""
    return torch.float64 if dtype == torch.float64 else torch.float32


def axis_view(shape: tuple[int, ...], axis: int) -> tuple[int, int, int]:
    """Return ``(outer, axis_len, inner)`` such that ``shape == (outer, axis_len, inner)``.

    Shapes:
        ``shape``: a tensor shape (tuple of ints). ``axis``: the quantized axis, possibly negative.

    Assumptions / limitations:
        Pure arithmetic on the shape; no tensor is touched. Used by the accounting module so that
        the block count is derived from the quantizer's own axis convention rather than re-derived.
    """
    rank = len(shape)
    if rank == 0:
        raise ValueError("quantization requires a tensor with at least one dimension")
    axis_norm = axis + rank if axis < 0 else axis
    if not 0 <= axis_norm < rank:
        raise IndexError(f"axis {axis} is out of range for shape {tuple(shape)}")
    outer = math.prod(shape[:axis_norm])
    inner = math.prod(shape[axis_norm + 1 :])
    return outer, shape[axis_norm], inner


def _validate_quantizable(w: torch.Tensor) -> None:
    """Reject non-float, empty and 0-dim tensors (no block would have a finite range)."""
    if not w.is_floating_point():
        raise TypeError(f"fake quantization requires a float tensor, got {w.dtype}")
    if w.ndim == 0:
        raise ValueError("cannot quantize a 0-dim tensor; add a trailing dimension")
    if w.numel() == 0:
        raise ValueError("cannot quantize an empty tensor: no block has a finite range")


def params_shape(spec: QuantSpec, outer: int, axis_len: int, inner: int) -> tuple[int, int, int]:
    """Return the block shape of :class:`QuantParams` tensors for this view and spec.

    Shapes:
        ``(outer, axis_len, inner)`` is the canonical view of the quantized tensor (see
        :func:`axis_view`). Returns ``(1, 1, 1)`` for ``per_tensor``, ``(1, axis_len, 1)`` for
        ``per_channel`` and ``(outer, ceil(axis_len / group_size), inner)`` for ``per_group``.

    Assumptions / limitations:
        Pure arithmetic. ``math.prod(params_shape(...))`` is the ``n_blocks`` total of
        ``docs/protocols/memory-accounting.md`` section 2 — the accounting module derives its block
        count from this function so the byte model cannot drift from the quantizer.
    """
    if spec.granularity == "per_tensor":
        return (1, 1, 1)
    if spec.granularity == "per_channel":
        return (1, axis_len, 1)
    return (outer, spec.n_blocks_along_axis(axis_len), inner)


def _expand_blocks(
    values: torch.Tensor, outer: int, axis_len: int, inner: int, spec: QuantSpec
) -> torch.Tensor:
    """Broadcast block-shaped ``values`` to the element view ``(outer, axis_len, inner)``.

    ``per_tensor``/``per_channel`` parameters broadcast directly (their collapsed dimensions are
    size 1); ``per_group`` parameters are gathered with the block index ``a // group_size``.

    Raises:
        ValueError: If ``values`` does not have exactly the block shape implied by
            ``spec``/``(outer, axis_len, inner)`` — this catches a ``QuantParams`` built for a
            different shape or granularity.
    """
    expected = params_shape(spec, outer, axis_len, inner)
    if tuple(values.shape) != expected:
        raise ValueError(
            f"expected per-block parameters of shape {expected} for granularity="
            f"{spec.granularity!r} and view {(outer, axis_len, inner)}, got "
            f"{tuple(values.shape)}"
        )
    if spec.granularity == "per_group":
        assert spec.group_size is not None  # guaranteed by __post_init__
        index = torch.arange(axis_len, device=values.device) // spec.group_size
        return values.index_select(1, index)
    return values.expand(outer, axis_len, inner)


def _block_scale_and_zero_point(
    vmin: torch.Tensor, vmax: torch.Tensor, qmin: int, qmax: int, symmetric: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn per-block min/max into ``(scale, zero_point)`` for either convention.

    Symmetric blocks use ``scale = max|w| / qmax`` and ``zero_point = 0``; because no element exceeds
    ``max|w|``, symmetric quantization **never clips** and the round-off bound ``scale / 2`` holds for
    every element. A zero block gets ``scale = 1.0`` and every code is 0, so it round-trips exactly
    instead of producing NaN.

    Asymmetric blocks first widen the observed range to **include zero**
    (``lo = min(min(w), 0)``, ``hi = max(max(w), 0)`` — the convention of ONNX Runtime's
    ``DynamicQuantizeLinear``), then use ``scale = (hi - lo) / (qmax - qmin)`` and
    ``zero_point = clip(qmin - round(lo / scale), qmin, qmax)``. Without the zero inclusion a block
    whose values sit far from the origin (e.g. ``w in [99, 100]``) would demand a zero-point outside
    ``[qmin, qmax]`` and the clamp would destroy the mapping; with it, the clip is inactive except
    for floating-point noise and zero is represented exactly.

    The final guard is against **underflow**, not just a zero range: a block of subnormal values can
    have ``spread > 0`` while ``spread / (qmax - qmin)`` still rounds to 0 in float32, which would
    make the codes ``inf``/``NaN``. Every block whose computed scale is not strictly positive falls
    back to ``scale = 1.0`` (and ``zero_point = 0`` when asymmetric), which rounds such values to
    code 0 — a bounded error of at most the block's own magnitude.
    """
    if symmetric:
        absmax = torch.maximum(vmin.abs(), vmax.abs())
        scale = torch.where(absmax > 0, absmax / qmax, torch.ones_like(absmax))
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        return scale, torch.zeros_like(scale)

    zero = torch.zeros_like(vmin)
    low = torch.minimum(vmin, zero)
    high = torch.maximum(vmax, zero)
    spread = high - low
    scale = torch.where(spread > 0, spread / (qmax - qmin), torch.ones_like(spread))
    usable = scale > 0
    scale = torch.where(usable, scale, torch.ones_like(scale))
    zero_point = torch.where(
        usable,
        torch.clamp(torch.round(qmin - low / scale), qmin, qmax),
        zero,
    )
    return scale, zero_point


def quant_params(w: torch.Tensor, spec: QuantSpec) -> QuantParams:
    """Compute per-block scales and zero-points for ``w`` under ``spec``.

    Shapes:
        ``w``: float tensor of rank >= 1, viewed as ``(outer, axis_len, inner)`` around
        ``spec.axis``. ``QuantParams.scales``/``.zero_points`` take the block shape documented on
        :class:`QuantParams`; their element count is exactly the ``n_blocks`` total of
        ``memory-accounting.md`` section 2.

    Dtypes / device:
        Statistics accumulate in float32 on the input's device (float64 for float64 input). The
        returned tensors are on that same device and are never placed on CUDA.

    Assumptions / limitations:
        * Zero-range blocks yield ``scale = 1.0`` (never 0 or NaN); when asymmetric, the zero-point
          is chosen to reproduce the constant exactly.
        * An outlier sets its block's scale and therefore degrades resolution for every other
          element of that block; clipping is not compensated.
        * Materialises a tensor of ``w``'s size while forming statistics (see module docstring).

    Raises:
        TypeError: If ``w`` is not a float tensor.
        ValueError: If ``w`` is empty, 0-dim, or the spec is inconsistent with itself.
        IndexError: If ``spec.axis`` is out of range for ``w``.
    """
    _validate_quantizable(w)
    outer, axis_len, inner = axis_view(tuple(w.shape), spec.axis)
    work = w.detach().to(_work_dtype(w.dtype)).reshape(outer, axis_len, inner)

    if spec.granularity == "per_tensor":
        vmin = work.amin().reshape(1, 1, 1)
        vmax = work.amax().reshape(1, 1, 1)
    elif spec.granularity == "per_channel":
        vmin = work.amin(dim=(0, 2)).reshape(1, axis_len, 1)
        vmax = work.amax(dim=(0, 2)).reshape(1, axis_len, 1)
    else:
        assert spec.group_size is not None  # guaranteed by __post_init__
        group = spec.group_size
        blocks = spec.n_blocks_along_axis(axis_len)
        padded_len = blocks * group
        if padded_len != axis_len:
            # Replicate the last element: it already belongs to the trailing short group, so that
            # group's min/max is unchanged by the padding.
            index = torch.arange(padded_len, device=work.device).clamp(max=axis_len - 1)
            work = work.index_select(1, index)
        grouped = work.reshape(outer, blocks, group, inner)
        # Reduce over the group axis only: every (outer, inner) row keeps its own blocks.
        vmin = grouped.amin(dim=2)
        vmax = grouped.amax(dim=2)

    scale, zero_point = _block_scale_and_zero_point(
        vmin, vmax, spec.qmin, spec.qmax, spec.symmetric
    )
    return QuantParams(scales=scale, zero_points=zero_point, qmin=spec.qmin, qmax=spec.qmax)


def quantize_to_codes(w: torch.Tensor, params: QuantParams, spec: QuantSpec) -> torch.Tensor:
    """Return the integer codes for ``w`` as a float tensor shaped like ``w``.

    Shapes:
        ``w``: float tensor, rank >= 1. Returns the same shape; values are whole numbers in
        ``[params.qmin, params.qmax]``.

    Dtypes / device:
        float32 (float64 for float64 input), on ``w``'s device. Codes are floats so the
        straight-through estimator and dequantization stay graph-friendly; the packer casts them to
        integers.

    Assumptions:
        ``params`` must come from :func:`quant_params` for the same ``spec`` and shape.

    Raises:
        ValueError: If the parameter shape does not match ``spec``/``w.shape``.
    """
    _validate_quantizable(w)
    outer, axis_len, inner = axis_view(tuple(w.shape), spec.axis)
    dtype = _work_dtype(w.dtype)
    scale = _expand_blocks(params.scales.to(dtype), outer, axis_len, inner, spec)
    zero_point = _expand_blocks(params.zero_points.to(dtype), outer, axis_len, inner, spec)
    work = w.detach().to(dtype).reshape(outer, axis_len, inner)
    codes = torch.round(work / scale) + zero_point
    return torch.clamp(codes, float(params.qmin), float(params.qmax)).reshape(w.shape)


def fake_dequantize(q: torch.Tensor, params: QuantParams, spec: QuantSpec) -> torch.Tensor:
    """Reconstruct float values from codes: ``w_hat = (q - zero_point) * scale``.

    Shapes:
        ``q``: integer-valued float (or integer) tensor, same shape as the original weight. Returns
        a float tensor with that shape.

    Dtypes / device:
        float32, or float64 when ``q`` is float64; on ``q``'s device.

    Assumptions / limitations:
        ``params`` must match ``spec`` and ``q.shape`` (enforced by :func:`_expand_blocks`).
        Dequantization is exact arithmetic on the codes: it adds no rounding beyond the float32
        (or float64) multiply, so a code grid that is exactly representable round-trips bit-for-bit.
    """
    outer, axis_len, inner = axis_view(tuple(q.shape), spec.axis)
    dtype = _work_dtype(q.dtype)
    scale = _expand_blocks(params.scales.to(dtype), outer, axis_len, inner, spec)
    zero_point = _expand_blocks(params.zero_points.to(dtype), outer, axis_len, inner, spec)
    codes = q.to(dtype).reshape(outer, axis_len, inner)
    return ((codes - zero_point) * scale).reshape(q.shape)


def fake_quantize(
    w: torch.Tensor, spec: QuantSpec, params: QuantParams | None = None
) -> torch.Tensor:
    """Quantize-then-dequantize ``w`` in float, with an identity straight-through gradient.

    Shapes:
        ``w``: float tensor, rank >= 1. Returns a float tensor of the same shape.

    Dtypes / device:
        Same dtype and device as ``w`` (codes are computed internally in float32/float64).

    Gradients:
        ``d out / d w = 1`` (straight-through estimator): the forward value is the dequantized
        tensor, while the rounding step's local gradient is replaced by identity so
        quantization-aware training can backpropagate through this op.

    Assumptions:
        The result is **float** storage: measurement class 2 (quality simulation). It MUST NOT be
        reported as low-bit storage or as accelerated inference (``AGENTS.md`` section 4.3).
        ``params`` may be passed to reuse calibration statistics; otherwise they are derived from
        ``w`` itself.

    Raises:
        TypeError / ValueError / IndexError: See :func:`quant_params`.
    """
    with torch.no_grad():
        resolved = quant_params(w, spec) if params is None else params
        codes = quantize_to_codes(w, resolved, spec)
        dequantized = fake_dequantize(codes, resolved, spec)
    adjusted = dequantized.to(dtype=w.dtype)
    # The forward value equals `dequantized` (the added term is exactly zero); the backward pass
    # sends the incoming gradient straight to `w`.
    return adjusted + (w - w.detach())


def exact_round_trip_ok(w: torch.Tensor, spec: QuantSpec) -> bool:
    """Return ``True`` iff ``dequantize(quantize(w))`` reproduces ``w`` bit-for-bit.

    Shapes / dtypes / device:
        ``w``: float tensor, rank >= 1; any float dtype; any device. The comparison is exact
        (``torch.equal``), never tolerance-based.

    Assumptions / limitations:
        * ``True`` requires every element to already lie on its block's code grid *and* to survive
          the float32 round trip ``round(w / scale) * scale``; build grid-aligned inputs from integer
          codes multiplied by a power-of-two scale to observe ``True``.
        * An outlier does not by itself prevent ``True``: it changes the block scale, and the check
          is about the grid, not about the value distribution.
    """
    params = quant_params(w, spec)
    codes = quantize_to_codes(w, params, spec)
    round_tripped = fake_dequantize(codes, params, spec).to(dtype=w.dtype)
    return bool(torch.equal(round_tripped, w))


def quantization_error(
    reference: torch.Tensor, reconstructed: torch.Tensor, *, norm: str = "fro"
) -> dict[str, float]:
    """Error metrics between a float tensor and its quantized reconstruction.

    Shapes:
        ``reference`` / ``reconstructed``: same-shape float tensors. Returns a dict with ``max_abs``,
        ``mean_abs``, ``rmse``, ``relative_fro`` and ``reference_absmax``.

    Dtypes / device:
        Computed in float64 on the input's device; the returned values are Python floats.

    Assumptions / limitations:
        Absolute error is **scale-dependent** (``memory-accounting.md`` section 7.5.1): the same
        quantizer on the same distribution returns a proportionally different ``max_abs`` when the
        weight scale changes, while ``relative_fro`` is invariant. Publish ``relative_fro`` together
        with ``reference_absmax`` and the fixture distribution — never a bare ``max_abs``.

    Raises:
        ValueError: If ``norm`` is not ``"fro"`` or the shapes differ.
    """
    if norm != "fro":
        raise ValueError(f"norm must be 'fro', got {norm!r}")
    if reference.shape != reconstructed.shape:
        raise ValueError(
            f"shape mismatch: reference {tuple(reference.shape)} vs "
            f"reconstructed {tuple(reconstructed.shape)}"
        )
    ref = reference.detach().to(torch.float64)
    diff = (ref - reconstructed.detach().to(torch.float64)).abs()
    reference_norm = float(ref.norm())
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rmse": float(diff.pow(2).mean().sqrt()),
        "relative_fro": float(diff.norm()) / reference_norm if reference_norm > 0 else 0.0,
        "reference_absmax": float(ref.abs().max()),
    }
