"""Fake-quantization core: spec validation, numerics, straight-through gradient, edge cases.

Fixture distributions are stated wherever an error figure is asserted
(``docs/protocols/memory-accounting.md`` section 8.8: a bare ``max_abs`` assertion is invalid).
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from spectraquant.quantization import (
    QuantParams,
    QuantSpec,
    exact_round_trip_ok,
    fake_dequantize,
    fake_quantize,
    quant_params,
    quantization_error,
    quantize_to_codes,
)


def _normal(shape: tuple[int, ...] = (6, 8), *, seed: int = 0, sigma: float = 1.0) -> torch.Tensor:
    """Standard normal fixture (``torch.randn`` with a fixed generator) scaled by ``sigma``."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator) * sigma


def _element_scale(w: torch.Tensor, spec: QuantSpec) -> torch.Tensor:
    """Per-element block scale, recovered through the public API.

    ``fake_dequantize(ones)`` equals ``(1 - 0) * scale``, i.e. the element's block scale.
    """
    params = quant_params(w, spec)
    zeros = torch.zeros_like(params.zero_points)
    scale_only = QuantParams(params.scales, zeros, 0, 1)
    return fake_dequantize(torch.ones_like(w), scale_only, spec)


# --------------------------------------------------------------------------------------
# QuantSpec validation
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("bits", [1, 0, 5, 6, 7, 16, -4])
def test_spec_rejects_unsupported_bits(bits: int) -> None:
    with pytest.raises(ValueError, match="bits must be one of"):
        QuantSpec(bits, "per_tensor", None, True)


def test_spec_rejects_non_int_bits() -> None:
    with pytest.raises(ValueError, match="bits must be an int"):
        QuantSpec(4.0, "per_tensor", None, True)  # type: ignore[arg-type]


@pytest.mark.parametrize("granularity", ["per_tensor", "per_channel", "per_group"])
def test_spec_accepts_documented_granularities(granularity: str) -> None:
    size = 8 if granularity == "per_group" else None
    spec = QuantSpec(4, granularity, size, True)
    assert spec.granularity == granularity


def test_spec_rejects_unknown_granularity() -> None:
    with pytest.raises(ValueError, match="granularity must be one of"):
        QuantSpec(4, "per_row", None, True)


def test_spec_requires_group_size_for_per_group() -> None:
    with pytest.raises(ValueError, match="group_size is required"):
        QuantSpec(4, "per_group", None, True)


def test_spec_rejects_non_positive_group_size() -> None:
    with pytest.raises(ValueError, match="group_size must be >= 1"):
        QuantSpec(4, "per_group", 0, True)


@pytest.mark.parametrize("granularity", ["per_tensor", "per_channel"])
def test_spec_forbids_group_size_outside_per_group(granularity: str) -> None:
    with pytest.raises(ValueError, match="group_size must be None"):
        QuantSpec(4, granularity, 8, True)


def test_spec_rejects_unsupported_round_mode() -> None:
    with pytest.raises(ValueError, match="round_mode must be one of"):
        QuantSpec(4, "per_tensor", None, True, round_mode="stochastic")


def test_spec_rejects_non_bool_symmetric() -> None:
    with pytest.raises(ValueError, match="symmetric must be a bool"):
        QuantSpec(4, "per_tensor", None, 1)  # type: ignore[arg-type]


def test_spec_rejects_non_int_axis() -> None:
    with pytest.raises(ValueError, match="axis must be an int"):
        QuantSpec(4, "per_channel", None, True, axis=1.5)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("bits", "symmetric", "qmin", "qmax", "n_levels"),
    [
        (2, True, -2, 1, 4),
        (2, False, 0, 3, 4),
        (3, True, -4, 3, 8),
        (4, True, -8, 7, 16),
        (4, False, 0, 15, 16),
        (8, True, -128, 127, 256),
        (8, False, 0, 255, 256),
    ],
)
def test_spec_code_ranges(bits: int, symmetric: bool, qmin: int, qmax: int, n_levels: int) -> None:
    spec = QuantSpec(bits, "per_tensor", None, symmetric)
    assert (spec.qmin, spec.qmax, spec.n_levels) == (qmin, qmax, n_levels)


def test_spec_is_frozen_and_hashable() -> None:
    spec = QuantSpec(4, "per_group", 8, True, axis=1)
    assert spec == QuantSpec(4, "per_group", 8, True, axis=1)
    assert len({spec, QuantSpec(4, "per_group", 8, True, axis=1)}) == 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.bits = 8  # type: ignore[misc]


@pytest.mark.parametrize(
    ("granularity", "group_size", "scale_bytes", "zero_bytes"),
    [
        ("per_tensor", None, 4, 0),
        ("per_channel", None, 4, 0),
        ("per_group", 8, 2, 0),
        ("per_group", 8, 2, 1),
    ],
)
def test_spec_declared_storage_widths(
    granularity: str, group_size: int | None, scale_bytes: int, zero_bytes: int
) -> None:
    spec = QuantSpec(4, granularity, group_size, symmetric=zero_bytes == 0)
    assert spec.scale_dtype_bytes() == scale_bytes
    assert spec.zero_point_bytes() == zero_bytes


# --------------------------------------------------------------------------------------
# Parameter shapes and the block layout
# --------------------------------------------------------------------------------------


def test_quant_params_shapes_per_granularity() -> None:
    w = _normal((6, 8))
    assert quant_params(w, QuantSpec(4, "per_tensor", None, True)).scales.shape == (1, 1, 1)
    # axis=0: one scale per row (6 rows), shared across the input axis.
    channel0 = quant_params(w, QuantSpec(4, "per_channel", None, True, axis=0))
    assert channel0.scales.shape == (1, 6, 1)
    # axis=1: one scale per column (8 columns), shared across the output axis.
    channel1 = quant_params(w, QuantSpec(4, "per_channel", None, True, axis=1))
    assert channel1.scales.shape == (1, 8, 1)
    # axis=1, group 4: 6 rows x ceil(8/4) = 2 groups per row.
    grouped = quant_params(w, QuantSpec(4, "per_group", 4, True, axis=1))
    assert grouped.scales.shape == (6, 2, 1)
    assert grouped.scales.numel() == 6 * ((8 + 3) // 4)


def test_per_channel_axis0_and_axis1_are_different_quantizers() -> None:
    w = _normal((6, 8), seed=3)
    spec0 = QuantSpec(4, "per_channel", None, True, axis=0)
    spec1 = QuantSpec(4, "per_channel", None, True, axis=1)
    error0 = (fake_quantize(w, spec0) - w).abs().max()
    error1 = (fake_quantize(w, spec1) - w).abs().max()
    assert not torch.equal(quant_params(w, spec0).scales, quant_params(w, spec1).scales)
    # Row-wise quantization of a matrix whose rows share a scale differs from column-wise.
    assert float(error0) != float(error1)


def test_group_count_for_non_divisible_and_oversized_groups() -> None:
    w = _normal((5, 7))
    # in = 7, g = 3 -> ceil(7/3) = 3 groups per row (the trailing group holds 1 element).
    three = quant_params(w, QuantSpec(4, "per_group", 3, True, axis=1))
    assert three.scales.shape == (5, 3, 1)
    # in = 7, g = 4 -> ceil(7/4) = 2 groups per row.
    four = quant_params(w, QuantSpec(4, "per_group", 4, True, axis=1))
    assert four.scales.shape == (5, 2, 1)
    # group larger than the axis -> every row is one block.
    huge = quant_params(w, QuantSpec(4, "per_group", 16, True, axis=1))
    assert huge.scales.shape == (5, 1, 1)


def test_non_divisible_group_keeps_the_half_step_bound() -> None:
    w = _normal((5, 7), seed=5)
    spec = QuantSpec(3, "per_group", 3, True, axis=1)
    error = (fake_quantize(w, spec) - w).abs()
    bound = _element_scale(w, spec) / 2
    assert torch.all(error <= bound * (1 + 1e-6) + 1e-7)


def test_trailing_short_group_uses_only_its_own_elements() -> None:
    """A trailing group holding one element gets a scale from that element, not from padding."""
    w = torch.zeros(1, 7)
    w[0, :6] = 4.0
    w[0, 6] = 0.25
    spec = QuantSpec(8, "per_group", 3, True, axis=1)
    scales = quant_params(w, spec).scales.reshape(-1)
    # groups: [0:3] = 4.0, [3:6] = 4.0, [6:7] = 0.25 -> 4/127, 4/127, 0.25/127.
    assert scales.shape == (3,)
    assert scales[0] == scales[1]
    assert float(scales[2]) == pytest.approx(0.25 / 127, rel=1e-6)


# --------------------------------------------------------------------------------------
# Numerics
# --------------------------------------------------------------------------------------


def test_per_tensor_symmetric_matches_the_manual_formula_exactly() -> None:
    w = _normal((4, 8), seed=7)
    spec = QuantSpec(4, "per_tensor", None, True)
    scale = w.abs().max() / 7.0  # float32 tensor scalar, as the implementation computes it
    expected = torch.clamp(torch.round(w / scale), -8, 7) * scale
    assert torch.equal(fake_quantize(w, spec), expected)
    assert float(quant_params(w, spec).scales) == float(scale)


def test_codes_are_integral_and_within_range_for_every_bit_width() -> None:
    w = _normal((7, 5), seed=11)
    for bits in (2, 3, 4, 8):
        for symmetric in (True, False):
            spec = QuantSpec(bits, "per_group", 3, symmetric, axis=1)
            params = quant_params(w, spec)
            codes = quantize_to_codes(w, params, spec)
            assert torch.equal(codes, codes.round())
            assert float(codes.min()) >= params.qmin
            assert float(codes.max()) <= params.qmax


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_max_error_is_bounded_by_half_a_step(bits: int) -> None:
    w = _normal((6, 8), seed=13)
    spec = QuantSpec(bits, "per_group", 5, True, axis=1)
    error = (fake_quantize(w, spec) - w).abs()
    half_step = _element_scale(w, spec) / 2
    assert torch.all(error <= half_step * (1 + 1e-6) + 1e-7)


def test_fake_quantize_preserves_shape_dtype_and_device() -> None:
    w = _normal((3, 4)).to(torch.float64)
    assert fake_quantize(w, QuantSpec(4, "per_channel", None, False, axis=1)).dtype == torch.float64
    w32 = _normal((3, 4))
    out = fake_quantize(w32, QuantSpec(4, "per_channel", None, False, axis=1))
    assert out.shape == w32.shape
    assert out.dtype == w32.dtype
    assert out.device == w32.device


def test_fake_dequantize_accepts_reused_params() -> None:
    calibration = _normal((6, 8), seed=2)
    other = _normal((6, 8), seed=99)
    spec = QuantSpec(4, "per_group", 4, True, axis=1)
    params = quant_params(calibration, spec)
    codes = quantize_to_codes(other, params, spec)
    assert torch.equal(fake_dequantize(codes, params, spec), fake_quantize(other, spec, params))


# --------------------------------------------------------------------------------------
# Straight-through estimator
# --------------------------------------------------------------------------------------


def test_ste_gradient_is_identity_on_a_leaf_tensor() -> None:
    w = _normal((4, 6), seed=17).requires_grad_(True)
    spec = QuantSpec(4, "per_group", 3, True, axis=1)
    out = fake_quantize(w, spec)
    assert out.requires_grad
    out.sum().backward()
    assert w.grad is not None
    assert torch.equal(w.grad, torch.ones_like(w))


def test_ste_gradient_equals_the_gradient_of_the_dequantized_value() -> None:
    """d loss / d w must be the identity, so it equals the analytic gradient at the dequantized w."""
    weight = _normal((5, 4), seed=23, sigma=0.5)
    leaf = weight.clone().requires_grad_(True)
    spec = QuantSpec(8, "per_channel", None, False, axis=1)
    dequantized = fake_quantize(leaf, spec)
    (dequantized.pow(2).sum() / 2).backward()
    assert leaf.grad is not None
    expected = fake_quantize(weight, spec)  # forward value == dequantized tensor
    assert torch.allclose(leaf.grad, expected, rtol=1e-6, atol=1e-7)


def test_qat_step_updates_weights_through_a_linear_layer() -> None:
    torch.manual_seed(0)
    layer = torch.nn.Linear(6, 3)
    spec = QuantSpec(4, "per_channel", None, True, axis=0)
    inputs = _normal((8, 6), seed=29)
    before = layer.weight.detach().clone()
    loss = fake_quantize(layer.weight, spec).pow(2).sum() + layer(inputs).pow(2).sum()
    loss.backward()
    assert layer.weight.grad is not None
    assert torch.isfinite(layer.weight.grad).all()
    assert torch.all(layer.weight.grad != 0)
    with torch.no_grad():
        layer.weight -= 0.01 * layer.weight.grad
    assert not torch.equal(before, layer.weight)


# --------------------------------------------------------------------------------------
# Edge cases: constant / zero-range / outlier / asymmetric clipping / grid values
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("constant", [0.0, 0.5, -3.25, 1e-8, -1e-8])
@pytest.mark.parametrize("symmetric", [True, False])
def test_constant_tensor_is_finite_and_round_trips_exactly(
    constant: float, symmetric: bool
) -> None:
    w = torch.full((4, 6), constant)
    spec = QuantSpec(4, "per_group", 3, symmetric, axis=1)
    params = quant_params(w, spec)
    assert torch.isfinite(params.scales).all()
    assert torch.isfinite(params.zero_points).all()
    assert bool((params.scales > 0).all())
    dequantized = fake_quantize(w, spec)
    assert torch.isfinite(dequantized).all()
    assert torch.equal(dequantized, w)
    assert exact_round_trip_ok(w, spec)


def test_all_zero_tensor_produces_zero_codes_and_zero_scales_are_guarded() -> None:
    w = torch.zeros(3, 5)
    for symmetric in (True, False):
        spec = QuantSpec(4, "per_tensor", None, symmetric)
        params = quant_params(w, spec)
        assert float(params.scales) == 1.0  # guarded, not 0 and not NaN
        assert torch.equal(quantize_to_codes(w, params, spec), torch.zeros(3, 5))
        assert torch.equal(fake_quantize(w, spec), w)


def test_constant_block_inside_a_varying_tensor_round_trips_exactly() -> None:
    w = _normal((4, 6), seed=31)
    w[:, 3] = -2.5  # one fully constant column, quantized as its own channel group
    spec = QuantSpec(8, "per_channel", None, True, axis=1)
    dequantized = fake_quantize(w, spec)
    assert torch.equal(dequantized[:, 3], w[:, 3])


def test_single_outlier_sets_the_block_scale_and_stays_within_half_a_step() -> None:
    w = _normal((1, 16), seed=37, sigma=0.05)
    w[0, 0] = 50.0  # single outlier, 1000x the rest
    spec = QuantSpec(4, "per_group", 16, True, axis=1)
    params = quant_params(w, spec)
    scale = float(params.scales)
    assert scale == pytest.approx(50.0 / 7, rel=1e-6)
    dequantized = fake_quantize(w, spec)
    error = (dequantized - w).abs()
    assert float(error.max()) <= scale / 2 * (1 + 1e-6)
    # The outlier itself is represented almost exactly; the small entries lose resolution.
    assert float(error[0, 0]) <= 1e-4
    assert float(error[0, 1:].max()) > 1e-4


@pytest.mark.parametrize("shift", [0.0, 99.0, -99.0, 1e-6, -1e-6])
def test_asymmetric_zero_point_clipping_keeps_the_mapping_usable(shift: float) -> None:
    """A block far from the origin must not overflow the zero-point clamp (ONNX convention)."""
    w = _normal((3, 8), seed=41, sigma=0.1) + shift
    spec = QuantSpec(8, "per_channel", None, False, axis=0)
    params = quant_params(w, spec)
    assert bool((params.zero_points >= params.qmin).all())
    assert bool((params.zero_points <= params.qmax).all())
    codes = quantize_to_codes(w, params, spec)
    assert float(codes.min()) >= 0
    assert float(codes.max()) <= 255
    error = (fake_quantize(w, spec) - w).abs()
    half_step = _element_scale(w, spec) / 2
    assert float(error.max()) <= float(half_step.max()) * (1 + 1e-6)


def test_asymmetric_represents_zero_exactly() -> None:
    w = _normal((4, 8), seed=43) + 7.0  # range far from zero
    spec = QuantSpec(8, "per_tensor", None, False)
    w[0, 0] = 0.0
    dequantized = fake_quantize(w, spec)
    assert float(dequantized[0, 0]) == 0.0
    assert float((dequantized - w).abs().max()) <= float(quant_params(w, spec).scales) / 2 * 1.0001


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_exact_round_trip_for_values_on_the_grid(bits: int) -> None:
    """Grid-aligned weights (integer codes times a power of two) round-trip bit-for-bit.

    The fixture is built so that ``max|codes| == qmax`` exactly: only then does the derived scale
    equal the construction scale and the codes reproduce. Codes stay inside the signed range.
    """
    qmax = (1 << (bits - 1)) - 1
    scale = 0.125  # power of two -> integer codes times it are exactly representable
    codes = torch.tensor(
        [
            [qmax, -qmax, 0, 1],
            [qmax - 1, -(qmax - 1), min(3, qmax), -min(2, qmax)],
        ],
        dtype=torch.float32,
    )
    assert float(codes.abs().max()) == float(qmax)
    w = codes * scale
    spec = QuantSpec(bits, "per_tensor", None, True)
    assert exact_round_trip_ok(w, spec)
    assert torch.equal(fake_quantize(w, spec), w)


def test_exact_round_trip_is_false_for_values_off_the_grid() -> None:
    w = _normal((4, 4), seed=47)
    assert not exact_round_trip_ok(w, QuantSpec(4, "per_tensor", None, True))


def test_subnormal_values_do_not_underflow_the_scale_into_nan() -> None:
    """A block of subnormal float32 values can have ``spread > 0`` while ``spread / qmax`` underflows.

    Found by a Hypothesis property run (``values=[0.0, 1.0, 2.0, 1.401298464324817e-45]``, i.e. the
    smallest float32 subnormal): for ``bits=2`` asymmetric the block scale ``2**-149 / 3`` rounds to
    zero, which produced NaN codes before the guard. Every configuration must now stay finite with a
    positive scale; the underflowing block falls back to ``scale = 1.0`` and rounds its subnormal
    element to code 0.
    """
    tiny = torch.tensor([[0.0, 1.0, 2.0, 1.401298464324817e-45]])
    for symmetric in (True, False):
        spec = QuantSpec(2, "per_group", 3, symmetric, axis=1)
        params = quant_params(tiny, spec)
        assert bool((params.scales > 0).all())
        assert bool(torch.isfinite(params.zero_points).all())
        codes = quantize_to_codes(tiny, params, spec)
        assert bool(torch.isfinite(codes).all())
        assert torch.equal(codes, codes.round())
        assert float(codes.min()) >= spec.qmin
        assert float(codes.max()) <= spec.qmax
        dequantized = fake_quantize(tiny, spec)
        assert bool(torch.isfinite(dequantized).all())
        error = (dequantized - tiny).abs()
        half_step = _element_scale(tiny, spec) / 2
        # One float32 ulp of slack: a code exactly half a step away can round a ulp over half.
        assert torch.all(error <= half_step * (1 + 1e-6))

    # The underflowing configuration specifically: scale floored to 1.0, subnormal -> code 0.
    asymmetric = QuantSpec(2, "per_group", 3, False, axis=1)
    assert float(quant_params(tiny, asymmetric).scales.reshape(-1)[1]) == 1.0
    assert float(fake_quantize(tiny, asymmetric)[0, 3]) == 0.0
    # The symmetric configuration at bits=2 has qmax = 1, so no division happens: the subnormal
    # block keeps its own value as the scale and round-trips exactly. (The first block does not:
    # with one code above zero the grid is {0, ±2}, so 1.0 rounds to 0.)
    symmetric_spec = QuantSpec(2, "per_group", 3, True, axis=1)
    assert float(quant_params(tiny, symmetric_spec).scales.reshape(-1)[1]) == 1.401298464324817e-45
    assert float(fake_quantize(tiny, symmetric_spec)[0, 3]) == 1.401298464324817e-45


def test_exact_round_trip_ok_is_true_for_a_constant_tensor() -> None:
    assert exact_round_trip_ok(torch.full((2, 2), 0.3), QuantSpec(3, "per_channel", None, False))


# --------------------------------------------------------------------------------------
# Error metrics
# --------------------------------------------------------------------------------------


def test_quantization_error_reports_scale_and_relative_error() -> None:
    w = _normal((6, 8), seed=53)
    spec = QuantSpec(4, "per_group", 4, True, axis=1)
    metrics = quantization_error(w, fake_quantize(w, spec))
    assert set(metrics) == {
        "max_abs",
        "mean_abs",
        "rmse",
        "relative_fro",
        "reference_absmax",
    }
    assert 0 < metrics["relative_fro"] < 1
    assert metrics["max_abs"] <= metrics["reference_absmax"]


def test_quantization_error_relative_fro_is_scale_invariant() -> None:
    w = _normal((6, 8), seed=59)
    spec = QuantSpec(4, "per_group", 4, True, axis=1)
    small = quantization_error(w, fake_quantize(w, spec))
    large = quantization_error(1000.0 * w, fake_quantize(1000.0 * w, spec))
    assert small["relative_fro"] == pytest.approx(large["relative_fro"], rel=1e-5)
    assert large["max_abs"] > 100.0 * small["max_abs"]  # absolute error is scale-dependent


def test_quantization_error_rejects_bad_norm_and_shape_mismatch() -> None:
    w = _normal((2, 2))
    with pytest.raises(ValueError, match="norm must be"):
        quantization_error(w, w, norm="l2")
    with pytest.raises(ValueError, match="shape mismatch"):
        quantization_error(w, _normal((3, 3)))


# --------------------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------------------


def test_quantize_rejects_non_float_tensor() -> None:
    with pytest.raises(TypeError, match="float tensor"):
        quant_params(torch.ones(2, 2, dtype=torch.int64), QuantSpec(4, "per_tensor", None, True))


def test_quantize_rejects_empty_and_zero_dim_tensors() -> None:
    spec = QuantSpec(4, "per_tensor", None, True)
    with pytest.raises(ValueError, match="empty tensor"):
        quant_params(torch.zeros(0, 3), spec)
    with pytest.raises(ValueError, match="0-dim"):
        quant_params(torch.zeros(()), spec)


def test_quantize_rejects_out_of_range_axis() -> None:
    with pytest.raises(IndexError, match="out of range"):
        quant_params(torch.zeros(2, 2), QuantSpec(4, "per_channel", None, True, axis=2))


def test_dequantize_rejects_params_from_a_different_shape() -> None:
    spec = QuantSpec(4, "per_group", 4, True, axis=1)
    params = quant_params(_normal((6, 8), seed=2), spec)
    with pytest.raises(ValueError, match="expected per-block parameters"):
        fake_dequantize(torch.zeros(6, 16), params, spec)


def test_negative_axis_matches_its_positive_equivalent() -> None:
    w = _normal((6, 8), seed=61)
    positive = QuantSpec(4, "per_group", 4, True, axis=1)
    negative = QuantSpec(4, "per_group", 4, True, axis=-1)
    assert torch.equal(fake_quantize(w, positive), fake_quantize(w, negative))
