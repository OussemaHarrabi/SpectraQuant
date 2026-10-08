"""Hypothesis properties of the quantizer: idempotence, half-step bound, error-vs-bits behaviour.

The bound properties are stated for the symmetric convention (which never clips) and hold exactly
for asymmetric blocks whose range includes zero. Where a property is only an empirical regularity,
the deterministic companion test states its fixture.
"""

from __future__ import annotations

import torch
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from spectraquant.quantization import (
    QuantParams,
    QuantSpec,
    accounted_bytes,
    exact_round_trip_ok,
    fake_dequantize,
    fake_quantize,
    measure_serialized_bytes,
    quant_params,
    quantize_to_codes,
)

_SHAPES = st.tuples(st.integers(1, 4), st.integers(1, 8))
_BITS = st.sampled_from((2, 3, 4, 8))
_FINITE = st.floats(
    min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False, width=32
)


def _element_scale(w: torch.Tensor, spec: QuantSpec) -> torch.Tensor:
    """Per-element block scale recovered through the public API (see the unit test module)."""
    params = quant_params(w, spec)
    scale_only = QuantParams(params.scales, torch.zeros_like(params.zero_points), 0, 1)
    return fake_dequantize(torch.ones_like(w), scale_only, spec)


def _tensor(shape: tuple[int, int], values: st.SearchStrategy[list[float]]) -> st.SearchStrategy:
    return st.builds(lambda vals: torch.tensor(vals, dtype=torch.float32).reshape(shape), values)


@given(
    bits=_BITS,
    scale_power=st.integers(min_value=-6, max_value=6),
    codes=st.lists(st.integers(min_value=-8, max_value=7), min_size=8, max_size=8),
)
@settings(max_examples=150, deadline=None)
def test_grid_aligned_tensors_round_trip_exactly(
    bits: int, scale_power: int, codes: list[int]
) -> None:
    """A tensor built as ``integer_codes * 2**k`` with ``max|codes| == qmax`` is exactly representable.

    The fixture forces one code to the extreme so the derived scale equals the construction scale;
    only then is the tensor on the quantizer's own grid.
    """
    qmax = (1 << (bits - 1)) - 1
    bounded = [max(-qmax, min(qmax, code)) for code in codes]
    bounded[0] = qmax
    assume(max(abs(code) for code in bounded) == qmax)
    scale = float(2.0**scale_power)
    w = torch.tensor(bounded, dtype=torch.float32).reshape(2, 4) * scale
    spec = QuantSpec(bits, "per_tensor", None, True)
    assert exact_round_trip_ok(w, spec)
    once = fake_quantize(w, spec)
    assert torch.equal(once, w)
    assert torch.equal(fake_quantize(once, spec), once)


@given(
    bits=_BITS,
    symmetric=st.booleans(),
    granularity=st.sampled_from(("per_tensor", "per_channel", "per_group")),
    group_size=st.integers(min_value=1, max_value=9),
    values=st.lists(_FINITE, min_size=4, max_size=32, unique=True),
)
@settings(max_examples=200, deadline=None)
def test_max_error_is_bounded_by_half_a_step(
    bits: int,
    symmetric: bool,
    granularity: str,
    group_size: int,
    values: list[float],
) -> None:
    """``|w - w_hat| <= block_scale / 2`` for every element, in every configuration."""
    w = torch.tensor(values, dtype=torch.float32).reshape(1, -1)
    if granularity == "per_group":
        spec = QuantSpec(bits, granularity, group_size, symmetric, axis=1)
    else:
        spec = QuantSpec(bits, granularity, None, symmetric, axis=1)
    error = (fake_quantize(w, spec) - w).abs()
    half_step = _element_scale(w, spec) / 2
    tolerance = 1e-6 * max(1.0, float(w.abs().max()))
    assert torch.all(error <= half_step + tolerance)


@given(
    bits=_BITS,
    values=st.lists(_FINITE, min_size=4, max_size=32, unique=True),
)
@settings(max_examples=150, deadline=None)
def test_codes_stay_inside_the_declared_range(bits: int, values: list[float]) -> None:
    w = torch.tensor(values, dtype=torch.float32).reshape(1, -1)
    for symmetric in (True, False):
        spec = QuantSpec(bits, "per_group", 3, symmetric, axis=1)
        codes = quantize_to_codes(w, quant_params(w, spec), spec)
        assert torch.equal(codes, codes.round())
        assert float(codes.min()) >= spec.qmin
        assert float(codes.max()) <= spec.qmax
        assert bool(torch.isfinite(codes).all())


@given(
    bits=_BITS,
    other=_BITS,
    values=st.lists(
        st.floats(min_value=0.5, max_value=8.0, allow_nan=False, width=32), min_size=4, max_size=16
    ),
)
@settings(max_examples=120, deadline=None)
def test_the_guaranteed_error_bound_shrinks_as_bits_grow(
    bits: int, other: int, values: list[float]
) -> None:
    """The worst-case error bound ``max|w| / (2 * qmax)`` decreases strictly with the bit width.

    Realized round-off is *not* monotone per element (a coarse grid can hit a value exactly while a
    finer one misses it by a few ulps), so the property asserted for arbitrary inputs is the bound;
    the realized mean error is checked on a fixed fixture below.
    """
    w = torch.tensor(values, dtype=torch.float32).reshape(1, -1)
    absmax = float(w.abs().max())
    bound = {b: absmax / (2 * ((1 << (b - 1)) - 1)) for b in (bits, other)}
    if bits != other:
        assert bound[min(bits, other)] > bound[max(bits, other)]
    spec = QuantSpec(bits, "per_tensor", None, True)
    realized = float((fake_quantize(w, spec) - w).abs().max())
    assert realized <= bound[bits] + 1e-6 * max(1.0, absmax)


def test_mean_error_decreases_with_bits_on_a_fixed_fixture() -> None:
    """Deterministic fixture: 1024 values uniform on [-1, 1] (``torch.linspace``), symmetric."""
    w = torch.linspace(-1.0, 1.0, 1024).reshape(1, -1)
    errors = []
    for bits in (2, 3, 4, 8):
        mean_error = float(
            (fake_quantize(w, QuantSpec(bits, "per_tensor", None, True)) - w).abs().mean()
        )
        errors.append(mean_error)
    assert errors[0] > errors[1] > errors[2] > errors[3]


@given(shape=_SHAPES, bits=_BITS, group_size=st.integers(min_value=1, max_value=8))
@settings(max_examples=80, deadline=None)
def test_accounted_bytes_equal_measured_bytes_for_random_shapes(
    shape: tuple[int, int], bits: int, group_size: int
) -> None:
    """Class-1 model vs class-3 measurement must agree for arbitrary shapes and widths."""
    generator = torch.Generator().manual_seed(shape[0] * 100 + shape[1] * 10 + bits)
    w = torch.randn(shape, generator=generator)
    spec = QuantSpec(bits, "per_group", group_size, False, axis=1)
    measured = measure_serialized_bytes({"w": w}, {"w": spec})
    assert measured == accounted_bytes(shape, spec)
