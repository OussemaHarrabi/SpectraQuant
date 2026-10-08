"""Byte accounting: the memory-accounting worked examples, and measured-vs-accounted reconciliation.

Every number asserted here comes from ``docs/protocols/memory-accounting.md`` section 7, which is
normative and machine-checked. Class 1 assertions (formulas) and class 3 assertions (real
serialization) are kept in separate tests and never mixed inside one claim.
"""

from __future__ import annotations

import pytest
import torch

from spectraquant.quantization import (
    QuantSpec,
    accounted_bytes,
    analytical_bits_per_param,
    byte_breakdown,
    deserialize_state,
    fake_dequantize,
    fake_quantize,
    measure_serialized_bytes,
    n_blocks,
    quant_params,
    quantize_to_codes,
    serialize_state,
    serialized_size_bytes,
    theoretical_bits,
    write_serialized_state,
)


def _state(shape: tuple[int, ...], *, seed: int, sigma: float = 1.0) -> torch.Tensor:
    """Standard-normal fixture with a stated shape/sigma (memory-accounting section 8.8)."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator) * sigma


# --------------------------------------------------------------------------------------
# Section 7.1 — tiny tensor, out = 4, in = 8, N = 32
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "total_bytes", "bits_per_param"),
    [
        (QuantSpec(8, "per_tensor", None, True), 36, 9.0),
        (QuantSpec(8, "per_group", 4, True, axis=1), 48, 12.0),
        (QuantSpec(8, "per_group", 8, True, axis=1), 40, 10.0),
        (QuantSpec(8, "per_group", 8, False, axis=1), 44, 11.0),
        (QuantSpec(4, "per_tensor", None, True), 20, 5.0),
        (QuantSpec(4, "per_group", 4, True, axis=1), 32, 8.0),
        (QuantSpec(4, "per_group", 8, True, axis=1), 24, 6.0),
    ],
)
def test_worked_example_7_1_rows(spec: QuantSpec, total_bytes: int, bits_per_param: float) -> None:
    breakdown = byte_breakdown((4, 8), spec)
    assert breakdown.total_bytes == total_bytes
    assert breakdown.bits_per_param == pytest.approx(bits_per_param, abs=1e-12)
    assert breakdown.payload_bytes == 32 * spec.bits // 8


def test_worked_example_7_1_row_8_differs_by_the_zero_point_convention() -> None:
    """The doc's 26 B row models 4-bit zero-points (``z_b = 0.5``); our packer stores int8.

    ``memory-accounting.md`` section 3.2 declares both conventions legitimate and says the zero-point
    width "is part of the claim". Our container writes int8 zero-points (the ORT dynamic-int8
    convention), so the same configuration costs 28 B here, not 26 B. The difference is a
    *convention*, not an arithmetic disagreement: recomputing the doc's row with ``z_b = 0.5`` from
    our own primitives reproduces 26 B exactly.
    """
    spec = QuantSpec(4, "per_group", 8, False, axis=1)
    breakdown = byte_breakdown((4, 8), spec)
    assert breakdown.zero_points_bytes == breakdown.n_blocks * 1  # int8, our convention
    assert breakdown.total_bytes == 28
    doc_total = breakdown.payload_bytes_padded + breakdown.scales_bytes + breakdown.n_blocks * 0.5
    assert doc_total == 26.0
    assert QuantSpec(4, "per_group", 8, False, axis=1).zero_point_bytes() == 1


def test_worked_example_7_1b_row_alignment() -> None:
    """in = 5, 4-bit, out = 4: 10 B unaligned -> 16 B aligned (+60 %)."""
    breakdown = byte_breakdown((4, 5), QuantSpec(4, "per_group", 5, True, axis=1))
    assert breakdown.payload_bytes == 10
    assert breakdown.payload_bytes_padded == 16
    assert breakdown.alignment_bytes == 6
    assert breakdown.scales_bytes == 4 * 2  # 4 rows x 1 group, fp16
    assert breakdown.total_bytes == 24


# --------------------------------------------------------------------------------------
# Section 7.2 — transformer projection, out = in = 4096
# --------------------------------------------------------------------------------------


def test_worked_example_7_2_int8_per_channel() -> None:
    breakdown = byte_breakdown((4096, 4096), QuantSpec(8, "per_channel", None, True, axis=0))
    assert breakdown.n_blocks == 4096
    assert breakdown.payload_bytes_padded == 16_777_216
    assert breakdown.scales_bytes == 16_384
    assert breakdown.zero_points_bytes == 0
    assert breakdown.total_bytes == 16_793_600
    assert breakdown.bits_per_param == pytest.approx(8.0078, abs=5e-5)


def test_worked_example_7_2_int4_group128_symmetric() -> None:
    breakdown = byte_breakdown((4096, 4096), QuantSpec(4, "per_group", 128, True, axis=1))
    assert breakdown.n_blocks == 131_072  # 4096 * ceil(4096/128)
    assert breakdown.payload_bytes_padded == 8_388_608
    assert breakdown.scales_bytes == 262_144
    assert breakdown.total_bytes == 8_650_752
    assert breakdown.bits_per_param == pytest.approx(4.1250, abs=5e-5)


def test_worked_example_7_2_int4_group128_zero_point_conventions() -> None:
    int8_zp = byte_breakdown((4096, 4096), QuantSpec(4, "per_group", 128, False, axis=1))
    assert int8_zp.zero_points_bytes == 131_072
    assert int8_zp.total_bytes == 8_781_824  # the doc's int8-zero-point row
    # The doc's 4-bit zero-point row, recomputed from our primitives with z_b = 0.5.
    assert (
        int8_zp.payload_bytes_padded + int8_zp.scales_bytes + int8_zp.n_blocks * 0.5 == 8_716_288.0
    )


def test_ideal_payload_matches_the_no_metadata_row_of_7_2() -> None:
    spec = QuantSpec(4, "per_group", 128, True, axis=1)
    assert theoretical_bits((4096, 4096), spec) == 4.0
    assert analytical_bits_per_param((4096, 4096), spec) == pytest.approx(4.1250, abs=5e-5)


# --------------------------------------------------------------------------------------
# Block-count arithmetic (sections 2 and 8.3)
# --------------------------------------------------------------------------------------


def test_n_blocks_matches_section_2_formula_for_every_granularity() -> None:
    assert n_blocks((4, 8), QuantSpec(8, "per_tensor", None, True)) == 1
    assert n_blocks((4, 8), QuantSpec(8, "per_channel", None, True, axis=0)) == 4
    assert n_blocks((4, 8), QuantSpec(8, "per_channel", None, True, axis=1)) == 8
    assert n_blocks((4, 8), QuantSpec(8, "per_group", 8, True, axis=1)) == 4
    assert n_blocks((4096, 4096), QuantSpec(4, "per_group", 128, True, axis=1)) == 4096 * 32


def test_trailing_short_group_costs_a_full_block() -> None:
    """A trailing group of one element still costs one scale (the ``ceil`` of section 2)."""
    divisible = byte_breakdown((4, 8), QuantSpec(4, "per_group", 4, True, axis=1))
    nondivisible = byte_breakdown((4, 5), QuantSpec(4, "per_group", 4, True, axis=1))
    assert divisible.n_blocks == 8  # 4 rows x 2 groups
    assert nondivisible.n_blocks == 8  # 4 rows x ceil(5/4) = 2 groups
    assert nondivisible.scales_bytes == divisible.scales_bytes


def test_group_metadata_is_counted_and_not_omitted() -> None:
    breakdown = byte_breakdown((4, 8), QuantSpec(4, "per_group", 4, False, axis=1))
    assert breakdown.group_metadata_bytes == breakdown.scales_bytes + breakdown.zero_points_bytes
    assert breakdown.group_metadata_bytes == breakdown.n_blocks * 3  # fp16 scale + int8 zero-point
    assert breakdown.total_bytes == (
        breakdown.payload_bytes_padded + breakdown.group_metadata_bytes
    )


def test_serialized_size_bytes_is_the_same_quantity_as_accounted_bytes() -> None:
    for spec in (
        QuantSpec(4, "per_tensor", None, True),
        QuantSpec(8, "per_channel", None, True, axis=0),
        QuantSpec(4, "per_group", 3, False, axis=1),
    ):
        assert serialized_size_bytes((7, 5), spec) == accounted_bytes((7, 5), spec)


@pytest.mark.parametrize(
    ("shape", "spec"),
    [
        ((4, 8), QuantSpec(4, "per_group", 3, True, axis=1)),
        ((4, 8), QuantSpec(8, "per_channel", None, False, axis=0)),
        ((4, 6), QuantSpec(4, "per_group", 2, True, axis=0)),
        ((3, 4, 5), QuantSpec(4, "per_group", 2, True, axis=-1)),
        ((3, 4, 5), QuantSpec(8, "per_channel", None, False, axis=1)),
        ((3, 4, 5), QuantSpec(3, "per_group", 5, True, axis=2)),
        ((2, 3, 4, 5), QuantSpec(2, "per_group", 7, False, axis=0)),
        ((2, 3, 4, 5), QuantSpec(4, "per_tensor", None, True)),
    ],
)
def test_accounted_blocks_equal_the_quantizer_parameter_count(
    shape: tuple[int, ...], spec: QuantSpec
) -> None:
    """Design-note invariant 1: the byte model's ``n_blocks`` is the quantizer's own parameter count.

    If these ever diverge, ``accounted_bytes`` would be charging scales the quantizer never
    produces (or omitting ones it does), which would silently corrupt every equal-memory comparison.
    """
    weight = _state(shape, seed=len(shape) * 7 + spec.bits)
    assert n_blocks(shape, spec) == quant_params(weight, spec).scales.numel()
    assert n_blocks(shape, spec) == byte_breakdown(shape, spec).n_blocks
    expected_parameter_bytes = n_blocks(shape, spec) * (
        spec.scale_dtype_bytes() + spec.zero_point_bytes()
    )
    assert byte_breakdown(shape, spec).group_metadata_bytes == expected_parameter_bytes


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_accounting_is_consistent_for_every_supported_width(bits: int) -> None:
    breakdown = byte_breakdown((4, 5), QuantSpec(bits, "per_group", 3, True, axis=1))
    assert breakdown.total_bytes == (
        breakdown.payload_bytes_padded + breakdown.scales_bytes + breakdown.zero_points_bytes
    )
    assert breakdown.bits_per_param == pytest.approx(
        8 * breakdown.total_bytes / breakdown.n_params, rel=1e-12
    )
    assert breakdown.alignment_bytes == breakdown.payload_bytes_padded - breakdown.payload_bytes
    assert 0 <= breakdown.alignment_bytes < 4 * 5  # at most one word per row


def test_theoretical_bits_is_a_lower_bound_on_the_analytical_figure() -> None:
    for spec in (
        QuantSpec(4, "per_tensor", None, True),
        QuantSpec(4, "per_group", 8, True, axis=1),
        QuantSpec(8, "per_channel", None, False, axis=0),
    ):
        assert theoretical_bits((6, 8), spec) <= analytical_bits_per_param((6, 8), spec)
    assert theoretical_bits((127, 8), QuantSpec(3, "per_tensor", None, True)) == pytest.approx(
        3.0, abs=0.01
    )


def test_accounting_rejects_invalid_shapes_and_axes() -> None:
    spec = QuantSpec(4, "per_tensor", None, True)
    with pytest.raises(ValueError, match="at least one dimension"):
        byte_breakdown((), spec)
    with pytest.raises(ValueError, match="strictly positive"):
        byte_breakdown((4, 0), spec)
    with pytest.raises(IndexError, match="out of range"):
        byte_breakdown((4, 8), QuantSpec(4, "per_channel", None, True, axis=3))


# --------------------------------------------------------------------------------------
# Section 8.7 — measured-vs-accounted reconciliation (class 3 vs class 1)
# --------------------------------------------------------------------------------------


RECONCILIATION_CASES = [
    ("int8 per-channel", (64, 32), QuantSpec(8, "per_channel", None, True, axis=0)),
    ("int4 per-group symmetric", (128, 128), QuantSpec(4, "per_group", 32, True, axis=1)),
    ("int4 per-group asymmetric", (128, 128), QuantSpec(4, "per_group", 32, False, axis=1)),
    ("int4 per-group non-divisible groups", (64, 35), QuantSpec(4, "per_group", 16, True, axis=1)),
    ("int8 odd row length", (8, 7), QuantSpec(8, "per_group", 3, False, axis=1)),
    ("int2 per-tensor", (16, 16), QuantSpec(2, "per_tensor", None, True)),
    ("int3 per-channel", (9, 5), QuantSpec(3, "per_channel", None, True, axis=1)),
]


@pytest.mark.parametrize(
    ("label", "shape", "spec"),
    RECONCILIATION_CASES,
    ids=[case[0] for case in RECONCILIATION_CASES],
)
def test_measured_serialized_bytes_equal_accounted_bytes(
    label: str, shape: tuple[int, ...], spec: QuantSpec, capsys: pytest.CaptureFixture[str]
) -> None:
    """Class 3 measured bytes must equal the class 1 model, exactly (no header, 4-byte row words).

    The reconciliation rule of ``memory-accounting.md`` section 6.1 requires that any container
    overhead be measured and reported; our container is headerless and its only alignment is the
    per-row 32-bit word padding that ``byte_breakdown`` already counts, so the residual is 0 and is
    asserted to be 0 rather than absorbed into a tolerance.
    """
    tensor = _state(shape, seed=sum(label.encode()) % 997)
    measured = measure_serialized_bytes({"w": tensor}, {"w": spec})
    accounted = accounted_bytes(shape, spec)
    with capsys.disabled():
        print(
            f"reconciliation {label:38s} shape={shape!s:10s} measured={measured:7d} B "
            f"accounted={accounted:7d} B residual={measured - accounted} B"
        )
    assert measured == accounted
    assert measured == serialized_size_bytes(shape, spec)
    assert measured == byte_breakdown(shape, spec).total_bytes


def test_measured_bytes_equal_the_sum_over_a_multi_tensor_state() -> None:
    state = {"a": _state((4, 8), seed=1), "b": _state((16, 16), seed=2)}
    specs = {
        "a": QuantSpec(4, "per_group", 4, True, axis=1),
        "b": QuantSpec(8, "per_channel", None, False, axis=0),
    }
    expected = sum(accounted_bytes(tuple(t.shape), specs[name]) for name, t in state.items())
    assert measure_serialized_bytes(state, specs) == expected


def test_container_written_to_disk_has_the_predicted_size(tmp_path) -> None:
    state = {"w": _state((4, 5), seed=3)}
    specs = {"w": QuantSpec(4, "per_group", 4, False, axis=1)}
    path = write_serialized_state(state, specs, tmp_path)
    assert path.stat().st_size == accounted_bytes((4, 5), specs["w"])
    assert path.name == "weights.sqpack"
    assert len(path.read_bytes()) == path.stat().st_size


# --------------------------------------------------------------------------------------
# Serialize / deserialize semantics
# --------------------------------------------------------------------------------------


def test_serialize_is_deterministic_and_key_order_independent() -> None:
    first = {"a": _state((4, 8), seed=4), "b": _state((3, 3), seed=5)}
    specs = {
        "a": QuantSpec(4, "per_group", 4, True, axis=1),
        "b": QuantSpec(8, "per_tensor", None, True),
    }
    reordered = {"b": first["b"], "a": first["a"]}
    assert serialize_state(first, specs) == serialize_state(reordered, specs)


def test_deserialize_recovers_fp32_scale_configurations_bit_exactly() -> None:
    w = _state((6, 8), seed=6)
    spec = QuantSpec(8, "per_channel", None, False, axis=0)  # fp32 scales
    blob = serialize_state({"w": w}, {"w": spec})
    restored = deserialize_state(blob, {"w": (6, 8)}, {"w": spec})["w"]
    in_memory = fake_quantize(w, spec)
    assert torch.equal(restored, in_memory)


def test_deserialize_of_per_group_differs_only_by_the_fp16_scale_rounding() -> None:
    """Per-group scales are stored fp16 (the accounting convention): bound the load-back drift."""
    w = _state((8, 32), seed=7)
    spec = QuantSpec(4, "per_group", 8, True, axis=1)
    blob = serialize_state({"w": w}, {"w": spec})
    restored = deserialize_state(blob, {"w": (8, 32)}, {"w": spec})["w"]
    in_memory = fake_quantize(w, spec)
    drift = (restored - in_memory).abs()
    # Each element's drift is at most |q - zp| * scale * 2**-11 (fp16 relative rounding of the scale).
    params = quant_params(w, spec)
    codes = quantize_to_codes(w, params, spec)
    per_element = fake_dequantize((codes.abs() + 1), params, spec)
    assert float(drift.max()) <= float(per_element.max()) * 2**-11 * 1.01


def test_deserialize_round_trips_codes_exactly_for_every_width() -> None:
    w = _state((5, 7), seed=8)
    for bits in (2, 3, 4, 8):
        spec = QuantSpec(bits, "per_group", 3, False, axis=1)
        blob = serialize_state({"w": w}, {"w": spec})
        restored = deserialize_state(blob, {"w": (5, 7)}, {"w": spec})["w"]
        codes = quantize_to_codes(w, quant_params(w, spec), spec)
        expected = fake_dequantize(codes, quant_params(w, spec), spec)
        # fp16 scale storage bounds the drift; the code recovery itself is exact (checked in
        # tests/unit/test_quant_packing.py).
        assert torch.allclose(restored, expected, rtol=2**-10, atol=1e-6)


def test_deserialize_and_measure_reject_malformed_containers() -> None:
    w = _state((4, 8), seed=9)
    spec = QuantSpec(4, "per_group", 4, True, axis=1)
    blob = serialize_state({"w": w}, {"w": spec})
    with pytest.raises(ValueError, match="truncated"):
        deserialize_state(blob[:-1], {"w": (4, 8)}, {"w": spec})
    with pytest.raises(ValueError, match="trailing bytes"):
        deserialize_state(blob + b"\x00", {"w": (4, 8)}, {"w": spec})
    with pytest.raises(ValueError, match="keys differ"):
        serialize_state({"w": w}, {"other": spec})
    with pytest.raises(ValueError, match="keys differ"):
        deserialize_state(blob, {"w": (4, 8)}, {"other": spec})
    with pytest.raises(TypeError, match="float tensor"):
        serialize_state({"w": torch.zeros(4, dtype=torch.int64)}, {"w": spec})
    with pytest.raises(ValueError, match="rank >= 1"):
        serialize_state({"w": torch.tensor(1.0)}, {"w": spec})
    with pytest.raises(ValueError, match="container truncated"):
        deserialize_state(blob, {"w": (4, 64)}, {"w": spec})


def test_empty_state_is_zero_bytes() -> None:
    assert serialize_state({}, {}) == b""
    assert measure_serialized_bytes({}, {}) == 0
    assert deserialize_state(b"", {}, {}) == {}
