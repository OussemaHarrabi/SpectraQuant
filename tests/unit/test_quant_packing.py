"""Packing layout: 32-bit row padding, little-endian nibble order, odd counts, error handling."""

from __future__ import annotations

import pytest
import torch

from spectraquant.quantization import (
    axis_padding,
    pack_codes,
    pack_int4,
    pack_int8,
    row_bytes,
    row_code_capacity,
    unpack_codes,
    unpack_int4,
    unpack_int8,
)

ALL_BITS = (2, 3, 4, 8)


def _reference_pack(codes: torch.Tensor, bits: int) -> bytes:
    """Independent little-endian bit-contiguous packer built from Python ints.

    Written from ``docs/protocols/memory-accounting.md`` section 3.1 + the documented nibble order,
    not from the implementation, so agreeing with it is evidence the layout is what we claim.
    """
    rows = codes.reshape(-1, codes.shape[-1]).tolist()
    mask = (1 << bits) - 1
    out = bytearray()
    for row in rows:
        accumulator = 0
        for index, code in enumerate(row):
            accumulator |= (int(code) & mask) << (index * bits)
        out += accumulator.to_bytes(row_bytes(len(row), bits), "little")
    return bytes(out)


# --------------------------------------------------------------------------------------
# Layout arithmetic (must reproduce memory-accounting.md section 3.1)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("row_len", "bits", "expected"),
    [
        (5, 4, 4),  # section 7.1b: 5 int4 values -> 4 B
        (8, 4, 4),
        (9, 4, 8),
        (5, 8, 8),
        (4, 8, 4),
        (7, 8, 8),
        (7, 4, 4),
        (7, 2, 4),
        (32, 4, 16),
        (4096, 4, 2048),
        (4096, 8, 4096),
    ],
)
def test_row_bytes_matches_the_documented_word_alignment(
    row_len: int, bits: int, expected: int
) -> None:
    assert row_bytes(row_len, bits) == expected
    assert expected % 4 == 0


@pytest.mark.parametrize(
    ("row_len", "bits", "expected"),
    [(5, 4, 3), (8, 4, 0), (7, 4, 1), (7, 8, 1), (8, 8, 0), (7, 2, 9), (7, 3, 4)],
)
def test_axis_padding_is_the_code_capacity_minus_the_data(
    row_len: int, bits: int, expected: int
) -> None:
    assert axis_padding((4, row_len), bits) == expected
    assert row_code_capacity(row_len, bits) - row_len == expected


def test_row_bytes_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="bits must be an int"):
        row_bytes(8, 0)
    with pytest.raises(ValueError, match="row_len must be >= 1"):
        row_bytes(0, 4)


# --------------------------------------------------------------------------------------
# Layout semantics
# --------------------------------------------------------------------------------------


def test_int4_is_little_nibble_first() -> None:
    codes = torch.tensor([[1, -1, 7, -8]], dtype=torch.int64)
    packed = pack_int4(codes)
    assert packed.dtype == torch.uint8
    assert packed.tolist() == [[0xF1, 0x87, 0x00, 0x00]]  # rows fill whole 32-bit words
    assert unpack_int4(packed, (1, 4), 4).tolist() == codes.tolist()


def test_int4_int8_extremes_round_trip() -> None:
    extremes4 = torch.tensor([list(range(-8, 8))], dtype=torch.int64)
    assert unpack_int4(pack_int4(extremes4), (1, 16), 0).tolist() == extremes4.tolist()
    extremes8 = torch.tensor([[-128, -1, 0, 1]], dtype=torch.int64)
    assert pack_int8(extremes8).shape == (1, 4)
    assert unpack_int8(pack_int8(extremes8), (1, 4), 0).tolist() == extremes8.tolist()


def test_nibble_order_matches_the_independent_reference() -> None:
    codes = torch.tensor([[1, -1, 7, -8, 3, 0, -5, 2, 6]], dtype=torch.int64)
    packed = pack_int4(codes)
    assert packed.reshape(-1).numpy().tobytes() == _reference_pack(codes, 4)


@pytest.mark.parametrize("bits", ALL_BITS)
def test_packing_matches_the_independent_reference_for_every_width(bits: int) -> None:
    qmax = (1 << (bits - 1)) - 1
    codes = torch.arange(-qmax - 1, qmax + 1, dtype=torch.int64).reshape(2, -1)[:, :7]
    packed = pack_codes(codes, bits)
    assert packed.reshape(-1).numpy().tobytes() == _reference_pack(codes, bits)


@pytest.mark.parametrize("bits", ALL_BITS)
@pytest.mark.parametrize("row_len", [1, 2, 3, 5, 7, 8, 15, 16, 17])
def test_round_trip_is_exact_for_every_width_and_row_length(bits: int, row_len: int) -> None:
    qmin, qmax = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    generator = torch.Generator().manual_seed(row_len * 31 + bits)
    codes = torch.randint(qmin, qmax + 1, (3, row_len), generator=generator, dtype=torch.int64)
    packed = pack_codes(codes, bits)
    assert packed.shape == (3, row_bytes(row_len, bits))
    recovered = unpack_codes(packed, (3, row_len), bits, axis_padding((3, row_len), bits))
    assert recovered.dtype == torch.int8
    assert torch.equal(recovered.to(torch.int64), codes)


def test_odd_row_length_round_trips_and_padding_is_dropped() -> None:
    codes = torch.tensor([[3, -4, 5, -6, 7]], dtype=torch.int64)  # row_len = 5 -> 3 padding slots
    packed = pack_int4(codes)
    assert packed.shape == (1, 4)
    assert axis_padding((1, 5), 4) == 3
    assert unpack_int4(packed, (1, 5), 3).tolist() == codes.tolist()


def test_int8_row_padding_is_one_byte_for_row_len_seven() -> None:
    codes = torch.tensor([[-128, 0, 1, 2, 3, 4, 5]], dtype=torch.int64)
    packed = pack_int8(codes)
    assert packed.shape == (1, 8)
    assert torch.equal(unpack_int8(packed, (1, 7), 1).to(torch.int64), codes)


def test_three_bit_word_padding_is_bit_level_not_code_level() -> None:
    """7 three-bit codes need 21 bits -> one 32-bit word; the last padding slot is partial."""
    codes = torch.tensor([[3, -4, 3, -1, 0, 1, -2]], dtype=torch.int64)
    packed = pack_codes(codes, 3)
    assert packed.shape == (1, 4)
    assert row_code_capacity(7, 3) == 11
    assert torch.equal(
        unpack_codes(packed, (1, 7), 3, axis_padding((1, 7), 3)).to(torch.int64), codes
    )


def test_multi_dimensional_rows_are_packed_row_wise() -> None:
    codes = torch.randint(-8, 8, (2, 3, 5), generator=torch.Generator().manual_seed(5))
    packed = pack_int4(codes)
    assert packed.shape == (2, 3, 4)
    assert torch.equal(unpack_int4(packed, (2, 3, 5), 3).to(torch.int64), codes)
    assert packed.reshape(-1).numpy().tobytes() == _reference_pack(codes, 4)


def test_one_dimensional_tensor_is_a_single_row() -> None:
    codes = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)
    packed = pack_int4(codes)
    assert packed.shape == (4,)
    assert torch.equal(unpack_int4(packed, (5,), 3).to(torch.int64), codes)


# --------------------------------------------------------------------------------------
# Rejections
# --------------------------------------------------------------------------------------


def test_out_of_range_codes_are_rejected() -> None:
    with pytest.raises(ValueError, match="codes out of range"):
        pack_int4(torch.tensor([[8]], dtype=torch.int64))
    with pytest.raises(ValueError, match="codes out of range"):
        pack_int4(torch.tensor([[-9]], dtype=torch.int64))


def test_non_integer_codes_are_rejected() -> None:
    with pytest.raises(TypeError, match="integer tensor"):
        pack_int4(torch.zeros(2, dtype=torch.float32))


def test_zero_dim_and_empty_rows_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least one dimension"):
        pack_int4(torch.tensor(3, dtype=torch.int64))
    with pytest.raises(ValueError, match="zero codes"):
        pack_int4(torch.zeros(3, 0, dtype=torch.int64))


def test_unpack_rejects_wrong_padding_byte_count_and_row_count() -> None:
    packed = pack_int4(torch.zeros(2, 8, dtype=torch.int64))
    with pytest.raises(ValueError, match="axis_padding=1 does not match"):
        unpack_int4(packed, (2, 8), 1)
    with pytest.raises(ValueError, match="expected 4 bytes per row"):
        unpack_int4(packed[:, :2], (2, 8), 0)
    with pytest.raises(ValueError, match="row count"):
        unpack_int4(packed, (3, 8), 0)


def test_unpack_rejects_float_packed_input() -> None:
    with pytest.raises(TypeError, match="unsigned integer tensor"):
        unpack_int4(torch.zeros(2, 4), (2, 8), 0)


def test_unpack_rejects_empty_shape() -> None:
    with pytest.raises(ValueError, match="at least one dimension"):
        unpack_int4(torch.zeros(4, dtype=torch.uint8), (), 0)


def test_unsupported_bit_width_is_rejected_by_the_packer() -> None:
    with pytest.raises(ValueError, match="bits must be one of"):
        pack_codes(torch.zeros(4, dtype=torch.int64), 7)
    with pytest.raises(ValueError, match="bits must be one of"):
        unpack_codes(torch.zeros(4, dtype=torch.uint8), (8,), 5, 0)
