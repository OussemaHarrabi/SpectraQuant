"""Bit packing for quantized codes — measurement class 3 (real serialized bytes).

The routines here turn integer codes produced by :mod:`spectraquant.quantization.fake_quant` into
the exact byte stream our container writes, so that ``packed_bytes`` in a run manifest is a
*measured* number. The layout is defined once, here, and mirrored by
:func:`spectraquant.quantization.accounting.byte_breakdown`; the two are asserted equal for
2/3/4/8-bit codes by ``tests/unit/test_quant_packing.py``.

Layout (frozen)
---------------
* Codes are signed two's-complement integers in ``[-(2**(b-1)), 2**(b-1) - 1]``.
* Packing is **row-wise**: the last axis is one row and every preceding index combination is another
  row (``n_rows = numel / shape[-1]``).
* Within a row, codes are packed **little-endian bit-contiguously**: code ``i`` occupies bits
  ``[i*b, (i+1)*b)`` of the row's bit stream and bit ``j`` of that stream is bit ``j`` of byte
  ``j // 8`` (least significant bit first). For 4-bit codes this is the familiar
  **little-nibble-first** layout — code ``2i`` in the low nibble of byte ``i``, code ``2i+1`` in the
  high nibble — and for 8-bit codes it is one byte per code.
* Each row is padded with zero **bits** to a whole number of 32-bit words:
  ``row_bytes = ceil(row_len * b / 32) * 4``, exactly the ``bytes_per_row_padded`` of
  ``docs/protocols/memory-accounting.md`` section 3.1.

Consequences (documented, tested)
---------------------------------
* For the shipped widths (2, 4, 8 bits) the word padding is also a whole number of zero *codes*, so
  :func:`axis_padding` reports how many codes the padded row can hold beyond the real data and the
  unpack routine checks it (a misaligned read cannot happen). For 3-bit codes the word padding is
  bit-level and the last padded code slot is partial; :func:`axis_padding` then still reports
  ``capacity - row_len`` but the decoder reads the first ``row_len`` codes from the bit stream.
* Odd row lengths are legal: with 4-bit codes and ``row_len = 5`` the row becomes 4 bytes and
  ``axis_padding == 3`` (the ``+60 %`` overhead example of ``memory-accounting.md`` section 7.1b);
  with 8-bit codes and ``row_len = 7`` the row becomes 8 bytes and ``axis_padding == 1``.

Numerical / resource limitations
--------------------------------
* The 4-bit and 8-bit paths are direct and exact. The 2- and 3-bit paths materialise an
  ``(n_rows, row_len, bits)`` bit matrix — the reference implementation, adequate for the Tier-0
  fixtures in scope, not for multi-million-element tensors.
* Packing is pure torch and never moves a tensor to a CUDA device; a CUDA tensor is rejected, and no
  CUDA-only library is importable from this project (``AGENTS.md`` section 2).
"""

from __future__ import annotations

import torch

__all__ = [
    "INT4_MAX",
    "INT4_MIN",
    "INT8_MAX",
    "INT8_MIN",
    "SUPPORTED_PACK_BITS",
    "axis_padding",
    "pack_codes",
    "pack_int4",
    "pack_int8",
    "row_bytes",
    "row_code_capacity",
    "signed_codes_to_unsigned",
    "unpack_codes",
    "unpack_int4",
    "unpack_int8",
    "unsigned_codes_to_signed",
]

INT4_MIN, INT4_MAX = -8, 7
INT8_MIN, INT8_MAX = -128, 127

#: Code widths this packer serializes (the widths ``QuantSpec`` accepts).
SUPPORTED_PACK_BITS: tuple[int, ...] = (2, 3, 4, 8)


def _check_pack_bits(bits: int) -> None:
    """Reject code widths we do not claim to serialize."""
    if bits not in SUPPORTED_PACK_BITS:
        raise ValueError(f"bits must be one of {SUPPORTED_PACK_BITS}, got {bits!r}")


def row_bytes(row_len: int, bits: int) -> int:
    """Padded byte count of one packed row: ``ceil(row_len * bits / 32) * 4``.

    Args:
        row_len: Number of codes in the unpadded row.
        bits: Code width in bits.

    Returns:
        Bytes per row, always a positive multiple of 4 (the ``bytes_per_row_padded`` of
        ``memory-accounting.md`` section 3.1).

    Raises:
        ValueError: If ``bits`` is outside ``[1, 64]`` or ``row_len`` is not positive.
    """
    if not isinstance(bits, int) or isinstance(bits, bool) or not 1 <= bits <= 64:
        raise ValueError(f"bits must be an int in [1, 64], got {bits!r}")
    if row_len < 1:
        raise ValueError(f"row_len must be >= 1, got {row_len}")
    return -(-(row_len * bits) // 32) * 4


def row_code_capacity(row_len: int, bits: int) -> int:
    """Codes the padded row can hold: ``ceil(row_bytes * 8 / bits)``.

    Shapes / dtypes / device:
        Pure arithmetic on two ints; no tensor is touched.

    Assumptions / limitations:
        For the shipped widths (2, 4, 8) this is an exact whole-code capacity; for 3-bit codes the
        last slot is partial, because the row is padded with zero *bits* rather than whole codes
        (:func:`pack_codes` zeroes the tail of the bit stream). The decoder therefore reads the
        first ``row_len`` codes and ignores the tail.

    Raises:
        ValueError: See :func:`row_bytes`.
    """
    return -(-(row_bytes(row_len, bits) * 8) // bits)


def axis_padding(shape: tuple[int, ...], bits: int) -> int:
    """Number of padding code slots the packer adds to each row of ``shape``.

    Shapes:
        ``shape``: the *unpadded* tensor shape; only ``shape[-1]`` matters. Returns
        ``row_code_capacity(shape[-1], bits) - shape[-1]``, which is 0 when the row already fills
        whole 32-bit words.

    Raises:
        ValueError: If ``shape`` is empty, ``shape[-1] < 1`` or ``bits`` is unsupported.
    """
    if len(shape) == 0:
        raise ValueError("shape must have at least one dimension")
    return row_code_capacity(shape[-1], bits) - shape[-1]


def _validate_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Return ``codes`` as contiguous int64 after validating device, dtype, rank and value range."""
    if codes.is_cuda:  # pragma: no cover - no CUDA device exists on the target host
        raise ValueError("packing runs on CPU only: CUDA tensors are rejected by design")
    if codes.is_complex() or codes.is_floating_point():
        raise TypeError(f"codes must be an integer tensor, got {codes.dtype}")
    if codes.ndim == 0:
        raise ValueError("codes must have at least one dimension")
    if codes.shape[-1] == 0:
        raise ValueError("cannot pack a row with zero codes")
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    work = codes.detach().to(torch.int64).contiguous()
    if int(work.min()) < qmin or int(work.max()) > qmax:
        raise ValueError(
            f"codes out of range for {bits}-bit two's complement [{qmin}, {qmax}]: "
            f"observed [{int(work.min())}, {int(work.max())}]"
        )
    return work


def _bit_matrix_to_bytes(bit_matrix: torch.Tensor) -> torch.Tensor:
    """Pack a ``(rows, total_bits)`` 0/1 matrix into little-endian bytes ``(rows, total_bits/8)``."""
    n_rows, total_bits = bit_matrix.shape
    byte_weights = 1 << torch.arange(8, dtype=torch.int64, device=bit_matrix.device)
    grouped = bit_matrix.reshape(n_rows, total_bits // 8, 8)
    return (grouped * byte_weights).sum(dim=-1).to(torch.uint8)


def _bytes_to_bit_matrix(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_bit_matrix_to_bytes`: ``(rows, bytes)`` uint8 -> 0/1 bits."""
    n_rows, n_bytes = packed.shape
    shift_amounts = torch.arange(8, dtype=torch.int64, device=packed.device)
    bits = (packed.to(torch.int64).unsqueeze(-1) >> shift_amounts) & 1
    return bits.reshape(n_rows, n_bytes * 8)


def _pad_rows(rows: torch.Tensor, target_len: int) -> torch.Tensor:
    """Right-pad ``rows`` (second dim) with zeros to ``target_len`` codes."""
    if rows.shape[1] == target_len:
        return rows
    filler = rows.new_zeros((rows.shape[0], target_len - rows.shape[1]))
    return torch.cat([rows, filler], dim=1)


def _sign_extend(values: torch.Tensor, bits: int) -> torch.Tensor:
    """Reinterpret ``bits``-wide unsigned values as two's-complement signed integers."""
    sign_bit = (values >> (bits - 1)) & 1
    return values - sign_bit * (1 << bits)


def unsigned_codes_to_signed(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Reinterpret unsigned ``bits``-bit codes in ``[0, 2**bits - 1]`` as two's-complement.

    Asymmetric quantization produces unsigned codes (``qmin = 0``) while :func:`pack_codes` stores
    two's-complement patterns; the two views are the same ``bits``-bit pattern, so this conversion is
    lossless and changes no byte.

    Shapes / dtypes / device:
        Integer tensor in, int64 out, same shape and device.

    Raises:
        ValueError: For an unsupported ``bits`` or a code outside ``[0, 2**bits - 1]``.
    """
    _check_pack_bits(bits)
    work = codes.to(torch.int64)
    if work.numel() and (int(work.min()) < 0 or int(work.max()) > (1 << bits) - 1):
        raise ValueError(
            f"unsigned codes must lie in [0, {(1 << bits) - 1}] for bits={bits}, "
            f"observed [{int(work.min())}, {int(work.max())}]"
        )
    half = 1 << (bits - 1)
    return torch.where(work >= half, work - (1 << bits), work)


def signed_codes_to_unsigned(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Inverse of :func:`unsigned_codes_to_signed`: signed codes -> ``[0, 2**bits - 1]``.

    Shapes / dtypes / device:
        Integer tensor in, int64 out, same shape and device.

    Raises:
        ValueError: For an unsupported ``bits`` or a code outside the signed range.
    """
    _check_pack_bits(bits)
    work = codes.to(torch.int64)
    qmin, qmax = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    if work.numel() and (int(work.min()) < qmin or int(work.max()) > qmax):
        raise ValueError(
            f"signed codes must lie in [{qmin}, {qmax}] for bits={bits}, "
            f"observed [{int(work.min())}, {int(work.max())}]"
        )
    return torch.remainder(work, 1 << bits)


def pack_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack signed ``bits``-bit codes into the row-padded uint8 stream of the module docstring.

    Shapes:
        ``codes``: integer tensor of shape ``(*head, row_len)`` with values in
        ``[-(2**(bits-1)), 2**(bits-1) - 1]``. Returns a ``uint8`` tensor of shape
        ``(*head, ceil(row_len * bits / 32) * 4)``.

    Dtypes / device:
        The input is cast to int64 internally; the output is ``uint8`` on the input's device. Rows
        are ordered so that ``packed.reshape(-1).numpy().tobytes()`` is the exact container payload.

    Assumptions / limitations:
        See the module docstring for the bit order and the per-row 32-bit word padding. The value
        range is validated, so an out-of-range code raises instead of wrapping silently. 2- and
        3-bit codes take a bit-matrix reference path (memory proportional to ``n_rows*row_len*bits``).

    Raises:
        TypeError: If ``codes`` is not an integer tensor.
        ValueError: For a 0-dim or empty-row tensor, an out-of-range code, an unsupported ``bits``,
            or a CUDA tensor.
    """
    _check_pack_bits(bits)
    work = _validate_codes(codes, bits)
    head = tuple(work.shape[:-1])
    row_len = work.shape[-1]
    padded_bytes = row_bytes(row_len, bits)
    rows = work.reshape(-1, row_len)

    if bits == 8:
        packed = _pad_rows(rows, padded_bytes).to(torch.uint8)
    elif bits == 4:
        capacity = row_code_capacity(row_len, bits)  # == 2 * padded_bytes
        padded = _pad_rows(rows, capacity)
        packed = ((padded[:, 0::2] & 0xF) | ((padded[:, 1::2] & 0xF) << 4)).to(torch.uint8)
    else:
        total_bits = padded_bytes * 8
        bit_matrix = (rows.unsqueeze(-1) >> torch.arange(bits, dtype=torch.int64)) & 1
        bit_matrix = bit_matrix.reshape(rows.shape[0], row_len * bits)
        if row_len * bits < total_bits:
            filler = bit_matrix.new_zeros((bit_matrix.shape[0], total_bits - row_len * bits))
            bit_matrix = torch.cat([bit_matrix, filler], dim=1)
        packed = _bit_matrix_to_bytes(bit_matrix)
    return packed.reshape(*head, padded_bytes)


def unpack_codes(
    packed: torch.Tensor, shape: tuple[int, ...], bits: int, axis_padding: int
) -> torch.Tensor:
    """Undo :func:`pack_codes`: return signed ``int8`` codes of the original ``shape``.

    Shapes:
        ``packed``: ``uint8`` tensor of shape ``(*head, row_bytes)`` as produced by
        :func:`pack_codes`. ``shape``: the original (unpadded) shape; ``shape[-1]`` and the row count
        are used. Returns an ``int8`` tensor of exactly ``shape``.

    Dtypes / device:
        ``uint8`` in, ``int8`` out, on ``packed``'s device.

    Assumptions / limitations:
        ``axis_padding`` must be the value :func:`axis_padding` returns for ``(shape, bits)``;
        anything else is a misaligned read and raises. Values beyond ``shape[-1]`` are discarded, so
        the zero padding never leaks into a result. Sign extension is from the declared width, so
        2- and 3-bit codes also come back as signed ``int8``.

    Raises:
        ValueError: If ``bits`` is unsupported, if the padded-row length implied by
            ``shape``/``axis_padding`` disagrees with ``packed``, or if the row count differs.
    """
    if len(shape) == 0:
        raise ValueError("shape must have at least one dimension")
    if packed.is_cuda:  # pragma: no cover - no CUDA device exists on the target host
        raise ValueError("packing runs on CPU only: CUDA tensors are rejected by design")
    if packed.is_floating_point() or packed.is_complex():
        raise TypeError(f"packed codes must be an unsigned integer tensor, got {packed.dtype}")
    _check_pack_bits(bits)
    row_len = shape[-1]
    padded_bytes = row_bytes(row_len, bits)
    expected_padding = row_code_capacity(row_len, bits) - row_len
    if axis_padding != expected_padding:
        raise ValueError(
            f"axis_padding={axis_padding} does not match the layout for row_len={row_len}, "
            f"bits={bits} (expected {expected_padding})"
        )
    if packed.shape[-1] != padded_bytes:
        raise ValueError(
            f"packed last dimension is {packed.shape[-1]}, expected {padded_bytes} bytes per row"
        )
    if tuple(packed.shape[:-1]) != tuple(shape[:-1]):
        raise ValueError(
            f"packed row count {tuple(packed.shape[:-1])} does not match shape {tuple(shape[:-1])}"
        )
    rows = packed.contiguous().reshape(-1, padded_bytes).to(torch.int64)

    if bits == 8:
        values = rows & 0xFF
    elif bits == 4:
        low = rows & 0xF
        high = (rows >> 4) & 0xF
        values = torch.stack([low, high], dim=-1).reshape(rows.shape[0], -1)
    else:
        bit_matrix = _bytes_to_bit_matrix(rows)[:, : row_len * bits]
        grouped = bit_matrix.reshape(rows.shape[0], row_len, bits)
        code_weights = 1 << torch.arange(bits, dtype=torch.int64, device=packed.device)
        values = (grouped * code_weights).sum(dim=-1)
    signed = _sign_extend(values[:, :row_len], bits)
    return signed.to(torch.int8).reshape(shape)


def pack_int4(codes: torch.Tensor) -> torch.Tensor:
    """Pack signed int4 codes into ``uint8`` nibble pairs, **little-nibble-first**.

    Shapes:
        ``codes``: integer tensor of shape ``(*head, row_len)`` with values in ``[-8, 7]``. Returns
        ``uint8`` of shape ``(*head, ceil(row_len / 8) * 4)``.

    Dtypes / device:
        int64 internally, ``uint8`` output, same device as ``codes`` (CUDA is rejected).

    Bit order:
        Element ``2i`` occupies the low nibble (bits 0-3) of byte ``i``; element ``2i+1`` the high
        nibble (bits 4-7). Negative codes are stored as 4-bit two's complement (``-1 -> 0xF``). Each
        row is zero-padded to a whole number of 32-bit words, so ``row_len = 5`` yields 4 bytes and
        ``axis_padding(shape, 4) == 3``.

    Raises:
        TypeError / ValueError: See :func:`pack_codes`.
    """
    return pack_codes(codes, 4)


def unpack_int4(packed: torch.Tensor, shape: tuple[int, ...], axis_padding: int) -> torch.Tensor:
    """Undo :func:`pack_int4`; returns signed ``int8`` codes of shape ``shape``.

    Shapes:
        ``packed``: ``uint8`` from :func:`pack_int4``; ``shape``: the original shape.

    Dtypes / device:
        ``uint8`` in, ``int8`` out, same device.

    Assumptions / limitations:
        ``axis_padding`` must equal ``axis_padding(shape, 4)``; 4-bit two's complement is
        sign-extended to ``int8``. Every representable int4 code is recovered exactly.

    Raises:
        ValueError: On a mismatched shape, padding or byte count.
    """
    return unpack_codes(packed, shape, 4, axis_padding)


def pack_int8(codes: torch.Tensor) -> torch.Tensor:
    """Pack signed int8 codes one byte per element, rows padded to 32-bit words.

    Shapes:
        ``codes``: integer tensor of shape ``(*head, row_len)`` in ``[-128, 127]``. Returns ``uint8``
        of shape ``(*head, ceil(row_len / 4) * 4)``.

    Dtypes / device:
        int64 internally, ``uint8`` output, same device (CUDA is rejected).

    Assumptions / limitations:
        Each row is padded with zero bytes to a multiple of 4 bytes, so ``row_len = 7`` yields 8 bytes
        and ``axis_padding(shape, 8) == 1``. The padding is part of the measured payload, exactly as
        ``memory-accounting.md`` section 3.1 requires.

    Raises:
        TypeError / ValueError: See :func:`pack_codes`.
    """
    return pack_codes(codes, 8)


def unpack_int8(packed: torch.Tensor, shape: tuple[int, ...], axis_padding: int) -> torch.Tensor:
    """Undo :func:`pack_int8`; returns signed ``int8`` codes of shape ``shape``.

    Raises:
        ValueError: On a mismatched shape, padding or byte count.
    """
    return unpack_codes(packed, shape, 8, axis_padding)
