"""Byte accounting (class 1) and measured packed serialization (class 3).

**This module is the one byte-accounting source of truth** for the project
(``docs/coordination/design-m2-interfaces.md`` invariant 1): allocation, training, conversion and
reporting all obtain costs from :func:`accounted_bytes` / :func:`serialized_size_bytes` rather than
re-deriving them. The arithmetic implements ``docs/protocols/memory-accounting.md`` section 3 and
reproduces every worked example in its section 7 (asserted by
``tests/unit/test_quant_accounting.py``).

Three distinct quantities, never conflated (``AGENTS.md`` section 4.2)
----------------------------------------------------------------------
* :func:`theoretical_bits` — the **ideal** payload cost per parameter, no metadata, no padding.
  Class 1. A lower bound.
* :func:`accounted_bytes` — the analytical per-tensor cost: padded payload + per-block scales +
  zero-points. Class 1. **This is the model the packer must match.**
* :func:`measure_serialized_bytes` — the bytes of a container we actually write and re-read.
  Class 3 (measured). Presenting an analytical number as this one is forbidden
  (``memory-accounting.md`` section 6).

Model (``memory-accounting.md`` section 3)
-----------------------------------------
With ``N = numel``, payload bit width ``b = spec.bits``, ``row_len = shape[-1]`` and
``n_rows = N / row_len``::

    payload_bytes          = ceil(N * b / 8)                     # ideal, unpadded
    bytes_per_row_padded   = ceil(row_len * b / 32) * 4          # 32-bit word per row
    payload_bytes_padded   = n_rows * bytes_per_row_padded
    alignment_bytes        = payload_bytes_padded - payload_bytes
    n_blocks               = prod(params_shape(spec, *axis_view(shape, spec.axis)))
    scales_bytes           = n_blocks * spec.scale_dtype_bytes()   # 2 for per_group, else 4
    zero_points_bytes      = n_blocks * spec.zero_point_bytes()    # 0 symmetric, 1 unsigned byte else
    total_bytes            = payload_bytes_padded + scales_bytes + zero_points_bytes

``n_blocks`` is taken from :func:`spectraquant.quantization.fake_quant.params_shape`, i.e. from the
quantizer's own block layout, so the byte model cannot drift from the tensor semantics. The scale
and zero-point widths are the project's declared convention (``QuantSpec`` methods); the width is
part of the claim and must accompany it.

**Group metadata is counted, not omitted.** It *is* the scales array plus, when asymmetric, the
zero-point array — one scale (and one zero-point) per block. No additional per-group index or length
is stored because groups are contiguous and uniformly sized along ``axis`` and the layout is a
function of ``(shape, axis, group_size, bits)``, which the container's spec carries once for the
whole tensor (amortised below one byte per block for any realistic shape). :class:`ByteBreakdown`
therefore exposes ``group_metadata_bytes = scales_bytes + zero_points_bytes`` explicitly.

Container (what :func:`serialize_state` writes)
-----------------------------------------------
A headerless concatenation, in sorted key order, of one region per tensor::

    region = packed_payload || scales || zero_points

* ``packed_payload`` — :func:`spectraquant.quantization.packing.pack_codes`, i.e. little-endian
  bit-contiguous rows padded to whole 32-bit words.
* ``scales`` — ``n_blocks`` little-endian values at ``spec.scale_dtype_bytes()`` each.
* ``zero_points`` — ``n_blocks`` little-endian **unsigned** bytes (zero-points are unsigned codes),
  present only when asymmetric; the width is 1 byte, the int8-width convention of
  ``memory-accounting.md`` section 3.2.

The format is deliberately headerless: :func:`measure_serialized_bytes` and
:func:`deserialize_state` both take the shapes and specs as arguments, so no self-description needs
to be stored — and the measured byte count equals the accounted sum **exactly** (the only alignment
rule is the per-row 32-bit word padding above; there is no inter-region padding and no header). The
layout is pinned by :data:`SQ_CONTAINER_FORMAT_ID` (``"spectraquant-sqpack-v1"``), the value run
manifests carry in ``compression.serializer``; changing any term above requires a new identifier.

Numerical limitations
---------------------
* Per-group scales are stored at fp16 (the accounting convention). A load-back dequantization
  therefore differs from the in-memory fp32-parameter dequantization by up to ``2**-11`` relative
  per scale; per-tensor/per-channel scales are fp32 and round-trip bit-exactly. See
  ``tests/integration/test_pack_roundtrip.py``, which measures both.
* Zero-points are stored as 1 unsigned byte per block, the int8-width convention of
  ``memory-accounting.md`` section 3.2 (ONNX Runtime's dynamic-int8 path) and the zero-point value is
  an unsigned code in ``[0, 2**bits - 1]``. The cheaper 4-bit zero-point convention (``z_b = 0.5``)
  is a documented alternative we do not pack.
* Byte counts are integers; a *ratio* derived from them must name its numerator and denominator
  (``memory-accounting.md`` section 1).
"""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from spectraquant.quantization.fake_quant import (
    QuantParams,
    QuantSpec,
    axis_view,
    fake_dequantize,
    params_shape,
    quant_params,
    quantize_to_codes,
)
from spectraquant.quantization.packing import (
    axis_padding,
    pack_codes,
    row_bytes,
    signed_codes_to_unsigned,
    unpack_codes,
    unsigned_codes_to_signed,
)

__all__ = [
    "SQ_CONTAINER_FORMAT_ID",
    "SQ_CONTAINER_SUFFIX",
    "ByteBreakdown",
    "accounted_bytes",
    "analytical_bits_per_param",
    "byte_breakdown",
    "deserialize_state",
    "measure_serialized_bytes",
    "n_blocks",
    "scale_storage_dtype",
    "serialize_state",
    "serialized_size_bytes",
    "theoretical_bits",
    "write_serialized_state",
]

#: File suffix of the SpectraQuant packed container written by :func:`write_serialized_state`.
SQ_CONTAINER_SUFFIX = ".sqpack"

#: Frozen identifier of the container format written by :func:`serialize_state`. Run manifests carry
#: it in ``compression.serializer`` (class 3): a headerless concatenation, in sorted key order, of
#: packed payload (little-endian bit-contiguous rows padded to 32-bit words) + per-block scales
#: (fp16 for ``per_group``, else fp32) + per-block unsigned zero-point bytes (asymmetric only).
#: Changing any of those terms changes ``accounted_bytes`` and REQUIRES a new identifier.
SQ_CONTAINER_FORMAT_ID = "spectraquant-sqpack-v1"


@dataclass(frozen=True)
class ByteBreakdown:
    """Per-tensor byte accounting (class 1) for one quantized tensor.

    Fields:
        shape: The tensor shape the accounting was computed for.
        n_params: ``prod(shape)``.
        n_blocks: Number of quantization blocks (the ``n_blocks`` of
            ``memory-accounting.md`` section 2).
        bits: Payload bit width.
        payload_bytes: Ideal payload, ``ceil(n_params * bits / 8)`` — no padding.
        payload_bytes_padded: Payload as the packer writes it (rows padded to 32-bit words).
        alignment_bytes: ``payload_bytes_padded - payload_bytes`` (the section 3.1 overhead).
        scales_bytes: ``n_blocks * spec.scale_dtype_bytes()``.
        zero_points_bytes: ``n_blocks * spec.zero_point_bytes()``.
        group_metadata_bytes: ``scales_bytes + zero_points_bytes`` — the per-block metadata that
            exists *because* the tensor is grouped. Reported explicitly so it is never omitted; it
            is **not** an extra term in ``total_bytes``.
        total_bytes: ``payload_bytes_padded + scales_bytes + zero_points_bytes``.
        bits_per_param: ``8 * total_bytes / n_params`` — the analytical figure of
            ``memory-accounting.md`` section 6. Class 1: never quote it as measured bytes.
    """

    shape: tuple[int, ...]
    n_params: int
    n_blocks: int
    bits: int
    payload_bytes: int
    payload_bytes_padded: int
    alignment_bytes: int
    scales_bytes: int
    zero_points_bytes: int
    group_metadata_bytes: int
    total_bytes: int
    bits_per_param: float

    def to_dict(self) -> dict[str, object]:
        """Return the breakdown as a plain JSON-serializable dict (for manifests and prints)."""
        return {
            "shape": list(self.shape),
            "n_params": self.n_params,
            "n_blocks": self.n_blocks,
            "bits": self.bits,
            "payload_bytes": self.payload_bytes,
            "payload_bytes_padded": self.payload_bytes_padded,
            "alignment_bytes": self.alignment_bytes,
            "scales_bytes": self.scales_bytes,
            "zero_points_bytes": self.zero_points_bytes,
            "group_metadata_bytes": self.group_metadata_bytes,
            "total_bytes": self.total_bytes,
            "bits_per_param": self.bits_per_param,
        }


def _normalize_shape(w_shape: tuple[int, ...] | torch.Size | list[int]) -> tuple[int, ...]:
    """Return ``w_shape`` as a tuple of ints, rejecting empty or non-positive shapes."""
    shape = tuple(int(dim) for dim in w_shape)
    if not shape:
        raise ValueError("w_shape must have at least one dimension")
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"w_shape must be strictly positive, got {shape}")
    return shape


def n_blocks(w_shape: tuple[int, ...] | torch.Size | list[int], spec: QuantSpec) -> int:
    """Number of quantization blocks for a tensor of this shape under ``spec``.

    Shapes:
        ``w_shape``: the tensor shape (only shape, never data, is used).

    Returns:
        ``1`` (per_tensor), ``shape[axis]`` (per_channel) or
        ``(numel / shape[axis]) * ceil(shape[axis] / group_size)`` (per_group).

    Assumptions / limitations:
        Derived from :func:`spectraquant.quantization.fake_quant.params_shape`, so it is identical to
        the element count of the quantizer's own ``scales`` tensor — the property that keeps the byte
        model and the numerics in agreement.

    Raises:
        ValueError: For a non-positive shape.
        IndexError: If ``spec.axis`` is out of range.
    """
    shape = _normalize_shape(w_shape)
    outer, axis_len, inner = axis_view(shape, spec.axis)
    return math.prod(params_shape(spec, outer, axis_len, inner))


def scale_storage_dtype(spec: QuantSpec) -> torch.dtype:
    """Torch dtype our packer writes scales in: ``float16`` for ``per_group``, else ``float32``.

    Shapes / device:
        Pure spec arithmetic; returns a dtype, touches no tensor. This is the dtype
        :func:`serialize_state` casts to, and the dtype whose element width
        (:meth:`QuantSpec.scale_dtype_bytes`) :func:`byte_breakdown` counts.

    Assumptions / limitations:
        The dtype is a project convention, not a law (``memory-accounting.md`` section 3.2): a
        different scale dtype changes the byte count and must be declared with the claim.
    """
    return torch.float16 if spec.scale_dtype_bytes() == 2 else torch.float32


def byte_breakdown(
    w_shape: tuple[int, ...] | torch.Size | list[int], spec: QuantSpec
) -> ByteBreakdown:
    """Full class-1 byte accounting for one tensor of ``w_shape`` under ``spec``.

    Shapes:
        ``w_shape``: tensor shape; rows for alignment are the **last axis**
        (``row_len = w_shape[-1]``, ``n_rows = numel / row_len``), as in
        ``memory-accounting.md`` section 3.1.

    Dtypes / device:
        Pure integer arithmetic on the shape; nothing is allocated or executed. Class 1
        (analytical) — the returned numbers MUST NOT be reported as measured bytes.

    Assumptions / limitations:
        * Scale width is fp16 for ``per_group`` and fp32 otherwise; zero-point width is int8 (1 byte)
          when asymmetric and 0 when symmetric (``QuantSpec`` methods; the width is part of the
          claim).
        * Alignment is per-row 32-bit words only (``section 3.1``); no container header is counted
          because a SpectraQuant container is headerless (see the module docstring). Container
          overhead of a *third-party* format is a measured, per-artifact quantity and must be added
          outside this function (``section 6.1``).
        * A trailing short group still costs one full block of scale/zero-point bytes (the ``ceil``),
          which matches the quantizer's block layout.

    Raises:
        ValueError / IndexError: For a non-positive shape or an out-of-range axis.
    """
    shape = _normalize_shape(w_shape)
    n_params = math.prod(shape)
    blocks = n_blocks(shape, spec)
    row_len = shape[-1]
    n_rows = n_params // row_len

    payload_bytes = math.ceil(n_params * spec.bits / 8)
    padded_row_bytes = row_bytes(row_len, spec.bits)
    payload_bytes_padded = n_rows * padded_row_bytes
    scales_bytes = blocks * spec.scale_dtype_bytes()
    zero_points_bytes = blocks * spec.zero_point_bytes()
    total = payload_bytes_padded + scales_bytes + zero_points_bytes
    return ByteBreakdown(
        shape=shape,
        n_params=n_params,
        n_blocks=blocks,
        bits=spec.bits,
        payload_bytes=payload_bytes,
        payload_bytes_padded=payload_bytes_padded,
        alignment_bytes=payload_bytes_padded - payload_bytes,
        scales_bytes=scales_bytes,
        zero_points_bytes=zero_points_bytes,
        group_metadata_bytes=scales_bytes + zero_points_bytes,
        total_bytes=total,
        bits_per_param=8.0 * total / n_params,
    )


def accounted_bytes(w_shape: tuple[int, ...] | torch.Size | list[int], spec: QuantSpec) -> int:
    """Analytical bytes of one tensor: padded payload + scales + zero-points (class 1).

    Shapes / dtypes / device:
        Shape-only arithmetic; see :func:`byte_breakdown` for the full term list. This is the
        function other slices MUST call for costs (design note invariant 1).

    Assumptions / limitations:
        Same conventions as :func:`byte_breakdown`. Equal to the bytes our headerless container
        actually writes, which is asserted for the three headline configurations in the
        reconciliation test and for 2/3/4/8-bit layouts in the packing tests.

    Raises:
        ValueError / IndexError: See :func:`byte_breakdown`.
    """
    return byte_breakdown(w_shape, spec).total_bytes


def serialized_size_bytes(
    w_shape: tuple[int, ...] | torch.Size | list[int], spec: QuantSpec
) -> int:
    """Bytes this tensor contributes to a SpectraQuant-serialized container.

    Shapes / dtypes / device:
        Shape-only arithmetic; identical to :func:`accounted_bytes` **by construction** — the
        container has no header, no inter-region padding and stores exactly the accounted terms, so
        ``serialized_size_bytes == accounted_bytes`` (asserted by tests). The two names exist because
        the design note lists both intents: "what we account for" and "what a serialize call will
        write".

    Assumptions / limitations:
        A container written by another tool adds its own container overhead, which is a *measured*
        per-artifact quantity and must be summed separately (``memory-accounting.md`` section 6.1).

    Raises:
        ValueError / IndexError: See :func:`byte_breakdown`.
    """
    return accounted_bytes(w_shape, spec)


def theoretical_bits(w_shape: tuple[int, ...] | torch.Size | list[int], spec: QuantSpec) -> float:
    """Ideal payload bits per parameter, with no metadata and no padding (class 1).

    Shapes:
        ``w_shape``: tensor shape. Returns ``8 * ceil(numel * bits / 8) / numel``, which equals
        ``spec.bits`` whenever ``numel * bits`` is a whole number of bytes and is slightly larger for
        degenerate tiny tensors (a 4-bit scalar needs a byte).

    Dtypes / device:
        Pure arithmetic; no tensor is touched.

    Assumptions / limitations:
        This is a **lower bound**, not a storage figure: it excludes scales, zero-points and padding.
        It MUST NOT be presented as measured storage or compared against
        :func:`measure_serialized_bytes` without stating the difference
        (``memory-accounting.md`` section 6). Use :func:`analytical_bits_per_param` for the
        ``bits_per_param`` column of the memory model.

    Raises:
        ValueError: For a non-positive shape or unsupported bit width.
    """
    shape = _normalize_shape(w_shape)
    if spec.bits not in (2, 3, 4, 8):
        raise ValueError(f"unsupported bit width {spec.bits!r}")
    numel = math.prod(shape)
    return 8.0 * math.ceil(numel * spec.bits / 8) / numel


def analytical_bits_per_param(
    w_shape: tuple[int, ...] | torch.Size | list[int], spec: QuantSpec
) -> float:
    """``8 * accounted_bytes / numel`` — the analytical ``bits_per_param`` of the memory model.

    Shapes / dtypes / device:
        Shape-only; no execution. Class 1.

    Assumptions / limitations:
        Includes padding, scales and zero-points but **not** container overhead, so it is the
        ``bits_per_param_analytical`` column of ``memory-accounting.md`` section 6 and must never be
        printed as ``bits_per_param_measured``.

    Raises:
        ValueError / IndexError: See :func:`byte_breakdown`.
    """
    breakdown = byte_breakdown(w_shape, spec)
    return breakdown.bits_per_param


def _checked_state(state: Mapping[str, torch.Tensor], specs: Mapping[str, QuantSpec]) -> list[str]:
    """Validate a state/spec mapping pair and return its keys in the container's write order."""
    if set(state) != set(specs):
        missing = sorted(set(state) - set(specs))
        extra = sorted(set(specs) - set(state))
        raise ValueError(f"state and specs keys differ (missing specs: {missing}, extra: {extra})")
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"state[{name!r}] must be a torch.Tensor, got {type(tensor).__name__}")
        if tensor.is_cuda:  # pragma: no cover - no CUDA device exists on the target host
            raise ValueError("serialization runs on CPU only: CUDA tensors are rejected by design")
        if not tensor.is_floating_point():
            raise TypeError(f"state[{name!r}] must be a float tensor, got {tensor.dtype}")
        if tensor.numel() == 0 or tensor.ndim == 0:
            raise ValueError(f"state[{name!r}] must be a non-empty tensor of rank >= 1")
    return sorted(state)


def serialize_state(state: Mapping[str, torch.Tensor], specs: Mapping[str, QuantSpec]) -> bytes:
    """Quantize and pack a whole state dict into the container bytes (class 3 payload).

    Shapes:
        ``state``: mapping of tensor name -> float tensor (rank >= 1). ``specs``: mapping of the same
        names to specs. The output is one ``bytes`` object of length
        ``sum(serialized_size_bytes(shape, spec))``.

    Dtypes / device:
        Inputs are quantized in float32/float64 on their own device; the return value is host
        ``bytes``. CUDA tensors are rejected (no CUDA device exists on this host).

    Assumptions / limitations:
        * Regions are written in **sorted key order** and concatenated with no header and no
          inter-region padding; the reader must be given the same shapes and specs.
        * Per-group scales are cast to fp16 (the accounting convention), so a load-back dequantizes
          with up to ``2**-11`` relative scale error; per-tensor/per-channel scales stay fp32.
        * This is a serialization of *quantized* weights: the same fake-quantized float tensor is
          never itself written (that would be class 2 storage, which does not exist here).

    Raises:
        ValueError: If the key sets differ, for a non-float/empty/0-dim tensor, or for a shape whose
            last dimension is not positive.
        TypeError: If a state value is not a float tensor.
        IndexError: If a spec axis is out of range for its tensor.
    """
    chunks: list[bytes] = []
    for name in _checked_state(state, specs):
        weight = state[name]
        spec = specs[name]
        params = quant_params(weight, spec)
        codes = quantize_to_codes(weight, params, spec).to(torch.int64)
        if not spec.symmetric:
            # Asymmetric codes are unsigned; the packer stores two's-complement patterns, which is
            # the same `bits`-bit pattern and therefore the same byte count.
            codes = unsigned_codes_to_signed(codes, spec.bits)
        packed = pack_codes(codes, spec.bits).contiguous()
        chunks.append(packed.numpy().tobytes())

        scales = params.scales.reshape(-1).to(scale_storage_dtype(spec)).contiguous()
        chunks.append(scales.numpy().tobytes())

        if not spec.symmetric:
            # Zero-points are unsigned codes in [0, 2**bits - 1]: store them as unsigned bytes.
            zero_points = params.zero_points.reshape(-1).round().to(torch.uint8).contiguous()
            chunks.append(zero_points.numpy().tobytes())
    return b"".join(chunks)


def write_serialized_state(
    state: Mapping[str, torch.Tensor], specs: Mapping[str, QuantSpec], directory: str | Path
) -> Path:
    """Write :func:`serialize_state` output into ``directory`` and return the file path.

    Shapes / dtypes / device:
        See :func:`serialize_state`. Writes exactly one file,
        ``directory/weights{SQ_CONTAINER_SUFFIX}``.

    Assumptions / limitations:
        The directory is created if missing. A real artifact may contain more files (sidecar data,
        config, tokenizer); ``memory-accounting.md`` section 6.1 requires summing **every** file of
        an artifact, so callers that add files must add their sizes too.

    Raises:
        ValueError / TypeError / IndexError: See :func:`serialize_state`.
        OSError: If the file cannot be written.
    """
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"weights{SQ_CONTAINER_SUFFIX}"
    path.write_bytes(serialize_state(state, specs))
    return path


def measure_serialized_bytes(
    state: Mapping[str, torch.Tensor], specs: Mapping[str, QuantSpec]
) -> int:
    """Serialize ``state`` for real and return the measured container size in bytes (class 3).

    Shapes / dtypes / device:
        See :func:`serialize_state`. The measurement is ``os.path.getsize`` of the file that
        :func:`write_serialized_state` actually wrote into a temporary directory, summed over every
        file the write produced — i.e. a measured byte count, not a derivation
        (``memory-accounting.md`` section 6.1).

    Assumptions / limitations:
        * The temporary directory is removed afterwards; nothing is left on disk.
        * The number includes exactly the accounted terms (padded payload + scales + zero-points)
          because the container is headerless; a third-party container's overhead would have to be
          measured and reported separately.

    Returns:
        Measured bytes, equal to ``sum(accounted_bytes(shape, spec) for ...)`` for our container.

    Raises:
        ValueError / TypeError / IndexError: See :func:`serialize_state`.
        OSError: If the temporary file cannot be written.
    """
    with tempfile.TemporaryDirectory(prefix="spectraquant-sqpack-") as tmp:
        write_serialized_state(state, specs, tmp)
        return sum(os.path.getsize(entry.path) for entry in os.scandir(tmp) if entry.is_file())


def deserialize_state(
    blob: bytes,
    shapes: Mapping[str, tuple[int, ...] | torch.Size | list[int]],
    specs: Mapping[str, QuantSpec],
) -> dict[str, torch.Tensor]:
    """Read a :func:`serialize_state` container back and dequantize it.

    Shapes:
        ``blob``: the container bytes. ``shapes``/``specs``: the same keys, shapes and specs used to
        write it (the order is re-derived as sorted key order, so no header is needed). Returns a
        mapping name -> float32 tensor of the original shape.

    Dtypes / device:
        Returns float32 CPU tensors; the arithmetic mirrors :func:`fake_dequantize`.

    Assumptions / limitations:
        * Regions are located from the accounting model
          (:func:`byte_breakdown`); any length mismatch raises rather than silently misreading.
        * Per-group scales are fp16 in the container, so per-group reconstructions differ from an
          in-memory fp32-parameter dequantization by up to ``2**-11`` relative scale error;
          per-tensor/per-channel reconstructions are bit-identical to the in-memory ones.
        * Load-back comparison is a class-3 evidence requirement (``measurement-taxonomy.md``
          section 1, "Required evidence"): report ``max_abs`` error together with the fixture
          distribution and the reference definition.

    Raises:
        ValueError: If the key sets differ, a region is truncated, or a shape does not match its
            region length.
        IndexError: If a spec axis is out of range.
    """
    if set(shapes) != set(specs):
        raise ValueError("shapes and specs keys differ")
    reconstructed: dict[str, torch.Tensor] = {}
    offset = 0
    for name in sorted(shapes):
        shape = _normalize_shape(shapes[name])
        spec = specs[name]
        breakdown = byte_breakdown(shape, spec)
        end = offset + breakdown.total_bytes
        if end > len(blob):
            raise ValueError(
                f"container truncated reading {name!r}: need {breakdown.total_bytes} bytes at "
                f"offset {offset}, have {len(blob) - offset}"
            )
        payload = blob[offset : offset + breakdown.payload_bytes_padded]
        cursor = offset + breakdown.payload_bytes_padded
        if spec.scale_dtype_bytes() == 2:
            scales = torch.frombuffer(
                bytearray(blob[cursor : cursor + breakdown.scales_bytes]), dtype=torch.float16
            )
        else:
            scales = torch.frombuffer(
                bytearray(blob[cursor : cursor + breakdown.scales_bytes]), dtype=torch.float32
            )
        cursor += breakdown.scales_bytes
        if breakdown.zero_points_bytes:
            zero_points = torch.frombuffer(
                bytearray(blob[cursor : cursor + breakdown.zero_points_bytes]), dtype=torch.uint8
            ).to(torch.float32)
        else:
            zero_points = torch.zeros(breakdown.n_blocks, dtype=torch.float32)

        padded_row_bytes = row_bytes(shape[-1], spec.bits)
        n_rows = math.prod(shape[:-1])
        if len(payload) != n_rows * padded_row_bytes:
            raise ValueError(
                f"payload for {name!r} is {len(payload)} bytes, "
                f"expected {n_rows * padded_row_bytes}"
            )
        packed = torch.frombuffer(bytearray(payload), dtype=torch.uint8).reshape(
            *shape[:-1], padded_row_bytes
        )
        codes = unpack_codes(packed.clone(), shape, spec.bits, axis_padding(shape, spec.bits))
        code_values: torch.Tensor = codes.to(torch.float32)
        if not spec.symmetric:
            code_values = signed_codes_to_unsigned(codes, spec.bits).to(torch.float32)
        params = QuantParams(
            scales=scales.to(torch.float32)
            .clone()
            .reshape(params_shape(spec, *axis_view(shape, spec.axis))),
            zero_points=zero_points.clone().reshape(
                params_shape(spec, *axis_view(shape, spec.axis))
            ),
            qmin=spec.qmin,
            qmax=spec.qmax,
        )
        reconstructed[name] = fake_dequantize(code_values, params, spec)
        offset = end
    if offset != len(blob):
        raise ValueError(
            f"container has {len(blob) - offset} trailing bytes beyond the accounted regions"
        )
    return reconstructed
