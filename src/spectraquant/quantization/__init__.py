"""Quantization package — Milestone 2 (owner: quantization stream, agent ``QuantizationCore``).

Public surface, frozen by ``docs/coordination/design-m2-interfaces.md`` section 1:

* :mod:`spectraquant.quantization.fake_quant` — :class:`QuantSpec`, :class:`QuantParams`,
  :func:`quant_params`, :func:`fake_quantize` (straight-through estimator),
  :func:`fake_dequantize`, :func:`exact_round_trip_ok`. Measurement **class 2** (float execution that
  simulates quantization numerics): never low-bit storage, never accelerated inference.
* :mod:`spectraquant.quantization.packing` — :func:`pack_int4`/:func:`unpack_int4` (signed int4,
  little-nibble-first, rows padded to 32-bit words), the int8 path, and the generic
  :func:`pack_codes`/:func:`unpack_codes` used by the container.
* :mod:`spectraquant.quantization.codebook` — the 2-bit NF-style codebook of the frozen M3/R1
  LoftQ arms and its uniform-int2 control: :class:`CodebookSpec`,
  :func:`nf_levels`, :func:`uniform_int2_levels`, :func:`fake_quantize_codebook`,
  :func:`codebook_codes`, the pinned reference's affine control
  :func:`fake_quantize_uniform_int2`, and the class-1 cost model
  :func:`codebook_accounted_bytes`. Measurement **class 2** (float simulation of a nearest-level
  lookup): never low-bit storage, never accelerated inference.
* :mod:`spectraquant.quantization.accounting` — **the one byte-accounting source of truth**
  (design note invariant 1): :func:`theoretical_bits` (class 1, ideal payload),
  :func:`accounted_bytes` / :func:`serialized_size_bytes` (class 1 model),
  :func:`measure_serialized_bytes` (class 3, real serialization),
  :func:`serialize_state` / :func:`deserialize_state`.

CUDA-only kernels (bitsandbytes, TorchAO CUDA paths, GPTQ/AWQ GPU kernels, vLLM) are deliberately
absent and are deferred behind a documented GPU prerequisite (ADR-0002): packers and quantizers here
reject CUDA tensors outright rather than falling back to different numerics.
"""

from __future__ import annotations

from spectraquant.quantization.accounting import (
    SQ_CONTAINER_FORMAT_ID,
    ByteBreakdown,
    accounted_bytes,
    analytical_bits_per_param,
    byte_breakdown,
    deserialize_state,
    measure_serialized_bytes,
    n_blocks,
    scale_storage_dtype,
    serialize_state,
    serialized_size_bytes,
    theoretical_bits,
    write_serialized_state,
)
from spectraquant.quantization.codebook import (
    CODEBOOK_KINDS,
    CODEBOOK_SCALE_BYTES,
    FROZEN_CODEBOOK_BITS,
    NF_OFFSET,
    CodebookSpec,
    codebook_accounted_bytes,
    codebook_codes,
    fake_quantize_codebook,
    fake_quantize_uniform_int2,
    nf_levels,
    uniform_int2_levels,
)
from spectraquant.quantization.fake_quant import (
    GRANULARITIES,
    ROUND_MODES,
    SUPPORTED_BITS,
    QuantParams,
    QuantSpec,
    axis_view,
    exact_round_trip_ok,
    fake_dequantize,
    fake_quantize,
    params_shape,
    quant_params,
    quantization_error,
    quantize_to_codes,
)
from spectraquant.quantization.packing import (
    INT4_MAX,
    INT4_MIN,
    INT8_MAX,
    INT8_MIN,
    axis_padding,
    pack_codes,
    pack_int4,
    pack_int8,
    row_bytes,
    row_code_capacity,
    signed_codes_to_unsigned,
    unpack_codes,
    unpack_int4,
    unpack_int8,
    unsigned_codes_to_signed,
)

__all__ = [
    "CODEBOOK_KINDS",
    "CODEBOOK_SCALE_BYTES",
    "FROZEN_CODEBOOK_BITS",
    "GRANULARITIES",
    "INT4_MAX",
    "INT4_MIN",
    "INT8_MAX",
    "INT8_MIN",
    "NF_OFFSET",
    "ROUND_MODES",
    "SQ_CONTAINER_FORMAT_ID",
    "SUPPORTED_BITS",
    "ByteBreakdown",
    "CodebookSpec",
    "QuantParams",
    "QuantSpec",
    "accounted_bytes",
    "analytical_bits_per_param",
    "axis_padding",
    "axis_view",
    "byte_breakdown",
    "codebook_accounted_bytes",
    "codebook_codes",
    "deserialize_state",
    "exact_round_trip_ok",
    "fake_dequantize",
    "fake_quantize",
    "fake_quantize_codebook",
    "fake_quantize_uniform_int2",
    "measure_serialized_bytes",
    "n_blocks",
    "nf_levels",
    "pack_codes",
    "pack_int4",
    "pack_int8",
    "params_shape",
    "quant_params",
    "quantization_error",
    "quantize_to_codes",
    "row_bytes",
    "row_code_capacity",
    "scale_storage_dtype",
    "serialize_state",
    "serialized_size_bytes",
    "signed_codes_to_unsigned",
    "theoretical_bits",
    "uniform_int2_levels",
    "unpack_codes",
    "unpack_int4",
    "unpack_int8",
    "unsigned_codes_to_signed",
    "write_serialized_state",
]
