"""Integration: quantize -> pack -> write to disk -> read back -> dequantize, with byte reconciliation.

Fixture distributions are stated explicitly (``memory-accounting.md`` section 8.8 requires the
reference definition and the ``W``/``X`` distributions with any error figure):

* ``W_a``: shape ``(out=4, in=5)``, ``numpy``-equivalent ``Normal(0, 0.1)`` drawn with
  ``torch.Generator().manual_seed(0)`` — an **odd** row length, so packing pads 3 code slots.
* ``W_b``: shape ``(out=8, in=7)``, ``Normal(0, 1)`` with ``manual_seed(1)``.
* Reference: ``fake_quantize(W)`` in float32 (the in-memory reference quantization). The measured
  container is then read back and compared against it; the comparison is the class-3 load-back round
  trip required by ``measurement-taxonomy.md`` section 1.
"""

from __future__ import annotations

import os

import pytest
import torch

from spectraquant.quantization import (
    QuantSpec,
    accounted_bytes,
    axis_padding,
    byte_breakdown,
    deserialize_state,
    fake_quantize,
    quant_params,
    quantize_to_codes,
    serialize_state,
    unpack_codes,
    unsigned_codes_to_signed,
    write_serialized_state,
)
from spectraquant.quantization.packing import row_bytes

pytestmark = pytest.mark.integration

W_A_SHAPE = (4, 5)
W_B_SHAPE = (8, 7)
SPEC_A = QuantSpec(4, "per_group", 3, True, axis=1)  # fp16 scales, rows padded to 4 bytes
SPEC_B = QuantSpec(8, "per_channel", None, False, axis=0)  # fp32 scales + 1-byte zero-points


def _fixture() -> dict[str, torch.Tensor]:
    """The two stated-distribution weights used by every test in this module."""
    a = torch.randn(W_A_SHAPE, generator=torch.Generator().manual_seed(0)) * 0.1
    b = torch.randn(W_B_SHAPE, generator=torch.Generator().manual_seed(1))
    return {"W_a": a, "W_b": b}


def _specs() -> dict[str, QuantSpec]:
    return {"W_a": SPEC_A, "W_b": SPEC_B}


def _payload_of(
    blob: bytes, state: dict[str, torch.Tensor], specs: dict[str, QuantSpec], key: str
) -> torch.Tensor:
    """Slice one tensor's packed-payload region out of the container (sorted key order)."""
    offset = 0
    for name in sorted(state):
        breakdown = byte_breakdown(tuple(state[name].shape), specs[name])
        if name == key:
            payload = blob[offset : offset + breakdown.payload_bytes_padded]
            return torch.frombuffer(bytearray(payload), dtype=torch.uint8).reshape(
                *state[name].shape[:-1], row_bytes(state[name].shape[-1], specs[name].bits)
            )
        offset += breakdown.total_bytes
    raise KeyError(key)


def test_pack_roundtrip_reconciles_bytes_and_reconstructs_the_weights(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    state = _fixture()
    specs = _specs()
    path = write_serialized_state(state, specs, tmp_path)
    measured = sum(entry.stat().st_size for entry in tmp_path.iterdir() if entry.is_file())
    accounted = sum(accounted_bytes(tuple(t.shape), specs[name]) for name, t in state.items())

    blob = path.read_bytes()
    assert len(blob) == measured == os.path.getsize(path)
    assert measured == accounted, (measured, accounted)

    reconstructed = deserialize_state(blob, {k: tuple(v.shape) for k, v in state.items()}, specs)

    with capsys.disabled():
        print(
            f"\ncontainer: {path.name} measured={measured} B accounted={accounted} B "
            f"residual={measured - accounted} B (class 3 vs class 1)"
        )
        for name in sorted(state):
            breakdown = byte_breakdown(tuple(state[name].shape), specs[name])
            print(
                f"  {name}: shape={tuple(state[name].shape)} spec={specs[name].granularity}"
                f"/{specs[name].bits}bit/sym={specs[name].symmetric} "
                f"payload={breakdown.payload_bytes_padded} scales={breakdown.scales_bytes} "
                f"zp={breakdown.zero_points_bytes} total={breakdown.total_bytes} "
                f"axis_padding={axis_padding(tuple(state[name].shape), specs[name].bits)}"
            )

    for name, weight in state.items():
        spec = specs[name]
        reference = fake_quantize(weight, spec)  # in-memory float32 reference
        restored = reconstructed[name]
        error = (restored - reference).abs()
        relative = float(error.max()) / max(float(reference.abs().max()), 1e-12)
        with capsys.disabled():
            print(
                f"  load-back {name}: max_abs={float(error.max()):.3e} "
                f"mean_abs={float(error.mean()):.3e} relative_max={relative:.3e} "
                f"reference_absmax={float(reference.abs().max()):.4f} dtype={restored.dtype}"
            )
        if spec.granularity == "per_group":
            # Only per-group scales are stored fp16 (the accounting convention), so the drift is
            # bounded by the fp16 relative rounding of the scale, 2**-11.
            assert relative <= 2**-11 * 1.01
        else:
            assert torch.equal(restored, reference)  # fp32 scales: bit-exact load back


def test_packed_codes_are_recovered_exactly_from_the_container_bytes() -> None:
    """The payload region decodes to exactly the codes the quantizer produced (layout proof)."""
    state = _fixture()
    specs = _specs()
    blob = serialize_state(state, specs)
    for name, weight in state.items():
        spec = specs[name]
        params = quant_params(weight, spec)
        expected = quantize_to_codes(weight, params, spec).to(torch.int64)
        if not spec.symmetric:
            # The container stores two's-complement patterns; convert the unsigned codes the same way.
            expected = unsigned_codes_to_signed(expected, spec.bits)
        packed = _payload_of(blob, state, specs, name)
        decoded = unpack_codes(
            packed, tuple(weight.shape), spec.bits, axis_padding(tuple(weight.shape), spec.bits)
        )
        assert torch.equal(decoded.to(torch.int64), expected)


def test_odd_row_length_padding_is_explicit_and_functional() -> None:
    """A 5-wide 4-bit row costs 4 bytes (3 padded slots) and still round-trips."""
    assert axis_padding(W_A_SHAPE, 4) == 3
    assert row_bytes(5, 4) == 4
    breakdown = byte_breakdown(W_A_SHAPE, SPEC_A)
    assert breakdown.payload_bytes == 10  # 20 codes x 4 bits / 8
    assert breakdown.payload_bytes_padded == 16  # 4 rows x 4 bytes
    assert breakdown.alignment_bytes == 6


def test_container_survives_two_tensors_with_different_granularities(tmp_path) -> None:
    state = _fixture()
    specs = _specs()
    first = serialize_state(state, specs)
    second = serialize_state(state, specs)
    assert first == second  # deterministic container
    path = write_serialized_state(state, specs, tmp_path)
    assert path.read_bytes() == first
    restored = deserialize_state(first, {k: tuple(v.shape) for k, v in state.items()}, specs)
    assert set(restored) == set(state)
    assert torch.equal(restored["W_b"], fake_quantize(state["W_b"], SPEC_B))
