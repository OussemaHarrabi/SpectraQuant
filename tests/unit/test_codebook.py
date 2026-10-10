"""2-bit codebook quantizer: spec validation, reference numerics, oracle fixture, byte accounting.

Two independent sources of truth are asserted here:

* the **hand-computed** 4-element block below (an exact, self-contained derivation), and
* the **pinned oracle fixture** ``artifacts/sample-results/m3-oracle/loftq-nf2-block64.json``,
  captured from the reference implementation (``yxli2123/LoftQ`` @ ``ae33fd4f``,
  ``glue/utils_qaunt.py``, MIT). That reference was cloned **outside** this repository; only its
  numbers and provenance are committed, and no third-party source is imported or vendored here.
  The fixture is the contract: our levels and block outputs must reproduce it to its own stated
  ``tolerance.absolute_per_element`` (1e-10, i.e. the 10-decimal JSON rounding of float32 values).

Measurement classes: the numerics asserted here are class 2 (float simulation of a codebook
lookup); the byte counts are class 1 (analytical). See ``AGENTS.md`` section 5.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import pytest
import torch

from spectraquant.quantization import (
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
from spectraquant.quantization.accounting import byte_breakdown
from spectraquant.quantization.fake_quant import QuantSpec

REPO_ROOT = Path(__file__).resolve().parents[2]
ORACLE_FIXTURE = REPO_ROOT / "artifacts" / "sample-results" / "m3-oracle" / "loftq-nf2-block64.json"

NF_SPEC = CodebookSpec(bits=2, kind="nf", block_size=64)
UNIFORM_SPEC = CodebookSpec(bits=2, kind="uniform", block_size=64)

#: The reference's 2-bit NormalFloat table, as produced by its own float32 choreography
#: (float32 ``linspace`` probabilities -> float64 quantile -> float32 cast -> float32 / max).
NF2_EXPECTED = torch.tensor([-1.0, 0.0, 0.33791524171829224, 1.0])


def _normal(shape: tuple[int, ...], *, seed: int = 0, sigma: float = 1.0) -> torch.Tensor:
    """Standard normal fixture: ``torch.randn`` from a fixed generator, scaled by ``sigma``."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator) * sigma


def _block_absmax_like(w: torch.Tensor, block_size: int) -> torch.Tensor:
    """Per-element block absmax for the documented row-major flatten blocking (test-side model).

    Computed independently of the implementation (``reshape(-1)`` then chunks) so that the
    "output values are a level times the block scale" check does not reuse the code under test.
    """
    flat = w.reshape(-1)
    n_blocks = math.ceil(flat.numel() / block_size)
    padded = torch.zeros(n_blocks * block_size, dtype=w.dtype)
    padded[: flat.numel()] = flat
    absmax = padded.reshape(n_blocks, block_size).abs().amax(dim=1)
    return absmax.repeat_interleave(block_size)[: w.numel()].reshape(w.shape)


def _oracle() -> dict:
    """Load the pinned oracle fixture, skipping when the producing workstream has not landed it."""
    if not ORACLE_FIXTURE.is_file():
        pytest.skip(f"oracle fixture not present: {ORACLE_FIXTURE.relative_to(REPO_ROOT)}")
    with ORACLE_FIXTURE.open(encoding="utf-8") as handle:
        return json.load(handle)


def _f32(values: object) -> torch.Tensor:
    """Fixture numbers -> the float32 tensors the reference pipeline produced them with."""
    return torch.tensor(values, dtype=torch.float32)  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# CodebookSpec validation
# --------------------------------------------------------------------------------------


def test_spec_defaults_are_the_frozen_arm_setting() -> None:
    spec = CodebookSpec(bits=2, kind="nf")
    assert (spec.bits, spec.kind, spec.block_size, spec.axis) == (2, "nf", 64, -1)
    assert spec.n_levels == 4
    assert FROZEN_CODEBOOK_BITS == 2
    assert CODEBOOK_KINDS == ("nf", "uniform")


@pytest.mark.parametrize("bits", [1, 3, 4, 8, -2, 0])
def test_spec_refuses_every_width_but_two(bits: int) -> None:
    """The frozen plan fixes the R1 codebook at 2 bits; other widths must not be accepted."""
    with pytest.raises(ValueError, match="reproduction-plan") as excinfo:
        CodebookSpec(bits, "nf")
    assert "2 bits" in str(excinfo.value)


def test_spec_refuses_non_int_bits() -> None:
    with pytest.raises(ValueError, match="bits must be an int"):
        CodebookSpec(2.0, "nf")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="bits must be an int"):
        CodebookSpec(True, "nf")


def test_spec_refuses_unknown_kind() -> None:
    with pytest.raises(ValueError, match="kind must be one of"):
        CodebookSpec(2, "nf4")


@pytest.mark.parametrize("block_size", [0, -1])
def test_spec_refuses_non_positive_block_size(block_size: int) -> None:
    with pytest.raises(ValueError, match="block_size must be >= 1"):
        CodebookSpec(2, "nf", block_size)


def test_spec_refuses_non_int_block_size_and_axis() -> None:
    with pytest.raises(ValueError, match="block_size must be an int"):
        CodebookSpec(2, "nf", 64.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="axis must be an int"):
        CodebookSpec(2, "nf", 64, 1.5)  # type: ignore[arg-type]


def test_spec_is_frozen_and_hashable() -> None:
    spec = CodebookSpec(2, "nf")
    assert hash(spec) == hash(CodebookSpec(2, "nf", 64, -1))
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.bits = 4  # type: ignore[misc]


def test_spec_levels_bind_to_the_requested_table() -> None:
    assert torch.equal(NF_SPEC.levels(), nf_levels(2))
    assert torch.equal(UNIFORM_SPEC.levels(), uniform_int2_levels())


# --------------------------------------------------------------------------------------
# Level tables
# --------------------------------------------------------------------------------------


def test_nf2_levels_equal_the_reference_value() -> None:
    """Pin the 2-bit table to the reference's numbers (see the module docstring's derivation)."""
    levels = nf_levels(2)
    assert levels.dtype == torch.float32
    assert levels.numel() == 4
    torch.testing.assert_close(levels, NF2_EXPECTED, rtol=0.0, atol=0.0)


def test_nf_table_is_monotone_with_exact_endpoints_and_one_zero() -> None:
    levels = nf_levels(2)
    assert bool((torch.diff(levels) > 0).all()), "levels must be strictly increasing"
    assert float(levels[0]) == -1.0 and float(levels[-1]) == 1.0, "normalized to [-1, 1]"
    assert int((levels == 0).sum()) == 1, "the reference table carries exactly one exact zero"


def test_nf_table_is_not_symmetric_about_zero() -> None:
    """Measured property of the reference's QLoRA-default (asymmetric) map.

    ``create_normal_map(symmetric=False)`` gives three non-positive levels and one positive level at
    2 bits, so the table differs from its own sign-flipped mirror. This is deliberate: the frozen
    assignment's expectation of a symmetric table does not hold for the pinned reference, whose
    asymmetry is the QLoRA default (the fixture's ``outputs.nf2_levels`` carries the same numbers).
    """
    levels = nf_levels(2)
    assert not torch.allclose(levels, -levels.flip(0)), "expected an asymmetric reference table"
    assert float(levels[1]) == 0.0 and float(levels[2]) > 0.0
    assert abs(float(levels[0])) > float(levels[2]), "negative side reaches further than positive"


def test_nf_table_reproduces_the_nf4_family_at_four_bits() -> None:
    """The same construction at 4 bits reproduces the published NF4 codebook (sanity anchor)."""
    levels = nf_levels(4)
    assert levels.numel() == 16
    assert float(levels[0]) == -1.0 and float(levels[-1]) == 1.0
    # Anchor values of the bitsandbytes/QLoRA NF4 table (float32-computed; 1e-6 covers the
    # published table's own last-ulp differences from this float64-quantile path).
    for index, expected in (
        (1, -0.6961928009986877),
        (8, 0.07958029955625534),
        (14, 0.7229568362236023),
    ):
        torch.testing.assert_close(levels[index], torch.tensor(expected), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("bits", [3, 4, 8])
def test_nf_levels_are_generic_but_spec_restricts_to_two(bits: int) -> None:
    levels = nf_levels(bits)
    assert levels.numel() == 2**bits
    assert bool((torch.diff(levels) > 0).all())
    assert float(levels.abs().max()) == 1.0
    with pytest.raises(ValueError, match="bits must be 2"):
        CodebookSpec(bits, "nf")


def test_nf_levels_rejects_unsupported_and_non_int_widths() -> None:
    with pytest.raises(ValueError, match="bits must be one of"):
        nf_levels(5)
    with pytest.raises(TypeError, match="bits must be an int"):
        nf_levels(True)


def test_uniform_int2_levels_are_the_uniform_codebook_grid() -> None:
    levels = uniform_int2_levels()
    assert levels.dtype == torch.float32 and levels.numel() == 4
    torch.testing.assert_close(
        levels, torch.tensor([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0]), rtol=0.0, atol=0.0
    )
    assert bool((torch.diff(levels) > 0).all()), "monotone"
    # Uniform spacing: the float32 grid 1/3, 2/3 makes the outermost gap one ulp wider.
    torch.testing.assert_close(torch.diff(levels), torch.full((3,), 2.0 / 3.0), rtol=0.0, atol=1e-7)
    assert float(levels[0]) == -1.0 and float(levels[-1]) == 1.0
    assert not bool((levels == 0).any()), "the uniform codebook carries no zero level"


def test_every_codebook_table_contains_both_extremes() -> None:
    """The property that makes the absmax round trip idempotent, asserted for both tables.

    With four *uniformly spaced* levels spanning ``[-1, 1]`` only one table can carry both extremes
    (``[-1, -a, a, 1]`` forces ``a = 1/3``), so the uniform codebook table is necessarily symmetric —
    a one-sided grid such as the affine control's ``{0, 1/3, 2/3, 1}`` drops ``-1`` and its round
    trip is not idempotent for signed weights.
    """
    for spec in (NF_SPEC, UNIFORM_SPEC):
        levels = spec.levels()
        assert float(levels.abs().max()) == 1.0
        assert bool((levels == -1.0).any()) and bool((levels == 1.0).any())


def test_uniform_codebook_table_is_symmetric_about_zero() -> None:
    levels = uniform_int2_levels()
    torch.testing.assert_close(levels, -levels.flip(0), rtol=0.0, atol=0.0)


def test_reference_affine_control_grid_is_not_symmetric_about_zero() -> None:
    """The reference ``quant_uniform`` rounds onto ``k/3``, ``k = 0..3``: one-sided and asymmetric.

    The grid is recovered from the control's own outputs (``out = q·(alpha + 1e-8) + beta`` for
    ``q`` on that grid) rather than from :func:`uniform_int2_levels`, which is the *codebook* table.
    """
    weight = torch.randn(32, dtype=torch.float32, generator=torch.Generator().manual_seed(31))
    clip = (-4.0, 4.0)
    clamped = weight.clamp(clip[0], clip[1])
    alpha = clamped.max() - clamped.min()
    beta = clamped.min()
    control_grid = torch.arange(4, dtype=torch.float32) / 3.0
    assert not torch.allclose(control_grid, -control_grid.flip(0))
    out = fake_quantize_uniform_int2(weight, clip=clip)
    distances = (out.unsqueeze(-1) - (control_grid * (alpha + 1e-8) + beta)).abs()
    assert bool((distances.min(dim=-1).values == 0).all()), "outputs lie on the affine grid"


# --------------------------------------------------------------------------------------
# Hand-computed block
# --------------------------------------------------------------------------------------


def test_hand_computed_block_codes_and_dequantized_values() -> None:
    """Hand-computed 4-element block (``block_size=4``, so the whole row is one block).

    ``w = [-2.0, 1.8, 0.8, -0.1]``; block absmax ``= 2.0``; table ``= [-1.0, 0.0, a, 1.0]`` with
    ``a = 0.33791524171829224``; ``scale = absmax / max|level| = 2.0 / 1.0 = 2.0``, so
    ``w / scale = [-1.0, 0.9, 0.4, -0.05]`` and the nearest-level distances are

    ==========  ================================  =================
    element     |normalized - level|              code (level)
    ==========  ================================  =================
    0 (-1.0)    |0.0, 1.0,  1.3379, 2.0|         0 (-1.0) exact
    1 ( 0.9)    |1.9, 0.9,  0.5621, 0.1|         3 ( 1.0)
    2 ( 0.4)    |1.4, 0.4,  0.0621, 0.6|         2 ( a  )
    3 (-0.05)   |0.95, 0.05, 0.3879, 1.05|       1 ( 0.0)
    ==========  ================================  =================

    Dequantized ``= levels[codes] * 2.0 = [-2.0, 2.0, 0.6758304834365845, 0.0]``.
    """
    weight = torch.tensor([[-2.0, 1.8, 0.8, -0.1]])
    spec = CodebookSpec(2, "nf", block_size=4)

    codes = codebook_codes(weight, spec)
    assert codes.dtype == torch.int64
    torch.testing.assert_close(codes, torch.tensor([[0, 3, 2, 1]]))

    expected = torch.tensor([[-2.0, 2.0, 2.0 * 0.33791524171829224, 0.0]])
    torch.testing.assert_close(fake_quantize_codebook(weight, spec), expected, rtol=0.0, atol=1e-6)


def test_ties_resolve_to_the_lowest_level_index() -> None:
    """``|-0.5 - 0| == |-0.5 - (-1)|``: the reference's ``argmin`` picks the *lower* level."""
    weight = torch.tensor([[-1.0, -0.5]])
    spec = CodebookSpec(2, "nf", block_size=2)
    torch.testing.assert_close(codebook_codes(weight, spec), torch.tensor([[0, 0]]))
    torch.testing.assert_close(
        fake_quantize_codebook(weight, spec), torch.tensor([[-1.0, -1.0]]), rtol=0.0, atol=0.0
    )


def test_all_zero_block_dequantizes_to_exactly_zero() -> None:
    """A zero absmax would make the reference return ``NaN``; here the block collapses to 0."""
    weight = torch.zeros(1, 8)
    out = fake_quantize_codebook(weight, CodebookSpec(2, "nf", block_size=8))
    assert bool(torch.isfinite(out).all())
    torch.testing.assert_close(out, weight, rtol=0.0, atol=0.0)


# --------------------------------------------------------------------------------------
# Round trip, determinism, layout
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("spec", [NF_SPEC, UNIFORM_SPEC])
def test_round_trip_is_idempotent_and_values_are_levels_times_block_scale(
    spec: CodebookSpec,
) -> None:
    weight = _normal((6, 8), seed=3, sigma=1.7)
    quantized = fake_quantize_codebook(weight, spec)

    assert quantized.shape == weight.shape and quantized.dtype == weight.dtype
    assert torch.equal(quantized, fake_quantize_codebook(quantized, spec)), "must be idempotent"

    codes = codebook_codes(weight, spec)
    assert codes.shape == weight.shape
    assert int(codes.min()) >= 0 and int(codes.max()) < spec.n_levels
    element_scale = _block_absmax_like(weight, spec.block_size)
    expected = spec.levels()[codes] * element_scale
    assert torch.equal(quantized, expected), "output must be level * (block absmax / max|level|)"


@pytest.mark.parametrize("spec", [NF_SPEC, UNIFORM_SPEC])
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 5, 8, 13])
def test_round_trip_is_idempotent_on_every_seed(spec: CodebookSpec, seed: int) -> None:
    """Idempotence must not be seed-luck: it follows from both extremes being in the table.

    A one-sided table (such as the affine control's ``{0, 1/3, 2/3, 1}``) loses the negative extreme
    whenever a block's absmax element is negative, so its second pass rescales and changes the
    output. Both codebook tables here carry ``±1``.
    """
    weight = _normal((4, 16), seed=seed, sigma=0.5 * (seed + 1))
    quantized = fake_quantize_codebook(weight, spec)
    assert torch.equal(quantized, fake_quantize_codebook(quantized, spec))
    assert torch.equal(codebook_codes(quantized, spec), codebook_codes(weight, spec))


@pytest.mark.parametrize("spec", [NF_SPEC, UNIFORM_SPEC])
def test_two_calls_are_bit_identical(spec: CodebookSpec) -> None:
    weight = _normal((5, 13), seed=7)
    first = fake_quantize_codebook(weight, spec)
    second = fake_quantize_codebook(weight, spec)
    assert torch.equal(first, second)
    assert torch.equal(codebook_codes(weight, spec), codebook_codes(weight, spec))


def test_quantization_does_not_mutate_its_input() -> None:
    weight = _normal((4, 9), seed=11)
    before = weight.clone()
    fake_quantize_codebook(weight, NF_SPEC)
    assert torch.equal(weight, before)


def test_float64_input_is_quantized_in_float64() -> None:
    weight = _normal((4, 8), seed=5).to(torch.float64)
    quantized = fake_quantize_codebook(weight, CodebookSpec(2, "nf", block_size=8))
    assert quantized.dtype == torch.float64
    assert torch.equal(quantized, fake_quantize_codebook(quantized, CodebookSpec(2, "nf", 8)))


def test_trailing_short_block_is_accepted() -> None:
    """The reference's in-place ``resize`` requires divisibility; this module pads instead.

    10 elements with ``block_size=7`` give two blocks: ``[1..7]`` (absmax 7) and ``[8, 9, 10]``
    (absmax 10).
    """
    weight = torch.arange(1.0, 11.0).reshape(1, 10)
    spec = CodebookSpec(2, "nf", block_size=7)
    codes = codebook_codes(weight, spec)
    assert codes.shape == (1, 10)
    # Block 1: normalized [1/7 .. 7/7] = [0.1429, 0.2857, 0.4286, 0.5714, 0.7143, 0.8571, 1.0];
    # nearest levels of [-1, 0, a, 1] (a = 0.3379, half-gap 0.1690) are
    # 0 (|0.1429| < |0.1429 - a|), then a, a, a, 1, 1, 1.
    # Block 2: normalized [8, 9, 10] / 10 = [0.8, 0.9, 1.0] -> nearest 1.0 for all three.
    torch.testing.assert_close(codes[0, 7:], torch.tensor([3, 3, 3]))
    torch.testing.assert_close(codes[0, :7], torch.tensor([1, 2, 2, 2, 3, 3, 3]))


def test_axis_default_is_the_reference_row_major_flatten() -> None:
    weight = torch.arange(24, dtype=torch.float32).reshape(4, 6) / 7.0
    default = fake_quantize_codebook(weight, CodebookSpec(2, "nf", block_size=3))
    explicit_last = fake_quantize_codebook(weight, CodebookSpec(2, "nf", block_size=3, axis=-1))
    assert torch.equal(default, explicit_last)
    # axis=0 flattens the transposed view, so it must block differently (measured, not assumed).
    other = fake_quantize_codebook(weight, CodebookSpec(2, "nf", block_size=3, axis=0))
    assert not torch.equal(default, other)


def test_flatten_blocking_equals_per_row_blocking_when_the_row_length_divides() -> None:
    """For ``shape[-1] % block_size == 0`` a block never crosses a row (the frozen arm's regime)."""
    weight = _normal((8, 64), seed=13)
    whole = fake_quantize_codebook(weight, CodebookSpec(2, "nf", block_size=64))
    per_row = torch.cat(
        [
            fake_quantize_codebook(weight[i : i + 1], CodebookSpec(2, "nf", block_size=64))
            for i in range(weight.shape[0])
        ],
        dim=0,
    )
    assert torch.equal(whole, per_row)


def test_axis_out_of_range_is_rejected() -> None:
    weight = _normal((2, 3), seed=1)
    with pytest.raises(IndexError, match="axis 2 is out of range"):
        fake_quantize_codebook(weight, CodebookSpec(2, "nf", block_size=3, axis=2))


def test_non_float_and_degenerate_inputs_are_rejected() -> None:
    spec = CodebookSpec(2, "nf", block_size=4)
    with pytest.raises(TypeError, match="requires a float tensor"):
        fake_quantize_codebook(torch.arange(8), spec)
    with pytest.raises(ValueError, match="0-dim"):
        fake_quantize_codebook(torch.tensor(1.0), spec)
    with pytest.raises(ValueError, match="empty tensor"):
        fake_quantize_codebook(torch.zeros(0), spec)


# --------------------------------------------------------------------------------------
# Uniform-int2 control (reference ``quant_uniform``)
# --------------------------------------------------------------------------------------


def test_uniform_control_matches_the_reference_recipe_inline() -> None:
    """Re-derive the reference recipe in the test and require bit-exact agreement."""
    weight = _normal((4, 16), seed=17, sigma=3.0)
    clip = (-2.5, 2.5)

    def reference(x: torch.Tensor) -> torch.Tensor:
        y = torch.where(x < clip[1], x, torch.tensor(clip[1]))
        y = torch.where(y > clip[0], y, torch.tensor(clip[0]))
        alpha = y.max() - y.min()
        beta = y.min()
        normalized = (y - beta) / (alpha + 1e-8)
        quantized = torch.round(normalized * 3).div(3)
        return quantized * (alpha + 1e-8) + beta

    torch.testing.assert_close(
        fake_quantize_uniform_int2(weight, clip=clip), reference(weight), rtol=0.0, atol=0.0
    )


def test_uniform_control_default_clip_is_two_sigma() -> None:
    weight = _normal((8, 8), seed=19, sigma=2.0)
    mean = weight.mean()
    std = weight.std()
    torch.testing.assert_close(
        fake_quantize_uniform_int2(weight),
        fake_quantize_uniform_int2(weight, clip=(float(mean - 2 * std), float(mean + 2 * std))),
        rtol=1e-6,
        atol=1e-6,
    )


def test_uniform_control_outputs_lie_on_the_affine_grid() -> None:
    """Every output is ``q·(alpha + 1e-8) + beta`` for ``q`` on the reference's ``k/3`` grid."""
    weight = _normal((4, 16), seed=23, sigma=2.0)
    clip = (-3.0, 3.0)
    out = fake_quantize_uniform_int2(weight, clip=clip)
    clamped = weight.clamp(clip[0], clip[1])
    alpha = clamped.max() - clamped.min()
    beta = clamped.min()
    grid = torch.arange(4, dtype=torch.float32) / 3.0 * (alpha + 1e-8) + beta
    distances = (out.reshape(-1).unsqueeze(-1) - grid).abs()
    assert bool((distances.min(dim=-1).values == 0).all()), "every output is q*(alpha+eps)+beta"
    assert out.shape == weight.shape


def test_uniform_control_rejects_a_bad_clip() -> None:
    weight = _normal((2, 4), seed=29)
    with pytest.raises(ValueError, match="low <= high"):
        fake_quantize_uniform_int2(weight, clip=(3.0, -3.0))
    with pytest.raises(ValueError, match="low <= high"):
        fake_quantize_uniform_int2(weight, clip=(1.0,))  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# Byte accounting (class 1)
# --------------------------------------------------------------------------------------


def test_accounting_of_the_frozen_768_matrix() -> None:
    """768x768 at bits=2, block 64: 147456 B payload + 9216 fp32 scales = 184320 B (class 1)."""
    numel = 768 * 768
    payload = math.ceil(numel * 2 / 8)
    scales = numel // 64
    assert (payload, scales) == (147456, 9216)
    for spec in (NF_SPEC, UNIFORM_SPEC):
        assert codebook_accounted_bytes((768, 768), spec) == payload + 4 * scales == 184320


def test_accounting_delegates_the_payload_and_block_count_to_the_central_helper() -> None:
    equivalence = QuantSpec(bits=2, granularity="per_group", group_size=64, symmetric=True)
    for shape in ((768, 768), (576, 576), (1536, 576), (7,)):
        numel = math.prod(shape)
        breakdown = byte_breakdown((numel,), equivalence)
        assert breakdown.payload_bytes == math.ceil(numel * 2 / 8)
        assert breakdown.n_blocks == math.ceil(numel / 64)
        assert codebook_accounted_bytes(shape, NF_SPEC) == (
            breakdown.payload_bytes + CODEBOOK_SCALE_BYTES * breakdown.n_blocks
        )


def test_accounting_ignores_axis_and_kind_but_tracks_block_size() -> None:
    assert codebook_accounted_bytes((768, 768), NF_SPEC) == codebook_accounted_bytes(
        (768, 768), CodebookSpec(2, "uniform", block_size=64, axis=0)
    )
    # Halving the block size doubles the scale count.
    coarse = codebook_accounted_bytes((768, 768), NF_SPEC)
    fine = codebook_accounted_bytes((768, 768), CodebookSpec(2, "nf", block_size=32))
    assert fine == 147456 + 4 * 18432
    assert fine > coarse


@pytest.mark.parametrize("shape", [(), (0, 4), (4, -1)])
def test_accounting_rejects_degenerate_shapes(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="shape must"):
        codebook_accounted_bytes(shape, NF_SPEC)


# --------------------------------------------------------------------------------------
# Oracle fixture (pinned reference capture)
# --------------------------------------------------------------------------------------


def test_oracle_fixture_levels() -> None:
    fixture = _oracle()
    tolerance = fixture["tolerance"]["absolute_per_element"]
    torch.testing.assert_close(
        nf_levels(2), _f32(fixture["outputs"]["nf2_levels"]), rtol=0.0, atol=tolerance
    )


@pytest.mark.parametrize(
    ("input_key", "output_key"),
    [("small_8x8", "nf2_block64_8x8"), ("big_16x16", "nf2_block64_16x16")],
)
def test_oracle_fixture_nf_block_outputs(input_key: str, output_key: str) -> None:
    """``quant_nf4_block(weight, block_size=64, num_bits=2)`` on the captured inputs.

    The reference flattens the weight in row-major order, so a 64-element 8x8 weight is **one**
    64-element block and a 256-element 16x16 weight is four. The capture is float32 throughout
    (``provenance.torch == "2.14.1+cpu"``), so the inputs are cast to float32 before quantizing —
    our float64 path reproduces the same 2-bit decisions but differs from the fixture by ~1e-8 in
    the last float32 ulp.
    """
    fixture = _oracle()
    tolerance = fixture["tolerance"]["absolute_per_element"]
    side = 8 if input_key == "small_8x8" else 16
    weight = _f32(fixture["inputs"][input_key]).reshape(side, side)
    expected = _f32(fixture["outputs"][output_key]).reshape(side, side)
    torch.testing.assert_close(
        fake_quantize_codebook(weight, NF_SPEC), expected, rtol=0.0, atol=tolerance
    )


def test_oracle_fixture_uniform_control() -> None:
    """``quant_uniform(input, num_bits=2, clip_val=[mean-2*std, mean+2*std])``."""
    fixture = _oracle()
    tolerance = fixture["tolerance"]["absolute_per_element"]
    weight = _f32(fixture["inputs"]["small_8x8"]).reshape(8, 8)
    clip = tuple(float(value) for value in fixture["inputs"]["clip_val_8x8"])
    torch.testing.assert_close(
        fake_quantize_uniform_int2(weight, clip=clip),
        _f32(fixture["outputs"]["uniform_int2_8x8"]).reshape(8, 8),
        rtol=0.0,
        atol=tolerance,
    )


def test_oracle_fixture_provenance_pins_the_reference() -> None:
    """The fixture must name the pinned commit and state that no third-party source is committed."""
    fixture = _oracle()
    reference = fixture["reference"]
    assert reference["commit"] == "ae33fd4fd05fd4ba146555cd77c13d307eb4e9b3"
    assert reference["license"] == "MIT"
    assert reference["path"] == "glue/utils_qaunt.py"
    assert "OUTSIDE" in fixture["provenance"]["how"]
    assert NF_OFFSET == 0.9677083


def test_levels_match_the_oracle_construction_not_a_symmetric_midpoint_table() -> None:
    """Guard against the plausible-but-wrong symmetric "quantile midpoints" reading.

    A symmetric midpoint table at 2 bits would be ``[-1.0, -0.2521..., 0.2521..., 1.0]``; the
    reference is the asymmetric QLoRA default. The fixture's own numbers settle which one is right.
    """
    fixture = _oracle()
    fixture_levels = _f32(fixture["outputs"]["nf2_levels"])
    assert not torch.allclose(fixture_levels, -fixture_levels.flip(0))
    assert float(fixture_levels[1]) == 0.0
    torch.testing.assert_close(nf_levels(2), fixture_levels, rtol=0.0, atol=1e-10)
