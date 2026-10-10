"""LoftQ alternating initialization: hand-computed schedule, alternation, limitations.

The schedule under test is the paper's (Li et al., arXiv:2310.08659, Eq. 7-8), transcribed here so
the test states the semantics rather than the implementation:

.. code-block:: text

    A_0, B_0 <- SVD_rank(W)
    for t = 1 .. T:
        Q_t   = q_N(W - B_{t-1} @ A_{t-1})
        A_t, B_t <- SVD_rank(W - Q_t)

All fixtures are CPU float32/float64, no download, no GPU.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch.testing import assert_close

from spectraquant.factorization.initialization import initialize_svd
from spectraquant.factorization.loftq import loftq_initialise
from spectraquant.quantization import QuantSpec, fake_quantize

#: 2-bit symmetric per-tensor quantization (``qmin=-2, qmax=1``, so ``scale = max|W|``).
SPEC_2BIT = QuantSpec(2, "per_tensor", None, True)
#: 4-bit asymmetric per-group quantization along the output axis.
SPEC_4BIT_GROUP = QuantSpec(4, "per_group", 4, False, axis=0)

#: Hand-built 2x2 weight whose SVD is exact by construction:
#: ``W = 1.0 * u1 v1ᵀ + 0.5 * u2 v2ᵀ`` with ``u1 = v1 = (3, 1)/sqrt(10)`` and
#: ``u2 = v2 = (1, -3)/sqrt(10)`` (both orthonormal pairs), i.e.
#: ``W = (1/10)[[9.5, 1.5], [1.5, 5.5]]``.
TOY_W = torch.tensor([[0.95, 0.15], [0.15, 0.55]], dtype=torch.float32)


def _aligned(t: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    """Flip ``t`` by ``±1`` to match ``expected``'s leading sign (SVD sign ambiguity)."""
    idx = int(torch.argmax(expected.flatten().abs()))
    return t * torch.sign(t.flatten()[idx] * expected.flatten()[idx])


def _companion_objective(
    w: torch.Tensor, a: torch.Tensor, b: torch.Tensor, spec: QuantSpec
) -> float:
    """Paper Eq. 6 objective ``||W - Q - B @ A||_F`` at a self-consistent pair.

    The companion base ``Q = fake_quantize(W - B @ A, spec)`` is the one a caller reconstructs from
    the returned two-tensor result (module docstring), so this is the reconstruction error of the
    pair ``(Q, A, B)`` against ``w``.
    """
    q = fake_quantize(w - b @ a, spec)
    return float(torch.linalg.matrix_norm(w - (q + b @ a), ord="fro"))


# --------------------------------------------------------------------------------------
# Hand-computed toy: one iteration, every step by hand
# --------------------------------------------------------------------------------------


def test_loftq_hand_computed_one_iteration_matches_the_closed_form() -> None:
    """Every step of the first iteration is derived by hand and asserted explicitly.

    The schedule starts from ``A_0 = B_0 = 0`` (paper Algorithm 1, amendment A-0014), so the first
    iteration is: quantize ``W`` itself, then take the rank-1 SVD of what the quantization left.

    Step 1 (Eq. 7) — quantize ``W = [[0.95, 0.15], [0.15, 0.55]]`` with 2-bit symmetric per-tensor:
        ``scale = max|W| = 0.95``; ``W / scale = [[1, 0.1579], [0.1579, 0.5789]]``; nearest rounding
        of ``0.1579`` and ``0.5789`` gives codes ``[[1, 0], [0, 1]]`` (both are far from the ``1/2``
        boundary, so this is insensitive to float32 round-off), hence ``Q = [[0.95, 0], [0, 0.95]]``.

    Step 2 (Eq. 8 input) — the residual of the quantization:
        ``R = W - Q = [[0, 0.15], [0.15, -0.40]]``, symmetric with ``trace = -0.40`` and
        ``det = -0.0225``, so its eigenvalues are ``(-0.40 ± sqrt(0.16 + 0.09)) / 2 = 0.05`` and
        ``-0.45``. The dominant *singular* direction is the one of largest magnitude, ``λ1 = -0.45``,
        with unit eigenvector ``q = (1, -3) / sqrt(10)`` (from ``(R - λ1 I) q = 0``:
        ``[[0.45, 0.15], [0.15, 0.05]] q = 0``).

    Step 3 — the rank-1 truncation is therefore ``B1 @ A1 = λ1 q qᵀ = (-0.45/10) [[1, -3], [-3, 9]]
    = [[-0.045, 0.135], [0.135, -0.405]]``, and the leftover is the *other* eigenpair,
    ``λ2 q2 q2ᵀ`` with ``λ2 = 0.05`` and ``q2 = (3, 1) / sqrt(10)``:
    ``(0.05/10) [[9, 3], [3, 1]] = [[0.045, 0.015], [0.015, 0.005]]``.
    """
    # Step 1: the quantized base, hand-computed from the scale and the two codes.
    quantized = torch.tensor([[0.95, 0.0], [0.0, 0.95]], dtype=torch.float32)
    assert_close(fake_quantize(TOY_W, SPEC_2BIT), quantized, atol=1e-6, rtol=0)

    # Step 2: the residual of the quantization.
    residual = torch.tensor([[0.0, 0.15], [0.15, -0.40]], dtype=torch.float32)
    assert_close(TOY_W - quantized, residual, atol=1e-6, rtol=0)

    # Step 3: the rank-1 truncation of the residual, in closed form.
    q = torch.tensor([1.0, -3.0]) / math.sqrt(10.0)
    lambda1 = -0.45
    expected_product = lambda1 * torch.outer(q, q)
    svd_of_residual = initialize_svd(residual, 1)
    assert_close(svd_of_residual.B @ svd_of_residual.A, expected_product, atol=1e-6, rtol=0)

    # End to end: `iterations=1` performs exactly those steps.
    a, b = loftq_initialise(TOY_W, rank=1, spec=SPEC_2BIT, iterations=1)
    assert_close(_aligned(a, svd_of_residual.A), svd_of_residual.A, atol=1e-6, rtol=0)
    assert_close(_aligned(b, svd_of_residual.B), svd_of_residual.B, atol=1e-6, rtol=0)

    # The pair (Q, A, B) reconstructs W up to the discarded second spectral direction of the residual.
    q2 = torch.tensor([3.0, 1.0]) / math.sqrt(10.0)
    lambda2 = 0.05
    assert_close(TOY_W - (quantized + b @ a), lambda2 * torch.outer(q2, q2), atol=1e-6, rtol=0)
    # ... and that leftover is strictly smaller than the quantization error the factors compensate.
    assert lambda2 < float(torch.linalg.matrix_norm(TOY_W - quantized, ord="fro"))


# --------------------------------------------------------------------------------------
# The schedule alternates
# --------------------------------------------------------------------------------------


def test_one_iteration_is_not_the_svd_initialization() -> None:
    """The quantization half-step changes the factors, so ``iterations=1`` is not plain SVD."""
    w = torch.randn(6, 4, generator=torch.Generator().manual_seed(0))
    init = initialize_svd(w, 2)
    a, b = loftq_initialise(w, rank=2, spec=SPEC_2BIT, iterations=1)

    assert not torch.equal(b @ a, init.B @ init.A)
    # The quantization half-step moved the factors by an amount set by the quantizer, not by noise.
    gap = float(torch.linalg.matrix_norm((b @ a) - (init.B @ init.A), ord="fro"))
    assert gap > 1e-3


def test_two_iterations_differ_from_one_when_the_second_step_changes_the_factors() -> None:
    """``iterations=2`` really runs a second quantize+SVD cycle (not a no-op repeat)."""
    w = torch.randn(6, 4, generator=torch.Generator().manual_seed(0))
    a1, b1 = loftq_initialise(w, rank=2, spec=SPEC_2BIT, iterations=1)
    a2, b2 = loftq_initialise(w, rank=2, spec=SPEC_2BIT, iterations=2)

    assert not torch.equal(b2 @ a2, b1 @ a1)
    gap = float(torch.linalg.matrix_norm((b2 @ a2) - (b1 @ a1), ord="fro"))
    assert gap > 1e-3  # observed 3.6e-2 for this fixture
    # ... and the third iteration moves it again, so the loop is a loop and not a clamp.
    a3, b3 = loftq_initialise(w, rank=2, spec=SPEC_2BIT, iterations=3)
    assert float(torch.linalg.matrix_norm((b3 @ a3) - (b2 @ a2), ord="fro")) > 1e-3


def test_iterations_count_is_enforced() -> None:
    """Zero is rejected: one step is the first residual step, and "no LoftQ" is not an input."""
    for bad in (0, -1, -5):
        with pytest.raises(ValueError, match="iterations must be >= 1"):
            loftq_initialise(TOY_W, rank=1, spec=SPEC_2BIT, iterations=bad)
    for bad_type in (1.0, True, "2"):  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="iterations must be an int"):
            loftq_initialise(TOY_W, rank=1, spec=SPEC_2BIT, iterations=bad_type)  # type: ignore[arg-type]


def test_the_paper_schedule_starts_from_zero_and_the_sequence_is_recorded_not_asserted() -> None:
    """The first step is the QLoRA weight plus the residual SVD; later steps are *recorded*.

    The schedule starts from ``A_0 = B_0 = 0`` (amendment A-0014), so ``iterations=1`` must equal the
    hand-checkable pair ``Q = fake_quantize(W)`` with factors ``SVD(W - Q)`` - that identity is
    asserted. Whether the alternating sequence improves the self-consistent objective
    ``||W - Q - B @ A||_F`` with ``Q = fake_quantize(W - B @ A)`` is *measured and printed*, not
    asserted: the discrete quantizer re-estimates the scale from a shrinking residual at every step,
    so monotonicity is not guaranteed by anything this module claims.
    """
    w = torch.randn(6, 4, generator=torch.Generator().manual_seed(0))

    # The asserted identity: the first step is quantize(W) plus the SVD of its residual.
    a1, b1 = loftq_initialise(w, rank=2, spec=SPEC_2BIT, iterations=1)
    reference = initialize_svd(w - fake_quantize(w, SPEC_2BIT), 2)
    assert_close(_aligned(a1, reference.A), reference.A, atol=1e-6, rtol=0)
    assert_close(_aligned(b1, reference.B), reference.B, atol=1e-6, rtol=0)

    two_bit = [
        _companion_objective(
            w, *loftq_initialise(w, rank=2, spec=SPEC_2BIT, iterations=t), SPEC_2BIT
        )
        for t in (1, 2, 3)
    ]
    group = [
        _companion_objective(
            w, *loftq_initialise(w, rank=2, spec=SPEC_4BIT_GROUP, iterations=t), SPEC_4BIT_GROUP
        )
        for t in (1, 2, 3)
    ]
    print(f"2-bit per-tensor objective at T=1,2,3: {two_bit}")
    print(f"4-bit per-group  objective at T=1,2,3: {group}")

    # Asserted: the sequence is finite and non-negative, and the schedule is deterministic.
    assert all(math.isfinite(value) and value >= 0.0 for value in two_bit + group)
    again = loftq_initialise(w, rank=2, spec=SPEC_2BIT, iterations=2)
    assert torch.equal(again[0], loftq_initialise(w, rank=2, spec=SPEC_2BIT, iterations=2)[0])
    assert torch.equal(again[1], loftq_initialise(w, rank=2, spec=SPEC_2BIT, iterations=2)[1])


# --------------------------------------------------------------------------------------
# Shapes, dtypes, determinism, edge cases
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("iterations", [1, 3])
def test_shapes_dtype_and_determinism(dtype: torch.dtype, iterations: int) -> None:
    w = torch.randn(5, 3, generator=torch.Generator().manual_seed(11)).to(dtype)
    first = loftq_initialise(w, rank=2, spec=SPEC_2BIT, iterations=iterations)
    second = loftq_initialise(w, rank=2, spec=SPEC_2BIT, iterations=iterations)

    a, b = first
    assert a.shape == (2, 3) and b.shape == (5, 2)
    assert a.dtype is dtype and b.dtype is dtype
    assert a.device.type == "cpu" and b.device.type == "cpu"
    assert torch.equal(a, second[0]) and torch.equal(b, second[1])


def test_rank_is_clamped_like_the_rest_of_the_package() -> None:
    """A rank above ``min(shape)`` means "keep everything" (documented clamping), not an error."""
    w = torch.randn(5, 3, generator=torch.Generator().manual_seed(12))
    a, b = loftq_initialise(w, rank=9, spec=SPEC_2BIT, iterations=1)

    assert a.shape == (3, 3) and b.shape == (5, 3)


def test_rank_zero_returns_empty_factors() -> None:
    w = torch.randn(4, 2, generator=torch.Generator().manual_seed(13))
    a, b = loftq_initialise(w, rank=0, spec=SPEC_2BIT, iterations=1)

    assert a.shape == (0, 2) and b.shape == (4, 0)
    assert (b @ a).shape == (4, 2)
    assert torch.count_nonzero(b @ a) == 0


def test_returned_factors_are_detached_constants() -> None:
    """Documented: the schedule runs under ``no_grad`` and returns constants to train."""
    w = torch.randn(5, 3, generator=torch.Generator().manual_seed(14), requires_grad=True)
    a, b = loftq_initialise(w, rank=2, spec=SPEC_2BIT, iterations=2)

    assert not a.requires_grad and not b.requires_grad


def test_rejects_non_matrix_unsupported_dtype_and_negative_rank() -> None:
    with pytest.raises(ValueError, match="must be 2-D"):
        loftq_initialise(torch.randn(4), rank=1, spec=SPEC_2BIT)
    with pytest.raises(ValueError, match="dtype"):
        loftq_initialise(torch.randn(4, 2).half(), rank=1, spec=SPEC_2BIT)
    # The rank is validated before the zero start is built, so a negative rank raises the module's
    # own ValueError rather than a torch RuntimeError from `torch.zeros(-1, ...)`.
    with pytest.raises(ValueError, match="rank must be an int >= 0"):
        loftq_initialise(TOY_W, rank=-1, spec=SPEC_2BIT)
    with pytest.raises(ValueError, match="rank must be an int >= 0"):
        loftq_initialise(TOY_W, rank=1.5, spec=SPEC_2BIT)  # type: ignore[arg-type]
