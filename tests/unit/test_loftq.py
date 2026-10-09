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
    """Every step of the first iteration is computed by hand and asserted explicitly.

    Step 1 (paper Eq. 7 input) — rank-1 SVD of ``W``:
        ``σ1 = 1.0`` with ``u1 = v1 = (3, 1)/sqrt(10)`` (exact by construction of ``TOY_W``), so
        ``A_0 = σ1 v1ᵀ = [0.9486832980505138, 0.31622776601683794]`` and
        ``B_0 = u1`` as a column, and ``B_0 @ A_0 = (1/10)[[9, 3], [3, 1]] = [[0.9, 0.3], [0.3, 0.1]]``.
        Residual ``R = W - B_0 @ A_0 = (0.5/10)[[1, -3], [-3, 9]] = [[0.05, -0.15], [-0.15, 0.45]]``.

    Step 2 (Eq. 7) — quantize ``R`` with 2-bit symmetric per-tensor (``qmax = 1``):
        ``scale = max|R| = 0.45``; ``R / scale = [[1/9, -1/3], [-1/3, 1]]``; nearest rounding gives
        codes ``[[0, 0], [0, 1]]`` (no tie: ``1/9`` and ``1/3`` are far from ``1/2``, so this is
        insensitive to float32 round-off), hence ``Q = [[0, 0], [0, 0.45]]``.

    Step 3 (Eq. 8) — SVD of the residual of the quantization ``Rq = W - Q = [[0.95, 0.15],
    [0.15, 0.10]]``. ``Rq`` is symmetric positive definite with ``trace = 1.05`` and
    ``det = 0.095 - 0.0225 = 0.0725``, so its eigenvalues are ``(1.05 ± sqrt(0.8125)) / 2``; the
    larger, ``σ1 = 0.9756939094329986``, supplies the rank-1 truncation. Its unit eigenvector is
    ``q = (1, c)/sqrt(1 + c²)`` with ``c = (σ1 - 0.95)/0.15 = (sqrt(0.8125) - 0.85)/0.3 =
    0.17129273...``, giving ``A_1 = σ1 qᵀ``, ``B_1 = q`` and the hand-computed product
    ``B_1 @ A_1 = σ1 q qᵀ`` asserted below.
    """
    # Step 1: the rank-1 SVD initialization is the closed form above.
    factors = initialize_svd(TOY_W, 1)
    expected_a0 = torch.tensor([[0.9486832980505138, 0.31622776601683794]])
    expected_b0 = torch.tensor([[0.9486832980505138], [0.31622776601683794]])
    expected_b0a0 = torch.tensor([[0.9, 0.3], [0.3, 0.1]])
    assert_close(_aligned(factors.A, expected_a0), expected_a0, atol=1e-6, rtol=0)
    assert_close(_aligned(factors.B, expected_b0), expected_b0, atol=1e-6, rtol=0)
    assert_close(factors.B @ factors.A, expected_b0a0, atol=1e-6, rtol=0)

    # Step 2: the residual and its hand-computed quantization.
    residual = torch.tensor([[0.05, -0.15], [-0.15, 0.45]], dtype=torch.float32)
    assert_close(TOY_W - expected_b0a0, residual, atol=1e-6, rtol=0)
    quantized = torch.tensor([[0.0, 0.0], [0.0, 0.45]], dtype=torch.float32)
    assert_close(fake_quantize(residual, SPEC_2BIT), quantized, atol=1e-6, rtol=0)

    # Step 3: SVD of the residual of the quantization (closed form above).
    quantization_residual = TOY_W - quantized
    assert_close(
        quantization_residual, torch.tensor([[0.95, 0.15], [0.15, 0.10]]), atol=1e-6, rtol=0
    )
    sigma1 = (1.05 + math.sqrt(0.8125)) / 2.0
    c = (math.sqrt(0.8125) - 0.85) / 0.3
    norm = math.sqrt(1.0 + c * c)
    q = torch.tensor([1.0 / norm, c / norm])
    expected_a1 = (sigma1 * q).reshape(1, 2)
    expected_b1 = q.reshape(2, 1)
    expected_product = sigma1 * torch.outer(q, q)

    svd_of_residual = initialize_svd(quantization_residual, 1)
    assert_close(svd_of_residual.B @ svd_of_residual.A, expected_product, atol=1e-6, rtol=0)

    # End to end: `iterations=1` performs exactly the three steps above.
    a, b = loftq_initialise(TOY_W, rank=1, spec=SPEC_2BIT, iterations=1)
    assert_close(_aligned(a, expected_a1), expected_a1, atol=1e-6, rtol=0)
    assert_close(_aligned(b, expected_b1), expected_b1, atol=1e-6, rtol=0)
    assert_close(b @ a, expected_product, atol=1e-6, rtol=0)

    # The returned factors are handed the *quantization* residual Rq = W - Q, so the pair (Q, A, B)
    # with the hand-computed Q reconstructs W up to the discarded second spectral direction of Rq:
    # W - Q - B @ A = λ2 * q2 q2ᵀ with q2 ⊥ q the other eigenvector of the symmetric 2x2 Rq.
    q2 = torch.tensor([q[1].item(), -q[0].item()])
    lambda2 = (1.05 - math.sqrt(0.8125)) / 2.0
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


def test_first_step_improves_the_loftq_objective_and_later_steps_are_not_guaranteed_to() -> None:
    """Documented observation: the schedule is *not* proven monotone in the Eq. 6 objective.

    Measured on this fixture (6x4 standard normal, ``seed=0``, rank 2, 2-bit symmetric per-tensor),
    with the objective ``||W - Q - B @ A||_F`` at the self-consistent pair
    ``Q = fake_quantize(W - B @ A)``:

    * plain SVD initialization: 0.95916
    * ``iterations=1``:          0.48682   (a large first-step improvement)
    * ``iterations=2``:          0.52730   (increases again)
    * ``iterations=3``:          0.55369

    and for the 4-bit asymmetric per-group spec (same fixture) the sequence is monotone decreasing:
    0.05367 -> 0.02349 -> 0.02285 -> 0.02272. Both regimes are asserted as measured: the paper
    minimizes Eq. 6 by alternating, but the discrete quantizer means a step can move the
    *self-consistent* objective either way, and nothing in this module claims otherwise.
    """
    w = torch.randn(6, 4, generator=torch.Generator().manual_seed(0))
    init = initialize_svd(w, 2)
    objectives = [
        _companion_objective(
            w, *loftq_initialise(w, rank=2, spec=SPEC_2BIT, iterations=t), SPEC_2BIT
        )
        for t in (1, 2, 3)
    ]
    init_objective = _companion_objective(w, init.A, init.B, SPEC_2BIT)

    assert objectives[0] < init_objective  # first residual step helps
    assert objectives[1] > objectives[0]  # observed: the second step raises it again
    assert objectives[2] > objectives[1]

    group_objectives = [
        _companion_objective(
            w, *loftq_initialise(w, rank=2, spec=SPEC_4BIT_GROUP, iterations=t), SPEC_4BIT_GROUP
        )
        for t in (1, 2, 3)
    ]
    assert group_objectives[0] > group_objectives[1] > group_objectives[2]  # monotone here


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
    with pytest.raises(ValueError, match="rank must be >= 0"):
        loftq_initialise(TOY_W, rank=-1, spec=SPEC_2BIT)
    with pytest.raises(TypeError, match="rank must be an int"):
        loftq_initialise(TOY_W, rank=1.5, spec=SPEC_2BIT)  # type: ignore[arg-type]
