"""LR-QAT pair: hand-computed initialization, residual identity, effective weight.

The mechanism under test (Bondarenko et al., arXiv:2406.06385; upstream
``Qualcomm-AI-research/LR-QAT`` @ ``8795afe0…``): a quantized main weight ``Wq`` plus a trainable
low-rank auxiliary weight ``B @ A`` added during training, initialised from the quantization error
``W - Wq`` by a rank-``rank`` SVD. All fixtures are CPU float32/float64, no download, no GPU.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch.testing import assert_close

from spectraquant.factorization.lr_qat import effective_weight, lr_qat_pair
from spectraquant.quantization import QuantSpec

#: 2-bit symmetric per-tensor quantization (``qmin=-2, qmax=1``, so ``scale = max|W|``).
SPEC_2BIT = QuantSpec(2, "per_tensor", None, True)
#: 4-bit asymmetric per-group quantization along the output axis.
SPEC_4BIT_GROUP = QuantSpec(4, "per_group", 4, False, axis=0)

#: Hand-built 2x2 weight (same fixture as ``test_loftq.py``): symmetric positive definite with
#: eigenvalues 1.0 and 0.5, so every closed form below is exact arithmetic.
TOY_W = torch.tensor([[0.95, 0.15], [0.15, 0.55]], dtype=torch.float32)


def _aligned(t: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    """Flip ``t`` by ``±1`` to match ``expected``'s leading sign (SVD sign ambiguity)."""
    idx = int(torch.argmax(expected.flatten().abs()))
    return t * torch.sign(t.flatten()[idx] * expected.flatten()[idx])


# --------------------------------------------------------------------------------------
# Hand-computed toy
# --------------------------------------------------------------------------------------


def test_lr_qat_pair_hand_computed_2x2() -> None:
    """``Wq``, the residual, the SVD and the reconstructed pair are all hand-computed.

    * Quantization of ``TOY_W`` with 2-bit symmetric per-tensor: the block scale is
      ``max|W| = 0.95`` (``qmax = 1``), ``W / 0.95 = [[1, 0.15789], [0.15789, 0.57895]]`` rounds to
      codes ``[[1, 0], [0, 1]]`` (no tie: ``0.15789`` is far from ``1/2``), so
      ``Wq = [[0.95, 0], [0, 0.95]]`` and the quantization error is
      ``E = W - Wq = [[0, 0.15], [0.15, -0.4]]``.
    * ``E`` is symmetric with ``trace = -0.4`` and ``det = -0.0225``, so its eigenvalues are
      ``(-0.4 ± sqrt(0.16 + 0.09)) / 2 = 0.05`` and ``-0.45``; its singular values are therefore
      ``0.45`` (eigenvector ``(1, -3)/sqrt(10)``) and ``0.05`` (eigenvector ``(3, 1)/sqrt(10)``).
    * The rank-1 truncation is the ``0.45`` term with the sign absorbed into the left factor:
      ``B @ A = -0.45 * (1/10) [[1, -3], [-3, 9]] = [[-0.045, 0.135], [0.135, -0.405]]``.
    * ``effective_weight = Wq + B @ A = [[0.905, 0.135], [0.135, 0.545]]``, and the residual norms
      are ``||W - Wq||_F = sqrt(0.205) = 0.4527692...`` and ``||W - (Wq + B @ A)||_F = 0.05`` — the
      discarded singular value, exactly as Eckart-Young requires.
    """
    wq, (a, b) = lr_qat_pair(TOY_W, rank=1, spec=SPEC_2BIT)

    expected_wq = torch.tensor([[0.95, 0.0], [0.0, 0.95]])
    assert_close(wq, expected_wq, atol=1e-6, rtol=0)

    residual = torch.tensor([[0.0, 0.15], [0.15, -0.4]])
    assert_close(TOY_W - wq, residual, atol=1e-6, rtol=0)

    # ``q1 = (1, -3)/sqrt(10)``; the negative eigenvalue becomes a sign on the left factor, so
    # ``B = q1`` and ``A = -0.45 * q1ᵀ`` with the convention ``B @ A = -0.45 q1 q1ᵀ``.
    q1 = torch.tensor([1.0, -3.0]) / math.sqrt(10.0)
    expected_b = q1.reshape(2, 1)
    expected_a = -0.45 * q1.reshape(1, 2)
    assert_close(_aligned(b, expected_b), expected_b, atol=1e-6, rtol=0)
    assert_close(_aligned(a, expected_a), expected_a, atol=1e-6, rtol=0)
    expected_product = torch.tensor([[-0.045, 0.135], [0.135, -0.405]])
    assert_close(b @ a, expected_product, atol=1e-6, rtol=0)

    effective = effective_weight(wq, a, b)
    expected_effective = torch.tensor([[0.905, 0.135], [0.135, 0.545]])
    assert_close(effective, expected_effective, atol=1e-6, rtol=0)
    assert_close(
        torch.tensor(float(torch.linalg.matrix_norm(TOY_W - effective, ord="fro"))),
        torch.tensor(0.05),
        atol=1e-6,
        rtol=0,
    )


# --------------------------------------------------------------------------------------
# The residual identity (the reason the auxiliary weight exists)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("spec", [SPEC_2BIT, SPEC_4BIT_GROUP])
@pytest.mark.parametrize("rank", [1, 2])
def test_residual_identity_is_strict_and_equals_the_discarded_spectrum(
    spec: QuantSpec, rank: int
) -> None:
    """``||W - (Wq + B@A)|| < ||W - Wq||`` strictly, and equals the discarded singular values."""
    w = torch.randn(8, 6, generator=torch.Generator().manual_seed(3))
    wq, (a, b) = lr_qat_pair(w, rank=rank, spec=spec)

    before = float(torch.linalg.matrix_norm(w - wq, ord="fro"))
    after = float(torch.linalg.matrix_norm(w - (wq + b @ a), ord="fro"))
    assert before > 0.0
    assert after < before

    # Eckart-Young: the residual after compensation is the Euclidean norm of the discarded
    # singular values of the quantization error (an independent oracle, not a re-run of the module).
    discarded = torch.linalg.svdvals(w - wq)[rank:]
    assert_close(
        torch.tensor(after), torch.linalg.vector_norm(discarded, ord=2), atol=1e-5, rtol=1e-5
    )


def test_rank_zero_leaves_the_quantized_weight_uncompensated() -> None:
    w = torch.randn(6, 4, generator=torch.Generator().manual_seed(4))
    wq, (a, b) = lr_qat_pair(w, rank=0, spec=SPEC_2BIT)

    assert a.shape == (0, 4) and b.shape == (6, 0)
    assert torch.count_nonzero(b @ a) == 0
    assert_close(effective_weight(wq, a, b), wq, atol=0.0, rtol=0)


def test_returned_tensors_are_detached_constants() -> None:
    """Documented: the initializer runs under ``no_grad`` and returns constants to train."""
    w = torch.randn(6, 4, generator=torch.Generator().manual_seed(8), requires_grad=True)
    wq, (a, b) = lr_qat_pair(w, rank=2, spec=SPEC_2BIT)

    assert not wq.requires_grad and not a.requires_grad and not b.requires_grad


# --------------------------------------------------------------------------------------
# Shapes, dtypes, determinism, validation
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_shapes_dtype_and_determinism(dtype: torch.dtype) -> None:
    w = torch.randn(7, 5, generator=torch.Generator().manual_seed(5)).to(dtype)
    wq, (a, b) = lr_qat_pair(w, rank=3, spec=SPEC_2BIT)
    wq2, (a2, b2) = lr_qat_pair(w, rank=3, spec=SPEC_2BIT)

    assert wq.shape == (7, 5) and a.shape == (3, 5) and b.shape == (7, 3)
    assert wq.dtype is dtype and a.dtype is dtype and b.dtype is dtype
    assert wq.device.type == "cpu" and a.device.type == "cpu" and b.device.type == "cpu"
    assert torch.equal(wq, wq2) and torch.equal(a, a2) and torch.equal(b, b2)


def test_rank_is_clamped_like_the_rest_of_the_package() -> None:
    w = torch.randn(7, 5, generator=torch.Generator().manual_seed(6))
    _, (a, b) = lr_qat_pair(w, rank=99, spec=SPEC_2BIT)

    assert a.shape == (5, 5) and b.shape == (7, 5)


def test_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="must be 2-D"):
        lr_qat_pair(torch.randn(4), rank=1, spec=SPEC_2BIT)
    with pytest.raises(ValueError, match="dtype"):
        lr_qat_pair(TOY_W.half(), rank=1, spec=SPEC_2BIT)
    with pytest.raises(ValueError, match="rank must be >= 0"):
        lr_qat_pair(TOY_W, rank=-1, spec=SPEC_2BIT)
    with pytest.raises(TypeError, match="rank must be an int"):
        lr_qat_pair(TOY_W, rank=1.5, spec=SPEC_2BIT)  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# effective_weight
# --------------------------------------------------------------------------------------


def test_effective_weight_is_the_sum_and_is_differentiable() -> None:
    """``effective_weight`` must be the trained representation: ``wq + b @ a``, with gradients."""
    w = torch.randn(6, 4, generator=torch.Generator().manual_seed(7), requires_grad=False)
    wq, (a, b) = lr_qat_pair(w, rank=2, spec=SPEC_2BIT)
    a = a.clone().requires_grad_(True)
    b = b.clone().requires_grad_(True)

    effective = effective_weight(wq, a, b)
    assert_close(effective, wq + b @ a, atol=0.0, rtol=0)

    effective.square().sum().backward()
    assert a.grad is not None and b.grad is not None
    assert torch.count_nonzero(a.grad) > 0 and torch.count_nonzero(b.grad) > 0


def test_effective_weight_rejects_inconsistent_shapes_instead_of_broadcasting() -> None:
    wq = torch.zeros(6, 4)
    good_a, good_b = torch.zeros(2, 4), torch.zeros(6, 2)

    assert_close(effective_weight(wq, good_a, good_b), wq, atol=0.0, rtol=0)
    with pytest.raises(ValueError, match="expected wq"):
        effective_weight(wq, good_b, good_a)  # transposed factors must not broadcast silently
    with pytest.raises(ValueError, match="expected wq"):
        effective_weight(wq, torch.zeros(3, 4), good_b)
    with pytest.raises(TypeError, match="must be a torch"):
        effective_weight(wq, good_a, [[0.0]])  # type: ignore[arg-type]
