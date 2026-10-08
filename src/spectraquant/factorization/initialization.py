"""Factor-initialization strategies for low-rank preparation.

All three strategies return :class:`~spectraquant.factorization.decomposition.LowRankFactors`, so
the frozen ``W ~= B @ A`` convention of
:mod:`spectraquant.factorization.decomposition` (``A: (rank, in_features)``,
``B: (out_features, rank)``) applies to every object produced here — see that module's docstring.

Available in Milestone 2
------------------------
``svd``
    Plain truncated SVD of the weight (:func:`initialize_svd`). The deterministic baseline; exact
    for a rank-``rank`` matrix.
``svd_residual``
    Truncated SVD **plus the explicitly returned residual** ``R = W - B @ A``
    (:func:`initialize_svd_residual`), so a downstream quantizer can spend bits on the part the
    factors did not capture instead of discarding it.
``random``
    Seeded Gaussian initialization scaled so the expected factor norm matches ``||W||_F``
    (:func:`initialize_random`). Used as a control arm and as an initialization for training-time
    schemes that refine factors later.

LoftQ-style alternating initialization is *not* implemented here
----------------------------------------------------------------
LoftQ (Li et al., 2023) alternates between (i) quantizing the current base and computing the
residual ``R = W - Q(B @ A)`` and (ii) taking the SVD of that residual to update ``B``/``A``. The
**first half** of that loop is exactly :func:`initialize_svd_residual`; the full alternating scheme
additionally needs the quantizer (Milestone 2's ``spectraquant.quantization`` slice) and belongs to
the training/preparation milestone (Milestone 3). It is deliberately not implemented here so that no
caller mistakes a single residual step for LoftQ. When it lands, it will expose the same
``LowRankFactors`` return type and the same convention.

Execution scope: CPU only, float32/float64 only (frozen-SVD and CPU RNG paths).
"""

from __future__ import annotations

import torch

from spectraquant.factorization.decomposition import (
    LowRankFactors,
    _validate_matrix,
    _validate_rank,
    truncated_svd,
)

__all__ = [
    "initialize_random",
    "initialize_svd",
    "initialize_svd_residual",
]


def initialize_svd(
    w: torch.Tensor,
    rank: int,
    *,
    energy_keep: float | None = None,
) -> LowRankFactors:
    """Plain truncated-SVD initialization (``W ~= B @ A``; see the module docstring).

    Thin, documented alias for :func:`spectraquant.factorization.decomposition.truncated_svd` so
    that all initialization strategies share one name and return type.

    Args:
        w: ``(out_features, in_features)`` float32/float64 CPU tensor.
        rank: requested rank, clamped to ``min(rank, min(w.shape))``.
        energy_keep: optional energy fraction in ``(0, 1]`` that may *reduce* the effective rank.

    Returns:
        :class:`LowRankFactors` with the exact truncation error recorded in float64.
    """
    return truncated_svd(w, rank, energy_keep=energy_keep)


def initialize_svd_residual(
    w: torch.Tensor,
    rank: int,
    *,
    energy_keep: float | None = None,
) -> tuple[LowRankFactors, torch.Tensor]:
    """Truncated-SVD initialization that also returns the residual ``W - B @ A``.

    This is the SVD step of the LoftQ alternating schedule (module docstring): the factors capture
    the dominant subspace and the residual is handed back explicitly, in the input's dtype, so the
    caller can quantize it, keep outlier channels from it, or feed it into the next LoftQ iteration.

    Args:
        w: ``(out_features, in_features)`` float32/float64 CPU tensor.
        rank: requested rank, clamped to ``min(rank, min(w.shape))``.
        energy_keep: optional energy fraction in ``(0, 1]`` that may *reduce* the effective rank.

    Returns:
        ``(factors, residual)`` where ``factors.reconstruct() + residual == w`` up to float64
        round-off and ``residual`` has the same shape/dtype/device as ``w``. ``factors`` carries the
        absolute Frobenius reconstruction error; ``residual`` is ``W - B @ A`` in ``w.dtype``
        (computed in float64, then cast), i.e. its Frobenius norm equals that error up to cast
        round-off.

    Raises:
        ValueError: ``w`` is not a supported 2-D CPU float tensor, or ``rank``/``energy_keep``
            invalid.
    """
    factors = truncated_svd(w, rank, energy_keep=energy_keep)
    residual = (w.to(torch.float64) - factors.reconstruct().to(torch.float64)).to(w.dtype)
    return factors, residual


def initialize_random(
    w: torch.Tensor,
    rank: int,
    *,
    seed: int,
    scale: float | None = None,
) -> LowRankFactors:
    """Seeded Gaussian random initialization scaled to the weight's Frobenius norm.

    Entries of ``A`` and ``B`` are drawn ``N(0, scale^2)`` from a dedicated CPU
    :class:`torch.Generator`. With the default ``scale=None`` the scale is chosen so that the
    *expected* squared Frobenius norm of ``B @ A`` equals ``||W||_F^2``. Since
    ``(B @ A)_ij = sum_k B_ik A_kj`` has variance ``rank * scale^4``, the expected squared Frobenius
    norm is ``out_features * in_features * rank * scale^4``, hence

    .. code-block:: text

        scale = ( ||W||_F^2 / (out_features * in_features * rank) ) ** 0.25

    This is a control/reference initialization, **not** a compression method: it captures no
    structure of ``W``.

    Args:
        w: ``(out_features, in_features)`` float32/float64 CPU tensor. Only its Frobenius norm is
            used, so a zero matrix is handled (returns zero factors with error ``||W||_F``).
        rank: requested rank, clamped to ``min(rank, min(w.shape))``.
        seed: generator seed; a fixed ``(seed, shape, dtype)`` triple is bit-reproducible on CPU and
            does not touch global RNG state.
        scale: optional explicit standard deviation for both factors (``> 0``). ``None`` uses the
            norm-matched default above.

    Returns:
        :class:`LowRankFactors` whose ``reconstruction_error`` (absolute Frobenius, float64) is the
        measured error of this random draw.

    Raises:
        ValueError: ``w`` invalid, ``rank`` negative, or ``scale`` provided but not positive.
    """
    _validate_matrix(w)
    _validate_rank(rank)
    out_features, in_features = w.shape
    max_rank = min(out_features, in_features)
    effective = min(rank, max_rank)

    if scale is not None and not (float(scale) > 0.0):
        raise ValueError(f"scale must be > 0 when provided, got {scale}")

    if effective == 0:
        A = w.new_zeros((0, in_features))
        B = w.new_zeros((out_features, 0))
        return LowRankFactors.from_weight(w, A, B)

    if scale is None:
        weight_norm_sq = float((w.to(torch.float64) ** 2).sum())
        sigma = (weight_norm_sq / float(out_features * in_features * effective)) ** 0.25
    else:
        sigma = float(scale)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    A = (
        torch.randn((effective, in_features), dtype=w.dtype, device="cpu", generator=generator)
        * sigma
    )
    B = (
        torch.randn((out_features, effective), dtype=w.dtype, device="cpu", generator=generator)
        * sigma
    )
    return LowRankFactors.from_weight(w, A, B)
