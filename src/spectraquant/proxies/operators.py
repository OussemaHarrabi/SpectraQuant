"""Compression operators shared by every proxy variant.

One definition of "the compressed operator" so that no variant can silently disagree about what
``W_hat`` is. The pipeline is always: truncated SVD of ``W`` into ``(A, B)`` with ``A: (rank, in)``
and ``B: (out, rank)`` (the frozen convention of ``design-m2-interfaces.md`` section 0.2), then fake
quantization of **both factors**, then reconstruction ``W_hat = Q(B) @ Q(A)``.

Shapes:
    ``w``: ``(out_features, in_features)`` float32/float64 CPU tensor. ``rank``: int, clamped to
    ``min(shape)``; ``rank <= 0`` yields the zero reconstruction (documented, not an error, because
    an allocator may legitimately assign rank 0).

Dtypes / device:
    Output has the dtype and device of ``w``. No CUDA path exists (``AGENTS.md`` section 2).

Numerical limitations:
    Everything here is fake quantization (measurement class 2): the factors are rounded and then
    stored back in floating point. No packed bytes and no kernel are involved, so nothing produced by
    this module may be reported as low-bit storage or as accelerated inference.
"""

from __future__ import annotations

import torch

from spectraquant.factorization import truncated_svd
from spectraquant.quantization import QuantSpec, fake_quantize

__all__ = [
    "compression_delta",
    "low_rank_delta",
    "low_rank_reconstruction",
    "quantization_only_delta",
    "quantized_factor_reconstruction",
]


def low_rank_reconstruction(w: torch.Tensor, rank: int) -> torch.Tensor:
    """Return ``B @ A``, the truncated-SVD reconstruction of ``w`` without quantization.

    Args:
        w: ``(out, in)`` weight.
        rank: target rank; clamped to ``min(w.shape)``; ``rank <= 0`` returns zeros.

    Returns:
        ``(out, in)`` tensor with ``w``'s dtype and device.
    """
    if w.dim() != 2:
        raise ValueError(f"w must be 2-D (out, in), got {tuple(w.shape)}")
    if rank <= 0:
        return torch.zeros_like(w)
    factors = truncated_svd(w, rank)
    return factors.B @ factors.A


def quantized_factor_reconstruction(w: torch.Tensor, rank: int, spec: QuantSpec) -> torch.Tensor:
    """Return ``Q(B) @ Q(A)``: the low-rank factors rounded to ``spec``'s grid.

    The quantization is applied to each factor independently (per the frozen unit), so the returned
    operator is what a deployment of the low-rank form would actually compute under fake
    quantization.
    """
    if w.dim() != 2:
        raise ValueError(f"w must be 2-D (out, in), got {tuple(w.shape)}")
    if rank <= 0:
        return torch.zeros_like(w)
    factors = truncated_svd(w, rank)
    q_a = fake_quantize(factors.A, spec)
    q_b = fake_quantize(factors.B, spec)
    return q_b @ q_a


def compression_delta(w: torch.Tensor, rank: int, spec: QuantSpec) -> torch.Tensor:
    """Return ``W - Q(B) Q(A)``, the operator error the proxy unit is defined on."""
    return w - quantized_factor_reconstruction(w, rank, spec)


def low_rank_delta(w: torch.Tensor, rank: int) -> torch.Tensor:
    """Return ``W - B A``: the factorization error alone (no quantization)."""
    return w - low_rank_reconstruction(w, rank)


def quantization_only_delta(w: torch.Tensor, rank: int, spec: QuantSpec) -> torch.Tensor:
    """Return ``B A - Q(B) Q(A)``: the error introduced *by rounding the factors* alone.

    This is the quantity the ``quant_residual_stats`` variant scores; it isolates rounding from
    truncation so the two error sources can be reported separately.
    """
    return low_rank_reconstruction(w, rank) - quantized_factor_reconstruction(w, rank, spec)
