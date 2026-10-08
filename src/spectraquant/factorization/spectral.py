"""Spectral analysis of weight matrices.

All functions consume a 2-D weight ``W`` of shape ``(out_features, in_features)`` and return
quantities derived from its singular values ``s_1 >= s_2 >= ... >= s_k``, ``k = min(W.shape)``.
They are **exact** (a dense CPU SVD, no sampling), so callers may treat their outputs as
measurements of the matrix rather than estimates — but note that the *inputs* are weight-space
quantities: they are not the output-error unit of ``design-m2-interfaces.md`` §0.3 and must not be
mixed with it.

Convention note: the singular values themselves are convention-free, but when these summaries are
converted back into factors, the frozen ``W ~= B @ A`` convention from
:mod:`spectraquant.factorization.decomposition` applies.

Execution scope: CPU only, float32/float64 only (``torch.linalg.svdvals`` has no half-precision CPU
kernel). A zero matrix yields ``0.0``/``nan`` placeholders that are documented per key rather than
raising, so a proxy can consume a summary of any real tensor.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "effective_rank",
    "singular_values",
    "spectral_summary",
    "stable_rank",
]

#: Number of leading singular values surfaced individually in :func:`spectral_summary` as
#: ``sv_0`` .. ``sv_7``. Fixed so downstream feature vectors have a stable width across layers.
TOP_K_SUMMARY = 8

#: Fraction of the leading *directions* whose energy share is reported by
#: :func:`spectral_summary` under ``energy_top_1pct``.
TOP_ENERGY_FRACTION = 0.01


def _validate(w: torch.Tensor) -> None:
    """Reject anything other than a 2-D float32/float64 CPU tensor."""
    if not isinstance(w, torch.Tensor):
        raise TypeError(f"w must be a torch.Tensor, got {type(w).__name__}")
    if w.ndim != 2:
        raise ValueError(f"w must be 2-D (out_features, in_features), got shape {tuple(w.shape)}")
    if w.dtype not in (torch.float32, torch.float64):
        raise ValueError(
            f"w has dtype {w.dtype}; spectral analysis is CPU-only and supports float32/float64 "
            "(torch.linalg.svdvals has no CPU kernel for half precision)"
        )
    if w.device.type != "cpu":
        raise ValueError(f"w is on device {w.device}; only CPU execution is supported")


def _sorted_singular_values(w: torch.Tensor) -> torch.Tensor:
    """Descending singular values as a float64 1-D tensor (float64 for stable downstream stats)."""
    s = torch.linalg.svdvals(w.to(torch.float64))
    return torch.sort(s, descending=True).values


def singular_values(w: torch.Tensor, *, top_k: int | None = None) -> torch.Tensor:
    """Singular values of ``w``, descending, in the input's float32/float64 precision.

    Args:
        w: ``(out_features, in_features)`` float32/float64 CPU tensor.
        top_k: if given, return only the ``top_k`` largest values (``>= 0``); clamped to
            ``min(w.shape)``. ``None`` (default) returns all ``min(w.shape)`` values.

    Returns:
        1-D tensor of length ``min(top_k, min(w.shape))`` when ``top_k`` is given, else
        ``min(w.shape)``, dtype ``float32``/``float64`` matching ``w``, on CPU, descending.

    Raises:
        ValueError: ``w`` invalid, or ``top_k`` negative.
    """
    _validate(w)
    if top_k is not None and top_k < 0:
        raise ValueError(f"top_k must be >= 0, got {top_k}")
    s = torch.linalg.svdvals(w)
    if top_k is not None:
        s = s[: min(int(top_k), int(s.numel()))]
    return s


def effective_rank(w: torch.Tensor, *, energy: float = 0.99) -> float:
    """Number of singular directions needed to capture ``energy`` of the squared energy.

    Defined as the smallest ``k`` such that ``sum_{i<=k} s_i^2 / sum_i s_i^2 >= energy``, returned
    as a ``float``. This is the *energy-threshold* notion of effective rank used by the allocator to
    bound how much rank is worth spending; it is **not** the entropy-based
    ``exp(-sum p_i log p_i)`` definition (which takes no threshold).

    Args:
        w: ``(out_features, in_features)`` float32/float64 CPU tensor.
        energy: target fraction of total squared energy, in ``(0, 1]``. ``0.99`` (default) is the
            standard "99 % energy" cut.

    Returns:
        ``0.0`` for the zero matrix (no direction carries energy), otherwise a value in
        ``[1, min(w.shape)]``.

    Raises:
        ValueError: ``w`` invalid or ``energy`` outside ``(0, 1]``.
    """
    _validate(w)
    if not (0.0 < float(energy) <= 1.0):
        raise ValueError(f"energy must be in (0, 1], got {energy}")
    s = _sorted_singular_values(w)
    total = float((s * s).sum())
    if total <= 0.0:
        return 0.0
    cumulative = torch.cumsum(s * s, dim=0) / total
    target = torch.as_tensor(float(energy), dtype=cumulative.dtype)
    idx = int(torch.searchsorted(cumulative, target, right=False))
    return float(min(idx + 1, int(s.numel())))


def stable_rank(w: torch.Tensor) -> float:
    """Stable rank ``||W||_F^2 / ||W||_2^2 = sum_i s_i^2 / s_1^2`` (Rudelson–Vershynin).

    Bounded by ``min(w.shape)``; equals the effective rank of a perfectly flat spectrum and is
    close to the intrinsic rank of a spectrally concentrated matrix.

    Args:
        w: ``(out_features, in_features)`` float32/float64 CPU tensor.

    Returns:
        ``0.0`` for the zero matrix, otherwise a value in ``(0, min(w.shape)]``.

    Raises:
        ValueError: ``w`` invalid.
    """
    _validate(w)
    s = _sorted_singular_values(w)
    top = float(s[0]) if s.numel() else 0.0
    if top <= 0.0:
        return 0.0
    return float((s * s).sum()) / (top * top)


def _power_law_decay(s: torch.Tensor) -> tuple[float, float]:
    """Least-squares fit of ``ln s_i`` on ``ln i`` over positive singular values.

    Returns ``(exponent, r2)`` where the fitted model is ``s_i ~= c * i^(-exponent)``. Returns
    ``(nan, nan)`` when fewer than two positive singular values exist (no fit is defined).
    """
    positive = s[s > 0]
    if int(positive.numel()) < 2:
        return float("nan"), float("nan")
    idx = torch.arange(1, int(positive.numel()) + 1, dtype=torch.float64)
    x = torch.log(idx)
    y = torch.log(positive.to(torch.float64))
    x_mean, y_mean = x.mean(), y.mean()
    denominator = float(((x - x_mean) ** 2).sum())
    if denominator == 0.0:
        return float("nan"), float("nan")
    slope = float(((x - x_mean) * (y - y_mean)).sum()) / denominator
    intercept = float(y_mean) - slope * float(x_mean)
    predicted = slope * x + intercept
    residual = float(((y - predicted) ** 2).sum())
    total = float(((y - y_mean) ** 2).sum())
    r2 = 1.0 - residual / total if total > 0.0 else float("nan")
    return -slope, r2


def spectral_summary(w: torch.Tensor) -> dict[str, float]:
    """Flat, layer-comparable spectral feature vector consumed by the proxy slice.

    Every returned value is a Python ``float`` so the result can be concatenated into a fixed-width
    feature matrix. Keys (all measured, float64 internally):

    Leading spectrum
        ``n_singular_values`` — number of singular values (``min(w.shape)``).
        ``sv_0`` .. ``sv_7`` — the ``TOP_K_SUMMARY`` (8) largest singular values, descending;
            absent when the matrix has fewer singular values than that.

    Decay
        ``spectral_decay_exponent`` — power-law exponent ``alpha`` from the least-squares fit
            ``ln s_i = ln c - alpha * ln i`` over the positive singular values. ``nan`` if fewer
            than two positive singular values (e.g. the zero matrix).
        ``spectral_decay_r2`` — coefficient of determination of that fit (1.0 = perfect power law);
            ``nan`` alongside the exponent.

    Concentration
        ``effective_rank`` — the threshold rank at 99 % energy (see :func:`effective_rank`).
        ``stable_rank`` — see :func:`stable_rank`.
        ``condition_number`` — ``s_1 / s_k``; ``inf`` if ``s_k == 0 < s_1`` and ``0.0`` for the
            zero matrix.
        ``condition_number_nonzero`` — ``s_1 / min positive s_i``, finite for rank-deficient
            matrices (``0.0`` for the zero matrix).
        ``energy_top_1pct`` — fraction of total squared energy in the top
            ``max(1, ceil(0.01 * n))`` directions (``0.0`` for the zero matrix).
        ``energy_top_1pct_count`` — how many directions that was.

    Quantile spread of the singular values
        ``sv_min``, ``sv_q25``, ``sv_median``, ``sv_q75``, ``sv_max`` — quantiles of the singular
        value *values* (not energies).
        ``sv_iqr`` — interquartile range ``sv_q75 - sv_q25``.
        ``sv_iqr_over_median`` — ``sv_iqr / sv_median`` (``0.0`` when the median is zero).
        ``sv_coefficient_of_variation`` — ``std(s) / mean(s)`` (``0.0`` for the zero matrix);
            ``sqrt((n - 1) / n)``-normalised (population) standard deviation.

    Scale
        ``frobenius_norm`` — ``||W||_F = sqrt(sum s_i^2)``.
        ``spectral_norm`` — ``s_1``.
        ``nuclear_norm`` — ``sum_i s_i``.

    Args:
        w: ``(out_features, in_features)`` float32/float64 CPU tensor.

    Returns:
        ``dict[str, float]``; keys are always present except ``sv_j`` entries beyond the matrix's
        rank. Exact (dense SVD), not an estimator.

    Raises:
        ValueError: ``w`` is not a supported 2-D CPU float tensor.
    """
    _validate(w)
    s64 = _sorted_singular_values(w)
    n = int(s64.numel())
    summary: dict[str, float] = {}

    summary["n_singular_values"] = float(n)
    for i in range(min(TOP_K_SUMMARY, n)):
        summary[f"sv_{i}"] = float(s64[i])

    exponent, r2 = _power_law_decay(s64)
    summary["spectral_decay_exponent"] = exponent
    summary["spectral_decay_r2"] = r2

    summary["effective_rank"] = effective_rank(w)
    summary["stable_rank"] = stable_rank(w)

    top = float(s64[0]) if n else 0.0
    smallest = float(s64[-1]) if n else 0.0
    if top <= 0.0:
        summary["condition_number"] = 0.0
        summary["condition_number_nonzero"] = 0.0
    else:
        summary["condition_number"] = math.inf if smallest <= 0.0 else top / smallest
        positive_min = float(s64[s64 > 0].min()) if bool((s64 > 0).any()) else 0.0
        summary["condition_number_nonzero"] = top / positive_min

    energy = s64 * s64
    total_energy = float(energy.sum())
    k_top = max(1, math.ceil(TOP_ENERGY_FRACTION * n))
    summary["energy_top_1pct_count"] = float(k_top)
    summary["energy_top_1pct"] = (
        float(energy[:k_top].sum()) / total_energy if total_energy > 0.0 else 0.0
    )

    quantiles = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
    q = torch.quantile(s64, quantiles) if n else torch.zeros(5, dtype=torch.float64)
    summary["sv_min"] = float(q[0])
    summary["sv_q25"] = float(q[1])
    summary["sv_median"] = float(q[2])
    summary["sv_q75"] = float(q[3])
    summary["sv_max"] = float(q[4])
    iqr = float(q[3] - q[1])
    summary["sv_iqr"] = iqr
    median = float(q[2])
    summary["sv_iqr_over_median"] = iqr / median if median > 0.0 else 0.0
    mean = float(s64.mean()) if n else 0.0
    summary["sv_coefficient_of_variation"] = (
        float(s64.std(unbiased=False)) / mean if mean > 0.0 else 0.0
    )

    summary["frobenius_norm"] = math.sqrt(total_energy)
    summary["spectral_norm"] = top
    summary["nuclear_norm"] = float(s64.sum()) if n else 0.0
    return summary
