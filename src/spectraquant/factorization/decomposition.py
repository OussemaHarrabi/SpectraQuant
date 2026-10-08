"""Low-rank factorization primitives for weight matrices.

Factor convention (frozen, load-bearing)
----------------------------------------
For a weight matrix ``W`` of shape ``(out_features, in_features)`` and a target ``rank``, every
factorization in this package obeys **one** convention, frozen in
``docs/coordination/design-m2-interfaces.md`` §0.2 and §2:

.. code-block:: text

    W  ~=  B @ A
    A : (rank, in_features)        # "down" factor, input side
    B : (out_features, rank)       # "up"   factor, output side
    W : (out_features, in_features)

so that ``B @ A`` has the shape of ``W`` and the dense parameter count is
``rank * (in_features + out_features)`` instead of ``out_features * in_features`` (see
:func:`factor_bytes` and ``docs/protocols/memory-accounting.md`` §4). The convention is stated here
**once**; every function below that returns factors (:class:`LowRankFactors`, :func:`truncated_svd`,
:func:`randomized_svd`, the strategies in :mod:`spectraquant.factorization.initialization`)
references this docstring rather than restating it. Do not transpose a factor anywhere else.

Approximation labelling
-----------------------
``truncated_svd`` and ``randomized_svd`` are *approximations* unless the matrix is rank-``rank``
(AGENTS.md §4.13). The residual norm is returned with every factorization
(:attr:`LowRankFactors.reconstruction_error`) instead of being silently dropped; a caller that needs
an exactness statement must use it.

Execution scope
---------------
Pure CPU PyTorch. ``torch.linalg.svd`` is not implemented for half precision on CPU, so this module
accepts **float32 and float64 only** and raises ``ValueError`` for other dtypes; no CUDA path exists
(``AGENTS.md`` §2, binding rule 2). All error measurements are accumulated in float64 regardless of
the input dtype, so float32 reconstructions are not reported with float32-level measurement noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "LowRankFactors",
    "factor_bytes",
    "randomized_svd",
    "reconstruction_error",
    "truncated_svd",
]

#: dtypes this CPU-only module performs linear algebra in. ``torch.linalg.svd`` has no CPU kernel
#: for float16/bfloat16, so those are rejected up front with an actionable message.
SUPPORTED_DTYPES: tuple[torch.dtype, ...] = (torch.float32, torch.float64)


# --------------------------------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------------------------------


def _validate_matrix(w: torch.Tensor, *, name: str = "w") -> None:
    """Validate that ``w`` is a 2-D CPU tensor of a supported floating dtype.

    Raises:
        TypeError: ``w`` is not a :class:`torch.Tensor`.
        ValueError: ``w`` is not 2-D, has an unsupported dtype, or lives on a non-CPU device.
    """
    if not isinstance(w, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(w).__name__}")
    if w.ndim != 2:
        raise ValueError(
            f"{name} must be 2-D (out_features, in_features), got shape {tuple(w.shape)}"
        )
    if w.dtype not in SUPPORTED_DTYPES:
        raise ValueError(
            f"{name} has dtype {w.dtype}; this CPU-only module supports "
            f"{', '.join(str(d) for d in SUPPORTED_DTYPES)} (torch.linalg.svd has no CPU kernel "
            "for half precision)"
        )
    if w.device.type != "cpu":
        raise ValueError(f"{name} is on device {w.device}; only CPU execution is supported")


def _validate_rank(rank: int) -> None:
    """Validate a requested rank without clamping it (clamping is done per-function)."""
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise TypeError(f"rank must be an int, got {type(rank).__name__}")
    if rank < 0:
        raise ValueError(f"rank must be >= 0, got {rank}")


def _validate_energy(energy_keep: float | None) -> None:
    """Validate ``energy_keep``: ``None`` (disabled) or a fraction in ``(0, 1]``."""
    if energy_keep is None:
        return
    if not (0.0 < float(energy_keep) <= 1.0):
        raise ValueError(f"energy_keep must be in (0, 1], got {energy_keep}")


def _rank_for_energy(s: torch.Tensor, energy_keep: float, cap: int) -> int:
    """Smallest number of singular values whose squared sum reaches ``energy_keep`` of the total.

    Args:
        s: 1-D singular values, descending, non-negative, same dtype as the matrix.
        energy_keep: target fraction of the total squared energy in ``(0, 1]``.
        cap: upper bound on the returned rank (including "keep everything").

    Returns:
        ``k`` with ``0 <= k <= min(cap, s.numel())``. A matrix whose total energy is zero (the zero
        matrix) returns ``0``: no direction carries energy, so no direction is "kept".
    """
    total = float((s * s).sum())
    if total <= 0.0:
        return 0
    cumulative = torch.cumsum(s * s, dim=0) / total
    target = torch.as_tensor(energy_keep, dtype=cumulative.dtype)
    # 'left' insertion point: first index i with cumulative[i] >= energy_keep.
    idx = int(torch.searchsorted(cumulative, target, right=False))
    return min(idx + 1, cap, int(s.numel()))


def _residual_error(
    w: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    norm: str = "fro",
) -> float:
    """Float64 residual norm of ``W - B @ A`` (shared by the factors and the public API)."""
    w64 = w.to(torch.float64)
    residual = w64 - (B.to(torch.float64) @ A.to(torch.float64))
    numerator = float(torch.linalg.matrix_norm(residual, ord="fro"))
    if norm == "fro":
        return numerator
    if norm in ("relative_fro", "fro_relative"):
        denominator = float(torch.linalg.matrix_norm(w64, ord="fro"))
        if denominator > 0.0:
            return numerator / denominator
        return 0.0 if numerator == 0.0 else float("inf")
    raise ValueError(f"unknown norm {norm!r}; expected 'fro' or 'relative_fro'")


# --------------------------------------------------------------------------------------------------
# Factor container
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LowRankFactors:
    """A rank-``rank`` factorization ``W ~= B @ A`` plus its measured reconstruction error.

    Convention (frozen; see the module docstring): ``A`` is ``(rank, in_features)`` and ``B`` is
    ``(out_features, rank)``.

    Attributes:
        A: ``(rank, in_features)`` tensor, same dtype/device as the source weight.
        B: ``(out_features, rank)`` tensor, same dtype/device as the source weight.
        rank: the effective rank actually stored (``0 <= rank <= min(out_features, in_features)``).
            It is the *effective* rank after clamping/energy selection, not the requested one.
        dtype: dtype shared by ``A`` and ``B`` (float32 or float64 on CPU).
        reconstruction_error: **absolute** Frobenius reconstruction error ``||W - B @ A||_F``,
            measured in float64 at construction time (``norm="fro"`` in
            :func:`reconstruction_error`). This is an approximation label, not a guarantee: it is
            ``0.0`` only for an exact factorization (AGENTS.md §4.13).

    Notes:
        Frozen. Construct via the factory :meth:`from_weight` or via ``truncated_svd`` /
        ``randomized_svd``; the raw constructor is validated in :meth:`__post_init__` but does not
        itself know ``W`` and therefore expects ``reconstruction_error`` to be passed in.
    """

    A: torch.Tensor
    B: torch.Tensor
    rank: int
    dtype: torch.dtype
    reconstruction_error: float

    def __post_init__(self) -> None:
        if self.A.ndim != 2 or self.B.ndim != 2:
            raise ValueError("A and B must both be 2-D")
        if self.A.shape[0] != self.rank or self.B.shape[1] != self.rank:
            raise ValueError(
                f"rank {self.rank} inconsistent with A.shape={tuple(self.A.shape)} "
                f"and B.shape={tuple(self.B.shape)}"
            )
        if self.A.dtype != self.B.dtype:
            raise ValueError(f"A dtype {self.A.dtype} != B dtype {self.B.dtype}")
        if self.A.dtype != self.dtype:
            raise ValueError(f"stored dtype {self.dtype} != factor dtype {self.A.dtype}")
        if self.rank < 0:
            raise ValueError(f"rank must be >= 0, got {self.rank}")

    @property
    def in_features(self) -> int:
        """Input feature count of the factorized weight (``A.shape[1]``)."""
        return int(self.A.shape[1])

    @property
    def out_features(self) -> int:
        """Output feature count of the factorized weight (``B.shape[0]``)."""
        return int(self.B.shape[0])

    @property
    def num_parameters(self) -> int:
        """Stored factor parameters, ``rank * (in_features + out_features)``."""
        return self.rank * (self.in_features + self.out_features)

    def reconstruct(self) -> torch.Tensor:
        """Return ``B @ A`` — the low-rank approximation of the source weight.

        Returns:
            ``(out_features, in_features)`` tensor with this object's dtype/device.
        """
        return self.B @ self.A

    @classmethod
    def from_weight(
        cls,
        w: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        *,
        norm: str = "fro",
    ) -> LowRankFactors:
        """Build factors from ``A``/``B`` and measure the reconstruction error against ``w``.

        Args:
            w: ``(out_features, in_features)`` float32/float64 CPU tensor.
            A: ``(rank, in_features)`` factor.
            B: ``(out_features, rank)`` factor.
            norm: error norm forwarded to :func:`reconstruction_error` (``"fro"`` or
                ``"relative_fro"``). The stored ``reconstruction_error`` uses whatever is passed;
                ``"fro"`` (absolute) is the default.

        Returns:
            A :class:`LowRankFactors` with ``reconstruction_error`` measured in float64.
        """
        _validate_matrix(w)
        if B.ndim != 2 or A.ndim != 2 or B.shape[0] != w.shape[0] or A.shape[1] != w.shape[1]:
            raise ValueError(
                f"factor shapes B{tuple(B.shape)}@A{tuple(A.shape)} do not match w{tuple(w.shape)}"
            )
        return cls(
            A=A,
            B=B,
            rank=int(A.shape[0]),
            dtype=A.dtype,
            reconstruction_error=_residual_error(w, A, B, norm=norm),
        )


# --------------------------------------------------------------------------------------------------
# Reconstruction error
# --------------------------------------------------------------------------------------------------


def reconstruction_error(
    w: torch.Tensor,
    f: LowRankFactors,
    *,
    norm: str = "fro",
) -> float:
    """Reconstruction error of the factorization ``f`` against the weight ``w``.

    The residual ``R = W - B @ A`` (convention: module docstring) is formed in **float64** so the
    measurement is not limited by float32 accumulation, then reduced by the requested norm.

    Args:
        w: ``(out_features, in_features)`` float32/float64 CPU tensor.
        f: :class:`LowRankFactors` whose ``B``/``A`` factorize ``w`` (checked by shape).
        norm: one of

            * ``"fro"`` (default) — absolute Frobenius norm ``||W - B @ A||_F``. For a truncated
              SVD of rank ``r`` this equals ``sqrt(sum_{i > r} s_i^2)``, the Euclidean norm of the
              discarded singular values (asserted by the property test).
            * ``"relative_fro"`` (alias ``"fro_relative"``) — ``||W - B @ A||_F / ||W||_F``, the
              scale-free variant. If ``||W||_F == 0`` the result is ``0.0`` when the residual is also
              zero and ``inf`` otherwise (the ratio is undefined for a zero denominator).

    Returns:
        The error as a Python ``float``. Absolute for ``"fro"``; in ``[0, inf]`` for the relative
        variants. An exact factorization returns ``0.0`` (within float64 round-off).

    Raises:
        TypeError: ``f`` is not a :class:`LowRankFactors`.
        ValueError: shape mismatch between ``w`` and the factors, or an unknown ``norm``.
    """
    _validate_matrix(w)
    if not isinstance(f, LowRankFactors):
        raise TypeError(f"f must be a LowRankFactors, got {type(f).__name__}")
    if f.B.shape[0] != w.shape[0] or f.A.shape[1] != w.shape[1]:
        raise ValueError(
            f"factor shapes B{tuple(f.B.shape)}@A{tuple(f.A.shape)} do not match w{tuple(w.shape)}"
        )
    return _residual_error(w, f.A, f.B, norm=norm)


# --------------------------------------------------------------------------------------------------
# Factorizations
# --------------------------------------------------------------------------------------------------


def truncated_svd(
    w: torch.Tensor,
    rank: int,
    *,
    energy_keep: float | None = None,
) -> LowRankFactors:
    """Truncated SVD of ``w`` in the frozen ``W ~= B @ A`` convention.

    Uses the exact dense SVD (``torch.linalg.svd``, ``full_matrices=False``). With ``U S Vh = W``:

    .. code-block:: text

        A = diag(S[:r]) @ Vh[:r]     # (r, in_features)
        B = U[:, :r]                 # (out_features, r)
        B @ A = U[:, :r] diag(S[:r]) Vh[:r]

    Args:
        w: ``(out_features, in_features)`` float32/float64 **CPU** tensor.
        rank: requested rank. Clamped to ``min(rank, min(w.shape))``: a request larger than the
            intrinsic rank is treated as "keep everything" rather than raising, because callers
            (e.g. the allocator) may legitimately ask for more rank than the layer can use.
            ``rank=0`` is allowed and yields zero-filled factors whose reconstruction error is
            ``||W||_F``.
        energy_keep: optional fraction in ``(0, 1]``. When given, the effective rank becomes the
            smallest ``r'`` whose retained squared-singular-value fraction reaches ``energy_keep``,
            **capped at the requested (clamped) rank**. It can only reduce the rank, never raise it;
            a zero matrix selects rank 0.

    Returns:
        :class:`LowRankFactors` with the exact truncation error recorded (float64 measurement).

    Notes:
        Deterministic: the dense SVD is a deterministic CPU LAPACK routine; repeated calls on the
        same input return bit-identical factors.

    Raises:
        ValueError: ``w`` is not a supported 2-D CPU float tensor; ``rank`` is negative; or
            ``energy_keep`` is outside ``(0, 1]``.
    """
    _validate_matrix(w)
    _validate_rank(rank)
    _validate_energy(energy_keep)

    out_features, in_features = w.shape
    max_rank = min(out_features, in_features)
    effective = min(rank, max_rank)

    if effective == 0 and energy_keep is None:
        A = w.new_zeros((0, in_features))
        B = w.new_zeros((out_features, 0))
        return LowRankFactors.from_weight(w, A, B)

    U, S, Vh = torch.linalg.svd(w, full_matrices=False)
    if energy_keep is not None:
        effective = min(effective, _rank_for_energy(S, energy_keep, cap=max_rank))

    # diag(S[:r]) @ Vh[:r] without materialising a diagonal matrix.
    A = Vh[:effective] * S[:effective].unsqueeze(1)
    B = U[:, :effective].contiguous()
    return LowRankFactors.from_weight(w, A, B)


def randomized_svd(
    w: torch.Tensor,
    rank: int,
    *,
    n_oversamples: int,
    n_iter: int,
    seed: int,
) -> LowRankFactors:
    """Randomized (Halko–Martinsson–Tropp) SVD in the frozen ``W ~= B @ A`` convention.

    Draws a Gaussian test matrix ``Omega`` of shape ``(in_features, l)`` with
    ``l = min(rank + n_oversamples, min(w.shape))`` using a dedicated CPU
    :class:`torch.Generator`, forms ``Y = W @ Omega``, optionally applies ``n_iter`` power
    iterations with a QR re-orthonormalisation after every application to keep the basis
    well-conditioned, then projects to a small ``(l, in_features)`` matrix whose exact SVD supplies
    the factors. The result is an **approximation** even when ``l`` exceeds the intrinsic rank: the
    residual norm is recorded in :attr:`LowRankFactors.reconstruction_error`.

    Args:
        w: ``(out_features, in_features)`` float32/float64 **CPU** tensor.
        rank: requested rank, clamped to ``min(rank, min(w.shape))`` exactly as in
            :func:`truncated_svd` (``rank=0`` returns zero factors).
        n_oversamples: extra random directions ``l - rank`` used to protect against an unlucky
            sketch. Must be ``>= 0``. ``0`` means ``l == rank`` (no oversampling).
        n_iter: number of power iterations. Must be ``>= 0``. ``0`` is fastest and least accurate;
            ``2``–``4`` is the usual range in Halko et al. Power iterations improve the accuracy of
            slowly-decaying spectra at the cost of ``2 * n_iter`` extra matmuls.
        seed: seed for the internal CPU generator; a fixed ``(seed, shape, dtype)`` triple gives
            bit-identical factors across runs and processes (CPU RNG only, no global state touched).

    Returns:
        :class:`LowRankFactors` whose ``reconstruction_error`` (absolute Frobenius, float64) is the
        *measured* error of this randomized approximation — never the exact-SVD error.

    Notes:
        The error bound used to validate this implementation against the exact SVD is the Halko et
        al. (2011) expectation bound
        ``E ||W - Q Qᵀ W|| <= (1 + sqrt(k / (l - k + 1))) * sigma_{k+1} + ...``; the measured
        randomized-vs-exact ratios and the conservative empirical bound we enforce are recorded in
        both the test module and ``docs/research/spectral-notes.md``.

    Raises:
        ValueError: ``w`` invalid; ``rank`` negative; ``n_oversamples`` or ``n_iter`` negative.
    """
    _validate_matrix(w)
    _validate_rank(rank)
    if n_oversamples < 0:
        raise ValueError(f"n_oversamples must be >= 0, got {n_oversamples}")
    if n_iter < 0:
        raise ValueError(f"n_iter must be >= 0, got {n_iter}")

    out_features, in_features = w.shape
    max_rank = min(out_features, in_features)
    effective = min(rank, max_rank)

    if effective == 0:
        A = w.new_zeros((0, in_features))
        B = w.new_zeros((out_features, 0))
        return LowRankFactors.from_weight(w, A, B)

    sketch = min(effective + int(n_oversamples), max_rank)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    omega = torch.randn((in_features, sketch), dtype=w.dtype, device="cpu", generator=generator)
    Y = w @ omega
    Q, _ = torch.linalg.qr(Y)
    for _ in range(int(n_iter)):
        Z = w.mT @ Q
        Qz, _ = torch.linalg.qr(Z)
        Y = w @ Qz
        Q, _ = torch.linalg.qr(Y)

    small = Q.mT @ w  # (sketch, in_features)
    U_small, S_small, Vh_small = torch.linalg.svd(small, full_matrices=False)
    r = min(effective, int(S_small.numel()))
    A = Vh_small[:r] * S_small[:r].unsqueeze(1)
    B = Q @ U_small[:, :r]
    return LowRankFactors.from_weight(w, A, B)


# --------------------------------------------------------------------------------------------------
# Class-1 analytical cost
# --------------------------------------------------------------------------------------------------


def factor_bytes(
    in_features: int,
    out_features: int,
    rank: int,
    dtype_bytes: int = 2,
) -> int:
    """Analytical (class-1) byte cost of rank-``rank`` factors of a ``(out, in)`` weight.

    ``rank * (in_features + out_features) * dtype_bytes`` — the stored factor payload only. It
    excludes quantization scales/zero-points and container overhead; for the quantized-factor
    formula see ``docs/protocols/memory-accounting.md`` §4. This is an **estimate** (AGENTS.md §4.13,
    measurement class 1) and MUST NOT be reported as a measured artifact size.

    Args:
        in_features: ``in_features`` of the weight (``A.shape[1]``).
        out_features: ``out_features`` of the weight (``B.shape[0]``).
        rank: factor rank (effective rank, ``>= 0``).
        dtype_bytes: bytes per factor element: ``2`` for fp16/bf16 (default, the locked factor
            dtype), ``4`` for fp32. Any positive int is accepted.

    Returns:
        Non-negative integer byte count (exact for the analytical model).

    Raises:
        ValueError: any argument is negative, or ``dtype_bytes`` is not positive.
    """
    for name, value in (
        ("in_features", in_features),
        ("out_features", out_features),
        ("rank", rank),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an int, got {type(value).__name__}")
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")
    if isinstance(dtype_bytes, bool) or not isinstance(dtype_bytes, int):
        raise TypeError(f"dtype_bytes must be an int, got {type(dtype_bytes).__name__}")
    if dtype_bytes <= 0:
        raise ValueError(f"dtype_bytes must be > 0, got {dtype_bytes}")
    return rank * (in_features + out_features) * dtype_bytes
