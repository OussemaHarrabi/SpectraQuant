"""LoftQ alternating quantize/SVD initialisation (Milestone 3) — measurement class 2.

Reference: Li et al., *LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models*,
arXiv:2310.08659 (``docs/research/upstream-lockfile.md`` pins ``yxli2123/LoftQ`` at
``ae33fd4f…``; the reference behaviour implemented there is also in HF PEFT's
``peft/utils/loftq_utils.py::loftq_init``, pinned in the same lockfile). LoftQ initialises a
*quantized base* ``Q`` and a *low-rank part* ``B @ A`` **jointly**, so that the pair reconstructs
the full-precision weight better than ``Q`` alone:

.. code-block:: text

    min_{Q, A, B} || W - Q - B @ A ||_F                                       (paper Eq. 6)

The alternating schedule (paper §3.2, Eq. 7-8; ``A``/``B`` below use this repository's frozen
convention ``W ~= B @ A`` with ``A: (rank, in_features)``, ``B: (out_features, rank)`` — the
paper writes the low-rank part as ``A Bᵀ`` in the transposed convention):

.. code-block:: text

    A_0, B_0 <- SVD_rank(W)                     # factor initialization (see "Difference" below)
    for t = 1 .. T:
        Q_t   = q_N(W - B_{t-1} @ A_{t-1})      # Eq. 7: quantize the residual the factors leave
        R_t   = W - Q_t                         # Eq. 8: "residual of the quantization"
        A_t, B_t <- SVD_rank(R_t)               # Eq. 8: update the factors from that residual

``iterations == 1`` is **not** LoftQ
-----------------------------------
A single iteration is one quantization half-step plus one SVD half-step. It is the *first
residual step* of the schedule, not the method: the paper's schedule alternates the two halves
``T`` times (``T = 1, 3, 5`` in their ablations, ``--iter`` in the reference code) and the published
quality claims are made for ``T >= 3``. Because one iteration only re-estimates the factors of the
*current* quantized base, a caller that reports ``iterations=1`` as "LoftQ" would mislabel a
single-step initialization; :func:`loftq_initialise` therefore raises for ``iterations < 1``
instead of silently treating ``0`` as "no LoftQ".

Difference from the paper's initialization
------------------------------------------
The paper starts the alternation from ``A_0 = B_0 = 0`` (so ``Q_1 = q_N(W)``, the plain
post-training-quantization base). This module starts from ``SVD_rank(W)`` as instructed by the
Milestone-3 interface, which makes the first quantized base ``Q_1 = q_N(W - B_0 @ A_0)`` instead of
``q_N(W)``; from ``t = 2`` on the two schedules coincide. The SVD step itself is the repository's
existing :func:`~spectraquant.factorization.initialization.initialize_svd`, not a re-implementation.

What the two-tensor return does *not* contain
---------------------------------------------
The LoftQ pair is a triple ``(Q_T, A_T, B_T)``; this function returns the low-rank half
``(A_T, B_T)`` only. A caller that needs the companion base recomputes it from the returned
factors as

.. code-block:: text

    Q = fake_quantize(W - B_T @ A_T, spec)

which is the *next* quantization half-step of the same alternation and reproduces ``Q_T`` whenever
the re-derived block scales agree with those computed from ``R_T`` (exact for the fixed-scale
blocks of a symmetric per-tensor spec, approximate otherwise). Unlike ``Q_T``, it is computable
from the returned pair alone, and ``Q + B_T @ A_T`` is the paper's Eq. 6 reconstruction of ``W``.

Numerical limitations
---------------------
* The SVD is sign-ambiguous (``U``, ``Vh`` rows may both flip sign); the *product* ``B @ A`` is
  invariant, which is the only object the reconstruction uses.
* ``SVD_rank(W - Q_t)`` is rank-deficient whenever ``W - Q_t`` has numerical rank ``< rank`` (e.g.
  ``rank`` above the intrinsic rank of the quantized residual, or a residual that is exactly
  low-rank). The truncated SVD then returns zero singular directions and ``B @ A`` no longer spans
  ``rank`` dimensions; the product is still the Eckart-Young best rank-``rank`` approximation.
* Quantization here is float *simulated* quantization (measurement class 2): the returned factors
  are float tensors and this function makes no storage or latency claim
  (``AGENTS.md`` sections 4.3-4.4).
* Deterministic on CPU float32/float64: the dense SVD and the quantizer are deterministic, so a
  fixed ``(w, rank, spec, iterations)`` reproduces the same factors bit-for-bit.

Execution scope: CPU only, float32/float64 only, no network, no GPU, ``torch.no_grad()``.
"""

from __future__ import annotations

import torch

from spectraquant.factorization.initialization import initialize_svd
from spectraquant.quantization.fake_quant import QuantSpec, fake_quantize

__all__ = ["loftq_initialise"]


def _validate_iterations(iterations: int) -> None:
    """Validate the alternation count: an ``int >= 1`` (bool is rejected as a non-int)."""
    if isinstance(iterations, bool) or not isinstance(iterations, int):
        raise TypeError(f"iterations must be an int, got {type(iterations).__name__}")
    if iterations < 1:
        raise ValueError(
            f"iterations must be >= 1, got {iterations}: a single iteration is only the first "
            "residual step of the LoftQ alternating schedule, so 'no alternation' is not a "
            "supported input"
        )


def loftq_initialise(
    w: torch.Tensor,
    *,
    rank: int,
    spec: QuantSpec,
    iterations: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Alternating quantize/SVD initialization of the LoftQ low-rank half (paper Eq. 7-8).

    Shapes:
        * ``w``: ``(out_features, in_features)`` float32/float64 CPU tensor.
        * ``rank``: requested rank, clamped to ``min(rank, min(w.shape))`` by the underlying
          :func:`~spectraquant.factorization.initialization.initialize_svd` (a request larger than
          the intrinsic rank means "keep everything" rather than an error; ``rank=0`` is allowed
          and returns empty factors, which quantizes ``W`` with no low-rank compensation).
        * returns ``(A, B)`` with ``A`` of shape ``(rank_effective, in_features)`` and ``B`` of
          shape ``(out_features, rank_effective)`` — the frozen ``W ~= B @ A`` convention.

    Dtypes / device:
        ``A`` and ``B`` have ``w``'s dtype (float32 in production) and live on CPU.

    Determinism:
        Runs under :func:`torch.no_grad`; the returned factors are constants, not graph leaves.
        A fixed ``(w, rank, spec, iterations)`` is bit-reproducible on CPU.

    Assumptions / limitations:
        ``spec`` is interpreted against ``w``'s axes exactly as in
        :func:`~spectraquant.quantization.fake_quant.fake_quantize` (``spec.axis`` selects the
        block axis). The companion quantized base is **not** returned; see the module docstring for
        the recomputation from ``(A, B)`` and for the difference between one iteration and the
        paper's alternating schedule. SVD sign ambiguity is irrelevant to the product; a quantized
        residual of numerical rank ``< rank`` makes the returned factors rank-deficient. Class 2
        (float simulation of quantization numerics), never class 3/4.

    Raises:
        TypeError: ``iterations`` is not an int, or ``w`` is not a tensor.
        ValueError: ``iterations < 1``; ``w`` is not a 2-D CPU float tensor; ``rank`` is negative;
            or ``w`` is empty (no quantizable block).

    Example:
        >>> w = torch.tensor([[0.9, 0.2], [0.2, 0.6]])
        >>> spec = QuantSpec(2, "per_tensor", None, True)
        >>> a, b = loftq_initialise(w, rank=1, spec=spec)
        >>> a.shape, b.shape
        (torch.Size([1, 2]), torch.Size([2, 1]))
    """
    _validate_iterations(iterations)
    # Validates w (2-D, CPU, float32/float64) and rank (>= 0), and clamps rank to min(w.shape).
    factors = initialize_svd(w, rank)
    a, b = factors.A, factors.B

    with torch.no_grad():
        for _ in range(iterations):
            # Eq. 7: quantize the residual left by the current factors.
            quantized = fake_quantize(w - b @ a, spec)
            # Eq. 8: SVD of the residual of the quantization, R_t = W - Q_t.
            factors = initialize_svd(w - quantized, rank)
            a, b = factors.A, factors.B

    return a, b
