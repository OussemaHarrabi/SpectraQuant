"""LR-QAT initialisation: quantized main weight plus a low-rank auxiliary weight (Milestone 3).

Reference: Bondarenko et al., *Low-Rank Quantization-Aware Training for LLMs* (LR-QAT),
arXiv:2406.06385. Upstream: ``Qualcomm-AI-research/LR-QAT`` at ``8795afe0…``
(``docs/research/upstream-lockfile.md``), whose ``QuantizedLinear.forward``/``_apply_lora_qat_*``
and ``lora_merge_weights`` path implement the mechanism reproduced here.

Central mechanism
-----------------
LR-QAT keeps a **quantized main weight** ``Wq = q_N(W)`` and adds a **trainable low-rank
auxiliary weight** ``B @ A`` to it during training:

.. code-block:: text

    W_eff = Wq + B @ A                 # this module's `effective_weight`
    Wq    = fake_quantize(W, spec)     # the main weight, class-2 simulated quantization

so the quantization error ``W - Wq`` is compensated by a term the optimizer can train. The
initialisation that makes step 0 already better than plain quantization is the rank-``rank`` SVD of
that error, ``(A, B) <- SVD_rank(W - Wq)`` — the low-rank part of the *same* residual the later
training steps keep refining.

Convention: this repository's frozen ``W ~= B @ A`` with ``A: (rank, in_features)``,
``B: (out_features, rank)`` (``factorization/decomposition.py``). The upstream code stores the
auxiliary weight as ``lora_A @ lora_Bt`` scaled by ``lora_scaling``; no scaling factor is applied
here — :func:`effective_weight` is deliberately the unscaled sum, and a caller that wants the
upstream scaling multiplies its own factors before calling it.

What this module does *not* do
------------------------------
* It does not implement the grid-aware variants of the upstream code, which add the auxiliary
  weight **before** rounding in integer space (``s * clip(round(W_int + AB), q_min, q_max)``,
  ``lora_method='lr_qat_int'``) or in the φ down-cast domain (``lr_qat_fp16``/``lr_qat_bf16``/
  ``lr_qat_fixed_point8``). Here the auxiliary term is added to the already-dequantized weight.
* It does not fuse the auxiliary weight into the quantized weight (upstream
  ``lora_merge_weights``). Fusing changes the storage story, not the weights, and belongs to the
  training loop that owns the factors.
* It adds no byte accounting. :mod:`spectraquant.quantization.accounting` is the single
  byte-accounting source of truth (its module docstring, invariant 1), and
  :func:`~spectraquant.factorization.decomposition.factor_bytes` already covers the factor size;
  duplicating that formula here would create a second source of truth.

Measurement class: the quantized main weight is *float* dequantized storage (class 2). The low-rank
auxiliary weight is a **training-memory economy, not a compression reason** — the upstream method
fuses it away at inference, so nothing here may be reported as low-bit storage or as accelerated
inference (``AGENTS.md`` sections 4.3-4.4).

Numerical limitations
---------------------
* SVD sign ambiguity: the individual factor signs are implementation-defined; only the product
  ``B @ A`` is meaningful and it is invariant.
* If ``rank`` exceeds the numerical rank of ``W - Wq`` (e.g. ``Wq == W`` exactly, or a residual
  whose rank is smaller), the truncated SVD returns zero singular directions: the *product* stays
  the Eckart-Young best rank-``rank`` approximation, but the factors are rank-deficient.
* Deterministic on CPU float32/float64: dense SVD plus the deterministic quantizer reproduce the
  same pair bit-for-bit for a fixed ``(w, rank, spec)``.

Execution scope: CPU only, float32/float64 only, no network, no GPU.
"""

from __future__ import annotations

import torch

from spectraquant.factorization.decomposition import _validate_matrix, _validate_rank
from spectraquant.factorization.initialization import initialize_svd
from spectraquant.quantization.fake_quant import QuantSpec, fake_quantize

__all__ = ["effective_weight", "lr_qat_pair"]


def lr_qat_pair(
    w: torch.Tensor,
    *,
    rank: int,
    spec: QuantSpec,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Quantize ``w`` and initialise the LR-QAT auxiliary factors from its quantization error.

    Shapes:
        * ``w``: ``(out_features, in_features)`` float32/float64 CPU tensor.
        * ``rank``: requested auxiliary rank, clamped to ``min(rank, min(w.shape))`` by the
          underlying :func:`~spectraquant.factorization.initialization.initialize_svd`
          (``rank=0`` returns empty factors, i.e. no compensation).
        * returns ``(Wq, (A, B))`` with ``Wq`` shaped like ``w``, ``A`` of shape
          ``(rank_effective, in_features)`` and ``B`` of shape ``(out_features, rank_effective)`` —
          the frozen ``W ~= B @ A`` convention, so ``Wq + B @ A`` is the trained representation
          (see :func:`effective_weight`).

    Dtypes / device:
        ``Wq``, ``A`` and ``B`` have ``w``'s dtype (float32 in production) and live on CPU.

    Determinism:
        Runs under :func:`torch.no_grad` (an initializer, so the returned ``Wq``/``A``/``B`` are
        detached constants and no graph through the source weight is retained; wrap the factors in
        :class:`torch.nn.Parameter` to train them, exactly as upstream LR-QAT does). The
        quantization parameters are derived from ``w`` itself (no calibration data), and the SVD is
        the deterministic dense CPU LAPACK routine, so a fixed ``(w, rank, spec)`` is
        bit-reproducible.

    Assumptions / limitations:
        ``spec`` is interpreted against ``w``'s axes exactly as in
        :func:`~spectraquant.quantization.fake_quant.fake_quantize` (``spec.axis`` selects the block
        axis). ``Wq`` is float dequantized storage (measurement class 2), never low-bit storage.
        The returned factors minimise ``||(W - Wq) - B @ A||_F`` at ``rank`` (Eckart-Young), so
        ``||W - (Wq + B @ A)||_F`` is strictly smaller than ``||W - Wq||_F`` whenever ``rank > 0``
        and ``W - Wq`` is non-zero. SVD sign ambiguity is irrelevant to the product; a
        rank-deficient residual makes the factors rank-deficient.

    Raises:
        TypeError: ``w`` is not a tensor, or ``rank`` is not an int.
        ValueError: ``w`` is not a 2-D CPU float tensor, ``rank`` is negative, or ``w`` is empty
            (no quantizable block).

    Example:
        >>> w = torch.tensor([[0.95, 0.15], [0.15, 0.55]])
        >>> spec = QuantSpec(2, "per_tensor", None, True)
        >>> wq, (a, b) = lr_qat_pair(w, rank=1, spec=spec)
        >>> wq.shape, a.shape, b.shape
        (torch.Size([2, 2]), torch.Size([1, 2]), torch.Size([2, 1]))
    """
    _validate_matrix(w)
    _validate_rank(rank)

    with torch.no_grad():
        wq = fake_quantize(w, spec)
        # `initialize_svd` re-clamps rank and validates the residual as a 2-D CPU float tensor.
        factors = initialize_svd(w - wq, rank)
    return wq, (factors.A, factors.B)


def effective_weight(wq: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """The trained LR-QAT representation ``Wq + B @ A`` (the weight the forward pass uses).

    Shapes:
        * ``wq``: ``(out_features, in_features)`` — the (dequantized) quantized main weight.
        * ``a``: ``(rank, in_features)`` — auxiliary factor (frozen ``W ~= B @ A`` convention).
        * ``b``: ``(out_features, rank)`` — auxiliary factor; ``b @ a`` must be shaped like ``wq``.
        * returns ``(out_features, in_features)``.

    Dtypes / device:
        The result follows PyTorch type promotion of ``(wq, b, a)``; on CPU with float32 inputs it
        is float32. Graphs are preserved: this is a differentiable function of ``a`` and ``b`` and
        is safe to call inside a training step.

    Assumptions / limitations:
        * ``wq`` is already a *dequantized* float tensor (class 2). The sum is computed in float, in
          the natural dtype: a low-bit kernel that never materialises ``Wq + B @ A`` may execute a
          different numerics, and no kernel-equivalence claim is made here.
        * This is the **training-time** representation. LR-QAT fuses the auxiliary weight into the
          quantized weights before inference, so a stored/deployed byte count must come from the
          fused artifact, not from this sum.
        * Shape mismatches raise rather than broadcasting: a silent broadcast would hide a
          transposed factor, which is the failure mode this convention is most exposed to.

    Raises:
        TypeError: an argument is not a tensor.
        ValueError: shapes are not ``(out, in)``, ``(rank, in)``, ``(out, rank)`` consistently.
    """
    for name, tensor in (("wq", wq), ("a", a), ("b", b)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    _validate_matrix(wq, name="wq")
    if b.ndim != 2 or a.ndim != 2:
        raise ValueError(f"a and b must both be 2-D, got a{tuple(a.shape)} and b{tuple(b.shape)}")
    if b.shape[0] != wq.shape[0] or a.shape[1] != wq.shape[1] or b.shape[1] != a.shape[0]:
        raise ValueError(
            f"expected wq{tuple(wq.shape)}, a(rank, in) and b(out, rank) consistent "
            f"(out={wq.shape[0]}, in={wq.shape[1]}), got a{tuple(a.shape)} and b{tuple(b.shape)}"
        )
    return wq + b @ a
