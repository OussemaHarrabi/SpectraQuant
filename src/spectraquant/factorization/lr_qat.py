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
* It does not implement the *training-time* grid-aware variants of the upstream code, which add the
  auxiliary weight **before** rounding in integer space (``s * clip(round(W_int + AB), q_min,
  q_max)``, ``lora_method='lr_qat_int'``) or in the φ down-cast domain (``lr_qat_fp16``/
  ``lr_qat_bf16``/``lr_qat_fixed_point8``). The φ domain's **merge** — the arithmetic evaluated
  once, after training — *is* implemented here (:func:`merge_lr_qat`, :func:`fixed_point8_downcast`,
  :func:`fixed_point8_upcast`); the training loop that owns the factors and the learned step size
  ``s`` is not.
* It adds no byte accounting. :mod:`spectraquant.quantization.accounting` is the single
  byte-accounting source of truth (its module docstring, invariant 1), and
  :func:`~spectraquant.factorization.decomposition.factor_bytes` already covers the factor size;
  duplicating that formula here would create a second source of truth.

Grid-aware φ-variant and its merge (R2-LRQAT)
---------------------------------------------
The R2-LRQAT arm (``docs/research/reproduction-plan.md`` §3.2) keeps the main weight in the paper's
fixed-point φ₀ container and adds the auxiliary term **inside** the rounding, where ``s`` is a
learned step size, ``alpha`` the LoRA scaling numerator and ``r`` the rank:

.. code-block:: text

    Phi_0 = fixed_point8_downcast(W / s)                                  # Q4.4 int8 container
    W_int = clip(round(Phi_0 + (alpha / r) * (B @ A)), q_min, q_max)      # 4-bit signed codes
    W_hat = s * W_int                                                     # dequantized weight

Merging (upstream ``lora_merge_weights`` / ``get_params``) replaces the whole training-time module
by the integer codes plus the scalar ``s``; nothing changes numerically, which is the paper's "no
inference overhead" claim. :func:`merge_lr_qat` performs exactly that fold and returns both the
integer codes and the dequantized weight, so the merge identity can be checked by **exact integer
equality** (``docs/research/reproduction-plan.md`` §5.3, gate T1-a).

Measurement class: the quantized main weight is *float* dequantized storage (class 2). The low-rank
auxiliary weight is a **training-memory economy, not a compression reason** — the upstream method
fuses it away at inference, so nothing here may be reported as low-bit storage or as accelerated
inference (``AGENTS.md`` sections 4.3-4.4). :func:`merge_lr_qat` likewise produces float codes in
simulation: it is not a packed container and makes no class-3/class-4 claim.

Numerical limitations
---------------------
* SVD sign ambiguity: the individual factor signs are implementation-defined; only the product
  ``B @ A`` is meaningful and it is invariant.
* If ``rank`` exceeds the numerical rank of ``W - Wq`` (e.g. ``Wq == W`` exactly, or a residual
  whose rank is smaller), the truncated SVD returns zero singular directions: the *product* stays
  the Eckart-Young best rank-``rank`` approximation, but the factors are rank-deficient.
* Deterministic on CPU float32/float64: dense SVD plus the deterministic quantizer reproduce the
  same pair bit-for-bit for a fixed ``(w, rank, spec)``.
* Rounding is ``torch.round`` (ties-to-**even**) everywhere, matching the upstream
  ``round_ste``/``torch.round``. In the φ domain the base itself carries only 4 fractional bits
  (:data:`Q44_INTEGER_BITS`), so a caller that folds the adapter in loses the sub-``2**-4`` part of
  ``Phi_0`` *before* the final 4-bit rounding — that loss is a property of the paper's container,
  not a defect of the merge.

Execution scope: CPU only, float32/float64 only, no network, no GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from spectraquant.factorization.decomposition import _validate_matrix, _validate_rank
from spectraquant.factorization.initialization import initialize_svd
from spectraquant.quantization.fake_quant import QuantSpec, fake_quantize

__all__ = [
    "FIXED_POINT8_BITS",
    "Q44_INTEGER_BITS",
    "LrQatMerge",
    "effective_weight",
    "fixed_point8_downcast",
    "fixed_point8_upcast",
    "lr_qat_pair",
    "merge_lr_qat",
]

#: Total bits of the paper's fixed-point φ₀ container (upstream ``lr_qat_fixed_point8``,
#: ``quantization/hijacker.py``): an **8-bit** signed integer holding ``2**(8 - integer_bits)``
#: fractional steps.
FIXED_POINT8_BITS = 8

#: Frozen integer-bit count of the R2-LRQAT φ₀ downcast
#: (``docs/research/reproduction-plan.md`` §3.2): 4 integer + 4 fractional bits — "**Q4.4**" —
#: i.e. resolution ``2**-4`` over the 4-bit code domain ``[-8, 7]``.
Q44_INTEGER_BITS = 4


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


# --------------------------------------------------------------------------------------------------
# The φ₀ fixed-point container ("Q4.4") and the merged integer weight (R2-LRQAT, gate T1-a)
# --------------------------------------------------------------------------------------------------


def _validate_integer_bits(integer_bits: int) -> int:
    """Validate the integer-bit count of the 8-bit φ₀ container and return it unchanged."""
    if isinstance(integer_bits, bool) or not isinstance(integer_bits, int):
        raise TypeError(f"integer_bits must be an int, got {type(integer_bits).__name__}")
    if not 1 <= integer_bits <= FIXED_POINT8_BITS:
        raise ValueError(f"integer_bits must be in [1, {FIXED_POINT8_BITS}], got {integer_bits}")
    return integer_bits


def _code_domain(integer_bits: int) -> tuple[int, int]:
    """Signed symmetric code range ``(qmin, qmax)`` of an ``integer_bits``-wide code."""
    return -(1 << (integer_bits - 1)), (1 << (integer_bits - 1)) - 1


def _validate_phi0_codes(codes: torch.Tensor) -> None:
    """Validate a φ₀ container tensor: 2-D, non-empty, CPU, real (integer or float) dtype."""
    if not isinstance(codes, torch.Tensor):
        raise TypeError(f"phi0_codes must be a torch.Tensor, got {type(codes).__name__}")
    if codes.ndim != 2:
        raise ValueError(f"phi0_codes must be 2-D, got {codes.ndim} dimension(s)")
    if codes.numel() == 0:
        raise ValueError("phi0_codes must not be empty")
    if codes.device.type != "cpu":
        raise ValueError(f"phi0_codes is on device {codes.device}; only CPU is supported")
    if codes.dtype.is_complex or codes.dtype == torch.bool:
        raise ValueError(f"phi0_codes must have a real integer or float dtype, got {codes.dtype}")


def fixed_point8_downcast(x: torch.Tensor, *, integer_bits: int = Q44_INTEGER_BITS) -> torch.Tensor:
    """Down-cast φ₀ into the paper's 8-bit fixed-point container ("**Q4.4**" by default).

    Reproduces the upstream ``lr_qat_fixed_point8`` down-cast
    (``round(clamp(x, q_min, q_max) * 2**(8 - integer_bits)).to(int8)``,
    ``quantization/hijacker.py:156-180``): the value is first clamped to the *weight* code domain
    ``[q_min, q_max] = [-2**(integer_bits - 1), 2**(integer_bits - 1) - 1]``, then scaled by
    ``2**(FIXED_POINT8_BITS - integer_bits)`` and rounded to the nearest integer.

    Args:
        x: the base weight **in step-size units**, i.e. ``Phi_0 = W / s`` (the paper folds the
            learned step size into this domain). Float tensor, any shape (this function is
            element-wise; the 2-D restriction is imposed by :func:`merge_lr_qat` and by the
            training loop).
        integer_bits: integer bits of the container. ``4`` (the frozen default) is Q4.4 — 4 integer
            + 4 fractional bits, resolution ``2**-4``.

    Shapes / dtypes / device:
        Returns an ``torch.int8`` tensor of ``x``'s shape and device. The container is always 8
        bits wide; only the binary point moves with ``integer_bits``.

    Range:
        ``[q_min * 2**(8 - integer_bits), q_max * 2**(8 - integer_bits)]``. For Q4.4 that is
        ``[-128, 112]``, so every representable value fits ``int8`` and the final cast cannot
        saturate for any valid input (the explicit clip to ``int8`` is therefore inert defence).

    Rounding:
        ``torch.round`` — half-to-**even** (banker's), identical to the upstream ``round_ste``
        wrapper around ``torch.round``. ``x`` must be finite; NaN/inf are not validated because the
        paper's container has no representation for them.

    Assumptions / limitations:
        This is an element-wise *simulation* of the container (measurement class 2 in the sense of
        ``AGENTS.md`` §5): it returns float-backed ``int8`` codes, not a packed artifact, and makes
        no storage or latency claim. Only the frozen default ``integer_bits=4`` is used by the M3
        reproduction plan; the other widths are provided because the upstream code parametrises
        them.

    Raises:
        TypeError: ``x`` is not a tensor, or ``integer_bits`` is not an int.
        ValueError: ``integer_bits`` is outside ``[1, 8]``.
    """
    _validate_integer_bits(integer_bits)
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")
    qmin, qmax = _code_domain(integer_bits)
    scale = 1 << (FIXED_POINT8_BITS - integer_bits)
    scaled = torch.clamp(x, float(qmin), float(qmax)) * float(scale)
    return torch.clamp(torch.round(scaled), -128.0, 127.0).to(torch.int8)


def fixed_point8_upcast(
    codes: torch.Tensor, *, integer_bits: int = Q44_INTEGER_BITS
) -> torch.Tensor:
    """Up-cast an 8-bit fixed-point φ₀ container back to the code domain (inverse of the down-cast).

    Reproduces the upstream up-cast ``x / 2**(8 - integer_bits)``
    (``quantization/hijacker.py:156-180``).

    Shapes / dtypes / device:
        ``codes``: the ``torch.int8``-style tensor produced by :func:`fixed_point8_downcast` (any
        real integer or float dtype is accepted). Returns a float tensor of the same shape and
        device: ``float64`` when ``codes`` is ``float64``, otherwise ``float32``.

    Assumptions / limitations:
        Exactly the inverse of :func:`fixed_point8_downcast` on its range — the map is injective,
        so ``upcast(downcast(x))`` is the nearest Q-integer-bits grid point of ``x`` and
        ``downcast(upcast(c)) == c`` for every in-range ``c``. No clamping is applied on the way up
        (out-of-container inputs a caller supplied by hand are returned as-is rather than silently
        saturated).

    Raises:
        TypeError: ``codes`` is not a tensor, or ``integer_bits`` is not an int.
        ValueError: ``integer_bits`` is outside ``[1, 8]``.
    """
    _validate_integer_bits(integer_bits)
    if not isinstance(codes, torch.Tensor):
        raise TypeError(f"codes must be a torch.Tensor, got {type(codes).__name__}")
    scale = float(1 << (FIXED_POINT8_BITS - integer_bits))
    dtype = torch.float64 if codes.dtype == torch.float64 else torch.float32
    return codes.to(dtype) / scale


@dataclass(frozen=True)
class LrQatMerge:
    """The result of folding a trained LR-QAT φ-adapter into the stored weight (paper Eq. merge).

    Attributes:
        codes: the merged integer weight matrix ``clip(round(Phi_0 + (alpha / r) * B @ A))``,
            ``torch.int8``, shaped like φ₀ (``(out_features, in_features)``). This is what a packed
            container would store, so the merge identity can be checked by exact integer equality
            (``docs/research/reproduction-plan.md`` §5.3, gate T1-a).
        weight: the dequantized weight ``s * codes`` (float, ``codes``' shape) — the weight the
            merged module's forward pass uses.

    Dtypes / device:
        ``codes`` is ``torch.int8``; ``weight`` follows the promotion of φ₀ and ``B @ A`` (float32
        for float32 inputs). CPU only.

    Assumptions / limitations:
        Float simulation (class 2), not a packed artifact; ``weight`` is *not* a proof of low-bit
        storage or of any kernel behaviour. ``codes`` alone loses the sub-``2**-integer_bits``
        content that φ₀ could not hold in the first place.
    """

    codes: torch.Tensor
    weight: torch.Tensor


def merge_lr_qat(
    phi0_codes: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    step_size: float,
    alpha: float = 1.0,
    rank: int | None = None,
    integer_bits: int = Q44_INTEGER_BITS,
) -> LrQatMerge:
    """Fold a trained LR-QAT φ-adapter into the stored integer weight (upstream ``lora_merge_weights``).

    Implements the paper's merge (``docs/research/reproduction-plan.md`` §3.2, gate T1-a):

    .. code-block:: text

        Phi_0       = fixed_point8_upcast(phi0_codes)              # Q4.4 -> code-domain float
        merged_codes = clip(round(Phi_0 + (alpha / rank) * (B @ A)), q_min, q_max)
        W_merged     = step_size * merged_codes

    which is the *same* arithmetic that the training-time φ-forward (``_apply_lora_qat_w_phi``)
    performs per step, evaluated once: the auxiliary term is added **inside** the rounding, and the
    result is an integer code matrix scaled by ``s`` (hence "merging costs nothing").

    Args:
        phi0_codes: the φ₀ container of the trained layer, shaped ``(out_features, in_features)``.
            An integer tensor (e.g. :func:`fixed_point8_downcast` output) or any real float tensor;
            values are interpreted at the container's ``1 / 2**(8 - integer_bits)`` resolution.
        a: auxiliary factor, ``(rank, in_features)`` — frozen ``W ~= B @ A`` convention.
        b: auxiliary factor, ``(out_features, rank)``.
        step_size: the learned step size ``s`` (a positive finite scalar).
        alpha: LoRA scaling numerator ``alpha`` in ``(alpha / rank) * B @ A``; finite scalar. The
            paper fixes ``alpha = 1.0``.
        rank: the ``r`` used in the scaling. ``None`` (default) takes it from the factors
            (``a.shape[0] == b.shape[1]``); a value that disagrees with the factors is rejected
            rather than silently re-scaling the adapter. ``rank=0`` (empty factors) is the package's
            "no compensation" convention and folds only φ₀ — ``alpha / rank`` is never evaluated.
        integer_bits: integer bits of the φ₀ container and of the merged code domain
            (``4`` = Q4.4, the frozen default).

    Returns:
        :class:`LrQatMerge` with ``codes`` (``torch.int8``, φ₀'s shape, integer path) and ``weight``
        (``s * codes``, float, the dequantized merged weight).

    Dtypes / device:
        CPU only. ``codes`` is ``torch.int8``; ``weight`` has the promotion dtype of the upcast φ₀
        and ``b @ a`` (float32 for the production float32 path).

    Assumptions / limitations:
        * Float simulation (measurement class 2). The returned ``codes`` are not a packed container;
          no class-3 (storage) or class-4 (kernel) claim follows from this function.
        * ``s`` is a single scalar here. The R2 arm's W4-g128 configuration learns one scale per
          group; folding a per-group scale into this call is the training loop's responsibility and
          is not implemented (the merge identity is stated per scalar ``s`` in §3.2/§5.3).
        * The integer path is exact by construction, which is what gate T1-a checks: the same
          ``clip``/``round`` over the same floats on both paths gives bit-identical ``int8`` codes.
          The fp path compares ``s * (x @ codesᵀ)`` with ``x @ (s * codes)ᵀ`` and therefore differs
          only by fp32 accumulation order (~1e-7 relative), not by any rounding difference.
        * ``phi0_codes`` must already lie in the container; the upcast does not re-clip, so a caller
          that hand-writes out-of-range codes gets them clipped only at the final 4-bit rounding.

    Raises:
        TypeError: an argument has the wrong type (``step_size``/``alpha`` must be real scalars,
            ``rank`` an int when given).
        ValueError: shape inconsistency between ``phi0_codes``, ``a`` and ``b``; ``rank`` disagrees
            with the factors; ``step_size`` is not finite and positive; ``alpha`` is not finite;
            ``integer_bits`` outside ``[1, 8]``.

    Example:
        >>> w = torch.tensor([[0.9, 0.2], [0.2, 0.6]])
        >>> phi0 = fixed_point8_downcast(w / 0.125)
        >>> a = torch.zeros(1, 2)
        >>> b = torch.zeros(2, 1)
        >>> merged = merge_lr_qat(phi0, a, b, step_size=0.125)
        >>> merged.codes.dtype, merged.weight.shape
        (torch.int8, torch.Size([2, 2]))
    """
    _validate_integer_bits(integer_bits)
    _validate_phi0_codes(phi0_codes)
    _validate_matrix(a, name="a")
    _validate_matrix(b, name="b")
    if b.shape[1] != a.shape[0]:
        raise ValueError(
            f"a and b must agree on the rank: a{tuple(a.shape)} (rank, in) and b{tuple(b.shape)} "
            f"(out, rank), got rank {a.shape[0]} vs {b.shape[1]}"
        )
    if a.shape[1] != phi0_codes.shape[1] or b.shape[0] != phi0_codes.shape[0]:
        raise ValueError(
            f"factors do not match phi0_codes{tuple(phi0_codes.shape)}: expected "
            f"a(rank, {phi0_codes.shape[1]}) and b({phi0_codes.shape[0]}, rank), got "
            f"a{tuple(a.shape)} and b{tuple(b.shape)}"
        )

    effective_rank = a.shape[0]
    if rank is None:
        rank = effective_rank
    elif isinstance(rank, bool) or not isinstance(rank, int):
        raise TypeError(f"rank must be an int or None, got {type(rank).__name__}")
    elif rank != effective_rank:
        raise ValueError(
            f"rank={rank} disagrees with the factors (a has {effective_rank} rows); the scaling "
            f"would change silently, so it is rejected"
        )

    if isinstance(step_size, bool) or not isinstance(step_size, (int, float)):
        raise TypeError(f"step_size must be a real scalar, got {type(step_size).__name__}")
    if not math.isfinite(float(step_size)) or float(step_size) <= 0.0:
        raise ValueError(f"step_size must be finite and > 0, got {step_size!r}")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError(f"alpha must be a real scalar, got {type(alpha).__name__}")
    if not math.isfinite(float(alpha)):
        raise ValueError(f"alpha must be finite, got {alpha!r}")

    phi0 = fixed_point8_upcast(phi0_codes, integer_bits=integer_bits)
    # ``rank == 0`` is the frozen convention for "no compensation" (see `lr_qat_pair` and
    # `loftq_initialise`): the adapter term is empty, so ``b @ a`` is the zero matrix and the
    # scaling ``alpha / rank`` would divide by zero. Fold nothing.
    z = phi0 if effective_rank == 0 else phi0 + (float(alpha) / rank) * (b @ a)
    qmin, qmax = _code_domain(integer_bits)
    codes = torch.clamp(torch.round(z), float(qmin), float(qmax)).to(torch.int8)
    weight = codes.to(dtype=z.dtype) * float(step_size)
    return LrQatMerge(codes=codes, weight=weight)
