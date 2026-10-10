"""2-bit NF-style codebook fake quantization and its uniform-int2 control (M3/R1 arms).

Measurement **class 2** (``AGENTS.md`` section 5): everything here runs in float arithmetic and
*simulates* the numerics of a 2-bit codebook lookup. A tensor produced by this module is still a
float tensor — it is never low-bit **storage** (class 3) and never accelerated inference
(class 4). The byte cost of a configuration is computed analytically from shapes only
(:func:`codebook_accounted_bytes`, class 1) and never presented as measured bytes.

Why this module exists
----------------------
The frozen M3 LoftQ arms (``docs/research/reproduction-plan.md`` section 3.1) quantize the base
weight with an **"NF-style codebook, block 64"** — the paper's NF2 — while the uniform-int2 control
uses the same 2-bit budget with a uniform code grid. Neither is expressible through
:class:`spectraquant.quantization.fake_quant.QuantSpec`, whose codes are uniformly spaced integers
with an affine (scale, zero-point) map: a NormalFloat codebook is *non-uniformly* spaced, and its
quantizer is a nearest-level lookup with a per-block absmax scale. This module adds exactly that,
reusing the existing :mod:`~spectraquant.quantization.fake_quant` and
:mod:`~spectraquant.quantization.accounting` modules rather than forking them.

Reference (pinned, not vendored)
--------------------------------
``yxli2123/LoftQ`` @ ``ae33fd4fd05fd4ba146555cd77c13d307eb4e9b3`` (MIT),
``glue/utils_qaunt.py`` — the CPU-readable reference the frozen plan names for the T2-a oracle
(``docs/research/upstream-notes.md`` section 3.2; the LLM path of that repository delegates to PEFT,
whose current ``loftq_init`` supports only ``num_bits in {4, 8}``, requires bitsandbytes and a CUDA
device, and has no ``method``/``block_size`` parameters — it is **not** this oracle). No upstream
source is copied or imported here; the numbers live in
``artifacts/sample-results/m3-oracle/loftq-nf2-block64.json`` and are reproduced by
``tests/unit/test_codebook.py``.

The reference's two quantizers, reproduced exactly
--------------------------------------------------
1. **NormalFloat** (``create_normal_map(offset=0.9677083, symmetric=False, num_bits=2)`` plus
   ``quant_nf4_block``): the **asymmetric** QLoRA-default table. Its four 2-bit levels are the
   sorted, max-normalized values of ``{ppf(offset), ppf((offset + 0.5) / 2), 0, -ppf(offset)}``
   where ``ppf`` is the standard-normal quantile function. The reference normalizes by the largest
   magnitude (``values /= values.max()``), so the table spans ``[-1, 1]`` with an exact ``0`` —
   **not symmetric about zero** (``[-1.0, 0.0, 0.3379152417, 1.0]`` at 2 bits; the asymmetry is
   what makes the QLoRA table a better match to a normal weight distribution than a symmetric one).
   Quantization is a **nearest-level lookup by absolute difference** (``argmin``, so a tie resolves
   to the *lowest* level index) after dividing the block by its absmax, with a **per-block** scale
   of ``max|block|`` (a block is 64 consecutive elements of the row-major flatten).
2. **Uniform-int2 control** (``quant_uniform(input, num_bits=2, clip_val)``): a **per-tensor affine**
   quantizer, not a codebook — clip to ``clip_val`` (the dispatcher's default is
   ``mean ± 2·std``), then ``round((x − min)/(max − min + 1e-8) · 3)/3 · (max − min + 1e-8) + min``.
   Its code grid is therefore the one-sided ``{0, 1/3, 2/3, 1}``, and
   :func:`fake_quantize_uniform_int2` reproduces it verbatim.

   The **codebook** form of the control — the same 2-bit budget and the same absmax machinery as the
   NF arm, with only the table changed — is ``CodebookSpec(kind="uniform")`` with the table
   :func:`uniform_int2_levels` returns, ``[-1, -1/3, 1/3, 1]``. It is symmetric and carries no zero:
   with four uniformly spaced levels spanning ``[-1, 1]`` that is the only table containing both
   extremes, which is exactly what :func:`fake_quantize_codebook` needs for an idempotent round trip.
   The two grids are deliberately different objects — one is an affine quantizer's rounding grid, the
   other a codebook table — and are never interchanged.

Numerical conventions (deliberate, matching the reference bit-for-bit)
---------------------------------------------------------------------
* The level table is built in the reference's own dtype order: the **probabilities** come from a
  float32 ``torch.linspace`` (``torch.linspace``'s default dtype in the reference), the quantile
  ``ppf`` is evaluated in float64, the values are cast to **float32**, sorted, and divided by their
  float32 maximum. Building the table in float64 instead shifts the 2-bit middle level by one
  float32 ulp (``0.33791527…`` instead of ``0.33791524…``) and breaks the oracle tolerance; see
  ``tests/unit/test_codebook.py``.
* ``ppf`` is evaluated with ``torch.special.erfinv`` — ``Φ⁻¹(p) = √2 · erfinv(2p − 1)`` — because
  scipy is not a project dependency. The reference calls ``scipy.stats.norm.ppf``; both agree to
  float64 round-off, far below the float32 cast.
* Work dtype is float32 for float32 input and float64 for float64 input (the
  :func:`~spectraquant.quantization.fake_quant.fake_quantize` convention). The reference is
  float32 throughout, so an exact oracle comparison uses float32 inputs.
* The codebook dequantizes as ``level · (block_absmax / max|level|)``; since every table is
  normalized to ``max|level| = 1`` the scale is the block absmax, as in the reference.

Blocking (and the one deliberate generalization)
-----------------------------------------------
A block is ``block_size`` **consecutive elements of the tensor flattened with ``spec.axis`` moved
to the last position** — ``w.movedim(axis, -1).reshape(-1)`` cut into ``ceil(numel / block_size)``
chunks. With the default ``axis=-1`` this is exactly the reference's
``weight.resize(numel // block_size, block_size)`` row-major flatten-and-cut, so a block may span
more than one row of a matrix (for a 64-element 8x8 weight, the reference and this module both
produce **one** 64-element block). Two differences from the reference are deliberate and documented:
the reference's in-place ``Tensor.resize`` requires ``numel % block_size == 0`` and raises otherwise,
whereas a trailing short block is accepted here and quantized against its own absmax; and an
all-zero block yields exactly ``0`` here, whereas the reference divides by a zero absmax and returns
``NaN``.

Numerical limitations
---------------------
* The nearest-level rule cannot represent a value exactly between two levels: the lower index wins
  (``torch.argmin``), so such inputs are biased downward by half a level gap.
* Accuracy is bounded by the table: the 2-bit NF table's widest gap is 1.0, so a block whose values
  spread over ``[-absmax, absmax]`` can carry an element error of up to ``absmax/2``.
* The implementation materializes an ``(numel / block_size, block_size, n_levels)`` difference
  tensor — deliberate (clarity, no hidden broadcasts) and adequate for the Tier-0 fixtures and the
  ≤1B-parameter arms in scope; it is not a memory-optimised kernel.
* ``block_absmax`` is ``NaN``/``inf`` for non-finite input; non-finite weights are not supported.

Execution scope: CPU only, float32/float64 only, no network, no GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from spectraquant.quantization.accounting import byte_breakdown
from spectraquant.quantization.fake_quant import SUPPORTED_BITS, QuantSpec

__all__ = [
    "CODEBOOK_KINDS",
    "CODEBOOK_SCALE_BYTES",
    "FROZEN_CODEBOOK_BITS",
    "NF_OFFSET",
    "CodebookSpec",
    "codebook_accounted_bytes",
    "codebook_codes",
    "fake_quantize_codebook",
    "fake_quantize_uniform_int2",
    "nf_levels",
    "uniform_int2_levels",
]

#: Codebook families this module holds: the NormalFloat table and the uniform-int2 codebook table
#: (the reference's *affine* uniform control is a separate quantizer:
#: :func:`fake_quantize_uniform_int2`).
CODEBOOK_KINDS: tuple[str, ...] = ("nf", "uniform")

#: Bit width of every frozen M3/R1 codebook arm (``reproduction-plan.md`` section 3.1).
FROZEN_CODEBOOK_BITS: int = 2

#: Bytes per stored codebook scale. A codebook block's scale is kept at fp32 — unlike the
#: ``per_group`` fp16 convention of ``memory-accounting.md`` section 3.2 (see
#: :func:`codebook_accounted_bytes`).
CODEBOOK_SCALE_BYTES: int = 4

#: Tail-probability offset of the reference NormalFloat construction
#: (``create_normal_map(offset=0.9677083)``). It is the constant of the QLoRA/NF4 family;
#: :func:`nf_levels` documents where it enters.
NF_OFFSET: float = 0.9677083

#: Number of distinct levels a ``bits``-wide codebook has.
_N_LEVELS = {bits: 1 << bits for bits in SUPPORTED_BITS}


@dataclass(frozen=True)
class CodebookSpec:
    """A validated 2-bit codebook configuration (the frozen M3/R1 quantization setting).

    Args:
        bits: Code width. **Must be 2**: the frozen M3 plan fixes the R1 codebook at 2 bits
            (``docs/research/reproduction-plan.md`` section 3.1, "NF-style codebook, block 64");
            other widths are rejected rather than silently approximated.
        kind: ``"nf"`` (NormalFloat table, :func:`nf_levels`) or ``"uniform"`` (uniform-int2 table,
            :func:`uniform_int2_levels`).
        block_size: Elements sharing one absmax scale. Defaults to 64, the reference's block size.
        axis: Axis moved to the last position before the row-major flatten that defines the block
            order (negative values count from the end). Defaults to ``-1``, which is the
            reference's own plain row-major flatten; the frozen arms use the default.

    Raises:
        ValueError: For a non-int (or bool) field; ``bits != 2``; an unknown ``kind``;
            ``block_size < 1``; or a non-int ``axis``.

    Dtypes / device:
        A pure, immutable configuration — ints and strings only, hashable, no tensor and no device.
        Every function here takes it by value.
    """

    bits: int
    kind: str
    block_size: int = 64
    axis: int = -1

    def __post_init__(self) -> None:
        if isinstance(self.bits, bool) or not isinstance(self.bits, int):
            raise ValueError(f"bits must be an int, got {type(self.bits).__name__}")
        if self.bits != FROZEN_CODEBOOK_BITS:
            raise ValueError(
                f"bits must be {FROZEN_CODEBOOK_BITS}, got {self.bits!r}: the frozen M3 repertoire "
                "fixes the R1 codebook at 2 bits (docs/research/reproduction-plan.md section 3.1, "
                "'NF-style codebook, block 64'); no other width is part of the frozen design"
            )
        if self.kind not in CODEBOOK_KINDS:
            raise ValueError(f"kind must be one of {CODEBOOK_KINDS}, got {self.kind!r}")
        if isinstance(self.block_size, bool) or not isinstance(self.block_size, int):
            raise ValueError(f"block_size must be an int, got {type(self.block_size).__name__}")
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}")
        if isinstance(self.axis, bool) or not isinstance(self.axis, int):
            raise ValueError(f"axis must be an int, got {type(self.axis).__name__}")

    @property
    def n_levels(self) -> int:
        """Number of distinct code values (``2 ** bits``)."""
        return _N_LEVELS[self.bits]

    def levels(self) -> torch.Tensor:
        """The codebook's level table as a sorted float32 tensor of :attr:`n_levels` values.

        Dtypes / device:
            float32 on the CPU. The table is a constant of the *configuration*, not of a tensor:
            it is rebuilt on every call (four values — no caching, no shared mutable state).
        """
        return nf_levels(self.bits) if self.kind == "nf" else uniform_int2_levels()


def _normal_ppf(p: torch.Tensor) -> torch.Tensor:
    """Standard-normal quantile function ``Φ⁻¹(p)`` for ``p`` in ``(0, 1)``.

    Shapes / dtypes:
        ``p`` is a float64 tensor; the result has the same shape and dtype. Implemented as
        ``√2 · erfinv(2p − 1)`` (the inverse error-function identity) rather than through
        ``scipy.stats.norm.ppf``, because scipy is not a project dependency. The two agree to
        float64 round-off, which is far below the float32 cast the table is built with.

    Assumptions / limitations:
        ``p = 0`` or ``p = 1`` map to ``±inf`` (``erfinv(±1)``). The construction only evaluates
        ``p`` in ``[1 − NF_OFFSET, NF_OFFSET] ≈ [0.0323, 0.9677]``, so the tails are never reached.
    """
    return math.sqrt(2.0) * torch.special.erfinv(2.0 * p - 1.0)


def nf_levels(bits: int) -> torch.Tensor:
    """The NormalFloat level table of the pinned reference, normalized to ``[-1, 1]``.

    Derivation (``glue/utils_qaunt.py::create_normal_map``, ``symmetric=False``)
    ---------------------------------------------------------------------------
    With ``variations = 2**bits`` and ``offset = 0.9677083``::

        probs_hi  = linspace(offset, 0.5, variations // 2 + 1)[:-1]   # variations // 2 values
        probs_lo  = linspace(offset, 0.5, variations // 2)[:-1]       # variations // 2 - 1 values
        values    = sort([ Φ⁻¹(probs_hi), 0, -Φ⁻¹(probs_lo) ])
        levels    = values / max(values)                              # -> spans [-1, 1]

    The two sides have **different lengths** (the "+1" in the high-side linspace supplies one extra
    probability, and the literal ``0`` in the middle), which is exactly what makes this the
    *asymmetric* QLoRA-default map: at 2 bits the table is ``[-1.0, 0.0, 0.3379152…, 1.0]`` — three
    non-positive values and one positive value, so it is monotone but **not symmetric about zero**.
    ``offset`` is the tail probability at which the outermost level is placed; the reference's value
    reproduces the bitsandbytes NF4 codebook at ``bits = 4`` (the table of the QLoRA family).

    Dtype choreography (must not be "improved")
    ------------------------------------------
    The reference calls ``torch.linspace`` at its default float32 dtype, evaluates the quantile in
    float64 and then does ``torch.Tensor(values).sort().values /= values.max()`` in **float32**. The
    probabilities are therefore float32 numbers and the division is a float32 division; doing either
    in float64 moves the 2-bit middle level by one float32 ulp and fails the oracle fixture's
    ``1e-10`` tolerance. This function follows the reference's order exactly.

    Args:
        bits: Code width, one of ``fake_quant.SUPPORTED_BITS``.

    Shapes / dtypes / device:
        Returns a sorted, strictly increasing float32 CPU tensor of ``2**bits`` values, spanning
        exactly ``[-1.0, 1.0]`` (the endpoints are exact after the max-normalization).

    Assumptions / limitations:
        Only the asymmetric (QLoRA-default) branch is implemented; the reference's
        ``symmetric=True`` branch is a different table and is not part of the frozen design.

    Raises:
        TypeError: ``bits`` is not an int (bools are rejected).
        ValueError: ``bits`` is not in :data:`spectraquant.quantization.fake_quant.SUPPORTED_BITS`.
    """
    if isinstance(bits, bool) or not isinstance(bits, int):
        raise TypeError(f"bits must be an int, got {type(bits).__name__}")
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"bits must be one of {SUPPORTED_BITS}, got {bits!r}")

    variations = 1 << bits
    half = variations // 2
    # float32 linspace: the reference's default dtype, and part of its published numerics.
    probs_hi = torch.linspace(NF_OFFSET, 0.5, half + 1, dtype=torch.float32)[:-1]
    probs_lo = torch.linspace(NF_OFFSET, 0.5, half, dtype=torch.float32)[:-1]
    values = torch.cat(
        [
            _normal_ppf(probs_hi.to(torch.float64)),
            torch.zeros(1, dtype=torch.float64),
            -_normal_ppf(probs_lo.to(torch.float64)),
        ]
    )
    # torch.Tensor(...) in the reference: float32 cast, then sort, then float32 max-normalization.
    table = values.sort().values.to(torch.float32)
    return table / table.max()


def uniform_int2_levels() -> torch.Tensor:
    """The uniform-int2 **codebook** table: four uniform levels spanning ``[-1, -1/3, 1/3, 1]``.

    This is the control arm's *code table*: the uniform counterpart of :func:`nf_levels`, run
    through the identical absmax codebook machinery so that an R1 arm can be repeated with the same
    bit budget and only the codebook shape changed.

    Why these four values, and why they are symmetric
    ------------------------------------------------
    ``fake_quantize_codebook`` picks the nearest level to ``w / (block_absmax / max|level|)``, so it
    reproduces a block's largest-magnitude element **exactly** only when the table contains both
    ``-max|level|`` and ``+max|level|``. That is precisely what makes the round trip idempotent
    (recomputing the absmax from the output gives the same scale, hence the same codes). With four
    *uniformly spaced* levels spanning ``[-1, 1]`` there is exactly one such table — ``[-1, -a, a, 1]``
    with ``2a = 1 - a``, i.e. ``a = 1/3`` — and it is necessarily symmetric about zero, with **no
    zero level**. Any four-level uniform grid that carries a zero (``[-1, -1/3, 0, 1/3]``,
    ``{-1, -1/3, 1/3, 1}`` shifted, or the affine ``{0, 1/3, 2/3, 1}``) drops one of the extremes and
    loses idempotence.

    The pinned reference's own uniform control is a **different** quantizer: ``quant_uniform`` is
    per-tensor *affine* (it rounds onto ``k / (2**bits - 1)`` for ``k = 0..3``, i.e.
    ``{0, 1/3, 2/3, 1}``, and then rescales onto the clipped ``[min, max]``), so its grid *is*
    one-sided and asymmetric. :func:`fake_quantize_uniform_int2` reproduces that control verbatim and
    is the function the oracle fixture pins; ``CodebookSpec(kind="uniform")`` uses the table returned
    here.

    Shapes / dtypes / device:
        Returns a sorted, strictly increasing float32 CPU tensor of 4 values, normalized to
        ``max|level| = 1`` so that ``fake_quantize_codebook``'s scale is the block absmax.

    Assumptions / limitations:
        The spacing is ``2/3`` (equal on both sides up to one float32 ulp on the outermost gap).
        The table contains no zero level, so a block's near-zero values are rounded to ``±1/3``.
    """
    return torch.tensor([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0], dtype=torch.float32)


def _validate_weight(w: torch.Tensor) -> None:
    """Reject non-tensor, CUDA, non-float, 0-dim and empty inputs (mirrors the packing guards)."""
    if not isinstance(w, torch.Tensor):
        raise TypeError(f"w must be a torch.Tensor, got {type(w).__name__}")
    if w.is_cuda:  # pragma: no cover - no CUDA device exists on the target host
        raise ValueError(
            "codebook quantization runs on CPU only: CUDA tensors are rejected by design"
        )
    if not w.is_floating_point():
        raise TypeError(f"codebook quantization requires a float tensor, got {w.dtype}")
    if w.ndim == 0:
        raise ValueError("cannot quantize a 0-dim tensor; add a trailing dimension")
    if w.numel() == 0:
        raise ValueError("cannot quantize an empty tensor: no block has a finite absmax")


def _moved_axis(rank: int, axis: int) -> int:
    """Normalize ``axis`` for a tensor of this rank, raising ``IndexError`` when out of range."""
    axis_norm = axis + rank if axis < 0 else axis
    if not 0 <= axis_norm < rank:
        raise IndexError(f"axis {axis} is out of range for rank {rank}")
    return axis_norm


def _block_codes_and_scale(
    w: torch.Tensor, spec: CodebookSpec
) -> tuple[torch.Tensor, torch.Tensor, torch.dtype]:
    """Return ``(codes, element_scale, work_dtype)`` in ``w``'s shape for one codebook.

    ``codes`` is int64 in ``[0, n_levels)``; ``element_scale`` holds, for every element, the absmax
    scale of the block it belongs to (``block_absmax / max|level|``, i.e. the block absmax itself
    for every normalized table). Both are shaped like ``w``; the block order is the one documented
    in the module docstring.
    """
    _validate_weight(w)
    rank = w.ndim
    axis = _moved_axis(rank, spec.axis)
    work = torch.float64 if w.dtype == torch.float64 else torch.float32
    levels = spec.levels().to(work)

    moved_shape = (*(w.shape[i] for i in range(rank) if i != axis), w.shape[axis])
    flat = w.movedim(axis, -1).reshape(-1).to(work)
    numel = flat.numel()
    n_blocks = math.ceil(numel / spec.block_size)
    padded = n_blocks * spec.block_size
    if padded > numel:
        # Zero padding cannot raise a block's absmax (|0| <= any absmax) and is discarded below.
        flat = F.pad(flat, (0, padded - numel))

    blocks = flat.reshape(n_blocks, spec.block_size)
    absmax = blocks.abs().amax(dim=1, keepdim=True)
    scale = absmax / levels.abs().max()
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    codes = (blocks / safe).unsqueeze(-1).sub(levels).abs().argmin(dim=-1)
    element_scale = scale.expand(n_blocks, spec.block_size)

    codes = codes.reshape(-1)[:numel].reshape(moved_shape).movedim(-1, axis).reshape(w.shape)
    element_scale = (
        element_scale.reshape(-1)[:numel].reshape(moved_shape).movedim(-1, axis).reshape(w.shape)
    )
    return codes, element_scale, work


def codebook_codes(w: torch.Tensor, spec: CodebookSpec) -> torch.Tensor:
    """The integer level index chosen for every element of ``w`` (the payload of the container).

    Shapes:
        ``w``: any rank >= 1 float tensor. Returns an int64 tensor of the same shape whose values
        are in ``[0, spec.n_levels)`` — the index into ``spec.levels()``. This is the code the
        :func:`fake_quantize_codebook` dequantization multiplies back out, and the index array whose
        ``bits``-wide payload :func:`codebook_accounted_bytes` prices.

    Dtypes / device:
        float32 input is quantized in float32, float64 in float64 (the
        :func:`~spectraquant.quantization.fake_quant.fake_quantize` convention); the returned index
        tensor is int64 and lives on ``w``'s (CPU) device.

    Assumptions / limitations:
        Nearest level by absolute difference, ties to the **lowest** index (``torch.argmin``) — the
        reference's rule. Ties therefore round toward the smaller level index.

    Raises:
        TypeError: ``w`` is not a float tensor.
        ValueError: ``w`` is 0-dim, empty, or on CUDA.
        IndexError: ``spec.axis`` is out of range for ``w``'s rank.

    Example:
        >>> w = torch.tensor([[-2.0, 1.8, 0.8, -0.1]])
        >>> codebook_codes(w, CodebookSpec(2, "nf", block_size=4)).tolist()
        [[0, 3, 2, 1]]
    """
    codes, _, _ = _block_codes_and_scale(w, spec)
    return codes


def fake_quantize_codebook(w: torch.Tensor, spec: CodebookSpec) -> torch.Tensor:
    """Quantize ``w`` to the codebook and dequantize it back in float (measurement class 2).

    Shapes:
        ``w``: any rank >= 1 float tensor. Returns a float tensor of the same shape and dtype whose
        values are ``levels[code] · scale``, where ``scale = block_absmax / max|level|`` is the
        block's absmax divided by the table's largest magnitude (``1`` for every normalized table
        here, so the scale is the block absmax, as in the pinned reference).

    Dtypes / device:
        Quantization runs in float32 for float32 input and float64 for float64 input, on CPU; the
        result is cast back to ``w``'s dtype. Non-float input is rejected. The reference is float32
        throughout, so an exact oracle comparison uses float32 inputs.

    Determinism:
        Pure elementwise arithmetic plus ``amax``/``argmin`` over one dimension — deterministic and
        bit-reproducible on CPU: the same ``(w, spec)`` yields the same tensor bit for bit.

    Assumptions / limitations:
        * The output is a **float** tensor simulating 2-bit numerics: never low-bit storage
          (class 3) and never accelerated inference (class 4).
        * Idempotent for the tables here, by construction: every table is normalized to
          ``max|level| = 1`` **and contains both ``-1`` and ``+1``**, so a block's
          largest-magnitude element is reproduced exactly and the recomputed absmax — hence the
          scales and the indices — is unchanged. (A table missing one extreme, e.g. the affine
          control's one-sided ``{0, 1/3, 2/3, 1}``, would lose idempotence; that grid is therefore
          not used as a codebook table — see :func:`uniform_int2_levels`.)
        * An all-zero block dequantizes to exactly ``0`` (the reference returns ``NaN`` there).
        * Blocks span the row-major flatten (see the module docstring), so a block may cross rows of
          a matrix; with the default ``axis`` and a weight whose last dimension is a multiple of
          ``block_size`` this is identical to the reference's per-row cutting.

    Raises:
        TypeError: ``w`` is not a float tensor.
        ValueError: ``w`` is 0-dim, empty, or on CUDA.
        IndexError: ``spec.axis`` is out of range for ``w``'s rank.

    Example:
        >>> w = torch.tensor([[-2.0, 1.8, 0.8, -0.1]])
        >>> fake_quantize_codebook(w, CodebookSpec(2, "nf", block_size=4)).tolist()
        [[-2.0, 2.0, 0.6758304834365845, 0.0]]
    """
    codes, element_scale, work = _block_codes_and_scale(w, spec)
    levels = spec.levels().to(work)
    return (levels[codes] * element_scale).to(w.dtype)


def fake_quantize_uniform_int2(
    w: torch.Tensor, *, clip: tuple[float, float] | None = None
) -> torch.Tensor:
    """The pinned reference's uniform-int2 control: a per-tensor affine fake quantizer (class 2).

    This reproduces ``glue/utils_qaunt.py::quant_uniform(input, num_bits=2, clip_val=clip)``:

    .. code-block:: text

        x     = clamp(w, clip)                      # both ends, when clip is given
        alpha = x.max() - x.min();  beta = x.min()
        z     = (x - beta) / (alpha + 1e-8)         # map to [0, 1]
        out   = round(z * 3) / 3 * (alpha + 1e-8) + beta

    so it rounds onto the one-sided grid ``{0, 1/3, 2/3, 1}`` (``k / (2**bits − 1)``, the reference's
    own code grid — **not** the codebook table :func:`uniform_int2_levels` returns) and rescales onto
    the clipped range of the *whole tensor* (per-tensor affine, no blocks, no codebook, and **not**
    :func:`fake_quantize_codebook` with ``kind="uniform"``, which uses a per-block absmax codebook).
    The reference's ``weight_quant_fn`` wraps this in ``clip_val = mean ± 2·std``; passing
    ``clip=None`` reproduces that default.

    Shapes:
        ``w``: any rank >= 1 float tensor. Returns a tensor of the same shape and dtype. The
        quantization statistics are per **tensor**, so the result depends on all elements of ``w``.

    Args:
        w: The tensor to quantize.
        clip: ``(low, high)`` clipping range, or ``None`` for the reference dispatcher's default
            ``(mean − 2·std, mean + 2·std)`` computed over ``w`` (``torch.std``'s default, i.e. the
            unbiased estimator, matching the reference's ``weight.std()``).

    Dtypes / device:
        Runs in float32 for float32 input and float64 for float64 input, on CPU. The reference runs
        in float32 (its ``+ 1e-8`` epsilon and clamping are float32 operations), so float32 input is
        required for a bit-exact oracle comparison.

    Assumptions / limitations:
        * ``torch.round`` ties to even, exactly as the reference; no per-block information is used.
        * The ``1e-8`` epsilon in the reference's normalization and rescaling is reproduced verbatim
          (it matters: removing it shifts the 8x8 fixture output by ~8e-8).
        * ``clip`` is applied with the reference's ``torch.where`` expressions, whose semantics are
          "keep ``x`` where it is below the upper bound, else the bound" and symmetrically for the
          lower bound.

    Raises:
        TypeError: ``w`` is not a float tensor.
        ValueError: ``w`` is 0-dim, empty, or on CUDA; or ``clip`` does not have exactly two entries
            with ``low <= high``.

    Example:
        >>> w = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        >>> fake_quantize_uniform_int2(w, clip=(-3.0, 3.0)).tolist()
        [[0.0, 1.0, 2.0, 3.0]]
    """
    _validate_weight(w)
    work = torch.float64 if w.dtype == torch.float64 else torch.float32
    x = w.to(work)

    if clip is not None:
        if len(clip) != 2:
            raise ValueError(f"clip must be a (low, high) pair with low <= high, got {clip!r}")
        low, high = float(clip[0]), float(clip[1])
        if low > high:
            raise ValueError(f"clip must be a (low, high) pair with low <= high, got {clip!r}")
        x = torch.where(x < high, x, torch.tensor(high, dtype=work))
        x = torch.where(x > low, x, torch.tensor(low, dtype=work))
    else:
        mean = x.mean()
        std = x.std()  # unbiased (correction=1), as the reference's torch.std default
        x = torch.where(x < mean + 2.0 * std, x, mean + 2.0 * std)
        x = torch.where(x > mean - 2.0 * std, x, mean - 2.0 * std)

    alpha = x.max() - x.min()
    beta = x.min()
    scale = 2**FROZEN_CODEBOOK_BITS - 1
    normalized = (x - beta) / (alpha + 1e-8)
    quantized = torch.round(normalized * scale).div(scale)
    return (quantized * (alpha + 1e-8) + beta).to(w.dtype)


def codebook_accounted_bytes(
    shape: tuple[int, ...] | torch.Size | list[int], spec: CodebookSpec
) -> int:
    """Analytical bytes of one codebook-quantized tensor: index payload + fp32 scales (class 1).

    Model
    -----
    ``numel = prod(shape)``; ``n_blocks = ceil(numel / block_size)`` (the flatten blocking of
    ``codebook_codes``)::

        payload_bytes = ceil(numel * bits / 8)          # one `bits`-wide index per element
        scales_bytes  = n_blocks * 4                    # one fp32 scale per block
        total_bytes   = payload_bytes + scales_bytes

    Delegation, and why the central helper could not be used directly
    ----------------------------------------------------------------
    The payload and the block count are **not** re-derived here: they are read off
    :func:`spectraquant.quantization.accounting.byte_breakdown`, the project's one byte-accounting
    source of truth. The equivalence spec is ``QuantSpec(bits=2, granularity="per_group",
    group_size=block_size, symmetric=True)`` evaluated on the flattened shape ``(numel,)``, whose
    ``per_group`` block count is exactly ``ceil(numel / block_size)`` and whose ``payload_bytes`` is
    the packed index size ``ceil(numel * bits / 8)`` (with one row, the 32-bit row padding of
    ``memory-accounting.md`` section 3.1 is the byte ceiling itself, so no padding term appears).

    :func:`accounted_bytes` itself **cannot express a codebook container**, and calling it would
    mis-state the cost in two ways: ``QuantSpec`` has no codebook kind (its codes are uniformly
    spaced integers, while a NormalFloat codebook is not), and its ``per_group`` scale is stored at
    **fp16** (``memory-accounting.md`` section 3.2) whereas a codebook block's absmax scale is kept
    at fp32 here (:data:`CODEBOOK_SCALE_BYTES`). It would also count a zero-point array, which a
    codebook index payload does not have. The two terms it *can* supply are therefore taken from it
    and only the declared scale width is applied on top. When ``shape[-1] % block_size == 0`` — true
    for every frozen arm's weight (576/1536/2048/5632 = 9/24/32/88 blocks per row) and for the
    768x768 accounting example — the flatten count and the per-row count coincide, so the number is
    the same as the un-flattened ``per_group`` accounting.

    Class 1 (analytical): derived from shapes and bit widths only. It is **not** measured storage and
    must never be quoted as such (``AGENTS.md`` section 5). No codebook container is serialized yet.

    Args:
        shape: The tensor shape (the same shape :func:`codebook_codes` would receive).
        spec: The codebook configuration; its ``axis`` does not affect the byte count (the payload
            and scale counts are layout-independent — flatten blocking prices one index per element
            and one scale per ``block_size`` elements whatever the axis).

    Raises:
        ValueError: For a non-positive or empty shape.
    """
    normalized_shape = tuple(int(dim) for dim in shape)
    if not normalized_shape:
        raise ValueError("shape must have at least one dimension")
    if any(dim <= 0 for dim in normalized_shape):
        raise ValueError(f"shape must be strictly positive, got {normalized_shape}")
    numel = math.prod(normalized_shape)

    equivalence = QuantSpec(
        bits=spec.bits,
        granularity="per_group",
        group_size=spec.block_size,
        symmetric=True,
    )
    breakdown = byte_breakdown((numel,), equivalence)
    return breakdown.payload_bytes + CODEBOOK_SCALE_BYTES * breakdown.n_blocks
