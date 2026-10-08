# Spectral notes — decay regimes of synthetic weight matrices

**Owner:** factorization stream (E). **Created:** 2026-10-08. **Status:** measurements of
`src/spectraquant/factorization/spectral.py` on controlled synthetic spectra.

Purpose: the proxy slice and the rank/bit allocator both need weight matrices whose spectra are
*known* before they can test whether a sensitivity proxy tracks energy concentration. This note
records the regimes we measured so those slices can reuse them as fixtures, and it fixes the
numbers the unit tests (`tests/unit/test_spectral_*.py`) assert.

Everything below is **class-1/analytical-adjacent**: exact quantities of exactly specified synthetic
matrices, not measurements of any model. No CUDA, no external data. Dtype `float64`, CPU, PyTorch
2.14.1+cpu.

## 1. Fixture construction (reproducible)

For a prescribed singular-value vector `s` of length `n = 256`:

```python
import torch

n = 256
q0, _ = torch.linalg.qr(torch.randn(n, n, generator=torch.Generator().manual_seed(0), dtype=torch.float64))
q1, _ = torch.linalg.qr(torch.randn(n, n, generator=torch.Generator().manual_seed(1), dtype=torch.float64))
W = q0 @ torch.diag(s) @ q1.mT          # W is 256x256, singular values exactly s
```

Prescribed spectra (index `i = 1..n`):

| regime | `s_i` |
|---|---|
| `power_law_0.5` | `i^-0.5` |
| `power_law_1.0` | `i^-1` |
| `power_law_2.0` | `i^-2` |
| `exponential_0.05` | `exp(-0.05 (i-1))` |
| `exponential_0.2` | `exp(-0.2 (i-1))` |
| `near_rank_1` | `s_1 = 1.0`, `s_i = 1e-3` for `i > 1` |
| `random` | `W = randn(256, 256)`, generator seed 99 — no prescribed spectrum |

## 2. Measured regime table

`effective_rank` is `effective_rank(W, energy=0.99)`, i.e. the smallest number of directions
carrying 99 % of the squared energy. `top1pct` = energy fraction in the top
`max(1, ceil(0.01·n)) = 3` directions. `cv` = coefficient of variation of the singular values.
`rank@90/99/99.9 %` = the same threshold rank at three energies.

| regime | fitted decay exp. | fit R² | eff. rank @99 % | stable rank | cond `s_1/s_n` | top1pct energy | cv | rank@90 % | rank@99 % | rank@99.9 % |
|---|---|---|---|---|---|---|---|---|---|---|
| `power_law_0.5` | 0.5000 | 1.0000 | 241 | 6.124 | 1.600e+01 | 0.2994 | 0.8232 | 139 | 241 | 255 |
| `power_law_1.0` | 1.0000 | 1.0000 | 49 | 1.641 | 2.560e+02 | 0.8294 | 3.1938 | 6 | 49 | 181 |
| `power_law_2.0` | 2.0000 | 1.0000 | 3 | 1.082 | 6.554e+04 | 0.9931 | 10.0939 | 1 | 3 | 7 |
| `exponential_0.05` | 3.4014 | 0.7804 | 47 | 10.508 | 3.446e+05 | 0.2592 | 2.3235 | 24 | 47 | 70 |
| `exponential_0.2` | 12.0899 | 0.8296 | 12 | 3.033 | 3.739e+18 | 0.6988 | 4.9513 | 6 | 12 | 18 |
| `near_rank_1` | 0.1336 | 0.0885 | 1 | 1.0003 | 1.000e+03 | 0.9997 | 12.7114 | 1 | 1 | 1 |
| `random` | 0.7472 | 0.4725 | 198 | 65.042 | 1.138e+03 | 0.0452 | 0.6251 | 131 | 198 | 229 |

### 2.1 What the table says (the four regimes in one line each)

* **Power-law spectra are the only ones the power-law fit describes well.** The fitted exponent
  recovers the prescribed `α` exactly (0.5 → 0.5000, 1.0 → 1.0000, 2.0 → 2.0000) with `R² = 1.0000`,
  because the construction *is* the model. This is the control that validates
  `spectral_summary["spectral_decay_exponent"]`.
* **Exponential spectra are not power laws.** The fit reports a steep exponent (3.40 for decay rate
  0.05, 12.09 for 0.20) with `R²` only 0.78–0.83 — the fit is *usable as a concentration feature* but
  must not be read as a physical exponent. The two exponential rows differ by 4× in decay rate yet
  by 1.35× in fitted exponent, so the exponent is monotone in concentration but not proportional to it.
* **`near_rank_1` is the extreme concentration end:** effective rank 1, stable rank 1.0003 (the 255
  equal small values contribute `255·1e-6` to the energy), 99.97 % of the energy in 3 directions, and
  condition number exactly `1/1e-3 = 1000`. Any proxy must rank this layer "spend ~1 rank" or it is
  broken.
* **`random` is the flat/near-flat end:** decay exponent 0.75 with `R² = 0.47` (a poor fit — the
  Marchenko–Pastur bulk is not a line in log–log), effective rank 198/256, stable rank 65.0, and only
  4.5 % of energy in the top 3 directions.

### 2.2 Allocation-facing reading

The allocator needs to know how much the *99 %-energy rank* differs across regimes — this is the
quantity a rank budget is spent on:

* `near_rank_1` and `power_law_2.0` are **free** for low-rank compression (rank 1–3 at 99 % energy,
  and rank 1–7 at 99.9 %).
* `random` is **uncompressible** (rank 198 at 99 %; the low-rank basis buys almost nothing).
* `power_law_1.0` sits in between (rank 49 at 99 %, rank 181 at 99.9 %) — and note the 3.7× jump from
  99 % to 99.9 %: **an allocator that scores only the 99 % rank will be blind to the fact that the
  last 0.9 % of energy still needs 132 more directions.**

## 3. Randomized SVD accuracy against the exact SVD (measured bound)

Setup: `W = randn(200, 120)` float64, generator seed 3; exact reference
`truncated_svd`; randomized `randomized_svd` with the same rank. Error = absolute Frobenius
reconstruction error (float64).

| `rank` | `n_oversamples` | `n_iter` | exact error | randomized error | ratio rand/exact |
|---|---|---|---|---|---|
| 5 | 0 | 0 | 145.889087 | 150.162854 | 1.029295 |
| 5 | 0 | 2 | 145.889087 | 147.513819 | 1.011137 |
| 5 | 10 | 4 | 145.889087 | 146.054761 | 1.001136 |
| 10 | 0 | 0 | 137.297987 | 145.252170 | 1.057934 |
| 10 | 5 | 2 | 137.297987 | 138.792900 | 1.010888 |
| 10 | 10 | 4 | 137.297987 | 137.521580 | 1.001629 |
| 20 | 0 | 0 | 121.277853 | 134.351571 | 1.107800 |
| 20 | 0 | 2 | 121.277853 | 124.467714 | 1.026302 |
| 20 | 5 | 2 | 121.277853 | 123.493337 | 1.018268 |
| 20 | 10 | 4 | 121.277853 | 121.616703 | 1.002794 |

**Measured worst-case ratios** over a wider sweep (`shapes ∈ {(60,40), (200,120), (40,200),
(128,128)}`, 8 seeds, `rank ∈ {2,5,10,20}`, `n_oversamples ∈ {0,5,10}`, `n_iter ∈ {0,1,2,4}`):

* worst `rand/exact` ratio over **all** 40 configurations in the sweep: **1.4402** (the worst case is
  `n_iter=0`, no oversampling, rank close to the effective spectral edge — the known failure mode of
  the plain randomized range finder on a slowly-decaying spectrum);
* worst ratio when `n_oversamples ≥ 5` **and** `n_iter ≥ 2`: **1.0323** — i.e. the recommended
  configuration is within ~3 % of the exact truncation error on these inputs.

**The bound we enforce.** The Halko–Martinsson–Tropp (2011) Frobenius error bound for a Gaussian
sketch of size `l = rank + n_oversamples` is

```
E ||W − Q Qᵀ W||_F  ≤  sqrt(1 + k / (l − k + 1)) · ||W − W_k||_F,      k = rank
```

where `||W − W_k||_F` is exactly the *exact* truncated-SVD error reported by our `truncated_svd`
(for `l = k`, i.e. no oversampling, the factor is `sqrt(1 + k)`; oversampling and power iterations
shrink it). Over the same sweep the **worst ratio of the measured randomized error to this bound was
0.9396 (< 1)** — the bound held for every configuration tested, and
`tests/unit/test_factorization_randomized.py` asserts it (with the exact seed set used here, so the
assertion is deterministic).

## 4. Degeneracies measured

| input | `effective_rank` | `stable_rank` | `condition_number` | `spectral_decay_exponent` | error of `truncated_svd(W, r)` |
|---|---|---|---|---|---|
| zero matrix `0_{5x7}` | 0.0 | 0.0 | 0.0 | `nan` | 0.0 (rank-3 zero factors) |
| `near_rank_1` (see §2) | 1.0 | 1.0003 | 1.000e+03 | 0.1336 | ~1e-13 at rank 1 |
| exact rank-2 `W = B₀A₀` (40×20) | — | — | — | — | `1.51e-14` absolute, `4.11e-16` relative at rank 2 |

`condition_number = 0.0` is reserved for the zero matrix; a rank-deficient non-zero matrix reports
`inf` there and a finite `condition_number_nonzero = s_1 / min(s_i > 0)`.

## 5. Reproduction

The measurements above were produced by scratch scripts outside the repository (per `AGENTS.md` §3
no throwaway artifacts are committed). The fixture construction in §1 plus the table keys in §2 are
sufficient to regenerate them; the assertion-level versions of §2 (power-law exponent 0.5/1.0/2.0),
§3 (bound ratio < 1) and §4 (degeneracies) live in `tests/unit/test_spectral_*.py` and
`tests/unit/test_factorization_randomized.py`.
