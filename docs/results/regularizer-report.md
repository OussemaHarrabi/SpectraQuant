# Regularizer report (Milestone 5, hypothesis H3 — exploratory Tier-0 fixture)

Owner: regularizers/training stream. Evidence: `scripts/experiments/regularizer_sweep.py` and the committed document `artifacts/sample-results/regularizer/sweep.json` (format `spectraquant-regularizer-sweep-v1`, commit `b2ea1edea9845be47266a3e6868261fc73ee8b9f`, dirty=`True`).

**This document is EXPLORATORY and `LOCAL-FIXTURE`.** It is method-design work on the Tier-0 synthetic fixture (`docs/research/preregistration.md` section 8, tag `LOCAL-FIXTURE`, measurement class 1-2). It is **not** a confirmatory test and no confirmatory claim can be sourced from it: the confirmatory H3 cell is Tier 1 on the cloud substrate (`CLOUD-COLAB`, pinned `L_b=8`/`d=256` transformer, >= 5 seeds) and is **NOT RUN** — there is no validated `run_manifest.json` for it, so per section 8 it is *not run*, never "in progress".

## 1. What was run

- Fixture: `spectraquant.training.tiny_lm.TinyCharTransformer` (spectraquant arms from `configs/experiment/regularizer_tier0.yaml`), dense reference pretrained for 60 steps on the synthetic LCG corpus (24 train / 8 dev sequences, corpus checksum `sha256:b0204d500dff244442c0e8bda54c38fdbd6e7833df74917c23f1a9ebfe4908c6`).
- Preparation: 150 steps, batch 16, AdamW lr 0.003, per-cell objective, then rank-then-quantize deployment `Q(B)Q(A)` at rank [8], 4-bit per_group (group size 16, symmetric=True, axis 0).
- Seeds: [0, 1, 2, 3, 4] (the preregistration's Tier-0 floor of 5). Cells: 11 (`factorization`, `full`, `none`, `residual=1`, `residual=10`, `round=0.3`, `round=3`, `round=30`, `rounding`, `spectral`, `spectrum=0.1`).
- Reported on the **dev split only**. The Tier-0 synthetic corpus exposes train/val and no test split, so the sweep touches train + dev and no test set; the frozen two-read test rule (section 3.3) is untouched.
- Arm identities are the validated Hydra configs `configs/method/regularizer_*.yaml` (`none`, `factorization`, `rounding`, `spectral`, `full`); all arms share one compression plan, so they are at equal stored bytes by construction.

## 2. Which objective terms were needed

The predeclared first candidate, the exact rounding-residual ratio `||Q(B)Q(A) - BA||_F^2 / ||BA||_F^2`, is implemented (`rounding_residual_ratio`) and monitored as the objective's `residual` term, but it is **not** the term the working objective uses. Its straight-through gradient is degenerate: with `fq(X) = deq(X) + (X - X.detach())` the Jacobian of `fq` is the identity, so the surviving gradient of `||fq(B)fq(A) - BA||^2` is a product of two rounding errors and carries almost no descent information about the rounding residual itself. Section 4 shows the measurement: sweeping `lambda_residual` does not lower the measured residual.

The term that does the work is `rounding_grid_penalty`: a smooth, scale-weighted surrogate of the factors' squared rounding error on the target grid, `(s^2/4) sin^2(pi x / s)` with the detached block scale `s`. It has the same zero set (the code grid), the same cell amplitude and the same scale weighting as `delta^2`, and a genuinely non-zero gradient. The spectral term (`spectral_tail_energy`) is the plain-spectral **control** of the frozen falsifier; it is numerically zero (~1e-15, float32 SVD noise on a rank-deficient product) when `tail_rank >= rank`, which is why it is off by default and why the spectral arm is configured with `tail_rank` below the factor rank.

## 3. Arms (mean ± sd over seeds, at equal accounted bytes)

| cell | group | λfactor | λround | λresidual | λspectrum | dev NLL (quantized) | output err vs dense | factor rounding residual | product rounding residual | bytes |
|---|---|---|---|---|---|---|---|---|---|---|
| `factorization` | arm | 1 | 0 | 0 | 0 | 0.108 ± 0.013426 | 5.3323 ± 0.11847 | 0.074035 ± 0.00051165 | 0.10821 ± 0.00077608 | 11680 |
| `full` | arm | 1 | 1 | 0 | 1 | 0.17547 ± 0.023514 | 6.6626 ± 0.32073 | 0.040028 ± 0.00040991 | 0.067796 ± 0.00042525 | 11680 |
| `none` | arm | 0 | 0 | 0 | 0 | 0.023765 ± 0.0041281 | 10.366 ± 0.33927 | 0.073368 ± 0.00060587 | 0.10466 ± 0.00079901 | 11680 |
| `residual=1` | lambda_residual | 1 | 0 | 1 | 0 | 0.10707 ± 0.013011 | 5.4168 ± 0.13187 | 0.074835 ± 0.00052683 | 0.10907 ± 0.00063801 | 11680 |
| `residual=10` | lambda_residual | 1 | 0 | 10 | 0 | 0.097252 ± 0.016884 | 6.1692 ± 0.15932 | 0.079091 ± 0.00045573 | 0.11306 ± 0.00052213 | 11680 |
| `round=0.3` | lambda_round | 1 | 0.3 | 0 | 0 | 0.10537 ± 0.013253 | 5.3838 ± 0.11224 | 0.057665 ± 0.0008188 | 0.09069 ± 0.0011458 | 11680 |
| `round=3` | lambda_round | 1 | 3 | 0 | 0 | 0.098693 ± 0.0099101 | 5.5934 ± 0.11294 | 0.021973 ± 0.0005811 | 0.043519 ± 0.00096523 | 11680 |
| `round=30` | lambda_round | 1 | 30 | 0 | 0 | 0.13455 ± 0.0077112 | 5.768 ± 0.1493 | 0.0044536 ± 0.00021354 | 0.010158 ± 0.00056961 | 11680 |
| `rounding` | arm | 1 | 1 | 0 | 0 | 0.10108 ± 0.010768 | 5.4253 ± 0.11534 | 0.037728 ± 0.00074475 | 0.066774 ± 0.00093818 | 11680 |
| `spectral` | arm | 1 | 0 | 0 | 1 | 0.18323 ± 0.027492 | 6.8484 ± 0.36827 | 0.075155 ± 0.00031926 | 0.10147 ± 0.00051435 | 11680 |
| `spectrum=0.1` | lambda_spectrum | 1 | 0 | 0 | 0.1 | 0.10645 ± 0.011217 | 5.4826 ± 0.10651 | 0.074163 ± 0.00044766 | 0.10732 ± 0.00087404 | 11680 |

Two quality columns are reported and they answer different questions: `dev NLL (quantized)` is the task quality of the deployed model on the dev split, while `output err vs dense` is the squared logits error of the *deployed* model against the **dense uncompressed reference** — the predeclared Tier-0 analogue of P1 in `preregistration.md` section 1 (fidelity to the uncompressed model). On this fixture the two disagree in sign: the unregularized `none` arm has the best dev NLL and the worst fidelity-to-dense, because it is free to move away from the dense weights. Both are reported; neither alone decides H3.

### 3.1 The pre-registered arm contrast

- dev NLL of the deployed (rank-then-quantized) model: unprepared `none` 0.023765 -> full `full` 0.17547 (+638.35%).
- squared logits error vs the dense reference: unprepared `none` 10.366 -> full `full` 6.6626 (-35.73%).
- measured factor rounding residual: unprepared `none` 0.073368 -> full `full` 0.040028 (-45.44%).
- measured product rounding residual: unprepared `none` 0.10466 -> full `full` 0.067796 (-35.22%).

## 4. Direction checks (measured, not asserted)

Every row is a *measured* quantity on the dev split, computed with hard rounding and no gradient — not the training-time surrogate. Values are means over the seed set. Each family contains the two-arm endpoints (λ = 0 is the `factorization` arm, the swept coefficient's arm value is included) so the trend is not read off a single point.

### 4.1 `lambda_round` (rounding-grid term)

| λround | measured factor rounding residual | measured product rounding residual | output err vs dense | dev NLL (quantized) | objective `rounding` term |
|---|---|---|---|---|---|
| 0 | 0.074035 | 0.10821 | 5.3323 | 0.108 | 0.0082353 |
| 0.3 | 0.057665 | 0.09069 | 5.3838 | 0.10537 | 0.0069537 |
| 1 | 0.037728 | 0.066774 | 5.4253 | 0.10108 | 0.0051046 |
| 3 | 0.021973 | 0.043519 | 5.5934 | 0.098693 | 0.0032487 |
| 30 | 0.0044536 | 0.010158 | 5.768 | 0.13455 | 0.00096297 |

**Verdict (measured).** Over λround ∈ [0.0, 0.3, 1.0, 3.0, 30.0], the measured factor rounding residual is monotone non-increasing (strictly decreasing), relative change -93.98%. The measured **product** residual, which is what a rank-then-quantize deployment actually incurs, moves -90.61% (monotone non-increasing).

The factor residual is the quantity the penalty claims to control (its hard-rounding counterpart); the product residual is the prescribed H3 quantity. Both are reported. The quality columns are reported too, and they do **not** improve with λround on this fixture (dev NLL relative change +24.59%, monotone non-increasing: False) — an exploratory negative observation, stated rather than hidden; deciding the H3 quality clause needs the Tier-1 cell.

### 4.2 `lambda_spectrum` (plain-spectral control)

| λspectrum | measured tail energy beyond tail_rank | dev NLL (quantized) | output err vs dense |
|---|---|---|---|
| 0 | 0.65481 | 0.108 | 5.3323 |
| 0.1 | 0.57875 | 0.10645 | 5.4826 |
| 1 | 0.2582 | 0.18323 | 6.8484 |

**Verdict (measured).** The measured tail energy (root of the pooled squared-energy fraction beyond tail_rank 4, recomputed independently of the penalty) is monotone non-increasing in λspectrum (relative change -60.57%). The control term therefore reduces the quantity it claims to, so it is an active penalty rather than a no-op.

### 4.3 `lambda_residual` (the predeclared STE ratio)

| λresidual | measured factor rounding residual | measured product rounding residual | dev NLL (quantized) |
|---|---|---|---|
| 0 | 0.074035 | 0.10821 | 0.108 |
| 1 | 0.074835 | 0.10907 | 0.10707 |
| 10 | 0.079091 | 0.11306 | 0.097252 |

**Verdict (measured).** Over λresidual ∈ [0.0, 1.0, 10.0] the measured factor residual is not monotone (relative change +6.83%), i.e. the penalty does not steer the factors onto the grid. This is the documented straight-through degeneracy of the predeclared ratio: its gradient is a product of two rounding errors, so the objective ships it as a monitored diagnostic with default weight 0 and uses the smooth grid surrogate for the activated term.

## 5. Equal memory

All arms declare the identical compression plan, so the class-1 analytical stored bytes are identical by construction: `factorization` 11680 B, `full` 11680 B, `none` 11680 B, `rounding` 11680 B, `spectral` 11680 B.
The gate `spectraquant.reporting.comparability.assert_equal_memory` was applied to the real run manifests of 6 arm pairs at the predeclared 0.5% relative tolerance (`preregistration.md` section 5) and accepted every pair (byte difference 0). These are class-1 analytical bytes, not class-3 measured storage: nothing in this document is a packed-storage or kernel measurement.

## 6. Measurement classes and units

- **class 1** (analytical): `accounted_bytes` and the compression plan (from shapes and bit widths, via `spectraquant.quantization.accounting`).
- **class 2** (fake-quantization quality): every NLL, every output error, both measured rounding residuals and the measured tail energy; all float execution simulating quantize-on-grid numerics.
- **not measured here**: class 3 (packed storage), class 4-CPU/4-GPU (kernel execution), class 5 (service). No latency, throughput or storage claim is made or implied.

## 7. The H3 falsifier, and what this fixture says

Frozen falsifier (`preregistration.md` section 11.1, quoted verbatim):

> The regularizer gives no improvement over unprepared factorization at equal bytes (CI includes 0), **or** the improvement vanishes when the rounding-grid term is replaced by a plain spectral penalty.

**Verdict: cannot yet speak to it at confirmatory scope.** The falsifier is stated at equal *stored bytes* on the confirmatory Tier-1 vehicle (`CLOUD-COLAB`, the pinned L_b=8/d=256 transformer, ≥ 5 seeds) with the preregistration's paired arm contrast (its section 7.1); that cell is **NOT RUN** and no validated manifest exists for it. What this fixture provides instead is exploratory, mechanistic evidence about the *terms*:

1. The rounding-grid term reduces the measured factor rounding residual in the direction claimed (-93.98% over the swept range) — the mechanism works on the quantity it targets.
2. The plain-spectral control reduces the tail energy it claims (-60.57%), so the control is a real, active penalty rather than a no-op.
3. The predeclared exact STE residual ratio does not steer the factors (section 4.3), so the working objective uses the smooth grid surrogate instead; the report states this rather than claiming the predeclared form works.
4. Whether the arm-level *quality* improvement at equal bytes exists, and whether it vanishes under the plain-spectral control (the two clauses of the falsifier), is **not decided here**: the fixture's dev-NLL and output-error differences are reported in section 3 for the Tier-1 cell to be judged against, and the frozen analysis is a paired test at Tier 1, not these fixture means.
The fixture's own exploratory signal is **mixed and is reported as such**: the rounding-residual direction is clean and large, but the deployed dev NLL does not improve with the rounding coefficient (it degrades at the largest coefficient), the spectral arm is the worst arm on dev NLL, and the unregularized `none` arms score best on dev NLL while scoring worst on fidelity to the dense reference. Nothing in this section upgrades or downgrades H3; it records what the fixture measured.

## 8. Limits and next steps

- One fixture (~0.1 M parameters), one synthetic task, one compression plan (rank [8], 4-bit, group 16); the seed count is the Tier-0 floor of 5, and the quantities are single-fixture means, not a powered test.
- The dense reference is a *from-scratch* fixture model, not a pretrained LM; the preparation setting is faithful in structure (SVD init, task training, rank-then-quantize deployment) but not in scale.
- The next step is the frozen confirmatory cell, unchanged: `CLOUD-COLAB`, Tier 1, `configs/tier1/smollm2_135m.yaml`-style arms (d)/(e) plus the plain-spectral control, ≥ 5 seeds, paired test at equal measured bytes. Nothing in this document substitutes for it.

_Wall-clock of the sweep: 155.7 s on the CPU workstation of record (all cells, all seeds: 55 training runs). Generated by `scripts/experiments/regularizer_sweep.py`._
