# SpectraQuant — Research Report (Milestone 10, local evidence)

**Budget-Aware Joint Low-Rank and Low-Bit Training for Efficient Transformers**

Status: **local half only.** This report describes what the committed artifacts measure on the
Tier-0 CPU workstation and marks every cell that has not run as **NOT RUN**. It contains no cloud,
GPU or service number, because none exists. Nothing here is a state-of-the-art claim.

Every headline number below is copied from the artifact it cites; the tables under `reports/tables/`
and the figure under `reports/figures/` are generated from the same artifacts by
`scripts/reproduce/generate_release_artifacts.py` and cannot drift from them. A number marked
`NOT RUN` has no artifact and is never replaced by an estimate.

Substrate labels follow `AGENTS.md` §2b (`LOCAL-FIXTURE`, `LOCAL-CPU-MEASUREMENT`, `CLOUD-COLAB`,
`CLOUD-GPU`); measurement classes follow `AGENTS.md` §5.

---

## Abstract

Transformer compression stacks two lossy transforms — low-rank factorization and quantization — and
their interaction is under-studied at *equal stored memory*. SpectraQuant asks whether preparing a
transformer for low-rank factorization helps or hurts its subsequent quantization (H1), whether an
output-aware sensitivity proxy ranks layers better than weight-space Frobenius error (H2), whether a
rounding-aware spectral regularizer improves quality at equal memory (H3), whether layer-wise mixed
rank and precision dominate uniform compression on the quality–memory frontier (H4), and whether
fake-quantization gains survive packed storage and a real low-bit kernel (H5).

On the **Tier-0 CPU fixture** (`LOCAL-FIXTURE`, measurement classes 1–3 and 4-CPU) this report
establishes: (i) the declared candidate proxy `gain_aware_composed` beats every predeclared
comparator in **15 of 15** aggregate Fisher-z contrasts (model-level, 5 seeds × 12 cells × 2
fixtures), e.g. a Fisher-z advantage over weight-space Frobenius of `0.603921` (CI
[`0.375098`, `0.832745`]) pooled over all fixtures, while the untuned `combined` variant is
*refuted* against the same comparator (`-0.215946`, CI [`-0.35468`, `-0.0772131`]) — a published negative;
(ii) exact layer-wise mixed rank+bit allocation reduces measured hidden-state damage to a ratio of
`0.3667698487962366` of uniform at identical measured bytes (`10.477057393238306` vs
`28.565754321476305` at 16 928 B — better than 2× smaller), with the
allocator's byte accounting reconciling to `0.0` relative difference over 18 arms; (iii) the
rounding-aware regularizer's rounding-residual mechanism works (measured factor rounding residual
−93.98 % over the swept coefficient) but its *quality* signal is mixed — dev NLL of the deployed
model worsens from `0.023765427246689796` (unprepared) to `0.1754724234342575` (full) — so H3 is
**not supported locally**;
and (iv) a self-serialized int4 ONNX container runs through a real CPU `MatMulNBits` kernel with
relative error `0.07712923924319749`— class 4-CPU, no speedup claimed.
The confirmatory Tier-1/Tier-2 cells (H1–H4 and the H5 quality/GPU halves) are **NOT RUN**.
A validated Tier-1 *pilot* slice now exists — `fp16`, per-group int8 and int4 PTQ, and two untrained
low-rank arms on SmolLM2-135M, one seed, 21 758 scored tokens
(`docs/results/tier1-cloud-run-2026-10-09.md`) — and it supports **no confirmatory claim**: it measures
the comparator side of the frontier and two negative results for untrained factorization.

---

## 1. Introduction

Quantization-aware low-rank training (LR-QAT, LoftQ, QLoRA) and low-rank-plus-quantized
decomposition (LQ-LoRA, SVDQuant) each compress transformers, but they are evaluated on different
models, budgets and metrics, and rarely at *equal stored bytes* with a real serialized container.
The result is a crowded but under-controlled question: when a weight matrix is first truncated to
rank $r$ and then rounded to $b$ bits, is the surviving quality better or worse than compressing a
full-rank weight to the same stored size?

This repository builds the machinery to answer that question honestly on a CPU-only workstation and
to run the confirmatory half on a cloud notebook substrate. The local contribution is the *machinery*:
one byte-accounting source of truth, an equal-memory comparison gate, a proxy whose ranking ability
is validated against measured damage, an exact allocator bound by measured bytes, a preparation
objective with a rounding-aware term, and a class-4-CPU path that executes a container SpectraQuant
serializes itself. The empirical contribution is bounded: it is a **fixture-scale** signal, clearly
labelled, that feeds pre-registered confirmatory cells which have not run.

Three negative results are reported rather than hidden: the naive per-layer proxy loses ranking
ability under LayerNorm; the untuned `combined` proxy variant is worse than its comparators; and the
rounding-aware regularizer improves fidelity to the dense reference while worsening task NLL.

---

## 2. Related work

Drawn from `docs/research/literature-review.md` (metadata verified 2026-10-08; upstream pins in
`docs/research/upstream-lockfile.md`). Numbers from other papers are *their* claims; we have not
reproduced them (M3 is **NOT RUN as specified** — its frozen arm set is unimplemented; see §8).

**Foundations.** Gholami et al. (arXiv:2103.13630) and Nagel et al. (arXiv:2106.08295) taxonomize
quantization granularity and the PTQ/QAT split; OBD/OBS (LeCun et al. 1990; Hassibi & Stork 1993)
introduce second-order saliency, which today's sensitivity proxies still echo. Classical compression
(Deep Compression, arXiv:1510.00149) already showed that stacking lossy transforms is non-trivial,
but it is CNN-era.

**Low-rank-then-quantize and quantized low-rank decomposition.** LoftQ (arXiv:2310.08659) alternates
quantize/SVD to initialize adapters on a quantized base; LQ-LoRA (arXiv:2311.12023) decomposes a
matrix into a high-precision low-rank part plus a quantized part, allocating per-matrix rank,
bit-width and block size under one budget with an integer program. SRR/Preserve-Then-Quantize
(arXiv:2602.02001) splits the rank budget between a preserved subspace and quantization-residual
reconstruction — the closest published neighbour to our regularizer. SVDQuant (arXiv:2411.05007)
uses a high-precision low-rank branch to absorb outliers with a co-designed GPU kernel; MLoRQ
(arXiv:2507.09616), the ASP-DAC-2026 joint compression work, AutoQRA (arXiv:2602.22268), KV-COBRA
(arXiv:2609.24298) and JoLT (arXiv:2607.12550) all allocate rank and bits jointly. **Joint rank × bit
allocation is therefore previously known**, which is why our claim is narrowed to the *output-aware
cost proxy* and the *equal-stored-bytes protocol* rather than to joint allocation itself
(`docs/research/novelty-risk.md`).

**PTQ baselines and sensitivity proxies.** GPTQ (arXiv:2210.17323), AWQ (arXiv:2306.00978),
SmoothQuant (arXiv:2211.10438), LLM.int8() (arXiv:2208.07339), OWQ (arXiv:2306.02272) and
SqueezeLLM (arXiv:2306.07629) define the baseline family. HAWQ/HAWQ-V2 (arXiv:1905.03696,
arXiv:1911.03852) is the canonical mixed-precision sensitivity proxy and our predeclared bit-only
comparator; APTQ (arXiv:2402.14866), CoopQ (arXiv:2509.15455) and KronQ (arXiv:2607.07964) push
toward interaction-aware sensitivity; MixQuant (arXiv:2607.23047) marginalizes over upstream bit
widths; RAM (arXiv:2609.33923) shows a propagation-weighted random probe ranks layers well. None of
these allocates *rank* from an output-aware cost.

**Regularizers and spectra.** Spectral Normalization (arXiv:1802.05957), the dynamical low-rank
spectral regularizer (arXiv:2505.08022) and Spectrum (arXiv:2406.06623) control spectral properties
for robustness or select modules to train; none ties the penalty to a quantization grid.

**Gap.** What remains least covered is whether an output-aware cost proxy that couples the two
transforms yields a better frontier than these alternatives *at equal stored bytes* — the question
this repository targets with H2 and H4.

---

## 3. Formal problem definition

Let a transformer have $L$ compressible linear modules; module $\ell$ has weight
$W_\ell \in \mathbb{R}^{d^{\text{out}}_\ell \times d^{\text{in}}_\ell}$. A compression plan assigns
each module a rank $r_\ell$ and a bit width $b_\ell$; the plan's **accounted bytes** (class 1) is the
sum, over modules, of the stored size of the padded payload, per-block scales and zero-points as
computed by `spectraquant.quantization.accounting`, and the **measured bytes** (class 3) is the size
of the serialized container. The factorization convention is one and only one
(`docs/coordination/design-m2-interfaces.md` §0.2): $W \approx B A$ with
$A\in\mathbb{R}^{r\times d^{\text{in}}}$, $B\in\mathbb{R}^{d^{\text{out}}\times r}$.

The allocation problem is

$$\min_{\{(r_\ell,b_\ell)\}} \; \sum_\ell \hat{E}_\ell(r_\ell,b_\ell)
\quad \text{s.t.} \quad \sum_\ell \text{bytes}_\ell(r_\ell,b_\ell) \le B,$$

where $\hat{E}_\ell$ is a per-module error proxy in the common unit of
`design-m2-interfaces.md` §0.3/§7 (per-layer squared-Frobenius *output* error), and $B$ is a
predeclared budget. **Comparisons are only valid at equal measured bytes**: any quality comparison is
gated by `spectraquant.reporting.comparability.assert_equal_memory` at the predeclared `0.5 %`
relative tolerance (`docs/research/preregistration.md` §5, §7.0).

Two error notions matter and are never conflated: the factorization reconstruction error
$\lVert W - BA\rVert_F$ and the *rounding sensitivity*, i.e. the additional output distortion
incurred when the factors are rounded onto the quantization grid. H1 asks whether these can be made
to move in opposite directions; H3's regularizer is the mechanism that would test it. All numbers in
this report are labelled with one measurement class (`AGENTS.md` §5): class 1 (analytical), class 2
(fake-quantization quality), class 3 (packed-storage bytes measured), class 4-CPU (a real CPU
low-bit kernel executing *our* container). Classes are never mixed inside a claim.

---

## 4. Method

### 4.1 Declared candidate proxy

Per `design-m2-interfaces.md` §7 and amendment A-0007, the **declared candidate** is
`gain_aware_composed`: the per-layer output error multiplied by an estimated downstream gain,
scored through the public `Proxy.score_model` entry point with option-specific `LayerInputs`. The
naive pre-normalisation per-layer output error (`per_layer_output_error`) is a **baseline** whose
failure the review measured (ρ = 0.119 under LayerNorm) and which we publish (§11). The candidate's
estimator reports `exact: bool` and its sampling information, because sampling variance is not the
dominant error term — the activation-distribution shift under joint compression is
(`design-m2-interfaces.md` §7). Predeclared comparators are weight-space Frobenius error
(`weight_frobenius`, the primary H2 comparator), weight magnitude, activation magnitude, and the
HAWQ-V2-style average-Hessian-eigenvalue surrogate `hessian_diag`; secondary variants
`in_situ_output_error`, `quant_residual_stats`, `spectral_summary` and `combined` are reported.

### 4.2 Preparation objective (rounding-aware spectral regularizer)

The preparation loop (`scripts/experiments/regularizer_sweep.py`, `src/spectraquant/regularizers/`)
optimizes a task term plus optional auxiliary terms: a factorization term, a **rounding-grid penalty**
`smooth grid surrogate = (s²/4)·sin²(πx/s)` with the detached block scale `s`, a spectral tail-energy
term, and a monitored rounding-residual diagnostic. The predeclared exact ratio
`||Q(B)Q(A) − BA||_F² / ||BA||_F²` is implemented and monitored, but its straight-through gradient is
degenerate (a product of two rounding errors), so the *activated* term is the smooth grid surrogate —
stated, not hidden (`docs/results/regularizer-report.md` §2; amendment A-0008). Every arm shares one
compression plan, so the arms are at equal accounted bytes by construction.

### 4.3 Allocator

`src/spectraquant/allocation/` provides `solve_uniform`, `solve_greedy` (marginal error reduction per
byte, deterministic), `solve_exhaustive` (oracle) and `solve_ortools` (CP-SAT; the `alloc` extra).
The cost function is backed by the single byte-accounting source of truth; `solve_exhaustive ==
solve_ortools` on tiny instances (tested). The proxy feeds the solver as its `error_fn` through
`src/spectraquant/allocation/proxy_adapter.py`, with option-specific context so the proxy is not held
at one reference compression. Allocations are emitted as deterministic manifests.

### 4.4 Equal-memory gate

`assert_equal_memory(run_a, run_b, tolerance)` compares **class-3 measured** bytes at the
predeclared `0.5 %` relative tolerance, converted to an integer byte budget by `ceil`. A pair
outside tolerance is not an equal-memory comparison and is refused. A dedicated negative test
(`test_equal_memory_gate_accepts_parity_and_rejects_a_gap`) proves the rejection is caused by the
tolerance check.

---

## 5. Experimental protocol (frozen preregistration)

The study protocol is frozen at `docs/research/preregistration.md` (FROZEN 2026-10-09 at commit
`6281c56`, amendment A-0006); amendments are append-only in
`docs/research/preregistration-amendments.md`. The binding elements reproduced here:

* **Primary outcomes.** P1 = held-out per-token cross-entropy (nats), reported as perplexity, class 2
  (fake quantization); on Tier 0 the analogue is mean-squared output error of the compressed linear
  map. P2 = stored bytes of the serialized weight artifact, class 3 (measured) with a class-1
  cross-check. P1 and P2 are compared only at equal P2.
* **Datasets and splits.** Tier 0: synthetic controlled-spectrum matrices and a tiny net; Tier 1:
  WikiText-2 (raw, later document-level) and a deterministic TinyStories slice; Tier 2: WikiText-2
  test plus a pinned lm-evaluation-harness subset. Calibration is drawn **only** from the training
  split, disjoint from dev and test; a programmatic overlap check runs before every run. The test set
  is read **exactly 2 times** per arm (one confirmatory read plus at most one recorded re-read).
* **Tier-1 confirmatory vehicle (pinned, not chosen after seeing results):** `L_b = 8` transformer
  blocks, width `d = 256`, sequence length 256, vocab 50,257 (GPT-2 BPE), ≈32 M parameters, a fixed
  5 epochs on the cloud notebook substrate (`CLOUD-COLAB`).
* **Grids.** Bits `{2,3,4,8}`, ranks `{2,4,8,16,32}`, group sizes `{32,64,128}`; the H5 chain applies
  to $b \in \{4,8\}$ only (there is no `pack_int2`/`pack_int3` and no permitted kernel executes
  2/3-bit weights, so those arms are class-2 only and excluded from every H5 statement).
* **Arms.** (a) uniform-$b$ full-rank; (b) uniform-$r$ then uniform-$b$; (c) quantize-then-low-rank
  residual; (d) proxy-allocated $(r_\ell,b_\ell)$; (e) proxy-allocated with the regularizer (H3 arm);
  (f) fp16 reference. Q2 (order of transforms) is a *mechanistic comparator, not a hypothesis*.
* **Statistics.** Arm-vs-arm unit = seed; paired bootstrap 95 % CI (≥10,000 resamples) + Cliff's
  delta. Proxy validity (H2) unit = **model**; per model a Spearman ρ over that model's `n=32` linear
  modules is transformed to Fisher-z and combined by a DerSimonian–Laird random-effects model; the
  H2 statistic is the paired difference in Fisher-z between candidate and comparator. Frontier
  comparison (H4) = Pareto hypervolume difference with a paired bootstrap CI over seeds; dominance
  requires the hypervolume-difference CI to exclude 0, a stored-byte saving ≥ 5 % at matched quality
  (`|ΔNLL| ≤ 0.01` nats/token), and the byte saving's CI to exclude 0. Families F1 (1 test), F2 (H1,
  H3, H4) and F3 (H5a, H5b) control multiplicity (Holm–Bonferroni); exploratory sweeps use
  Benjamini–Hochberg and support no primary claim.
* **Power.** Predeclared MDE(z) at S = 5 seeds is 0.33 (optimistic, `n_eff = 32`) to 0.79
  (conservative, `n_eff = 8`). A null CI wider than the MDE is *inconclusive*, never a refutation of a
  smaller effect.
* **Byte parity and serializer.** Two arms are at byte parity iff `|bytes_a − bytes_b| / bytes_b ≤
  0.5 %`. One serialization entry point is frozen (`spectraquant-sqpack-v1`); container overhead is
  invocation-dependent and measured, never assumed.
* **Exclusions (automatic, constants).** Non-finite loss; dev-NLL divergence ≥ +0.693 nats/token over
  500 steps; OOM or wall-clock above the per-tier cap; schema-invalid manifest; detected train/eval
  overlap; byte-accounting error > 0.5 %. Unfavourable seeds and unsupported hypotheses are **not**
  exclusion reasons.
* **Substrate.** The local workstation produces Tier-0 fixtures, the class-4-CPU measurement and all
  analysis; Tiers 1–5 execute only on the cloud notebook substrate. A cloud cell is a result only
  once a validated `run_manifest.json` and checksum-validated artifacts exist; until then it is
  **NOT RUN**.

---

## 6. Reproduction status per baseline

M3 (bounded reproduction of LR-QAT and LoftQ) is **NOT RUN as specified**. The cloud substrate
itself is no longer the blocker: two runs are collected and checksum-validated on Kaggle
(`tier1_smollm2_135m-cloud`, CPU; `repro_lr_qat_loftq_smollm2_135m-cloud`, Tesla T4), and
SmolLM2-135M has been loaded and evaluated. What blocks M3 is that the **frozen protocol's eight
arms are not implemented**: `R1-FP16-LoRA`, `R1-std-2bit`, `R1-loftq-2bit` and `R1-loftq-2bit-T1`
(2-bit NF-style codebook, block 64, rank 16), `R2-FP16`, `R2-RTN-4bit-g128`,
`R2-LRQAT-4bit-g128` (rank 32 plus a learned step size, `Φ₀` downcast to Q4.4) and
`R2-fullQAT-4bit-g128`, together with the validation-split learning-rate search
(`docs/research/reproduction-plan.md` §3). The trainable arms that have run use the plans' default
grid (rank 8, uniform 4-bit per-group 32) and are reported as **diagnostics**, not as M3 cells: at
one seed and 300 steps they show the expected direction (training improves on the initialisation —
LoftQ 17.79 → 16.69, LR-QAT 17.72 → 17.07 perplexity against an fp16 reference of 14.02) but they
test neither the predeclared bit width nor the predeclared verdict rule. Baselines below list what
*would* be compared and the pinned revision at which it would be reproduced; none is claimed as
measured here.

| baseline / comparator | role | pinned upstream (verified) | license | status |
|---|---|---|---|---|
| LR-QAT (arXiv:2406.06385) | QAT-with-low-rank reference trend | `Qualcomm-AI-research/LR-QAT` @ `8795afe054cf951b714299e01083a1b354721829` | BSD-3-Clause-Clear | **NOT RUN as specified** (frozen arm set unimplemented) |
| LoftQ (arXiv:2310.08659) | quantize-SVD initialization trend | `yxli2123/LoftQ` @ `ae33fd4fd05fd4ba146555cd77c13d307eb4e9b3` | MIT | **NOT RUN as specified** (frozen arm set unimplemented) |
| LQ-LoRA (arXiv:2311.12023) | rank/bit-budget comparator | `HanGuo97/lq-lora` @ `c2424b3adc27197815da1ac9e1304565168d824d` | MIT | **NOT RUN as specified** (frozen arm set unimplemented) |
| GPTQ (arXiv:2210.17323) / AWQ (arXiv:2306.00978) | PTQ baselines | not pinned this wave | unverified | **NOT RUN** (`CLOUD-GPU`) |
| HAWQ-V2 (arXiv:1911.03852) | bit-only allocation comparator (H4) | community | n/a | **NOT RUN** (confirmatory-cell arm) |
| uniform rank/bit | primary equal-memory reference | in-repo (`solve_uniform`) | Apache-2.0 | **measured on the Tier-0 fixture** (§7, §10) |
| greedy | simplistic allocator reference | in-repo (`solve_greedy`) | Apache-2.0 | **measured on the Tier-0 fixture** (§10) |

The reproduction plan (`configs/repro/lr_qat_smollm2_135m.yaml`) fixes the predeclared tolerance: a
trend counts as reproduced when the method's advantage over its comparator has the same sign and at
least 25 % of the published effect size on perplexity at matched rank/bit. That plan's execution is
**NOT RUN**.

---

## 7. Results (Tier-0 fixture, `LOCAL-FIXTURE`)

All numbers in this section are class 1/2/3 on the Tier-0 fixture (one seed, 13 modules, synthetic
next-token task) unless stated otherwise. Generated tables: `reports/tables/allocator-frontier.md`
and `reports/tables/proxy-fixture.md`; figure: `reports/figures/pareto-damage-vs-bytes.svg`.

The fixture is a 3-block pre-norm transformer (`d_model=48`, 2 heads, `d_ff=96`, `vocab=24`,
`seq_len=24`) trained 120 steps (`artifacts/sample-results/allocation-frontier/frontier.json` →
`fixture.training.final_loss = 0.008360037580132484`). It is **not** a language model.

**Allocator frontier.** At the first equal-byte point the exact mixed rank+bit allocation reduces
measured hidden-state damage versus uniform: at 16 928 B the uniform `(8,8)` arm measures
`28.565754321476305` hidden damage while the mixed arm measures `10.477057393238306`, a ratio of
`0.3667698487962366` (`artifacts/sample-results/allocation-frontier/frontier.json`, points[3] and
`equal_memory_wins[2]`). At 9 496 B the mixed arm measures `31.911922669134054` versus uniform's
`51.92979658800838`, ratio `0.6145204635079035` (same artifact, `equal_memory_wins[0]`). Predicted
error and measured damage agree in direction and magnitude: proxy `0.9708333333333333` Spearman
across all 18 arms, per-arm per-layer mean `0.865689865689866` (`rank_agreement`).

**Byte accounting.** All 18 allocated configurations reconcile class-3 measured bytes with class-1
accounting at `max_relative_difference = 0.0` over 18 reconciliations
(`artifacts/sample-results/allocation-frontier/frontier.json`, `byte_reconciliation`): the
SpectraQuant container is headerless and stores exactly the accounted terms, so the container
overhead is `0` bytes by construction.

**Proxy ranking (H2, local half).** Across all fixtures the declared candidate
`gain_aware_composed` reaches model-level ρ mean `0.824908`, versus `0.488187` for weight-space
Frobenius, `0.304762` for the naive baseline and `0.312271` for the untuned `combined`
(`artifacts/sample-results/proxy-validation/proxy-validation.json`, `aggregate.all_fixtures.variants`).
The achieved MDE at the achieved `n_eff` is `0.475544` (inside the predeclared 0.33–0.79 band).

**Proxy fixture failure of the naive baseline.** On the single-seed proxy fixture, the naive
`per_layer_output_error` scores ρ = `-0.06043956043956044` under LayerNorm while the gain-aware
variant scores `0.8296703296703297`; per-layer damage composes super-additively, with joint damage /
Σ(single-layer damage) = `1.3751080626424674`
(`artifacts/sample-results/proxy-fixture/proxy-fixture.json`).

**Systems (class 4-CPU).** A self-serialized int4 ONNX container (`1662` measured bytes) executes
through ONNX Runtime's `MatMulNBits` CPU kernel with relative max error `0.07712923924319749` at input
shape (4, 64) against an fp32 baseline measured in the same session; the int8 container measures
`2727` bytes and the fp32 reference `8321` bytes
(`artifacts/sample-results/class4cpu/class4cpu.json`). See §12.

---

## 8. Ablations

Generated table: `reports/tables/regularizer-arms.md` (from
`artifacts/sample-results/regularizer/sweep.json`). Exploratory, `LOCAL-FIXTURE`, class 1–2.

The regularizer sweep varies one auxiliary coefficient at a time (11 cells, 5 seeds each, equal
accounted bytes 11 680). The predeclared contrast between the unprepared arm `none` and the full arm
`full`:

| quantity | `none` | `full` | relative change |
|---|---|---|---|
| dev NLL (quantized) | `0.023765427246689796` | `0.1754724234342575` | +638.35 % |
| output err vs dense | `10.366287429666171` | `6.662638838174908` | −35.73 % |
| factor rounding residual | `0.07336750664452484` | `0.04002804431809499` | −45.44 % |
| product rounding residual | `0.10466320542682295` | `0.06779602722568183` | −35.22 % |

(`artifacts/sample-results/regularizer/sweep.json`, `cells.none.metrics`, `cells.full.metrics`.) The
"relative change" column is **derived** from the two artifact values in the same row (it is not itself
stored in the artifact); the −93.98 % figure cited elsewhere is derived the same way from
`direction_checks.lambda_round_vs_factor_rounding_residual.relative_change`
(`-0.9398451098149285`). The
`lambda_round` family lowers the measured factor rounding residual monotonically with relative change
`-0.9398451098149285` (`direction_checks.lambda_round_vs_factor_rounding_residual`) while dev NLL is
non-monotone (`direction_checks.lambda_round_vs_dev_nll_quantized`, relative change
`0.2458935455360959`). The H3 confirmatory cell (`CLOUD-COLAB`, Tier 1) is **NOT RUN**
(`confirmatory_cell.status`).

**Component / calibration / outlier / seed-count ablations** are a sibling slice's evidence and live
in `docs/results/ablation-report.md` (machine-readable:
`artifacts/sample-results/ablations/ablations.json`). They are not reproduced here.

---

## 9. Proxy validation (H2)

The H2 measurement is reported in `docs/results/proxy-validation-report.md` with its machine-readable
artifact and generated table `reports/tables/proxy-ranking.md`. Summary (model-level, 5 seeds, 12
cells, 10 trained models, substrate `LOCAL-FIXTURE`, class 2):

* **Candidate `gain_aware_composed`**: 15 of 15 aggregate contrasts (2 fixtures + pooled, 5
  comparators) have a positive Fisher-z advantage whose random-effects CI excludes 0. Pooled over
  all fixtures against the predeclared primary comparator `weight_frobenius`: delta_z
  `0.603921`, CI [`0.375098`, `0.832745`], p `2.30576e-07` →
  **supported** (`aggregate.all_fixtures.contrasts.gain_aware_composed.weight_frobenius`).
* **`combined` (untuned ablation)**: refuted against `weight_frobenius` pooled, delta_z `-0.215946`,
  CI [`-0.35468`, `-0.0772131`] (`aggregate.all_fixtures.primary_verdict`).
* **Cloud Tier-1 cells (H2, H4)**: **NOT RUN**. One Tier-1 run is now collected and validated
  (`tier1_smollm2_135m-cloud`, commit `9e7c8768b31a91ec726cedfdc832cfc8ccd04743`, 4 artifacts
  checksum-verified), but it covers only the non-trainable comparator arms at one seed: fp16
  reference 14.0181 perplexity at 212 336 640 accounted bytes, per-group int8 PTQ 14.0412 at
  112 803 840 (1.88x fewer bytes), per-group int4 PTQ 18.6190 at 59 719 680 (3.56x), and two
  untrained low-rank arms that are degenerate (mean relative Frobenius error 0.9427 and 0.9440).
  H2 (the proxy ranking) and H4 (the allocator frontier) remain **NOT RUN**; no Tier-1 number for
  either is implied anywhere.

The H2 verdict is stated against the achieved MDE: `0.475544` pooled at the achieved `n_eff`,
inside the predeclared 0.33–0.79 band. This is *exploratory fixture evidence*; the frozen
confirmatory H2 cell is Tier 1.

---

## 10. Negative results

Reported because `AGENTS.md` §4.10 requires it, and because each one bounds a claim.

1. **The naive per-layer proxy fails under LayerNorm.** `per_layer_output_error` scores ρ =
   `-0.06043956043956044` under LayerNorm (and `0.24175824175824176` without) on the proxy fixture,
   versus `0.8296703296703297` for the gain-aware variant
   (`artifacts/sample-results/proxy-fixture/proxy-fixture.json`, `fixtures.layernorm.proxies`). The
   per-layer damages also compose super-additively (`1.3751080626424674`), so no single constant
   rescales them. This is why the candidate must carry a composition/gain term.
2. **The untuned `combined` variant is worse than its comparators.** Pooled delta_z `-0.215946`
   against weight-space Frobenius (`artifacts/sample-results/proxy-validation/proxy-validation.json`,
   `aggregate.all_fixtures.primary_verdict`), and it is refuted against weight magnitude too. The
   equal-weight combination is not the candidate (amendment A-0007).
3. **Greedy allocation is myopic and can be far worse than uniform.** At 17 392 B greedy measures
   hidden damage `37.83093875405708` versus uniform's `10.254079443113978` (a more-than-3× increase)
   and at 9 728 B `51.50522987295377` versus `29.04945948402356`
   (`artifacts/sample-results/allocation-frontier/frontier.json`, points[4], points[2]). Greedy is
   *not* a safe approximation on this fixture; only the exact solver is used for the headline.
4. **The M5 regularizer result is mixed.** The rounding-grid term reduces the quantity it targets
   (factor rounding residual −93.98 % over the swept range), but the *deployed* dev NLL does not
   improve with the rounding coefficient and the `spectral` arm is the worst arm on dev NLL (dev NLL
   `0.1832280784845352`); the unregularized `none` arm scores best on dev NLL
   (`0.023765427246689796`) and worst on fidelity to the dense reference (`10.366287429666171`)
   (`artifacts/sample-results/regularizer/sweep.json`). H3 is therefore **not supported** locally;
   the confirmatory H3 cell is **NOT RUN**.

---

## 11. Limitations and broader impact

**Limitations.**

* Everything measured here is **Tier 0**: tiny synthetic fixtures and a single weight matrix. No
  language model was trained or evaluated locally. The confirmatory H1–H4 cells and the H5
  quality/GPU halves are **NOT RUN** (`CLOUD-COLAB` / `CLOUD-GPU`).
* Quality numbers are **class 2** (fake quantization); the class-3 numbers are *bytes*, never a
  quality claim. The class-4-CPU number is a self-serialized fixture, not the Tier-1 H5 chain.
* The allocator optimises the **final hidden state**; logits damage is reported separately and
  worsens at mixed points (e.g. `58.667224998386` vs `41.47838123188296` at 16 928 B), because the
  proxy's unit is the hidden state and `head` is downstream of the measurement point.
* The proxy's `n_eff` is a bootstrap estimate over 13 modules and is noisy; the achieved MDE
  (`0.475544` pooled) is coarser than the predeclared optimistic end (0.33). Layer-count power is low
  by predeclaration, which is why the model is the unit.
* No latency, throughput or service claim is made, and none follows from the CPU kernel path. The
  class-4-CPU latencies are overhead-dominated at fixture scale.
* Integer byte tolerance: the 0.5 % tolerance becomes an integer byte budget via `ceil`, so a
  relative difference marginally above 0.5 % can still pass; the relative difference is always
  reported.

**Broader impact.** The deliverable is compression machinery for open, CPU-checkable models; the
positive near-term impact is a stricter, equal-memory accounting discipline for a crowded literature
and a reproducible harness for researchers without GPU access. Compression research can lower
inference cost but also lowers the cost of deploying harmful models, and quantization can degrade
model behaviour in ways that are hard to audit; this is why measurement-class labelling and the
"no state-of-the-art claim" rule are binding here. No dual-use code (e.g. kernels, datasets) is
released beyond what is already pinned upstream.

---

## 12. Systems results (class 4-CPU)

Generated tables: `reports/tables/class4cpu-bytes.md`, `reports/tables/class4cpu-latency.md`
(from `artifacts/sample-results/class4cpu/class4cpu.json`; substrate `LOCAL-FIXTURE`, measurement
class **4-CPU**).

A single weight-only linear map (`W ~ N(0, 0.1²)`, `(K,N) = (64, 32)`, `|W|.max = 0.3171`) is
serialized by SpectraQuant into three ONNX containers and executed by ONNX Runtime 1.30.0's CPU
provider:

| backend | op / domain | total B (class 3) | relative max error (X shape (4,64)) |
|---|---|---|---|
| fp32 `MatMul` | `MatMul` | `8321` | `0.0` |
| int4 `MatMulNBits` | `MatMulNBits` / `com.microsoft` | `1662` | `0.07712923924319749` |
| int8 dynamic | `MatMulInteger` | `2727` | `0.007851041544258917` |

The int4 figure reproduces the independently published `0.2040593922138214` absolute / `0.077129`-class
relative error from `docs/research/backend-capability.md` §2.4c, confirming the container executes
through the real kernel. **Scope line (verbatim from the artifact):** *"CPU-only measurement on AMD
Ryzen AI 7 350 w/ Radeon 860M with 8 threads; not comparable to published GPU latency or throughput
figures."* No speedup is claimed or implied; the mandatory scope line is present
(`scope_line_required_verbatim`), the same-session fp32 baseline uses identical inputs and threads,
and `dequantized_path = false`.

This makes the H5 CPU-kernel cell **runnable and measured**; it does not by itself confirm H5 (the
quality half is `CLOUD-COLAB`, the GPU half `CLOUD-GPU`, both **NOT RUN**). H5 is reported
**partially supported / inconclusive at GPU scope**.

---

## 13. Reproducibility statement

All local evidence is reproducible on a CPU-only workstation from a clean checkout with the pinned
`uv.lock`; the commands are in `reports/REPRODUCIBILITY.md`. Determinism is enforced by seeding
Python/NumPy/PyTorch from one integer, `torch.use_deterministic_algorithms(True)` and single-threaded
CPU reductions. Each sample artifact records its git commit and dirty flag; the class-4-CPU artifact
records the SHA-256 of every container, and the allocator manifests replay deterministically
(`diff -rq` clean). The tables and figure in `reports/tables/` and `reports/figures/` are generated
solely by `scripts/reproduce/generate_release_artifacts.py` from the committed artifacts, and the
regression test `tests/unit/test_release_artifacts.py` fails if they drift.

What cannot be reproduced locally: **Tiers 1–5** (all model training, QAT, large-scale inference and
GPU evaluation) — they require the cloud notebook substrate and **have not run**; **class 4-GPU and
class 5** (CUDA-only kernels and service measurement) — unavailable on this workstation; and the
**M3 baseline reproduction** — **NOT RUN**. Nothing in this report estimates those cells.

---

## Appendix A — Configurations

Configuration values are quoted from the committed configs; no value is retyped.

**Tier-0 fixture model** (`configs/model/tiny.yaml`): `vocab_size: 32`, `d_model: 64`,
`n_heads: 4`, `n_layers: 2`, `max_seq_len: 32`, `dropout: 0.0` (the CI smoke fixture; ≤200 k params
by construction, `src/spectraquant/training/tiny_lm.py`).

**Tier-0 synthetic data** (`configs/data/synthetic.yaml`): `vocab_size: 32`, `seq_len: 32`,
`n_train_sequences: 24`, `n_val_sequences: 8`, generator `lcg` (`lcg_a: 3`, `lcg_b: 7`),
`train_start: 0`, `val_start: 24` (disjoint by construction).

**Evaluation fixture** (`src/spectraquant/evaluation/toy.py`, `TinyConfig`): `n_blocks: 3`,
`d_model: 48`, `n_heads: 2`, `d_ff: 96`, `vocab: 24`, `seq_len: 24`, `use_norm: True`,
`final_norm: True`, `seed: 0` (the fixture behind the proxy/allocation artifacts; 13 compressible
linear modules).

**Tier-0 regularizer experiment** (`configs/experiment/regularizer_tier0.yaml`): `steps: 150`,
`batch_size: 16`, `lr: 0.003`, `grad_clip: 1.0`, `seed: 2026`; method group
`configs/method/regularizer_*.yaml`; compression plan rank 8, int4, per-group 16.

**CI smoke experiment** (`configs/experiment/smoke.yaml`): `steps: 50`, `batch_size: 16`,
`lr: 0.003`, `seed: 1234`.

**Tier-1 pilot plan** (`configs/tier1/smollm2_135m.yaml`, frozen): model
`HuggingFaceTB/SmolLM2-135M` @ `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` (apache-2.0, 134 515 008
params); datasets `EleutherAI/wikitext_document_level` @
`647234772b9554e208af6c826f23b99e3cac88c8` (cc-by-sa-3.0) and `allenai/c4` @
`1588ec454efa1a09f29cd18ddd04fe05fc8653a2` (odc-by); seeds master `20261008`, reported
`[0, 1, 2, 3, 4]` (floor 5); grids bits `[2, 3, 4, 8]`, ranks `[2, 4, 8, 16, 32]`, group sizes
`[32, 64, 128]`; budget ladder `[26906055, 40359082, 53812110, 67265138, 88789981]` bytes; harness
`ddd67220430a2470529f25fd5c05a576ca1057a0` (`v0.4.13`); free-tier only (`max_cost_authorized_usd:
0.0`).

**Bounded reproduction plan** (`configs/repro/lr_qat_smollm2_135m.yaml`, frozen): model and datasets
as above; grids bits `[4, 8]`, ranks `[8, 16, 32]`; budget ladder
`[53812110, 67265138, 88789981]`; predeclared tolerance same sign and ≥25 % of the published effect
size. **NOT RUN.**

**Tier-1 pilot result (2026-10-09, first collected cloud run).** `tier1_smollm2_135m-cloud` on the
Kaggle CPU substrate, code commit `9e7c8768b31a91ec726cedfdc832cfc8ccd04743` (clean tree), seed 0,
`seq_len` 2048, 21 758 scored tokens over the `test` split of the pinned WikiText-2 document-level
corpus: fp16 reference 14.0181 perplexity at 212 336 640 accounted bytes; per-group int8 PTQ 14.0412
at 112 803 840 (+0.17 % perplexity for 1.88x fewer bytes); per-group int4 PTQ 18.6190 at 59 719 680
(+32.8 % for 3.56x); untrained rank-8 truncation and rank-8-then-4-bit both degenerate (1.397e16 and
4.649e19, mean relative Frobenius error 0.9427 and 0.9440, flagged `perplexity_degenerate`). All five
arms ran in one collected, checksum-validated bundle with no `--rank`/`--bits` flag, because the plan
binds each arm's grid point. All byte figures cover the 210 targeted projection tensors (106 168 320
params), not the whole model, and are class-1 analytical estimates; the quality figures are class 2
(fake quantization). Rank 8 at 0.94 relative error places the plan's rank grid `[2, 4, 8, 16, 32]` far
below the viable region for this model — recorded here before the trainable arms run, so it is not a
post-hoc choice. Full record: `docs/results/tier1-cloud-run-2026-10-09.md`.

Committed plans are schema-validated by `tests/unit/test_experiment_plan.py`.
