# SpectraQuant — technical report

**Budget-Aware Joint Low-Rank and Low-Bit Training for Efficient Transformers**

Version: 2026-10-10 · repository `github.com/OussemaHarrabi/SpectraQuant` · branch
`research/tier1-closure` · the exact commit, tag and CI links are in
`docs/coordination/handoff-2026-10-10-final.md`.

---

## 1. Abstract

SpectraQuant asks whether *preparing* a transformer for low-rank factorization helps or hurts its
subsequent quantization, and whether an output-aware sensitivity proxy can choose layer-wise ranks and
bit widths that beat uniform compression at equal stored bytes. The project delivers: a joint low-rank
/ low-bit training method with a gain-aware sensitivity proxy and a CP-SAT allocator under a measured
byte budget; a full equal-memory evaluation harness with a measurement taxonomy that never mixes
analytical, fake-quantized, packed-storage and kernel-backed numbers; a reproduction of the LR-QAT
direction at pilot scale; and an honest record of what failed. On SmolLM2-135M with WikiText-2, at
**equal measured bytes (59 719 488 vs 59 719 680, a 0.0003 % match)**, the trained allocated arm scores
**23.1698** perplexity against uniform int4 PTQ's **18.6193** — **the central hypothesis is not
supported** in this configuration. The LR-QAT direction reproduces (24.1265 vs 26.9432) but its
magnitude reaches **21.8 %** of the PTQ damage against a predeclared 25 % requirement. Every number is
labelled by substrate and measurement class; every failure is preserved.

## 2. Research question

Primary: *does preparing a transformer for low-rank factorization improve or damage its subsequent
quantization, and can an output-aware sensitivity proxy select layer-wise ranks and bit widths that
produce a better quality–memory Pareto frontier than uniform compression?* Pre-registered hypotheses
H1–H5 and the frozen protocol are in `docs/research/preregistration.md` (FROZEN 2026-10-09 at
`6281c56`) and `docs/research/reproduction-plan.md`.

## 3. Motivation

Deployed models are often both low-rank *and* low-bit, but the two compressions are studied
separately. If the preparation that makes a matrix factorizable also makes its factors unusually
sensitive to rounding, then optimizing them independently is suboptimal — and a cheap, layer-wise
proxy could allocate the two budgets jointly.

## 4. Related methods and baselines

LR-QAT (arXiv 2406.06385), LoftQ (2310.08659), LQ-LoRA (2311.12023), QLoRA, GPTQ/AWQ, HAWQ-V2,
torchao QAT, and the LM Evaluation Harness. Upstream commits and licences are pinned in
`docs/research/upstream-lockfile.md`; the novelty audit's verdicts are in
`docs/results/verification/m1-novelty-review.md`.

## 5. The SpectraQuant method

`ADR-0004` fixes the specification; `docs/decisions/design-m3-arms.md` and
`docs/decisions/design-trainable-arms.md` fix the arm interfaces.

**Objective.** For a target layer with weight `W`, factors `A (r, in)`, `B (out, r)` on a frozen
quantized base `Q`, the trained loss is

```
L = L_task + λ_factor·factorization_proxy + λ_round·rounding_grid_penalty
           + λ_residual·rounding_residual_ratio + λ_spectrum·spectral_tail_energy
```

with the deployed weight `W_eff = Q + B@A`. **The primary claim is about held-out quality at equal
measured bytes, never about a penalty term** — a run that lowers a penalty while raising perplexity is
a negative result (this is the A-0008 failure mode, made structurally hard to repeat). The Tier-1 plan
declares `λ_factor = 1.0, λ_round = 1.0`, residual and spectral off (A-0020).

**Trainable tensors.** Only `A` and `B`. The base is a buffer; a trainable base would make the result a
QAT result and destroy the comparison against the QAT baselines. The `r2_fullqat_4bit` reference arm
uses a straight-through quantized layer, and the method does not.

**Base / low-rank decomposition.** Two initialisations, both from the paper's Algorithm 1: the
`quantize-then-decompose` step (`Q = q(W)`, `(A,B) = SVD_r(W − Q)`, LoftQ's `T = 1`) and the
**alternating** schedule at `T = 5`. Amendment A-0014 corrected the implementation: it had started from
`SVD_r(W)`, which made its `T = 1` the reference's *second* step.

**Straight-through estimator.** Not used by the method. Documented bias where it is used.

## 6. Gain-aware sensitivity

The proxy unit is `E_X ‖X Wᵀ − X (Q + BA)ᵀ‖²`, estimated on a held-out **calibration** batch. The
declared candidate is `gain_aware_composed` (a gain-aware output-error form); its predeclared
comparators are weight-Frobenius, weight-magnitude, activation-magnitude, Hessian-diagonal and naive
output-error. On the Tier-0 fixture it beats every comparator in **15/15** aggregate Fisher-z contrasts
(model-level unit, 5 seeds × 12 cells × 2 fixtures) — **at an achieved MDE wider than the predeclared
conservative end for its own contrast**, i.e. supported at fixture scale and underpowered for that
contrast. On a real model it did not translate into a better configuration (§13).

## 7. CP-SAT rank/precision allocation

`AllocationProblem(layer_shapes, ranks, bits, budget_bytes, cost_fn, error_fn)` with
`cost_fn = make_accounting_cost_fn` (the single byte-accounting source of truth) and `error_fn` = the
proxy. Solved by CP-SAT, with uniform and greedy baselines and an exhaustive oracle on tiny cases
(CP-SAT == exhaustive at three budgets). On SmolLM2-135M at a 59 719 680-byte budget it chose
**bits 2 (5 layers), 3 (76), 4 (122), 8 (7)** and **rank 32 (206 of 210; four clamped by the SVD)**,
landing **192 bytes** under the budget.

## 8. Quantization and low-rank training details

Uniform symmetric per-group quantization (class 2) plus a 2-bit NF-style codebook (block 64) whose
level table and block semantics reproduce the pinned reference **exactly** (`max_abs_diff = 0.0`); a
Q4.4 fixed-point container and the paper's merge for LR-QAT; `QuantizedPlusLowRankLinear` (frozen base
buffer + trainable factors); checkpoint/resume; the numerical primitives run on CPU and their results
reach the device, with the preparation penalty attached by a manual vector–Jacobian product so the
regularizer cannot silently become a no-op.

## 9. Experimental setup

| element | value |
|---|---|
| Model | `HuggingFaceTB/SmolLM2-135M` @ `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` (Apache-2.0, 134 515 008 params) |
| Corpus | `EleutherAI/wikitext_document_level` config `wikitext-2-raw-v1` @ `647234772b9554e208af6c826f23b99e3cac88c8`, split `test` |
| Train / dev | streamed `allenai/c4` (`en`) train role; WikiText-2 `validation` — never the test split |
| Evaluation | non-overlapping-window token-level cross-entropy; **21 758 tokens**, seq 512 (train) / 2048 (eval) |
| Substrate | Kaggle free tier, Tesla T4 (sm_75), CUDA 12.8, torch 2.11.0+cu128 |
| Seeds | 0 for every cloud cell; the frozen pilot's 5-seed floor was **not** reachable (A-0018) |
| Measurement classes | 1 (analytical bytes), 2 (fake-quantization quality), 3 (packed storage), 4-CPU (ONNX Runtime kernels) |

## 10. Compute environment and budget

The workstation is CPU-only (no CUDA), so all training and evaluation ran on the cloud substrate via
generated notebooks; the local machine orchestrates, validates and analyses. Consumption is estimated
at **≈ 26 GPU-hours** of a **30-hour ceiling** — Kaggle exposes no quota API — with a mandated 3-hour
reserve. The ledger is `docs/results/compute-budget-ledger.md`.

## 11. Frozen protocol and its amendment

The frozen M3 protocol (5 seeds × 1000 steps × 8 arms, ≈ 19.5 h) **does not fit**. Amendment **A-0018**
preserves it unchanged, marks it partially executed, records what was already visible, and defines the
resource-constrained subset (the minimum arm set that can decide T1) with its criteria fixed before the
run. A-0013, A-0014, A-0015, A-0016, A-0017, A-0019 and A-0020 record the other corrections; the log is
append-only.

## 12. Results

### 12.1 Tier-1 equal-memory comparison (`docs/results/tier1-method-comparison.md`)

| arm | perplexity | accounted B | measured B | class |
|---|---|---|---|---|
| `fp16_reference` | 14.0183 | 212 336 640 | — | — |
| `ptq_uniform_8` | 14.0415 | 112 803 840 | 112 803 840 | 2 |
| **`ptq_uniform_4`** | **18.6193** | **59 719 680** | 59 719 680 | 2 |
| `proxy_allocated` (untrained) | 64.5477 | 59 719 488 | 59 719 488 | 2 |
| `spectraquant_regularized` (**regularizer off** — A-0020) | 23.1698 | 59 719 488 | 59 719 488 | 2 |
| `low_rank_only`, `rank_then_quant` | degenerate | 4 884 480 / 1 598 400 | — | 2 |

### 12.2 M3 Tier-1 subset (`docs/results/m3-verdict.md`)

`r2_fp16` 14.0183 · `r2_rtn_4bit` **26.9432** @ 54 858 240 B · `r2_lrqat_4bit` **24.1265** @
74 397 000 B, 1000 steps, rate selected on validation.

### 12.3 Tier-0 fixture evidence

Proxy 15/15 contrasts; allocation at identical measured bytes reduces hidden-state damage to a ratio of
`0.3668` of uniform (**2.73×**); 18/18 byte reconciliations at zero difference; the regularizer reduces
the rounding residual by **93.98 %** while **worsening** dev NLL; a self-serialized int4 ONNX container
runs through ONNX Runtime's `MatMulNBits` (relative error 0.0771, class 4-CPU, **no speedup claimed**).

## 13. Equal-memory comparison and what it shows

The method and uniform int4 PTQ are compared at a **192-byte** (0.0003 %) match, with class-3 measured
bytes equal to class-1 accounted bytes for the allocated arms. The untrained allocation is **3.5× worse**
than uniform int4; training the factors recovers **2.8×** of that but still leaves **+24.4 %**
perplexity. **The gain-aware proxy's ranking did not produce a better configuration at this budget** —
the most likely reasons are that a 135 M model at 3.56× compression cannot afford 2- and 3-bit layers
on most of its matrices, and that the proxy ranks *per-layer* sensitivity without accounting for how
errors compound across 30 layers. That is a finding, not a failure of the harness.

## 14. Ablations

| ablation | arms | result |
|---|---|---|
| regularizer on vs off | `spectraquant_regularized` (λ=0) vs the λ-declared run | the λ=0 arm is the 23.1698 row above; the regularized run is reported separately (A-0020) |
| allocation without training | `proxy_allocated` 64.5477 | isolates the training effect: 2.8× |
| proxy ranking vs comparators | 15/15 contrasts on the fixture | supported at fixture scale, underpowered for the candidate's own contrast |
| LoftQ `T = 5` vs `T = 1` | not executed (A-0018) | — |
| CP-SAT vs uniform at equal bytes | uniform int4 18.6193 vs allocated 23.1698 | uniform wins in this configuration |

## 15. Statistical analysis

**Unit and power.** The proxy result's unit is the **model** (one ρ per trained model per cell) with a
within-model bootstrap for `n_eff` and a DerSimonian–Laird random-effects model; no pseudoreplication
was found (seeds and modules are never treated as independent model-level replicates). The cloud cells
are **one seed each**; five seeds provide limited power in any case, and the frozen 5-seed floor was not
reachable. Language used: "across the seeds run", "the observed mean difference was", "the confidence
interval remained wide", "supports / does not support the preregistered hypothesis under the tested
configuration". No "proved universally", no "statistically significant" beyond what the analysis
supports, no "state of the art", no "production-ready".

## 16. Negative and unsupported findings (published, not hidden)

1. **The method does not beat uniform int4 at equal bytes** (+24.4 %).
2. **The untrained proxy allocation is much worse** (+246.7 %).
3. **The LR-QAT magnitude falls short** of the predeclared 25 % (21.8 %).
4. **The regularizer worsened dev NLL on the fixture** while reducing the rounding residual (A-0008).
5. **The first "method" run measured the unregularized ablation** because the plan declared no
   coefficients (A-0020).
6. **An M3 run trained all eight arms and then failed validation at export** (disk exhaustion), so its
   numbers are not claimable (A-0018).
7. **The LoftQ `T = 1` implementation had been the reference's second step** (A-0014).
8. **The `1e-6` elementwise oracle tolerance is unachievable** on a near-degenerate rank-16 subspace
   (A-0015).
9. **The hidden-state-damage improvement did not translate into task-level quality** — the two axes
   disagree in sign, and the disagreement is reported.

## 17. Threats to validity

Single seed per cloud cell; one model and one corpus; 21 758 evaluation tokens (a cap, not the full
split); the byte figures cover the 210 targeted projection tensors, not the deployed model; the
allocated arms' T1 comparison is at equal bit width, not equal memory; class 2 is fake quantization and
licenses no storage or latency claim; T4 is sm_75, so bf16 and int4 `tinygemm` are unavailable and no
class-4-GPU number exists; the regularized method's own cell is a single run.

## 18. Reproducibility

`uv sync --all-extras`, then `uv run pytest -q`, `uv run ruff check .`, `uv run ruff format --check .`,
`uv run pyright`. A **clean clone** of the pushed branch installs and passes: **1160 passed /
4 skipped** at `8f73024`, ruff and pyright clean, and the smoke experiment writes its manifest. The
four skips are the conditional bundle-presence paths (`artifacts/runs/` is gitignored); the local tree
with the collected bundles passes 1161 / 3 skipped. Every quoted count names its commit. Cloud cells are
reproduced with the four commands in §12.1/§12.2. Every claim maps to an artifact in
`docs/results/experiment-index.md`.

## 19. Limitations

No Tier-2 confirmation; no multi-seed cloud evidence; no LM-Eval task suite (DEV-0009); no class-4-GPU
or class-5 measurement; no allocator frontier on a real model beyond the single budget exercised; the
proxy's real-model ranking was not measured (no H2 cell).

## 20. Future Tier-2 work

Run the frozen protocol on TinyLlama-1.1B at three seeds with the pinned plan, after a corpus-sized
schedule is pinned from a measured step cost; add the LM-Eval suite; measure the proxy's real-model
ranking against the observed damage; and re-solve the method at the *baseline's* bytes so the central
comparison is equal-memory by construction.

## 21. Conclusion

The harness, the taxonomy and the honesty machinery work: an exact equal-memory comparison, class-3
bytes matching class-1 accounting, a mixed CP-SAT allocation solved to 192 bytes, a reproduced
direction with a missed magnitude, and every failure preserved in an append-only record. The central
claim is **not supported** at this scale: joint low-rank/low-bit training with a gain-aware proxy did
not beat uniform int4 quantization at equal measured bytes on SmolLM2-135M.
