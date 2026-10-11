# Compute-budget ledger — Kaggle GPU, 2026-10-10

Machine-readable twin: `artifacts/manifests/compute-budget-ledger.json`. Written before the next GPU
launch, per the continuation brief. Every rate below is either **measured** (named where) or marked
**estimated**.

## 1. Opening balance

> **Superseded by §8.** The table below is the **pre-launch** balance written on 2026-10-10 *before*
> the Tier-1 comparison and the M3 subset ran. The final accounting is in §8.

Kaggle exposes no quota API, so consumption is estimated from session wall-clock:

| Session | GPU wall-clock | Source |
|---|---|---|
| `repro_lr_qat_loftq_smollm2-135m-cloud` (2 collected + 1 operator-stopped) | ≈ 8.5 h | last-run timestamps + the incident log |
| `m3-tier1-smollm2-135m-cloud` (1 attempt whose output expired ≈ 6 h; 1 running) | ≈ 6 h + in flight | kernel last-run timestamps |
| `tier1-smollm2_135m-cloud` (2 failed method attempts) | ≈ 1 h | logs (300 steps each + install) |
| **Estimated consumed** | **≈ 15.5 h** | |

Against the brief's **30 GPU-hour ceiling** this leaves **≈ 14.5 h**, less whatever the running M3
pilot consumes. **Planning budget: 11 h of new work with a 3 h untouched reserve.** If the running
pilot consumes ~4 h, the new-work budget is ~7 h. The ledger is written to fit **either** case.

## 2. Measured rates

| Quantity | Value | Source |
|---|---|---|
| Optimiser step, batch 8 × seq 512, T4, rank-8/4-bit diagnostic arms | **2.17 s** | `repro_lr_qat_loftq_smollm2-135m-cloud` (433 s / 200 steps) |
| Tier-1 plan: 300 steps, one trainable arm | ≈ 11 min | same rate |
| Per-run fixed cost (uv sync + CUDA torch + model + dataset) | ≈ 15 min | every cloud run's log |
| Perplexity evaluation, 21 758 tokens | ≈ 1–2 min | every cloud run's log |
| CP-SAT allocation, 210 layers × 5 ranks × 4 bits | < 1 min | the method run's log |
| Activation capture, 256 rows/layer × 210 layers | < 1 min | the method run's log |

**Unknown and needed:** the per-step rate of the *frozen M3 arms* (rank 16/32 with the 2-bit
codebook, and `r2_fullqat_4bit` which trains every linear weight). The running seed-0 pilot logs it
per arm; the plan below is provisional until that log is collected, and no M3 seed beyond seed 0 is
launched before it is.

## 3. Arm ledger

`Priority`: **P1** central comparison · **P2** necessary baseline · **P3** multi-seed · **P4** ablation
· **P5** analysis. `Det.` = deterministic output for a fixed configuration (one seed suffices).

### 3.1 Tier-1 method plan (`configs/tier1/smollm2_135m.yaml`) — the central comparison

| Arm | Purpose | Train | Det. | Seeds done | Seeds planned | Steps | Est. per seed | Prio | Status |
|---|---|---|---|---|---|---|---|---|---|
| `spectraquant_regularized` | **the method** (proxy allocation + regularized factor training) | yes | no | 0 | **3** | 300 | 11 min | **P1/P3** | wired, serialization fixed, resubmit pending |
| `proxy_allocated` | ablation: same allocation, **no** regularizer | no | yes | 0 | 1 | — | 2 min | **P4** | scored 64.5477 at equal bytes in the failed run (to be re-measured) |
| `ptq_uniform_4` | strongest equal-memory baseline (uniform int4) | no | yes | 0 | 1 | — | 2 min | **P2** | 18.6190 measured on the CPU pilot |
| `ptq_uniform_8` | uniform int8 baseline | no | yes | 0 | 1 | — | 2 min | P2 | 14.0412 measured on the CPU pilot |
| `fp16_reference` | upper reference | no | yes | 0 | 1 | — | 2 min | P2 | 14.0181/14.0183 measured twice |
| `low_rank_only`, `rank_then_quant` | untrained factorization baselines | no | yes | 0 | 1 | — | 2 min | P2 | degenerate on the CPU pilot |
| `quant_then_residual` | quantize-then-residual baseline | no | yes | 0 | 1 | — | 2 min | P2 | wired |

**One Tier-1 run executes all eight arms**: ≈ 11 min of training + ≈ 6 × 2 min of evaluation + ≈ 15 min
fixed ≈ **31 min**. Three seeds (the method arm re-trained per seed; the deterministic arms need no
repetition, but they are cheap and re-measured on the same substrate for comparability) ≈ **1.6 h**.

### 3.2 Frozen M3 plan (`configs/m3/tier1_smollm2_135m.yaml`) — the reproduction

| Arm | Purpose | Train | Det. | Seeds done | Seeds planned | Steps | Est. per seed | Prio | Status |
|---|---|---|---|---|---|---|---|---|---|
| `r1_fp16_lora` | T2 reference (fp16 + standard adapter) | yes | no | 0 | 1 | 1000 | ≈ 36 min | P2 | wired |
| `r1_std_2bit` | T2 baseline (2-bit base, standard init) | yes | no | 0 | 1 | 1000 | ≈ 36 min | P2 | wired |
| `r1_loftq_2bit` | T2 treatment (T=5) | yes | no | 0 | 1 | 1000 | ≈ 36 min | **P1(T2)** | wired |
| `r1_loftq_2bit_t1` | T2 ablation (T=1) | yes | no | 0 | 1 | 1000 | ≈ 36 min | **P4** | wired |
| `r2_fp16` | T1 reference | no | yes | 0 | 1 | — | 2 min | P2 | wired |
| `r2_rtn_4bit` | T1 baseline (RTN 4-bit g128) | no | yes | 0 | 1 | — | 2 min | P2 | wired |
| `r2_lrqat_4bit` | T1 treatment | yes | no | 0 | 1 | 1000 | ≈ 36 min | **P1(T1)** | wired |
| `r2_fullqat_4bit` | T1 upper reference (all weights trained) | yes | no | 0 | 1 | 1000 | **unknown, likely ≫ 36 min** | P2 | wired; DEV-0012 flags the memory risk |

**Per seed ≈ 3.6 h** at the 2.17 s/step estimate (6 trainable arms), plus ≈ 15 min fixed and ≈ 2 min
per evaluation-only arm → **≈ 3.9 h per seed**. The frozen protocol wants **5 seeds ⇒ ≈ 19.5 h**,
which **does not fit** the remaining budget (§4).

## 4. Does the frozen protocol fit? **No.**

The frozen M3 protocol (`reproduction-plan.md` §3, §5.1: 5 seeds × 1000 steps × 8 arms) costs
≈ 19.5 h. The remaining budget is ≈ 14.5 h minus the running pilot (≈ 4 h) ⇒ **≈ 10.5 h**. Even at
3 seeds it costs ≈ 11.7 h, i.e. more than everything else combined and more than the reserve allows.

Therefore, per Phase 2 of the brief, the original protocol is **preserved unchanged** and a
**timestamped compute-budget amendment** (A-0018) will record: the original frozen design, the pilot
observations that were already visible, the resource-constrained confirmatory subset, and the
unexecuted work. The subset is chosen on scientific relevance, not on which pilot looks strongest,
and its primary comparison and success criteria are fixed **before** the remaining seeds run.

## 5. Planned run order (each step gated on the previous one's collection)

| # | Run | Est. GPU | Gate before it starts | Stop condition |
|---|---|---|---|---|
| 1 | **M3 seed 0** (running) | in flight | — | collect within minutes of completion; if it fails, read the log and fix before any resubmission |
| 2 | **Tier-1 method, seed 0** | ≈ 31 min | serialization fix committed; the run must export per-arm results | if it fails, fix locally; no second attempt without a diagnosis |
| 3 | **Tier-1 method, seeds 1–2** | ≈ 1.1 h | seed 0 produced a valid bundle | stop if seed 0 is divergent or the bundle is uncollectable |
| 4 | **M3 confirmatory subset** (per A-0018; provisional 2 seeds of the stochastic arms only) | ≈ 7.2 h at the estimate, **re-planned from the pilot's measured per-arm rates** | the pilot's log gives per-arm costs | never launch if it would take the remaining budget below the 3 h reserve |
| 5 | **H2/H4 analysis** | **0 GPU** | — | CPU/local only, from the captured activations and recorded proxy scores |

**Projected usage if everything succeeds:** ≈ 31 min + 1.1 h + 7.2 h ≈ **9 h**, inside the 11 h
new-work budget with the reserve intact. If the pilot's measured rates are worse than 2.17 s/step,
step 4 shrinks (fewer seeds), never the reserve.

## 6. What will not consume GPU time

Documentation, plot styling, statistics, unit tests, configuration generation, schema work, report
writing, artifact indexing, the compute ledger, the claims audit and the H2/H4 analysis — all local or
CPU. The oversized oracle artifact's compaction (Phase 6) is local.

## 7. Reserve policy

At least **3 GPU-hours** stay untouched. When the remaining estimate reaches 3 h, **no new GPU work
starts**; the project reports what it has. Improving an already-usable headline number is not a
reason to spend the reserve.

---

## 8. Final accounting (2026-10-11) — **estimate, not a measurement**

**There is no measured GPU-hours figure in this repository, and none can be obtained.** Kaggle exposes
no quota API, and the run manifests record only `timestamp_utc` (completion) — never a start time or a
duration. Every number below is reconstructed from kernel wall-clock, so treat it as **±3 h**.

| # | Session / attempt | GPU wall-clock | Outcome |
|---|---|---|---|
| 1 | `repro_lr_qat_loftq_smollm2-135m-cloud` — 2 collected + 1 operator-stopped | ≈ 8.50 h | diagnostic only (no M3 verdict) |
| 2 | `m3-tier1-smollm2-135m-cloud` — attempt 1: output expired before collection | ≈ 6.00 h | **nothing** |
| 3 | `m3-tier1-smollm2-135m-cloud` — attempt 2: all 8 arms trained, **failed validation at export** | ≈ 6.50 h | **nothing** |
| 4 | `m3-tier1-smollm2-135m-cloud` — attempt 3: the Tier-1 subset (3 arms) | ≈ 1.50 h | **validated** → `docs/results/m3-verdict.md` |
| 5 | `tier1_smollm2_135m-cloud` — attempts 1-2: serialization and granularity defects | ≈ 1.00 h | **nothing** |
| 6 | `tier1_smollm2_135m-cloud` — attempt 3: the equal-memory comparison | ≈ 0.50 h | **validated** → `docs/results/tier1-method-comparison.md` |
| 7 | `tier1_smollm2_135m-cloud` — attempt 4: the regularizer-on ablation | ≈ 0.75 h | **validated** (same record) |
| | **Total** | **≈ 24.75 h** | of the brief's **30 h ceiling** |

### What the hours bought

| | GPU-hours | share |
|---|---|---|
| Runs that produced a **claimable** artifact | ≈ 11.25 h | **45 %** |
| Runs that produced **nothing** (expiry, failed export, tooling defects) | ≈ **13.50 h** | **55 %** |

**More than half the GPU budget was spent on runs that produced no result**, and two of them
(§ rows 2-3, ≈ 12.5 h) were the *same* run attempted twice at full 8-arm scale. The lessons are
recorded rather than glossed: the export-validation failure was a disk-exhaustion defect that a
pre-flight size check would have caught, and the expiry was an unbounded-wall-clock submission
(A-0018).

### Reserve

The mandated **3 h reserve was never touched**. Work stopped at ≈ 24.75 h, leaving ≈ 5.25 h of
headroom against the ceiling — deliberately unused, because the only remaining GPU work was Tier-2
(≈ 19.5 h for the frozen protocol), which could not fit and would have consumed the reserve for a
partial result.

