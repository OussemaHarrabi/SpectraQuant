# SpectraQuant — Pre-registration

Status: **DRAFT pending freeze.** This document is frozen **at milestone M1** (preregistration
freeze) and, in any case, before the first confirmatory run of any kind. Once frozen, it is modified
**only** through `preregistration-amendments.md` (append-only, timestamped, reason-stated).
Exploratory Tier-0/1 work that informs *method design* may precede M1 and is labelled
**exploratory**; no confirmatory claim may be sourced from it.

Authored 2026-10-08. Compute envelope per `docs/research/environment.md` and `AGENTS.md` §6.

---

## 1. Objectives and the two primary outcomes

A confirmatory result must declare **one primary quality outcome and one primary compression
outcome**. Both are fixed here; no substituting a friendlier metric after seeing results.

- **Primary quality outcome (P1 — quality).** Language-modelling quality of the compressed model,
  measured as **per-token cross-entropy in nats on a held-out set, reported as perplexity** (and the
  raw mean NLL, which is the statistic actually tested). Execution is **fake-quantization**
  (float arithmetic simulating quantize→dequantize) ⇒ **measurement class 2**. On Tier 0 the analogue
  is **mean-squared output error** of the compressed linear map against the uncompressed map on
  held-out inputs (also class 2).
- **Primary compression outcome (P2 — memory).** **Stored bytes of the compressed weight artifact**,
  measured by actually serializing the artifact (weights at the chosen format + all side information:
  per-group scales, zero-points, low-rank factors, any codebook) ⇒ **measurement class 3** for
  formats we serialize ourselves (int8/int4 packing, low-rank factor storage), accompanied by the
  **class-1 analytical estimate** for cross-checking. Compression ratio is stored bytes relative to
  an fp16 checkpoint of the same model.

P1 and P2 are **only ever compared at equal P2**. Every compression comparison reports the
equal-memory counterpart of the baseline (`AGENTS.md` §4.5). Training-time memory and deployed
inference representation are reported **separately** and never conflated with P2.

**H5 uses additional classes.** The H5 chain is P1 (class 2, fake quantization) → packed storage
(class 3, measured serialized bytes) → **class 4-CPU** execution of our own serialized container by a
real CPU low-bit kernel (ONNX Runtime `MatMulNBits`/`MatMulInteger` or torchao intx weight-only),
always against an fp32 CPU baseline measured in the same session. Class 4-CPU carries no
latency/throughput claim and is never compared to published GPU numbers; class 4-GPU and class 5 are
"not measured" here.

## 2. Secondary outcomes

- **S1.** Per-layer post-compression output distortion (class 2) — used for proxy-validity analysis.
- **S2.** Spearman and Pearson correlation between proxy scores and measured per-layer damage.
- **S3.** Pareto-frontier **hypervolume** over (quality, stored-bytes) points at predeclared budgets.
- **S4.** Zero-shot accuracy on a small fixed task set (Tier 1: a predeclared subset; Tier 2:
  lm-evaluation-harness tasks) — reported but never used as the primary claim.
- **S5.** Byte-level accounting accuracy: |class-3 measured − class-1 analytical| / class-1.
- **S6.** Wall-clock and memory of the proxy itself (probe count, calibration activations used).
- **S7.** Rank/allocation stability across seeds (same layers get high rank/bit?).

## 3. Datasets, splits, and contamination control

| Tier | Dataset | Train | Dev (validation) | Test (held-out) |
|---|---|---|---|---|
| 0 | Synthetic matrices with controlled spectra (power-law, exponential, near-rank-1, random) + tiny MLP regression | generated with fixed RNG seeds | disjoint RNG seeds | disjoint RNG seeds (fixed once, reused for all arms) |
| 1 | Primary LM corpus: WikiText-2 (raw, official splits) subsampled deterministically | official train | official validation | official test |
| 1 | Secondary LM corpus for trainability/robustness: TinyStories subset (deterministic slice) | train slice | validation slice | test slice |
| 2 | WikiText-2 test (official) + lm-evaluation-harness tasks | (pretrained model; no training unless QAT/LoRA arm) | calibration from train, dev = WikiText-2 validation | WikiText-2 test evaluated **once** per arm at freeze |

Tier 0 runs locally (`LOCAL-FIXTURE`); **Tier 1 and Tier 2 datasets are used on the cloud substrate
only** (`AGENTS.md` §2b; see §4). Local copies exist solely for fixture-scale unit tests and for
analysing downloaded artifacts.

**Calibration / dev / test separation (mandatory).**

1. **Calibration set** (for GPTQ-style layer input statistics, AWQ-style scaling, and any activation
   statistic used by the proxy) is drawn **only from the training split** (or the synthetic train
   generator). It MUST be disjoint from dev and test. A programmatic train/eval overlap check runs
   before every run and its result is recorded in the manifest.
2. **Dev/validation** is used freely for method selection, budget sweeps, and hyperparameter choice.
   That is its purpose; it is not a test set.
3. **Test** is touched only by the frozen confirmatory protocol. The number of test evaluations per
   arm is predeclared (≤ 2: one confirmatory read, one re-read after a recorded bug fix) and logged.
   **No tuning on test labels; no repeated test peeking to steer method decisions** (`AGENTS.md`
   §4.6).
4. **No training on LM Evaluation Harness evaluation questions** (`AGENTS.md` §4.7); the harness task
   list is fixed at freeze and its data provenance recorded.

## 4. Model families and execution substrate

**Substrate policy (binding, `AGENTS.md` §2b; amendment A-0004).** The local workstation may not run
research training, QAT, large-scale inference or GPU evaluation. Local work = repository management,
CPU unit/property tests, tiny synthetic fixtures, static analysis, config validation, notebook
generation, result analysis, figures/tables/reports, and the class 4-CPU kernel *measurement*.
**All model training and evaluation (Tier 1–5) runs on the cloud notebook substrate** — Google Colab
(developer's chosen vehicle; notebooks generated from versioned configs, executed by the developer),
Kaggle Notebooks (preferred unattended backend, programmatic CLI), or Colab Enterprise with an
authorized GCP project. Cloud is a **plan, not a result**, until a validated `run_manifest.json` and
checksum-validated artifacts exist.

- **Tier 0 (required, `LOCAL-FIXTURE`):** synthetic weight matrices with controlled spectral decay; a
  small MLP regression net. Purpose: exact per-layer damage by enumeration; proxy-vs-exact agreement.
- **Tier 1 primary (required, `CLOUD-COLAB`):** a **decoder-only transformer trained from scratch on
  the cloud substrate** on the Tier-1 corpus (small: a few layers, modest width; exact config pinned in
  `configs/` at freeze, not chosen after seeing results). This is the confirmatory vehicle for H1–H5.
  It is **not** a local workload.
- **Tier 1 secondary (inference-only, `CLOUD-COLAB`):** a ~125M pretrained LM (e.g. GPT-2-class) used
  **forward-only** for proxy-validity checks (S2). No finetuning.
- **Tier 2 (`CLOUD-GPU`, planned):** TinyLlama-1.1B-class + WikiText-2.
- **Tier 3 (`CLOUD-GPU`, planned):** one second model in the 0.6–1.7B range.
- **Tier 4 (optional, `CLOUD-GPU`):** ViT/DeiT on CIFAR-100/ImageNet-100.
- **Tier 5 (optional, `CLOUD-GPU`):** 3B–7B. **Core validity MUST NOT depend on Tier 5.**
- **Class 4-CPU measurement (`LOCAL-CPU-MEASUREMENT`):** executing a **self-serialized** low-bit
  container through a real CPU kernel (ONNX Runtime `MatMulNBits`/`MatMulInteger`, torchao intx) — a
  measurement on an artifact (local for Tier-0 fixtures, or on an artifact exported from a cloud run),
  not training.

Final checkpoint identifiers and licenses are recorded in `docs/research/upstream-lockfile.md` before
any run; no model enters the core path without a recorded license.

## 5. Compression configs (predeclared grids)

- **Bit widths $b$:** {2, 3, 4, 8} for weights; per-group symmetric and asymmetric as separate arms
  (group sizes predeclared, e.g. 32/64/128).
- **Ranks $r$:** {2, 4, 8, 16, 32} (scaled to layer dimensions; final list pinned at freeze).
- **Byte budgets** for the Pareto sweep: a predeclared geometric ladder of stored-byte targets.
- **Arms:** (a) uniform-$b$ full-rank; (b) uniform-$r$ then uniform-$b$ (rank-then-quantize);
  (c) quantize-then-low-rank-residual (LoftQ-style); (d) proxy-allocated $(r_\ell,b_\ell)$;
  (e) proxy-allocated with the regularizer (**H3 arm**); (f) reference: fp16 uncompressed.
- **Ordering arms (Q2 comparator, *not* a hypothesis):** (b) vs (c) at identical $r$ and $b$ and
  identical stored bytes. Reported as a mechanistic comparison feeding H1 and H4.
- **Comparators:** weight-space Frobenius error is the **predeclared H2 comparator**; HAWQ-V2-style
  trace allocation is the **H4 bit-only comparator** at the same byte budget.
- **H5 arm (packed-storage and CPU-kernel survival):** for each arm that shows a fake-quantization
  quality gain, (i) re-measure quality after our **own packed serialization** (int4/int8) with CPU
  dequantization at measured stored-byte parity, then (ii) re-measure it when the same serialized
  artifact is executed by a **real CPU low-bit kernel** (class 4-CPU) against an fp32 CPU baseline in
  the same session. Report which stage the gain survives; make no GPU or latency claim.

## 6. Seed plan

- **Master seed** fixed (recorded in every manifest). Derived streams: init, data order, calibration
  sampling, probe sampling — separated by documented derivation so that changing one does not
  perturb the others.
- **Tier 0 (local fixtures):** ≥ **5 seeds** for every reported cell; exact enumeration where feasible.
- **Tier 1 (cloud, trainable comparisons):** ≥ **5 seeds** for cheap configurations; ≥ **3 seeds**
  where a full-factorial sweep is too expensive, and the reduction to 3 MUST be justified in a
  manifest note. All Tier-1 training runs execute on the cloud substrate (Colab/Kaggle), never locally;
  a session pre-emption is resumed via the persisted remote run id, or recorded as failed (see §10).
- **Tier 2+ (cloud GPU):** ≥ **3 seeds** for trainable arms.
- A result may not be reported from fewer seeds than the floor. Seeds are not dropped because their
  outcome is unfavourable (see §9 exclusion).

## 7. Statistical analysis plan

1. **Paired comparisons.** Arm A vs arm B at equal stored bytes: paired differences over matched
   seeds (and, for Tier 0, matched matrices). Report mean difference, **paired bootstrap** 95% CI
   (≥ 10,000 resamples, percentile + BCa), and **Cliff's delta** effect size.
2. **Proxy validity (S2, tests H2).** Spearman ρ and Pearson r between proxy scores and measured
   per-layer (and end-to-end) degradation, with **bootstrap CIs** (per-layer resampling) and Fisher-z
   intervals. **Predeclared primary comparator: weight-space Frobenius error** (the exact wording of
   H2). Secondary comparators: weight magnitude, activation magnitude, Hessian trace, and (if
   implementable) the random-probe estimator of arXiv:2609.33923. H2 is tested against the primary
   comparator with Holm–Bonferroni; secondary comparators are labelled exploratory.
3. **Frontier comparison (H4).** Pareto hypervolume difference with bootstrap CI; dominance declared
   only if the CI excludes 0 and the byte saving at matched quality exceeds a predeclared equivalence
   margin.
4. **Multiple comparisons.** Within each hypothesis family, **Holm–Bonferroni** across the predeclared
   tests. Exploratory sweeps (not in the confirmatory family) use **Benjamini–Hochberg** and are
   labelled exploratory; they cannot support a primary claim.
5. **No optional stopping.** Confirmatory tests use the fixed seed count and fixed test reads
   (§3.3). No peeking to decide whether to continue.
6. **Reporting.** Every reported number carries a measurement class label (`AGENTS.md` §5) and its
   seed count; negative and null results are reported (`AGENTS.md` §4.10).

## 8. Confirmatory test matrix, execution substrate, and what CANNOT run locally

**Substrate tags** (policy `AGENTS.md` §2b; amendment A-0004):
`LOCAL-FIXTURE` = tiny synthetic/unit fixtures on the CPU workstation; `LOCAL-CPU-MEASUREMENT` = a
class-4-CPU kernel measurement on a **self-serialized** low-bit artifact (not training);
`CLOUD-COLAB` = Tier-1 training/evaluation on the cloud notebook substrate (Colab, developer-run) —
Kaggle Notebooks may substitute as the unattended backend; `CLOUD-GPU` = Tier 2–5 on the cloud
substrate. **"Cloud" is a plan, not a result**, until a validated `run_manifest.json` and
checksum-validated artifacts exist. Anything requiring CUDA **on the local host** stays tagged
**NOT RUNNABLE locally** and must fail loudly.

| Cell | Hypothesis tested (canonical ID) | Tier | Substrate | Milestone | Status |
|---|---|---|---|---|---|
| Factorization error vs. rounding sensitivity, synthetic spectra + tiny net, equal bytes | **H1** | 0 | `LOCAL-FIXTURE` | M2/M4 | **runnable** |
| Factorization error vs. rounding sensitivity, tiny LM, ≥5 seeds | **H1** | 1 | `CLOUD-COLAB` | M4 | **planned (cloud)** |
| Q2 comparator: rank-then-quantize vs. quantize-then-residual (mechanistic, *not* a hypothesis) | Q2 (feeds H1, H4) | 1 | `CLOUD-COLAB` | M4 | **planned (cloud)** |
| Proxy vs. **weight-space Frobenius error** rank correlation (activation covariance + quantization residual in the proxy) | **H2** | 0–1 | `LOCAL-FIXTURE` + `CLOUD-COLAB` | M4 | **runnable (Tier 0) / planned (Tier 1)** |
| Proxy comparison on a ~125M pretrained LM (forward-only) | **H2** | 1 | `CLOUD-COLAB` | M4 | **planned (cloud)** |
| Rounding-aware spectral regularizer, with/without + plain-spectral control, equal bytes | **H3** | 1 | `CLOUD-COLAB` | M4 | **planned (cloud)** |
| Layer-wise mixed rank + mixed precision vs. uniform and vs. HAWQ-V2-style, ≥5 seeds | **H4** | 1 | `CLOUD-COLAB` | M4 | **planned (cloud)** |
| **Packed-storage survival:** re-measure at parity after our own int4/int8 serialization + CPU dequant | **H5** | 0/1 | `LOCAL-CPU-MEASUREMENT` | M4 | **runnable** (class 2 → class 3) |
| **CPU-kernel survival:** our own serialized int4/int8 container executed by a real **CPU** low-bit kernel (ONNX Runtime `MatMulNBits` int4 weight-only / `MatMulInteger` int8, or torchao intx) vs. an fp32 CPU baseline in the same session | **H5** | 0/1 | `LOCAL-CPU-MEASUREMENT` | M4 | **runnable** (class 4-CPU; CPU scope only — **not** comparable to published GPU numbers; **no** latency/throughput claim) |
| TinyLlama-1.1B PTQ / QLoRA / LoftQ / LR-QAT / SpectraQuant, ≥3 seeds | **H1–H5** | 2 | `CLOUD-GPU` | M7 | **planned (cloud GPU)** |
| Second 0.6–1.7B model | **H1–H5** | 3 | `CLOUD-GPU` | M7 | **planned (cloud GPU; paid instance ⇒ prior authorization + cost ceiling)** |
| ViT/DeiT CIFAR-100 / ImageNet-100 | **H1, H2, H4** | 4 | `CLOUD-GPU` | later | **planned (cloud GPU; authorization + cost ceiling required)** |
| 3B–7B scale | — | 5 | `CLOUD-GPU` | later | **planned (cloud GPU; authorization + cost ceiling required; not required for validity)** |
| **H5 GPU half:** class 4-GPU kernel-backed inference and class 5 service latency/throughput | **H5** | any | `CLOUD-GPU` | M7 | **claimable ONLY from a validated cloud run with a real supported kernel** (`AGENTS.md` §2b rule 9); otherwise "not measured" |
| Any upstream CUDA-only path (bitsandbytes 4-bit, GPTQ/AWQ CUDA kernels, TorchAO CUDA, vLLM) invoked on the local host | — | any | local | — | **[NOT RUNNABLE locally — must fail loudly; CUDA paths are cloud-only]** |

**Binding statement.** The local machine produces the **Tier-0 fixtures**, the **class 4-CPU
measurement**, and all analysis/figures/reports. **Tier 1–5 execute only on the cloud substrate**; no
Tier-1+ number may be reported as measured locally (`AGENTS.md` §2b; `environment.md` §5). A cloud cell
counts as a **result** only once a validated `run_manifest.json` and checksum-validated artifacts
exist; until then it is **not run** and must be listed as planned. **H5 is locally testable only in the
CPU scope** — packed-storage survival (class 2 → class 3) and CPU-kernel survival (class **4-CPU**) —
and those results are **not** comparable to published GPU latency/throughput; its class 4-GPU and
class 5 halves require a validated cloud run with a real supported kernel. H5 is therefore reported
**partially supported / inconclusive at GPU scope**, never fully confirmed.

## 9. Exclusion rules (predeclared, automated)

A run is excluded **only** by these predeclared, automatic criteria, and every exclusion is logged
with the raw manifest:

1. Non-finite loss or activation (NaN/Inf) at any step.
2. Training divergence: dev loss exceeds the initial dev loss by a predeclared factor over a
   predeclared patience window.
3. Resource failure: OOM, or wall-clock exceeded the predeclared per-run cap.
4. Corrupted or schema-invalid manifest (per `artifacts/schemas/`).
5. Detected train/eval overlap (contamination) in the run's data pipeline.
6. Byte-accounting l1 error beyond a predeclared tolerance (the run's storage claim is untrustworthy).

**Not admissible exclusion reasons:** an unfavourable seed, a poor arm, a hypothesis not being
supported, or any post-hoc visual judgement. Excluded runs are reported in an exclusions table with
counts per arm so that differential attrition is visible.

## 10. Stopping rules

1. Confirmatory testing is **fixed-n**: stop after the predeclared seed count and the predeclared
   test reads. No optional stopping.
2. Training-time early stopping uses **dev** only, with identical patience and criteria across arms.
3. If an arm fails to train on ≥ 3 seeds at Tier 1, it is declared **infeasible at this tier** and
   reported (not deleted).
4. **Cloud substrate and cost.**
   (a) The **free-tier cloud substrate** (Colab / Kaggle) is assumed for Tier 1–2, subject to session
       and quota limits; runs are sized to fit within a session.
   (b) **Tier 3–5 and any paid instance require explicit user authorization and a stated maximum
       estimated cost before submission** (`AGENTS.md` §2b rule 8). No paid resource is started
       without it.
   (c) A **pre-empted or interrupted session** is resumed via the persisted remote run id, or recorded
       as **failed** with its logs; it is **never silently retried into a success**.
   (d) A run whose **artifact bundle fails checksum validation does not exist** — it is recorded as
       failed and excluded from the research record (`AGENTS.md` §2b rules 4–5).
5. The Pareto sweep terminates at the predeclared budget ladder; no budget may be added after seeing
   results.

## 11. Failure-mode analysis — what result would falsify each hypothesis

### 11.0 Traceability table (spec hypothesis → where tested → falsifier)

| Spec ID (canonical wording, abbreviated) | Tested in (section / row) | Falsifying result (see §11.1) |
|---|---|---|
| **H1** preparation can reduce factorization error while *increasing* rounding sensitivity | §8 rows 1–2 (`LOCAL-FIXTURE` Tier 0; `CLOUD-COLAB` Tier 1); §5 arms (a)/(b)/(c); §7.1 paired comparison | no regime where (i) factorization error improves while (ii) quantization sensitivity worsens/does not improve |
| **H2** proxy with activation covariance + quantization residual beats weight-space Frobenius error | §8 rows 4–5 (`LOCAL-FIXTURE` + `CLOUD-COLAB`); §7.2 (correlation study); §5 comparator = Frobenius error | proxy rank/linear correlation not higher than Frobenius error after Holm–Bonferroni |
| **H3** rounding-aware spectral regularizer improves quality at equal memory | §8 row 6 (`CLOUD-COLAB`); §5 arms (d)/(e) + plain-spectral control; §7.1 | no improvement at equal bytes, or improvement disappears without the rounding-grid term |
| **H4** layer-wise mixed rank+precision dominates uniform on part of the frontier | §8 row 7 (`CLOUD-COLAB`); §5 arms (d) vs (a)/(b) vs HAWQ-V2-style; §7.3 hypervolume | no predeclared budget where it dominates beyond CIs, or matched by the bit-only allocator |
| **H5** fake-quant gains need a packed kernel to become real gains | §8 rows 8–9 (`LOCAL-CPU-MEASUREMENT`: packed-storage survival class 2→3 and CPU-kernel survival class 4-CPU) + §8 row 14 (`CLOUD-GPU`, class 4-GPU/5, claimable only from a validated cloud run) | fake-quant gain inverts at packed storage or at the CPU kernel (falsified), **or** the GPU/service half is not measurable without a validated cloud run (inconclusive at GPU scope — never claimed confirmed) |
| Q2 (not a hypothesis) order of the two transforms | §8 row 3 (`CLOUD-COLAB`); §5 arms (b) vs (c) | n/a (mechanistic comparator) |

### 11.1 Failure modes

| Hypothesis | Falsifying result (predeclared) | Consequence / how we report |
|---|---|---|
| **H1 (factorization error vs. rounding sensitivity)** | No regime found where truncation reduces reconstruction error while simultaneously increasing rounding sensitivity at equal stored bytes (95% CI of the sensitivity difference includes 0, or the two move together), on ≥5 Tier-1 seeds (cloud). | H1 refuted at Tier 1: factorization error and rounding sensitivity are not dissociable at this scale. Report the null; the "double-edged" framing is dropped and H3/H4 no longer rely on it as a premise. |
| **H2 (proxy validity)** | The proxy's correlation with measured degradation is **not** higher than weight-space Frobenius error (bootstrap CIs overlap / the paired difference is not significant after Holm–Bonferroni). | Proxy refuted as an improvement over the predeclared comparator; fall back to the best comparator and re-scope contribution A (novelty-risk §1 verdict downgrade). |
| **H3 (rounding-aware spectral regularizer)** | The regularizer gives no improvement over unprepared factorization at equal bytes (CI includes 0), **or** the improvement vanishes when the rounding-grid term is replaced by a plain spectral penalty. | Regularizer refuted (either "no effect" or "not rounding-aware"); downgrade to a spectral-regularizer variant and report which term failed. |
| **H4 (mixed rank + mixed precision under a global budget)** | Proxy-allocated $(r_\ell,b_\ell)$ does **not** dominate uniform rank/bit at any predeclared budget beyond CIs, or is matched by the HAWQ-V2-style bit-only allocator at equal bytes. | Frontier-advantage claim refuted; report a null. "Joint allocation" remains previously-known (novelty-risk §3). |
| **H5 (fake quantization → real gains)** | Three-part, canonical: (a) **falsified** if the fake-quantization gain inverts at our **packed int4/int8 serialization** at byte parity (the gain was a fake-path artifact); (b) **falsified** if it inverts when the same artifact is executed by a **real CPU low-bit kernel** (class 4-CPU: ONNX Runtime `MatMulNBits`/`MatMulInteger` or torchao intx) against an fp32 CPU baseline in the same session — though a CPU-kernel failure is evidence about the CPU path, not about GPU kernels; (c) **inconclusive at GPU scope** because class 4-GPU and class 5 require a validated cloud run with a real supported kernel and **have not been measured**. | Report exactly which of (a)/(b)/(c) was tested. Never convert "not measured" into "no gain", and never convert a class 4-CPU result into a latency/throughput claim or a comparison with published GPU numbers. H5 may be reported at most as **partially supported / inconclusive at GPU scope**; it is confirmed only if the gain survives packed storage, the CPU kernel, **and** a validated cloud GPU run with a real supported kernel (Tier 2, M7). |
| **H2 cell on pretrained LM** | Forward-only proxy correlation on the ~125M model is null while Tier-0/1 correlation is positive, or vice versa. | Scale-dependence of the proxy: report both, state the proxy is only validated at the scales actually measured, and make no scale-transfer claim. |

Any hypothesis may be refuted. Refutation is a completed result, not a failure of the project.
An **inconclusive** hypothesis (locally the case for H5's kernel half) is reported as inconclusive;
it is never upgraded to confirmed and never downgraded to refuted by default. Likewise, a cell that is
**planned on the cloud but not yet executed with a validated manifest** is "not run", never "in progress
towards a positive result".

## 12. Reproducibility obligations

- Every run writes a manifest validated against `artifacts/schemas/`, recording commit SHA, dirty
  flag, seeds, dataset identifiers (with hashes where feasible), model/checkpoint identifiers,
  calibration-set hash, config, measurement classes, and byte accounting.
- **Every cloud run additionally writes `run_manifest.json`** with run id, resolved config, dependency
  lock information, git commit SHA, dataset versions/checksums, seeds, hardware metadata, start/end
  times, GPU-hours, status, metrics and artifact checksums (`AGENTS.md` §2b rule 3); artifacts are
  collected and **checksum-validated locally** before they enter the research record, and a run whose
  artifacts fail validation is recorded as failed (rule 5).
- **Notebooks are generated, not hand-edited:** produced from version-controlled experiment configs by
  a generator; a hand-edited notebook that cannot be regenerated is not evidence (`AGENTS.md` §2b
  rules 1–2). All scientific logic lives in importable `src/spectraquant/**` modules.
- **Credentials never touch the repo** — platform secrets / environment variables only (`AGENTS.md`
  §2b rule 7).
- Tier-0 fixture runs must be bit-for-bit reproducible on CPU given the pinned environment (`uv.lock`).
  Cloud runs are reproducible given the recorded lock information, hardware metadata and seeds.
- Proxy/allocator/regularizer hyperparameters are frozen before the confirmatory run; any change
  after freeze is an amendment.
- The exact commands for each confirmatory cell are recorded in `scripts/reproduce/` (owned by
  stream B) and referenced from `reports/`; cloud submission/collection is idempotent and resumable via
  the persisted remote run id (`AGENTS.md` §2b rule 6).

## 13. Freeze checklist (to be completed at freeze; not yet done)

- [ ] Tier-1 corpus subsample indices hashed and committed.
- [ ] Random seed list committed.
- [ ] Byte-budget ladder committed.
- [ ] Rank/group/bit grids committed.
- [ ] lm-evaluation-harness task list committed.
- [ ] Calibration size and sampling rule committed.
- [ ] Comparators (weight magnitude, activation magnitude, Hessian trace, HAWQ-V2-style, random-probe,
      **weight-space Frobenius error**) implemented and unit-checked.
- [ ] **Class 4-CPU fixtures** committed: int4 (`MatMulNBits`) and int8 (`MatMulInteger`) containers
      **serialized by us**, the CPU-kernel runner, and the fp32 same-session baseline; header/claim
      template records backend, op, container, thread count and CPU model.
- [ ] Tier-0 model configs committed; Tier-1+ configs committed for cloud execution.
- [ ] This document marked FROZEN with date and commit SHA by the orchestrator (**freeze = M1**).
- [ ] Traceability table (§11.0) reviewed: every spec H1–H5 still maps to a test row and a falsifier.
- [ ] **Cloud substrate ready**: notebook generator wired to the versioned configs (no hand-edited
      notebooks as evidence); `run_manifest.json` schema committed under `artifacts/schemas/`;
      local collection + checksum validation path (`spectraquant cloud collect`) exercised once end-to-end.
- [ ] **Cloud credentials policy** confirmed (platform secrets/env vars only, never repo/logs) and the
      **cost ceiling** for any paid Tier-3/4/5 instance stated before submission.
