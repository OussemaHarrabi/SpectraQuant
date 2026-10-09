# SpectraQuant — Pre-registration

Status: **FROZEN 2026-10-09 at milestone M1** by the orchestrator (freeze commit SHA recorded in
`docs/coordination/status.md` §5). Once frozen, this document is modified **only** through
`preregistration-amendments.md` (append-only, timestamped, reason-stated); amendments A-0001..A-0006
precede or accompany this freeze. Tier-0/1 work that informed *method design* before the freeze is
labelled **exploratory**; no confirmatory claim may be sourced from it.

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
   arm is predeclared as **exactly 2** — one confirmatory read plus **at most one** re-read after a
   recorded bug fix — and logged. **No tuning on test labels; no repeated test peeking to steer
   method decisions** (`AGENTS.md` §4.6).
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
  the cloud substrate** on the Tier-1 corpus. The confirmatory configuration is **pinned here**, not
  chosen after seeing results: **`L_b = 8` transformer blocks, width `d = 256`, sequence length 256,
  vocab 50,257 (GPT-2 BPE), ≈32 M parameters** (approximate — interpolated between the adversarial
  review's measured `d=256, L=4` (28.9 M) and `d=256, L=6` (30.5 M) configurations), trained for a
  **fixed 5 epochs**. `L_b = 8` is the smallest block count inside the review's measured `L = 8–12`
  power band (`docs/results/verification/raw/probe_stats.out`); the nearest measured local CPU
  throughput anchors are 1,879 tok/s (`d=256, L=4`) and 1,506 tok/s (`d=256, L=6`, seq 512)
  ⇒ ≈0.35–0.44 h per WikiText-2 epoch locally, hence cheap on the cloud substrate. This is the
  confirmatory vehicle for H1–H5.
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
  (group sizes predeclared, e.g. 32/64/128). **Scope of the packed-storage/CPU-kernel chain
  (H5): the H5 chain applies to $b \in \{4, 8\}$ only.** The frozen packers are
  `spectraquant.quantization.packing.pack_int4` / `pack_int8`, and the permitted class-4-CPU kernels
  are int4 (`MatMulNBits` weight-only) and int8 (`MatMulInteger`). There is **no `pack_int2`/`pack_int3`**
  and **no permitted kernel executes 2/3-bit weights**, so the 2/3-bit arms are evaluated under **fake
  quantization (class 2) only** and are excluded from every H5 statement — this is exactly where
  fake-quant gains are largest, and the exclusion is stated rather than hidden.
- **Ranks $r$:** {2, 4, 8, 16, 32} (scaled to layer dimensions; final list pinned at freeze).
- **Byte budgets** for the Pareto sweep: a predeclared geometric ladder of stored-byte targets.
- **Arms:** (a) uniform-$b$ full-rank; (b) uniform-$r$ then uniform-$b$ (rank-then-quantize);
  (c) quantize-then-low-rank-residual (LoftQ-style); (d) proxy-allocated $(r_\ell,b_\ell)$;
  (e) proxy-allocated with the regularizer (**H3 arm**); (f) reference: fp16 uncompressed.
- **Ordering arms (Q2 comparator, *not* a hypothesis):** (b) vs (c) at identical $r$ and $b$ and
  identical stored bytes. Reported as a mechanistic comparison feeding H1 and H4.
- **Comparators:** weight-space Frobenius error is the **predeclared H2 comparator**; **HAWQ-V2-style
  average-Hessian-eigenvalue Pareto allocation** is the **H4 bit-only comparator** at the same byte budget.
- **H5 arm (packed-storage and CPU-kernel survival):** for each arm that shows a fake-quantization
  quality gain, (i) re-measure quality after our **own packed serialization** (int4/int8) with CPU
  dequantization **at measured stored-byte parity**, then (ii) re-measure it when the same serialized
  artifact is executed by a **real CPU low-bit kernel** (class 4-CPU) against an fp32 CPU baseline in
  the same session. Report which stage the gain survives; make no GPU or latency claim.
  - **Frozen serializer invocation (B12).** Exactly **one** serialization entry point is used for the
    whole H5 chain, frozen at M1 and identified in every manifest by the hash
    `compare.serializer_build`. Container overhead is **invocation-dependent** — measured on the
    project's own fixture (`memory-accounting.md` §7.5: a 1,280 B payload yields **221–363 B** of
    container overhead depending only on how the ONNX export is invoked, so *nominal 4.0 bits/param
    becomes **5.86–6.42 measured***). The build hash, not the nominal bit width, defines "the same
    container"; two arms are comparable only if their `serializer_build` values are identical.
  - **Predeclared byte-parity tolerance.** Two arms are "at measured stored-byte parity" iff
    `|bytes_a − bytes_b| / bytes_b ≤ **0.5 %**`; a comparison whose relative byte difference exceeds
    this is **not** an equal-memory comparison and is rejected by the equal-memory gate (§7.1;
    `assert_equal_memory`). The class-1 analytical estimate used inside the allocator MUST include a
    per-format `overhead_fn` measured once at freeze, so the constrained and reported quantities are
    the same number.

## 6. Seed plan

- **Master seed** fixed (recorded in every manifest). Derived streams: init, data order, calibration
  sampling, probe sampling — separated by documented derivation so that changing one does not
  perturb the others.
- **Tier 0 (local fixtures):** ≥ **5 seeds** for every reported cell; exact enumeration where feasible.
- **Tier 1 (cloud, trainable comparisons):** the seed floor is ≥ **5 seeds** and applies to the
  **pinned confirmatory config** (§4: `L_b=8`, `d=256`, 5 epochs). "Cheap" is **defined
  operationally**: a Tier-1 cell is cheap iff its full arm × seed matrix fits the cloud budget line
  of §10.4(a). Where it does not fit, the **arm count is reduced first** (drop exploratory arms, then
  the Q2 comparator's secondary cells), **not the seed count**; only if the ≥5-seed floor still does
  not fit is the floor lowered to **≥3 seeds**, and that reduction MUST be justified in a manifest
  note against the budget line. Seed counts are never reduced per-arm (no differential attrition,
  §9). All Tier-1 training runs execute on the cloud substrate (Colab/Kaggle), never locally; a
  session pre-emption is resumed via the persisted remote run id, or recorded as failed (see §10).
- **Tier 2+ (cloud GPU):** ≥ **3 seeds** for trainable arms.
- A result may not be reported from fewer seeds than the floor. Seeds are not dropped because their
  outcome is unfavourable (see §9 exclusion).

## 7. Statistical analysis plan

### 7.0 Predeclared units, granularity, and constants (binding)

- **Layer granularity.** A "layer" in this plan is a **linear module** — the four linear maps per
  transformer block (attention QKV projection, attention output projection, MLP up-projection, MLP
  down-projection). At the pinned Tier-1 size (`L_b = 8`, §4) a model therefore has **n = 32 linear
  modules**. This matches the proxy's native unit (`per_layer: dict[str, float]`, keyed by module
  name) and resolves the block-vs-module ambiguity flagged during review.
- **Primary resampling unit for H2: the MODEL.** One correlation is computed per **trained model**
  (never per layer); models are the exchangeable unit and layers/modules are a **within-model
  nuisance**. Justification is measured, not asserted: modules are *heterogeneous strata* — in probe
  E3 the per-module proxy spans **0.919 (`proj`) to 178.2 (`fc1`) at the same depth** (a 194×
  spread driven by module *type*), the per-layer bootstrap is **degenerate at small n** (width 0.000
  at L=4 under a distinctness guard), and the procedure had **no tie/duplicate rule** (probe E1,
  `raw/probe_stats.out`). Seeds *are* exchangeable.
- **Arm-vs-arm unit (§7.1): the seed** (and, at Tier 0, the matrix).
- **Byte tolerance (equal-memory gate):** `|bytes_a − bytes_b| / bytes_b ≤ 0.5 %` (§5), enforced by
  `spectraquant.reporting.comparability.assert_equal_memory`. A comparison outside tolerance is **not
  an equal-memory comparison** and is rejected.
- **Pinned constants** (all predeclared; mirrored in §13): test reads **2** (§3.3); frontier
  equivalence margin **5 %** relative byte saving; matched-quality tolerance **0.01 nats/token**;
  exclusion constants as in §9.

1. **Paired arm comparisons (§7.1).** Arm A vs arm B **at measured stored-byte parity** (`rel_diff ≤ 0.5 %`,
   §5): paired differences over matched **seeds** (matched matrices at Tier 0). Report the mean
   difference, a **paired bootstrap** 95 % CI (≥ 10,000 resamples, percentile + BCa), and **Cliff's
   delta**. Paired units are seeds, not layers.
2. **Proxy validity (S2, tests H2) — model-level (§7.2).** Per **trained model** `m`, compute Spearman ρ and
   Pearson r between each proxy's scores and the measured degradation over that model's `n=32` linear
   modules, **and separately against the end-to-end (joint) damage**; transform to Fisher
   `z = atanh(r)`. The **H2 statistic** is the **paired difference in Fisher-z between the candidate
   proxy and the predeclared comparator across models**, combined by a **Fisher-z random-effects
   model** (DerSimonian–Laird τ²; models = unit; seeds exchangeable). A model's within-model (layer)
   uncertainty enters as `var(z_m) = 1/(n_eff − 3)` with **`n_eff` reported** per model (estimated by
   a within-model bootstrap over the 32 modules: resample with replacement, require **≥ ⌈n/2⌉
   distinct indices**, else redraw). Layers/modules never appear as resampling units in the primary
   test. **Predeclared primary comparator: weight-space Frobenius error** (the exact wording of H2).
   Secondary comparators — weight magnitude, activation magnitude, HAWQ-V2 average-Hessian-eigenvalue
   sensitivity, and (if implementable) the random-probe estimator of arXiv:2609.33923 — are
   **exploratory**.
3. **Frontier comparison (H4) (§7.3).** Pareto **hypervolume** difference with a paired bootstrap CI over
   seeds. **Dominance** is declared only if (i) the hypervolume-difference CI excludes 0, **and**
   (ii) the arm's stored bytes are **≥ 5 % lower** (the **predeclared equivalence margin**) than the
   baseline's at **matched quality**, where matched quality means `|ΔNLL| ≤ 0.01 nats/token` on the
   held-out set (the **predeclared matched-quality tolerance**; at NLL ≈ 4.5 this is ≈1 % perplexity),
   **and** (iii) the byte saving's own 95 % CI excludes 0. A byte saving below the 5 % margin is
   reported **not distinguishable at this tolerance**, never as a win.
4. **Confirmatory families and multiplicity (§7.4).** The confirmatory tests are enumerated **in advance**
   with their family sizes:

   | Family | Tests (`m`) | Correction |
   |---|---|---|
   | **F1 — H2 primary contrast** | 1 (candidate proxy vs the predeclared Frobenius comparator, model-level) | **none** — with `m = 1` Holm–Bonferroni is the identity and is **deleted**; reported at α = 0.05 two-sided |
   | **F2 — arm-level hypotheses** | 3 (**H1**, **H3**, **H4**) | **Holm–Bonferroni** |
   | **F3 — H5 chain** | 2 (**H5a** packed-storage survival; **H5b** class-4-CPU kernel survival) | **Holm–Bonferroni** |

   Each hypothesis contributes **exactly one** confirmatory test (its primary cell, §8); no post-hoc
   row may be added to a family. Exploratory sweeps (not in F1–F3) use **Benjamini–Hochberg** and
   cannot support a primary claim.
5. **No optional stopping (§7.6).** Confirmatory tests use the fixed seed count and the fixed **2** test
   reads per arm (§3.3). No peeking to decide whether to continue.
6. **Reporting (§7.7).** Every reported number carries a measurement class label (`AGENTS.md` §5) and its
   seed count; negative and null results are reported (`AGENTS.md` §4.10). A null whose CI extends
   beyond the MDE of §7.5 is reported **inconclusive**, not as a refutation of a smaller effect.

### 7.5 Power / minimum detectable effect (MDE) — predeclared, from the review's measured numbers

This is a **design** statement from the adversarial review's probes, not a result.

- **Layer-level power (rejected as the unit).** For the paired layer-bootstrap test at true Δρ = 0.17,
  probe E1 (`raw/probe_stats.out`) measures power **0.036** at L=8 units and **0.082** at L=12 (the
  band containing the pinned `L_b = 8`), rising only to **0.446** at L=48; the L=4 layer bootstrap is
  degenerate (CI width 0.000) and the procedure has no tie rule. Layer-level power is therefore far
  below the conventional 0.8 across the whole Tier-1 size range — this, not compute, is why the model
  is the unit.
- **Model-level MDE (declared unit).** With the model as unit, per-model Fisher-z from `n = 32`
  linear modules, and S = 5 seeds, the paired Fisher-z difference has MDE (80 % power, α = 0.05
  two-sided) of **MDE(z) ≈ 0.33**, i.e. **Δρ ≈ 0.12 at ρ ≈ 0.8**, if the within-model modules are
  treated as ≈independent (optimistic); and **MDE(z) ≈ 0.79, Δρ ≈ 0.29** if the 8 *blocks* are taken
  as the effective sample (conservative). The truth lies between; the **achieved `n_eff` is reported**
  per model and H2's verdict is stated against the achieved MDE. Derivation:
  `MDE(z) = (z_{0.975} + z_{0.80}) · sqrt(2 / (n_eff − 3)) / sqrt(S)` (command run by the research
  stream on 2026-10-08; output recorded in amendment A-0005).
- **Consequence.** Δρ = 0.17 is detectable at S = 5 **only at the optimistic end** of the `n_eff`
  range; the honest expectation is that H2's Tier-1 CI is wide. This is **predeclared**, not
  discovered after the fact.

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
| **Packed-storage byte measurement:** serialize our own int4/int8 containers through the **frozen serializer invocation** and measure stored bytes | **H5** | 0/1 | `LOCAL-CPU-MEASUREMENT` | M4 | **runnable** (bytes are class 1 → class 3; no quality claim) |
| **Packed-storage quality re-measurement:** re-measure P1 quality (class 2) at the predeclared byte parity (`rel_diff ≤ 0.5 %`) after the packed round-trip | **H5** | 1 | `CLOUD-COLAB` | M4 | **planned (cloud)** — quality re-evaluation is not local work (`AGENTS.md` §2.3) |
| **CPU-kernel survival:** our own serialized **int4/int8** container ($b \in \{4,8\}$ only, §5) executed by a real **CPU** low-bit kernel (ONNX Runtime `MatMulNBits` int4 weight-only / `MatMulInteger` int8, or torchao intx) vs. an fp32 CPU baseline in the same session | **H5** | 0/1 | `LOCAL-CPU-MEASUREMENT` | M4 | **runnable** (class 4-CPU; requires the pinned `onnx`/`onnxruntime`/torchao extra — B14/§13; CPU scope only — **not** comparable to published GPU numbers; **no** latency/throughput claim) |
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

A run is excluded **only** by these predeclared, automatic criteria (all constants are **numbers**, not
"predeclared factors"), and every exclusion is logged with the raw manifest:

1. Non-finite loss or activation (NaN/Inf) at any step.
2. **Training divergence:** dev NLL exceeds the initial dev NLL by a factor of **2.0** (equivalently
   ΔNLL ≥ +0.693 nats/token) sustained over a **patience window of 500 consecutive optimizer steps**.
3. **Resource failure:** OOM, or wall-clock exceeded the predeclared per-run cap — **Tier 0 ≤ 1 h**;
   **Tier-1 arm-seed ≤ 4 h** (cloud); **Tier-2/3 model-seed ≤ 24 h**; **Tier-4/5 model-seed ≤ 48 h**
   (paid tiers only, with prior authorization).
4. Corrupted or schema-invalid manifest (per `artifacts/schemas/`).
5. Detected train/eval overlap (contamination) in the run's data pipeline.
6. **Byte-accounting error:** `|class-3 measured − (class-1 analytical + overhead_fn)| / class-3 >
   **0.5 %**` — the same tolerance as the byte-parity/equal-memory gate (§5, §7.0); the run's storage
   claim is untrustworthy and the run is excluded.

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
   (a) **Cloud budget line.** The **free-tier cloud substrate** (Colab / Kaggle) is assumed for
       Tier 1–2. Assumptions: **Kaggle weekly GPU quota ≈ 30 GPU-h** and Colab free sessions
       **≤ 12 h** each (conservative; actual quotas vary).
       **Basis for cost-bounding.** The review measured this workstation's CPU at **1,158–2,375
       tok/s** for tiny LMs (`docs/results/verification/raw/probe_throughput.out`: `d=128,L=4` 2,375;
       `d=256,L=6` 1,506; `d=384,L=6` 1,158 tok/s; 0.28–0.58 h per WikiText-2 epoch). Scaling to a
       T4-class cloud GPU at a conservative **20–50×** gives **≈0.9–9 GPU-h per Tier-1 arm-seed** at
       5 epochs, so the full 6-arm × 5-seed matrix is **≈27–270 GPU-h** — up to ~9× the
       **30 GPU-h/week** free-tier ceiling. The project's own cost model
       (`docs/protocols/eval-protocol.md` §6.4) gives the confirmatory S4 suite alone as
       **0.4–18.3 GPU-h per model per seed** (× ≥3 seeds).
       **Ceiling and maximum arm×seed count.** The predeclared Tier-1 ceiling is **30 GPU-h per
       calendar week**; at the optimistic 0.9 GPU-h per arm-seed this permits **≤ 33 arm-seed
       units/week**, at the pessimistic 9 GPU-h it permits **≥ 3**. Until the first cloud run measures
       the real per-arm cost, the confirmatory matrix is sized to the **pessimistic** end.
       **When the ceiling binds** (predeclared ladder, applied in order): (1) drop exploratory arms;
       (2) drop the Q2 comparator's secondary cells; (3) spread across weeks using the resumable-slice
       design (`AGENTS.md` §2b rule 6); (4) if the pinned confirmatory cells still do not fit, the
       affected H3/H4 cells are reported **not run** — never imputed and never silently reseeded.
       **Paid instances** (any tier) require the user's explicit authorization and a stated maximum
       estimated cost before submission (`AGENTS.md` §2b rule 8), and may raise the ceiling only by
       that recorded authorization.
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
| **H2** proxy with activation covariance + quantization residual beats weight-space Frobenius error | §8 rows 4–5 (`LOCAL-FIXTURE` + `CLOUD-COLAB`); §7.0 (model unit, per-module granularity) + §7.2 (model-level Fisher-z random-effects correlation study) + §7.5 (MDE); §5 comparator = Frobenius error | proxy's model-level paired Fisher-z difference not higher than Frobenius error (random-effects CI includes 0 and the CI is narrower than the §7.5 MDE); a null wider than the MDE is **inconclusive**, not a refutation |
| **H3** rounding-aware spectral regularizer improves quality at equal memory | §8 row 6 (`CLOUD-COLAB`); §5 arms (d)/(e) + plain-spectral control; §7.1 | no improvement at equal bytes, or improvement disappears without the rounding-grid term |
| **H4** layer-wise mixed rank+precision dominates uniform on part of the frontier | §8 row 7 (`CLOUD-COLAB`); §5 arms (d) vs (a)/(b) vs HAWQ-V2-style; §7.3 hypervolume | no predeclared budget where it dominates beyond CIs, or matched by the bit-only allocator |
| **H5** fake-quant gains need a packed kernel to become real gains | §8 rows **8–10** (`LOCAL-CPU-MEASUREMENT`: packed-storage **byte** measurement class 1→3; `CLOUD-COLAB`: packed-storage **quality** re-measurement class 2; CPU-kernel survival class 4-CPU) + §8 row **15** (`CLOUD-GPU`, class 4-GPU/5, claimable only from a validated cloud run) | fake-quant gain inverts at packed storage or at the CPU kernel (falsified), **or** the GPU/service half is not measurable without a validated cloud run (inconclusive at GPU scope — never claimed confirmed) |
| Q2 (not a hypothesis) order of the two transforms | §8 row 3 (`CLOUD-COLAB`); §5 arms (b) vs (c) | n/a (mechanistic comparator) |

### 11.1 Failure modes

| Hypothesis | Falsifying result (predeclared) | Consequence / how we report |
|---|---|---|
| **H1 (factorization error vs. rounding sensitivity)** | No regime found where truncation reduces reconstruction error while simultaneously increasing rounding sensitivity at equal stored bytes (95% CI of the sensitivity difference includes 0, or the two move together), on ≥5 Tier-1 seeds (cloud). | H1 refuted at Tier 1: factorization error and rounding sensitivity are not dissociable at this scale. Report the null; the "double-edged" framing is dropped and H3/H4 no longer rely on it as a premise. |
| **H2 (proxy validity)** | The proxy's **model-level** correlation with measured degradation is **not** higher than weight-space Frobenius error — the Fisher-z random-effects CI on the paired difference includes 0 and is narrower than the predeclared §7.5 MDE (a CI wider than the MDE is reported **inconclusive**, not refuting). | Proxy refuted as an improvement over the predeclared comparator; fall back to the best comparator and re-scope contribution A (novelty-risk §1 verdict downgrade). |
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

## 13. Freeze checklist — completed 2026-10-09

Legend: **[x]** satisfied and machine-checked; **[~]** satisfied by a frozen rule with the remaining
artefact produced by the run (reason given); **[ ]** deferred, with the milestone that owns it.

- [x] **Random seed list committed** — `configs/tier1/smollm2_135m.yaml` §`seeds` (master 20261008,
  four derived streams, reported `[0,1,2,3,4]`, floor 5); `configs/tier2/tinyllama_1_1b.yaml`
  (reported `[0,1,2]`, floor 3); validated by `tests/unit/test_experiment_plan.py`.
- [x] **Byte-budget ladder committed** — geometric stored-byte targets at 10/15/20/25/33 % of each
  model's fp16 checkpoint, in the same plan files (class-3 targets, not estimates).
- [x] **Rank/group/bit grids committed** — `bits [2,3,4,8]`, `ranks [2,4,8,16,32]`,
  `group_sizes [32,64,128]`, asserted against the predeclared values by the plan tests.
- [x] **lm-evaluation-harness task list committed** — commit `ddd67220430a2470529f25fd5c05a576ca1057a0`
  (`v0.4.13`) and tasks `[hellaswag, arc_easy, arc_challenge, piqa, winogrande, boolq]` in every plan
  and in `docs/protocols/eval-protocol.md`; the plan tests assert the match.
- [x] **Calibration size and sampling rule committed** — `eval-protocol.md` §3.2: 256 × 2048 tokens,
  `sha256(text) mod 1000 < 1` over the C4 `en` train stream, seed 20261008, slice hash recorded.
- [x] **Candidate proxy declared (A-0007)** — `gain_aware_composed`; `combined` is an ablation with
  untuned weights, not the candidate. The M4 local-fixture half is complete
  (`docs/results/proxy-validation-report.md`: 15/15 aggregate contrasts favour the candidate over
  every predeclared comparator); the cloud Tier-1 cells remain **NOT RUN**.
- [x] **Comparator set implemented and unit-checked** — `weight_magnitude`, `activation_magnitude`,
  `hessian_diag` and `weight_frobenius` in `src/spectraquant/proxies/variants.py`, each tested
  (`tests/unit/test_proxy_variants.py`); the unit-defining variants match float64 toy ground truth at
  `rtol=1e-9`.
- [x] **Environment pinned for the class-4-CPU path (B14)** — `onnx` extra (`onnx~=1.23.2`,
  `onnxruntime~=1.30.0`, `onnx-ir~=1.0.0`, `torchao~=0.18.0`) in `pyproject.toml` + `uv.lock`.
- [x] **Serializer frozen (B12)** — our own container is `spectraquant-sqpack-v1`
  (`SQ_CONTAINER_FORMAT_ID`) with measured overhead (12.5–37.5 % of payload for per-group int4);
  `accounted_bytes == measured` with zero residual across 16 configurations.
- [x] **Statistical plan constants committed (B7–B10)** — one primary unit (the model, §7.0); F1–F3
  family table with `m` (§7.4); equivalence margin 5 %, matched-quality tolerance 0.01 nats/token,
  byte-parity tolerance 0.5 %, test reads 2; the four §9 exclusion constants; MDE statement (§7.5).
- [x] **Cloud budget line recorded (B15)** — plan-level ceilings (Tier-1 6 GPU-h, Tier-2 12 GPU-h,
  reproduction 8 GPU-h), free-tier-only, zero paid authorization, and the incompleteness rule.
- [x] **Tier-0 and Tier-1+ configs committed** — `configs/experiment/smoke.yaml` (Tier-0 fixture) and
  the three cloud plans under `configs/{tier1,tier2,repro}/`, all schema-validated.
- [x] **Traceability table reviewed (§11.0)** — every spec H1–H5 maps to a test row and a falsifier.
- [x] **Substrate decision recorded (A-0004)** — cloud notebook substrate for Tiers 1–5; locally
  available classes 1, 2, 3, 4-CPU; no "classes 4–5 unavailable" phrasing remains.
- [x] **Cloud substrate ready** — notebook generator wired to versioned configs
  (`spectraquant cloud notebook`), `run_manifest.json` schema committed under `artifacts/schemas/`,
  and `spectraquant cloud collect` exercised end-to-end (orchestrator: intact bundle validated,
  tampered artifact and commit mismatch rejected).
- [x] **This document marked FROZEN** — see the header; the freeze commit SHA is recorded in
  `docs/coordination/status.md` §5.
- [~] **Tier-1 corpus subsample hash** — the selection rule and seed are frozen; the hash is produced
  by the run and recorded in `run_manifest.json`, then checked at collection. Locally hashing would
  require downloading the corpus, which the substrate policy assigns to the run (A-0006).
- [x] **Class-4-CPU ONNX fixture** (int4/int8 container written by us, its runner, the same-session
  fp32 baseline) — implemented and measured: `src/spectraquant/quantization/onnx_export.py`,
  `src/spectraquant/benchmarking/kernel_cpu.py`, artifact `artifacts/sample-results/class4cpu/`
  (int4 1662 B, int8 2727 B, int4 relative error 0.0771 with the weight distribution stated). The H5
  CPU-kernel cell is runnable; the result does **not** by itself confirm H5 (A-0007).
- [ ] **Random-probe comparator** (arXiv:2609.33923) — optional addition, not part of the predeclared
  comparator set; deferred (A-0006).
