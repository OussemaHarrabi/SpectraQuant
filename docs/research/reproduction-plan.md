# Milestone 3 reproduction plan (predeclared)

**Status: PREDECLARED — nothing in this plan has been run.** No reproduction result exists yet.
Every number in this file is either (a) quoted from a published paper with its table/figure reference,
(b) measured on this workstation as an *environment* fact (CPU throughput), or (c) an explicitly
labelled analytical estimate (measurement class 1, AGENTS.md §5). Execution is split by substrate
(`AGENTS.md` §2b): training and evaluation are **CLOUD-RUN** on generated notebooks; fixtures,
validation, collection, analysis and the class-4-CPU kernel measurement are local. Milestone 3 starts
only after the entry conditions in §10.1 and closes only at the M3 gate in §10.3.

Owner: agent `ReproPlan` (planning stream). Companion documents:
`docs/research/upstream-notes.md` (code-level evidence, pinned SHAs, licence reads, blocker file:line),
`references/method-cards/*.md` (per-method cards), `docs/research/environment.md` (hardware audit),
`docs/coordination/design-m2-interfaces.md` (frozen interfaces this plan consumes),
`docs/coordination/design-cloud-adapter.md` (frozen cloud execution contract this plan is written
against).

---

## 1. Objective and framing

Milestone 3 asks a deliberately narrow question: **can the two published central trends that this
project builds on be re-verified at reduced scale, on a free-tier cloud GPU substrate, with the failure
modes stated in advance?** It is not an attempt to reproduce the published *numbers*: the papers use
7B–70B models on A100s, and neither is reachable in the free-tier envelope (§8). Execution is
**cloud-only** for everything that trains or evaluates a model (`AGENTS.md` §2b): the workstation
orchestrates, validates, generates notebooks, collects and analyses; it does not train.

The two trends to be reproduced are:

| ID | Published trend | Source (exact reference) |
|---|---|---|
| **T1 (LR-QAT)** | At equal bit width, **PTQ < LR-QAT ≤ full-model QAT**: quantization-aware training with low-rank auxiliary weights recovers most of the accuracy that post-training quantization destroys, and merging the adapters into the integer weights costs nothing. | LR-QAT, arXiv:2406.06385v3, Table 3 (LLaMA-2 7B, W4 pc: RTN-equivalent full-QAT 5.77 / LR-QAT 5.66 vs FP16 5.47) and Table 4 (W4 pc RTN 6.14 vs LR-QAT 5.66); merge claim §3 + Table 1 |
| **T2 (LoftQ)** | At equal bit width and equal rank, **LoftQ-initialised LoRA beats standard (Kaiming-A / zero-B) LoRA initialisation** after fine-tuning, and the advantage grows as the bit width falls. | LoftQ, arXiv:2310.08659v4, Table 5 (LLaMA-2 7B/13B: at 2-bit QLoRA does not converge, LoftQ reaches 7.85/7.69 ppl) and Tables 1–2 (2-bit DeBERTaV3: MNLI-m 79.9 → 88.0); mechanism Figure 2 |

Both trends are **ordering/relative** claims, not absolute-value claims. This is the only form in which
they can be tested at a different scale, and it is the form this project's own research question
depends on (does low-rank preparation help or hurt subsequent quantization?).

Two support targets are reproduced **exactly** (deterministic, no training, no noise):

| ID | Invariant | Why it is in scope |
|---|---|---|
| **T1-a** | Merge identity: for LR-QAT's `Ŵ = s·clip(⌊Φ₀ + (α/r)·AB⌉)`, the *merged* integer weight matrix produces the same outputs as the un-merged adapter forward, exactly. | It is the paper's "no inference overhead" claim (§3, Table 1) and the only part of T1 that is exactly checkable without a GPU. |
| **T2-a** | Initialisation residual dominance: the alternating projection of LoftQ strictly reduces `‖W − (Q + BA)‖_F` relative to both the standard initialisation and the T=1 truncation. | It is the mechanism the paper's claim rests on (Eq. 6, Algorithm 1, Figure 2) and it is deterministic. |

---

## 2. Pinned upstream baselines (summary; full evidence in `upstream-notes.md`)

| Repo | Pinned commit | Licence (file read) | Role in Milestone 3 |
|---|---|---|---|
| `Qualcomm-AI-research/LR-QAT` | `8795afe054cf951b714299e01083a1b354721829` | Qualcomm BSD-3-style, **no patent grant** | **Specification reference only.** Reimplemented from the paper's equations; no code copied. |
| `yxli2123/LoftQ` | `ae33fd4fd05fd4ba146555cd77c13d307eb4e9b3` | MIT | Specification reference. The LLM path delegates to PEFT; the in-repo GLUE path is a second, CPU-readable reference. |
| `HanGuo97/lq-lora` | `c2424b3adc27197815da1ac9e1304565168d824d` | MIT | Context only (mixed-precision allocation + Fisher-weighted SVD). Not part of T1/T2 execution. |
| `pytorch/ao` | `cff77b46ef85ba8472b4a286598073d6c01e18e5` (`version.txt` 0.19.0-dev; released **0.18.0**) | BSD-3-Clause | **Dependency** (pure-Python wheel `torchao-0.18.0-py3-none-any.whl`), used for the full-model-QAT reference arm only. |

Clones live **outside the working tree** at `%TEMP%/sq-upstream/` (see `upstream-notes.md` §0); the
repository contains no third-party source. Confirmed by `git status --short` run **after** all four
clones existed: no `LR-QAT/`, `LoftQ/`, `lq-lora/` or `ao/` path appears in the repository
(`upstream-notes.md` §0.1). No upstream code is copied, vendored or imported from a local clone; the
only third-party code that may enter the experiment path is a **pinned published wheel**
(`torchao==0.18.0`, `peft`, `onnxruntime`, `bitsandbytes`, `transformers`) declared in `uv.lock`.

---

## 3. The reproduction designs

Common substrate (frozen before running):

| Element | Choice | Justification |
|---|---|---|
| Model (pilot, Tier 1) | `HuggingFaceTB/SmolLM2-135M` (`LlamaForCausalLM`, 30 layers, d=576, FFN 1536, GQA 9/3 heads, vocab 49 152, tied embeddings, 134.5 M params, `max_position_embeddings` 8192; revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`). Licence **Apache-2.0** (verified via the HF API, §11). | LLaMA architecture, so the paper's Llama-specific quant setup (linear layers only; embedding/head/RMSNorm kept in FP) transfers; pretrained, so PTQ damage is measurable; it is the cheap Tier-1 pilot that proves the harness end-to-end before GPU-hours are spent. Same architecture as the project's registered Tier-1 workhorse (`HuggingFaceTB/SmolLM2-360M`, `docs/research/model-dataset-licenses.md` M5). |
| Model (primary, Tier 2) | `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` — the **upstream LR-QAT README's own smoke model**, already registered in `docs/research/model-dataset-licenses.md` row M1 (Apache-2.0, revision `59f6f375b26bde864a6ca194a9a3044570490064`). | This is the configuration AGENTS.md §6 names for the reference-reproduction class ("TinyLlama-1.1B-class + WikiText-2, FP16/PTQ/QLoRA/LoftQ/LR-QAT … ≥3 seeds"), and it is 6× closer to the papers' LLaMA family than any 135 M model. It becomes affordable only because M3.2 is a **cloud** workload (§2b). |
| Execution substrate | **All training and evaluation (R1, R2, hyperparameter selection) is CLOUD-RUN** on the notebook substrate of `AGENTS.md` §2b / `docs/coordination/design-cloud-adapter.md`. Local work is limited to fixtures, validation, notebook generation, collection, analysis and the class-4-CPU measurement. | `AGENTS.md` §2b and `docs/research/environment.md` §5 (superseded): the local machine may not run research training, QAT or large-scale inference. Every cloud run must emit a checksum-validated `run_manifest.json`. |
| Token budget per arm | **1000 optimizer steps × 512 tokens = 512 k tokens** (≈25 % of one WikiText-2 train epoch, assumed ≈2.1 M tokens ±10 % — to be measured at M3.0). | Sized from the cost model in §4; large enough for a QAT/LoRA effect to move, small enough to fit the free-tier GPU envelope with 5 seeds at the pilot size and 3 seeds at the primary size. |
| Sequence length | 512 (train), 1024 (perplexity evaluation, non-overlapping windows) | The paper evaluates WikiText-2 at 2048 (LR-QAT §4). 1024 is a declared deviation (§7) that halves evaluation cost; perplexity is only ever compared **within** this project, never against the paper's numbers. |
| Quantization | Our own `spectraquant.quantization.fake_quantize` (measurement class 2, fake quantization in float). LR-QAT arms: **symmetric int-N, group size 128** (`QuantSpec(bits=4, granularity="per_group", group_size=128, symmetric=True)`), matching the paper's W4 g128 setting. LoftQ arms: **NF-style codebook, block 64** (the paper's NF2/NF4), plus a uniform-int2 control. | Class 2 is the quality-measurement class; class 3 (packed storage) is measured separately from **our own** int4/int8 ONNX containers via `accounted_bytes`, and class 4-CPU timing of those same containers is permitted for our own artifacts (`environment.md` §1 correction, `backend-capability.md` §4). The three classes are never merged into one claim (AGENTS.md §5). |
| Factors | `W ≈ B @ A`, `A: (rank, in_features)`, `B: (out_features, rank)` (frozen in `design-m2-interfaces.md` §0.2) | Note this is the transpose of LoftQ's `L/R` naming; conversions are explicit in the method card. |
| Seeds | **{0, 1, 2, 3, 4}** for the Tier-1 pilot (matching LR-QAT Table 3's "5 runs" and LoftQ Table 5's "median over five random seeds"); **{0, 1, 2}** for the Tier-2 primary runs (AGENTS.md §6 requires ≥3 seeds for Tier 2). Seed controls python/numpy/torch RNG, data-order shuffling, LoRA init, and the calibration sample used for range estimation; the seed list is part of `RunSpec.seeds` and is recorded in every manifest. | Same n as the published protocol at pilot scale; the Tier-2 reduction to 3 seeds is a declared deviation (§7) forced by the free-tier GPU envelope. |
| Determinism | Local (CPU fixtures): `torch.set_num_threads(8)` fixed, `torch.use_deterministic_algorithms(True)` where supported, no dropout, no data-parallel — the determinism control requires **bitwise-identical** perplexity between two seed-0 fixture runs. Cloud (GPU): `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, TF32 off, fixed thread count, and a **reproducibility control** that re-runs seed 0 once and requires the perplexity to agree within 1e-6 relative; if it does not, the observed spread is reported and used as the noise floor instead of a tolerance. | CPU determinism is achievable and must be proven; GPU kernels may be non-deterministic in reduction order, so the cloud control is stated as a measured tolerance rather than an assumption. |

### 3.1 Design R1 — LoftQ (trend T2)

Arms (per seed), all with the **same** rank `r`, the same target modules
(`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`), the same optimizer (AdamW), the same
learning-rate search grid as the paper (`{1e-5, 5e-5, 1e-4, 3e-4}`, selected on the *validation* split
only), and the same number of steps. **The arm set is model-agnostic**: it is run first on the Tier-1
pilot (SmolLM2-135M, 5 seeds) and then on the Tier-2 primary (TinyLlama-1.1B, 3 seeds); both are
CLOUD-RUN, and the Tier-2 result is the one that carries the claim (§5.1).

| Arm | Base weights | Adapter initialisation |
|---|---|---|
| `R1-FP16-LoRA` | fp32 (no quantization) | standard (Kaiming A, B = 0) |
| `R1-std-2bit` | 2-bit fake quant (NF2-style codebook, block 64) | standard (Kaiming A, B = 0) — the QLoRA/fixup scheme |
| `R1-loftq-2bit` | 2-bit fake quant, **same** quantized matrix `Q` | LoftQ initialisation, `T = 5` alternating steps (paper: "5 iterations for all GLUE tasks", Appendix D.2) |
| `R1-loftq-2bit-T1` | same `Q` | LoftQ with `T = 1` (paper: T=1 is the QLoRA-quantized weight + SVD of the residual, §3.2) |

Metric: **WikiText-2 test perplexity** = `exp(mean cross-entropy)` over non-overlapping 1024-token
windows (the metric the upstream code prints: `utils/huggingface_utils.py:629-635` in LR-QAT;
`train_clm.py:697-701` in LoftQ). Secondary metric: **initialisation output error**
`E_X ‖X Wᵀ − X (Q + BA)ᵀ‖²` (the frozen proxy unit of `design-m2-interfaces.md` §0.3) measured on a
held-out calibration batch — this is the quantity the paper's Figure 2 tracks.

Rank: `r = 16` (paper's DeBERTaV3 setting) and `r = 64` (paper's LLaMA-2 setting) for the
initialisation-level arms; the model-level arms use **`r = 16`** only, to bound cost (declared
deviation).

### 3.2 Design R2 — LR-QAT (trend T1)

Arms (per seed), run on both model sizes exactly as R1 (Tier-1 pilot first, Tier-2 primary second):

| Arm | Weights | Trainable | Source |
|---|---|---|---|
| `R2-FP16` | fp32 | none (eval only) | reference point |
| `R2-RTN-4bit-g128` | 4-bit symmetric fake quant, g128, RTN scales | none (eval only) — this is exactly the upstream "PTQ baseline" recipe (`README.md`: run the QAT script with `--max-train-steps 0`) | PTQ baseline |
| `R2-LRQAT-4bit-g128` | 4-bit fake quant, g128; `Φ₀` downcast to the paper's fixed-point `Q4.4` form | low-rank `A,B` (rank 32, as the paper's default `r = 32`) **plus** the step size `s` (learned, `lr_s` searched in `{0, 1e-5}` as in Table B1) | the method under test |
| `R2-fullQAT-4bit-g128` | 4-bit fake quant, g128 | **all linear weights** (`lr_W` searched in `{1,5,10,50,100}·1e-5`) | the paper's upper reference arm; implemented with the pinned `torchao` QAT *prepare* path (BSD-3-Clause dependency) rather than reimplemented |

Training recipe (from LR-QAT Table B1, applied unchanged where the scale allows): AdamW,
`β = (0.9, 0.95)`, weight decay 0 for `A,B` and `s`, linear warmup + linear decay, 10 % warmup,
1000 steps, batch 512 tokens, gradient-clipping max-norm 1.0, `α = 1.0` in `(α/r)·AB`, embeddings /
LM head / RMSNorm frozen. Learning-rate grid searched on the validation split: `lr_AB ∈
{1e-5, 1e-4, 1e-3, 1e-2}` (paper's grid), `lr_s ∈ {0, 1e-5}`.

Metric: WikiText-2 test perplexity (same evaluator as R1), plus the **merge-identity check** (T1-a)
performed on the trained model: `max |logits(merged) − logits(adapter)|` relative, and exact integer
equality of the merged `clip(⌊Φ₀ + C⌉)` matrix.

**Additional arm — LOCAL-CPU-MEASUREMENT (not training, stays on the workstation).** Because a real CPU
low-bit kernel path exists for artifacts *we serialize ourselves* (`docs/research/environment.md` §1
"Corrected on 2026-10-08"; `docs/research/backend-capability.md` §2.4/§3 rows 2, 11), the merged
LR-QAT model and the RTN model are additionally exported to an **int4 weight-only ONNX container**
(`MatMulNBits`, block size 32/128) that this project writes, and executed with ONNX Runtime 1.30.0 on
CPU. That yields a **class-4-CPU** measurement of the paper's "no inference overhead after merging"
claim, comparing like with like (both artifacts are ours, same container format, same runtime). It is
explicitly **not** a class-4-GPU or class-5 claim, and it may not be compared with any published
latency number. Outputs of the ORT session are also checked against the fake-quantized forward
(numerical agreement), so the arm doubles as an independent-format validation of the merge.

---

## 4. Cost estimate: local CPU-hours vs cloud GPU-hours

The M3 workload is now split by substrate (`AGENTS.md` §2b; `docs/research/environment.md` §5,
superseded). Local cost is measured; cloud cost is an **analytical estimate (class 1)** that the first
`run_manifest.json` replaces with measured GPU-hours.

### 4.1 Basis A — local CPU (measured on this workstation, 2026-10-08, `torch 2.14.1+cpu`, 8 threads)

```
fp32 matmul  (1024,1024,1024) : 7.57 ms/call   283.7 GFLOP/s
fp32 matmul  (4096,4096,4096) : 421.91 ms/call  325.8 GFLOP/s
fp32 matmul  (5120,5120,5120) : 844.19 ms/call  318.0 GFLOP/s
tiny 6-layer d512 ffn2048, seq512 fp32 forward : 57.9 ms  (8844 tok/s; 39.9 M params)
   -> 251 GFLOP/s effective on a real (small-GEMM) transformer forward
SmolLM2-135M layer shapes at 512 tokens:
  (512x576)@(576x576)   0.736 ms   461.4 GFLOP/s
  (512x576)@(576x1536)  1.876 ms   482.9 GFLOP/s
  (512x1536)@(1536x576) 1.914 ms   473.4 GFLOP/s
  backward grad_x       0.690 ms   492.3 GFLOP/s
  backward grad_w       0.684 ms   496.5 GFLOP/s
```

Used for the local blocks only: fixture work is dominated by SVD (`torch.linalg.svd` on 576×576 to
4096×4096 matrices), fake quantization of a few hundred matrices, ONNX serialization, and perplexity
arithmetic on collected artifacts — all sub-minute-to-minutes jobs at the throughput above.

### 4.2 Basis B — cloud GPU (analytical, class 1)

FLOPs model: forward `2·N_ne` FLOP/token, training step ≈3× that, attention terms ≈5 % at sequence
length 512; `N_ne` (non-embedding parameters) from the published configs:

| Model | `N_ne` | fwd+bwd FLOPs / 512-token step | Assumed effective bf16 throughput | Steps only | With 2–3× harness overhead |
|---|---|---|---|---|---|
| `SmolLM2-135M` (pilot) | ≈106.2 M | ≈0.33 TFLOP | 10–20 TFLOPS | ≈0.03 s/step (33 s) | **2–4 min per arm** |
| `TinyLlama-1.1B` (primary) | ≈968 M (22 layers, d 2048, FFN 5632, 4 KV heads; 1.03 B total incl. the tied 32 k×2048 embedding) | ≈2.97 TFLOP | 10–20 TFLOPS | ≈0.15–0.30 s/step (2.5–5 min) | **10–20 min per arm** |

The 10–20 TFLOPS effective figure is 15–30 % of a T4's 65 TFLOPS bf16 peak, i.e. deliberately
pessimistic for memory-bound small-model training. **These are estimates, not measurements**: the
first cloud run's `run_manifest.json` GPU-hours replace them, and M3.0's cost probe is what closes the
loop (§10.2). If measured GPU-hours exceed 2× the estimate, the budget is re-planned by amendment
*before* the remaining arms start.

### 4.3 Cost table

**Local (CPU-hours; fixtures, validation, generation, collection, analysis — no training):**

| Block | Runs | Per run | Total |
|---|---|---|---|
| Tier-0 fixtures: T2-a residual dominance, merge-identity algebra, synthetic + extracted-matrix sweeps | ~12 short jobs | 2–10 min | ≤ 1.5 h |
| Reimplementation-vs-oracle agreement (PEFT `loftq_init` on CPU) + local determinism control | 2 jobs | 5–15 min | ≤ 0.5 h |
| Notebook generation + `RunSpec` validation + registry checks (determinism digest tests) | per config | < 1 min | ≤ 0.5 h |
| Class-3 packed storage: serialize the quantized/merged models to our int4 + int8 ONNX containers and measure bytes | 2 formats × 5 seeds | 2–5 min | ≤ 0.5 h |
| **Class-4-CPU arm** (local by definition): ONNX Runtime 1.30.0 `MatMulNBits` int4 session on the merged vs PTQ artifacts, 100 timed forwards each, plus numerical agreement vs the fake-quant forward | 2 artifacts × 5 seeds | 3–10 min | ≤ 1 h |
| Collection, checksum validation, statistics, figures, report text | — | — | ≤ 3 h |
| **Local total** | | | **≈ 6–8 CPU-hours** |

**Cloud (GPU-hours; all training and evaluation):**

| Block | Runs | Per run | Total |
|---|---|---|---|
| Tier-1 pilot (SmolLM2-135M, 5 seeds): `FP16`+`RTN` eval-only; `LRQAT`+`fullQAT`; R1's four arms; LR selection (validation split only) | 2×5 + 2×5 + 4×5 + 8 | 1–4 min | ≈ 1.5–2.5 h |
| Tier-1 pilot contingency (3-bit fallback, §5.4) | 3 arms × 5 seeds | 2–4 min | ≈ 0.5–1 h |
| Tier-2 primary (TinyLlama-1.1B, 3 seeds): `FP16`+`RTN` eval-only; `LRQAT`; `fullQAT`; R1's four arms; LR selection | 2×3 + 1×3 + 1×3 + 4×3 + 8 | 3–25 min | ≈ 5–9 h |
| Tier-2 contingency (3-bit fallback) | 3 arms × 3 seeds | 10–20 min | ≈ 1.5–3 h |
| Failed/aborted attempts preserved (pre-emption, OOM on the full-QAT arm) — budgeted, never hidden | — | — | ≈ 1–3 h |
| **Cloud total** | | | **≈ 10–18 GPU-hours** |

### 4.4 Free-tier envelope, quota, and pre-emption handling

| Platform | Envelope targeted | Pre-emption / failure handling |
|---|---|---|
| **Kaggle Notebooks** (preferred unattended backend) | free tier: weekly GPU quota (T4×2 or P100 class); the adapter submits via the official CLI, persists the remote run id, polls with bounded backoff, and fetches outputs | a pre-empted or quota-killed session is detected by `status()`; the run is **resumed** from the persisted remote run id if the adapter reports `can_resume`, otherwise it is recorded `failed` with its logs and enters the research record — **never silently retried into a success** |
| **Google Colab** (developer's chosen vehicle) | consumer Colab exposes **no submission API**: `build_notebook()` emits a ready-to-open notebook that the developer runs; free tier is T4-class with session/usage limits | the adapter records `submitted_by: "human"` and never claims to have started the run; an interrupted session is re-run from the same generated notebook (identical spec ⇒ identical notebook digest) and the duplicate is recorded, not merged |
| **Colab Enterprise** | only with an authorized GCP project, `SPECTRAQUANT_ALLOW_PAID=1` and `RunSpec.max_cost_authorized_usd` set | paid resources are refused by `assert_submission_allowed()` without prior authorization and a stated ceiling; free tiers are the default (`AGENTS.md` §2b.8) |

**Caps for Milestone 3**: **12 local CPU-hours** and **20 cloud GPU-hours** (inside one Kaggle weekly
quota or the equivalent Colab budget), plus the rule that no paid resource starts without prior
authorization and a stated maximum cost. Quota consumption is logged per submission.

### 4.5 Memory

Local: peak RSS ≤ 3.5 GB for the fixture/measurement blocks (matrix sweeps, ONNX serialization, ORT
sessions) — far inside the 15.1 GiB usable. Docker containers (16 CPUs, **7.318 GiB**) are used only
where a Linux-only wheel is required and are *not* used for the memory-heavy blocks.

Cloud: the pilot (135 M) fits any T4-class session trivially. The primary (1.1 B) needs ≈2.2 GB for
bf16 weights plus LoRA/activations; the **`fullQAT` arm trains all weights**, so it must use bf16
weights + 8-bit AdamW + gradient checkpointing to fit a 16 GB T4, and if it still OOMs the failure is
recorded (not silently replaced) and trend T1 is reported on the `FP16` / `RTN` / `LRQAT` arms with the
deviation logged (§7, DEV-0012).

---

## 5. Predeclared tolerances and decision criteria

All of the following are fixed **before** any run. They are stated in terms of the *ordering* claims,
because absolute perplexity at 135 M parameters is not comparable to the papers' 7B/70B values.

### 5.1 Statistical criterion (direction)

Paired across seeds within each arm set (pilot: 5 seeds; primary: 3 seeds):
- **Confirmed**: the predicted sign holds in **all** seeds of that arm set — 5 of 5 at the Tier-1 pilot
  and 3 of 3 at the Tier-2 primary. Exact one-sided paired sign test: `p = (1/2)^5 = 0.031` (pilot) and
  `p = (1/2)^3 = 0.125` (primary). Predeclared: **the Tier-2 primary result carries the claim**; the
  Tier-1 pilot is a harness/feasibility check and may only be reported as supporting evidence. Where
  the two disagree, the Tier-2 result governs and the pilot is reported as such.
- **Suggestive, not confirmed**: one seed short of the above (4 of 5, or 2 of 3) — reported as
  suggestive; it may not be used to claim the trend.
- **Falsified**: two or more seeds short of the above, or the sign reverses in the pooled medians.

Justification for n = 5: it is the published protocol's own n (LR-QAT Table 3 "5 runs"; LoftQ Table 5
"median over five random seeds"), so the reproduction is compared with the paper on equal statistical
footing, and a paired design removes the between-seed variance that dominates a 135 M model's
perplexity.

### 5.2 Magnitude criterion (effect size), and how the thresholds were chosen

Define the two gaps **inside this reproduction**, using this project's own FP16 and baseline arms:

```
G_ptq  = PPL(RTN-4bit)          - PPL(FP16)            # damage done by PTQ
G_std  = PPL(std-init @ 2bit)   - PPL(FP16-LoRA @ 16bit)  # damage done by 2-bit + standard init
```

| Trend | Predeclared magnitude requirement |
|---|---|
| T1 (LR-QAT) | `PPL(RTN) − PPL(LR-QAT) ≥ 0.25 · G_ptq` — i.e. at least **25 %** of the PTQ damage is recovered |
| T2 (LoftQ) | `PPL(std-init) − PPL(LoftQ) ≥ 0.25 · G_std` — i.e. LoftQ closes at least **25 %** of the gap opened by 2-bit quantization + standard initialisation |
| T1 upper bound | `PPL(LR-QAT) ≤ PPL(fullQAT) + max(0.1, 0.1·PPL(fullQAT))` — LR-QAT must not be materially worse than training all weights |

Why 25 % — justified *a priori*, not fitted:
1. **The published effect is 61–72 % gap closure.** LR-QAT Table 4 (W4 pc, LLaMA-2 7B): RTN 6.14,
   LR-QAT 5.66, FP16 5.47 ⇒ closure `(6.14−5.66)/(6.14−5.47) = 72 %`; W4 g128: RTN 5.78, LR-QAT 5.59,
   FP16 5.47 ⇒ 61 %. LoftQ's 2-bit result is stronger still (QLoRA fails to converge, LoftQ reaches
   7.85 ppl; Table 5).
2. **We still expect systematic attenuation** from a smaller model (1.1 B primary vs 7 B; 135 M pilot),
   a 10× shorter training budget (10³ vs 10⁴ steps), a smaller learning-rate search, and fp32/bf16
   instead of the papers' setup. Requiring 25 % means the reproduction must capture roughly **one
   third** of the published closure fraction — deliberately conservative at both model sizes.
3. **25 % is well above the noise floor.** The paper's own seed spread is ±0.01–0.02 ppl on 7B
   (LR-QAT Table 3: 5.66 ± 0.00, 6.13 ± 0.02). At our scales we will *measure* the noise floor with the
   determinism/reproducibility control (§3) and the FP16-arm seed spread; the 25 % threshold is
   expressed as a fraction of a gap, so it scales with whatever gap the model actually exhibits.
4. **Choosing a threshold after seeing results would be tuning on the outcome.** 25 % is therefore
   fixed now and cannot be revised by a later amendment that follows a failed run.

### 5.3 Exactness tolerances (gates, not trends)

| Check | Predeclared tolerance | Justification |
|---|---|---|
| T1-a merge identity, integer path | **exact integer equality** of the merged weight matrix (`clip(⌊Φ₀ + C⌉)` computed on both paths) | both sides are the same integer arithmetic; any difference is a bug, not floating-point noise |
| T1-a merge identity, fp path | `max_rel_err = max|logits_merged − logits_adapter| / max|logits_adapter| ≤ 1e-5` | fp32 accumulation over 30 layers; 1e-5 is ~100× above the fp32 round-off floor of a 576-dim matmul (`~1e-7`) and ~10⁴× below any logit difference that could change a perplexity comparison |
| T2-a residual dominance | `‖W − (Q + BA)‖_F(LoftQ T=1) < ‖W − Q‖_F(std init)` for **≥ 95 % of targeted matrices**, and the alternating error sequence is non-increasing until the break rule fires | the T=1 residual is by construction the rank-r optimum of `W − Q` (`min_Q,A,B ‖W − Q − ABᵀ‖_F ≤ min_Q ‖W − Q‖_F`, LoftQ Eq. 6); a violation means our SVD/quantizer convention is wrong. The 5 % slack covers matrices where the 2-bit `Q` is so coarse that the rank-16 correction is negligible at fp32 precision |
| Reimplementation vs independent oracle | relative Frobenius difference between our LoftQ `T=1` output and PEFT's `loftq_init` (`num_bits=2`, `method="normal"`, `block_size=64`, Apache-2.0, CPU path) on a 768×768 matrix: **≤ 1e-6** on the returned quantized weight and on the residual | both are deterministic closed-form fp32 operations; the only legitimate difference is the SVD sign convention, which cancels in `L·R`. 1e-6 is ~10× above the fp32 floor for a 768×768 Frobenius norm and still tight enough to catch a wrong convention (transposed `L/R` would give O(1) differences) |
| Determinism control | two seed-0 runs must give **bitwise-identical** perplexity | CPU-only, single-node, fixed thread count; if violated, record the op and report the measured spread instead of asserting determinism |
| Class-3 container accounting | `accounted_bytes` from `spectraquant.quantization` must equal the serialized ONNX int4/int8 artifact's byte count, modulo an explicit metadata-overhead rule | the container is ours and its layout is known; the independent measurement (`backend-capability.md` §2.4: 2048 int4 values → exactly 1024 B payload) shows the packing is bit-exact, so a mismatch is a bug in our accounting |
| Class-4-CPU numerical agreement | ONNX Runtime `MatMulNBits` int4 output vs our fake-quantized forward: `max_rel_err ≤ 1e-3` on a fixed input batch | if the block axis, block size and scale placement agree, both sides multiply by the *same* dequantized fp32 weight matrix, so the only difference is accumulation order (≤1e-5 relative for a 576-dim reduction). 1e-3 is therefore ~100× slack while still ~1000× below any convention error, which would be O(1). The probe's own int4 error of `2.041e-01` on N(0,1) weights (`backend-capability.md` §2.4) is quantization error, not kernel error, and is *identical* on both sides because both use the same scales |

### 5.4 Powering gates (predeclared stop/no-claim conditions)

- If `G_ptq < 0.5` ppl, the T1 model-level design is declared **underpowered**; the result is reported
  as *inconclusive*, **not** as a negative result. (Calibration: the published W4 gaps are 0.67 ppl at
  pc and 0.31 ppl at g128 on 7B; 0.5 sits between them and is deliberately generous for a small model.)
- If `G_std < 0.5` ppl at 2-bit, the T2 design is underpowered; the predeclared contingency is to run
  the same three arms at **3-bit** (LR-QAT's own ablation scale, §4.1) and report the 3-bit trend as the
  T2 result, with the 2-bit outcome reported as inconclusive. No other configuration may be substituted.
- If any arm **diverges** (PPL `> 1000`, NaN, or loss not decreasing after 100 steps), the divergence
  rate per arm becomes the primary endpoint for that trend, and the predeclared criterion is: the
  method under test diverges in **strictly fewer** seeds than its baseline within the same arm set
  (pilot: out of 5; primary: out of 3).
- If a **gate** check in §5.3 fails, Milestone 3 stops before any model-level claim: the implementation
  is wrong and no trend statement is permitted.

---

## 6. Stopping rule (fixed in advance)

1. **Per-run cap**: any single run is terminated at 1000 steps *or* at `RunSpec.timeout_minutes`
   (pilot: 20 min; primary: 40 min), whichever comes first. A terminated run is recorded as
   `incomplete`/`failed` and is **not** replaced by a longer run and **not** silently dropped.
2. **Session pre-emption**: a session killed by the platform (Colab timeout, Kaggle quota) is handled
   by the adapter: if `can_resume(remote_run_id)` is true the run resumes from the persisted remote id;
   otherwise it is recorded `failed` with its logs and its failure reason. **A failed run is never
   silently retried into a success** (`AGENTS.md` §2b.4; `design-cloud-adapter.md` §5).
3. **Block cap**: each block in the §4 cost tables is capped at 1.5× its worst-case estimate; when a
   block hits its cap, remaining seeds of that block are reported as `not run (cap)`.
4. **Global caps**: **12 local CPU-hours** and **20 cloud GPU-hours** for the whole of Milestone 3, with
   quota consumption logged per submission. When a cap is reached, execution stops and the report
   contains exactly the arms completed, with the criteria applied to them; un-run arms are listed as
   such. No budget extension — local or cloud, free-tier or paid — without a timestamped amendment in
   `docs/research/preregistration-amendments.md` *before* the extension is used.
5. **No paid resource** is started without the user's explicit authorization and a stated maximum
   estimated cost (`SPECTRAQUANT_ALLOW_PAID=1` plus `RunSpec.max_cost_authorized_usd`); free tiers are
   the default (`AGENTS.md` §2b.8).
6. **No test-set peeking**: WikiText-2 **test** perplexity is computed **once per arm, at the end of
   that arm**. All selection (learning rate, rank, schedule) uses the **validation** split. The test
   split is never used to steer a decision, and no arm is re-run because its test number was
   disappointing.
7. **No substitution**: if a criterion fails, the failure is reported. The only permitted re-run is one
   caused by an infrastructure fault (crash, OOM, pre-emption), which must be logged with the fault's
   evidence — and the re-run's result is recorded as a *new* run, not merged into the failed one.
8. **Stop-on-gate-failure**: §5.3 gate failures stop the milestone immediately.
9. **Negative/inconclusive results are published** (AGENTS.md §4.10); the write-up must state which
   criteria were met and which were not, in the predeclared terms.

---

## 7. Deviations-from-paper log

Every departure from the published protocol is logged in a machine-readable file at
`artifacts/manifests/m3-deviations.jsonl` (one JSON object per line), created **before** the first run
and appended during execution. Predeclared schema:

```json
{
  "deviation_id": "DEV-0001",
  "date": "2026-10-XX",
  "method": "lr-qat | loftq",
  "paper_ref": "Table 3 | Table B1 | Appendix D.2 | §3.2",
  "paper_value": "LLaMA-2 7B, A100-80GB, 1e4 steps, bs 32, seq 1024",
  "reproduction_value": "SmolLM2-135M pilot / TinyLlama-1.1B primary, cloud GPU, 1e3 steps, 512 tokens/step, seq 512",
  "class": "scale | budget | hardware | quantizer | dataset | evaluation | metric",
  "reason": "cloud-only execution policy (AGENTS.md §2b); free-tier GPU envelope",
  "expected_impact": "attenuates effect size (smaller model, shorter training); no expected impact on ordering",
  "affects": ["T1", "T2"],
  "declared_before_run": true
}
```

The log is written by hand for the predeclared deviations (this plan enumerates them below) and by the
run harness for anything discovered mid-flight. An entry with `declared_before_run: false` that touches
a criterion, a hyperparameter grid, or a stopping decision is a **preregistration violation** and must
be reported as such.

Predeclared deviations (to be logged with the IDs above at M3 start):

| # | Paper | This reproduction | Class |
|---|---|---|---|
| DEV-0001 | LLaMA-1/2/3, Mistral 7B–13B (LR-QAT); LLaMA-2 7B/13B (LoftQ) | SmolLM2-135M (Tier-1 pilot) and TinyLlama-1.1B (Tier-2 primary) | scale |
| DEV-0002 | A100-80GB; 2.25 days per 10⁴ steps at 7B; 202 GPU-days total | cloud free-tier GPU (T4/P100 class) or Colab T4; 10³ steps; ≈10–18 cloud GPU-hours + ≈6–8 local CPU-hours | hardware/budget |
| DEV-0003 | LR-QAT: 10⁴ steps, batch 32, seq 1024, eval seq 2048 | 10³ steps, 512 tokens/step, seq 512, eval seq 1024 | budget |
| DEV-0004 | LoftQ LLM path uses bitsandbytes NF4 (4-bit) and a fake NF quantizer for sub-4-bit | our own NF-codebook + uniform fake quantizer (class 2) | quantizer |
| DEV-0005 | LR-QAT: SlimPajama subset for QAT, Wikipedia validation for LR selection | WikiText-2 train for QAT, WikiText-2 validation for LR selection | dataset |
| DEV-0006 | LR-QAT rank 32; LoftQ rank 64 (LLaMA-2) / 16–32 (DeBERTa) | rank 32 (LR-QAT arms) and 16 (LoftQ model-level arms), 16/64 at initialisation level | scale |
| DEV-0007 | LoftQ alpha never stated in the paper | `lora_alpha = 16` with `r = 16` (scaling 1.0), stated explicitly | unknown-in-paper |
| DEV-0008 | LR-QAT searches `lr_s ∈ {0, 1e-5}` with no results table | we run the learned-`s` arm only; the `lr_s = 0` arm is an optional exploratory control, never the primary | unknown-in-paper |
| DEV-0009 | Zero-shot accuracy on 6 lm-eval tasks | not reproduced (six tasks × 5 seeds exceeds the cloud GPU budget for M3); perplexity only | evaluation |
| DEV-0010 | Merge identity asserted by the paper without an explicit numeric tolerance | we predeclare exact integer equality + 1e-5 fp logit tolerance | metric |
| DEV-0011 | Papers' hardware: A100-80GB, fp16/bf16, large batches | cloud notebook GPU (T4/P100 class), bf16, 512 tokens/step, gradient checkpointing where needed | hardware |
| DEV-0012 | LR-QAT's full-model QAT baseline trains at 7B with checkpointing | at TinyLlama-1.1B the `fullQAT` arm must use bf16 weights + 8-bit AdamW + gradient checkpointing to fit 16 GB; if it OOMs it is recorded as failed and T1 is reported on the `FP16`/`RTN`/`LRQAT` arms | hardware/budget |
| DEV-0013 | Papers report no execution substrate | every run executes on the cloud notebook substrate via generated notebooks and returns a checksum-validated `run_manifest.json`; a run without one does not exist | infrastructure |
| DEV-0014 | Paper protocols use 5 runs at their scale | Tier-1 pilot uses 5 seeds; Tier-2 primary uses 3 seeds (AGENTS.md §6 minimum) because of the free-tier GPU envelope | budget |

---

## 8. What CANNOT be reproduced here, and why (explicit infeasibility statement)

**Scope of this statement.** Two corrections apply to it. (a) The initial reading of the *local*
compute envelope ("no CUDA ⇒ fake quantization only") was too strong and was corrected on 2026-10-08 by
measurement (`docs/research/environment.md` §1 "Corrected on 2026-10-08";
`docs/research/backend-capability.md` §2–§4): three real CPU low-bit surfaces exist on this host —
ONNX Runtime 1.30.0 (`MatMulNBits` int4 weight-only, `MatMulInteger` int8), torchao 0.18.0
`IntxWeightOnlyConfig(torch.int4, PerGroup(32))`, and bitsandbytes 0.50.2's Windows CPU backend (NF4
**storage** with fp32 compute) — and Linux containers run. (b) Under `AGENTS.md` §2b the local machine
may **not** run research training at all: Tiers 1–5 execute on the cloud notebook substrate. So the
correct statement is not "we cannot run this" but "we cannot run this **locally**; the runs happen on
cloud GPUs under the manifest/checksum rules". What remains genuinely impossible is listed below.
Nothing in this plan may be stated as "no low-bit kernel exists here".

1. **The published absolute numbers.** LR-QAT's results are LLaMA-1/2/3 7B–13B and Mistral-7B on a
   single A100-80GB, 10⁴ steps at batch 32 × seq 1024 (Table B1), 2.25–4.5 days per run (Appendix B);
   LoftQ's are LLaMA-2 7B/13B on unspecified A100s, and 2-bit LLaMA-2-7B alone needs ≈15 GB (Table 8).
   Our cloud runs are T4/P100-class at 10³ steps on 135 M/1.1 B models, so **no number in this plan's
   output may be compared with a published number**; only the predeclared ordering/magnitude criteria
   are testable. Free-tier GPU-hours also make the papers' 202 A100-GPU-days unreachable by design.
2. **Zero-shot / downstream task evaluation** (LR-QAT's 6-task average, LoftQ's GLUE/SQuAD/ANLI/GSM8K/
   ROUGE numbers). The GPU budget for M3 cannot carry six lm-eval tasks × seeds; GSM8K additionally
   needs sampled generation. Declared as DEV-0009.
3. **Local execution of any training or large-scale inference.** Forbidden by `AGENTS.md` §2b /
   `environment.md` §5 (superseded): the local machine is an orchestration, correctness and analysis
   host. A local training run would be a policy violation even where it is technically possible, and
   Tier 2+ results MUST NOT be reported as completed from local execution.
4. **Class 4-GPU and class 5 remain unavailable until a cloud GPU run actually executes a supported
   low-bit kernel.** No such run exists yet, and the papers' own kernels are out of reach: bitsandbytes'
   real low-bit matmuls (CC ≥ 6.0/7.5), torchao's `Int4WeightOnlyConfig` tinygemm
   (`ImportError: Requires mslk >= 1.0.0`), GPTQ/AWQ/Marlin/exllama/Triton, flash-attention. What *is*
   available is **class 4-CPU** on our own int4/int8 ONNX containers, measured **locally** (it is a
   kernel measurement on a self-serialized artifact, not training). A class-4-GPU label requires a
   cloud run that executes a real supported kernel under `docs/protocols/benchmark-protocol.md`;
   until then it is unavailable, and class 5 stays unavailable by contract.
5. **lq-lora's mixed-precision allocation** (ILP over a 3⁵ configuration grid, Gurobi, Fisher-weighted
   SVD). Its runtime is CUDA/ray/gurobi-bound (`upstream-notes.md` §3.3); its data prep requires
   `param.cuda()` (`models/lq_utils.py:160`) and cluster paths. Only its *decomposition mathematics* is
   in scope for this project, and even that is Milestone 2 work, not Milestone 3.
6. **Running any upstream repository unmodified.** Every upstream entry point remains blocked:
   LR-QAT by unconditional `torch.cuda.memory_*` calls (`utils/utils.py:188-190`) and a `time.clock`
   Windows path (`:57-59`); LoftQ by `device_map="auto"` / `.to('cuda')` / `torch.cuda.amp.GradScaler`
   in its scripts and by PEFT's LoftQ path requiring a CUDA compute device for the 4-bit branch;
   lq-lora by `bitsandbytes`/`pytorch_quantization` imports, `ray`, and `device="cuda"` hard-coding
   (`models/packbits_utils.py:241-264`, `models/lora_utils.py:131,197`); torchao's QAT only partially
   (its int4 *convert* target requires mslk+CUDA). A Linux container removes the Windows-specific
   blockers and the `ray[air]` install problem, but **not** the CUDA-kernel and hard-coded-device
   blockers; and our cloud runs use *our own* reimplementations, not their scripts. Full evidence with
   `file:line` in `docs/research/upstream-notes.md` §3–4.
   One nuance worth recording: because bitsandbytes 0.50.2 has a CPU backend, the *specific* 4-bit NF4
   model load used by LoftQ's scripts is no longer impossible on a CPU host
   (`backend-capability.md` §2.6 shows a real `Linear4bit` forward on CPU) — but that is 4-bit storage
   with fp32 compute, and the scripts' CUDA-typed training/eval plumbing still blocks them.
7. **The papers' Fisher/data-aware variants** (LQ-LoRA Fisher-weighted SVD; LoftQ's Figure 2 spectral
   diagnostics) beyond the frozen proxy unit in `design-m2-interfaces.md` §0.3.
8. **The published hardware claims** ("7B trainable on a 24 GB consumer GPU", "27 GB for 2.75-bit
   70B"). A free-tier T4/P100 session cannot test them; they are cited as context only.
9. **Any class-4 claim about a third-party format or about speed.** ORT's `MatMulNBits` executes a
   container *we* wrote, so it is authorized for our own artifacts only
   (`docs/protocols/measurement-taxonomy.md` §5; `docs/research/backend-capability.md` §4). Timings of
   our ONNX artifacts are engineering telemetry unless the taxonomy's authorization rule is satisfied;
   they are never a latency claim about the papers' methods on their hardware.

---

## 9. Decision table: candidate designs, substrates, cost and fidelity

Cost = the §4 split (measured local CPU-hours vs estimated cloud GPU-hours). Fidelity = how much of the
published claim the design can actually test.

| # | Design | Cost | Scientific fidelity | Verdict |
|---|---|---|---|---|
| D0 | **Matrix-level only** (T2-a + T1-a on extracted weight matrices + synthetic matrices; no training) — LOCAL-FIXTURE | **≤ 1.5 local CPU-h** | Tests the *mechanism* exactly and is the only place where the papers' equations can be checked to fp precision; reproduces no published trend (both papers' claims are fine-tuning claims) | **Mandatory as gate, insufficient alone** |
| D1 | **Tier-1 pilot**: SmolLM2-135M + WikiText-2, 10³ steps/arm, 5 seeds, fake quantization, R1 (4 arms) + R2 (4 arms) — CLOUD-RUN, plus LOCAL class-3 container accounting and the LOCAL class-4-CPU ORT arm | **≈1.5 local CPU-h + 2–3.5 cloud GPU-h** | Proves the harness end-to-end at 1/8 of the primary cost and gives a fallback dataset if quota is lost; on its own it is not the configuration AGENTS.md §6 names for the reference-reproduction class | **Mandatory pilot (runs first)** |
| D2 | **Tier-2 primary**: TinyLlama-1.1B (the upstream LR-QAT smoke model; registry M1) + WikiText-2, 10³ steps/arm, 3 seeds, R1 (4 arms) + R2 (4 arms incl. the torchao `fullQAT` reference) — CLOUD-RUN | **≈5–9 cloud GPU-h** (+ ≈2 local CPU-h analysis) | The configuration AGENTS.md §6 names for the reference-reproduction class; 6× closer to the papers' LLaMA family than any 135 M model; still 6× smaller than 7B and 10× shorter than the papers' training | **RECOMMENDED — this result carries the claim** |
| D3 | D2 with the papers' 10⁴ steps | **≈50–90 cloud GPU-h** | Closer to the published training regime | Rejected: exceeds any free-tier quota; would require paid authorization + a stated ceiling, i.e. a preregistration amendment |
| D4 | **Run upstream code unmodified** | n/a | Highest fidelity in principle | **Infeasible** — every entry point is hard-blocked by CUDA-typed calls and CUDA-only kernels (§8.6), locally *and* in a container |
| D5 | D1/D2 with the **full-model-QAT arm taken from `torchao`** (BSD-3-Clause dependency, CPU-verified `prepare` path, GPU-side same API) instead of a reimplementation | +0 GPU-h (replaces an arm) | Adds an independent third-party implementation of the QAT reference arm, reducing "we wrote both sides" risk; LR-QAT itself is still reimplemented (licence: no patent grant) | **Adopted inside D1/D2** |
| D6 | D2 with a larger LR search and rank sweep (4 LRs × 3 ranks × 3 seeds) | **+20–40 cloud GPU-h** | Better hyperparameter parity with the papers | Rejected: exceeds the free-tier envelope; LR search stays one-axis on the validation split |
| D7 | **PEFT-based LoftQ oracle** as the primary LoftQ implementation instead of our reimplementation | +0 | Stronger external validation of the initialisation, but PEFT's `replace_lora_weights_loftq` supports **only bitsandbytes 4-bit** (`backend-capability.md` §3 row 16), which is storage-with-fp32-compute, and using it would make our result depend on a library we do not control | **Adopted as a LOCAL oracle only** (gate in §5.3), never as the reported implementation |
| D8 | **Containerised (Linux) run of the upstream repositories** (local or cloud) | +5–20 h setup, then infeasible | Would remove the Windows-only blockers (the `time.clock` path, `ray[air]` installation) | **Rejected**: the blocking cause is CUDA-only kernels plus hard-coded `cuda` devices, which a container does not fix, and no upstream script is sized for a 135 M/1.1 B model |
| D9 | **Colab-only execution** (skip Kaggle) | same GPU-hours | Consumer Colab has **no submission API**, so runs are developer-driven and unattended batching is impossible; session/usage limits are opaque | Rejected as the *unattended* backend: Colab notebooks are generated and remain the developer's interactive vehicle (`design-cloud-adapter.md` §1), Kaggle is the programmatic one |
| D10 | **Skip Tier 2; claim from the Tier-1 pilot only** | 2–3.5 cloud GPU-h | Cheapest path to a number | Rejected: AGENTS.md §6 makes the TinyLlama-1.1B-class configuration the reference-reproduction requirement; a 135 M pilot cannot carry the claim |
| D11 | **Run the papers' actual 7B scale on paid cloud** | ≥ 10³ GPU-h + storage | Highest fidelity | Rejected: no paid resource without the user's explicit authorization and a stated ceiling (AGENTS.md §2b.8); out of scope for M3 |

**Recommendation: D2 as the claim-bearing design, with D1 as its mandatory pilot, D0 as the gate, D5
inside both, D7 as a local oracle, and Kaggle as the unattended backend (Colab notebooks generated for
the developer's interactive use).** Reasoning: D0 alone cannot touch either published trend; D3/D6/D11
exceed the free-tier envelope or require paid authorization; D4/D8 are blocked by CUDA-only kernels and
hard-coded CUDA devices, which neither a container nor a wheel upgrade removes; D9/D10 leave either the
unattended path or the required model class unsatisfied. The corrected local envelope is used where it
genuinely helps — byte-exact class-3 containers we write ourselves and a **local** class-4-CPU
execution of those containers — and is explicitly *not* stretched into a GPU-latency or service claim.
The residual risk — that a 1.1 B model at 10³ steps attenuates the effects below the 25 % magnitude
threshold — is handled by the predeclared powering gates (§5.4), which turn attenuation into a declared
*inconclusive* outcome rather than a silent failure.

---

## 10. Milestone mapping, execution order, and the M3 gate

Master-spec milestone ids (AGENTS.md §6, `docs/coordination/status.md`):
**M0** bootstrap · **M1** literature / protocol / preregistration · **M2** math + systems correctness ·
**M3** reference reproduction · **M4** proxy validation · **M5** method · **M6** allocator ·
**M7** primary matrix · **M8** ablation · **M9** deployment · **M10** release.

This plan *is* the M3 design. Its entry conditions come from M0–M2; its exit condition is the M3 gate
below. There is no separate "G" numbering in this plan.

### 10.1 Entry conditions (must already hold when M3 starts)

- **M0 (bootstrap)**: environment resolves on CPU (`uv sync --all-extras`), `torch 2.14.1+cpu`,
  `torchao 0.18.0`, `onnxruntime 1.30.0`, `bitsandbytes 0.50.2` and the pinned upstream SHAs are
  recorded; Docker Linux containers verified usable; the cloud adapter
  (`docs/coordination/design-cloud-adapter.md`) is implemented and its own tests are green
  (`RunSpec` validation, notebook determinism, registry `validated` unreachable without matching
  checksums, budget guard).
- **M1 (literature / protocol / preregistration)**: WikiText-2, SmolLM2-135M and TinyLlama-1.1B
  licences recorded in `docs/research/upstream-lockfile.md` / `model-dataset-licenses.md` (stream A);
  the measurement taxonomy (`docs/protocols/measurement-taxonomy.md`) is frozen, including the
  class-4-CPU authorization rule.
- **M2 (math + systems correctness)**: `spectraquant.quantization` (`QuantSpec`, `fake_quantize`,
  `accounted_bytes`, `pack_int4`/`unpack_int4`), `spectraquant.factorization` (frozen `W ≈ B @ A`
  convention) and the M2 verification gate of `design-m2-interfaces.md` §6 are green on CPU.

### 10.2 M3 execution order — every step tagged with its substrate

Tags: **LOCAL-FIXTURE** = generation/validation/analysis on the workstation (no training);
**LOCAL-CPU-MEASUREMENT** = the class-4-CPU ONNX Runtime kernel measurement (a measurement, not
training); **CLOUD-RUN** = any training or evaluation of SmolLM2-135M, TinyLlama-1.1B or any LM, on
the notebook substrate. Each step is a gate: the next one does not start if it fails.

| Step | Substrate | Content |
|---|---|---|
| **M3.0 — prerequisites & cost probe** | LOCAL-FIXTURE (all of it) | Dataset token count measured; fixture peak RSS measured; the Tier-1 pilot's generated notebook executed once on the cloud to obtain **measured GPU-hours** and replace the §4.2 estimate; `RunSpec`/registry/budget paths exercised. If measured GPU-hours exceed 2× the estimate, the budget is re-planned by a timestamped amendment in `docs/research/preregistration-amendments.md` **before** the remaining arms start (never a silent overrun). |
| **M3.1 — implementation validation** | LOCAL-FIXTURE, plus LOCAL-CPU-MEASUREMENT for the ORT check | T1-a merge identity (exact integer equality + the fp tolerance of §5.3); T2-a residual dominance; reimplementation-vs-oracle agreement against PEFT's `loftq_init`; the local determinism control; `accounted_bytes == measure_serialized_bytes` on our ONNX int4/int8 containers; the class-4-CPU numerical-agreement check against ONNX Runtime. |
| **M3.2a — Tier-1 pilot** | CLOUD-RUN (Kaggle unattended; Colab notebook generated for interactive use) | R2 (LR-QAT, trend T1) then R1 (LoftQ, trend T2) on SmolLM2-135M, 5 seeds each, 10³ steps × 512 tokens per arm, plus the LR selection runs (validation split only). |
| **M3.2b — Tier-2 primary** | CLOUD-RUN | The same arm set on TinyLlama-1.1B, 3 seeds each, plus the 3-bit contingency if a powering gate fires. |
| **M3.2c — collection** | LOCAL-FIXTURE | `spectraquant cloud collect` per run id: fetch artifacts + logs + the *executed* notebook, and validate them. |
| **M3.3 — analysis** | LOCAL-FIXTURE | The criteria of §5 applied verbatim; the stopping rule of §6 honoured; every reported number labelled with its measurement class (1, 2, 3 or 4-CPU only). |
| **M3 gate (exit)** | — | See below. |

**Notebook rule (binding).** The M3 run notebooks are **generated**, never hand-written:
`spectraquant cloud notebook --config configs/experiment/<m3-arm>.yaml --platform colab --out
notebooks/generated/<m3-arm>.ipynb` (contract: `docs/coordination/design-cloud-adapter.md` §3, §7).
All scientific logic stays in `src/spectraquant/**`; a notebook may only install the pinned
environment, fetch versioned data, call repo code, record the commit, capture hardware/dependency
metadata, run the declared experiment, write `run_manifest.json`, export artifacts, and tear down.
A notebook whose logic exists only in a cell is a defect, and a hand-edited notebook that cannot be
regenerated is not evidence.

### 10.3 M3 gate (exit)

M3 passes only if:

(a) every §5.3 exactness gate passed, **locally**;
(b) the trend statements are expressed exactly as §5.1/§5.2 allow (confirmed / suggestive / falsified
    / inconclusive), with no criterion renegotiated after the fact, and the Tier-2 primary result is
    the one that carries the claim;
(c) the deviations log is complete and every deviation that touches a criterion is flagged;
(d) **every collected artifact bundle passed checksum validation against the spec before any number
    entered the analysis** — `run_manifest.json` present and schema-valid, git commit matching the
    spec, every `expected_artifacts` entry present with a matching sha256 and ≥ `min_bytes`, executed
    notebook present (`design-cloud-adapter.md` §5). **A run without a validated `run_manifest.json`
    does not exist**: it is recorded `rejected`/`failed` and its numbers may not appear anywhere in the
    report, not even as "preliminary";
(e) failed and pre-empted runs are preserved in the registry with their failure reasons and are not
    counted as successes;
(f) the report is written with negative and inconclusive outcomes included; and
(g) **no new-method result is promoted** — SpectraQuant's own method (low-rank + low-bit joint
    training, the sensitivity proxy, the allocator) is *not* compared against, or claimed over, these
    baselines at M3. Method results belong to **M5** (method) and are validated against the M7 primary
    matrix; M3 only establishes that the reference baselines behave as published, at reduced scale,
    under predeclared criteria. Any promotion of a method claim before the M3 gate closes is a contract
    violation.

### 10.4 Acceptance for this plan document (already satisfied)

Numeric tolerances justified in advance (§5), a cost estimate split into measured local CPU-hours and
analytical cloud GPU-hours with its basis (§4), an explicit and precise infeasibility statement (§8),
pinned baselines with raw command evidence (`upstream-notes.md` §1), confirmation that no upstream code
is in the repository (§2 and `upstream-notes.md` §0.1), and no claim that any reproduction has been
performed.

---

## 11. Verified environment facts used above (with their sources)

| Fact | Value | How verified |
|---|---|---|
| SmolLM2-135M licence | `apache-2.0`, not gated, 1 729 903 downloads, revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` | `GET https://huggingface.co/api/models/HuggingFaceTB/SmolLM2-135M` |
| SmolLM2-135M architecture | `LlamaForCausalLM`; 30 layers, d 576, FFN 1536, 9 heads / 3 KV heads, vocab 49 152, tied embeddings, `max_position_embeddings` 8192, `torch_dtype` bfloat16 | `GET …/SmolLM2-135M/raw/main/config.json` (full JSON in the transcript; parameter count computed from it) |
| TinyLlama-1.1B licence + registry | `apache-2.0`; already registered as row M1 of `docs/research/model-dataset-licenses.md` with revision `59f6f375b26bde864a6ca194a9a3044570490064` and fp32 size 4.40 GB; also the model the upstream LR-QAT README uses for its smoke test | `docs/research/model-dataset-licenses.md` row M1 (stream A), cross-checked against the upstream `README.md` run command |
| WikiText-2 licence + size | HF tags `license:cc-by-sa-3.0`, `license:gfdl`; train parquet 6 357 543 B, validation 657 209 B, test 732 610 B; dataset revision `b08601e04326c79dfdd32d625aee71d232d685c3` | `GET https://huggingface.co/api/datasets/Salesforce/wikitext` and `…/tree/main/wikitext-2-raw-v1`; revision cross-checked against `docs/research/model-dataset-licenses.md` row D1 |
| torchao wheel availability on Windows/CPU | `torchao-0.18.0-py3-none-any.whl` (pure Python) plus a manylinux wheel; latest release `0.18.0` | `GET https://pypi.org/pypi/torchao/json` |
| CPU throughput basis | see §4 raw output | local `torch 2.14.1+cpu` benchmark, 8 threads |
| Real CPU low-bit kernel path (class 3/4-CPU) | ONNX Runtime 1.30.0 `MatMulNBits` (int4 weight-only, `com.microsoft`) and `MatMulInteger` (int8) execute on this CPU; int4 packing is bit-exact (2048 four-bit values → 1024 B payload); torchao 0.18.0 `IntxWeightOnlyConfig(torch.int4, PerGroup(32))` forwards on CPU; bitsandbytes 0.50.2 has a Windows CPU backend (NF4 storage, fp32 compute); torchao's `Int4WeightOnlyConfig` fails with `Requires mslk >= 1.0.0` | `docs/research/backend-capability.md` §2.3–§2.6 (raw probe output, owner D-lite) and `docs/research/environment.md` §1 "Corrected on 2026-10-08" (independently reproduced by the orchestrator) |
| Container budget | Docker Linux containers usable: kernel 6.18.33.2-WSL2, **16 CPUs, 7.318 GiB RAM** | `docs/research/backend-capability.md` §2.1 raw `docker info` / `docker run` output |
| Upstream SHAs / licences | see §2 and `upstream-notes.md` §1 | `git ls-remote`, `git rev-parse`, `gh api`, licence files read |
| Execution substrate policy | local machine may not run research training; Tiers 1–5 execute on the cloud notebook substrate (Colab = developer-run generated notebooks, no submission API; Kaggle = programmatic unattended; Colab Enterprise only with an authorized GCP project); every run emits `run_manifest.json` | `AGENTS.md` §2b (binding, added 2026-10-08), `AGENTS.md` §6 (scope ladder rewritten), `docs/research/environment.md` §5 (superseded), `docs/coordination/design-cloud-adapter.md` (frozen) |
| Cloud adapter contract used by this plan | `RunSpec` (run_id, config, platform, git_commit, dataset_refs+checksums, seeds, gpu_required, timeout_minutes, max_cost_authorized_usd, expected_artifacts, measurement_class_expected); `build_notebook`/`notebook_digest`; `submit/status/fetch/can_resume`; `collect()` validates the manifest + checksums before any metric is exposed; `assert_submission_allowed()` blocks paid platforms without authorization | `docs/coordination/design-cloud-adapter.md` §2–§6, §7 (CLI) |
| No upstream code in the repository | `git status --short` shows no `LR-QAT/`, `LoftQ/`, `lq-lora/` or `ao/` path; clones live at `%TEMP%/sq-upstream/` (outside the tree) | `upstream-notes.md` §0.1 raw output, re-run after all four clones were created |

---

## 12. Risks and open questions

1. **Effect attenuation** (the main scientific risk): a 1.1 B model at 10³ steps (or the 135 M pilot)
   may show a gap below the powering thresholds. Mitigation: predeclared inconclusive outcome +
   predeclared 3-bit contingency.
2. **2-bit divergence for every arm** would make the sign test degenerate; handled by the divergence-rate
   endpoint (§5.4).
3. **Tokenizer/dataset mismatch**: our tokenizer (SmolLM/TinyLlama BPE, 49 k/32 k) differs from the
   papers' LLaMA-2 tokenizer, so perplexity is not comparable to any published value — stated in §8.1
   and in the method cards.
4. **Full-QAT arm via torchao** is a different code path from LR-QAT's own full-QAT baseline; if its
   QAT recipe differs materially (e.g. it does not support per-group symmetric int4 static ranges in
   the same way), the arm is replaced by a reimplementation and the change logged as a deviation.
   On TinyLlama-1.1B it additionally needs bf16 + 8-bit AdamW + checkpointing to fit a 16 GB T4
   (DEV-0012), and an OOM is recorded as a failure rather than worked around silently.
5. **`references/method-cards/TEMPLATE.md`** is owned by another wave-1 stream and was not present at
   the time of writing; the four method cards in this milestone follow the field list given in the
   assignment. If the template later fixes different field names, the cards must be re-synchronised
   (recorded as a coordination dependency, not a scientific one).
6. **Peak RSS and cloud GPU-hours are estimates, not measurements** (§4.2, §4.5); M3.0 measures both
   and closes the loop before the bulk of the runs.
7. **ONNX Runtime integration caveats** (found by the D-lite probe, `backend-capability.md` §3 row 11):
   `onnx 1.23.2` writes IR version 14 while ORT 1.30 accepts ≤ 13 (the probe had to pin
   `model.ir_version`), and the int4 quantizer requires the extra `onnx_ir` package. Both must be
   handled in the container-writing code path before M3.1, and the pinned versions recorded.
8. **bitsandbytes CPU is not a low-bit compute path** (`backend-capability.md` §3 row 4: NF4 storage,
   dequantize-then-fp32). It is used, if at all, only for a *storage-format* cross-check of our class-3
   numbers — never for a speed or kernel claim, and never as the reported LoftQ quantizer.
9. **`torchao`'s QAT prepare path is CPU-supported but not an upstream-benchmarked target**
   (`backend-capability.md` §3 row 2: not listed in TorchAO's own hardware matrix). The D5 full-QAT arm
   therefore carries a declared dependency risk; if the arm cannot reproduce the paper's per-group
   symmetric int4 *static-range* behaviour, it is replaced by a reimplementation and the change is
   logged as a deviation (already listed in §12.4 above).
10. **PEFT's LoftQ oracle availability** changed with the corrected envelope: bitsandbytes 0.50.2 has a
    CPU backend, so PEFT's `replace_lora_weights_loftq` (which supports only bnb 4-bit) may now execute
    on CPU. Whether the *initialisation* itself (as opposed to model loading) runs bnb-free on CPU must
    be verified at M3.1; if it does not, the oracle falls back to the pinned PEFT `loftq_init` source
    path with `num_bits = 2` (its own pure-PyTorch NF quantizer), which was already CPU-runnable.
11. **Free-tier pre-emption and quota exhaustion** (the main infrastructure risk): a Kaggle weekly GPU
    quota or a Colab session limit can truncate a run mid-arm. Handled by the predeclared rules — resume
    from the persisted remote run id when `can_resume` is true, otherwise record `failed` with its logs;
    never silently retried into a success. The Tier-1 pilot runs first precisely so that a quota loss
    still leaves a complete, analysed arm set.
12. **Colab has no submission API**: the developer's interactive vehicle cannot be automated, so the
    unattended path depends on Kaggle (or an authorized GCP project for Colab Enterprise). If Kaggle
    access is unavailable, M3.2 slips to developer-driven Colab runs and the GPU-hour cap is enforced
    by hand — recorded as an infrastructure deviation, not a scientific one.
13. **GPU-hour estimates are analytical**: the 10–20 TFLOPS effective figure is unmeasured on the actual
    cloud GPU. If the real throughput is worse by more than 2×, the budget amendment path in §6.4
    applies *before* further arms start.
14. **Cloud GPU determinism**: cuBLAS/cuDNN reductions may be non-deterministic even with the flags in
    §3, so the cloud reproducibility control is a measured 1e-6 relative tolerance rather than a
    bitwise guarantee; if it fails, the measured spread becomes the noise floor.
