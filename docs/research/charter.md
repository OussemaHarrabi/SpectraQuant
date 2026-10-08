# SpectraQuant — Research Charter (distilled scope)

Status: **Wave 1, draft 0.1 (2026-10-08).** Derived from the master orchestrator prompt and
constrained by `docs/research/environment.md` and `AGENTS.md` (authoritative). This file states
*what we are claiming to study, what we refuse to study, and what the hardware actually allows*.
It is deliberately narrower than the prompt: the compute envelope in §7 is binding.

---

## 1. Problem

Modern transformer compression stacks two lossy transforms:

1. **Low-rank factorization** — replace a weight matrix $W \in \mathbb{R}^{d_{out}\times d_{in}}$ by
   $W \approx LR$ with $L \in \mathbb{R}^{d_{out}\times r}$, $R \in \mathbb{R}^{r\times d_{in}}$,
   storing $r(d_{out}+d_{in})$ parameters instead of $d_{out}d_{in}$.
2. **Quantization** — store (and, in deployment, compute with) values at a reduced bit width
   $b$, with a rounding rule, scale, and (usually) zero-point.

The two transforms interact. Low-rank truncation removes energy from the spectrum of $W$; which part
of the spectrum survives determines how much quantization noise is *exposed* (energy that rounding
will move) versus *recoverable* (energy already discarded before rounding). Conversely, adaptive
rounding (e.g. GPTQ) can partly compensate reconstruction error but has less room when the matrix is
already rank-limited. Whether preparing a model to be low-rank helps or hurts a *subsequent*
quantization step is an empirical question with competing published answers for the fine-tuning
regime (LoftQ/LQ-LoRA vs. LR-QAT vs. QA-LoRA) and almost no controlled study at *equal stored
memory* across rank and bit-width choices.

## 2. Primary research question

> **Does preparing a transformer for low-rank factorization improve or damage its subsequent
> quantization, and can an output-aware sensitivity proxy allocate layer-wise ranks and bit widths
> that produce a better quality–memory Pareto frontier than uniform compression at equal memory?**

"Output-aware" means the sensitivity signal is derived from how a layer's *output* (not just its
weights or its input activations) changes when the layer is compressed — i.e. it propagates
activation statistics through the compressed operator instead of scoring weights in isolation.

## 3. Supporting questions

| ID | Question | Why it matters |
|---|---|---|
| **Q1** | At **equal stored bytes**, does low-rank-preparation-before-quantization beat (a) direct quantization of the full-rank weight, and (b) quantization-error-reconstruction alone (LoftQ-style)? | Isolates "preparation helps" from "correction helps". |
| **Q2** | Does the *order* of the two transforms (rank-then-quantize vs. quantize-then-rank-residual) change quality at fixed rank and bit width? | Decides which pipeline the allocator should plan for. |
| **Q3** | How well does an output-aware proxy rank layers by true post-compression damage compared with weight-magnitude, activation-magnitude, and Hessian-trace heuristics? | The proxy is the allocator's only input; invalid ranking ⇒ invalid allocation. |
| **Q4** | Does allocating rank/bit width per layer from the proxy dominate uniform compression **on the quality–memory Pareto frontier** (not just at one budget)? | The actual contribution claim. |
| **Q5** | Does a **rounding-aware spectral regularizer** applied during low-rank preparation improve post-quantization quality at equal memory versus unregularized preparation? | Tests whether the regularizer is causal or cosmetic. |

Question→hypothesis mapping: Q1 → H1 and H4; **Q2 → comparator arm only (no H-id)**; Q3 → H2;
Q4 → H4; Q5 → H3. Measurable/measurement feasibility is H5's subject (fake quantization vs. real
packed storage/kernels).

## 4. Hypotheses (canonical specification IDs)

These are the **five hypotheses pre-registered by the master specification**. They keep those exact
IDs (`H1`–`H5`) everywhere: here, in `preregistration.md` (traceability table), in
`novelty-risk.md`, in the risk register, and in all results tables. Each states the claim, its local
operationalisation, and its comparison.

- **H1 — Factorisation error vs. rounding sensitivity (double-edged preparation).** Naive low-rank
  preparation can **reduce factorisation error while increasing the sensitivity of the factors to
  rounding**. *Operationalisation:* at equal stored bytes, measure (i) factorisation reconstruction
  error and (ii) output distortion under the target quantizer, for naive truncated-SVD preparation
  vs. unprepared full-rank quantization. H1 holds if some regime shows (i) improving while (ii) does
  not improve or worsens — i.e. the two error notions dissociate. Locally testable (Tier 0 synthetic
  spectra; Tier 1 tiny transformer).
- **H2 — Proxy validity.** A proxy incorporating the **activation covariance** and the
  **quantization residual** correlates more strongly with downstream degradation than weight-space
  Frobenius error alone. *Operationalisation:* Spearman/Pearson of candidate scores vs. measured
  per-layer and end-to-end degradation; **weight-space Frobenius error is the predeclared
  comparator**. Tier 0/1 and (forward-only) a ~125M pretrained LM.
- **H3 — Rounding-aware spectral regularizer.** A rounding-aware spectral regularizer improves
  compressed-model quality **at equal memory** relative to an unprepared factorisation.
  *Operationalisation:* with/without ablation at equal stored bytes, plus a control replacing the
  rounding-grid term with a plain spectral penalty (isolates whether "rounding-aware" does any work).
- **H4 — Layer-wise mixed rank and mixed precision under a global budget.** Allocation dominates
  uniform allocations **on at least part of the quality–memory frontier**. *Operationalisation:*
  Pareto dominance/hypervolume at the predeclared budget ladder vs. uniform rank/bit and vs. a
  HAWQ-V2-style bit-only allocator at equal stored bytes.
- **H5 — Fake-quantization gains need a packed kernel to become real gains.** Some apparent quality
  improvements under fake quantization will **not** translate into real latency or memory
  improvements unless a compatible packed kernel is used.
  **Operationalisation here (binding — see §6, §7 and amendments A-0001/A-0003):** H5 has **three
  distinct halves** and only the first two are locally testable:
  1. *packed-storage survival* — does the fake-quantization quality gain survive our own
     **packed int4/int8 serialization + CPU dequantization** at measured stored-byte parity
     (measurement classes 2 → 3)?
  2. *CPU-kernel survival* — does it survive execution by a **real CPU low-bit kernel on an artifact
     SpectraQuant serializes itself** (class **4-CPU**, e.g. ONNX Runtime `MatMulNBits`/`MatMulInteger`
     or torchao intx weight-only) versus an fp32 CPU baseline measured on the same machine in the
     same session? **Scope limit:** this is a CPU-kernel check only; it is **not** comparable to
     published GPU latency/throughput and no throughput/latency claim follows from it.
  3. *GPU-kernel / service half* (class 4-GPU and class 5) — **not locally runnable**, reported
     "not measured", never estimated; it is **claimable only from a validated cloud run** with a real
     supported kernel and a complete `run_manifest.json` (Tier 2+, M7).
  H5 is therefore reported as **partially supported / inconclusive at GPU scope**, never as fully
  confirmed from local data.

**Q2 is not a hypothesis.** "Does the order of the two transforms change quality?" is a **supporting
question** (§3) implemented as a **comparator arm** (rank-then-quantize vs.
quantize-then-low-rank-residual). It is analysed as a mechanistic comparison feeding H1 and H4, and
must not be cited with an H-id in results tables.

All five hypotheses are **directional** claims. Any may be falsified; a clean negative result is a
publishable outcome (`AGENTS.md` §4.10). The hypothesis → test → falsifier traceability table lives
in `preregistration.md` §11.0.

## 5. Non-goals

1. **Not** an agent / RAG / chatbot / orchestration / robotics project.
2. **Not** a LoRA-fine-tuning-only project: adaptation alone is not compression. LoRA/QLoRA appear
   only as *baselines* or as the vehicle for a quantization-aware training comparison.
3. **No latency or throughput claims.** A real **CPU** low-bit kernel path exists (class 4-CPU) but
   gives no basis for latency/throughput claims; class 4-GPU and class 5 are out of scope here
   (see §6).
4. **No state-of-the-art claims** from narrow or incomparable runs (`AGENTS.md` §4.9).
5. **No 7B+ models, no multi-GPU, no distributed training, no service infrastructure** in the core
   path. 3B–7B (Tier 5) is optional and must never carry core validity.
6. **No external APIs** in the core experiment path; all models/datasets are local and open.
7. **No fake quantization reported as low-bit storage or speedup** (`AGENTS.md` §4.3).
8. **Not** a new quantizer-kernel project; we do not implement CUDA kernels.

## 6. Measurement classes we may emit

Per `AGENTS.md` §5, every number carries exactly one label. Locally available:

| # | Class | Local availability |
|---|---|---|
| 1 | Analytical estimate (shapes/bit widths) | **Yes** |
| 2 | Fake-quantization quality (float execution of quantization numerics) | **Yes** |
| 3 | Packed storage (measured serialized bytes for formats we serialize ourselves, e.g. int8/int4 packing we own) | **Yes, only for our own packers** |
| 4 | Kernel-backed inference | **Split**: **4-CPU Yes** (only for artifacts we serialize ourselves, on a verified CPU kernel); **4-GPU No** (CUDA-only) |
| 5 | End-to-end service | **No** |

Class 4 is split by hardware (`AGENTS.md` §5, verified 2026-10-08 in `docs/research/backend-capability.md`
§2.4): **4-CPU** is available for a low-bit artifact **SpectraQuant itself serialized** and executed by
a real CPU kernel (ONNX Runtime `MatMulNBits` int4 weight-only / `MatMulInteger` int8; torchao intx
weight-only). A 4-CPU claim MUST name the backend, kernel/op, container, thread count and CPU model, and
MUST compare against an fp32 CPU baseline measured in the same session on the same machine — it is
**not** comparable to published GPU numbers. Class 4-GPU and class 5 are always reported "not measured",
never estimated — they are **claimable only from a validated cloud run** with a real supported kernel
and a complete `run_manifest.json` (hardware, driver, kernel, shapes, warmups, repeats). The three memory numbers
(training-time memory, stored checkpoint size, deployed inference representation) are reported
separately and never conflated.

## 7. Honest compute envelope (binding)

Measured 2026-10-08 (`environment.md`): AMD Ryzen AI 7 350 (8C/16T), 16.2 GB RAM, **no CUDA**, AMD
Radeon 860M iGPU unusable for research-grade accelerated training, Windows 11, ≈648 GiB free disk.

**Corrected on 2026-10-08 (see `environment.md` §1 and `backend-capability.md` §2.4):** the initial
reading "no CUDA ⇒ no real low-bit kernel" was too strong. A **CPU** low-bit kernel path exists
(ONNX Runtime 1.30.0 CPU `MatMulNBits` int4 weight-only and `MatMulInteger` int8; torchao 0.18.0
`IntxWeightOnlyConfig(torch.int4, PerGroup(g))` forward), so measurement class **4-CPU** is available
for artifacts we serialize ourselves (§6). Latency/throughput claims (class 4-GPU, class 5) remain
unavailable locally, and no throughput claim follows from the CPU kernel path.

**Substrate split (added 2026-10-08, `AGENTS.md` §2b):** the local machine is an **orchestration and
correctness host** — repository management, CPU unit/property tests, tiny synthetic fixtures, static
analysis, config validation, notebook generation, result analysis, figures/tables/reports, and the
class 4-CPU *measurement*. **All model training, QAT, large-scale inference and GPU evaluation
(Tier 1–5) run on the cloud notebook substrate** (Google Colab as the developer's chosen vehicle;
Kaggle Notebooks preferred for unattended runs; Colab Enterprise with an authorized GCP project).
Tier 1 is therefore **no longer a local workload**; Tier 2–5 additionally require a cloud GPU.

| Tier | Scope | Substrate (binding, `AGENTS.md` §2b) | Status here |
|---|---|---|---|
| **0** | Synthetic matrices, tiny linear nets, exact enumeration, unit/property tests | `LOCAL-FIXTURE` | **Local, required** |
| **1** | Tiny transformer proof: small LM, full-factorial where feasible, ≥5 seeds, proxy ranking validation | `CLOUD-COLAB` (Colab, developer-run; Kaggle unattended) | **Planned, cloud — NOT a local workload** |
| **2** | TinyLlama-1.1B-class + WikiText-2: FP16 / PTQ / QLoRA / LoftQ / LR-QAT vs. SpectraQuant, ≥3 seeds | `CLOUD-GPU` | **Planned, cloud GPU** |
| **3** | Second 0.6–1.7B model | `CLOUD-GPU` | **Planned, cloud GPU (paid ⇒ prior authorization + cost ceiling)** |
| **4** | ViT/DeiT on CIFAR-100/ImageNet-100 (cross-architecture) | `CLOUD-GPU` | **Optional, planned (authorization + cost ceiling)** |
| **5** | 3B–7B | `CLOUD-GPU` | **Optional, planned; core validity MUST NOT depend on it** |
| — | Class 4-CPU kernel measurement on a self-serialized artifact | `LOCAL-CPU-MEASUREMENT` | **Local, required for H5** |

Consequences that are contractual, not aspirational:

- **Tier 1–5 execute only on the cloud notebook substrate**; no Tier-1+ number may be reported as
  measured locally (`AGENTS.md` §2b; `environment.md` §5). A cloud cell is a **result** only once a
  validated `run_manifest.json` and checksum-validated artifacts exist; otherwise it is **not run**.
- Any code path requiring CUDA **on the local host must fail loudly** (`NotImplementedError` /
  explicit config error) and may never silently fall back to a numerically different path.
- Consequence for ambition: the local contribution can *validate the machinery* (proxy, allocator,
  regularizer, accounting, fairness-at-equal-memory, class 4-CPU measurement) and prepare/analyse the
  Tier-1+ cloud runs. The TinyLlama-scale headline result still requires a cloud GPU session.

## 8. Milestones (M-ids from the master specification)

Milestone IDs follow the master specification (**M0–M10**); `docs/coordination/status.md` is the
source of truth for their live status. Milestones relevant to this charter:

| Milestone | Requirement / exit criterion | Status |
|---|---|---|
| **M0** | Clean checkout passes lint, type checks, unit tests, and the tiny experiment. | in progress |
| **M1** | **Literature, protocol and baseline selection** — literature matrix + novelty/differentiation audit + frozen preregistration + pinned upstreams; its **gate is the independent novelty/differentiation verdict**. Records the **execution-substrate decision**: the cloud notebook substrate is the vehicle for Tier 1–5 (free tiers for Tier 1–2; paid instances need prior authorization + a cost ceiling). Tier 2+ remains **not run** until a validated `run_manifest.json` exists. | in progress |
| **M2** | Math fixtures pass; the allocator matches exhaustive search on tiny cases. | not started |
| **M3** | **Reproduction:** bounded reproduction of at least LR-QAT and LoftQ semantics from the **pinned** commits — fixtures and semantics checks run `LOCAL-FIXTURE`; the Tier-1-scale reproduction runs on the cloud substrate; deviations recorded in method cards. | not started |
| **M4** | **Proxy gate:** proxy-vs-exact agreement on Tier-0 fixtures (`LOCAL-FIXTURE`) and proxy ranking validated (**H2** measurable) on the tiny LM (`CLOUD-COLAB`), with CIs. | not started |
| **M7** | **Primary matrix:** Tier-2/Tier-3 confirmatory campaign executed on `CLOUD-GPU` against the **frozen** preregistration, with validated `run_manifest.json` + checksum-validated artifacts. | not started |
| M5, M6, M8–M10 | Defined by the master specification; maintained in `docs/coordination/status.md`. | — |

`preregistration.md` **MUST be frozen at M1** (and before any confirmatory run of any kind).
Exploratory Tier-0/1 work that informs method design is allowed before M1 and is labelled
exploratory; confirmatory claims may only come from the frozen protocol. Scope reductions are legal
only via a timestamped amendment in `preregistration-amendments.md`.

*Substrate note (amendment A-0004).* The earlier framing "GPU not yet secured ⇒ Tier 2+ planned" is
replaced by: "the cloud notebook substrate is the execution vehicle for Tier 1–5; Tier 2+ still
requires a cloud GPU session and is reported **not run** until a validated `run_manifest.json` and
checksum-validated artifacts exist." Local execution remains fixtures, the class 4-CPU measurement,
and analysis.

## 9. Success / failure semantics

- **Success** is not "TinyLlama improves". Success is: a correctly implemented, honestly accounted
  proxy + allocator whose validity (**H2**) and frontier advantage (**H4**) are either demonstrated
  or refuted with pre-registered tests, with every number labelled by measurement class and every
  compression comparison accompanied by an equal-memory counterpart.
- **Failure** is a claim that cannot be mapped to a pre-registered hypothesis, a measurement class,
  or an equal-memory control. Such a claim is retracted, not softened.
