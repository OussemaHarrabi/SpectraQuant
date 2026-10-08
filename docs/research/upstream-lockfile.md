# SpectraQuant — Upstream Lockfile

Every upstream repository, paper artifact, dataset, or documentation source that SpectraQuant may
consult, import, or reimplement is pinned here with a **verified** commit SHA and a **verified**
license. No component enters the core experiment path without an entry.

**Verification method.** SHAs were obtained on 2026-10-08 with
`gh api repos/<owner>/<repo>/commits/main --jq .sha` and licenses with
`gh api repos/<owner>/<repo> --jq .license.spdx_id`. Raw output is preserved verbatim in
`references/upstream-verification.log`. SHAs are content-addressed defaults of the `main` branch at
that time; they are **not** guesses and **not** copied from papers. Re-verify with the same command
before importing.

**Exception — immutable release pin (U6).** For `EleutherAI/lm-evaluation-harness` we pin the
**immutable release tag** `v0.4.13` rather than the moving branch head: the annotated tag object
`eb4678c395c2c1f157d462b8f3a09831160c861a` peels to commit
`ddd67220430a2470529f25fd5c05a576ca1057a0` (2026-08-31, "chore(version): 0.4.13"), verified with
`gh api repos/EleutherAI/lm-evaluation-harness/git/ref/tags/v0.4.13` and
`.../git/tags/<tag-object-sha>`. The `main` head observed on 2026-10-08 was
`d6de81643928d653435c431bae19945d41d32520`; the released tag is preferred for a frozen evaluation
protocol. The divergence is recorded in the stream I eval protocol and in the run manifests, which
carry the tag commit.

**License caveat.** `pytorch/ao` reports SPDX `NOASSERTION` through the GitHub API. Inspecting
`LICENSE` shows a BSD-3-Clause-style Meta license with an Arm section (Copyright 2023 Meta; Arm
2024–2026). Treat it as BSD-3-Clause-equivalent but confirm the exact text before vendoring code; we
do **not** vendor it.

---

## 1. Pinned upstream repositories

| # | Repo | Paper / arXiv | Pinned commit (verified) | License (verified) | Intend to import | Intend to reimplement | Known blockers here |
|---|---|---|---|---|---|---|---|
| U1 | `Qualcomm-AI-research/LR-QAT` | arXiv:2406.06385 | `8795afe054cf951b714299e01083a1b354721829` | `BSD-3-Clause-Clear` | Read the low-rank auxiliary-weight QAT formulation and quantizer semantics for method-card comparison (M3). | The QAT training loop at Tier-1 scale, own code, executed on the **cloud** substrate (only semantics/fixture checks run locally). | Official examples target GPU/CUDA; bitsandbytes-style low-bit paths are CUDA-only. Import is **reading/reference only**; no upstream code is executed locally. |
| U2 | `yxli2123/LoftQ` | arXiv:2310.08659 | `ae33fd4fd05fd4ba146555cd77c13d307eb4e9b3` | `MIT` | Reference the alternating quantize-SVD initialization for the Q2 ordering comparator arm. | Alternating LoftQ-style initialization: small-matrix fixtures locally (M3); the Tier-1 run on the **cloud** substrate. | Repo's experiments are GPU/LLM-scale; no CUDA-free reference path. |
| U3 | `HanGuo97/lq-lora` | arXiv:2311.12023 | `c2424b3adc27197815da1ac9e1304565168d824d` | `MIT` | Reference the data-aware quantized+low-rank decomposition and rank budgeting. | Rank-budget allocation for the H4 comparator: fixtures locally, **cloud** run for Tier-1 scale. | GPU-oriented; quantizer kernels require the cloud substrate. |
| U4 | `pytorch/ao` (TorchAO) | TorchAO QAT docs (page updated 2026-03-25) | `cff77b46ef85ba8472b4a286598073d6c01e18e5` | `NOASSERTION` (BSD-3-Clause-style Meta/Arm; see caveat) | Reference the `prepare`/`convert` QAT API and fake-quantizer semantics (Class-2 measurement design). | CPU fake-quantization semantics of int4/int8 weight-only per-group, if needed. | CUDA kernels (int4 tinygemm, float8, etc.) **deferred**; `prepare`-stage fake quantization is CPU-portable. |
| U5 | `huggingface/peft` | HF PEFT quantization guide | `8bb3e702976b48f16c1995c97b4cb32b038e1e15` | `Apache-2.0` | Use the library for LoRA/QLoRA baseline plumbing at Tier 1 if a pretrained model is used. | — | The 4-bit backends (bitsandbytes, GPTQ, AWQ, HQQ, EETQ) are CUDA/GPU-tier; CPU-only imports must avoid them. |
| U6 | `EleutherAI/lm-evaluation-harness` | arXiv:2405.14782 (Biderman et al.) | **`ddd67220430a2470529f25fd5c05a576ca1057a0`** (annotated tag `v0.4.13`, tag object `eb4678c395c2c1f157d462b8f3a09831160c861a`, peeled commit 2026-08-31) | `MIT` | Use for Tier-1/Tier-2 zero-shot evaluation protocol and task definitions. | A minimal local task runner only if the harness is too heavy on CPU. | Large task suite is CPU-slow; local runs limited to a predeclared small task subset. |
| U7 | `vllm-project/vllm` | arXiv:2309.06180 (PagedAttention) | `72d59adc2c76de8f77aad640c7a04d663a970735` | `Apache-2.0` | Reference only: deployment/quantization semantics for the discussion section. | — | **Not runnable here** (CUDA-only serving). Explicitly deferred; no latency claims. |

**Verify command (reproduce all rows):**

```bash
for r in Qualcomm-AI-research/LR-QAT yxli2123/LoftQ HanGuo97/lq-lora pytorch/ao \
         huggingface/peft EleutherAI/lm-evaluation-harness vllm-project/vllm; do
  printf '%s ' "$r"; gh api "repos/$r/commits/main" --jq .sha
  gh api "repos/$r" --jq '.license.spdx_id'
done
# U6 immutable release pin:
gh api repos/EleutherAI/lm-evaluation-harness/git/ref/tags/v0.4.13 --jq .object.sha
gh api repos/EleutherAI/lm-evaluation-harness/git/tags/eb4678c395c2c1f157d462b8f3a09831160c861a --jq .object.sha
```

## 2. Papers consulted (metadata verified; see `references/bibliography.bib`)

All arXiv IDs below were verified by fetching `arxiv.org/abs/<id>` on 2026-10-08. No entry in
`references/bibliography.bib` is unverified.

Core: LR-QAT (2406.06385), LoftQ (2310.08659), LQ-LoRA (2311.12023), QA-LoRA (2309.14717),
QLoRA (2305.14314), LoRA (2106.09685).

PTQ baselines: GPTQ (2210.17323), AWQ (2306.00978), SmoothQuant (2211.10438), LLM.int8() (2208.07339),
OWQ (2306.02272), SqueezeLLM (2306.07629), ZeroQuant (2206.01861), ZeroQuant-V2 (2303.08302),
LLM-FP4 (2310.16836), Atom (2310.19102), Outlier Suppression+ (2304.09145).

Allocation / proxies: HAWQ (1905.03696), HAWQ-V2 (1911.03852), OBC (2208.11580), APTQ (2402.14866),
SliM-LLM (2405.14917), Layer-Wise Quantization (2406.17415), MLoRQ (2507.09616), AutoQRA (2602.22268),
KV-COBRA (2609.24298), SRR / Preserve-Then-Quantize (2602.02001, ICML 2026), MixQuant (2607.23047),
CoopQ (2509.15455), KronQ (2607.07964), RAM / spectrally-flat probe (2609.33923).

Low-rank compression: ASVD (2312.05821), SVD-LLM (2403.07378), Palu (2407.21118), SliceGPT (2401.15024).

Regularizers / spectra: Spectral Normalization (1802.05957), Dynamical Low-Rank + spectral regularizer
(2505.08022), Spectrum (2406.06623).

Foundations: Gholami et al. survey (2103.13630), Nagel et al. white paper (2106.08295), OBD (LeCun
et al., NeurIPS 2, 1990), OBS (Hassibi & Stork, NeurIPS 5, 1993), Deep Compression (1510.00149),
Learning both Weights and Connections (1506.02626).

Evaluation / serving: Biderman et al. (2405.14782), vLLM/PagedAttention (2309.06180).

Non-arXiv item flagged: *When Low-Rank Meets Mixed-Precision* (ASP-DAC 2026, pp. 604–610) — no arXiv
preprint verified; cited only with its conference venue and treated as unverified for code/metadata.

## 3. Models

| Model family | Identifier | License | Status |
|---|---|---|---|
| Tier-1 primary | tiny decoder-only transformer trained from scratch | our own artifact | `CLOUD-COLAB` (not local) |
| Tier-1 secondary | a ~125M pretrained LM (GPT-2-class candidate) | TBD at freeze (must be recorded here before use) | `CLOUD-COLAB`, forward-only |
| Tier-2 | TinyLlama-1.1B-class | TBD at freeze (Apache-2.0 candidate) | `CLOUD-GPU` |
| Tier-3 | second 0.6–1.7B model | TBD at freeze | `CLOUD-GPU` |

No model is loaded into the core path until its row here carries a **verified** license string and a
pinned revision (HF revision SHA recorded at freeze).

## 4. Datasets

| Dataset | Source | License/terms | Splits used | Status |
|---|---|---|---|---|
| WikiText-2 (raw) | standard public distribution | record the dataset's terms at freeze | train / validation / test (test read once) | `CLOUD-COLAB` (Tier 1), `CLOUD-GPU` (Tier 2) |
| TinyStories subset | public distribution | record at freeze | deterministic slice | `CLOUD-COLAB` (Tier 1) |
| LM Eval Harness tasks | U6 pinned commit | per-task; recorded at freeze | predeclared subset | `CLOUD-COLAB` (small subset) |
| CIFAR-100 / ImageNet-100 | public | record at freeze | per standard | `CLOUD-GPU` (Tier 4) |

Dataset **revision hashes** and the subsample indices are recorded at the preregistration freeze
(§13 of `preregistration.md`).

## 5. Deferred (CUDA-only) components — explicitly out of the core path

`bitsandbytes` 4-bit/8-bit kernels, TorchAO CUDA kernels (int4 tinygemm, float8), GPTQ/AWQ/HQQ GPU
kernels, `vllm` CUDA serving, FlashAttention, `torch.profiler` GPU paths.
Rule (`AGENTS.md` §2.1–2.2): these MAY be referenced/pinned/documented as deferred but MUST NOT enter
the core experiment path, and any code path requiring CUDA MUST fail loudly rather than silently fall
back to different numerics.

**Not deferred — a verified CPU low-bit kernel path (class 4-CPU).** Measured 2026-10-08
(`AGENTS.md` §5 as amended; `docs/research/backend-capability.md` §2.4; `environment.md` §1
correction): ONNX Runtime 1.30.0 CPU executes `MatMulNBits` (int4 weight-only, `com.microsoft`) and
`MatMulInteger` (dynamic int8), and torchao 0.18.0 `IntxWeightOnlyConfig(torch.int4, PerGroup(g))`
forwards on CPU. Class 4-CPU is available **only** for containers SpectraQuant serializes itself, and
only when the claim names backend, op, container, thread count and CPU model and is compared against
an fp32 CPU baseline in the same session. Class 4-CPU results are **not** comparable to published GPU
numbers and carry **no** latency/throughput claim. Class 4-GPU and class 5 remain deferred
(unavailable).
