# Method card — LR-QAT (low-rank quantization-aware training)

Field list follows the assignment in the Milestone 3 ticket (wave 1). `references/method-cards/TEMPLATE.md`
was not present when this card was written (owned by another stream) — if the template fixes different
field names, this card must be re-synchronised.

---

## 1. Exact citation

- **Title**: *Low-Rank Quantization-Aware Training for LLMs*
- **Authors**: Yelysei Bondarenko, Riccardo Del Chiaro, Markus Nagel (Qualcomm AI Research)
- **arXiv**: `arXiv:2406.06385` (v1 2024-06-10; **v3 2024-09-03**, the version used here)
- **Venue**: ICLR 2025 poster (OpenReview `dIK0EfZFO9`); earlier version at the ICML 2024 ES-FoMo-II
  workshop. Recorded honestly: the venue was **not** confirmed from a primary proceedings page during
  this task (OpenReview's API returned HTTP 403 and the forum page is script-rendered); the arXiv
  preprint is the authoritative artifact for every number quoted below.
- **Upstream code**: `https://github.com/Qualcomm-AI-research/LR-QAT`

## 2. Upstream repo + pinned commit

| Field | Value |
|---|---|
| Repository | `Qualcomm-AI-research/LR-QAT` |
| Pinned commit | `8795afe054cf951b714299e01083a1b354721829` (2024-11-05) |
| Verified by | `git ls-remote … HEAD` and `gh api repos/…/commits/main --jq .sha` — both agree (raw output in `docs/research/upstream-notes.md` §1) |
| Licence (file read) | `LICENSE` — BSD-3-Clause-style, Qualcomm-modified, **plus an explicit patent non-grant**: *"NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE."* GitHub's SPDX classifier reports `BSD-3-Clause-Clear`. |
| Clone location | `%TEMP%/sq-upstream/LR-QAT` — **outside** the working tree; no upstream source in this repository |
| Entry points (reference only) | `clm_main.py` (`train-baseline`, `train-quantized`, `validate-baseline`, `validate-quantized`); PTQ baseline = `train-quantized` with `--max-train-steps 0 --learning-rate 0 --lr-ab 0` |
| Core file (reference only) | `quantization/hijacker.py` (`_apply_lora_qat_w_int` `:101-125`, `_apply_lora_qat_w_phi` `:128-153`, fixed-point down/upcast `:156-180`, merge `get_params` `:293-380`) |

## 3. Reimplemented vs imported vs adapted

**Reimplemented from the paper's equations. No upstream code is copied or vendored.** Reason: the
licence is a modified BSD-3-Clause with no patent grant, the runtime is CUDA/Windows-blocked, and the
mathematics is a small amount of pure PyTorch. The upstream files are read as a *specification cross-check*
only (their existence proves which quantities are learnable and where the adapter enters the graph).

Implementation target (this project's own interfaces, `docs/coordination/design-m2-interfaces.md`):
`spectraquant.quantization.fake_quantize` (class-2 fake quantization), `spectraquant.factorization`
(`LowRankFactors` with the frozen `W ≈ B @ A` convention), and a new
`spectraquant.regularizers`/`training` QAT wrapper that owns the adapter-inside-the-rounding forward.

The equations to implement (paper references):

```
Eq. (1)  affine uniform quantizer:  x̂ = s·( clip(⌊x/s⌉ + z, −2^{b−1}, 2^{b−1}−1) − z ),  STE on ⌊·⌉
Eq. (3)  full-model QAT (LSQ):      Ŵ = s·clip(⌊W/s⌉, −2^{b−1}, 2^{b−1}−1)
Eq. (4)  LR-QAT, learned scale:     Ŵ = s·clip(⌊W₀/s + (α/r)·AB⌉, −2^{b−1}, 2^{b−1}−1)
Eq. (5)  LR-QAT, frozen scale:      Ŵ = s·clip(⌊W₀/s₀ + (α/r)·AB⌉, …)
Eq. (6)  downcast:                  Φ₀ = φ(W₀/s₀)
Eq. (7)  fixed point:               φ(x) = INT8(⌊2^{8−b}·clip(x, −2^{b−1}, 2^{b−1}−1)⌉),  φ⁻¹(x) = BF16(x)/2^{8−b}
Eq. (8)  final:                     Ŵ = s·clip(⌊Φ₀ + (α/r)·AB⌉, −2^{b−1}, 2^{b−1}−1)
merge:                              Ŵ = s·W_Z with W_Z = clip(⌊Φ₀ + (α/r)AB⌉, …)  (single b-bit integer matrix)
```

The decisive structural point (paper §3, "QAT with low-rank adapters"): the adapter product enters
**inside** the rounding non-linearity, so `AB` lives on the quantization grid and can be absorbed into
the integer matrix afterwards — unlike QLoRA's `y = Ŵx + ABx`.

## 4. Model / data / preprocessing / seed / hyperparameters

**As published (reference configuration, LR-QAT Table B1 + §4):**

| Item | Published value |
|---|---|
| Models | LLaMA-1 7B, LLaMA-2 7B/13B, LLaMA-3 8B, Mistral-0.1 7B |
| Data | small subset of SlimPajama (QAT); hyperparameters selected on Wikipedia validation ppl over 512 sequences |
| Quantization | INT4/INT3/INT2, per-channel ("pc") and group-wise g128; symmetric except INT2 (asymmetric, for comparability); all linear layers except the classification head |
| Weight-activation | INT4 pc weights + per-token activations, KV-cache quantized (4-8-8 / 4-8-4 / 4-4-4) |
| Trainable | `A, B` (rank r) and the scale `s`; embeddings, LM head, RMSNorm frozen; `W₀` frozen |
| Rank | default `r = 32` (1.2 % of 7B parameters); ablations `r ∈ {1, 4, 32, 256}` |
| Init | `B = 0`, `A` Kaiming (He); alternative: LoftQ SVD init with `T = 1` applied **before** downcasting |
| Downcast default | `φ = Qb.(8−b)` (fixed point), `b ∈ {3,4}` ⇒ Q3.5 / Q4.4; BF16 compute |
| Optimizer | AdamW, `β = (0.9, 0.95)`, weight decay 0 (A,B and s); 0.1 for full-model W |
| LR | `lr_AB ∈ {1e-5, 1e-4, 1e-3, 1e-2}`; `lr_s ∈ {0*, 1e-5}` (INT4/INT3), `{1e-5, 1e-4}` (INT2); full-model `lr_W ∈ {1,5,10,50,100}·1e-5` (best 5e-5) |
| Schedule | linear warmup + linear decay; warmup 10 % of steps |
| Steps / batch / seq | 10⁴ steps, batch 32, train seq 1024, eval seq 2048 |
| Other | grad-clip max-norm 1.0; `α = 1.0`; gradient checkpointing; RTN range chosen by an Lᵖ-norm search over `p ∈ {2.0, 2.4, 3.0, 3.5, 4.0, 5.0}` |
| Seeds | 5 runs (Tables 2, 3, A8); not stated for Tables 4–5 |

**As planned for this reproduction (see `docs/research/reproduction-plan.md` §3.2):**

| Item | Value |
|---|---|
| Model | Tier-1 pilot `HuggingFaceTB/SmolLM2-135M` (LlamaForCausalLM, 134.5 M params, Apache-2.0); Tier-2 primary `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` (Apache-2.0, registry M1) |
| Data | `Salesforce/wikitext` `wikitext-2-raw-v1`; contiguous 512-token blocks, `input_ids → labels` |
| Quantization | symmetric int-N, group size 128, weights only; embeddings / LM head / RMSNorm kept in FP |
| Trainable | `A, B` (rank 32) + scale `s` |
| Steps / tokens / seq | 1000 steps × 512 tokens (512 k tokens); train seq 512, eval seq 1024 |
| LR grid | `lr_AB ∈ {1e-5, 1e-4, 1e-3, 1e-2}`, `lr_s ∈ {0, 1e-5}`, selected on the **validation** split only |
| Seeds | `{0,1,2,3,4}` |
| Preprocessing | WikiText-2 raw is *not* pre-tokenized: join lines, tokenize with the SmolLM tokenizer, concatenate and chunk to 512; a leading-BOS/eos convention identical across arms |

## 5. Training and evaluation compute

**Published** (Appendix B, Table 3, Table A8): all experiments on a **single NVIDIA A100 80 GB**.
Wall-clock for 10⁴ steps: 7B ≈ 2.25 days, 8B ≈ 2.75 days, 13B ≈ 4.5 days. Per 100 steps:
full-model QAT 3248 ± 7 s vs LR-QAT 1522 ± 5 s (4×8 gradient-accumulation configuration).
Peak memory (LLaMA-2 7B, Table 3): full-model QAT **62.2 GB** with checkpointing (**98.5 GB** without);
**LR-QAT 20.5 GB**. Total: 202 A100-GPU-days for all results, ≈750 including preliminary work.

**This reproduction**: **CLOUD-RUN** on the notebook substrate (`AGENTS.md` §2b) — local execution of
training is forbidden. Tier-1 pilot (SmolLM2-135M, 5 seeds): ≈1–2.5 cloud GPU-h for the R2 arms;
Tier-2 primary (TinyLlama-1.1B, 3 seeds): ≈3–6 cloud GPU-h for the R2 arms; analytical estimate,
replaced by measured GPU-hours from the first `run_manifest.json` (basis and envelope in
`docs/research/reproduction-plan.md` §4.2–§4.4). Peak host memory ≤ 3.5 GB locally for the fixture and
analysis blocks. No GPU is used locally. The merge claim is additionally measured **locally** as
**class 3** (byte-exact int4/int8 ONNX containers this project writes) and **class 4-CPU** (ONNX
Runtime 1.30.0 `MatMulNBits` executing those containers) — a kernel measurement on a self-serialized
artifact, not training (`docs/research/environment.md` §1 "Corrected on 2026-10-08"). Class 4-GPU and
class 5 remain unavailable until a cloud GPU run actually executes a supported low-bit kernel; no CPU
number may be compared with a published GPU latency.

## 6. Known deviations from the paper

See `docs/research/reproduction-plan.md` §7 for the machine-readable log format; the LR-QAT entries are
DEV-0001, DEV-0002, DEV-0003, DEV-0005, DEV-0006, DEV-0008, DEV-0009, DEV-0010. In summary:

1. **Scale**: 135 M vs 7B–13B (DEV-0001).
2. **Hardware and budget**: CPU-only, 10³ steps vs A100, 10⁴ steps (DEV-0002/0003).
3. **Dataset**: WikiText-2 train vs a SlimPajama subset; validation split is WikiText-2 validation, not
   the WikiPedia/Wiki40b English subset (DEV-0005).
4. **Rank**: 32 only; no rank ablation (DEV-0006).
5. **Learned vs frozen step size**: the paper only *searches* `lr_s ∈ {0, 1e-5}` and publishes no
   ablation; we run the learned-`s` arm as primary and treat `lr_s = 0` as an optional control
   (DEV-0008).
6. **Evaluation**: perplexity only; the paper's 6-task zero-shot average (lm-eval 0.4.2) is not
   reproduced (DEV-0009).
7. **Fixed-point downcast**: `Q4.4` is implemented, but the double-packing of two INT-b values into one
   INT8 (motivated by PyTorch lacking native INT4) is *not* implemented, because it is a storage
   optimisation with no effect on the fake-quantized forward.
8. **Merge identity tolerance**: the paper asserts "without any loss of accuracy" without a numeric
   tolerance; we predeclare exact integer equality plus a 1e-5 relative fp logit bound (DEV-0010).

## 7. Expected output (what the paper reports)

| Claim | Reference | Published value |
|---|---|---|
| PTQ < LR-QAT ≤ full QAT, LLaMA-2 7B W4 pc | Table 3 | full-model QAT (LSQ) WT2 **5.77 ± 0.02**, zero-shot 68.96 ± 0.29; LR-QAT **5.66 ± 0.00**, 69.72 ± 0.32 (FP16 reference 5.47 / 70.47) |
| LR-QAT vs PTQ, W4 pc / W4 g128, LLaMA-2 7B | Table 4 | RTN **6.14 / 5.78** → LR-QAT **5.66 / 5.59**; zero-shot 68.88 / 69.75 → 69.72 / 69.88 |
| LR-QAT vs PTQ, W3 pc, LLaMA-2 7B | Table 4 | RTN **26.73** → LR-QAT **6.13** (the collapse case) |
| W2, LLaMA-2 7B g128 | Table 4 | RTN 2.5e3 → LR-QAT **7.62** |
| Weight-activation 4-4-4, LLaMA-2 7B | Table 5 | RTN **18.98** → LR-QAT **8.46** |
| Memory: 7B trainable on one 24 GB GPU | §1, §6, Table 3 | **20.5 GB** vs 62.2 GB (full QAT, checkpointed) / 98.5 GB (un-checkpointed) |
| No inference overhead after merging | §3, Table 1 | adapters absorbed into a single b-bit integer matrix `W_Z`; "no extra overhead at inference time" |
| Downcast: `Q4.4` ≈ FP32 store, free | Table 2 | ΔPPL −0.01 (W4 pc), +0.01 (W3 pc) vs the FP32 row (absolute 5.69 / 6.21) |
| Learned-step-size ablation | **not reported** (only a search-space entry, Table B1) | `[UNKNOWN]` |

## 8. Observed

**`observed: not yet run (Milestone 3 pending)`**

No reproduction has been performed. No number in this card is a result of this project.
