# Method card — LoftQ (LoRA-fine-tuning-aware quantization)

Field list follows the assignment in the Milestone 3 ticket (wave 1). `references/method-cards/TEMPLATE.md`
was not present when this card was written (owned by another stream) — if the template fixes different
field names, this card must be re-synchronised.

---

## 1. Exact citation

- **Title**: *LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models*
- **Authors**: Yixiao Li, Yifan Yu, Chen Liang, Pengcheng He, Nikos Karampatziakis, Weizhu Chen, Tuo Zhao
- **arXiv**: `arXiv:2310.08659` (v1 2023-10-12; **v4 2023-11-28**, the version used here)
- **Venue**: **ICLR 2024** (proceedings listing verified: `proceedings.iclr.cc` paper hash
  `39ec972afab01e0d8ddc6834a9d12ac1`)
- **Upstream code**: `https://github.com/yxli2123/LoftQ`; checkpoints at `https://huggingface.co/LoftQ`

## 2. Upstream repo + pinned commit

| Field | Value |
|---|---|
| Repository | `yxli2123/LoftQ` |
| Pinned commit | `ae33fd4fd05fd4ba146555cd77c13d307eb4e9b3` (2024-06-11) |
| Verified by | `git ls-remote … HEAD` and `gh api repos/…/commits/main --jq .sha` — both agree (`upstream-notes.md` §1) |
| Licence (file read) | `LICENSE` — **MIT**, `Copyright (c) 2023 yxli2123`; GitHub SPDX classifier agrees (`MIT`) |
| Clone location | `%TEMP%/sq-upstream/LoftQ` — outside the working tree |
| **Structural finding** | The repository's **LLM path does not contain the algorithm**: `quantize_save.py:153-167` calls PEFT's `LoraConfig(init_lora_weights="loftq")`. The alternating loop lives in `peft/utils/loftq_utils.py` (**Apache-2.0**). The repository's own pure-PyTorch implementation is the encoder path: `glue/utils.py:89-100` (`quant_first_iter`) inside a `for i in range(args.num_iter)` loop (`:137-140`, `:164-165`), with fake quantizers in `glue/utils_qaunt.py`. |
| Entry points (reference only) | `python quantize_save.py --model_name_or_path <id> --bits 4 --iter 5 --rank 16 --save_dir …`; training `train_clm.py` (WikiText-2), `train_gsm8k.py`, `train_summarization.py`, `glue/run_glue.py`, `glue/run_qa.py`; evaluation `test_gsm8k.py` |

## 3. Reimplemented vs imported vs adapted

**Reimplemented** (formula below) — **not** imported and **not** adapted. MIT/Apache-2.0 would legally
permit copying; the project nevertheless chooses reimplementation because (a) the frozen M2 convention
`W ≈ B @ A` with `A: (r, in)`, `B: (out, r)` is the **transpose** of LoftQ's `L`/`R` naming, so a copy
would silently invert shapes, and (b) `src/spectraquant/**` must stay pure-torch/numpy with no
third-party quantized kernels (`design-m2-interfaces.md` §0).

**Imported, as an oracle only**: PEFT's `loftq_init` (Apache-2.0, pinned version in `uv.lock`) is used
as an **independent numerical reference** in a Milestone 3 gate — for `num_bits = 2` the PEFT path uses
its own pure-PyTorch NF lookup quantizer (`loftq_utils.py:196-198`), so it runs on CPU. It is never the
reported implementation (reproduction-plan §9, decision D7).

Formula to implement (paper §3, Algorithm 1; the code's own equivalent is `glue/utils.py:89-100`):

```
input:  W ∈ R^{d_out×d_in}, rank r, bits b, alternating steps T, quantizer q_b(·)
res_0 = W
for t = 0 … T−1:
    Q_t          = q_b(res_t)                      # quantize the residual      (paper Eq. 7)
    U,S,Vh       = svd(W − Q_t, full_matrices=False)   # exact rank-r projection (paper Eq. 8/9)
    L_t          = U[:, :r]·diag(sqrt(S[:r]))      # (out, r)
    R_t          = diag(sqrt(S[:r]))·Vh[:r, :]     # (r, in)
    res_{t+1}    = W − L_t·R_t
return Q_{T−1}, A = R_{T−1}, B = L_{T−1}           # then y = Q x + (α/r)·B(A(x))
```

Objective (paper Eq. 6): `min_{Q,A,B} ‖W − Q − A Bᵀ‖_F`. T=1 is the special case where `Q₁` is exactly
the QLoRA quantized weight and `(A₁,B₁)` is the SVD of the quantization residual `W − Q₁` (§3.2).
Convention conversion for this project: `A_ours = R_loftq` (`(r, in)`), `B_ours = L_loftq` (`(out, r)`).

**Not implemented**: LoftQ's bit-packing of the NF table (`uint8` LIFO packing) — it is a storage
optimisation with no effect on the fake-quantized forward; class-3 storage is reported separately by
`accounted_bytes`.

## 4. Model / data / preprocessing / seed / hyperparameters

**As published:**

| Item | Published value |
|---|---|
| Models | DeBERTaV3-base (GLUE/SQuAD/ANLI), BART-large (XSum/CNN-DailyMail), LLaMA-2 7B/13B (WikiText-2/GSM8K) |
| Bits | NF2 / NF4 (NormalFloat lookup tables, `T[i] = F⁻¹(i/(2^N−1))`, Eq. 2–3) and **uniform** int-N; the LLM path's 4-bit is bitsandbytes NF4; the sub-4-bit numbers (2/2.25/2.5/3) can only come from the fake path |
| Rank | DeBERTaV3 16 and 32; BART 8 and 16; LLaMA-2 **64** (Table 5) |
| Alternating steps T | **5** for all GLUE tasks (App. D.2); **1** for all BART experiments (App. E.1); `[UNKNOWN]` for the LLaMA-2 Table 5 results; swept as T ∈ {0,1,5,10} in Figure 3 |
| LoRA placement | weight matrices in MHA and FFN of all layers; embeddings also quantized for GLUE/SQuAD/ANLI |
| Alpha / dropout | **`[UNKNOWN]`** — the paper never states `lora_alpha` or LoRA dropout (grep-verified). The upstream LLM script sets `lora_alpha = 16` for 4-bit CausalLM, else `= rank` (`quantize_save.py:159`), and `lora_dropout = 0.1` at init |
| Finetune optimizer | AdamW (BART: Adam); LR grids: DeBERTa `{1e-5,5e-5,1e-4,5e-4}`, BART `{1e-5,…,4e-4}`, LLaMA-2 `{1e-5,…,4e-4}`; chosen LRs per task in Tables 12–17 |
| Epochs / batch | GLUE 5–60 epochs (task-dependent), batch 32 (SQuAD 16); BART CNN/DM 15, XSum 25 epochs, batch 64/32; LLaMA-2 2 epochs WikiText-2, 6 epochs GSM8K, batch 32/16 |
| Frozen / trained | `Q` frozen; only `A, B` trained; backbone frozen; bias kept in FP and trainable in the encoder path |
| Seeds | median over **four** seeds (DeBERTa, Tables 1/2/6/11) and **five** seeds (BART Table 3/4, LLaMA-2 Table 5); individual seed values not given |

**As planned for this reproduction** (`reproduction-plan.md` §3.1):

| Item | Value |
|---|---|
| Model | Tier-1 pilot `HuggingFaceTB/SmolLM2-135M` (LlamaForCausalLM, Apache-2.0); Tier-2 primary `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` (Apache-2.0, registry M1) |
| Data | `Salesforce/wikitext` `wikitext-2-raw-v1`, 512-token training blocks, 1024-token eval windows |
| Bits | **NF2-style codebook, block 64** for the primary arms (paper's 2-bit regime) + a uniform-int2 control |
| Rank | 16 for the model-level arms; 16 and 64 for the initialisation-level arms |
| T | 5 (primary) and 1 (mechanism control) |
| Alpha / dropout | `lora_alpha = 16` with `r = 16` (scaling 1.0), dropout 0 — declared explicitly because the paper is silent (DEV-0007) |
| Optimizer / LR | AdamW; `{1e-5, 5e-5, 1e-4, 3e-4}` selected on the validation split |
| Budget | 1000 steps × 512 tokens per arm, 5 seeds |
| Preprocessing | identical across arms (same tokenized stream, same block boundaries, same `Q`) |

## 5. Training and evaluation compute

**Published**: "All the experiments are conducted on **NVIDIA A100 GPUs**" (§4). **GPU count: `[UNKNOWN]`.
Wall-clock training time: `[UNKNOWN]`** — never reported. Memory footprint (Table 8): LLaMA-2-7B on
GSM8K at seq 384 / batch 1 → **15 GB**; 13B → **24 GB**. Initialisation cost (Table 9, measured on an
Intel Xeon E5-2650 v4 CPU, one matrix, T=5): 768×768 uniform **1 s**; 1024×1024 NF4 **1 s**;
4096×4096 NF4 **21 s**; 5120×5120 NF4 **43 s** — i.e. the initialisation is CPU-feasible by design.

**This reproduction**: **CLOUD-RUN** (`AGENTS.md` §2b) — Tier-1 pilot (SmolLM2-135M, 5 seeds, 4 arms)
≈1–2 cloud GPU-h; Tier-2 primary (TinyLlama-1.1B, 3 seeds, 4 arms) ≈2.5–5 cloud GPU-h; analytical
estimate, replaced by measured GPU-hours from the first manifest (`reproduction-plan.md` §4.2–§4.4).
The LoftQ initialisation itself is a local, pure-CPU float computation on extracted weight matrices
(minutes; the paper's own Table 9 measures 1 s for a 768×768 matrix and 21–43 s for 4096×4096 on a
2016-era Xeon). Storage is measured **locally** as **class 3** from byte-exact int4 ONNX containers this
project writes; the initialisation makes no kernel-backed or latency claim. Class 4-GPU/class 5 remain
unavailable until a cloud GPU run actually executes a supported low-bit kernel.

## 6. Known deviations from the paper

Full log format in `reproduction-plan.md` §7; LoftQ entries are DEV-0001, DEV-0002, DEV-0004, DEV-0006,
DEV-0007, DEV-0009. In summary:

1. **Scale**: 135 M vs 7B/13B (and the paper's encoder experiments, DeBERTaV3-base 184 M, are not
   reproduced because they require GLUE data and task metrics outside this milestone's budget).
2. **Quantizer**: our own NF-codebook/uniform fake quantizer (class 2) instead of bitsandbytes NF4; the
   paper's sub-4-bit LLM results already came from a fake path, so this is closer to the paper than the
   4-bit rows are.
3. **T for the model-level arms** is set to 5 by analogy with the GLUE setting; the paper does not state
   T for Table 5 (declared explicitly).
4. **Alpha and dropout are unknown in the paper**; we fix them and state them.
5. **No GLUE/SQuAD/ANLI/GSM8K/ROUGE reproduction**; perplexity only.
6. **Rank 64 is used only at the initialisation level**, not for model-level finetuning (cost).
7. **The published numbers are never compared against ours** (different tokenizer, model, and scale).

## 7. Expected output (what the paper reports)

| Claim | Reference | Published value |
|---|---|---|
| 2-bit LLaMA-2: QLoRA fails, LoftQ converges | Table 5 | QLoRA **N.A.** (no convergence) at 2/2.25/2.5-bit (7B); LoftQ **7.85** (7B) / **7.69** (13B) ppl at 2-bit |
| 4-bit LLaMA-2 WikiText-2 ppl | Table 5 | QLoRA 5.70 (7B) → LoftQ **5.24**; 13B 5.22 → **5.16** (16-bit LoRA reference 5.08 / 5.12) |
| 2-bit DeBERTaV3 GLUE/SQuAD (uniform) | Table 2 | MNLI-m 79.9 → **88.0**; SQuADv1.1 EM 71.6 → **85.2**; SST 86.9 → **94.7** |
| 2-bit DeBERTaV3 (NF2) | Table 1 | MNLI-m 75.4 → **84.7** (r16); SQuAD EM 61.5 → **81.5** |
| 4-bit BART ROUGE | Table 3 | XSum R-1 43.29 → **44.51** (NF4 r16); CNN/DM 43.42 → **43.96** |
| 2-bit BART ROUGE | Table 4 | QLoRA **N.A.**; LoftQ 40.81 / 42.52 R-1 at r16 |
| Mechanism: init discrepancy shrinks | Figure 2 | spectral / Frobenius norm of `W − (Q + BA)` is smaller for LoftQ than for the standard init, at 2-bit and 4-bit |
| Ablation: quantize-first beats SVD-first | Table 6 | MNLI-m 87.8 (SVD first) → **88.0** (quantization first) |
| Ablation: T sweep | Figure 3 | T ∈ {0,1,5,10}; "88.0 % MNLI-m using only 5 alternating steps and 21.14 ROUGE-2 using only 1 step" (§4.4) |
| 4-bit regime: LoftQ ≈ QLoRA | Table 11 | MNLI-m 89.9 vs 89.9 at 4-bit NF4 (the method's value is concentrated in the sub-4-bit regime) |

## 8. Observed

**`observed: not yet run (Milestone 3 pending)`**

No reproduction has been performed. No number in this card is a result of this project.
