# Method card — LQ-LoRA (low-rank plus quantized matrix decomposition)

Field list follows the assignment in the Milestone 3 ticket (wave 1). `references/method-cards/TEMPLATE.md`
was not present when this card was written (owned by another stream) — if the template fixes different
field names, this card must be re-synchronised.

**Milestone status: context only.** LQ-LoRA is not one of the two Milestone 3 reproduction targets
(those are LR-QAT and LoftQ, see `docs/research/reproduction-plan.md` §1). It is pinned, read, and
documented because (a) it is the closest published relative of this project's own research question
(joint low-rank + low-bit preparation with a *budgeted, per-layer allocation*), (b) its decomposition
loop is the same iteration as LoftQ's, and (c) its allocation step is the published antecedent of the
allocator this project builds in Milestone 6. Nothing from it is executed in Milestone 3.

---

## 1. Exact citation

- **Title**: *LQ-LoRA: Low-rank Plus Quantized Matrix Decomposition for Efficient Language Model Finetuning*
- **Authors**: Han Guo, Philip Greengard, Eric P. Xing, Yoon Kim
- **arXiv**: `arXiv:2311.12023` (v1 2023-11-20; **v4 2024-08-27**, the version used here)
- **Venue**: **ICLR 2024** (proceedings listing verified: `proceedings.iclr.cc` paper hash
  `d97a16cb83d74195b76e0bf1e85bf072`). Not verified as a spotlight.
- **Upstream code**: `https://github.com/HanGuo97/lq-lora`

## 2. Upstream repo + pinned commit

| Field | Value |
|---|---|
| Repository | `HanGuo97/lq-lora` |
| Pinned commit | `c2424b3adc27197815da1ac9e1304565168d824d` (2024-01-21) |
| Verified by | `git ls-remote … HEAD` and `gh api repos/…/commits/main --jq .sha` — both agree (`upstream-notes.md` §1) |
| Licence (file read) | `LICENSE` — **MIT**, `Copyright (c) 2024 Han Guo`; GitHub SPDX classifier agrees (`MIT`) |
| Clone location | `%TEMP%/sq-upstream/lq-lora` — outside the working tree |
| Entry points (reference only) | `run_clm.py` (quantize + QLoRA finetune), `run_lm_eval.py`, `run_mmlu_evaluation.py`, `run_legacy_evaluation.py`, `run_glue.py`, `run_oasst1.py`, `experiments/qlora.py` (legacy), drivers in `scripts/*.sh` |
| Core files (reference only) | `models/lq_utils.py:42-91` (alternating loop), `models/factorizations_utils.py:6-107` (SVD / weighted SVD), `models/quantization_utils.py:11,45-95,170-222,255-274` (NF codebook, blockwise absmax, storage model), `models/allocation_utils.py:33-81,135-146,183-281` (Fisher, 432-config grid, normalisation), `models/lq_utils.py:227-289` (ILP), `models/lora_utils.py:420-476` (LoRA merge identity) |
| Licence-relevant dependencies of that repo | `gurobipy` (**commercial**), `pytorch-quantization` (NVIDIA NGC terms), `bitsandbytes`, `ray[air]`, `peft==0.5.0`, `transformers==4.34.1` |

## 3. Reimplemented vs imported vs adapted

**Decision: reimplement the mathematics; import nothing from this repository; execute nothing from it in
Milestone 3.** MIT would permit copying with attribution, but the runtime is CUDA/ray/Gurobi-bound and
its packing path hardcodes `device="cuda"` (`models/packbits_utils.py:241-264`), so a port would drag in
a runtime that cannot run here. The equations are short and are already the same family as LoftQ's:

```
objective:            argmin_{Q,L1,L2} ‖W − (Q + L1·L2)‖_F            (paper Eq. 1)
alternating:          L1,L2 ← SVD_r(W − Q)                            (paper Eq. 2)
                      Q     ← Quantize(W − L1·L2, c)                  (paper Algorithm 2)
stopping:             break when ε_t > ε_{t−1},
                      ε_t = ‖W − (Q + L1L2)‖_F  (or ‖√F ⊙ (W − Q − L1L2)‖_F with the Fisher weights)
storage model:        storage(A,c) = sizeof(A)·(b0 + b1/B0 + b2/(B0·B1))     (paper Eq. 3)
allocation:           min Σ cost·x  s.t.  Σ weight·x ≤ budget,  Σ_qconfig x = 1 per layer   (binary ILP)
LoRA merge identity:  y = x Wᵀ + s·x Aᵀ Bᵀ = x (W + s·B A)ᵀ  ⇒  A = L2/√s,  B = L1/√s
```

Mapping onto this project: the ILP is implemented with `scipy.optimize.milp` (the repo's own
licence-free fallback, `models/lq_utils.py:253-263`) rather than Gurobi, and the *cost function must be
`spectraquant.quantization.accounting`-backed* per the frozen interface
(`design-m2-interfaces.md` §0.1, §4) — i.e. LQ-LoRA's storage model is a reference for the accounting
formula, not a second source of truth. The Fisher-weighted two-sided SVD is the published antecedent of
the `hessian_diag`/`fisher_diag` proxy required in `design-m2-interfaces.md` §3.

**Not implemented**: the int32 bit-packing runtime (`models/packbits_utils.py`,
`quantization_utils_2.py`), `ray`-distributed Fisher computation, the Gurobi backend, and the
`/export/share*` cluster paths.

## 4. Model / data / preprocessing / seed / hyperparameters

| Item | Published value |
|---|---|
| Models | RoBERTa-Large (GLUE), LLaMA-2 7B and 70B (continual LM on C4, instruction tuning on OpenAssistant, PTQ/compression) |
| Rank | **64** in the main experiments; rank ablation {32, 64, 128} (Table 5) |
| Quantizer | NF-style codebook with a two-level (double) quantization of the absmax values; configuration grid `b0 ∈ {2,3,4}` (NF bits), `b1 ∈ {2,3,4}` (absmax bits), `b2 ∈ {bf16,fp16,fp32}` (absmax cast), `B0 ∈ {16,32,64}`, `B1 ∈ {16,64,256}` ⇒ **|C| = 243** (the code's grid is 432 combinations, `allocation_utils.py:135-146`) |
| Bit budgets | target 2.50 / 2.75 / 3.00 / 3.25 / 3.50 / 3.75 / 4.00 bits/param; references: NF-4 QLoRA = 4.127, NF-3 QLoRA = 3.127, GPTQ-LoRA 3-bit = 3.148 |
| Training | rank 64, **no LoRA dropout**, default LR **2e-5** with exceptions; continual LM = half an epoch of one C4 partition, sequence length 1024 for train and eval; instruction tuning follows Dettmers et al. hyperparameters |
| Fisher | diagonal empirical Fisher over **10 000 C4 samples** at sequence length 1024 (`allocation_utils.py:33-81`, batch size must be 1); RoBERTa uses an MLM objective on C4 |
| Frozen / trained | `Q` frozen; only `L1, L2` (the low-rank factors) finetuned |
| LoRA alpha | **`[UNKNOWN]`** — never stated. Upstream code uses `lora_alpha=16` (`models/lora_utils.py:262-274`) |
| Optimizer / batch / seeds | optimizer name `[UNKNOWN]` for finetuning; C4 batch size `[UNKNOWN]`; **seeds `[UNKNOWN]`** |
| GLUE specifics | LR/epochs from Hu et al. for the QLoRA baseline, 5 epochs for MNLI and QQP "due to their sizes" |

## 5. Training and evaluation compute

**Published**: ILP pre-computation "takes a few hours when parallelized across **four A100 GPUs** for
LLaMA-2-7B" (§3.2); full forward/backward on the sub-3-bit 70B model runs on a **single 80 GB GPU** at
batch 2 × seq 2048 (§4.3); Appendix A benchmarks runtimes on A100 and A6000. Memory (Figure 3):
LLaMA-2 7B / 70B at 16 bits need **14 GB / 139 GB**; the 2.75-bit 70B needs **27 GB**. No total
GPU-hour figure is reported. Initialisation cost per matrix is "a few seconds on a modern GPU for a
4096×4096 matrix" (§3.1).

**This project**: nothing is executed at Milestone 3. If the allocator (Milestone 6) ever wants this
method as a baseline, the training cost is cloud GPU-hours of the same order as LoftQ's
(`reproduction-plan.md` §4.2), and the expensive part is the Fisher computation (10 000
forward/backward samples at sequence length 1024). A *local* CPU estimate for that Fisher step, at the
measured local throughput of `reproduction-plan.md` §4.1, is ≈1–2 CPU-hours at 135 M and ≈10–20
CPU-hours at 1.1 B — which is why any Fisher-weighted variant would have to run on the cloud substrate
too. Measurement classes available: 1, 2, 3, 4-CPU (4-CPU only locally, on our own containers);
class 4-GPU/class 5 remain unavailable until a cloud GPU run executes a supported kernel.

## 6. Known deviations from the paper (as designed, not as run)

1. **Not reproduced at Milestone 3** — the milestone targets LR-QAT and LoftQ only.
2. Any future use would substitute `scipy.optimize.milp` for Gurobi (the repo's own fallback path), which
   changes solver provenance but not the problem statement; the solver name is a required field of the
   frozen `Allocation` dataclass (`design-m2-interfaces.md` §4).
3. Any future use would compute the Fisher diagonal with this project's own sampler (documented sample
   count and dtype, per `design-m2-interfaces.md` §0.4) rather than the repo's 10 000-sample
   ray-distributed job, and would therefore not be bit-comparable with the paper.
4. Scale, hardware and dataset deviations are the same class as DEV-0001/0002/0005 in
   `reproduction-plan.md` §7, should this method ever be run.

## 7. Expected output (what the paper reports)

| Claim | Reference | Published value |
|---|---|---|
| Sub-3-bit compression quality (PTQ setting) | Table 3 | LQ-LoRA 2.75-bit (Fisher, rank 64): C4 7B **7.60** / 70B **5.88**; WikiText 7B **5.67** / 70B **3.65** at 2.95 / 2.85 effective bits (vs RTN 3-bit g128: 8.40 / 6.02, 6.66 / 3.97) |
| Parity with higher-bit baselines | §4.1 | "3.5-bit (Fisher) LQ-LoRA is generally comparable to NF-4-bit QLoRA (4.127 bits/param); 2.75-bit LQ-LoRA is competitive with NF-3-bit QLoRA (3.127 bits/param)" |
| Allocation matters at low bits | Table 6 | QLoRA + ILP at 2.50 bits **collapses** (C4 7B 2996.3) while LQ-LoRA at 2.50 reaches **10.00** — the ILP alone is not enough; the joint decomposition is what survives |
| GLUE (RoBERTa-Large) | Table 2 | Full FT 88.5; QLoRA (ILP) 2.5-bit 75.4 → LQ-LoRA 2.5-bit **85.7**, 3.25-bit **88.1** |
| Zero/few-shot (Open LLM Leaderboard) | Table 4 | LLaMA-2-7B at 2.95 bits: average **48.0** vs uncompressed 51.0; 70B at 2.85 bits: **65.3** vs 67.7; degradation concentrated on GSM8K/ARC |
| Rank sensitivity | Table 5 | LQ-LoRA improves with rank (C4 7B: 8.02 → 7.84 from rank 32 → 128) whereas "QLoRA is insensitive to the LoRA rank" (§4.3) |
| Decomposition error < quantization error | Figure 1 (right) | the low-rank component reduces the residual below the pure quantization error |

## 8. Observed

**`observed: not yet run (Milestone 3 pending)`** — and not scheduled for Milestone 3 at all; see the
milestone-status note at the top. No reproduction has been performed. No number in this card is a
result of this project.
