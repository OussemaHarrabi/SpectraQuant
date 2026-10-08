# Upstream notes — pinned baselines, what their code actually does, and what is CPU-blocked

Scope: this document is the **code-level** companion to `docs/research/reproduction-plan.md` and to the
method cards in `references/method-cards/`. It records, for each pinned upstream repository, (a) the
verified commit, (b) the license actually read, (c) the entry points and the pipeline the code really
implements, (d) the exact dependencies and the CUDA/bitsandbytes assumptions, (e) which parts import
and run on this CPU-only Windows workstation, with `file:line` for every blocker, and (f) which pure
mathematics may be **independently reimplemented** versus code that must not be copied.

Status: **planning evidence only.** No reproduction has been run. Nothing in this document is a result
claim. Milestone 3 results come only after the gate (see `docs/research/reproduction-plan.md` §8).

Author: agent `ReproPlan` (wave 1, stream: research docs — planning). Owned paths: this file,
`docs/research/reproduction-plan.md`, `references/method-cards/{lr-qat,loftq,lq-lora,torchao-qat}.md`.

---

## 0. Where the clones live (license rule compliance)

**Upstream code is never cloned, copied or vendored inside this repository.** All four repositories were
shallow-cloned outside the working tree, under the OS temp directory:

```
%TEMP%\sq-upstream\LR-QAT      (C:\Users\oussa\AppData\Local\Temp\sq-upstream\LR-QAT)
%TEMP%\sq-upstream\LoftQ
%TEMP%\sq-upstream\lq-lora
%TEMP%\sq-upstream\ao
```

These paths are machine-local, are not part of the repository, and are ignored by `.gitignore`
(they are outside the tree entirely). `src/spectraquant/**` contains no third-party source. Reproduce
the clones with:

```bash
mkdir -p "$TEMP/sq-upstream" && cd "$TEMP/sq-upstream"
for r in Qualcomm-AI-research/LR-QAT yxli2123/LoftQ HanGuo97/lq-lora pytorch/ao; do
  d=$(basename $r); [ -d "$d" ] || git clone --depth 1 "https://github.com/$r.git" "$d"
done
```

Clone cost observed: 24 s wall, 28.4 MB total on disk (LR-QAT 2.6 MB, LoftQ 1.2 MB, lq-lora 0.6 MB,
pytorch/ao 24 MB).

### 0.1 Repository cleanliness check (raw output)

```
$ cd C:/Users/oussa/oussema/SpectraQuant && git status --short
 M docs/coordination/status.md
?? LICENSE
?? docs/coordination/design-m2-interfaces.md
?? pyproject.toml
?? src/
?? uv.lock
```

No upstream path (`LR-QAT/`, `LoftQ/`, `lq-lora/`, `ao/`) appears. The untracked entries are the
concurrent wave-1 scaffold stream's own files (owned by stream B), not third-party code. Verification
was re-run after all clones were created (see §9).

---

## 1. Pin verification (raw command output)

### 1.1 Remote HEADs (`git ls-remote`)

```
$ git ls-remote https://github.com/<repo>.git HEAD
8795afe054cf951b714299e01083a1b354721829	HEAD      # Qualcomm-AI-research/LR-QAT
ae33fd4fd05fd4ba146555cd77c13d307eb4e9b3	HEAD      # yxli2123/LoftQ
c2424b3adc27197815da1ac9e1304565168d824d	HEAD      # HanGuo97/lq-lora
cff77b46ef85ba8472b4a286598073d6c01e18e5	HEAD      # pytorch/ao
```

### 1.2 Local clone HEADs equal the remote HEADs (`git rev-parse` / `git log -1`)

```
$ git -C <clone> rev-parse HEAD ; git -C <clone> log -1 --format=%cI
LR-QAT	8795afe054cf951b714299e01083a1b354721829	2024-11-05T02:37:35-08:00
LoftQ	ae33fd4fd05fd4ba146555cd77c13d307eb4e9b3	2024-06-11T09:24:24-07:00
lq-lora	c2424b3adc27197815da1ac9e1304565168d824d	2024-01-21T22:24:37-05:00
ao	cff77b46ef85ba8472b4a286598073d6c01e18e5	2026-10-07T01:25:05Z
```

Independent corroboration via the GitHub API (produced by a parallel wave-1 stream and recorded in
`references/upstream-verification.log`, not by this document):

```
$ gh api repos/Qualcomm-AI-research/LR-QAT/commits/main --jq .sha   → 8795afe054cf951b714299e01083a1b354721829
$ gh api repos/yxli2123/LoftQ/commits/main --jq .sha               → ae33fd4fd05fd4ba146555cd77c13d307eb4e9b3
$ gh api repos/HanGuo97/lq-lora/commits/main --jq .sha             → c2424b3adc27197815da1ac9e1304565168d824d
$ gh api repos/pytorch/ao/commits/main --jq .sha                   → cff77b46ef85ba8472b4a286598073d6c01e18e5
```

Two independent transports (HTTPS git protocol and the REST API) agree on all four SHAs.

### 1.3 Pinned revisions for the record

| Repo | Pinned commit (full SHA) | Commit date (UTC offset as printed) | Pinned as |
|---|---|---|---|
| `Qualcomm-AI-research/LR-QAT` | `8795afe054cf951b714299e01083a1b354721829` | 2024-11-05 | default-branch tip at pin time |
| `yxli2123/LoftQ` | `ae33fd4fd05fd4ba146555cd77c13d307eb4e9b3` | 2024-06-11 | default-branch tip at pin time |
| `HanGuo97/lq-lora` | `c2424b3adc27197815da1ac9e1304565168d824d` | 2024-01-21 | default-branch tip at pin time |
| `pytorch/ao` | `cff77b46ef85ba8472b4a286598073d6c01e18e5` | 2026-10-07 | default-branch tip at pin time; `version.txt` = `0.19.0` (unreleased dev line) |

Caveat recorded honestly: the clones are `--depth 1`, so **no tags are present locally**. For
`pytorch/ao` the PyPI metadata was read live (`https://pypi.org/pypi/torchao/json`) and the latest
**released** version is `0.18.0`; the pinned commit is *after* that release. Method cards therefore pin
the SHA, not the version string.

---

## 2. Licenses (the files actually read)

```
$ head -3 <clone>/LICENSE
--- LR-QAT
Copyright (c) 2024 Qualcomm Technologies, Inc.

All rights reserved.
--- LoftQ
MIT License

Copyright (c) 2023 yxli2123
--- lq-lora
MIT License

Copyright (c) 2024 Han Guo
--- ao
Copyright 2023 Meta
All contributions by Arm:
Copyright (c) 2024-2026 Arm Limited and/or its affiliates
```

| Repo | License text read (path) | Conclusion from the file | GitHub-detected SPDX (independent) | Reuse implication for this project |
|---|---|---|---|---|
| LR-QAT | `LR-QAT/LICENSE` (26 lines, full text read) | **BSD-3-Clause-style, Qualcomm-modified**: 3 redistribution clauses (source notice, binary notice, no-endorsement) + standard warranty disclaimer, plus one added sentence: *"NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE."* Not a verbatim SPDX BSD-3-Clause text; the addition is a patent non-grant. | `BSD-3-Clause-Clear` | **Reference-only. Do not copy source.** Reimplement the published equations (LSQ-style learned step size, STE uniform quantization, low-rank-on-quantized-weights) independently from the papers. No patent grant ⇒ avoid any implementation that could read as practising Qualcomm patent claims; this is a documented risk, not a legal opinion. |
| LoftQ | `LoftQ/LICENSE` (MIT, `Copyright (c) 2023 yxli2123`) | MIT | `MIT` | Copying is legally permitted with attribution; **this project nevertheless chooses reference-only** (see §5.2) because the M2 factor convention is frozen independently and the repo's LLM path delegates to `peft` anyway. |
| lq-lora | `lq-lora/LICENSE` (MIT, `Copyright (c) 2024 Han Guo`) | MIT | `MIT` | Same: permitted, but reference-only by project choice. Note `gurobipy` (commercial licence) and `pytorch_quantization` (NVIDIA NGC wheel) are *dependencies* of that repo and carry their own terms. |
| pytorch/ao | `ao/LICENSE` (BSD-3-Clause, `Copyright 2023 Meta` + Arm additions) and `ao/CITATION.cff` (`license: "BSD-3-Clause"`) | BSD-3-Clause | `NOASSERTION` (GitHub cannot classify the multi-holder text) | **Depend on the published wheel**, do not vendor source. Redistribution-as-dependency needs no extra notice obligation; vendoring would require shipping the notice and would fork numerics that upstream tests define. |

Note the discrepancy worth recording: GitHub reports `NOASSERTION` for `pytorch/ao`, while the file and
`CITATION.cff` both say BSD-3-Clause. The file-read conclusion governs; the automated classification is
recorded as a caveat.

Dependency licences that matter for the *plan* (not for copying): `peft` Apache-2.0 (it contains the
only real LoftQ LLM implementation), `bitsandbytes` MIT (CUDA-only binaries), `gurobipy` commercial,
`pytorch-quantization` NVIDIA NGC terms, `ray` Apache-2.0, `lm-evaluation-harness` MIT.

---

## 3. Per-repository findings

### 3.1 `Qualcomm-AI-research/LR-QAT` @ `8795afe0`

**Shape of the repo.** A single Click CLI (`clm_main.py`) with four sub-commands and a self-contained
quantization library. No `bitsandbytes`, no `peft`, no `deepspeed`, no `torch.compile`, no
flash-attention/Triton/xformers anywhere in-tree (repo-wide grep). This is unusually good news: the
quantization math is the authors' own pure-PyTorch code.

**Entry points** (`README.md`; `clm_main.py:1008/1022/1031/1042`):

| Command | Purpose |
|---|---|
| `train-baseline` | plain FP continued training |
| `train-quantized` | QAT; **PTQ (RTN) baseline = the same script with `--max-train-steps 0 --learning-rate 0 --lr-ab 0`** |
| `validate-baseline` | eval only |
| `validate-quantized` | PTQ (`ptq_main`, `utils/ptq_utils.py:17`) then eval |

Launched through `accelerate launch --config_file accelerate_configs/1gpu_no_mp.yaml clm_main.py …`.

**What the pipeline actually does.**

1. Data: `load_and_tokenize_datasets` (`utils/huggingface_utils.py:~330-566`) → tokenize, `group_texts`
   concatenation, chunk to `--block-size` (`:427-443`); `slimpajama_wiki` needs a pre-tokenized arrow
   dataset path.
2. Quantize the model: `get_quant_model` (`utils/quant_utils.py:88-113`) recursively replaces
   `nn.Linear/Embedding/LayerNorm` with hijacker subclasses (`quantization/autoquant_utils.py:38-58`),
   producing `QuantizedLlamaForCausalLM` / `QuantizedMistralForCausalLM`.
3. Ranges: `--learn-ranges` promotes the per-tensor/per-channel step size `_delta` to an `nn.Parameter`
   (LSQ-style, `quantization/quantizers/uniform_quantizers.py:264-268`, symmetric variant `:449-452`;
   gradient scaling `1/sqrt(Qp·n)` at `:182-195`); otherwise ranges come from a range estimator
   (`range_estimators.py`) or are fixed.
4. Adapters: LoRA `A (out,r)` and `Bt (r,in)` are attached **inside** each quantized Linear, the base
   weight is frozen (`utils/qat_utils.py:199-256`), `lora_scaling = lora_alpha/r` (`:226`); targets are
   `q_proj,k_proj,v_proj,o_proj` and `gate_proj,up_proj,down_proj` (`:171-193`).
5. **The LR-QAT core** (`quantization/hijacker.py`): the adapter product is added *inside the rounding
   non-linearity*:
   - `lr_qat_int` (`_apply_lora_qat_w_int`, `:101-125`): `C = (A @ Bt) * lora_scaling` (`:111`);
     `W_int_hat = clamp(W_int + round_ste(C), q_min, q_max)` (`:112-113`); `W_hat = scale * W_int_hat`
     (`:120`). Requires frozen integer weights (raises otherwise, `:206-211`).
   - φ-variants (`_apply_lora_qat_w_phi`, `:128-153`): `Z = Phi_0 + C` (`:142`);
     `W_int = clamp(round_ste(Z), q_min, q_max)` (`:143`); `W_hat = s * W_int` (`:150`), where
     `Phi_0` is the downcast fp weight (`lr_qat_fp16` `:256`, `lr_qat_bf16` `:254`,
     `lr_qat_fixed_point8` `:258`).
   - Down/upcast for fixed-point 8 (`:156-180`): `round(clamp(x,q_min,q_max) · 2^(8-b)).to(int8)` down,
     `x / 2^(8-b)` up.
6. Merge after training: `get_params` (`:293-380`) folds the adapter into the integer weights
   (`clip(W_int + round(C))`, or `round(φ0 + C)` for φ-variants, `:335-360`) and registers `weight`
   + `scale` buffers ⇒ **no inference-time overhead**, matching the paper's claim.
7. Optimizer split (`clm_main.py:72-251`, `utils/qat_utils.py:48-87`): LoRA params at `--lr-ab`, scale
   params at `--scales-lr`, base weights frozen.

**Dependencies actually assumed.** `torch==1.13.1+cu117` (`README.md`; `docker/Dockerfile:24`, base
image `nvcr.io/nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04`, `docker/Dockerfile:1`);
`transformers~=4.40.2` (`docker/requirements.txt:12`); `accelerate~=0.26.0` (`:1`);
`lm-eval~=0.4.2` (`:5`, not commit-pinned); `datasets~=2.19.0`, `numpy~=1.24.2`, `scipy~=1.10.1`,
`click~=8.1.7`, `sentencepiece==0.2.0`. Accelerate configs are GPU-targeted
(`use_cpu: false`, `mixed_precision: bf16|fp16|no`).

**CPU-only Windows feasibility.**

| Part | Status | Evidence |
|---|---|---|
| Quantizer + hijacker math (round/clamp/STE/merge) | **Runs on CPU** | pure `torch.round/clamp/matmul`; the authors even added CPU bf16/fp16 fallbacks: `uniform_quantizers.py:96-99` ("clamp fails due to missing implementation for bf16/fp16 on cpu"), `:141-148`, `:332-339` |
| `quantization/*`, `utils/utils.py`, `utils/enums.py` | **Import-safe** (torch/numpy/scipy/stdlib only) | module-level imports inspected |
| `models/quantized_llama.py`, `utils/quant_utils.py`, `utils/huggingface_utils.py` | Import-safe but need `transformers` | `models/quantized_llama.py:10-11` |
| Stock `clm_main.py` run | **Hard-blocked** | `get_and_log_cuda_memory` calls `torch.cuda.memory_allocated/reserved/max_memory_reserved` unconditionally (`utils/utils.py:188-190`), invoked at `clm_main.py:357,377,573,737,963` and `utils/ptq_utils.py:60` ⇒ `AssertionError: Torch not compiled with CUDA enabled` |
| Stock `clm_main.py` run (Windows) | **Hard-blocked (second, independent cause)** | `Stopwatch` binds `time.clock` on win32 (`utils/utils.py:57-59`); `time.clock` was removed in Python ≥3.8 ⇒ AttributeError; `Stopwatch` is instantiated at `clm_main.py:679` |
| Accelerate configs | Blocked | `use_cpu: false` in all three `accelerate_configs/*.yaml` |
| Docker path | Blocked | CUDA 11.7 base image (`docker/Dockerfile:1`) |
| `torch==1.13.1+cu117` pin | Unusable here | CUDA wheel; project env has `torch 2.14.1+cpu` |

**What may be reimplemented independently** (published equations, not this repo's expression of them):
affine uniform quantization with STE; symmetric scale `absmax/int_max` and asymmetric
`delta=(max-min)/int_max`, `zero=-min/delta`; per-token dynamic activation quantization; LSQ gradient
scaling; block-wise (group) reshaping; the LR-QAT update `Ŵ = s·clip(round(φ0 + α/r·A·Bᵀ))`; the
fixed-point-8 down/upcast; the alternating SVD+quantize adapter initialisation; MSE/LP-norm range
search. **What must not be copied:** every file of `quantization/`, `models/`, `utils/`, `lmeval/`,
`clm_main.py` (Qualcomm licence, no patent grant).

**Evaluation.** In-run metric is perplexity `exp(mean CE loss)`
(`utils/huggingface_utils.py:629-635`; printed at `clm_main.py:703`), plus optional extra-dataset
perplexities (`perplexity_wikitext`, `clm_main.py:280-300`). The lm-eval path
(`lmeval/evaluate.py:23-29`) flattens every key lm-eval returns as `<task>__<metric>`
(`clm_main.py:897-901`), with `DEFAULT_TASKS = boolq, piqa, winogrande, hellaswag, arc_easy,
arc_challenge` (`lmeval/tasks.py:2-9`) ⇒ concrete keys are `acc` / `acc_norm` per task.
`word_perplexity`/`byte_perplexity` are **never produced** by this harness path.

---

### 3.2 `yxli2123/LoftQ` @ `ae33fd4f`

**Shape of the repo — and the key structural finding.** There are two disjoint code paths:

1. **LLM path (the one used for the headline 2/2.25/2.5/3-bit results): a thin driver.** `quantize_save.py`
   does not implement LoftQ; it calls PEFT: `LoraConfig(..., init_lora_weights="loftq", loftq_config=…)`
   (`quantize_save.py:153-164`) and `get_peft_model` (`:167`). **The alternating SVD+quantization loop
   lives in PEFT** (`peft/utils/loftq_utils.py`, Apache-2.0), not in this repository. A grep for
   `loftq_initialization|find_low_rank` in the LoftQ clone returns nothing.
2. **Encoder/GLUE path: an in-repo pure-PyTorch reimplementation** (`glue/utils.py`, `glue/utils_qaunt.py`),
   CPU-clean, with its own fake quantizers.

This changes the pinning strategy: for the *algorithm* the authoritative LLM code is PEFT's
`loftq_init`; for an *inspectable, dependency-free reference* the LoftQ repo's `glue/utils.py` is the
better artifact. Both are recorded.

**Entry points.** `python quantize_save.py --model_name_or_path <HF_ID> --bits 4 --iter 5 --rank 16
--save_dir …` (`quantize_save.py:81-121`; `--iter` = "the alternating steps in LoftQ", default 1).
Training: `train_clm.py` (WikiText-2), `train_gsm8k.py`, `train_summarization.py`, `glue/run_glue.py`,
`glue/run_qa.py`; evaluation `test_gsm8k.py`. Scripts in `scripts/*.sh` fix seed 11 (LLM) / 0 (GLUE).

**The algorithm** (PEFT v0.9.0 `loftq_utils.py:181-225`, quoted structure): with `res = W`,
repeat `num_iter` times: `Q = dequant(quant_b(res))` (bitsandbytes NF4 when 4-bit+CUDA available,
otherwise a pure-PyTorch NF lookup quantizer, `:196-214`); `res = W − Q` (`:216`);
`(L,R) = low_rank_decomposition(res, r)` via `torch.linalg.svd(res, full_matrices=False)` with a
symmetric `sqrt(S)` split (`:174-177`); `res = W − L·R` (`:221`). Return `Q`, `lora_A = R`,
`lora_B = L` (`:223`). The in-repo GLUE equivalent is `quant_first_iter` (`glue/utils.py:89-100`)
inside a `for i in range(args.num_iter)` loop (`:137-140`, `:164-165`).

**Adapter semantics.** `lora_A = R (r,in)`, `lora_B = L (out,r)`; forward
`y = W_q x + (α/r)·B(A(x))`; `lora_alpha = 16` for 4-bit CausalLM else `= rank`
(`quantize_save.py:159`). The GLUE path uses scaling ≡ 1 and no LoRA dropout. The QLoRA comparison arm
in the same module is `lora_A` kaiming-uniform + `lora_B` zeros (`glue/utils.py:78-86`) — exactly the
"Gaussian + 0" baseline of the README table.

**Dependencies.** `requirements.txt` has **zero version pins and does not even list torch**;
`transformers`, `accelerate`, `peft`, `evaluate`, `bitsandbytes`, `deepspeed` (imported nowhere),
`scipy`, `einops`, … Exact versions are `[UNKNOWN]` from the repo; the API surface
(`LoftQConfig`, `init_lora_weights="loftq"`) matches peft ≈0.7–0.9 / transformers ≈4.36–4.40
`[INFERENCE]`.

**CPU-only Windows feasibility.**

| Part | Status | Evidence |
|---|---|---|
| Alternating projection + SVD + the repo's own fake quantizers | **Runs on CPU, unmodified** | `glue/utils.py:1-6` imports only torch/math/random/`utils_qaunt`; `torch.linalg.svd` (`:25`); `quant_uniform` uses only `torch.where/min/max/round/div` (`utils_qaunt.py:73-84`); `quant_nf4_block` uses `abs/max/argmin` + `torch.vmap` (`:52-70`); NF table from `scipy.stats.norm.ppf` (`:16-35`) |
| `torch.quantile` | Not used at all in this repo (0 grep hits); clipping is `mean ± 2·std` (`utils_qaunt.py:7-8`) | removes a common CPU performance trap |
| LoftQ init via PEFT | **Hard-blocked on CPU** | peft v0.9.0 hard-codes `compute_device = "cuda"` for the 4-bit branch (`loftq_utils.py:200`) and calls `bnb.nn.Params4bit`/`bnb.functional.dequantize_4bit` (`:207-211`); current peft raises `"bitsandbytes is not available, please install it to use LoftQ."` |
| Any training/eval script | **Blocked** | `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", …)` (`train_clm.py:474-479`, `train_gsm8k.py:213-218`, `test_gsm8k.py:110-115`), `device_map="auto"` (`quantize_save.py:135`), `.to('cuda')` (`test_gsm8k.py:159`), `torch.device(0)` (`glue/run_glue.py:75`), `torch.cuda.amp.GradScaler` (`:496`), `torch.autocast(device_type='cuda')` (`:592,624,665`) |
| README statement | Confirms the block | `README.md:19-23`: bitsandbytes "only support CUDA >= 11.0 and does not support CPU" |

**Reimplementable** (formula in §5.1 of the method card): `Q_t = fake_quant_intN(res_t)`,
`(L,R) = svd_r(W − Q_t)` with `sqrt(S)` split, `res_{t+1} = W − L·R`, return `(Q, A=R, B=L)`.
**Do not vendor:** nothing needs to be copied; MIT permits it but the M2 design note freezes an
independent convention (`W ≈ B @ A`, `A:(r,in)`, `B:(out,r)`, `docs/coordination/design-m2-interfaces.md`
§0.2) which is the *transpose* of LoftQ's `L/R` naming — a blind copy would silently invert shapes.

**Evaluation / reported numbers** (README): WikiText-2 perplexity for LoftQ vs full-precision backbone —
7B: 5.08 / 5.24 / 5.63 / 5.78 / 6.13 / 7.85 and 13B: 5.12 / 5.16 / 5.13 / 5.22 / 5.45 / 7.69 at
bits 16 / 4 / 3 / 2.5 / 2.25 / 2 (`README.md:175-183`). GLUE uses one metric per task
(`glue/run_glue.py:64-72`); GSM8K uses sampled generation + exact numeric match
(`test_gsm8k.py:205-259`); summarization uses ROUGE-1/2/L (`train_summarization.py:725`).

**Defects found while reading** (relevant because a port must not inherit them):
`quantize_save.py:30` imports `save_open` but calls `safe_open` at `:187` (NameError on modern
safetensors); `test_gsm8k.py` leaves `tokenizer` undefined on the `--full_precision` branch;
`glue/train_glue.sh:1` has `TASK_NAME=mnl9` (typo, rejected by argparse `choices`);
`glue/utils_qaunt.py:68` mutates its input via in-place `Tensor.resize` and requires
`numel % block_size == 0` (peft's equivalent raises otherwise, `loftq_utils.py:111-115`).

---

### 3.3 `HanGuo97/lq-lora` @ `c2424b3a`

**Entry points.** `run_clm.py` (quantize + QLoRA finetune), `run_lm_eval.py`,
`run_mmlu_evaluation.py`, `run_legacy_evaluation.py`, `run_glue.py`, `run_oasst1.py`,
`experiments/qlora.py` (legacy), drivers in `scripts/*.sh`. Example
(`scripts/c4_lora.sh:124-152`): `python run_clm.py --model_name_or_path … --dataset_name c4
--block_size 1024 --bf16 True --num_train_epochs 0.5 --per_device_train_batch_size 4
--gradient_accumulation_steps 32 --learning_rate 2e-5 --lora_num_ranks 64 --lora_dropout 0.0
--lora_model_name "llama-2-7b/lpq-64/None,budget=2.75" --lora_config lora-lpq`; 70B uses
`torchrun --nproc_per_node=4`.

**Pipeline actually implemented** (the README does *not* enumerate it; this is read from the code):

1. **Truncated / weighted SVD** — `models/factorizations_utils.py:6-29`
   (`torch.linalg.svd` or `torch.svd_lowrank`, `L1 = U·√S`, `L2 = √S·Vᵀ`, truncated) and the
   Fisher-weighted two-sided variant `:32-107` (`W1 = sqrt(mean(W,1))`, `W2 = sqrt(mean(W,0))`,
   factorize `W1·A·W2`, un-weight the factors).
2. **Alternating low-rank + quantized decomposition** — `models/lq_utils.py:42-91`
   ("Applying Robust PCA"): repeat {`(L1,L2) = svd(A − Q)`; `Q = quantize(A − L1·L2)`;
   `A_ = L1·L2 + Q`; break when the Frobenius error increases}.
3. **Quantization** — two-level blockwise NF: `create_normal_float_scheme` reproduces the NF4 codebook
   (`NF4_OFFSET = 0.9677083`, `scipy.special.erfinv`, quantiles via `Normal.icdf`, `torch.bucketize`,
   `models/quantization_utils.py:11,45-95`), `blockwise_absmax` (`:170-222`), storage model
   `bits = b + b0/B0 + b1/(B0·B1)` (`:255-274`). A second module does real packing into int32
   (`quantization_utils_2.py:445-593`) via `packbits_utils.py`.
4. **Fisher + ILP allocation** — `compute_empirical_Fisher_LLaMA`
   (`models/allocation_utils.py:33-81`, `fisher += grad²` over ~10 000 C4 samples, batch size must be 1);
   432-configuration grid (`:135-146`: `b,b0∈{2,3,4,8}`, `b1∈{bf16,fp16,fp32}`, `B0∈{16,32,64}`,
   `B1∈{16,64,256}`); ILP in `models/lq_utils.py:227-289` — binary x per (layer, qconfig),
   minimise `Σ cost·x` s.t. `Σ weight·x ≤ budget` and exactly one config per layer; Gurobi backend
   (`:268-289`) or **`scipy.optimize.milp` fallback** (`:253-263`).
5. **QLoRA finetune on the decomposed base** — `models/lora_utils.py:420-476`:
   `replace_weight_(module, Q)` and `lora_A = L2/√s`, `lora_B = L1/√s` (`:455-469`), with
   `LoraConfig(r=64, lora_alpha=16, target_modules=[q,k,v,o,gate,down,up]_proj)` (`:262-274`).

**Dependencies.** `peft==0.5.0`, `transformers==4.34.1` (`requirements.txt`), `ray[air]`, `gurobipy`,
`bitsandbytes`, `datasets`, `evaluate`, `accelerate`, `jaxtyping`, `wandb`, `optree>=0.9.1`.
`scripts/setup.sh` additionally installs `auto-gptq==0.4.2` (cu118 index) and
`pytorch-quantization==2.1.3` from `pypi.ngc.nvidia.com`. Docker base
`pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel`; no `deepspeed` anywhere.

**CPU-only Windows feasibility.**

| Blocker | Evidence | Consequence |
|---|---|---|
| `import bitsandbytes as bnb` | `models/quantization_utils.py:4` | import fails / CUDA-only; used only for the FP8 dynamic map at `:98` |
| `from pytorch_quantization import tensor_quant` | `models/quantization_utils.py:5`, `quantization_utils_2.py:8` | NVIDIA NGC wheel, unavailable on this box |
| `import ray` (module level) | `models/allocation_utils.py:1`, `allocation_utils_2.py:1`, `ray_utils.py:1` | `ray[air]` does not install cleanly on Windows |
| `import gurobipy` (module level) | `models/lq_utils.py:3` | import is free, `optimize()` needs a licence; `scipy.milp` backend exists (`:253-263`) |
| `A = param.cuda()` | `models/lq_utils.py:160` | hard CUDA in the allocation data prep |
| packing hardcodes `device="cuda"` | `models/packbits_utils.py:241-264` | the real packed path cannot run on CPU even with deps |
| `device="cuda"` / `map_location=cuda` | `models/lora_utils.py:131,197`, `run_legacy_evaluation.py:47` | checkpoint load/convert blocked |
| `torch._inductor.codecache.AsyncCompile` at import | `models/misc_utils.py:6` | hidden torch-inductor import side effect |
| internal cluster paths | `FILE_NAMES_DICT`/`MODEL_PATHS_DICT` → `/export/share*` (`allocation_utils.py:169-181`, `prepare_ilp_and_fisher_data.py:12-31`) | not usable without repointing; public ILP/Fisher artifacts exist on HF (`README.md:9-12`) |

Pure-torch and CPU-portable: `factorizations_utils.py` entirely; the `lq_utils.py:42-91` loop apart
from `maybe_sparsify_or_quantize`; the `blockwise-nf` quantization path of `quantization_utils.py`
(only the int/uint path via `pytorch_quantization.tensor_quant` and the FP8 map via bnb are foreign).

**Evaluation.** lm-eval harness with a pinned commit
`b281b0921b636bc36ad05c0b0b0763bd6dd43463` (`scripts/setup_lm_eval.sh`), tasks/metrics
(`experiments/lm_eval_utils.py:17-45`): `arc_challenge` acc_norm@25, `hellaswag` acc_norm@10,
`truthfulqa_mc` mc2@0, `hendrycksTest-*` acc@5, `winogrande` acc@5, `gsm8k` acc@5, averaged ×100.
WikiText-2 perplexity `exp(loss)` (`run_clm.py:835-838`); MMLU 5-shot accuracy
(`experiments/mmlu_utils.py:31-90`). **The repo contains no results tables**; the numbers live in the
paper. Bit budgets driven by `scripts/c4_lora.sh` are 2.5–4.0 plus nf3/nf4/GPTQ/dense baselines,
rank 64.

---

### 3.4 `pytorch/ao` @ `cff77b46` (QAT surface only)

**Surface.** `torchao/quantization/qat/` — 10 modules. Public API
(`torchao/quantization/qat/__init__.py:35-67`): `QATConfig`, `QATStep`, `FakeQuantizeConfigBase`,
`IntxFakeQuantizeConfig`, `Float8FakeQuantizeConfig`, `Int4WeightFakeQuantizeConfig`,
`FakeQuantizerBase`/`IntxFakeQuantizer`/`Float8FakeQuantizer`/`Int4WeightFakeQuantizer`,
`FakeQuantizedLinear`, `FakeQuantizedEmbedding`, `TwoStepQuantizer`, `ComposableQATQuantizer`,
legacy `Int8DynActInt4WeightQATQuantizer`, `Int4WeightOnlyQATQuantizer`, `Float8ActInt4WeightQATQuantizer`,
`initialize_fake_quantizers`.

**Documented pipeline** (`docs/source/workflows/qat.md:73-97`):

```python
from torchao.quantization import quantize_, Int4WeightOnlyConfig
from torchao.quantization.qat import QATConfig
quantize_(model, QATConfig(base_config, step="prepare"))   # nn.Linear -> FakeQuantizedLinear
train_loop(model)
quantize_(model, QATConfig(base_config, step="convert"))   # swap back + apply PTQ base config
```

Dispatch: `@register_quantize_module_handler(QATConfig)` (`qat/api.py:187`) →
`_qat_config_transform` (`:188-309`); prepare swaps modules (`:214-247`), convert swaps back and
re-applies the base config's transform (`:250-309`), optionally forwarding learned qparams when
`range_learning=True` (`:270-278`). Hot path: `FakeQuantizedLinear.forward`
(`qat/linear.py:108-114`) fake-quantizes activations and weights, then `F.linear`.

**Fake-quant arithmetic** (quote-able): `quant = clamp(round(x/scale) + zero_point, qmin, qmax)`
(`quant_primitives.py:512-514`), `dequant = (q − zero_point)·scale` (`:888-891`), STE `_Round`
(`:215-227`) and `_ClampSTE` (`:229-260`). Granularities `per_token|per_channel|per_group|per_tensor`
(`fake_quantize_config.py:249-291`); dynamic by default, static + `range_learning` supported
(`fake_quantizer.py:198-207,401-425`); int4 weight-only fake quant emulates MSLK numerics in pure ATen
(`fake_quantizer.py:144-188`).

**Dependencies.** `pyproject.toml`/`setup.py` declare **no torch requirement at all**; runtime expects
torch ≥ 2.11 for prebuilt `.so` (`torchao/__init__.py:69-79`); CI pins `torch==2.13.0`. `version.txt`
= `0.19.0` (dev line after the released `0.18.0`).

**CPU-only Windows feasibility.**

| Path | Status | Evidence |
|---|---|---|
| `step="prepare"` + intx/int4 **fake** quantization, and the whole `torch.ops.quantized_decomposed` flow | **CPU eager OK** | pure ATen arithmetic; module `_DEVICE` falls back to `"cpu"` (`test/quantization/test_qat.py:114-117`); ungated CPU tests at `test_qat.py:945-1022, 1177-1190, 1280-1285, 1943-1992`; CI CPU legs run this file (`.github/workflows/regression_test.yml:34-43,105-108`) |
| `step="convert"` **without** a base config (swap back to `nn.Linear`) | CPU OK | `test_qat.py:2548-2574` |
| `QATConfig(Int4WeightOnlyConfig(...), step="convert")` | **CUDA/mslk only** | `int4_tensor.py:139-140` raises `ImportError("Requires mslk >= 1.0.0")`; `torch.ops.mslk.*` (`:215-231`); all related tests CUDA-gated (`test_qat.py:2728+`) |
| `Int8DynamicActivationIntxWeightConfig`/`IntxWeightOnlyConfig` convert (`unpacked_to_int8`) | CPU-capable by code reading | `intx_unpacked_to_int8_tensor.py:315-340` dequantizes then `F.linear`; the upstream equivalence test is GPU-gated only because its harness needs an accelerator (`test_qat.py:2635`) — `[INFERENCE]` from reading, not from an executed run |
| Float8 / MX / NVFP4 QAT prototypes | CUDA/Triton only | `prototype/qat/mx.py:87-88,138-139`; `nvfp4.py:34`; `prototype/mx_formats/kernels.py:163-166` |

**Distribution note (checked live, not assumed):** `https://pypi.org/pypi/torchao/json` reports latest
`0.18.0`, and its file list includes `torchao-0.18.0-py3-none-any.whl` — a **pure-Python wheel**, i.e.
installable on Windows/CPU without any compiled extension (the other artifact is a Linux manylinux
wheel). So torchao can be taken as a *dependency* on this workstation; the pinned source SHA is still
recorded for provenance.

**Reimplement vs depend.** The affine fake-quant core, qparam choosers, granularity objects and the
`FakeQuantizedLinear` plumbing are reimplementable in a few hundred lines of pure PyTorch; the
`quantize_` dispatch/config machinery, tensor-subclass convert targets and packing formats are better
taken as a dependency. BSD-3-Clause ⇒ depending on the wheel is unconditionally fine; vendoring
obligates notice retention and forks numerics that upstream tests define. **Recommendation: depend on
`torchao==0.18.0`, vendor nothing.**

---

## 4. Consolidated CPU-only Windows blocker matrix

| Blocker class | LR-QAT | LoftQ | lq-lora | pytorch/ao |
|---|---|---|---|---|
| Unconditional CUDA API call at runtime | `utils/utils.py:188-190` (`torch.cuda.memory_*`) | — | `lq_utils.py:160` (`.cuda()`) | — |
| bitsandbytes CUDA kernels | not used at all | peft `loftq_utils.py:207-211`; bnb 4-bit model load in every script | `quantization_utils.py:4` | not used by QAT |
| Other CUDA-only kernels | — | — | `pytorch_quantization.tensor_quant`, `packbits_utils.py:241-264` | mslk int4 convert, Triton MX/NVFP4 |
| Python-version incompatibility on Windows | `time.clock` (`utils/utils.py:57-59`) | — | — | — |
| CUDA-only container | `docker/Dockerfile:1` (CUDA 11.7) | — | `Dockerfile` (cuda11.8) | — |
| Commercial licence dependency | — | — | `gurobipy` (scipy.milp fallback exists) | — |
| Cluster-only paths | `slimpajama` pre-tokenized path required for the repro recipe | — | `/export/share*` dicts | — |
| Non-CUDA blockers | accelerate `use_cpu: false`; torch 1.13.1+cu117 pin | unpinned deps, torch not listed | `ray` on Windows | none for the prepare path |

**Net conclusion.** Three of the four upstream *runtimes* are hard-blocked on this workstation, and the
fourth (torchao) is partially usable (its CPU-supported QAT surface). In every case the **mathematics is
pure PyTorch and CPU-runnable**; what is blocked is the shipping runtime — CUDA kernels, CUDA-only
containers, GPU device maps and cluster paths. The reproduction plan
(`docs/research/reproduction-plan.md`) therefore reimplements the published equations against this
project's own frozen `QuantSpec`/`fake_quantize`/`factorization` interfaces, and treats the upstream
repositories as **specification references, never as dependencies of the experiment path**.

**Correction (2026-10-08, affects the matrix above).** The compute envelope was re-measured after this
document's first draft: real CPU low-bit kernels *do* exist on this host for artifacts we serialize
ourselves — ONNX Runtime 1.30.0 `MatMulNBits` (int4 weight-only) and `MatMulInteger` (int8),
`torchao 0.18.0 IntxWeightOnlyConfig(torch.int4, PerGroup(32))`, and a bitsandbytes 0.50.2 Windows CPU
backend (NF4 storage with fp32 compute) — and Docker Linux containers run (16 CPUs, 7.318 GiB). See
`docs/research/environment.md` §1 "Corrected on 2026-10-08" and `docs/research/backend-capability.md`
§2–§4. Consequences for the table above: measurement **class 3** (packed storage) and **class 4-CPU**
(timing of *our own* containers) are available; the *upstream runtimes* remain blocked exactly as
listed, because their blockers are CUDA-only kernels and hard-coded CUDA devices, which neither a
container nor a wheel upgrade removes. Class 4-GPU and class 5 remain unavailable.

---

## 5. Reimplement vs. import vs. adapt (per-repo decision)

| Repo | Decision | Why |
|---|---|---|
| LR-QAT | **Reimplement** (from paper equations) | licence is a modified BSD-3 with **no patent grant**; runtime is CUDA/Windows-blocked; the math is ~40 lines of pure torch |
| LoftQ | **Reimplement**, with `peft` optionally used as an *oracle for the initialisation only* on a tiny CPU matrix (peft is Apache-2.0 and its NF lookup path runs on CPU when `num_bits ∈ {2,8}` — `loftq_utils.py:196-198`) | the authoritative LLM implementation is in peft, not in the LoftQ repo; the repo's GLUE path is CPU-clean and can be read as a second reference |
| lq-lora | **Reimplement the decomposition + allocation math**; use `scipy.optimize.milp` instead of Gurobi; skip packing entirely | MIT permits copying, but the runtime is CUDA/ray/gurobi-bound and the packing path hardcodes `device="cuda"` |
| pytorch/ao | **Import as a pinned dependency** (`torchao==0.18.0`, pure-Python wheel) for the QAT *prepare* path only; never use its int4 `convert` target | BSD-3-Clause, wheel install is clean, and the CPU-supported fake-quant surface is exactly the class-2 (fake-quantization) measurement this project is allowed to take |

---

## 6. Open questions / risks recorded (not resolved here)

1. **peft version drift**: the LoftQ LLM path depends on peft internals (`init_lora_weights="loftq"`)
   that moved (`loftq_utils.py` changed between 0.9 and current). If a peft-based oracle is used, the
   version must be pinned in `uv.lock` and recorded in the method card.
2. **bitsandbytes NF4 vs our int-N**: LoftQ's headline sub-4-bit numbers are produced by the *fake*
   NF quantizer, not by bnb; any reproduction must therefore state which quantizer it used
   (see `docs/research/reproduction-plan.md` §4).
3. **torchao 0.19.0-dev vs 0.18.0**: the pinned SHA is post-release. If the plan depends on torchao,
   it must depend on the released `0.18.0` (PyPI), and the SHA pin is provenance only.
4. **No upstream repository contains the numbers we intend to compare against** for lq-lora; the
   comparison values must be transcribed from the papers, not from the repos.
5. **`references/upstream-verification.log`** is owned by another wave-1 stream; this document cites it
   rather than editing it. `docs/research/upstream-lockfile.md` (stream A) is the canonical lockfile
   and does not yet exist at the time of writing — this document is a code-level companion, not a
   substitute.
