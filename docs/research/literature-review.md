# SpectraQuant — Literature Review

Status: **Wave 1, draft 0.1 (2026-10-08).** Scope: what is already known about (i) preparing models
for low-rank factorization before quantization, (ii) quantization-aware low-rank *training*, (iii)
post-training quantization (PTQ) methods we will use as baselines, (iv) mixed-precision / mixed-rank
allocation, and (v) sensitivity proxies and spectral regularizers that could inform our allocator.

Metadata (titles, authors, years, arXiv IDs) in this file and in `literature-matrix.csv` was
verified by fetching `arxiv.org/abs/<id>` (or the cited proceedings/repo) on 2026-10-08; see
`references/bibliography.bib` for BibTeX and `references/upstream-verification.log` for the pinned
upstream commits. Where a number is reported from an abstract, it is the *paper's own* claim and is
marked as such; we have not reproduced any of these results. Code licenses are stated as **verified**
only for the repositories pinned in `upstream-lockfile.md` (verified via
`gh api repos/<owner>/<repo> --jq .license.spdx_id`); all other repository licenses mentioned in this
file are marked **unverified** and must not be relied on before a run.

Reading guide: §1–§6 are the thematic body; §7 collects cross-cutting gaps that define our problem.
The **complete per-work field set** — problem setting, model scale, quantization type, rank treatment,
objective, memory accounting, datasets, metrics, code, pinned commit, license, and local relevance —
is tabulated for all 50 works in `literature-matrix.csv`; the sections below are the narrative and
may compress fields that the matrix states in full.

---

## 1. Foundations: quantization and low-rank compression

### 1.1 Gholami et al., *A Survey of Quantization Methods for Efficient Neural Network Inference* (2021), arXiv:2103.13630
- **Problem setting:** taxonomy of quantization (uniform vs. non-uniform, symmetric vs. asymmetric, per-tensor/axis/channel, PTQ vs. QAT).
- **Model scale:** CNN- and transformer-era methods survey (no single scale).
- **Quantization type:** integer and low-precision fixed point; 4-bit and below highlighted.
- **Rank treatment:** none.
- **Objective:** taxonomy; no single training objective.
- **Memory accounting:** qualitative (bits × parameters); no serialization protocol.
- **Datasets/metrics:** n/a (survey).
- **Code:** n/a. **License:** n/a.
- **Limitations:** pre-LLM; does not cover the low-rank × quantization interaction.

### 1.2 Nagel et al., *A White Paper on Neural Network Quantization* (2021), arXiv:2106.08295
- **Problem setting:** engineering pipelines for PTQ and QAT; hardware-motivated quantization noise model.
- **Model scale:** ResNet/MobileNet-class CNNs.
- **Quantization type:** int8 weights+activations primarily; per-channel weight ranges.
- **Rank treatment:** none.
- **Objective:** cross-entropy with fake quantizers (QAT); reconstruction-based PTQ.
- **Memory accounting:** bit-width × tensor shape (analytical only).
- **Datasets/metrics:** ImageNet accuracy.
- **Code:** no official repo for the white paper (ideas live in AIMET). **License:** n/a.
- **Limitations:** the "PTQ is enough for 8-bit, QAT for lower" heuristic is CNN-era and does not
  transfer directly to weight-only 4-bit LLM quantization.

### 1.3 LeCun, Denker & Solla, *Optimal Brain Damage* (NeurIPS 2, 1990); Hassibi & Stork, *Second Order Derivatives for Network Pruning: Optimal Brain Surgeon* (NeurIPS 5, 1993)
- **Problem setting:** which parameters can be removed with least loss increase; second-order saliency.
- **Model scale:** tiny MLPs.
- **Quantization type:** none (pruning).
- **Rank treatment:** none (structured removal of individual weights/units).
- **Objective:** loss increase estimated via diagonal (OBD) or full (OBS) Hessian.
- **Memory accounting:** parameter count.
- **Datasets/metrics:** XOR / small classification.
- **Code:** none official. **License:** n/a.
- **Limitations:** motivation for Hessian-based sensitivity, not a practical LLM method; the
  Hessian is intractable at LLM scale without the Kronecker/GPTQ approximations.

### 1.4 Han et al., *Deep Compression* (2015), arXiv:1510.00149; *Learning both Weights and Connections* (2015), arXiv:1506.02626
- **Problem setting:** prune + quantize + Huffman-code for CNN storage.
- **Model scale:** AlexNet/VGG.
- **Quantization type:** k-means/clustered weight sharing, 4–8 bit.
- **Rank treatment:** none.
- **Objective:** retrain after pruning; clustering by weight value.
- **Memory accounting:** stored bytes + codebook + Huffman table (a rare early example of *storage*
  accounting, not just parameter counting).
- **Datasets/metrics:** ImageNet top-5.
- **Code:** none official (widely reimplemented). **License:** n/a.
- **Limitations:** shows stacking compression transforms is non-trivial; no transformer results.

**Take-away for SpectraQuant.** Quantization and pruning/factorization both discard information, and
the ordering of the two lossy transforms changes the outcome — but the classical literature is
CNN-era and does not answer the transformer/LLM version of the ordering question.

---

## 2. Low-rank-then-quantize and quantized low-rank decomposition

### 2.1 Li et al., *LoftQ: LoRA-Fine-Tuning-Aware Quantization* (2023), arXiv:2310.08659
- **Problem setting:** QLoRA-style finetuning where low-rank *adapters* are trained on a quantized base; initialization is chosen to reduce quantization error.
- **Model scale:** up to 70B (finetuning); 7B/13B most reported.
- **Quantization type:** NF4 / 4-bit per-block weight-only (QLoRA stack).
- **Rank treatment:** rank-$r$ LoRA adapters; **alternating** minimization of
  $\lVert W - Q - LR\rVert_F^2$ over the quantized base $Q$ and the low-rank init $LR$.
- **Objective:** alternate between quantization of $W - LR$ and truncated SVD of $W - Q$.
- **Memory accounting:** stored = quantized base + adapters; adapters are *training* memory, not
  deployed compression (merged at inference in the original).
- **Datasets/metrics:** GLUE, GSM8K, WikiText-2 perplexity.
- **Code:** `yxli2123/LoftQ` @ `ae33fd4fd05fd4ba146555cd77c13d307eb4e9b3`. **License:** MIT.
- **Limitations:** the low rank is used to *reconstruct quantization error*, not to compress the
  base; the "preparation helps" signal is entangled with adapter capacity and finetuning. No
  equal-stored-byte frontier sweep.

### 2.2 Guo et al., *LQ-LoRA: Low-rank Plus Quantized Matrix Decomposition* (2023), arXiv:2311.12023
- **Problem setting:** factorize each pretrained matrix into a quantized part plus a low-rank part, then finetune.
- **Model scale:** 3B–70B.
- **Quantization type:** 4-bit (and 2-bit) per-group weight-only with data-aware quantization of the residual.
- **Rank treatment:** **column-wise adaptive ranks** across layers under a total parameter budget;
  DP/knapsack-style allocation of ranks. (Closest published precedent for *rank* allocation.)
- **Objective:** minimize $\lVert W - Q - LR\rVert$ with a data-aware (activation-weighted) quantizer.
- **Memory accounting:** total factorization parameters under a budget; int4 storage assumed.
- **Datasets/metrics:** WikiText-2 perplexity, MMLU/GSM8K.
- **Code:** `HanGuo97/lq-lora` @ `c2424b3adc27197815da1ac9e1304565168d824d`. **License:** MIT.
- **Limitations:** allocation is over *rank* given a fixed bit width; the quantization-cost term is
  a reconstruction proxy, not an output-aware one; no joint rank × bit search.

### 2.3 Cho et al., *Preserve-Then-Quantize: Balancing Rank Budgets for Quantization Error Reconstruction in LLMs* (SRR), ICML 2026, arXiv:2602.02001
- **Problem setting:** PTQ with quantization-error reconstruction (QER): $W \approx Q + LR$.
- **Model scale:** LLM PTQ across "diverse models"; 2-bit QPEFT on GLUE.
- **Quantization type:** weight-only PTQ (multi-bit settings); later 2-bit quantized PEFT.
- **Rank treatment:** **splits the rank budget**: top-$k$ singular subspace preserved *before*
  quantization, remaining $r-k$ ranks reconstruct the quantization residual. A **theory-guided
  criterion** chooses $k$ by balancing "quantization-exposed energy" against "unrecoverable error".
- **Objective:** minimize exposed quantization energy subject to rank constraint; then QPEFT with
  gradient scaling along preserved directions.
- **Memory accounting:** rank $r$ budget shared between the two roles; quantized $Q$ storage separate.
- **Datasets/metrics:** WikiText-2 perplexity, GLUE (5.9 pp average gain at 2-bit, *paper's claim*).
- **Code:** project page `ai-isl.github.io/srr` (no pinned repo verified by us yet). **License:** n/a (paper CC-BY-4.0 on arXiv).
- **Limitations / relevance:** **the closest published neighbour to our "rounding-aware spectral
  regularizer" and to the rank-splitting idea.** It is PTQ + a closed-form spectral criterion, not a
  *regularizer applied during training*, and not a joint rank × bit allocation.

### 2.4 Gordon et al., *MLoRQ: Bridging Low-Rank and Quantization for Transformer Compression* (2025), arXiv:2507.09616
- **Problem setting:** joint per-layer bit-width **and** rank assignment under a memory constraint.
- **Model scale:** Vision Transformers (ViT/DeiT-class); image classification, detection, segmentation.
- **Quantization type:** mixed-precision (hardware-realizable mixed low-rank + quantization), "adapted adaptive rounding".
- **Rank treatment:** explicit per-layer rank; two-stage optimization (intra-layer: enumerate
  low-rank × quantization candidates; inter-layer: assign under the memory constraint).
- **Objective:** minimize task loss under a memory budget; greedy/DP-style inter-layer assignment.
- **Memory accounting:** per-layer compressed size with rank × bits; total budget constraint.
- **Datasets/metrics:** ImageNet classification, COCO detection/segmentation; up to 15% improvement (*paper's claim*).
- **Code:** not verified. **License:** n/a.
- **Limitations / relevance:** **closest prior art to our mixed-rank/mixed-bit allocator**, but for
  vision transformers and a post-hoc search; the sensitivity signal is not an output-aware
  propagated proxy, and there is no spectral regularizer. Confirms the problem is legitimate and
  that joint rank × bit under a budget is already a published framing.

### 2.5 Wang et al., *When Low-Rank Meets Mixed-Precision: Training-Free Joint Compression for Efficient LLM Inference* (ASP-DAC 2026, pp. 604–610)
- **Problem setting:** training-free joint low-rank + mixed-precision compression of LLMs.
- **Model scale:** LLM (conference report).
- **Quantization type:** mixed-precision, per-layer.
- **Rank treatment:** joint per-layer low-rank + bit-width selection; input-aware sensitivity measure; "sample-aware decomposition" and a unified difference matrix for the combined error.
- **Objective:** minimize combined compression error under a size budget (20% of original size claimed).
- **Memory accounting:** joint size target.
- **Datasets/metrics:** not verified in detail (no arXiv preprint found; metadata from ASP-DAC program).
- **Code:** not verified. **License:** n/a.
- **Limitations / relevance:** another **joint rank × bit allocation** precedent, training-free and
  input-aware. Reinforces that our differentiation must rest on the *output-aware propagated cost
  proxy* + *training-time regularizer*, not on "joint allocation" itself.

### 2.6 Yao et al., *ZeroQuant-V2* (2023), arXiv:2303.08302
- **Problem setting:** comprehensive PTQ study; introduces **low-rank compensation (LRC)** to correct PTQ error.
- **Model scale:** up to 175B-class.
- **Quantization type:** INT8/INT4 weight + activation, various granularities.
- **Rank treatment:** low-rank correction term added *after* quantization (LRC), not a rank budget search.
- **Objective:** post-hoc error compensation.
- **Memory accounting:** additive LRC terms counted separately.
- **Datasets/metrics:** GLUE, WikiText-2 perplexity.
- **Code:** no standalone repo (integrated into DeepSpeed). **License:** n/a (DeepSpeed is Apache-2.0).
- **Limitations:** LRC rank is a free hyperparameter, not allocated across layers; no output-aware proxy.

---

## 3. Quantization-aware low-rank training

### 3.1 Bondarenko et al., *Low-Rank Quantization-Aware Training for LLMs* (LR-QAT) (2024), arXiv:2406.06385
- **Problem setting:** memory-efficient QAT for LLMs by training **low-rank auxiliary weights** that are fused into the quantized model.
- **Model scale:** up to 7B reported (single 24 GB GPU).
- **Quantization type:** weight+activation low-bit integer, quantization-grid-aware quantizer; per-group.
- **Rank treatment:** fixed low rank for the auxiliary weights; **fuses** LR into the quantized weights at inference (hence the low rank is training-memory, and the deployed model is quantized-only).
- **Objective:** standard QAT loss (cross-entropy) with fake-quantized forward; low-rank weights act
  as a "quantization grid" regularizer.
- **Memory accounting:** reports training-time memory; deployed size = quantized weights only.
- **Datasets/metrics:** WikiText-2 perplexity, zero-shot commonsense tasks.
- **Code:** `Qualcomm-AI-research/LR-QAT` @ `8795afe054cf951b714299e01083a1b354721829`. **License:** BSD-3-Clause-Clear.
- **Limitations / relevance:** the central reference for "low-rank preparation inside QAT"; but rank
  is **not** allocated per layer against a bit-width budget, and the low rank is not a compression
  reason — it is a training-memory economy. Our joint rank × bit allocation under a *stored-bytes*
  budget is a different object.

### 3.2 Xu et al., *QA-LoRA: Quantization-Aware Low-Rank Adaptation* (2023), arXiv:2309.14717
- **Problem setting:** reconcile group-wise quantization with LoRA by using group-wise operators so the quantized weights + adapters are mergeable.
- **Model scale:** LLaMA-7B/13B, LLaMA-2-7B, LLaMA-2-13B.
- **Quantization type:** group-wise INT4 (weight-only) finetuning then deployment.
- **Rank treatment:** fixed LoRA rank; balances quantization freedom vs. adaptation freedom.
- **Objective:** finetuning loss with group-wise quantized base; merge after training.
- **Memory accounting:** quantized base + adapters (deployed merged).
- **Datasets/metrics:** MMLU, commonsense QA.
- **Code:** community implementations (PEFT supports `gptq`/`awq` LoRA); no verified pinned repo of our own.
- **Limitations:** rank is a hyperparameter, not allocated; no output-aware sensitivity.

### 3.3 Dettmers et al., *QLoRA* (2023), arXiv:2305.14314
- **Problem setting:** finetune a 4-bit quantized base with LoRA + paged optimizers + double quantization.
- **Model scale:** up to 65B on one 48 GB GPU.
- **Quantization type:** NF4 4-bit weight-only, per-block; double quantization of the quantization constants.
- **Rank treatment:** fixed LoRA rank (adapters only).
- **Objective:** task cross-entropy.
- **Memory accounting:** careful accounting of optimizer/gradient memory and adapter size (training memory, not compression).
- **Datasets/metrics:** GLUE, MMLU, benchmark suite.
- **Code:** `artidoro/qlora` (license **unverified**; not pinned in this wave); concepts in `huggingface/peft`. **License (PEFT):** Apache-2.0 (verified, `huggingface/peft` pinned in `upstream-lockfile.md`).
- **Limitations:** adaptation, not compression; used by us only as a baseline/framing.

### 3.4 Liu et al., *LLM-QAT: Data-Free Quantization Aware Training* (2023), arXiv:2305.17888
- **Problem setting:** QAT for LLMs without the original training data, using a teacher model to generate data.
- **Model scale:** up to 30B.
- **Quantization type:** weight/activation/kv-cache 4-bit, distillation-based.
- **Rank treatment:** none.
- **Objective:** distillation + QAT loss.
- **Memory accounting:** deployed quantized size.
- **Datasets/metrics:** WikiText-2, C4, zero-shot.
- **Code:** community reimplementations. **License:** n/a.
- **Limitations:** data-free generation is out of scope locally; establishes QAT loss design.

### 3.5 PyTorch, *TorchAO — Quantization-Aware Training (QAT)* (docs; page last updated 2026-03-25)
- **Problem setting:** production QAT API: `prepare` inserts fake quantizers, `convert` materializes quantized ops.
- **Model scale:** any; examples include a small llama3 and gemma3-12b-it.
- **Quantization type:** int4 weight-only per-group (32/128), int8 per-token dynamic activation; float8 in the broader library.
- **Rank treatment:** none in the QAT path; QAT+LoRA recipe exists in torchtune integration.
- **Objective:** user-supplied training loss on fake-quantized model.
- **Memory accounting:** not the focus; "int4 baseline vs int4 QAT" eval table uses recovers-%.
- **Datasets/metrics:** WikiText perplexity, BBH (recovery percentages).
- **Code:** `pytorch/ao` @ `cff77b46ef85ba8472b4a286598073d6c01e18e5`. **License:** BSD-3-Clause-style Meta/Arm license; GitHub reports `NOASSERTION` (SPDX id absent).
- **Local relevance:** the QAT kernels' CUDA paths are **deferred** (no CUDA locally). The *fake-quantization*
  (`prepare`) semantics are CPU-portable and inform our Class-2 measurements; separately, a real
  **CPU** low-bit kernel path was verified locally on 2026-10-08 (ONNX Runtime `MatMulNBits`/
  `MatMulInteger`, and torchao intx weight-only forwards on CPU), which makes class **4-CPU** available
  for artifacts we serialize ourselves (`AGENTS.md` §5; `backend-capability.md` §2.4).

---

## 4. Post-training quantization (PTQ) baselines

### 4.1 Frantar et al., *GPTQ* (2022), arXiv:2210.17323
- **Setting:** one-shot weight-only PTQ via layer-wise second-order (Hessian) error minimization, sequential column quantization with error compensation. **Scale:** 175B. **Quant:** 4/3-bit per-row, group-wise; optional act-order. **Rank:** none. **Objective:** minimize $\lVert WX - \hat W X\rVert_F^2$ from layer input statistics. **Memory:** stored weight bits (+ scales/zeros). **Data/metrics:** WikiText-2, C4, PTB perplexity; OPT/BLOOM/LLaMA. **Code:** `IST-DASLab/gptq` (license **unverified**; not pinned this wave). **Relevance:** the **candidate output-aware signal source** for our proxy (GPTQ's layer objective is a function of layer input), and the standard baseline. **Limitation:** requires calibration activations; may diverge at 2-bit.

### 4.2 Lin et al., *AWQ* (2023), arXiv:2306.00978
- **Setting:** activation-aware weight-only PTQ: protect the ~1% salient weight channels scaled by activation magnitude. **Scale:** up to 70B. **Quant:** 4-bit, group-wise; also W4A16 kernels. **Rank:** none. **Objective:** minimize output error with per-channel scaling (no backprop). **Memory:** weights + scales. **Data/metrics:** WikiText-2, TinyChat, VLM. **Code:** `mit-han-lab/llm-awq` (license **unverified**; not pinned). **Relevance:** activation-awareness is a simpler proxy we must beat (H2 comparator baseline). **Limitation:** GPU kernels deferred locally.

### 4.3 Xiao et al., *SmoothQuant* (2022), arXiv:2211.10438
- **Setting:** migrate activation outliers into weights via a per-channel scale so activation quantization becomes easy. **Scale:** up to 530B. **Quant:** W8A8. **Rank:** none. **Objective:** control max/range migration. **Memory:** W8A8 storage. **Code:** `mit-han-lab/smoothquant` (license **unverified**; not pinned). **Relevance:** activation-outlier handling interacts with low-rank structure.

### 4.4 Dettmers et al., *LLM.int8()* (2022), arXiv:2208.07339
- **Setting:** mixed-precision decomposition: outlier feature dimensions in fp16, rest int8 matmul. **Scale:** up to 175B. **Quant:** int8 + fp16 outlier columns. **Rank:** none. **Code:** bitsandbytes (license **unverified**; CUDA-dependent → deferred). **Relevance:** defines the outlier phenomenon class and a fairness baseline at 8-bit.

### 4.5 Lee et al., *OWQ* (2023), arXiv:2306.02272
- **Setting:** weight quantization where weak columns (low activation magnitude) are quantized harder and the rest kept mixed-precision. **Quant:** W3/W4 with fp16 for salient columns. **Rank:** none. **Relevance:** a mixed-precision-under-budget precedent at the *column* rather than layer granularity.

### 4.6 Kim et al., *SqueezeLLM* (2023), arXiv:2306.07629
- **Setting:** sensitivity-based non-uniform quantization into dense + sparse (outlier) components. **Quant:** 4-bit with sparse outliers. **Rank:** none. **Relevance:** sensitivity-weighted rounding, another proxy family.

### 4.7 Yao et al., *ZeroQuant* (2022), arXiv:2206.01861
- **Setting:** end-to-end PTQ with group-wise weight quantization + layer-wise knowledge distillation. **Quant:** INT8/INT4. **Code:** DeepSpeed (license **unverified**). **Relevance:** shows fine-grained grouping reduces error; informs our fairness controls.

### 4.8 Liu et al., *LLM-FP4* (2023), arXiv:2310.16836; Zhao et al., *Atom* (2023), arXiv:2310.19102; Wei et al., *Outlier Suppression+* (2023), arXiv:2304.09145
- **Setting:** non-uniform / floating-point low-bit formats and outlier-aware scaling. **Quant:** FP4, mixed-precision W4A4 KV4, and equivalent shifting/scaling. **Rank:** none. **Relevance:** alternative numeric formats we deliberately exclude from the core path (our quantization is integer per-group, CPU-serializable).

---

## 5. Mixed-precision / mixed-rank allocation and sensitivity proxies

### 5.1 Dong et al., *HAWQ* (2019), arXiv:1905.03696 and *HAWQ-V2* (2019), arXiv:1911.03852
- **Setting:** mixed-precision via Hessian spectra: HAWQ uses top Hessian eigenvalues; HAWQ-V2 uses **trace-weighted** sensitivity (Hessian trace as a proxy for quantization perturbation) to allocate bit widths. **Scale:** CNNs (ResNet/Inception) + BERT. **Quant:** mixed 2/4/8-bit with the average constrained. **Rank:** none. **Objective:** minimize $\sum_i \text{tr}(H_i)\,\lVert\Delta W_i\rVert$ style surrogate under an average-bit budget. **Code:** community. **Relevance:** **the canonical mixed-precision sensitivity proxy we must beat or match** (H2). **Limitation:** Hessian-trace is a *weight-space* proxy; ignores propagation and rank coupling; no rank dimension.

### 5.2 Frantar & Alistarh, *Optimal Brain Compression* (2022), arXiv:2208.11580
- **Setting:** unify pruning and quantization as sequential greedy minimization of a layer-wise second-order objective. **Scale:** up to BERT-large / small LLMs. **Relevance:** formalizes "greedy reconstruction-error allocation under a budget" — the algorithmic skeleton our allocator will resemble. **Limitation:** single-modality (bit width only), calibration-dependent objective.

### 5.3 Guan et al., *APTQ* (2024), arXiv:2402.14866
- **Setting:** attention-aware PTQ mixed precision; Hessian-trace sensitivity plus attention-output effect. **Quant:** mixed 2/4-bit. **Rank:** none. **Relevance:** a 2024 output/attention-aware variant → direct prior art for "output-aware cost".

### 5.4 Huang et al., *SliM-LLM* (2024), arXiv:2405.14917
- **Setting:** salience-driven mixed precision at weight-group granularity (finer than layer). **Quant:** mixed per-group. **Relevance:** shows allocation granularity matters; our allocator operates at layer granularity locally.

### 5.5 Dumitru et al., *Layer-Wise Quantization* (2024), arXiv:2406.17415
- **Setting:** assign higher precision to important layers; presents an ordering as complementary to other quantizers. **Quant:** non-integer average bit widths. **Relevance:** argues layer-wise allocation is orthogonal and stacking-friendly — supports our framing.

### 5.6 Zhou et al., *AutoQRA* (2026), arXiv:2602.22268
- **Setting:** joint search over **per-layer bit width and LoRA rank** for mixed quantized finetuning; two-stage evolutionary + Bayesian optimization. **Scale:** LLMs (PEFT). **Rank treatment:** explicit per-layer rank of *adapters* (not base compression). **Objective:** downstream finetuning quality under a memory budget. **Relevance:** **closest joint bit × rank allocation precedent for LLMs**; differentiation = our proxy is closed-form/output-aware and our rank compresses the base, not adapters. **Limitation:** expensive search; no spectral regularizer.

### 5.7 Ha, Lee & Jeon, *KV-COBRA* (2026), arXiv:2609.24298
- **Setting:** co-optimized **bit-rank allocation per attention head** for KV-cache compression; balances rank-truncation loss vs. quantization loss, then redistributes budget. **Relevance:** near-identical *resource-allocation formalization* to ours, applied to KV cache rather than weights; Hadamard rotation + SVD-basis reordering by attention-KL is a query-aware variant of "output-aware". **Limitation:** KV-cache object; allocator is per-head and calibration-defined.

### 5.8 Misra et al., *MixQuant* (2026), arXiv:2607.23047
- **Setting:** budget-agnostic adaptive mixed precision; marginalizes each layer's distortion over random quantized upstream configurations; greedy pass serves any budget. **Scale:** Llama-3.2-3B, Llama-2-7B, Mistral-7B. **Quant:** AWQ/GPTQ wrappers. **Relevance:** **directly attacks the same "sensitivity depends on other layers' bit widths" issue our propagated proxy addresses**; a key comparator and prior-art risk. **Limitation:** bit width only, no rank.

### 5.9 Kennedy & Kennedy, *Quantization Error Is Spectrally Flat…* (RAM), 2026, arXiv:2609.33923
- **Setting:** a single random Gaussian probe gives an **unbiased, calibration-free estimate of a layer's squared quantization-error Frobenius norm** because round-to-nearest error is spectrally flat; the *propagated* form allocates bits under an exact byte budget via a knapsack solver. **Scale:** 1,683 tensors from a 35B MoE and 9B dense; seven architectures 8B–122B. **Quant:** mixed 2–8-bit. **Rank:** none. **Metrics:** WikiText-2 perplexity; Spearman vs. GPTQ objective (0.81–0.83 for propagated probe; the isolated estimator is uncorrelated). **Code:** described as from "Black Sheep Ai (baa.ai)"; repo not verified. **Relevance:** **the sharpest prior art for an output-aware, calibration-light quantization-cost proxy, and the strongest evidence that the propagated (output-aware) form is the right one while the isolated form is not.** Our proxy must be positioned explicitly against it. **Limitation:** bit-width only; no rank dimension; no low-rank preparation; no regularizer.

### 5.10 Zhao et al., *CoopQ* (2025), arXiv:2509.15455
- **Setting:** Shapley-value layer sensitivity capturing **inter-layer interactions**, cast as binary quadratic optimization; 2- or 4-bit per layer under a memory constraint. **Scale:** Llama-3, Gemma-2, Qwen-3 × Quanto/HQQ/GPTQ. **Relevance:** shows interaction-aware sensitivity beats isolated metrics at low average bits — an alternative to our propagated proxy that we must acknowledge. **Limitation:** bit width only; expensive Shapley estimation.

### 5.11 Lee et al., *KronQ* (2026), arXiv:2607.07964
- **Setting:** PTQ using the **gradient covariance** in addition to the activation covariance (Kronecker-factored Hessian); bidirectional incoherence processing; a new sensitivity metric for inter-layer mixed-precision allocation from gradient+activation Hessian traces. **Scale:** up to LLaMA-3-70B, 2-bit. **Relevance:** strongest recent evidence that *output/gradient-side* information improves allocation. **Limitation:** bit width only; PTQ; no rank.

### 5.12 Miyato et al., *Spectral Normalization* (2018), arXiv:1802.05957; Schotthöfer et al., *Dynamical Low-Rank Compression with a Spectral Regularizer* (2025), arXiv:2505.08022; Hartford, *Spectrum: Targeted Training on Signal to Noise Ratio* (2024), arXiv:2406.06623
- **Setting:** controlling spectral properties during training. Spectral normalization bounds the Lipschitz constant; the dynamical-low-rank work adds a **spectral regularizer on the condition number of the low-rank core** to improve robustness; Spectrum reweights training by SNR.
- **Relevance:** the closest neighbours to our **rounding-aware spectral regularizer**. The dynamical-low-rank regularizer controls conditioning for *robustness*, not for *quantization rounding*; none of these ties the penalty to a quantization grid.
- **Limitation:** no quantization grid in the penalty; no stored-bytes framing.

---

## 6. Low-rank compression-only methods (context for Q1/Q2)

- **ASVD** (Yuan et al., 2023, arXiv:2312.05821): activation-aware SVD of weight matrices, distribution-balanced; no quantization. *Limitation:* rank per layer chosen heuristically; quantization-unaware.
- **SVD-LLM** (Wang et al., 2024, arXiv:2403.07378): truncation-aware SVD that accounts for the *truncated* part when selecting the rank. **Relevance:** directly motivates **H1** (what truncation exposes to later rounding) and the **Q2**
ordering comparator. *Limitation:* no quantization step modeled.
- **Palu** (Chang et al., 2024, arXiv:2407.21118): low-rank projection of KV cache with an offline SVD basis. *Relevance:* joint low-rank+quantization of KV cache exists; weights are a different object.
- **SliceGPT** (Ashkboos et al., 2024, arXiv:2401.15024): structured row/column deletion (PCA-like) preserving the computation graph. *Relevance:* an alternative structured transform; used as a boundary for "what counts as low-rank".

---

## 7. Cross-cutting gaps and the resulting problem statement

1. **Ordering is studied only in fragments.** LoftQ/LQ-LoRA/ZeroQuant-V2 treat low rank as *error
   reconstruction after quantization*. SRR/MLoRQ/KV-COBRA treat the balance explicitly but not as a
   controlled, equal-stored-bytes comparison of "prepare-then-quantize" vs. "quantize-then-correct".
2. **Sensitivity proxies are converging on output-awareness but not on rank.** HAWQ/HAWQ-V2 use
   weight-space Hessian traces; APTQ/KronQ add attention/gradient information; RAM/MixQuant/CoopQ
   show the propagated, interaction-aware, calibration-light direction is the right one. **None of
   these allocate rank.**
3. **Joint rank × bit allocation exists** (MLoRQ for ViTs, ASP-DAC 2026 for LLMs, AutoQRA for
   adapters, KV-COBRA for KV cache, LQ-LoRA for rank-only). What is not established is whether an
   **output-aware quantization-cost proxy that couples the two transforms** yields a better frontier
   than these alternatives at equal stored bytes — the gap our primary question targets.
4. **Regularizers are spectral or rounding-aware, rarely both.** Spectral regularizers address
   conditioning/robustness; adaptive-rounding methods address rounding but not the spectrum. The
   intersection (a penalty that is *both* spectral and tied to the quantization grid) appears open.
5. **Measurement honesty is rare.** Few papers separate training memory, stored bytes, and deployed
   representation; our equal-memory controls and measurement-class labelling (per `AGENTS.md` §5)
   are a deliberate methodological difference, not a claim of superiority.

**Consequence for novelty.** Charter hypotheses H1–H5 are *legitimate but crowded*. The safest
defensible candidate contributions are (a) the **output-aware quantizaton-cost proxy with an explicit
rank dimension** and (b) the **rounding-aware spectral regularizer**, both evaluated under a strict
equal-stored-bytes protocol. "Joint rank × bit allocation" alone is **previously known**
(MLoRQ/ASP-DAC-2026). Details and the explicit difference statements are in `novelty-risk.md`; the
falsification protocol is in `preregistration.md`.
