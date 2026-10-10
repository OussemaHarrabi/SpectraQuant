# Tier-1 equal-memory comparison — validated run, 2026-10-10

**Run** `tier1_smollm2_135m-cloud` · **status** `success` (registry: `validated`) · **code commit**
`1932c3d74cc69b9ab00854484d1bcc5702069222` · **substrate** Kaggle Tesla T4 · **model**
`HuggingFaceTB/SmolLM2-135M` @ `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` · **corpus**
`EleutherAI/wikitext_document_level` config `wikitext-2-raw-v1` @ `647234772b9554e208af6c826f23b99e3cac88c8`,
split `test` · **21 758 scored tokens** · **seed** 0 · **class** 2 (fake quantization) with class-1 byte
accounting · **one run per arm**.

## 1. The comparison

| arm | perplexity | accounted B | measured B | bytes vs fp16 | vs int4 PTQ |
|---|---|---|---|---|---|
| `fp16_reference` | **14.0183** | 212 336 640 | — | 1.00× | — |
| `ptq_uniform_8` (per-group g=32) | 14.0415 | 112 803 840 | 112 803 840 | 1.88× | — |
| **`ptq_uniform_4`** (per-group g=32) | **18.6193** | **59 719 680** | 59 719 680 | 3.56× | **the baseline** |
| `proxy_allocated` (allocation, **untrained**) | 64.5477 | 59 719 488 | 59 719 488 | 3.56× | **+246.7 %** |
| `spectraquant_regularized` (allocation + training, regularizer **ON**, λ_factor = λ_round = 1.0) | **23.3131** | 59 719 488 | 59 719 488 | 3.56× | **+25.2 %** |
| `spectraquant_regularized` (allocation + training, regularizer **OFF**, all λ = 0) | **23.1698** | 59 719 488 | 59 719 488 | 3.56× | +24.4 % |
| `low_rank_only` (rank 8) | degenerate | 4 884 480 | — | 43.5× | — |
| `rank_then_quant` (rank 8, 4-bit factors) | degenerate | 1 598 400 | 1 598 400 | 132.9× | — |

**The equal-memory comparison is exact by construction.** The allocated arms were solved against a
59 719 680-byte budget (A-0017 added the measured comparator counts to the plan's ladder), and they
landed at 59 719 488 — **192 bytes** (0.0003 %) below the int4 PTQ baseline. For those two arms the
class-3 **measured** bytes equal the class-1 accounted bytes exactly, so the comparison is not a
class-1 estimate standing in for a serialized artifact.

## 2. What this establishes

* **The mixed rank/bit allocation is real and mixed.** CP-SAT chose, over 210 targeted tensors:
  **bits 2 (5 layers), 3 (76), 4 (122), 8 (7)**; **effective rank 32 (206 layers)** with four narrow
  layers clamped to 2/4/8 by the SVD. It solved to 192 B of headroom under the budget.
* **The allocation alone is far worse than uniform int4.** Untrained, the allocated configuration
  scores 64.5477 against 18.6193 — 3.5× worse perplexity at the same bytes. A sensitivity-guided
  allocation that mixes 2- and 3-bit into most layers is a *bad* allocation for a 135 M model at this
  budget, and the honest reading is that the proxy's ranking did not translate into a better
  configuration here.
* **Training the factors recovers most of that damage but not enough to beat uniform int4.**
  64.5477 → 23.1698 (a 2.8× reduction) still leaves **+24.4 %** over the uniform baseline at the same
  bytes, and the method's own configuration (regularizer on) is **23.3131, +25.2 %**. **The central
  hypothesis — that the joint low-rank/low-bit method beats equal-memory uniform quantization — is NOT
  supported.**
* **The quantizer ordering is clean.** int8 costs +0.17 % for 1.88× fewer bytes; int4 costs +32.8 %
  for 3.56×. That is the reference behaviour the method has to beat, and it is measured on the same
  substrate, model, corpus and token count as the method.

## 3. The method's own arm, and the regularizer ablation

Both configurations are now collected, validated runs on the same substrate, model, corpus, token
count, seed and byte budget:

| configuration | λ_factor | λ_round | perplexity | vs uniform int4 | vs the other |
|---|---|---|---|---|---|
| allocation + training, **regularizer ON** | 1.0 | 1.0 | **23.3131** | +25.2 % | — |
| allocation + training, **regularizer OFF** | 0.0 | 0.0 | **23.1698** | +24.4 % | **ON is 0.1433 worse (+0.62 %)** |
| allocation, untrained | — | — | 64.5477 | +246.7 % | — |

The first collected run had no `regularizer` block in the plan, so the objective defaulted to an exact
differentiable zero and the arm named `spectraquant_regularized` measured the *unregularized* ablation
(**A-0020**). The plan now declares the project's own `regularizer_full.yaml` coefficients, and the
second run measures the method with its mechanism on: `regularizer.is_noop = False`,
`lambda_factor = lambda_round = 1.0`, dev loss 3.5551 at step 300 against the unregularized arm's
3.5509.

**The regularizer's contribution is negative**: enabling it costs **0.1433 perplexity (+0.62 %)**. That
is the same sign as the Tier-0 fixture finding (A-0008: the term reduces the rounding residual but
worsens dev NLL), now measured at the model scale the method is claimed for. The mechanism is
implemented, differentiable (the penalty is attached to the device factors by a manual vector–Jacobian
product, so it cannot silently no-op) and **does not help**.

## 4. What may not be claimed from this document## 4. What may not be claimed from this document

* No claim that the method beats or matches uniform int4 — the method's own arm has not run.
* No multi-seed claim: every row is **one seed**, and §5.1 of the reproduction plan requires all seeds
  of an arm set before a direction is "confirmed". This is a **pilot-scale** result.
* No latency, throughput or speedup claim: nothing here is a class-4 measurement.
* No claim about the deployed model size: the byte figures cover the **210 targeted projection
  tensors** (106 168 320 parameters), not the embedding or the LM head.
* The `proxy_allocated` row is *not* an equal-memory competitor in the method's sense — it is the
  same allocation without training, i.e. the ablation that isolates training.

## 5. Reproducing

```bash
spectraquant cloud notebook --plan configs/tier1/smollm2_135m.yaml --platform kaggle \
  --runner-arg --max-tokens=20000 --runner-arg --max-documents=200 \
  --runner-arg --granularity=per_group --runner-arg --group-size=32 \
  --runner-arg --budget-bytes=59719680
spectraquant cloud submit  --spec notebooks/generated/tier1_smollm2_135m-cloud.ipynb.spec.json
spectraquant cloud fetch   --run-id tier1_smollm2_135m-cloud --dest <dir>
spectraquant cloud collect --run-id tier1_smollm2_135m-cloud --source <dir>
```

The bundle's own `metrics.json` and per-arm `run_manifest.json` are the authority for every number
above; `reports/tables/cloud-arms.md` is generated from them by
`scripts/reproduce/generate_release_artifacts.py`.
