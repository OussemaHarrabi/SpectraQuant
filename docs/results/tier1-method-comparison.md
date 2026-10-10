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
| `spectraquant_regularized` (**allocation + training, regularizer OFF** — see §3) | **23.1698** | 59 719 488 | 59 719 488 | 3.56× | **+24.4 %** |
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
  bytes. **The central hypothesis — that the joint low-rank/low-bit method beats equal-memory uniform
  quantization — is NOT supported by this run.**
* **The quantizer ordering is clean.** int8 costs +0.17 % for 1.88× fewer bytes; int4 costs +32.8 %
  for 3.56×. That is the reference behaviour the method has to beat, and it is measured on the same
  substrate, model, corpus and token count as the method.

## 3. Which arm is which — the labelling that matters

The collected run's `spectraquant_regularized` arm recorded
`regularizer.lambda_factor = lambda_round = lambda_residual = lambda_spectrum = 0.0` in its own
manifest. The objective's coefficients default to an exact differentiable zero and the Tier-1 plan
declared no `regularizer` block, so **the arm ran with the method's mechanism switched off**: it is the
*allocation + unregularized training* ablation, not the method. Amendment A-0020 records this, the plan
now declares the coefficients (λ_factor = 1.0, λ_round = 1.0, the project's own
`regularizer_full.yaml`), and the method arm is being re-run so the **regularizer on/off pair** is
reported at equal measured bytes. Until that run is collected, **no number in this document is the
method's**; the row is the ablation and is labelled as such.

## 4. What may not be claimed from this document

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
