# M3 verdict — resource-constrained Tier-1 subset, 2026-10-10

**Run** `m3_tier1_smollm2_135m-cloud` · **status** `success` (registry: `validated`) · **code commit**
`aee3d921c702` · **substrate** Kaggle Tesla T4 · **model** `HuggingFaceTB/SmolLM2-135M` @
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2` · **corpus** `EleutherAI/wikitext_document_level` config
`wikitext-2-raw-v1` @ `647234772b9554e208af6c826f23b99e3cac88c8`, split `test` · **21 758 scored
tokens** (the same basis as `docs/results/tier1-method-comparison.md`, so the two are directly
comparable) · **seed** 0 · **1000 optimiser steps**, batch 8, seq 512, warmup 100 · learning rate
selected from the frozen grid `{1e-5, 5e-5, 1e-4, 3e-4, 1e-3, 1e-2}` **on the validation split**
(selected `1e-2`) · **class 2** quality, **class 1** bytes.

## 1. What was executed, and what was not

Amendment **A-0018** fixes the subset: the original frozen protocol (5 seeds × 1000 steps × 8 arms) is
**preserved unchanged** and **not executed**; the executed subset is the minimum arm set that can
decide trend **T1**, chosen for scientific necessity, with the comparison and its criteria fixed
*before* the run. **T2 (LoftQ), the `fullQAT` upper-bound arm, seeds 1–4 and Tier-2 are NOT
executed.** No wording here may imply the reduced experiment completed the frozen protocol.

## 2. The measured arms

| arm | perplexity | accounted B | measured B | steps | role |
|---|---|---|---|---|---|
| `r2_fp16` | **14.0183** | 212 336 640 | — | — | the fp16 reference (eval only) |
| `r2_rtn_4bit` | **26.9432** | 54 858 240 | 54 858 240 | — | the PTQ baseline: 4-bit symmetric, g128, RTN scales (eval only) |
| `r2_lrqat_4bit` | **24.1265** | 74 397 000 | 74 397 000 | 1000 | the treatment: 4-bit g128 base + rank-32 low-rank auxiliary + a learned step size, `Φ₀` in Q4.4 |

All three ran on the same model, corpus, split, tokenisation and token count, with the same seed
pairing, in **one** invocation.

## 3. The frozen criteria, applied verbatim

§5.2 defines the T1 magnitude requirement as `PPL(RTN) − PPL(LR-QAT) ≥ 0.25 · G_ptq` with
`G_ptq = PPL(RTN) − PPL(FP16)`, and the upper bound as
`PPL(LR-QAT) ≤ PPL(fullQAT) + max(0.1, 0.1·PPL(fullQAT))`.

```
G_ptq   = 26.9432 − 14.0183 = 12.9249
closure = 26.9432 − 24.1265 =  2.8167
fraction = 2.8167 / 12.9249 =  0.2179  ->  21.8 %
```

| criterion | threshold | measured | verdict |
|---|---|---|---|
| §5.2 direction: `PPL(LR-QAT) < PPL(RTN-4bit)` | strict | 24.1265 < 26.9432 | **holds** |
| §5.2 magnitude: closure ≥ 25 % of `G_ptq` | ≥ 0.25 | **0.2179** | **NOT met** |
| §5.2 upper bound vs `fullQAT` | ≤ `PPL(fullQAT)·1.1 + 0.1` | `fullQAT` not run | **not assessable** |
| §5.1 confirmation: the sign in **all** seeds of the arm set | 5 of 5 at the pilot | 1 seed | **not reachable** — 1/1 |
| §5.4 divergence: PPL > 1000, NaN, or loss flat after 100 steps | — | none of the three arms | no divergence |

## 4. Verdict

**T1 (LR-QAT): partially reproduced — the direction holds, the predeclared magnitude does not.**

* The treatment beats the PTQ baseline by **2.82 perplexity** at equal bit width and equal group size,
  which is the *direction* the published trend predicts.
* That closure is **21.8 %** of the damage PTQ does, against the predeclared **25 %** requirement: it
  falls **3.2 percentage points short**, and the threshold cannot be revised after seeing this
  (reproduction-plan §5.2 point 4).
* With **one seed** the result cannot satisfy §5.1's all-seeds requirement, so it is **suggestive at
  pilot scale**, not "confirmed", and the frozen rule that the **Tier-2 primary result carries the
  claim** is untouched — Tier-2 is not executed.

**T2 (LoftQ): NOT EXECUTED.** No T2 number exists, and none is implied. The §5.4 divergence endpoint
observed in an *unclaimable* run (whose bundle failed validation at export, so under §10.3(d) it does
not exist) is recorded in A-0018 only as the reason this subset was prioritised, never as a result.

**Bytes caveat that belongs with the T1 number.** `r2_lrqat_4bit` costs **74 397 000 B** against
`r2_rtn_4bit`'s **54 858 240 B** — **+35.6 %** — because its stored object is the 4-bit base *plus* the
rank-32 factors. The frozen §5.2 criterion is defined at **equal bit width**, and that is what is
reported; the T1 result is therefore **not** an equal-memory result, and a frontier claim would need
the treatment re-solved at the baseline's bytes.

## 5. What may not be claimed from this document

* That the frozen M3 protocol was executed, or that the reproduction is complete: it is a
  resource-constrained subset (A-0018).
* That T1 is "confirmed" or "reproduced": the magnitude criterion failed and the seed floor was not met.
* Anything about T2, `fullQAT`, Tier-2, or a population-level effect.
* Any latency, throughput or speedup number: nothing here is class 4.
* That the method beats uniform quantization: that is the Tier-1 method comparison, and it is **also
  negative** at equal bytes (`docs/results/tier1-method-comparison.md`).

## 6. Reproducing

```bash
spectraquant cloud notebook --plan configs/m3/tier1_smollm2_135m.yaml --platform kaggle \
  --runner-arg --max-tokens=20000 --runner-arg --max-documents=200 \
  --runner-arg --arms=r2_fp16,r2_rtn_4bit,r2_lrqat_4bit
spectraquant cloud submit  --spec notebooks/generated/m3_tier1_smollm2_135m-cloud.ipynb.spec.json
spectraquant cloud fetch   --run-id m3_tier1_smollm2_135m-cloud --dest <dir>
spectraquant cloud collect --run-id m3_tier1_smollm2_135m-cloud --source <dir>
```

The subset is a *subset*: reproducing the frozen protocol means dropping `--arms` and running five
seeds, which costs ≈ 19.5 GPU-hours.
