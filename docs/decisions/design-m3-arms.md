# Design note — the frozen M3 arm set in the cloud runner

Status: **proposed** (implementation not started). Required by `AGENTS.md` §3: this interface is shared
between the plan schema, the cloud runner and the quantization/factorization primitives, so it is
specified in writing before it is implemented.

## 1. Why this note exists

`docs/research/reproduction-plan.md` §3 predeclares eight arms for M3. **None of them is implemented.**
What the runner has today (`fp16_reference`, `ptq_uniform`, `low_rank_only`, `rank_then_quant`, plus the
`loftq` and `lr_qat` arms added on 2026-10-09) uses the plans' *default* grid — rank 8, uniform 4-bit
per-group-32 — which is **not** the predeclared configuration. Every "M3" number produced so far is
therefore a diagnostic, not an M3 cell (`docs/coordination/claim-contradictions.md`). This note fixes
the mapping from the frozen text to code so the gap cannot be re-opened by improvisation.

## 2. The frozen arms, verbatim

### R1 — LoftQ (trend T2), rank 16 for the model-level arms
| arm kind | base weights | adapter initialisation |
|---|---|---|
| `r1_fp16_lora` | fp32, no quantization | standard (Kaiming A, B = 0) |
| `r1_std_2bit` | 2-bit NF-style codebook, block 64 | standard (Kaiming A, B = 0) |
| `r1_loftq_2bit` | the **same** quantized matrix `Q` | LoftQ init, `T = 5` |
| `r1_loftq_2bit_t1` | the **same** `Q` | LoftQ init, `T = 1` |

The two `2bit` arms must share `Q` exactly — the trend is about the *initialisation*, so the base must
be identical across them. Target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj,
down_proj`. Optimiser AdamW, identical step count, identical learning-rate grid.

### R2 — LR-QAT (trend T1), 4-bit symmetric, group size 128, RTN scales
| arm kind | weights | trainable |
|---|---|---|
| `r2_fp16` | fp32 | none (eval only) |
| `r2_rtn_4bit` | 4-bit g128 RTN | none (eval only) |
| `r2_lrqat_4bit` | 4-bit g128; `Φ₀` downcast to fixed-point Q4.4 | low-rank `A, B` (rank 32) **plus** the step size `s` |
| `r2_fullqat_4bit` | 4-bit g128 | **all** linear weights |

Recipe (LR-QAT Table B1): AdamW, `β = (0.9, 0.95)`, weight decay 0 for `A, B` and `s`, linear warmup +
linear decay with 10 % warmup, 1000 steps, batch 512 tokens, gradient clipping max-norm 1.0,
`α = 1.0`, embeddings / LM head / RMSNorm frozen.

## 3. Interface

### 3.1 Quantization
* `r1_*` arms use the **codebook** quantizer (`spectraquant.quantization.codebook`,
  `CodebookSpec(bits=2, block_size=64, kind="nf")`), whose level table and block semantics are pinned
  against the committed oracle fixture (`artifacts/sample-results/m3-oracle/loftq-nf2-block64.json`,
  amendment A-0013).
* `r2_*` arms use the existing uniform `QuantSpec(bits=4, granularity="per_group", group_size=128,
  symmetric=True)`.
* Byte accounting stays in `spectraquant.quantization.accounting` — one source of truth. A codebook
  container's payload is `ceil(numel * bits / 8)` plus one scale per block; the codebook module
  delegates there and documents the rule when the central helper cannot express the container.

### 3.2 Trainable representation
`r1_*` arms and `r2_lrqat_4bit` use
:class:`~spectraquant.training.low_rank.QuantizedPlusLowRankLinear` (frozen base buffer + trainable
factors), which already exists. `r2_fullqat_4bit` needs the **dense** linear layers to stay trainable
with a straight-through quantized forward: a new layer is required —
`StraightThroughQuantizedLinear` — whose forward quantizes the weight, dequantizes it, and passes the
gradient through the quantizer unchanged (`∂L/∂W` of the dequantized weight). It must be a *separate*
class, not a flag on the existing one, because the trainable parameter set differs (all weights vs
factors only).

### 3.3 The learned step size `s` (LR-QAT only)
`r2_lrqat_4bit` optimises `s` in addition to `A, B`. It is a scalar parameter, initialised from the
RTN scale of `Φ₀`, trained with `lr_s` searched in `{0, 1e-5}` where `lr_s = 0` means **frozen** (the
`0` grid point is a real arm, not a missing value). The merge that consumes it exists
(`merge_lr_qat`, the T1-a gate); the runner must record `s` per layer in the manifest.

### 3.4 Learning-rate search
Both designs search their grid on the **validation** split only and then run the chosen rate on the
test split once. The search is a separate, recorded step: `training.lr_search` in the manifest lists
every candidate and its validation perplexity, and `training.learning_rate` is the selected one.
Searching on the test split is a contract violation (`AGENTS.md` §4.6) and the manifest must make the
split explicit.

### 3.5 Plan shape
A new plan per model size — `configs/m3/tier1_smollm2_135m.yaml` and (later)
`configs/m3/tier2_tinyllama_1_1b.yaml` — declaring the eight arms above, the frozen seeds
(`{0,1,2,3,4}` pilot, `{0,1,2}` primary), the lr grids, ranks (16 for R1 model-level, 32 for R2),
`group_size: 128`, `block_size: 64`, and the 1000-step schedule. The existing
`configs/{tier1,tier2,repro}` plans are **not** repurposed: their arms are the project's own
comparator grid, and their recorded runs must keep their labels.

## 4. Invariants the implementation must hold

1. **Same `Q` across the R1 2-bit arms.** A test must assert that `r1_std_2bit` and both `r1_loftq_*`
   arms see a bit-identical quantized base for the same seed.
2. **No state leakage between arms** — each arm gets its own model copy (already enforced, with a
   regression test).
3. **The test split is never read during training or search.** Only dev (WikiText-2 validation).
4. **`lr_s = 0` means frozen**, and the manifest distinguishes it from "not applicable".
5. **Byte accounting is class 1** (analytical) unless a container is serialized, in which case the
   class-3 figure is recorded beside it and the equal-memory gate is applied.
6. **A run that cannot reach the declared steps is `failed`**, never a silently shorter run.
7. **Divergence rule (predeclared, §5.4):** perplexity `> 1000`, NaN, or a loss not decreasing after
   100 steps marks the arm divergent; the divergence *rate* then becomes that trend's primary
   endpoint. The runner must detect and record this rather than emit the number.

## 5. Acceptance criteria

1. All eight arms run end to end on a tiny synthetic fixture locally (no network, no GPU), each writing
   a schema-valid manifest with its own compression block and training provenance.
2. `r1_std_2bit` vs `r1_loftq_2bit` differ **only** in the adapter initialisation: a test asserts the
   base bytes are identical and the initialisation output error differs in the expected direction
   (LoftQ lower).
3. The lr search writes every candidate's validation perplexity and selects on validation only; a test
   proves the test split is not touched during the search.
4. `r2_lrqat_4bit`'s learned `s` changes during training when `lr_s > 0` and is bit-identical when
   `lr_s = 0`.
5. The T1-a and T2-a exactness gates (separate workstream) pass before any M3 cloud run starts
   (`reproduction-plan.md` §10.3(a)).

## 6. Open questions for the owner

1. **`r2_fullqat_4bit` cost.** It trains all linear weights: the plan's own cost table budgets 1–3 h
   of the cloud envelope for "failed/aborted attempts (pre-emption, OOM on the full-QAT arm)". On a T4
   with 135 M parameters it should fit, but the arm is the most likely to OOM at a larger model. Run it
   at Tier 1 first and record the peak memory before Tier 2 is attempted.
2. **The 2-bit arms' rank.** The plan fixes rank 16 for model-level arms and mentions rank 64 for the
   initialisation-level arms; the model-level run uses 16 only (a declared deviation). Confirm that the
   initialisation-level rank-64 arms are out of scope for this release.
