# Tier-1 cloud pilot, 2026-10-09 — first validated run

**Run id** `tier1_smollm2_135m-cloud` · **platform** Kaggle Notebooks (CPU, free tier) ·
**code commit** `9e7c8768b31a91ec726cedfdc832cfc8ccd04743` (clean tree) ·
**registry state** `validated` (4 artifacts checksum-verified) ·
**collection** `spectraquant cloud fetch` + `spectraquant cloud collect` on the downloaded bundle.

This is the first run in the project whose artifacts were downloaded, checksum-validated and recorded
from the platform. It is a **pilot slice**, not a confirmatory cell: the trainable arms (H1–H4, H3,
and the H5 quality half) are still **NOT RUN**, because they need the M5 training loop and a GPU
substrate. One seed (`0` of the predeclared `[0, 1, 2, 3, 4]`), 20 000-token cap, 200 documents.

## 1. What was measured

Model `HuggingFaceTB/SmolLM2-135M` @ `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` (134 515 008
params), loaded from the hub in float32. Corpus `EleutherAI/wikitext_document_level` config
`wikitext-2-raw-v1` @ `647234772b9554e208af6c826f23b99e3cac88c8`, split `test`. Protocol
`non-overlapping-window-token-ce-v1`, `seq_len` 2048, 5 documents, 12 windows, 21 758 scored tokens.
CPU, single-threaded; 673 s wall for the four arms.

**Compression scope.** Every arm accounts the same 210 tensors / 106 168 320 parameters — the
attention and MLP projection matrices the plan targets. The token embedding and the LM head
(≈ 28 M params) are **outside** the accounted scope in every arm, including the fp16 reference, so
the byte column below compares like with like but is **not** a deployed-model size. `bytes.
accounted_class` is `1` (analytical estimate from shapes and bit widths); no packed artifact was
serialized in this slice, so `bytes.measured_class` is `null`.

| arm | perplexity | accounted bytes | vs fp16 bytes | rel. Frobenius (mean) | class | degenerate |
|---|---|---|---|---|---|---|
| `fp16_reference` | 14.0181 | 212 336 640 | 1.00× | 0.0000 | — | no |
| `ptq_uniform_8` (per-group, g=32) | 14.0412 | 112 803 840 | 1.88× | — | 2 | no |
| `ptq_uniform_4` (per-group, g=32) | 18.6190 | 59 719 680 | 3.56× | 0.1039 | 2 | no |
| `low_rank_only` (rank 8) | 1.397e16 | 4 884 480 | 43.5× | 0.9427 | 2 | **yes** |
| `rank_then_quant` (rank 8, 4-bit) | 4.649e19 | 1 598 400 | 132.9× | 0.9440 | 2 | **yes** |

**This table is a frontier, not a comparison.** Its rows sit at deliberately different byte counts, and
no comparison is claimed *between* rows: `AGENTS.md` §4.5 requires an equal-memory counterpart for any
compression comparison, so each of these numbers is a point on the quality–memory plane that a
comparison must be built against at matched bytes — not a verdict that one row beats another.

This is the complete non-trainable comparator matrix of the plan: the only arms the runner skips
are the three that are not implemented in this slice (M4 `quant_then_residual`, M5 `proxy_allocated`,
M5 `spectraquant_regularized`). Wall time for the five arms: 807 s.

Class 2 is *fake-quantization quality*: float execution that simulates quantization numerics. No
latency, throughput or packed-storage claim is made here; none of these numbers is a class 3 or
class 4 measurement.

## 2. Reading

* **int8 PTQ is nearly free at this scale; int4 is not.** int8 costs +0.17 % perplexity (14.0412 vs
  14.0181) for 1.88× fewer accounted bytes — the cheap point of the frontier. int4 costs +32.8 %
  (18.6190) for 3.56× fewer bytes, with a 10.4 % mean relative Frobenius weight error. Both are the
  equal-memory-relevant comparators the method has to beat, and both are now measured rather than
  assumed.
* **Uniform rank-8 truncation destroys this model.** The mean relative Frobenius error is 0.94 —
  rank 8 retains 8 of 576 singular directions in a 576-wide matrix — and the perplexity is not a
  quality number at all (1.4e16, flagged `perplexity_degenerate: true`). The two low-rank arms are
  therefore **negative results for untrained factorization**, which is the expected outcome and the
  reason the project's trainable arms (M5: low-rank preparation, LR-QAT, LoftQ) exist. It also
  confirms that the runner must never present such a number as a quality measurement; the flag now
  travels into `metrics.json`.
* **The plan's rank grid is far below the viable region.** With a 0.94 relative error at rank 8, the
  useful ranks for a 135 M model are much larger (or the recovery training is essential). This is a
  finding about the *grid*, recorded before the trainable arms run, so it is not a post-hoc choice.

## 3. Arms that did not run, and why

The runner refuses to guess a grid point and reports the reason instead:

| arm | reason |
|---|---|
| `quant_then_residual` | not implemented in this slice (M4) |
| `proxy_allocated` | not implemented in this slice (M5) |
| `spectraquant_regularized` | not implemented in this slice (M5) |

An earlier collected attempt (the one recorded in §4 as exposing defects 9 and 10) ran only four
arms: `ptq_uniform_8` was skipped because the invocation's `--bits=4` conflicted with the width in
the arm's own name. That is why the arms' grid points now live in the plan (`ArmSpec.point`) instead
of in the invocation — the run recorded above needed no `--rank`/`--bits` flag at all.

## 4. Defects this run exposed

Every one of these was invisible to the 984-test suite, because the suite drives a fake runner. They
were found by executing the real path end to end and reading what came back.

| # | Defect | Evidence | Fix |
|---|---|---|---|
| 1 | Generated notebook imported the repository *before* installing it (`ModuleNotFoundError: spectraquant`) | attempt 1 | install precedes every import |
| 2 | The `models` extra (torch/transformers/datasets) was never installed on the platform | attempt 2 | install stage builds the pinned environment |
| 3 | Notebook template had a syntax error in the run cell | attempt 3 | fixed and compiled by a test that compiles every generated cell |
| 4 | The runner exited 1 with no reason recorded | attempt 4 | `run-plan` records the failing stage and the error |
| 5 | The teardown cell referenced names the stages own (`RUN_STATUS`, `MANIFEST_PATH`) | attempts 5–6 | teardown re-reads the stage context; template v2.0.1 |
| 6 | The plans pinned the dataset repository and revision but **not the config**, so the loader died with `Config name is missing` | attempt 5 | `DatasetRole.config` is required for a perplexity dataset and resolved from the plan |
| 7 | The registry stored the **redacted** remote id (`[REDACTED:KAGGLE_USERNAME]/…`), so every later platform call was denied `kernels.get` | first collection | `KAGGLE_USERNAME` is an identifier, not a secret; only real secrets are scrubbed |
| 8 | The collector picked the alphabetically first `run_manifest.json`, which was a nested **per-arm** manifest, and rejected a valid bundle | first collection | candidates are ranked by depth; the run's manifest is the shallowest |
| 9 | `--bits` silently overrode the width in an arm's name: `ptq_uniform_8` and `ptq_uniform_4` produced identical perplexity (18.6190) and identical bytes (59 719 680) | attempt 7 | the arm name is authoritative; a conflict is an error |
| 10 | A degenerate perplexity (1.4e16) was recorded like any other number | attempt 7 | `perplexity_degenerate` flag; a non-finite loss is a failed measurement |

## 5. Reproducing

```bash
spectraquant cloud notebook --plan configs/tier1/smollm2_135m.yaml --platform kaggle \
  --runner-arg --max-tokens=20000 --runner-arg --max-documents=200 \
  --runner-arg --granularity=per_group --runner-arg --group-size=32 \
  --runner-arg --rank=8 --runner-arg --bits=4
spectraquant cloud submit --spec notebooks/generated/tier1_smollm2_135m-cloud.ipynb.spec.json
spectraquant cloud fetch  --run-id tier1_smollm2_135m-cloud --dest <dir>
spectraquant cloud collect --run-id tier1_smollm2_135m-cloud --source <dir>
```

The bundle itself is not committed (it contains the model-independent artifacts under
`artifacts/runs/`, which is gitignored); the registry line and this report are the record.
