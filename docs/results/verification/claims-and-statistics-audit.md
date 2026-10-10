# Claims and statistics audit — cloud phase, 2026-10-10

Auditor: orchestrator (the delegated review was cancelled by the user's instruction and the review
was done directly). Scope: the claims added **since** `docs/results/verification/claim-audit.md`
(2026-10-09, which covers the fixture era and found no headline-number contradiction). Every claim
below is quoted from the committed text; every judgement names the artifact or code path that settles
it. Read-only except this file.

Method: read the public documents (`reports/paper/paper.md`, `README.md`,
`docs/coordination/status.md`, `docs/results/**`), then for each new claim locate its artifact
(`artifacts/sample-results/**`, `artifacts/runs/registry.jsonl`, the collected bundles under
`artifacts/runs/*/download/`) or the test that asserts it. Checks 2–7 were run as searches, not as
impressions; the search is named in each section.

---

## 1. Verdicts on the claims that changed

| # | Claim (quoted, abbreviated) | Source | Evidence | Class / substrate | Verdict |
|---|---|---|---|---|---|
| N1 | "fp16 reference on SmolLM2-135M / WikiText-2 test: perplexity 14.0181" | `docs/results/tier1-cloud-run-2026-10-09.md` | collected bundle, `tier1_smollm2_135m-cloud` `validated` | 2 / CLOUD-CPU | **supported-with-caveat** — one seed, 21 758 tokens; the doc says "one seed" and "not a confirmatory cell" |
| N2 | "int8 PTQ costs +0.17 % perplexity for 1.88× fewer accounted bytes" | same | same | 2 + 1 / CLOUD-CPU | **supported-with-caveat** — bytes are class-1 analytical over 210 targeted tensors, not a deployed-model size (the doc says so) |
| N3 | "untrained rank-8 truncation is degenerate (1.40e16; mean relative Frobenius 0.9427)" | same | same | 2 / CLOUD-CPU | **supported** — the degenerate flag is in the manifest and travels into `metrics.json` |
| N4 | "training improves both arms … LoftQ 17.793 → 16.6936, LR-QAT 17.722 → 17.0694" | `docs/results/m3-reproduction-2026-10-09.md` | collected bundle, `repro_lr_qat_loftq_smollm2_135m-cloud` `validated` | 2 / CLOUD-GPU | **supported-with-caveat** — a **single run per arm**, and the doc labels the whole table a diagnostic, not an M3 cell |
| N5 | "measured training cost: 2.17 s per optimiser step at batch 8 × seq 512" | same | same (433 s / 200 steps) | 1 / CLOUD-GPU | **supported** — a wall-clock measurement with its shape recorded |
| N6 | "the frozen protocol's eight arms now exist and their local prerequisites pass" | `reports/paper/paper.md` §8 | `tests/unit/test_m3_arms.py`, `tests/unit/test_m3_exactness.py`, `tests/unit/test_codebook.py` | 2 / LOCAL | **supported** — 22 + 8 + 70 tests, each asserting the frozen tolerance verbatim |
| N7 | "the merge identity is exact on the integer path and 5.03e-07 against the frozen 1e-5; residual-dominance 21/21" | same | `tests/unit/test_m3_exactness.py` | 2 / LOCAL | **supported** — the frozen constants appear in the test |
| N8 | "the oracle gate … agreeing on the quantized weight exactly at T = 1 and on the residual norm and spectrum to 1.5e-07 and 4.2e-07" | same | `artifacts/sample-results/m3-oracle/loftq-nf2-block64.json` | 2 / LOCAL | **supported-with-caveat** — A-0015 records that the *elementwise* comparison could not reach 1e-6 and is asserted at 1e-3 as a convention detector |
| N9 | "Nothing here is a state-of-the-art claim." | `reports/paper/paper.md` line 7 | — | — | **supported** — see §7 |
| N10 | "the released commit installs and passes from a clean clone" | `docs/results/experiment-index.md` | clone of `research/tier1-closure` @ `3f704e9`; 1133 passed / 2 skipped | — | **supported** — reproduced by the auditor at `3f704e9`; the skips are the PEFT cross-check and the inert dirty-tree guard |

No claim in the changed set is `unsupported` or `misleading` as written. Two are marked
`supported-with-caveat` for reasons the documents already state; one caveat (N4) is the one I would
strengthen (finding F2 below).

---

## 2. Statistical unit and power

Search: `grep -n "statistical unit|independent|n_eff|MDE"` over `docs/results/**` and
`reports/paper/paper.md`.

* The proxy result's unit is stated and correct: "Statistical unit: the **model** (one `rho` per
  trained model per cell); `n_eff` per model from a within-model module bootstrap … combined with a
  DerSimonian-Laird Fisher-z random-effects model; contrast CIs additionally bootstrapped over models"
  (`docs/results/proxy-validation-report.md` line 20). **No pseudoreplication found**: seeds and
  modules are never treated as independent model-level replicates.
* The power caveat is stated in the source, not hidden: "the candidate's own paired contrast is
  coarser: its achieved MDE is +0.968 (pooled) … *wider* than the predeclared conservative end (0.79).
  The correct reading is 'supported at fixture scale, underpowered relative to the predeclared band
  for this particular contrast'" (line 11).
* **Finding F3 (caveat, not blocking).** The paper's abstract-level sentence for that result
  ("beats every predeclared comparator in 15 of 15 aggregate Fisher-z contrasts … model-level,
  5 seeds × 12 cells × 2 fixtures") does not repeat the underpowering caveat; the caveat appears in
  §"Power" and in the report. A reader of the abstract alone would take the contrast as
  as-planned-powerful. Smallest fix: append ", at an achieved MDE wider than the predeclared
  conservative end for the candidate's own contrast" to that sentence.
* No population-level claim is made from a single model: the cloud claims are explicitly scoped to
  one model (SmolLM2-135M), one seed, and are labelled pilot/diagnostic.

## 3. Leakage

Search: `grep -n "test split|test_perplexity|validation"` over `src/spectraquant/cloud/**` and the
M3 tests; read `_training_corpus`, `_search_learning_rate` and the arm tests.

* `plan_runner._training_corpus` reads the plan's `train` role for training and its `development`
  role for dev; the plan validator refuses a dataset that carries both the test and a
  train/calibration role (`tests/unit/test_experiment_plan.py`, asserted for the M3 plan).
* The learning-rate search is asserted to select on dev only, with every candidate's dev perplexity
  recorded (`training.lr_search`), by `tests/unit/test_m3_arms.py`.
* The test split is read once, after training, by `token_level_perplexity`.
* **No leakage found.** The one path worth naming as a residual risk: the *plan* chooses which split
  each role maps to (`ROLE_SPLITS`), so a future plan could point `development` at a test split. That
  mapping is a constant in the runner and is covered by the plan-level validator above, not by a
  runtime assertion; adding an assertion that the dev split's dataset name differs from the test
  split's would close it. Recorded as finding F4 (hardening, not blocking).

## 4. Equal memory

Search: read every table in `docs/results/**` and `reports/paper/paper.md` that places two
compression configurations side by side.

* The Tier-1 comparator table (fp16 212.3 MB / int8 112.8 MB / int4 59.7 MB) is a **frontier, not a
  comparison**: its rows are deliberately at different byte counts, and the doc states that the byte
  column "compares like with like but is not a deployed-model size" and that the arms are the
  "equal-memory-relevant comparators the method has to beat". **Finding F1 (caveat):** the sentence
  should say explicitly that no comparison is claimed *between* rows at different byte counts —
  `AGENTS.md` §4.5 requires an equal-memory counterpart for a comparison, and a reader could take the
  table as one. Smallest fix: one sentence.
* The M3 diagnostic table already handles the case correctly: "`rank_then_quant` … is not a fair
  equal-memory point against the trained arms (64.6 MB) — it is a *lower-bound* artifact and is
  reported as such."
* Byte *sources* are never mixed silently: the class-1 (accounted) and class-3 (measured) figures are
  labelled per row, and the equal-memory gate refuses a mixed-source pair (asserted in CI).

## 5. Measurement-class mixing

Search: every table header and every claim sentence for a class label.

* Each cloud table carries a per-row class column; the local tables name their class in the text.
* The class-4-CPU row is labelled 4-CPU and states "no speedup claimed"; the paper says "Nothing here
  is a state-of-the-art claim" and "no latency/throughput claim".
* **No mixing found.** The one recurring risk is that the *frontier* table's class-1 bytes sit next to
  class-2 quality numbers; the table labels both, which is the required behaviour.

## 6. Unreported exclusions and failures

Search: `artifacts/runs/registry.jsonl` (every run id and state) against the reports and the incident
log in `docs/coordination/status.md`.

* Every non-success is in the registry with a reason: the seven early Tier-1 submissions
  (`rejected`/`resubmitted` cycles), the collector-rejected bundle, the operator-stopped C4 run
  (`failed`, with the reason), and the M3 seed-0 attempt whose artifacts expired before collection
  (`failed`, "artifacts lost … the session output was fetched ~7h later, when Kaggle had expired it").
* **No unreported exclusion found.** One honesty note the registry itself records and this audit
  confirms: the M3 seed-0 run *may have succeeded* and is nevertheless claimed as nothing, because a
  run without a collected artifact does not exist. That is the correct treatment, not a gap.

## 7. Novelty language

Search: the claim guard's own forbidden-novelty pattern list (`src/spectraquant/reporting/claim_guard.py`,
`NOVELTY_RE`) grepped over the public documents, plus a manual read of every abstract-level sentence.

* The only occurrences in the paper are negations ("Nothing here is a state-of-the-art claim") or
  pointers to the novelty-risk document. The M1 verdict is quoted with its narrow scope
  ("differentiated only through A"). **No overclaiming found.**

---

## 8. Ranked issues

| Rank | Issue | File | Smallest correct fix | Blocks release? |
|---|---|---|---|---|
| 1 | F2 — the diagnostic's direction sentence ("training improves both arms") is a single run per arm and could be read as a finding | `docs/results/m3-reproduction-2026-10-09.md` | say "in this single run per arm" in the sentence itself, not only in the table header | no (it is labelled a diagnostic), but fix before release |
| 2 | F1 — the comparator table is a frontier; no equal-memory *comparison* is claimed between rows | `docs/results/tier1-cloud-run-2026-10-09.md` | add one sentence saying so | no |
| 3 | F3 — the abstract-level proxy sentence omits the underpowering caveat | `reports/paper/paper.md` | append the caveat clause | no |
| 4 | F4 — the role→split mapping is a runner constant; a future plan could point `development` at a test split | `src/spectraquant/cloud/plan_runner.py` | assert the dev dataset name differs from the test dataset name | no |

No blocking issue found in the changed claim set. The four items are documentation or hardening; none
changes a number, a scope or a verdict.
