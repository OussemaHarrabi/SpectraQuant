# Claim contradiction audit — 2026-10-09/10

Every row compares a statement in a project document against the run registry and the collected
artifacts. "Evidence" names the artifact that settles it. The rule applied throughout: **a run without
a validated `run_manifest.json` does not exist** (`docs/research/reproduction-plan.md` §10.3(d)), and a
pilot is never promoted to a confirmatory result.

Registry states at audit time (`artifacts/runs/registry.jsonl`, append-only):

| run id | substrate | registry state | what it is |
|---|---|---|---|
| `tier1_smollm2_135m-cloud` | Kaggle, CPU | `validated` | Tier-1 **comparator slice**: fp16, int8 PTQ, int4 PTQ, two untrained low-rank arms; 1 seed |
| `repro_lr_qat_loftq_smollm2_135m-cloud` | Kaggle, Tesla T4 | `validated` | **trainable-arm diagnostic** (not the frozen M3 protocol): `lr_qat`, `loftq`, fp16, `rank_then_quant`; 1 seed; 300 steps |

## Contradiction table

| # | file | statement | evidence | correct replacement |
|---|---|---|---|---|
| 1 | `README.md` L16-20 | "Every Tier-1+ cloud cell — and the bounded LR-QAT/LoftQ reproduction — is **NOT RUN**" | registry: two runs `validated`; `docs/results/tier1-cloud-run-2026-10-09.md` | Tier-1 **comparator slice** and the trainable-arm **diagnostic** have run and are validated; the frozen M3 protocol and the Tier-1 confirmatory cells (H2/H3/H4) have **not** |
| 2 | `README.md` L90 | "TinyLlama-1.1B and SmolLM2-135M are pinned but **not loaded or evaluated** (cloud)" | both validated runs load and evaluate SmolLM2-135M @ `93efa2f0…` | SmolLM2-135M **has been** loaded and evaluated on the cloud substrate; TinyLlama-1.1B has not |
| 3 | `README.md` L100 | "adapter ready; **no cloud run has executed** — M3, Tier 1 and Tier 2 are NOT RUN" | registry, two `validated` runs | the adapter has executed end to end (generate → submit → poll → fetch → checksum-validate → registry); M3 *as specified*, Tier-1 confirmatory cells and Tier-2 are NOT RUN |
| 4 | `README.md` L112-117 | NOT RUN list includes "the Tier-1 pilot and the bounded LR-QAT/LoftQ reproduction" | `docs/results/tier1-cloud-run-2026-10-09.md`; `docs/results/m3-reproduction-2026-10-09.md` | the Tier-1 comparator **pilot** has run; the **frozen M3 protocol** has not, and cannot yet (see #7) |
| 5 | `docs/coordination/status.md` M3 row | M3 absent from the milestone table's verdict column | same | add an M3 row: **NOT RUN as specified**; the implemented trainable arms produced pilot diagnostics only |
| 6 | `docs/coordination/status.md` M4/M5 rows | "The confirmatory H2/H4 cells are `CLOUD-COLAB` and remain **NOT RUN**" | unchanged — no proxy or allocator cell has run on a real model | **correct as written**; keep, but note the cloud *substrate* is now verified and Kaggle is the executed backend (A-0011/A-0012) |
| 7 | `reports/paper/paper.md` L80, L250 | "M3 is **NOT RUN**: it requires the cloud substrate and no …" | the cloud substrate exists and two runs are validated; the *reason* M3 is not run is different | M3 is **NOT RUN as specified**: its eight frozen arms (2-bit NF-codebook block-64 LoftQ; 4-bit g128 LR-QAT with rank 32 and a learned step size; `R2-RTN-4bit-g128`; `R2-fullQAT-4bit-g128`; the validation-split learning-rate search) are **not implemented** in the runner. The implemented arms are a different configuration and are labelled diagnostics |
| 8 | `reports/paper/paper.md` L45 | "The confirmatory Tier-1/Tier-2 cells (H1–H4 and the H5 quality/GPU halves) are **NOT RUN**" | unchanged | **correct as written**; the pilot slice sentence added earlier must say explicitly that it is the comparator slice, not a confirmatory cell |
| 9 | `docs/coordination/handoff-2026-10-09.md` §4.2 | the M3 table's "after training" column, described as "the M3 cell" | the corrected run (commit `aa0ae0a`) supersedes it: loftq 17.79 → **16.69**, lr_qat 17.72 → **17.07** | the earlier negative was a corpus-size defect; the corrected run improves on its initialisation. **Neither is an M3 verdict** — the frozen protocol is unimplemented (#7) |
| 10 | `docs/results/m3-reproduction-2026-10-09.md` | titled and written as "the M3 cell" | #7 | relabel as the **trainable-arm diagnostic**; add the corrected-run numbers and state that the M3 protocol remains NOT RUN |

## What is genuinely measured, and at what scope

| Claim | Scope | Class | Where |
|---|---|---|---|
| fp16 reference perplexity 14.0181 on SmolLM2-135M / WikiText-2 test | 1 seed, 21 758 tokens, CPU, 210 targeted tensors | 1 (bytes) / 2 (quality) | `docs/results/tier1-cloud-run-2026-10-09.md` |
| int8 PTQ 14.0412 (+0.17 %), int4 PTQ 18.6190 (+32.8 %) | same | 2 | same |
| rank-8 untrained truncation degenerate (rel. Frobenius 0.9427) | same | 2 | same |
| trainable arms improve on their initialisation (loftq 17.79→16.69, lr_qat 17.72→17.07) | 1 seed, 300 steps, T4, 2 841 train windows | 2 | `docs/results/m3-reproduction-2026-10-09.md` |
| measured step cost 2.17 s/step (batch 8 × seq 512, T4) | 200 steps | 1 | same |
| class-4-CPU int4 `MatMulNBits` executes our container, rel. error 0.0771 | 1 fixture | 4-CPU | `artifacts/sample-results/class4cpu/class4cpu.json` |
| proxy candidate beats all predeclared comparators 15/15 contrasts | Tier-0 fixture, 10 trained models, 5 seeds | 2 | `docs/results/proxy-validation-report.md` |

## Explicitly NOT measured (never to be implied)

Tier-1 confirmatory H2 (proxy ranking on a real model), H3 (method), H4 (allocator frontier); the
frozen M3 protocol; Tier-2 (TinyLlama-1.1B); class 4-GPU; class 5; any latency, throughput, speedup
or state-of-the-art claim.
