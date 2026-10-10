# Experiment index — every claim, its evidence, and its substrate

One row per claim the project makes. "Evidence" names the artifact that a reader can open;
"report" names the prose that interprets it. A claim without a row here is not a claim.

Legend — **substrate**: `LOCAL` = this workstation (CPU, no CUDA) · `CLOUD-CPU` = Kaggle CPU ·
`CLOUD-GPU` = Kaggle Tesla T4. **Class** is the measurement taxonomy of `AGENTS.md` §5.

## Cloud runs (collected, checksum-validated)

| Claim | Evidence (run id, registry state) | Substrate | Class | Report |
|---|---|---|---|---|
| fp16 reference on SmolLM2-135M / WikiText-2 test: perplexity 14.0181 | `tier1_smollm2_135m-cloud` `validated` | CLOUD-CPU | 2 | `docs/results/tier1-cloud-run-2026-10-09.md` |
| int8 PTQ costs +0.17 % perplexity for 1.88× fewer accounted bytes (14.0412) | same run | CLOUD-CPU | 2 | same |
| int4 PTQ costs +32.8 % for 3.56× fewer bytes (18.6190) | same run | CLOUD-CPU | 2 | same |
| untrained rank-8 truncation is degenerate (1.40e16; mean relative Frobenius 0.9427) | same run | CLOUD-CPU | 2 | same |
| trainable arms improve on their own initialisation: LoftQ 17.793 → 16.6936, LR-QAT 17.722 → 17.0694 | `repro_lr_qat_loftq_smollm2_135m-cloud` `validated` | CLOUD-GPU | 2 | `docs/results/m3-reproduction-2026-10-09.md` (diagnostics, not M3 cells) |
| measured training cost: 2.17 s per optimiser step at batch 8 × seq 512 | same run | CLOUD-GPU | 1 | same |
| the first diagnostic's negative result was a corpus-size defect, not a method finding (99 windows vs 500 steps; task loss 0.06) | same run, superseded | CLOUD-GPU | 2 | same |

## Local measurements

| Claim | Evidence | Substrate | Class | Report |
|---|---|---|---|---|
| an int4 container we serialized runs through ONNX Runtime's `MatMulNBits` CPU kernel; int8 through `MatMulInteger`; relative max error 0.0771 at (4, 64); **no speedup claimed** | `artifacts/sample-results/class4cpu/class4cpu.json` | LOCAL | 4-CPU | `docs/results/class4cpu-report.md` |
| the equal-memory gate accepts an equal pair, labels a class-1 comparison, and **rejects** byte-unequal and mixed-source pairs | `artifacts/sample-results/comparability/*.json` | LOCAL | 1/3 | `docs/results/allocation-integration-report.md`, and the CI gate |
| 18/18 allocator arms reconcile class-3 measured bytes with class-1 accounting inside 0.5 % | `artifacts/sample-results/allocation-frontier/frontier.json` | LOCAL | 1/3 | `docs/results/allocator-report.md` |
| CP-SAT equals the exhaustive oracle at three budgets; uniform/greedy baselines recorded | `artifacts/sample-results/allocation-frontier/manifests/*.json` | LOCAL | 1 | same |
| the declared proxy `gain_aware_composed` beats every predeclared comparator in 15/15 aggregate contrasts (Fisher-z CI excluding 0); achieved MDE 0.476–0.715 | `artifacts/sample-results/proxy-validation/proxy-validation.json` | LOCAL | 2 | `docs/results/proxy-validation-report.md` |
| the regularizer lowers the rounding residual but **worsens** dev NLL (A-0008, H3 unsupported locally) | `artifacts/sample-results/ablations/ablations.json` | LOCAL | 2 | `docs/results/regularizer-report.md` |
| the smoke experiment reproduces its loss-sequence digest | `artifacts/sample-results/smoke-manifest.json` | LOCAL | 2 | CI |
| the 2-bit NF codebook reproduces the pinned reference exactly (`max_abs_diff = 0.0`); the LoftQ schedule agrees with it on the quantized weight (exact at T=1) and on the residual norm/spectrum (≤4.2e-07) | `artifacts/sample-results/m3-oracle/loftq-nf2-block64.json` | LOCAL | 2 | A-0013, A-0014, A-0015; `docs/decisions/design-m3-arms.md` |
| the T1-a merge identity holds exactly (integer path) and to 5.03e-07 (fp path, frozen 1e-5); the T2-a dominance gate is 21/21 (frozen 95 %) | `tests/unit/test_m3_exactness.py` (asserted, with the frozen tolerances quoted) | LOCAL | 2 | `docs/results/m3-reproduction-2026-10-09.md` |

## Not measured — no row exists because no artifact does

Tier-1 confirmatory H2 (proxy ranking on a real model) · H3 (the method) · H4 (the allocator
frontier on a real model) · the M3 verdict (the Tier-1 pilot is queued; the Tier-2 primary cell is
not started) · Tier-2 (TinyLlama-1.1B) · class 4-GPU · class 5 · any latency, throughput or speedup
figure.

## How to reproduce a row

```bash
uv sync --all-extras
uv run pytest -q                                   # the local rows and the gates
uv run spectraquant cloud notebook --plan <plan> --platform kaggle
uv run spectraquant cloud submit  --spec notebooks/generated/<run>.ipynb.spec.json
uv run spectraquant cloud fetch   --run-id <run> --dest <dir>
uv run spectraquant cloud collect --run-id <run> --source <dir>
uv run spectraquant cloud registry --run-id <run>  # the append-only attempt history
```

The registry (`artifacts/runs/registry.jsonl`, git-ignored runtime state) preserves every attempt,
including the failures and the operator-stopped run; a run whose artifacts were never collected is
recorded as such and contributes no row above.
