# Collected Phase3 numerical evidence

These are small numerical artifacts from Kaggle run
`oussemaharrabi/sq-phase3-controls-20261011`, version 1, scientific commit
`71b2e2b0f94bd7790a9389a0a0963839669494ab` (clean checkout).

- `metrics.json`: exact source report, including joint diagnostics, all
  development allocations and final heldout comparisons.
- `selection.json`: exact source development selection, frozen before the
  scientific runner loaded the new C4 validation holdout.
- `evidence-index.json`: extracted collection attestation with the original
  manifest/report hashes. This is not an original cloud manifest.

Eleven expected artifacts and six checkpoint files passed local collection
checks. Model weights, raw datasets, dependency/context dumps and logs remain
in the collected bundle outside Git. This is independently verified artifact
collection, not an independent rerun of the research experiment.

Clipping reduced heldout perplexity by 1.93% on one pretrained checkpoint at
identical bytes. It is a conventional control, not a novel allocation-method
gain. Quality uses dense reconstructed FP16 weights (class 2), storage uses
actual checkpoint files (class 3). No GPU kernel acceleration is claimed.
