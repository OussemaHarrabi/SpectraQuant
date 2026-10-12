# Frozen Phase4 Kaggle notebooks

These are unexecuted, generated input notebooks for the A-0025 transfer study.
Scientific code lives in `src/spectraquant/cloud/phase4.py`; the notebooks only
install the environment, check out the frozen source, invoke the runner and
export manifests, metrics, logs and packed checkpoints.

- `phase4-smol360-kaggle.ipynb`: SmolLM2-360M.
- `phase4-pythia410-kaggle.ipynb`: Pythia-410M.
- Scientific source: `71a1ec11cea6be90a6ab49356e81d980f4667784`.
- Protocol: `docs/decisions/ADR-0025-phase4-clipping-transfer.md`.
- Common frozen configuration: `configs/phase4/transfer.json`.

The archived specs contain the original submitted run IDs. Do not submit those
IDs again. Opening or inspecting a notebook does not authorize additional GPU
usage. Re-execution requires a new ID and a separately available compute budget.

The inputs are exact copies of the notebooks produced by
`scripts/experiments/phase4_cloud.py`. They are not hand-edited analysis notebooks
and do not contain measured output. After collection, numerical evidence and
the extracted collection attestations live in
`artifacts/sample-results/phase4-transfer/`; original executed notebooks and
large checkpoints remain in the private collected bundles outside Git.

All seven compressed arms freeze before holdout access. The primary four-cell
endpoint is axis0 MSE clipping versus axis0 RTN; grouping and activation-weighted
controls are predeclared secondary comparisons. No low-bit GPU acceleration
is measured by these notebooks.
