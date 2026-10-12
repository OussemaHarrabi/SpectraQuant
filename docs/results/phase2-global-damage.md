# SpectraQuant Phase 2: deployment-matched global-damage allocation

Source commit: `fe39413153c41dfc8e23ab2d88bfa9c4f217f72e`.

Validation selected **uniform4** before the test split was read.

Quality is float16 execution of reconstructed packed weights (class 2). Stored bytes are measured files (class 3).
Held-out evaluation: EleutherAI/wikitext_document_level / test; 49,951 predicted tokens.

| Arm | Test perplexity ↓ | Reduction vs INT4 | Target payload B | Full checkpoint B | Validation NLL |
|---|---:|---:|---:|---:|---:|
| uniform4 | 22.6213 | +0.00% | 59,719,680 | 116,522,726 | 2.896924 |
| bits_seed0 | 24.5318 | -8.45% | 59,719,680 | 116,522,743 | 2.989904 |
| bits_seed1 | 25.3152 | -11.91% | 59,719,680 | 116,522,750 | 3.029165 |
| bits_seed2 | 23.8709 | -5.52% | 59,719,680 | 116,522,748 | 2.954103 |
| residual_seed0 | 24.1160 | -6.61% | 59,719,200 | 116,533,379 | 2.971035 |
| residual_seed1 | 25.9582 | -14.75% | 59,719,680 | 116,533,331 | 3.042316 |
| residual_seed2 | 24.4054 | -7.89% | 59,719,680 | 116,536,525 | 2.979558 |

## Interpretation

one checkpoint; calibration-subset variation; class-2 quality and class-3 stored bytes.

Verified 21 checkpoint files against the collected report's checksums.

No proposed family met the prespecified validation improvement threshold. Retain the baseline; do not claim a method improvement.

## Limits

- One pretrained checkpoint and one previously used test benchmark.
- Three calibration subsets are not independent training replicates.
- Paired contiguous-block bootstrap is descriptive; adjacent blocks depend.
- No low-bit inference kernel, acceleration, downstream accuracy or generalization claim.
- Equal target-payload ceiling; full checkpoint bytes are separately measured.
- The marginal-NLL knapsack objective is approximate; interactions and dataset shift can break it.

## Resource record

Scientific runner wall time: 1110.4 s. Visible-device time upper bound through runner finish: 0.617 h. The outer cloud interval includes installation through runner finish but excludes final export and platform overhead; neither interval is exact Kaggle quota billing.

## What this proves for a research CV

A reproducible PyTorch/Hugging Face compression experiment with constrained OR-Tools allocation, actual serialization/reload, isolated calibration/validation/test logic, cloud execution, uncertainty reporting and honest negative-result analysis. Quantitative gain wording must use the selected-family table and this scope.
