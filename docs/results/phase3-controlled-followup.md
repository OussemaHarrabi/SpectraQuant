# SpectraQuant Phase 3: joint-loss diagnosis and controlled follow-up

Source commit: `71b2e2b0f94bd7790a9389a0a0963839669494ab`.

Validation selected **mseclip** before the test split was read.

Quality is float16 execution of reconstructed packed weights (class 2). Stored bytes are measured files (class 3).
Held-out evaluation: allenai/c4 / validation; 49,951 predicted tokens.

| Arm | Test perplexity ↓ | Reduction vs INT4 | Target payload B | Full checkpoint B | Validation NLL |
|---|---:|---:|---:|---:|---:|
| uniform4 | 28.0056 | +0.00% | 59,719,680 | 116,522,698 | 2.896924 |
| mseclip | 27.4656 | +1.93% | 59,719,680 | 116,522,698 | 2.876979 |

## Interpretation

one pretrained checkpoint; one deterministic clipping comparison; class-2 quality and class-3 stored bytes.

Verified 6 checkpoint files against the collected report's checksums.

The validation-selected conventional clipping control changes held-out perplexity by +1.93% reduction over INT4 at the measured bytes above. This is one deterministic comparison on one pretrained checkpoint, not a replicated allocation-method gain.

## Limits

- One pretrained checkpoint; new C4 validation holdout, with pretraining exposure unknown.
- Three calibration subsets are not independent training replicates.
- Paired contiguous-block bootstrap is descriptive; adjacent blocks depend.
- No low-bit inference kernel, acceleration, downstream accuracy or generalization claim.
- Equal target-payload ceiling; full checkpoint bytes are separately measured.
- The marginal-NLL knapsack objective is approximate; interactions and dataset shift can break it.

## Resource record

Scientific runner wall time: 218.6 s. Visible-device time upper bound through runner finish: 0.121 h. The outer cloud interval includes installation through runner finish but excludes final export and platform overhead; neither interval is exact Kaggle quota billing.

## What this proves for a research CV

A reproducible PyTorch/Hugging Face compression experiment with constrained OR-Tools allocation, actual serialization/reload, isolated calibration/validation/test logic, cloud execution, uncertainty reporting and honest negative-result analysis. Quantitative gain wording must use the selected-family table and this scope.


## Joint interaction diagnosis

Negative delta NLL means better. Nonadditivity = actual joint minus summed marginal delta on the same calibration blocks.

| Prior arm | Sum of marginal deltas | Actual joint delta | Nonadditivity gap | Disjoint training-block delta |
|---|---:|---:|---:|---:|
| bits_seed0 | -0.195256 | -0.052484 | +0.142772 | +0.059400 |
| bits_seed1 | -0.256668 | -0.053351 | +0.203317 | +0.077037 |
| bits_seed2 | -0.205278 | -0.116244 | +0.089034 | +0.047334 |
| residual_seed0 | -0.253794 | -0.098923 | +0.154871 | +0.049556 |
| residual_seed1 | -0.313331 | -0.058629 | +0.254703 | +0.098736 |
| residual_seed2 | -0.282477 | -0.131452 | +0.151025 | +0.057969 |

## Development selection

All family/cap choices were compared on WikiText validation before C4 validation was read. Selection is exploratory and may overfit development; there is no tuning on the new heldout.

| Family | Mean development NLL |
|---|---:|
| bits_k16 | 2.893219 |
| bits_k32 | 2.925715 |
| bits_k8 | 2.882619 |
| mseclip | 2.876979 |
| residual_k16 | 2.891725 |
| residual_k32 | 2.910234 |
| residual_k8 | 2.878364 |
| uniform4 | 2.896924 |

Clipping is a conventional weights-only control, not a novel SpectraQuant method. Its one deterministic checkpoint is not three independent replicates.

The fresh holdout is not the prior WikiText test. Full Tier 2 reproduction, cross-model transfer and trainable-method gains remain unproved.
