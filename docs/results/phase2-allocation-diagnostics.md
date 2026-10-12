# Phase 2: allocation size and calibration-subset instability

Offline analysis of the checksum-verified Phase2 report, scientific commit
`fe39413153c41dfc8e23ab2d88bfa9c4f217f72e`, metrics SHA256
`1cb35a549391fb47c0c7f679267cecb287e6e29a97d2200d3944c304d5d34456`.
No model was run locally and no new test evaluation occurred.

Each subset contains4x256 input tokens, or1,020 predicted tokens. There are210
compressible projection matrices. Count a change when `(bits,rank)!=(4,0)`.

| Family | Calibration subset | Changed matrices | Share of210 |
|---|---:|---:|---:|
| bits only | 0 | 77 | 36.7% |
| bits only | 1 | 85 | 40.5% |
| bits only | 2 | 78 | 37.1% |
| residual inclusive | 0 | 136 | 64.8% |
| residual inclusive | 1 | 135 | 64.3% |
| residual inclusive | 2 | 165 | 78.6% |

The solver is applying dozens or hundreds of interventions simultaneously,
although each scored candidate changed only one matrix in the INT4 background.
This does not prove interactions caused the degradation. A-0024 measures the
joint loss on the identical calibration blocks to test that explanation.

Compare complete `(bits,rank)` assignments between pairs of calibration subsets:

| Family | Subsets | Different assignments /210 | Share |
|---|---|---:|---:|
| bits only | 0 vs1 | 100 | 47.6% |
| bits only | 0 vs2 | 91 | 43.3% |
| bits only | 1 vs2 | 99 | 47.1% |
| residual inclusive | 0 vs1 | 156 | 74.3% |
| residual inclusive | 0 vs2 | 163 | 77.6% |
| residual inclusive | 1 vs2 | 176 | 83.8% |

These are descriptive counts from three subsets of one checkpoint, not
independent training runs or a statistical generalization claim. Large
assignment changes suggest selection sensitivity to the small calibration
sample. The follow-up caps of8/16/32 matrices test a bounded intervention
without tuning on the retired WikiText test. A larger calibration experiment
would be a separately preregistered future study, not an automatic retry.

Reproduce the counts from each arm's `allocation`: count tuples different from
`(4,0)`, then count pairwise tuple inequalities over the same210 layer names.
Uniform4's actual stored checkpoint is116,522,726B, and its targeted payload is
59,719,680B. Storage and simulated quality remain distinct measurement classes.
