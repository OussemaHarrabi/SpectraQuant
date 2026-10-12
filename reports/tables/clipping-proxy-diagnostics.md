# Clipping objectives and whole-model quality

Descriptive analysis of saved pre-holdout layer scores and verified model metrics. All reductions compare with each arm's own-axis RTN baseline.

Weight SSE sums layer MSE times the number of elements. The diagonal proxy sums each layer's per-output mean times its output dimension, using calibration subset0's input second moments. It omits cross-feature covariance and is not measured output MSE or a causal explanation. Calibration subsets are not training seeds.

| Model | Arm | Weight SSE reduction | Diagonal proxy reduction | Clipped groups | C4 PPL reduction | PG19 PPL reduction |
|---|---|---:|---:|---:|---:|---:|
| smol360 | act_axis1_seed0 | +2.75% | +47.83% | 71.43% | +7.39% | +5.89% |
| smol360 | act_axis1_seed1 | +2.63% | +47.48% | 71.41% | +7.70% | +6.18% |
| smol360 | act_axis1_seed2 | +2.74% | +47.54% | 71.42% | +7.59% | +6.05% |
| smol360 | mse_axis0 | +9.98% | +6.37% | 71.02% | -2.64% | -1.25% |
| smol360 | mse_axis1 | +10.19% | -8.76% | 70.88% | -1.83% | -1.67% |
| smol360 | rtn_axis0 | +0.00% | +0.00% | 0.00% | +0.00% | +0.00% |
| smol360 | rtn_axis1 | +0.00% | +0.00% | 0.00% | +0.00% | +0.00% |
| pythia410 | act_axis1_seed0 | +4.63% | +24.68% | 71.50% | +6.01% | -2.43% |
| pythia410 | act_axis1_seed1 | +4.51% | +24.60% | 71.52% | +5.75% | -1.09% |
| pythia410 | act_axis1_seed2 | +4.60% | +24.61% | 71.49% | +6.09% | -2.56% |
| pythia410 | mse_axis0 | +10.45% | +10.99% | 70.80% | +2.93% | +3.13% |
| pythia410 | mse_axis1 | +10.69% | -2.92% | 70.74% | +2.38% | -7.06% |
| pythia410 | rtn_axis0 | +0.00% | +0.00% | 0.00% | +0.00% | +0.00% |
| pythia410 | rtn_axis1 | +0.00% | +0.00% | 0.00% | +0.00% | +0.00% |
