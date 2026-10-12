# SpectraQuant: fixed clipping transfer across two models and text domains

The fixed primary comparison is MSE clipping vs RTN with group32/axis0. All arms were stored and frozen before heldout loading.

| Model | Corpus | RTN PPL | Clipped PPL | Reduction | Documents | Descriptive delta-NLL CI |
|---|---|---:|---:|---:|---:|---|
| smol360 | c4_reserved | 21.8140 | 22.3901 | -2.64% | 105 | [+0.01888, +0.03219] |
| smol360 | pg19 | 16.6734 | 16.8819 | -1.25% | 26 | [+0.00641, +0.01887] |
| pythia410 | c4_reserved | 32.3251 | 31.3777 | +2.93% | 108 | [-0.03929, -0.02039] |
| pythia410 | pg19 | 27.9244 | 27.0500 | +3.13% | 26 | [-0.05944, -0.00481] |

Positive primary cells: 2/4. Geometric mean reduction: 0.575%. Frozen transfer gate passed: **False**.

Perplexities are tokenizer-dependent; do not compare or average raw values between models. The summary combines within-cell log ratios only.

## Predeclared secondary comparisons

Axis1 MSE/activation-weighted clipping compare with their own axis1 RTN baseline. Calibration subsets are not training seeds. These are conventional criteria, not full AWQ or a novel SpectraQuant algorithm.

| Model | Corpus | Arm | PPL | Reduction vs same-axis RTN | Payload B | Full checkpoint B |
|---|---|---|---:|---:|---:|---:|
| smol360 | c4_reserved | act_axis1_seed0 | 20.9878 | +7.39% | 176,947,200 | 271,561,031 |
| smol360 | pg19 | act_axis1_seed0 | 15.6996 | +5.89% | 176,947,200 | 271,561,031 |
| smol360 | c4_reserved | act_axis1_seed1 | 20.9165 | +7.70% | 176,947,200 | 271,561,031 |
| smol360 | pg19 | act_axis1_seed1 | 15.6505 | +6.18% | 176,947,200 | 271,561,031 |
| smol360 | c4_reserved | act_axis1_seed2 | 20.9421 | +7.59% | 176,947,200 | 271,561,031 |
| smol360 | pg19 | act_axis1_seed2 | 15.6735 | +6.05% | 176,947,200 | 271,561,031 |
| smol360 | c4_reserved | mse_axis0 | 22.3901 | -2.64% | 176,947,200 | 271,561,031 |
| smol360 | pg19 | mse_axis0 | 16.8819 | -1.25% | 176,947,200 | 271,561,031 |
| smol360 | c4_reserved | mse_axis1 | 23.0776 | -1.83% | 176,947,200 | 271,561,031 |
| smol360 | pg19 | mse_axis1 | 16.9607 | -1.67% | 176,947,200 | 271,561,031 |
| smol360 | c4_reserved | rtn_axis0 | 21.8140 | +0.00% | 176,947,200 | 271,561,031 |
| smol360 | pg19 | rtn_axis0 | 16.6734 | +0.00% | 176,947,200 | 271,561,031 |
| smol360 | c4_reserved | rtn_axis1 | 22.6621 | +0.00% | 176,947,200 | 271,561,031 |
| smol360 | pg19 | rtn_axis1 | 16.6823 | +0.00% | 176,947,200 | 271,561,031 |
| pythia410 | c4_reserved | act_axis1_seed0 | 28.1465 | +6.01% | 198,844,416 | 302,615,137 |
| pythia410 | pg19 | act_axis1_seed0 | 20.6253 | -2.43% | 198,844,416 | 302,615,137 |
| pythia410 | c4_reserved | act_axis1_seed1 | 28.2251 | +5.75% | 198,844,416 | 302,615,137 |
| pythia410 | pg19 | act_axis1_seed1 | 20.3540 | -1.09% | 198,844,416 | 302,615,137 |
| pythia410 | c4_reserved | act_axis1_seed2 | 28.1210 | +6.09% | 198,844,416 | 302,615,137 |
| pythia410 | pg19 | act_axis1_seed2 | 20.6509 | -2.56% | 198,844,416 | 302,615,137 |
| pythia410 | c4_reserved | mse_axis0 | 31.3777 | +2.93% | 198,844,416 | 302,615,137 |
| pythia410 | pg19 | mse_axis0 | 27.0500 | +3.13% | 198,844,416 | 302,615,137 |
| pythia410 | c4_reserved | mse_axis1 | 29.2329 | +2.38% | 198,844,416 | 302,615,137 |
| pythia410 | pg19 | mse_axis1 | 21.5564 | -7.06% | 198,844,416 | 302,615,137 |
| pythia410 | c4_reserved | rtn_axis0 | 32.3251 | +0.00% | 198,844,416 | 302,615,137 |
| pythia410 | pg19 | rtn_axis0 | 27.9244 | +0.00% | 198,844,416 | 302,615,137 |
| pythia410 | c4_reserved | rtn_axis1 | 29.9459 | +0.00% | 198,844,416 | 302,615,137 |
| pythia410 | pg19 | rtn_axis1 | 20.1353 | +0.00% | 198,844,416 | 302,615,137 |

## Scope and reproducibility

Class2: dense FP16 execution of reconstructed packed checkpoints. Class3: actual stored target, layout and unchanged-weight files, tokenizer/runtime excluded. Same-axis comparisons have exact target and full-checkpoint byte equality. No low-bit GPU kernel or acceleration claim.

51,200 predicted tokens per corpus/model with document boundaries. Paired document bootstrap is descriptive corpus uncertainty; no model-seed or publication significance claim. Pretraining exposure is unknown. Original Tier2, original allocation efficacy and complete LR-QAT reproduction remain unproved.

The narrative progresses from a failed allocator, through measured nonadditivity/calibration-transfer failure, to a fixed simpler criterion and this external transfer test. Report all negative cells. Do not select or retune an arm on these exposed holdouts.

- smol360: scientific `71a1ec11cea6be90a6ab49356e81d980f4667784`, clean=True; runner 1059.9s, approximate visible-device time 0.589h, excluding installation/export and platform overhead; not exact quota billing.
- pythia410: scientific `71a1ec11cea6be90a6ab49356e81d980f4667784`, clean=True; runner 1515.5s, approximate visible-device time 0.842h, excluding installation/export and platform overhead; not exact quota billing.
