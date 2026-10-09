# Proxy validation report (Milestone 4 gate — local-fixture half)

Owner: proxy slice. Machine-readable artifact: `artifacts/sample-results/proxy-validation/proxy-validation.json` (regenerate with `uv run python scripts/experiments/proxy_validation_sweep.py`). This report is generated from that document, so the numbers below and the JSON cannot drift apart.

Substrate `LOCAL-FIXTURE` (CPU workstation); every number is measurement class **2** (`AGENTS.md` section 5). Seeds `[0, 1, 2, 3, 4]` x cells `[[8, 4], [16, 4], [32, 4], [8, 8], [16, 8], [32, 8]]` on fixtures `['layernorm', 'no_layernorm']`, i.e. 10 trained models and 12 reported cells; sweep wall time 19.0 s; git commit `1207b6942f65b585857385401292167a3e77facc` (dirty=True).

## 0. Headline

* **`combined`** — is the local-fixture evidence consistent with it beating every predeclared comparator? **No**: it is significantly worse than a comparator at layernorm/weight_frobenius, layernorm/weight_magnitude, all_fixtures/weight_frobenius, all_fixtures/weight_magnitude; its advantage is below the predeclared MDE (CI includes 0) at all_fixtures/activation_magnitude, all_fixtures/hessian_diag, all_fixtures/per_layer_output_error; and inconclusive at layernorm/activation_magnitude, layernorm/hessian_diag, layernorm/per_layer_output_error, no_layernorm/weight_frobenius, no_layernorm/weight_magnitude, no_layernorm/activation_magnitude, no_layernorm/hessian_diag, no_layernorm/per_layer_output_error.
* **`gain_aware_composed`** — is the local-fixture evidence consistent with it beating every predeclared comparator? **Yes**: every one of the 15 aggregate contrasts (2 fixtures + pooled, 5 comparators each) has a positive Fisher-z advantage whose random-effects CI excludes 0.
* **Achieved versus predeclared MDE.** The predeclared band at S=5 seeds is MDE(z) 0.33 (optimistic, n_eff=32) to 0.79 (conservative, n_eff=8); the achieved MDE at the achieved mean n_eff is +0.636 (layernorm), +0.715 (no_layernorm) and +0.476 (pooled), i.e. inside the predeclared band; individual cells span +0.543 to +1.318. The Tier-0 fixture has 13 modules per model, coarser than the predeclared optimistic end (32) and finer than the conservative end (8), which is why the achieved MDE lands between them; it is a *different* fixture from the pinned Tier-1 `L_b=8` cell, so this is not the Tier-1 MDE.
* **Cloud Tier-1 cell: NOT RUN.** The H2 and H4 confirmatory cells are `CLOUD-COLAB` and need the Colab run; no Tier-1 number is implied anywhere in this document (section 5).

## 1. Setup

* Fixture: 3-block pre-norm transformer, `d_model=48`, 2 heads, `d_ff=96`, `vocab=24`, `seq_len=24`; trained 120 steps on the deterministic synthetic next-token task (one training run per fixture x seed).
* 13 compressible linear modules per model (`blocks.0.qkv, blocks.0.proj, blocks.0.fc1, blocks.0.fc2, ... , head`).
* Compression per cell: rank-then-quantize, per-group symmetric, `group_size=32`; damage = mean squared change of the final hidden state when that module alone is compressed (measured, class 2).
* Two fixtures: LayerNorm present (`layernorm`) and the adversarial review's control (`no_layernorm`).
* Statistical unit: the **model** (one `rho` per trained model per cell); `n_eff` per model from a within-model module bootstrap (200 replicates, distinctness guard >= ceil(n/2)); combined with a DerSimonian-Laird Fisher-z random-effects model; contrast CIs additionally bootstrapped over models (2000 replicates, seed `20261009`).
* Candidate proxies: `combined`, `gain_aware_composed` (headline = `combined`); pre-declared comparators `weight_frobenius`, `weight_magnitude`, `activation_magnitude`, `hessian_diag` (primary = `weight_frobenius`); plus the naive baseline `per_layer_output_error`. The artifact reports the paired contrast for **both** candidates, because the design note calls `combined` the candidate interface while the repository status record calls the gain-aware composed variant the candidate; neither naming is resolved by fiat.
* Pre-declared MDE (`section 7.5`): MDE(z) ~ 0.33 optimistically (n_eff=32) and ~ 0.79 conservatively (n_eff=8) at S=5 seeds. Achieved MDE is reported per cell and per aggregate below.

## 2. Per-cell results

Mean over seeds of the model-level Spearman `rho` against single-layer damage:

| fixture | rank | bits | `weight_frobenius` | `weight_magnitude` | `activation_magnitude` | `per_layer_output_error` | `in_situ_output_error` | `gain_aware_composed` | `hessian_diag` | `quant_residual_stats` | `spectral_summary` | `combined` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| layernorm | 8 | 4 | +0.370 | +0.360 | +0.129 | -0.025 | +0.456 | +0.849 | -0.044 | -0.040 | -0.040 | -0.011 |
| layernorm | 16 | 4 | +0.426 | +0.362 | +0.141 | +0.222 | +0.455 | +0.876 | +0.209 | -0.047 | +0.209 | +0.223 |
| layernorm | 32 | 4 | +0.416 | +0.365 | +0.126 | +0.119 | +0.426 | +0.780 | +0.209 | -0.052 | +0.415 | +0.135 |
| layernorm | 8 | 8 | +0.375 | +0.358 | +0.129 | -0.026 | +0.463 | +0.849 | -0.041 | -0.048 | -0.041 | -0.011 |
| layernorm | 16 | 8 | +0.437 | +0.369 | +0.154 | +0.231 | +0.476 | +0.882 | +0.220 | -0.046 | +0.221 | +0.247 |
| layernorm | 32 | 8 | +0.433 | +0.378 | +0.119 | +0.415 | +0.858 | +0.742 | +0.427 | -0.034 | +0.429 | +0.438 |
| no_layernorm | 8 | 4 | +0.407 | +0.436 | +0.385 | +0.237 | -0.140 | +0.786 | +0.269 | +0.246 | +0.269 | +0.269 |
| no_layernorm | 16 | 4 | +0.555 | +0.555 | +0.404 | +0.411 | +0.107 | +0.800 | +0.427 | +0.299 | +0.443 | +0.391 |
| no_layernorm | 32 | 4 | +0.702 | +0.708 | +0.399 | +0.519 | +0.240 | +0.848 | +0.611 | +0.338 | +0.781 | +0.500 |
| no_layernorm | 8 | 8 | +0.415 | +0.446 | +0.389 | +0.246 | -0.132 | +0.786 | +0.280 | +0.274 | +0.280 | +0.280 |
| no_layernorm | 16 | 8 | +0.570 | +0.569 | +0.398 | +0.479 | +0.184 | +0.805 | +0.445 | +0.309 | +0.445 | +0.451 |
| no_layernorm | 32 | 8 | +0.751 | +0.777 | +0.393 | +0.830 | +0.649 | +0.895 | +0.830 | +0.378 | +0.823 | +0.834 |

Achieved MDE per cell (paired-contrast reference `n_eff`, S seeds):

| fixture | rank | bits | n_eff | MDE(z) achieved | predeclared optimistic | predeclared conservative |
|---|---|---|---|---|---|---|
| layernorm | 8 | 4 | +9.10 | +0.718 | 0.33 | 0.79 |
| layernorm | 16 | 4 | +13.22 | +0.554 | 0.33 | 0.79 |
| layernorm | 32 | 4 | +10.48 | +0.648 | 0.33 | 0.79 |
| layernorm | 8 | 8 | +8.33 | +0.767 | 0.33 | 0.79 |
| layernorm | 16 | 8 | +13.64 | +0.543 | 0.33 | 0.79 |
| layernorm | 32 | 8 | +9.74 | +0.682 | 0.33 | 0.79 |
| no_layernorm | 8 | 4 | +9.70 | +0.685 | 0.33 | 0.79 |
| no_layernorm | 16 | 4 | +10.62 | +0.642 | 0.33 | 0.79 |
| no_layernorm | 32 | 4 | +8.33 | +0.767 | 0.33 | 0.79 |
| no_layernorm | 8 | 8 | +10.65 | +0.641 | 0.33 | 0.79 |
| no_layernorm | 16 | 8 | +10.70 | +0.639 | 0.33 | 0.79 |
| no_layernorm | 32 | 8 | +4.81 | +1.318 | 0.33 | 0.79 |

Per-cell paired contrasts (compact form; full per-cell contrast table with both CIs is in `per_cell[].contrasts` of the JSON):

| fixture | rank | bits | candidate | comparator | delta_z | CI | boot CI | p | verdict |
|---|---|---|---|---|---|---|---|---|---|
| layernorm | 8 | 4 | `combined` | `weight_frobenius` | -0.404 | [-0.872, +0.064] | [-0.435, -0.364] | 0.0907 | inconclusive |
| layernorm | 8 | 4 | `combined` | `weight_magnitude` | -0.388 | [-0.892, +0.116] | [-0.413, -0.366] | 0.131 | inconclusive |
| layernorm | 8 | 4 | `combined` | `activation_magnitude` | -0.141 | [-0.673, +0.391] | [-0.161, -0.125] | 0.603 | inconclusive |
| layernorm | 8 | 4 | `combined` | `hessian_diag` | +0.032 | [-0.481, +0.546] | [+0.014, +0.058] | 0.901 | inconclusive |
| layernorm | 8 | 4 | `combined` | `per_layer_output_error` | +0.011 | [-0.499, +0.521] | [-0.019, +0.041] | 0.966 | inconclusive |
| layernorm | 8 | 4 | `gain_aware_composed` | `weight_frobenius` | +0.889 | [-0.200, +1.977] | [+0.736, +1.078] | 0.11 | inconclusive |
| layernorm | 8 | 4 | `gain_aware_composed` | `weight_magnitude` | +0.902 | [-0.203, +2.006] | [+0.745, +1.117] | 0.11 | inconclusive |
| layernorm | 8 | 4 | `gain_aware_composed` | `activation_magnitude` | +1.142 | [+0.027, +2.258] | [+0.992, +1.363] | 0.0447 | supported |
| layernorm | 8 | 4 | `gain_aware_composed` | `hessian_diag` | +1.322 | [+0.211, +2.433] | [+1.176, +1.514] | 0.0196 | supported |
| layernorm | 8 | 4 | `gain_aware_composed` | `per_layer_output_error` | +1.294 | [+0.181, +2.408] | [+1.154, +1.505] | 0.0227 | supported |
| layernorm | 16 | 4 | `combined` | `weight_frobenius` | -0.229 | [-0.652, +0.193] | [-0.247, -0.217] | 0.288 | inconclusive |
| layernorm | 16 | 4 | `combined` | `weight_magnitude` | -0.151 | [-0.586, +0.284] | [-0.166, -0.135] | 0.496 | inconclusive |
| layernorm | 16 | 4 | `combined` | `activation_magnitude` | +0.086 | [-0.356, +0.527] | [+0.074, +0.097] | 0.704 | inconclusive |
| layernorm | 16 | 4 | `combined` | `hessian_diag` | +0.013 | [-0.376, +0.403] | [-0.009, +0.032] | 0.947 | inconclusive |
| layernorm | 16 | 4 | `combined` | `per_layer_output_error` | +0.002 | [-0.379, +0.382] | [-0.014, +0.014] | 0.993 | inconclusive |
| layernorm | 16 | 4 | `gain_aware_composed` | `weight_frobenius` | +0.846 | [-0.255, +1.947] | [+0.691, +1.143] | 0.132 | inconclusive |
| layernorm | 16 | 4 | `gain_aware_composed` | `weight_magnitude` | +0.928 | [-0.179, +2.034] | [+0.784, +1.213] | 0.1 | inconclusive |
| layernorm | 16 | 4 | `gain_aware_composed` | `activation_magnitude` | +1.164 | [+0.055, +2.273] | [+1.013, +1.450] | 0.0396 | supported |
| layernorm | 16 | 4 | `gain_aware_composed` | `hessian_diag` | +1.089 | [+0.003, +2.175] | [+0.934, +1.385] | 0.0494 | supported |
| layernorm | 16 | 4 | `gain_aware_composed` | `per_layer_output_error` | +1.073 | [-0.008, +2.154] | [+0.924, +1.367] | 0.0517 | inconclusive |
| layernorm | 32 | 4 | `combined` | `weight_frobenius` | -0.310 | [-0.766, +0.146] | [-0.364, -0.252] | 0.183 | inconclusive |
| layernorm | 32 | 4 | `combined` | `weight_magnitude` | -0.248 | [-0.713, +0.216] | [-0.294, -0.199] | 0.295 | inconclusive |
| layernorm | 32 | 4 | `combined` | `activation_magnitude` | +0.005 | [-0.478, +0.487] | [-0.030, +0.049] | 0.985 | inconclusive |
| layernorm | 32 | 4 | `combined` | `hessian_diag` | -0.083 | [-0.521, +0.356] | [-0.120, -0.036] | 0.712 | inconclusive |
| layernorm | 32 | 4 | `combined` | `per_layer_output_error` | +0.017 | [-0.433, +0.467] | [-0.001, +0.035] | 0.94 | inconclusive |
| layernorm | 32 | 4 | `gain_aware_composed` | `weight_frobenius` | +0.578 | [-0.124, +1.280] | [+0.499, +0.746] | 0.107 | inconclusive |
| layernorm | 32 | 4 | `gain_aware_composed` | `weight_magnitude` | +0.645 | [-0.064, +1.355] | [+0.568, +0.795] | 0.0747 | inconclusive |
| layernorm | 32 | 4 | `gain_aware_composed` | `activation_magnitude` | +0.914 | [+0.200, +1.627] | [+0.823, +1.042] | 0.0121 | supported |
| layernorm | 32 | 4 | `gain_aware_composed` | `hessian_diag` | +0.819 | [+0.131, +1.508] | [+0.740, +0.970] | 0.0197 | supported |
| layernorm | 32 | 4 | `gain_aware_composed` | `per_layer_output_error` | +0.915 | [+0.219, +1.610] | [+0.792, +1.084] | 0.00998 | supported |
| layernorm | 8 | 8 | `combined` | `weight_frobenius` | -0.404 | [-0.895, +0.087] | [-0.437, -0.361] | 0.107 | inconclusive |
| layernorm | 8 | 8 | `combined` | `weight_magnitude` | -0.387 | [-0.901, +0.127] | [-0.416, -0.359] | 0.14 | inconclusive |
| layernorm | 8 | 8 | `combined` | `activation_magnitude` | -0.141 | [-0.668, +0.385] | [-0.161, -0.126] | 0.599 | inconclusive |
| layernorm | 8 | 8 | `combined` | `hessian_diag` | +0.031 | [-0.557, +0.618] | [+0.010, +0.060] | 0.919 | inconclusive |
| layernorm | 8 | 8 | `combined` | `per_layer_output_error` | +0.014 | [-0.521, +0.549] | [-0.021, +0.047] | 0.959 | inconclusive |
| layernorm | 8 | 8 | `gain_aware_composed` | `weight_frobenius` | +0.867 | [-0.167, +1.901] | [+0.728, +1.044] | 0.1 | inconclusive |
| layernorm | 8 | 8 | `gain_aware_composed` | `weight_magnitude` | +0.881 | [-0.164, +1.926] | [+0.743, +1.089] | 0.0983 | inconclusive |
| layernorm | 8 | 8 | `gain_aware_composed` | `activation_magnitude` | +1.129 | [+0.076, +2.182] | [+0.988, +1.335] | 0.0356 | supported |
| layernorm | 8 | 8 | `gain_aware_composed` | `hessian_diag` | +1.313 | [+0.209, +2.418] | [+1.176, +1.493] | 0.0198 | supported |
| layernorm | 8 | 8 | `gain_aware_composed` | `per_layer_output_error` | +1.283 | [+0.218, +2.348] | [+1.150, +1.481] | 0.0182 | supported |
| layernorm | 16 | 8 | `combined` | `weight_frobenius` | -0.216 | [-0.632, +0.200] | [-0.225, -0.209] | 0.309 | inconclusive |
| layernorm | 16 | 8 | `combined` | `weight_magnitude` | -0.135 | [-0.561, +0.290] | [-0.157, -0.113] | 0.533 | inconclusive |
| layernorm | 16 | 8 | `combined` | `activation_magnitude` | +0.097 | [-0.358, +0.553] | [+0.092, +0.103] | 0.676 | inconclusive |
| layernorm | 16 | 8 | `combined` | `hessian_diag` | +0.030 | [-0.356, +0.416] | [+0.017, +0.046] | 0.878 | inconclusive |
| layernorm | 16 | 8 | `combined` | `per_layer_output_error` | +0.017 | [-0.361, +0.395] | [+0.007, +0.027] | 0.931 | inconclusive |
| layernorm | 16 | 8 | `gain_aware_composed` | `weight_frobenius` | +0.831 | [+0.004, +1.658] | [+0.736, +1.122] | 0.049 | supported |
| layernorm | 16 | 8 | `gain_aware_composed` | `weight_magnitude` | +0.917 | [+0.094, +1.741] | [+0.833, +1.187] | 0.029 | supported |
| layernorm | 16 | 8 | `gain_aware_composed` | `activation_magnitude` | +1.154 | [+0.302, +2.006] | [+1.058, +1.425] | 0.00795 | supported |
| layernorm | 16 | 8 | `gain_aware_composed` | `hessian_diag` | +1.075 | [+0.271, +1.879] | [+0.983, +1.364] | 0.00874 | supported |
| layernorm | 16 | 8 | `gain_aware_composed` | `per_layer_output_error` | +1.065 | [+0.261, +1.869] | [+0.978, +1.344] | 0.00946 | supported |
| layernorm | 32 | 8 | `combined` | `weight_frobenius` | +0.006 | [-0.465, +0.477] | [-0.001, +0.014] | 0.979 | inconclusive |
| layernorm | 32 | 8 | `combined` | `weight_magnitude` | +0.072 | [-0.407, +0.552] | [+0.051, +0.086] | 0.768 | inconclusive |
| layernorm | 32 | 8 | `combined` | `activation_magnitude` | +0.348 | [-0.132, +0.829] | [+0.312, +0.390] | 0.155 | inconclusive |
| layernorm | 32 | 8 | `combined` | `hessian_diag` | +0.011 | [-0.468, +0.491] | [+0.002, +0.025] | 0.963 | inconclusive |
| layernorm | 32 | 8 | `combined` | `per_layer_output_error` | +0.030 | [-0.438, +0.499] | [+0.007, +0.048] | 0.899 | inconclusive |
| layernorm | 32 | 8 | `gain_aware_composed` | `weight_frobenius` | +0.503 | [-0.109, +1.116] | [+0.427, +0.596] | 0.107 | inconclusive |
| layernorm | 32 | 8 | `gain_aware_composed` | `weight_magnitude` | +0.568 | [-0.056, +1.192] | [+0.506, +0.654] | 0.0744 | inconclusive |
| layernorm | 32 | 8 | `gain_aware_composed` | `activation_magnitude` | +0.851 | [+0.221, +1.481] | [+0.759, +0.972] | 0.00811 | supported |
| layernorm | 32 | 8 | `gain_aware_composed` | `hessian_diag` | +0.504 | [-0.117, +1.125] | [+0.429, +0.596] | 0.112 | inconclusive |
| layernorm | 32 | 8 | `gain_aware_composed` | `per_layer_output_error` | +0.527 | [-0.088, +1.142] | [+0.454, +0.595] | 0.0929 | inconclusive |
| no_layernorm | 8 | 4 | `combined` | `weight_frobenius` | -0.159 | [-0.605, +0.287] | [-0.218, -0.108] | 0.485 | inconclusive |
| no_layernorm | 8 | 4 | `combined` | `weight_magnitude` | -0.193 | [-0.685, +0.300] | [-0.217, -0.158] | 0.443 | inconclusive |
| no_layernorm | 8 | 4 | `combined` | `activation_magnitude` | -0.130 | [-0.716, +0.455] | [-0.155, -0.095] | 0.662 | inconclusive |
| no_layernorm | 8 | 4 | `combined` | `hessian_diag` | +0.000 | [-0.479, +0.479] | [+0.000, +0.000] | 1 | inconclusive |
| no_layernorm | 8 | 4 | `combined` | `per_layer_output_error` | +0.033 | [-0.464, +0.530] | [+0.018, +0.049] | 0.897 | inconclusive |
| no_layernorm | 8 | 4 | `gain_aware_composed` | `weight_frobenius` | +0.673 | [+0.030, +1.316] | [+0.540, +0.768] | 0.0403 | supported |
| no_layernorm | 8 | 4 | `gain_aware_composed` | `weight_magnitude` | +0.639 | [-0.041, +1.319] | [+0.484, +0.737] | 0.0657 | inconclusive |
| no_layernorm | 8 | 4 | `gain_aware_composed` | `activation_magnitude` | +0.687 | [-0.118, +1.491] | [+0.539, +0.802] | 0.0942 | inconclusive |
| no_layernorm | 8 | 4 | `gain_aware_composed` | `hessian_diag` | +0.830 | [+0.159, +1.500] | [+0.692, +0.948] | 0.0153 | supported |
| no_layernorm | 8 | 4 | `gain_aware_composed` | `per_layer_output_error` | +0.864 | [+0.177, +1.552] | [+0.724, +0.968] | 0.0138 | supported |
| no_layernorm | 16 | 4 | `combined` | `weight_frobenius` | -0.194 | [-0.701, +0.312] | [-0.288, -0.133] | 0.452 | inconclusive |
| no_layernorm | 16 | 4 | `combined` | `weight_magnitude` | -0.215 | [-0.697, +0.266] | [-0.289, -0.163] | 0.381 | inconclusive |
| no_layernorm | 16 | 4 | `combined` | `activation_magnitude` | -0.016 | [-0.545, +0.513] | [-0.076, +0.054] | 0.954 | inconclusive |
| no_layernorm | 16 | 4 | `combined` | `hessian_diag` | -0.045 | [-0.482, +0.391] | [-0.065, -0.021] | 0.838 | inconclusive |
| no_layernorm | 16 | 4 | `combined` | `per_layer_output_error` | -0.025 | [-0.482, +0.432] | [-0.048, -0.003] | 0.915 | inconclusive |
| no_layernorm | 16 | 4 | `gain_aware_composed` | `weight_frobenius` | +0.478 | [-0.263, +1.218] | [+0.371, +0.650] | 0.206 | inconclusive |
| no_layernorm | 16 | 4 | `gain_aware_composed` | `weight_magnitude` | +0.455 | [-0.280, +1.190] | [+0.373, +0.642] | 0.225 | inconclusive |
| no_layernorm | 16 | 4 | `gain_aware_composed` | `activation_magnitude` | +0.653 | [-0.125, +1.432] | [+0.525, +0.834] | 0.1 | inconclusive |
| no_layernorm | 16 | 4 | `gain_aware_composed` | `hessian_diag` | +0.622 | [-0.091, +1.335] | [+0.521, +0.800] | 0.0872 | inconclusive |
| no_layernorm | 16 | 4 | `gain_aware_composed` | `per_layer_output_error` | +0.653 | [-0.076, +1.383] | [+0.563, +0.800] | 0.0793 | inconclusive |
| no_layernorm | 32 | 4 | `combined` | `weight_frobenius` | -0.348 | [-0.910, +0.214] | [-0.430, -0.279] | 0.225 | inconclusive |
| no_layernorm | 32 | 4 | `combined` | `weight_magnitude` | -0.315 | [-0.856, +0.225] | [-0.375, -0.280] | 0.253 | inconclusive |
| no_layernorm | 32 | 4 | `combined` | `activation_magnitude` | +0.134 | [-0.456, +0.724] | [+0.050, +0.220] | 0.657 | inconclusive |
| no_layernorm | 32 | 4 | `combined` | `hessian_diag` | -0.169 | [-0.683, +0.345] | [-0.223, -0.122] | 0.519 | inconclusive |
| no_layernorm | 32 | 4 | `combined` | `per_layer_output_error` | -0.022 | [-0.559, +0.515] | [-0.048, -0.006] | 0.935 | inconclusive |
| no_layernorm | 32 | 4 | `gain_aware_composed` | `weight_frobenius` | +0.361 | [-0.465, +1.187] | [+0.283, +0.551] | 0.392 | inconclusive |
| no_layernorm | 32 | 4 | `gain_aware_composed` | `weight_magnitude` | +0.357 | [-0.433, +1.146] | [+0.276, +0.522] | 0.376 | inconclusive |
| no_layernorm | 32 | 4 | `gain_aware_composed` | `activation_magnitude` | +0.746 | [-0.103, +1.595] | [+0.625, +1.059] | 0.0852 | inconclusive |
| no_layernorm | 32 | 4 | `gain_aware_composed` | `hessian_diag` | +0.524 | [-0.277, +1.326] | [+0.461, +0.678] | 0.2 | inconclusive |
| no_layernorm | 32 | 4 | `gain_aware_composed` | `per_layer_output_error` | +0.635 | [-0.174, +1.444] | [+0.563, +0.855] | 0.124 | inconclusive |
| no_layernorm | 8 | 8 | `combined` | `weight_frobenius` | -0.155 | [-0.594, +0.284] | [-0.222, -0.102] | 0.488 | inconclusive |
| no_layernorm | 8 | 8 | `combined` | `weight_magnitude` | -0.188 | [-0.690, +0.314] | [-0.220, -0.152] | 0.463 | inconclusive |
| no_layernorm | 8 | 8 | `combined` | `activation_magnitude` | -0.127 | [-0.705, +0.451] | [-0.152, -0.094] | 0.666 | inconclusive |
| no_layernorm | 8 | 8 | `combined` | `hessian_diag` | +0.000 | [-0.448, +0.448] | [+0.000, +0.000] | 1 | inconclusive |
| no_layernorm | 8 | 8 | `combined` | `per_layer_output_error` | +0.036 | [-0.457, +0.529] | [+0.018, +0.055] | 0.886 | inconclusive |
| no_layernorm | 8 | 8 | `gain_aware_composed` | `weight_frobenius` | +0.635 | [-0.148, +1.418] | [+0.522, +0.718] | 0.112 | inconclusive |
| no_layernorm | 8 | 8 | `gain_aware_composed` | `weight_magnitude` | +0.600 | [-0.215, +1.415] | [+0.486, +0.667] | 0.149 | inconclusive |
| no_layernorm | 8 | 8 | `gain_aware_composed` | `activation_magnitude` | +0.667 | [-0.217, +1.552] | [+0.585, +0.727] | 0.139 | inconclusive |
| no_layernorm | 8 | 8 | `gain_aware_composed` | `hessian_diag` | +0.784 | [-0.008, +1.576] | [+0.697, +0.855] | 0.0523 | inconclusive |
| no_layernorm | 8 | 8 | `gain_aware_composed` | `per_layer_output_error` | +0.822 | [+0.009, +1.635] | [+0.727, +0.888] | 0.0474 | supported |
| no_layernorm | 16 | 8 | `combined` | `weight_frobenius` | -0.146 | [-0.657, +0.365] | [-0.258, -0.068] | 0.574 | inconclusive |
| no_layernorm | 16 | 8 | `combined` | `weight_magnitude` | -0.170 | [-0.677, +0.337] | [-0.255, -0.099] | 0.512 | inconclusive |
| no_layernorm | 16 | 8 | `combined` | `activation_magnitude` | +0.066 | [-0.486, +0.617] | [+0.015, +0.117] | 0.815 | inconclusive |
| no_layernorm | 16 | 8 | `combined` | `hessian_diag` | +0.006 | [-0.436, +0.449] | [+0.000, +0.015] | 0.977 | inconclusive |
| no_layernorm | 16 | 8 | `combined` | `per_layer_output_error` | -0.040 | [-0.485, +0.405] | [-0.064, -0.010] | 0.861 | inconclusive |
| no_layernorm | 16 | 8 | `gain_aware_composed` | `weight_frobenius` | +0.523 | [-0.212, +1.257] | [+0.329, +0.730] | 0.163 | inconclusive |
| no_layernorm | 16 | 8 | `gain_aware_composed` | `weight_magnitude` | +0.510 | [-0.213, +1.234] | [+0.363, +0.725] | 0.167 | inconclusive |
| no_layernorm | 16 | 8 | `gain_aware_composed` | `activation_magnitude` | +0.721 | [-0.048, +1.490] | [+0.570, +0.869] | 0.0661 | inconclusive |
| no_layernorm | 16 | 8 | `gain_aware_composed` | `hessian_diag` | +0.680 | [-0.009, +1.370] | [+0.538, +0.843] | 0.0531 | inconclusive |
| no_layernorm | 16 | 8 | `gain_aware_composed` | `per_layer_output_error` | +0.633 | [-0.060, +1.326] | [+0.505, +0.781] | 0.0733 | inconclusive |
| no_layernorm | 32 | 8 | `combined` | `weight_frobenius` | +0.218 | [-0.612, +1.049] | [+0.167, +0.299] | 0.607 | inconclusive |
| no_layernorm | 32 | 8 | `combined` | `weight_magnitude` | +0.147 | [-0.724, +1.018] | [+0.117, +0.216] | 0.741 | inconclusive |
| no_layernorm | 32 | 8 | `combined` | `activation_magnitude` | +0.765 | [-0.061, +1.591] | [+0.604, +1.001] | 0.0694 | inconclusive |
| no_layernorm | 32 | 8 | `combined` | `hessian_diag` | +0.024 | [-0.879, +0.927] | [+0.000, +0.040] | 0.958 | inconclusive |
| no_layernorm | 32 | 8 | `combined` | `per_layer_output_error` | +0.013 | [-0.823, +0.849] | [+0.007, +0.019] | 0.976 | inconclusive |
| no_layernorm | 32 | 8 | `gain_aware_composed` | `weight_frobenius` | +0.449 | [-0.484, +1.382] | [+0.351, +0.602] | 0.346 | inconclusive |
| no_layernorm | 32 | 8 | `gain_aware_composed` | `weight_magnitude` | +0.404 | [-0.479, +1.287] | [+0.302, +0.540] | 0.37 | inconclusive |
| no_layernorm | 32 | 8 | `gain_aware_composed` | `activation_magnitude` | +0.929 | [-0.001, +1.859] | [+0.777, +1.169] | 0.0501 | inconclusive |
| no_layernorm | 32 | 8 | `gain_aware_composed` | `hessian_diag` | +0.246 | [-0.761, +1.253] | [+0.215, +0.320] | 0.632 | inconclusive |
| no_layernorm | 32 | 8 | `gain_aware_composed` | `per_layer_output_error` | +0.246 | [-0.680, +1.171] | [+0.193, +0.340] | 0.603 | inconclusive |

## 3. Model-level aggregation

### layernorm (5 model clusters, achieved n_eff mean +10.75, achieved MDE(z) +0.636 vs predeclared 0.33-0.79)

| variant | rho_mean | n_eff mean | z pooled | z CI | k units | tau^2 | I^2 |
|---|---|---|---|---|---|---|---|
| `weight_frobenius` | +0.4097 | +10.69 | +0.437 | [+0.308, +0.566] | 30 | +0.0000 | +0.000 |
| `weight_magnitude` | +0.3654 | +9.66 | +0.384 | [+0.246, +0.523] | 30 | +0.0000 | +0.000 |
| `activation_magnitude` | +0.1328 | +9.01 | +0.133 | [-0.013, +0.279] | 30 | +0.0000 | +0.000 |
| `per_layer_output_error` | +0.1559 | +11.11 | +0.183 | [+0.058, +0.309] | 30 | +0.0000 | +0.000 |
| `in_situ_output_error` | +0.5223 | +6.32 | +0.718 | [+0.522, +0.915] | 30 | +0.0000 | +0.000 |
| `gain_aware_composed` | +0.8299 | +4.49 | +1.110 | [+0.816, +1.403] | 30 | +0.0000 | +0.000 |
| `hessian_diag` | +0.1634 | +10.51 | +0.198 | [+0.068, +0.329] | 30 | +0.0000 | +0.000 |
| `quant_residual_stats` | -0.0445 | +8.38 | -0.042 | [-0.196, +0.113] | 30 | +0.0000 | +0.000 |
| `spectral_summary` | +0.1989 | +10.24 | +0.233 | [+0.100, +0.366] | 30 | +0.0000 | +0.000 |
| `combined` | +0.1703 | +10.75 | +0.195 | [+0.066, +0.323] | 30 | +0.0000 | +0.000 |

Paired contrasts (candidate − comparator), analytic random-effects CI and cluster-bootstrap CI:

| candidate | comparator | delta_z | CI (analytic) | boot CI | agree | p | MDE(z) achieved | verdict |
|---|---|---|---|---|---|---|---|---|
| `combined` | `weight_frobenius` | -0.256 | [-0.440, -0.071] | [-0.269, -0.238] | yes | 0.00662 | +0.636 | refuted |
| `combined` | `weight_magnitude` | -0.195 | [-0.386, -0.005] | [-0.220, -0.173] | yes | 0.0447 | +0.636 | refuted |
| `combined` | `activation_magnitude` | +0.056 | [-0.142, +0.253] | [+0.042, +0.070] | **no** | 0.581 | +0.636 | inconclusive |
| `combined` | `hessian_diag` | +0.004 | [-0.180, +0.188] | [-0.013, +0.026] | yes | 0.965 | +0.636 | inconclusive |
| `combined` | `per_layer_output_error` | +0.014 | [-0.166, +0.195] | [+0.002, +0.027] | **no** | 0.875 | +0.636 | inconclusive |
| `gain_aware_composed` | `weight_frobenius` | +0.683 | [+0.346, +1.021] | [+0.630, +0.763] | yes | 7.36e-05 | +1.453 | supported |
| `gain_aware_composed` | `weight_magnitude` | +0.746 | [+0.404, +1.087] | [+0.700, +0.806] | yes | 1.87e-05 | +1.453 | supported |
| `gain_aware_composed` | `activation_magnitude` | +1.004 | [+0.658, +1.349] | [+0.948, +1.065] | yes | 1.23e-08 | +1.453 | supported |
| `gain_aware_composed` | `hessian_diag` | +0.890 | [+0.552, +1.229] | [+0.857, +0.949] | yes | 2.53e-07 | +1.453 | supported |
| `gain_aware_composed` | `per_layer_output_error` | +0.912 | [+0.575, +1.249] | [+0.864, +0.970] | yes | 1.15e-07 | +1.453 | supported |

### no_layernorm (5 model clusters, achieved n_eff mean +9.13, achieved MDE(z) +0.715 vs predeclared 0.33-0.79)

| variant | rho_mean | n_eff mean | z pooled | z CI | k units | tau^2 | I^2 |
|---|---|---|---|---|---|---|---|
| `weight_frobenius` | +0.5667 | +9.37 | +0.609 | [+0.467, +0.751] | 30 | +0.0000 | +0.000 |
| `weight_magnitude` | +0.5819 | +8.76 | +0.669 | [+0.520, +0.819] | 30 | +0.0000 | +0.000 |
| `activation_magnitude` | +0.3947 | +6.80 | +0.409 | [+0.226, +0.593] | 30 | +0.0000 | +0.000 |
| `per_layer_output_error` | +0.4537 | +8.84 | +0.479 | [+0.331, +0.627] | 30 | +0.0000 | +0.000 |
| `in_situ_output_error` | +0.1513 | +8.92 | +0.145 | [-0.002, +0.292] | 30 | +0.0000 | +0.000 |
| `gain_aware_composed` | +0.8200 | +4.86 | +1.130 | [+0.867, +1.392] | 30 | +0.0000 | +0.000 |
| `hessian_diag` | +0.4771 | +9.57 | +0.473 | [+0.334, +0.613] | 30 | +0.0000 | +0.000 |
| `quant_residual_stats` | +0.3073 | +7.97 | +0.300 | [+0.140, +0.461] | 30 | +0.0000 | +0.000 |
| `spectral_summary` | +0.5070 | +8.89 | +0.468 | [+0.320, +0.615] | 30 | +0.0000 | +0.000 |
| `combined` | +0.4542 | +9.13 | +0.432 | [+0.287, +0.576] | 30 | +0.0000 | +0.000 |

Paired contrasts (candidate − comparator), analytic random-effects CI and cluster-bootstrap CI:

| candidate | comparator | delta_z | CI (analytic) | boot CI | agree | p | MDE(z) achieved | verdict |
|---|---|---|---|---|---|---|---|---|
| `combined` | `weight_frobenius` | -0.164 | [-0.375, +0.046] | [-0.210, -0.120] | **no** | 0.126 | +0.715 | inconclusive |
| `combined` | `weight_magnitude` | -0.191 | [-0.409, +0.027] | [-0.231, -0.164] | **no** | 0.0863 | +0.715 | inconclusive |
| `combined` | `activation_magnitude` | +0.053 | [-0.189, +0.295] | [+0.001, +0.121] | **no** | 0.668 | +0.715 | inconclusive |
| `combined` | `hessian_diag` | -0.033 | [-0.234, +0.168] | [-0.045, -0.021] | **no** | 0.748 | +0.715 | inconclusive |
| `combined` | `per_layer_output_error` | -0.004 | [-0.213, +0.205] | [-0.014, +0.004] | yes | 0.968 | +0.715 | inconclusive |
| `gain_aware_composed` | `weight_frobenius` | +0.537 | [+0.226, +0.848] | [+0.447, +0.646] | yes | 0.000721 | +1.299 | supported |
| `gain_aware_composed` | `weight_magnitude` | +0.503 | [+0.192, +0.815] | [+0.420, +0.608] | yes | 0.00154 | +1.299 | supported |
| `gain_aware_composed` | `activation_magnitude` | +0.726 | [+0.387, +1.065] | [+0.643, +0.823] | yes | 2.69e-05 | +1.299 | supported |
| `gain_aware_composed` | `hessian_diag` | +0.653 | [+0.343, +0.962] | [+0.571, +0.746] | yes | 3.6e-05 | +1.299 | supported |
| `gain_aware_composed` | `per_layer_output_error` | +0.668 | [+0.357, +0.980] | [+0.586, +0.763] | yes | 2.65e-05 | +1.299 | supported |

### all_fixtures (10 model clusters, achieved n_eff mean +9.94, achieved MDE(z) +0.476 vs predeclared 0.33-0.79)

| variant | rho_mean | n_eff mean | z pooled | z CI | k units | tau^2 | I^2 |
|---|---|---|---|---|---|---|---|
| `weight_frobenius` | +0.4882 | +10.03 | +0.515 | [+0.419, +0.610] | 60 | +0.0000 | +0.000 |
| `weight_magnitude` | +0.4736 | +9.21 | +0.517 | [+0.415, +0.618] | 60 | +0.0000 | +0.000 |
| `activation_magnitude` | +0.2637 | +7.91 | +0.240 | [+0.126, +0.354] | 60 | +0.0000 | +0.000 |
| `per_layer_output_error` | +0.3048 | +9.97 | +0.307 | [+0.211, +0.403] | 60 | +0.0000 | +0.000 |
| `in_situ_output_error` | +0.3368 | +7.62 | +0.351 | [+0.233, +0.468] | 60 | +0.0000 | +0.000 |
| `gain_aware_composed` | +0.8249 | +4.67 | +1.121 | [+0.925, +1.316] | 60 | +0.0000 | +0.000 |
| `hessian_diag` | +0.3202 | +10.04 | +0.326 | [+0.231, +0.422] | 60 | +0.0000 | +0.000 |
| `quant_residual_stats` | +0.1314 | +8.17 | +0.123 | [+0.011, +0.234] | 60 | +0.0000 | +0.000 |
| `spectral_summary` | +0.3529 | +9.56 | +0.338 | [+0.239, +0.437] | 60 | +0.0000 | +0.000 |
| `combined` | +0.3123 | +9.94 | +0.299 | [+0.203, +0.395] | 60 | +0.0000 | +0.000 |

Paired contrasts (candidate − comparator), analytic random-effects CI and cluster-bootstrap CI:

| candidate | comparator | delta_z | CI (analytic) | boot CI | agree | p | MDE(z) achieved | verdict |
|---|---|---|---|---|---|---|---|---|
| `combined` | `weight_frobenius` | -0.216 | [-0.355, -0.077] | [-0.247, -0.175] | yes | 0.00228 | +0.476 | refuted |
| `combined` | `weight_magnitude` | -0.193 | [-0.337, -0.050] | [-0.214, -0.175] | yes | 0.00827 | +0.476 | refuted |
| `combined` | `activation_magnitude` | +0.054 | [-0.098, +0.207] | [+0.030, +0.082] | **no** | 0.485 | +0.476 | refuted |
| `combined` | `hessian_diag` | -0.013 | [-0.149, +0.123] | [-0.029, +0.003] | yes | 0.853 | +0.476 | refuted |
| `combined` | `per_layer_output_error` | +0.006 | [-0.130, +0.143] | [-0.004, +0.016] | yes | 0.926 | +0.476 | refuted |
| `gain_aware_composed` | `weight_frobenius` | +0.604 | [+0.375, +0.833] | [+0.530, +0.682] | yes | 2.31e-07 | +0.968 | supported |
| `gain_aware_composed` | `weight_magnitude` | +0.613 | [+0.383, +0.843] | [+0.520, +0.707] | yes | 1.74e-07 | +0.968 | supported |
| `gain_aware_composed` | `activation_magnitude` | +0.862 | [+0.620, +1.104] | [+0.755, +0.965] | yes | 2.85e-12 | +0.968 | supported |
| `gain_aware_composed` | `hessian_diag` | +0.761 | [+0.532, +0.989] | [+0.672, +0.852] | yes | 6.66e-11 | +0.968 | supported |
| `gain_aware_composed` | `per_layer_output_error` | +0.781 | [+0.552, +1.009] | [+0.686, +0.873] | yes | 2.31e-11 | +0.968 | supported |

## 4. Hypothesis verdicts against the frozen section 11 falsifiers

### H2 — proxy validity

Frozen falsifier wording (section 11.1, H2): *"The proxy's **model-level** correlation with measured degradation is **not** higher than weight-space Frobenius error — the Fisher-z random-effects CI on the paired difference includes 0 and is narrower than the predeclared section 7.5 MDE (a CI wider than the MDE is reported **inconclusive**, not refuting)."*

Local-fixture evidence (S=5 seeds, model as unit, 13 modules per model):

* **layernorm**, candidate `combined` vs predeclared primary comparator `weight_frobenius`: delta_z = -0.256, analytic CI [-0.440, -0.071], bootstrap CI [-0.269, -0.238], p = 0.00662 -> **REFUTED** (decisive: random-effects CI [-0.440, -0.071] excludes 0 and delta_z = -0.256 < 0.)
* **layernorm**, candidate `gain_aware_composed` vs predeclared primary comparator `weight_frobenius`: delta_z = +0.683, analytic CI [+0.346, +1.021], bootstrap CI [+0.630, +0.763], p = 7.36e-05 -> **SUPPORTED** (decisive: random-effects CI [0.346, 1.021] excludes 0 and delta_z = +0.683 > 0.)
* **no_layernorm**, candidate `combined` vs predeclared primary comparator `weight_frobenius`: delta_z = -0.164, analytic CI [-0.375, +0.046], bootstrap CI [-0.210, -0.120], p = 0.126 -> **INCONCLUSIVE** (not decisive: random-effects CI [-0.375, 0.046] includes 0 and has width 0.421, which is not narrower than the predeclared MDE band [0.33, 0.79]; section 11.1 makes this inconclusive, not a refutation. NOTE: the cluster-bootstrap CI [-0.210, -0.120] disagrees about excluding 0.)
* **no_layernorm**, candidate `gain_aware_composed` vs predeclared primary comparator `weight_frobenius`: delta_z = +0.537, analytic CI [+0.226, +0.848], bootstrap CI [+0.447, +0.646], p = 0.000721 -> **SUPPORTED** (decisive: random-effects CI [0.226, 0.848] excludes 0 and delta_z = +0.537 > 0.)
* **all_fixtures**, candidate `combined` vs predeclared primary comparator `weight_frobenius`: delta_z = -0.216, analytic CI [-0.355, -0.077], bootstrap CI [-0.247, -0.175], p = 0.00228 -> **REFUTED** (decisive: random-effects CI [-0.355, -0.077] excludes 0 and delta_z = -0.216 < 0.)
* **all_fixtures**, candidate `gain_aware_composed` vs predeclared primary comparator `weight_frobenius`: delta_z = +0.604, analytic CI [+0.375, +0.833], bootstrap CI [+0.530, +0.682], p = 2.31e-07 -> **SUPPORTED** (decisive: random-effects CI [0.375, 0.833] excludes 0 and delta_z = +0.604 > 0.)

**Plain statement of the local-fixture evidence.**

* Candidate `combined` against the 5 comparators in each of the 3 fixture groups (15 aggregate contrasts): the candidate's advantage is **supported** in 0 (none); the candidate is **significantly worse** (analytic CI entirely below 0) in 4 (layernorm/weight_frobenius, layernorm/weight_magnitude, all_fixtures/weight_frobenius, all_fixtures/weight_magnitude); the advantage is **below the predeclared MDE** (CI includes 0 but is narrower than 0.33) in 3 (all_fixtures/activation_magnitude, all_fixtures/hessian_diag, all_fixtures/per_layer_output_error); **inconclusive** in 8 (layernorm/activation_magnitude, layernorm/hessian_diag, layernorm/per_layer_output_error, no_layernorm/weight_frobenius, no_layernorm/weight_magnitude, no_layernorm/activation_magnitude, no_layernorm/hessian_diag, no_layernorm/per_layer_output_error).
* Candidate `gain_aware_composed` against the 5 comparators in each of the 3 fixture groups (15 aggregate contrasts): the candidate's advantage is **supported** in 15 (layernorm/weight_frobenius, layernorm/weight_magnitude, layernorm/activation_magnitude, layernorm/hessian_diag, layernorm/per_layer_output_error, no_layernorm/weight_frobenius, no_layernorm/weight_magnitude, no_layernorm/activation_magnitude, no_layernorm/hessian_diag, no_layernorm/per_layer_output_error, all_fixtures/weight_frobenius, all_fixtures/weight_magnitude, all_fixtures/activation_magnitude, all_fixtures/hessian_diag, all_fixtures/per_layer_output_error); the candidate is **significantly worse** (analytic CI entirely below 0) in 0 (none); the advantage is **below the predeclared MDE** (CI includes 0 but is narrower than 0.33) in 0 (none); **inconclusive** in 0 (none).

* The local-fixture evidence is therefore **not consistent** with `combined` beating every predeclared comparator: it is significantly worse at layernorm/weight_frobenius, layernorm/weight_magnitude, all_fixtures/weight_frobenius, all_fixtures/weight_magnitude; its advantage is below the MDE at all_fixtures/activation_magnitude, all_fixtures/hessian_diag, all_fixtures/per_layer_output_error; and it is inconclusive at layernorm/activation_magnitude, layernorm/hessian_diag, layernorm/per_layer_output_error, no_layernorm/weight_frobenius, no_layernorm/weight_magnitude, no_layernorm/activation_magnitude, no_layernorm/hessian_diag, no_layernorm/per_layer_output_error.
* The local-fixture evidence is therefore **consistent** with `gain_aware_composed` beating every predeclared comparator and the naive baseline.

Interval agreement across all aggregate contrasts: 7 of 30 contrasts disagree: layernorm/combined vs activation_magnitude; layernorm/combined vs per_layer_output_error; no_layernorm/combined vs weight_frobenius; no_layernorm/combined vs weight_magnitude; no_layernorm/combined vs activation_magnitude; no_layernorm/combined vs hessian_diag; all_fixtures/combined vs activation_magnitude. In those rows the two intervals point in the same direction but one covers 0; the verdict follows the analytic random-effects interval and the disagreement is stated.

### H4 — layer-wise mixed rank + precision under a global budget

Frozen falsifier wording (section 11.1, H4): *"Proxy-allocated $(r_\ell,b_\ell)$ does **not** dominate uniform rank/bit at any predeclared budget beyond CIs, or is matched by the HAWQ-V2-style bit-only allocator at equal bytes."*; the traceability table (section 11.0) states the same falsifier as *"no predeclared budget where it dominates beyond CIs, or matched by the bit-only allocator"*.

**No local-fixture evidence bears on this statement.** The frozen section 8 matrix assigns H4's confirmatory cell (layer-wise mixed rank + mixed precision vs uniform and vs HAWQ-V2-style, >=5 seeds) to the `CLOUD-COLAB` substrate; there is no `LOCAL-FIXTURE` cell for H4. The local fixture informs only H4's premise (the proxy's ranking ability, section H2 above) and the ranking ability of the HAWQ-V2-style average-Hessian eigenvalue surrogate `hessian_diag`, which the candidate beats with a CI excluding 0. The H4 verdict therefore remains **not run / inconclusive** locally: it is not refuted and it is not supported by any measurement in this document.

## 5. Cloud cells — NOT RUN

| cell | hypothesis | tier | substrate | status | reason |
|---|---|---|---|---|---|
| Proxy vs weight-space Frobenius error rank correlation (activation covariance + quantization residual in the proxy), tiny LM, >=5 seeds | H2 | 1 | `CLOUD-COLAB` | **NOT RUN** | Tier-1 training/evaluation runs only on the cloud notebook substrate (AGENTS.md section 2.3/2b); no validated run_manifest.json exists yet. No Tier-1 number is implied anywhere in this document. |
| Layer-wise mixed rank + mixed precision vs uniform and vs HAWQ-V2-style, >=5 seeds | H4 | 1 | `CLOUD-COLAB` | **NOT RUN** | Same as above: the H4 confirmatory cell is CLOUD-COLAB (section 8 row 7) and has not been executed. |

No Tier-1 number is implied anywhere in this document. The H2 and H4 confirmatory cells require the Colab run (Colab/Kaggle cloud notebook substrate); they count as a result only once a validated `run_manifest.json` and checksum-validated artifacts exist (`AGENTS.md` section 2b, `preregistration.md` section 8).

## 6. Limits (must be quoted with any use of the tables above)

* This is the **local Tier-0 fixture** half of the M4 gate, not the confirmatory Tier-1 cell: 3 blocks / 13 modules per model versus the pinned Tier-1 `L_b=8` / 32 modules, and a synthetic next-token task rather than WikiText-2.
* The achieved MDE at the achieved `n_eff` is +0.476 on the combined set versus the predeclared 0.33-0.79. A null whose CI extends beyond the MDE is reported inconclusive, never as a refutation of a smaller effect (section 7.7).
* `n_eff` is a bootstrap estimate over 13 modules and is noisy; it is reported per model and its mean is what the achieved MDE uses.
* The `refuted` label covers two different situations and the artifact distinguishes them: an analytic CI entirely below 0 (the candidate is *significantly worse*) and an analytic CI that includes 0 but is narrower than the predeclared **optimistic** MDE of 0.33 (the advantage is below the MDE). Rule 3 measures "narrower" against the optimistic end of the predeclared band; using the conservative end (0.79) instead would leave those rows inconclusive. Both ends are recorded in the artifact, so the reading is auditable.
* The cluster-bootstrap CI resamples trained models (5 per fixture, 10 combined); with so few clusters its percentile coverage is coarse and it can disagree with the analytic CI about whether 0 is excluded. The verdict rule uses the analytic random-effects interval (section 7.2) and every disagreement is visible in the tables.
* Float values in the artifact are stored at 6 significant digits; the sweep itself is exact and deterministic, and re-running reproduces the analysis from unrounded values.
* Nothing here is a storage, latency or kernel measurement: class 1/2 only (`AGENTS.md` section 5). The equal-memory gate and the H5 chain are separate cells.

