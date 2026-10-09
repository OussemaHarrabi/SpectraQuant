# Ablation report (Milestone 8, local half — component, calibration, outlier, seed)

Owner: proxy/ablation slice. Machine-readable artifact: `artifacts/sample-results/ablations/ablations.json` (regenerate with `uv run python scripts/experiments/ablations.py`). This report is generated from that document, so the numbers below and the JSON cannot drift apart.

Substrate `LOCAL-FIXTURE` (CPU workstation); every number is measurement class **2** (`AGENTS.md` section 5). Seeds `[0, 1, 2, 3, 4]` x cells `[[8, 4], [16, 4], [32, 4], [8, 8], [16, 8], [32, 8]]` on fixtures `['layernorm', 'no_layernorm']` for the component ablation; the calibration and outlier studies use the representative cell `[8, 4]`. 10 trained models and 60 component cells; wall time 68.7 s; git commit `2f6d7a9be6cdcc64188dcc1c0948ef1aca5bbd63` (dirty=True).

**All Tier-1/Tier-2 cells are NOT RUN** (section 5 below). No Tier-1 number is implied by anything in this document; the local evidence is fixture scale only.

## 1. Proxy component ablation — which component carries the signal

The declared candidate is `gain_aware_composed` = `g^2 * mean_i ||x_i delta^T||^2` (the gain term applied to the **naive** pre-normalisation per-layer error). The component factorial spans the error basis `{naive, in-situ}` x `{no gain, gain}` plus the untuned `combined`:

| variant | in-situ | gain | role |
|---|---|---|---|
| `per_layer_output_error` | no | no | naive per-layer output error (no in-situ, no gain) |
| `in_situ_output_error` | yes | no | in-situ normalisation Jacobian only (no gain term) |
| `gain_aware_composed` | no | yes | declared candidate: gain-only, no in-situ (the frozen interface) |
| `gain_aware_in_situ` | yes | yes | full composition: in-situ basis x gain (local variant) |
| `combined` | no | yes* | untuned combination (+ Hessian-diagonal and rounding-residual terms) |

`combined` carries the gain through its `gain_aware` component. `gain_aware_in_situ` is implemented in `src/spectraquant/proxies/ablations.py` and is **not** in the frozen registry.

### 1.1 layernorm

| variant | rho_mean | n_eff_mean | k |
|---|---|---|---|
| `per_layer_output_error` | +0.1559 | +11.11 | 30 |
| `in_situ_output_error` | +0.5223 | +6.32 | 30 |
| `gain_aware_composed` | +0.8299 | +4.49 | 30 |
| `gain_aware_in_situ` | +0.7603 | +7.57 | 30 |
| `combined` | +0.1703 | +10.75 | 30 |

| candidate | comparator | delta_z | CI (analytic) | boot CI | p | verdict |
|---|---|---|---|---|---|---|
| `gain_aware_composed` | `per_layer_output_error` | +0.912 | [+0.575,+1.249] | [+0.864,+0.970] | 1.15e-07 | supported |
| `gain_aware_composed` | `in_situ_output_error` | +0.380 | [+0.007,+0.753] | [+0.321,+0.459] | 0.0459 | supported |
| `gain_aware_composed` | `gain_aware_in_situ` | +0.150 | [-0.217,+0.517] | [+0.096,+0.229] | 0.423 | inconclusive |
| `gain_aware_composed` | `combined` | +0.898 | [+0.561,+1.236] | [+0.845,+0.964] | 1.88e-07 | supported |
| `gain_aware_composed` | `weight_frobenius` | +0.683 | [+0.346,+1.021] | [+0.630,+0.763] | 7.36e-05 | supported |

### 1.2 no_layernorm

| variant | rho_mean | n_eff_mean | k |
|---|---|---|---|
| `per_layer_output_error` | +0.4537 | +8.84 | 30 |
| `in_situ_output_error` | +0.1513 | +8.92 | 30 |
| `gain_aware_composed` | +0.8200 | +4.86 | 30 |
| `gain_aware_in_situ` | +0.3749 | +5.97 | 30 |
| `combined` | +0.4542 | +9.13 | 30 |

| candidate | comparator | delta_z | CI (analytic) | boot CI | p | verdict |
|---|---|---|---|---|---|---|
| `gain_aware_composed` | `per_layer_output_error` | +0.668 | [+0.357,+0.980] | [+0.586,+0.763] | 2.65e-05 | supported |
| `gain_aware_composed` | `in_situ_output_error` | +1.013 | [+0.704,+1.323] | [+0.917,+1.134] | 1.42e-10 | supported |
| `gain_aware_composed` | `gain_aware_in_situ` | +0.750 | [+0.404,+1.095] | [+0.678,+0.844] | 2.13e-05 | supported |
| `gain_aware_composed` | `combined` | +0.680 | [+0.368,+0.992] | [+0.602,+0.768] | 1.92e-05 | supported |
| `gain_aware_composed` | `weight_frobenius` | +0.537 | [+0.226,+0.848] | [+0.447,+0.646] | 0.000721 | supported |

### 1.3 all_fixtures

| variant | rho_mean | n_eff_mean | k |
|---|---|---|---|
| `per_layer_output_error` | +0.3048 | +9.97 | 60 |
| `in_situ_output_error` | +0.3368 | +7.62 | 60 |
| `gain_aware_composed` | +0.8249 | +4.67 | 60 |
| `gain_aware_in_situ` | +0.5676 | +6.77 | 60 |
| `combined` | +0.3123 | +9.94 | 60 |

| candidate | comparator | delta_z | CI (analytic) | boot CI | p | verdict |
|---|---|---|---|---|---|---|
| `gain_aware_composed` | `per_layer_output_error` | +0.781 | [+0.552,+1.009] | [+0.686,+0.873] | 2.31e-11 | supported |
| `gain_aware_composed` | `in_situ_output_error` | +0.755 | [+0.517,+0.993] | [+0.525,+0.954] | 5.29e-10 | supported |
| `gain_aware_composed` | `gain_aware_in_situ` | +0.468 | [+0.216,+0.720] | [+0.268,+0.669] | 0.000268 | supported |
| `gain_aware_composed` | `combined` | +0.781 | [+0.552,+1.010] | [+0.694,+0.867] | 2.47e-11 | supported |
| `gain_aware_composed` | `weight_frobenius` | +0.604 | [+0.375,+0.833] | [+0.530,+0.682] | 2.31e-07 | supported |

### 1.4 Which component carries the signal

* **Gain term** — the pooled `gain_aware_composed - per_layer_output_error` contrast is delta_z = +0.781 with CI [+0.552,+1.009]: the gain term **carries the signal**.
* **In-situ term** — holding the gain fixed, `gain_aware_in_situ - gain_aware_composed (in-situ basis x gain, minus the gain-only candidate)` is delta_z = -0.468 with CI [-0.720,-0.216]: the in-situ normalisation basis **does not add signal** beyond the gain term (the reversed view is `gain_aware_composed - gain_aware_in_situ` = +0.468, CI [+0.216,+0.720]).
* **Extra terms** — `gain_aware_composed - combined` is delta_z = +0.781 with CI [+0.552,+1.010]: the untuned Hessian/rounding terms **do not help** relative to the gain-only candidate.
* **Naive vs in-situ-only basis** — `gain_aware_composed - in_situ_output_error` is delta_z = +0.755 with CI [+0.517,+0.993].

Pooled mean rho by component:

| variant | rho_mean |
|---|---|
| `per_layer_output_error` | +0.3048 |
| `in_situ_output_error` | +0.3368 |
| `gain_aware_composed` | +0.8249 |
| `gain_aware_in_situ` | +0.5676 |
| `combined` | +0.3123 |

**Statement.** The component that carries the ranking signal is the **downstream-gain term**: the candidate is ahead of the naive per-layer error (pooled delta_z = +0.781, CI [+0.552,+1.009], CI excludes 0) and of the in-situ-only variant (delta_z = +0.755, CI [+0.517,+0.993]). The ablation that **does not discriminate** is the in-situ normalisation term — at pooled fixture scale removing it is, if anything, better for the ranking (candidate - full composition = +0.468, CI [+0.216,+0.720]), and on the `layernorm` fixture alone the full composition versus the gain-only candidate is inconclusive (delta_z = +0.150, CI [-0.217,+0.517]). A null ablation is a result: the in-situ Jacobian is **not** the active ingredient at this scale, and the untuned `combined` variant is dominated by the candidate it contains (delta_z = +0.781, CI [+0.552,+1.010]).

## 2. Calibration-size sensitivity

The **target** (single-layer damage used for the ranking) is always measured on the fixed 2x24-token batch, so it does not move with the calibration size; only the proxy's calibration set (activations, residual errors, gains) grows. `rel_se` is the mean over layers of the naive plug-in estimator's relative standard error.

| tokens/layer | rho candidate | rho naive | rho in-situ | estimator rel_se |
|---|---|---|---|---|
| 8 | +0.7418 | +0.0918 | +0.1456 | +0.1171 |
| 16 | +0.7736 | +0.1049 | +0.1478 | +0.0911 |
| 32 | +0.7995 | +0.1000 | +0.1654 | +0.0670 |
| 64 | +0.8022 | +0.0918 | +0.1396 | +0.0480 |
| 128 | +0.7973 | +0.0868 | +0.1522 | +0.0340 |
| 256 | +0.7901 | +0.0885 | +0.1560 | +0.0237 |
| 512 | +0.8027 | +0.0841 | +0.1626 | +0.0167 |

**Where it stops improving.** the candidate's rho moves by less than 0.02 per step from 64 tokens per layer upward (the swept range ends at 512), so the proxy stops improving at or below 64. The relative SE falls roughly as 1/sqrt(tokens) across the whole range, but the ranking itself is flat well before the largest calibration set: more calibration data does not buy a better ranking here.

## 3. Outlier behaviour

**Injection procedure (exact).** train the fixture normally; select k = max(1, round(0.1 * in_features)) input channels per layer without replacement by random.Random(20261010) over the sorted layer order; deep-copy the model and multiply those channels of every compressible weight by the severity; measure on the fixed 2x24-token batch. severity 1.0 is the no-op control.

`res_group` is the summed per-group rounding residual (the study spec, `group_size=32`), `res_tensor` the per-tensor control; `t/g` is their ratio; `dmg/res` the damage per unit rounding residual; `rho_cand`/`rho_naive` are the model-level ranking correlations against the injected model's damage.

| severity | damage | res_group | res_tensor | t/g | dmg/res | rho candidate | rho naive |
|---|---|---|---|---|---|---|---|
| 1.0 | +28.376 | +13.751 | +27.259 | +2.00 | +2.429 | +0.818 | +0.106 |
| 2.0 | +53.612 | +31.225 | +132.564 | +4.15 | +2.218 | +0.736 | +0.029 |
| 4.0 | +220.504 | +484.769 | +3141.140 | +6.15 | +1.389 | +0.858 | -0.380 |
| 8.0 | +270.695 | +2582870.000 | +17126500.000 | +7.06 | +0.146 | +0.862 | -0.418 |
| 16.0 | +160.453 | +10372400000000.000 | +45450600000000.000 | +3.96 | +0.003 | +0.926 | -0.377 |

**Reading.** 5 severities were injected (1.0 .. 16.0). The measured damage is **non-monotone**: it peaks at severity 8.0 (+270.695) and falls to +160.453 at the highest severity (the injected model is itself re-measured, so an extreme spike changes what 'damage' the low-rank reconstruction incurs). The rounding residual rises monotonically with severity for both quantizers (+13.751 -> +10372400000000.000 per-group, +27.259 -> +45450600000000.000 per-tensor), and per-group is below per-tensor at every severity (`t/g` in [+2.00, +7.06]): the per-tensor global scale must absorb the spike, while per-group keeps one scale per block. At the top severity the residual is dominated by squared block scales (a spiked block rounds its non-spiked entries to zero), so the per-group residual saturates at a large value. The candidate's ranking does **not** collapse — it stays in [+0.736, +0.926] across all severities — whereas the naive per-layer error falls from +0.106 to -0.377. The damage / rounding-residual ratio falls with severity (+2.429 -> +0.003): the outliers push a larger fraction of the remaining error into the quantizer rather than the retained low-rank signal.

## 4. Seed-count sensitivity

Pooled `gain_aware_composed - weight_frobenius` contrast versus the number of seeds. This is a sensitivity view of the pre-registered fixed-n rule (section 7.6): it shows what a smaller seed count would have reported, it is **not** an optional-stopping licence.

| seeds | k units | delta_z | CI (analytic) | boot CI | MDE(z) | verdict |
|---|---|---|---|---|---|---|
| 1 | 12 | +0.574 | [+0.012,+1.137] | [+0.476,+0.642] | +2.451 | supported |
| 2 | 24 | +0.543 | [+0.167,+0.919] | [+0.466,+0.621] | +1.642 | supported |
| 3 | 36 | +0.545 | [+0.251,+0.839] | [+0.467,+0.620] | +1.232 | supported |
| 4 | 48 | +0.571 | [+0.313,+0.828] | [+0.495,+0.658] | +1.091 | supported |
| 5 | 60 | +0.604 | [+0.375,+0.833] | [+0.530,+0.682] | +0.968 | supported |

## 5. Hypothesis verdicts against the frozen section 11 falsifiers (fixture scale only)

### H1

**Frozen falsifier wording (section 11):** "No regime found where truncation reduces reconstruction error while simultaneously increasing rounding sensitivity at equal stored bytes (95% CI of the sensitivity difference includes 0, or the two move together), on >=5 Tier-1 seeds (cloud)."

**Verdict: INCONCLUSIVE** (not tested by this slice) — H1 requires the controlled factorization-error vs rounding-sensitivity regime comparison; this slice measures proxy components, calibration size, outlier robustness and seed sensitivity. No H1 contrast is computed here.

### H2

**Frozen falsifier wording (section 11):** "The proxy's **model-level** correlation with measured degradation is **not** higher than weight-space Frobenius error — the Fisher-z random-effects CI on the paired difference includes 0 and is narrower than the predeclared section 7.5 MDE (a CI wider than the MDE is reported **inconclusive**, not refuting)."

Contrast `gain_aware_composed - weight_frobenius`: delta_z = +0.604, analytic CI [+0.375,+0.833], bootstrap CI [+0.530,+0.682], p = 2.31e-07.

**Verdict: SUPPORTED** — random-effects CI [0.375, 0.833] excludes 0 and delta_z = +0.604 > 0.

### H4

**Frozen falsifier wording (section 11):** "Proxy-allocated $(r_\ell,b_\ell)$ does **not** dominate uniform rank/bit at any predeclared budget beyond CIs, or is matched by the HAWQ-V2-style bit-only allocator at equal bytes."

**Verdict: INCONCLUSIVE** (not tested by this slice) — H4 requires the Pareto-hypervolume frontier comparison against uniform and the HAWQ-V2-style allocator; this slice supplies only the proxy-component evidence. No H4 dominance contrast is computed here.

The vocabulary is mapped to the requested form: the frozen rule's `refuted` is reported as `contradicted`; `supported` and `inconclusive` are unchanged. Only **H2** is tested by this slice. **H1** and **H4** are `inconclusive` at fixture scale because the cells that test them are `NOT RUN` — never as a refutation.

### 5.1 Tier-1 / Tier-2 cells (all NOT RUN)

| cell | hypothesis | tier | substrate | status |
|---|---|---|---|---|
| Factorization error vs rounding sensitivity, tiny LM, >=5 seeds | H1 | 1 | `CLOUD-COLAB` | **NOT RUN** |
| Proxy vs weight-space Frobenius error rank correlation, activation covariance + quantization residual in the proxy, tiny LM, >=5 seeds | H2 | 1 | `CLOUD-COLAB` | **NOT RUN** |
| Layer-wise mixed rank + mixed precision vs uniform and vs HAWQ-V2-style, >=5 seeds | H4 | 1 | `CLOUD-COLAB` | **NOT RUN** |
| TinyLlama-1.1B-class + WikiText-2, all arms, >=3 seeds | H1-H5 | 2 | `CLOUD-GPU` | **NOT RUN** |

No Tier-1 or Tier-2 number is claimed, implied or estimated anywhere in this document. Each of those cells becomes a result only from a validated `run_manifest.json` produced on the cloud substrate (`preregistration.md` section 8).

