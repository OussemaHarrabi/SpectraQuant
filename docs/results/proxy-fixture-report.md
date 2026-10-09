# Proxy fixture report (Milestone 2, proxy slice)

Owner: proxy slice. Evidence: `scripts/experiments/proxy_fixture_measurement.py` and the committed
artifact `artifacts/sample-results/proxy-fixture/proxy-fixture.json` (regenerate with
`uv run python scripts/experiments/proxy_fixture_measurement.py`).

This is the Milestone-2 evidence for the proxy contract, and it deliberately **publishes the failure of
the naive proxy** rather than hiding it. It is a fixture result — one seed, one small model, one
(rank, bits) setting — and it is *not* the Milestone-4 validation, which repeats this across seeds and
compression configurations with the model as the statistical unit.

## 1. Setup

* Fixture: the committed `TinyConfig` defaults — 3-block pre-norm transformer, `d_model=48`, 2 heads,
  `d_ff=96`, `vocab=24`, `seq_len=24`, trained 120 steps on a deterministic synthetic next-token task
  (loss 3.24 → 0.008, ≈0.5 s on CPU). 13 compressible linear modules per model. (An earlier draft of
  this report described a smaller exploratory fixture; the committed artifact is authoritative.)
* Compression: rank-then-quantize, `rank=8`, `bits=4`, per-group symmetric, `group_size=32`.
* Two runs: LayerNorm present (the realistic case) and LayerNorm replaced by identity (the control
  used by the adversarial review).
* 9 compressible linear layers per model; damage = mean squared change of the final hidden state when
  the layer is compressed alone, measured end-to-end.

## 2. Measured results

| variant | ρ vs single-layer damage (LayerNorm) | ρ (no LayerNorm) | joint / Σproxy (LayerNorm) |
|---|---|---|---|
| `per_layer_output_error` (naive baseline) | **−0.060** | +0.242 | 6.9e-02 |
| `in_situ_output_error` (post-normalisation Jacobian) | +0.473 | +0.016 | 1.7e-01 |
| `gain_aware_composed` (× estimated downstream gain²) | **+0.830** | +0.731 | **8.5e-01** |
| `hessian_diag` | −0.044 | +0.286 | 8.3e-02 |
| `quant_residual_stats` | −0.016 | +0.269 | 4.4e+00 |
| `spectral_summary` | −0.044 | +0.286 | 8.4e-02 |
| `weight_frobenius` (H2 comparator) | +0.412 | +0.429 | 8.7e-02 |
| `weight_magnitude` (H2 comparator) | +0.357 | +0.467 | 6.1e-03 |
| `activation_magnitude` (H2 comparator) | +0.121 | +0.412 | 3.9e-02 |
| `combined` (equal-weight sum, untuned) | −0.011 | +0.286 | 7.4e-02 |

Joint damage / Σ(single-layer damage) = **1.375** (LayerNorm) and **1.083** (no LayerNorm), i.e. the
per-layer damages compose super-additively here, consistent with the review's 0.56–1.19 range.

## 3. What this says

1. **The naive per-layer output error does not rank layers by downstream damage once LayerNorm is
   present** (ρ = −0.06 here; the review measured 0.119 at r=32/b=4 on a larger fixture). It is
   retained as a baseline and its failure is a reportable result, not a bug to patch away.
2. **The gain-aware variant recovers ranking ability and beats every predeclared comparator on this
   fixture**: ρ = +0.830 with LayerNorm versus −0.060 (naive per-layer output error), +0.412
   (weight-space Frobenius), +0.357 (weight magnitude), +0.121 (activation magnitude) and −0.044
   (Hessian diagonal). Its summed value is also the only one within a factor of ~1.2 of the joint
   damage (0.85 vs 0.069 for the naive form). **This is one seed and one configuration: it is the
   M4 gate's job to confirm the ordering across seeds and (rank, bits) settings**, with the model as
   the resampling unit and the predeclared minimum detectable effect.
3. **The in-situ (normalisation-Jacobian) variant alone is not enough** (+0.473): mapping the layer
   error into the normalisation's output basis helps but does not model the gain product.
4. **The equal-weight `combined` variant is worse than its best component** (−0.011). The default
   weights are documented as untuned; this is a negative result about the naive combination, and it
   means the Milestone-4 work must tune or justify the weights rather than assume a sum helps.
5. **`quant_residual_stats` is a poor ranking signal on its own** (−0.016) but its value is the
   rounding-only error, which is the quantity hypotheses H1/H3 care about; it is kept for that role,
   not as a ranker.

## 4. Limits (must be quoted with any use of this table)

* One seed, one model, one `(rank, bits)` configuration, 9 layers. Rank correlations over 9 points are
  noisy; the Milestone-4 gate uses the model as the resampling unit across seeds and configurations
  (`preregistration.md` §7.0/§7.5) and reports a minimum detectable effect.
* Damage is measured in float32 through the network; proxy-versus-exact comparisons are float64 and
  are held to `rtol=1e-9` in `tests/unit/test_proxy_variants.py`.
* The gain estimator costs one extra forward pass per layer per calibration batch and measures a
  first-order RMS ratio on one batch; it is an estimate, and the variants that use it report
  `exact=False`.
* The artifact's `training.wall_time_s` varies between runs; the scientific content is deterministic
  (byte-identical apart from that field).
* Nothing here is a storage or latency measurement: every number is measurement class 1 or 2
  (`AGENTS.md` §5).
