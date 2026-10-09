# Allocation integration report (Milestone 6)

Owner: allocation-integration slice. Evidence: `scripts/experiments/allocation_frontier.py`,
`src/spectraquant/allocation/{proxy_adapter,validation}.py`, `tests/unit/test_allocation_integration.py`,
and the committed artifact `artifacts/sample-results/allocation-frontier/frontier.json` (plus one
deterministic manifest per budget/solver under `.../manifests/`). Regenerate with
`uv run python scripts/experiments/allocation_frontier.py`.

Every number below is generated from that artifact, so the tables and the JSON cannot drift apart.

## 1. Scope and measurement classes

This is a **Tier-0 fixture** result: one seed, one small synthetic model, one option grid. It is
**exploratory** and is **not** a Tier-1/2 result; the confirmatory H4 cell has not been run
(section 7). Every figure carries a class label (`AGENTS.md` section 5):

| quantity | class | how it is produced |
|---|---|---|
| `accounted_bytes` | **1** (analytical) | `spectraquant.quantization.accounting.accounted_bytes`, via the allocator's `cost_fn` |
| `measured_bytes` | **3** (packed storage) | real `measure_serialized_bytes` of the serialized factors |
| `predicted_error` | **2** (fake-quantization quality) | the `gain_aware_composed` proxy, evaluated under each option's `QuantSpec` |
| `measured_hidden_damage`, `measured_logits_damage` | **2** | float forward pass of the compressed fixture vs the uncompressed one |
| equal-memory gate | 1 / 3 | `spectraquant.reporting.comparability.assert_equal_memory` on class-3 measured bytes |

**Fixture scale.** `TinyConfig` defaults: a 3-block pre-norm transformer, `d_model=48`, 2 heads,
`d_ff=96`, `vocab=24`, `seq_len=24`, trained 120 steps on the deterministic synthetic next-token task
(final loss 0.0084). **13 compressible linear modules** per model
(`blocks.{0,1,2}.{qkv,proj,fc1,fc2}` plus `head`). Calibration/measurement batch: the fixed
`ids = arange(24) % 16` of the proxy fixture. This is a fixture, not a language model.

## 2. What was wired

* **Proxy -> allocator.** `src/spectraquant/allocation/proxy_adapter.py` turns per-layer
  `LayerInputs` plus the **declared candidate** proxy `gain_aware_composed` into the allocator's
  `error_fn`, scored through the frozen public `Proxy.score_model` entry point with a single-layer
  mapping. The context is **option-specific**: for each `(rank, bits)` the residual error and the
  downstream gain are measured at that option, so the proxy is not held at one reference compression.
* **Cost.** Unchanged: `make_problem` derives `cost_fn` from
  `spectraquant.quantization.accounting` (the one byte source of truth). `overhead_fn` is the default
  zero overhead; the container's own overhead is measured and reconciled (section 4).
* **Options.** ranks `{4, 8, 16}`, bits `{4, 8}` -> 6 options; per-group symmetric, `group_size=32`,
  factor axis 1. Ranks are restricted to `<= min(out, in)` so the factorization never clamps (a clamp
  would make the class-1 cost vacuous — `allocation_factor_state` refuses it).
* **Ladder.** The **predeclared** ladder is the sorted set of the six uniform-configuration accounted
  byte totals, `[7960, 9496, 9728, 16928, 17392, 31792]` B, fixed before any quality number is
  computed. Anchoring each point at a uniform total means every point has a uniform arm at exactly its
  own bytes, which is what makes the equal-memory gate meaningful. (Min feasible 7960 B; max 31792 B.)
* **Solvers.** `solve_uniform`, `solve_greedy`, and the exact `solve_ortools` (CP-SAT; the optional
  `alloc` extra is installed here). CP-SAT reproduces the exhaustive oracle exactly on small instances
  (allocator report section 1; `tests/unit/test_allocation_solvers.py`), so it is used as the exact
  solver on the 13-layer instance where enumeration (`6^13`) is infeasible.

## 3. Frontier (raw)

Per point, per solver: accounted bytes (class 1), measured bytes (class 3), their relative
difference, the proxy's predicted error (class 2), the measured end-to-end damage (class 2, hidden
state and logits), and the number of distinct `(rank, bits)` options used.

```
BUDGET  SOLVER   ACCT_B  MEAS_B  REL_DIFF  PRED_ERR   HID_DMG    LOG_DMG   DISTINCT
  7960  uniform    7960   7960  0.00e+00  5.2469e+01  5.1715e+01  7.0989e+01  1
  7960  greedy     7960   7960  0.00e+00  5.2469e+01  5.1715e+01  7.0989e+01  1
  7960  ortools    7960   7960  0.00e+00  5.2469e+01  5.1715e+01  7.0989e+01  1
  9496  uniform    9496   9496  0.00e+00  5.2337e+01  5.1930e+01  7.1409e+01  1
  9496  greedy     9496   9496  0.00e+00  5.0707e+01  5.0011e+01  6.9238e+01  4
  9496  ortools    9448   9448  0.00e+00  3.2695e+01  3.1912e+01  6.5667e+01  3
  9728  uniform    9728   9728  0.00e+00  3.3900e+01  2.9049e+01  4.1894e+01  1
  9728  greedy     9720   9720  0.00e+00  5.1798e+01  5.1505e+01  7.0524e+01  3
  9728  ortools    9672   9672  0.00e+00  3.1625e+01  3.0525e+01  6.4200e+01  3
 16928  uniform   16928  16928  0.00e+00  3.3206e+01  2.8566e+01  4.1478e+01  1
 16928  greedy    16888  16888  0.00e+00  3.0967e+01  2.7898e+01  6.3497e+01  4
 16928  ortools   16928  16928  0.00e+00  1.3159e+01  1.0477e+01  5.8667e+01  3
 17392  uniform   17392  17392  0.00e+00  1.3076e+01  1.0254e+01  1.1361e+01  1
 17392  greedy    17384  17384  0.00e+00  4.1136e+01  3.7831e+01  6.5840e+01  4
 17392  ortools   16960  16960  0.00e+00  1.3076e+01  1.0254e+01  5.8834e+01  2
 31792  uniform   31792  31792  0.00e+00  1.2235e+01  9.5506e+00  9.8375e+00  1
 31792  greedy    30784  30784  0.00e+00  1.2235e+01  9.5506e+00  5.8452e+01  2
 31792  ortools   30784  30784  0.00e+00  1.2235e+01  9.5506e+00  5.8452e+01  2
```

### Headline

At **16928 B** the exact (mixed rank+bit) allocation and the uniform `(8, 8)` allocation are at
**exactly equal measured bytes** (16928 B), and the mixed allocation reaches a measured hidden-state
damage of **10.48** versus uniform's **28.57** — a **2.73x reduction at equal bytes** (ratio 0.367).
The mixed map is eleven layers at `(16, 4)`, `blocks.1.proj` at `(8, 8)` and `head` at `(4, 4)`.

A second equal-memory point exists at **9496 B** (uniform `(4, 8)` = 9496 B, mixed = 9448 B, 0.51 %
apart and inside the gate's integer tolerance): hidden damage **31.91 vs 51.93** (ratio 0.615).
Greedy also edges uniform at 16928 B (ratio 0.977) and 9496 B (ratio 0.963), but by a much smaller
margin.

**The logits axis is a published caveat, not a win.** The proxy's unit is the *final hidden state*
(the head is downstream of the measurement point), so the allocator gives `head` its cheapest option
`(4, 4)` to free bytes for the block layers. Hidden-state damage improves sharply while **logits
damage worsens** at the mixed points (e.g. 58.67 vs 41.48 at 16928 B). The allocator optimises the
hidden-state target; it does not optimise logits. Both axes are reported so the trade is visible.

### Greedy is not a safe approximation here

Greedy is *worse* than uniform at 9728 B (hidden 51.51 vs 29.05, ratio 1.77) and at 17392 B (37.83 vs
10.25, ratio 3.69), matching the Milestone-2 finding that its marginal-error-per-byte heuristic is
myopic. Only the exact solver is used for the headline.

## 4. Byte reconciliation (class 3 vs class 1)

For **all 18 allocated configurations** the factors were actually truncated, fake-quantized and
**serialized** through `spectraquant.quantization.accounting.measure_serialized_bytes`, and the
measured container size compared with the allocator's class-1 `accounted_bytes`:

* `max_relative_difference = 0.0` over 18 reconciliations (`all_within_tolerance = true`);
* `total_container_overhead_bytes = 0`; the measured size equals the accounted size for every arm.

This is not a coincidence: the SpectraQuant container is **headerless** and stores exactly the
accounted terms (32-bit-word-padded payload + per-block fp16 scales + zero-points), so
`measured == accounted` by construction and the "container overhead" is **exactly zero bytes**. The
per-artifact container overhead the preregistration warns about (section 5: 221–363 B for an ONNX
invocation) belongs to *third-party* containers; none is used here, and the reconciliation would
surface such overhead in `container_overhead_bytes` rather than hide it. The preregistered **0.5 %**
byte-parity tolerance (`preregistration.md` section 5) is therefore met with margin, and
`assert_byte_parity` is a hard assertion in the experiment (a violation fails the run).

## 5. Equal-memory enforcement

Every quality comparison in the frontier is gated through
`spectraquant.reporting.comparability.assert_equal_memory` on the two arms' **class-3 measured
bytes**, using the tolerance the arm manifests declare (`0.5 %`). The gate is applied to the
uniform arm against each mixed arm at the same budget:

```
B=  7960  uniform->ortools  equal    rel=0.0000%  hidden ratio=1.000  pred ratio=1.000
B=  7960  uniform->greedy   equal    rel=0.0000%  hidden ratio=1.000  pred ratio=1.000
B=  9496  uniform->ortools  equal    rel=0.5055%  hidden ratio=0.615  pred ratio=0.625
B=  9496  uniform->greedy   equal    rel=0.0000%  hidden ratio=0.963  pred ratio=0.969
B=  9728  uniform->ortools  REFUSED  rel=0.5757%
B=  9728  uniform->greedy   equal    rel=0.0822%  hidden ratio=1.773  pred ratio=1.528
B= 16928  uniform->ortools  equal    rel=0.0000%  hidden ratio=0.367  pred ratio=0.396
B= 16928  uniform->greedy   equal    rel=0.2363%  hidden ratio=0.977  pred ratio=0.933
B= 17392  uniform->ortools  REFUSED  rel=2.4839%
B= 17392  uniform->greedy   equal    rel=0.0460%  hidden ratio=3.689  pred ratio=3.146
B= 31792  uniform->ortools  REFUSED  rel=3.1706%
B= 31792  uniform->greedy   REFUSED  rel=3.1706%
```

* Three pairs are **refused** because the mixed arm's measured bytes differ from the uniform arm's by
  more than tolerance (0.58 %, 2.48 %, 3.17 %); those arms are not compared for quality. The refusals
  are informative: at 17392 B and 31792 B the exact solver reaches the *same* quality as uniform at
  **fewer** bytes (16960 and 30784), so the uniform points are byte-dominated rather than beaten at
  equal bytes.
* The 9496 B row is accepted at a **0.5055 %** relative difference: `assert_equal_memory` converts the
  declared `0.5 %` into an integer byte budget with `ceil(0.005 * 9496) = 48 B` and the gap is exactly
  48 B. The relative difference is reported verbatim so the rounding is auditable.
* A dedicated **negative test** (`test_equal_memory_gate_accepts_parity_and_rejects_a_gap`) rejects a
  50-byte gap on a 1000-byte pair and then accepts the same pair once the tolerance is raised,
  proving the rejection is caused by the tolerance check and not by an unrelated error path.

## 6. Quality axis: predicted vs measured damage

* **Across all 18 arms**, Spearman rank agreement between the proxy's predicted error and the measured
  hidden-state damage is **+0.971**; per solver it is uniform **+0.943**, greedy **+1.000**, exact
  **+1.000**. The per-arm within-model agreement between the proxy's per-layer error and the measured
  single-layer damage at the allocated option averages **+0.866** over the 18 arms.
* The proxy ordering is therefore consistent with the measured ordering on this fixture, which is the
  premise H4 needs; it is a single-seed, class-2 fixture number and carries no confirmatory weight.
* At the exact-equal-bytes point (16928 B) the proxy predicts a **2.53x** reduction (13.16 vs 33.21)
  and the measurement confirms a **2.73x** reduction (10.48 vs 28.57): the proxy's predicted gain and
  the measured gain agree in direction and magnitude.

## 7. The H4 falsifier, and what this fixture does and does not say

Frozen falsifier (`preregistration.md` section 11.1, H4): *"Proxy-allocated `(r_l, b_l)` does **not**
dominate uniform rank/bit at any predeclared budget beyond CIs, or is matched by the HAWQ-V2-style
bit-only allocator at equal bytes."* The traceability table (section 11.0) states it as *"no
predeclared budget where it dominates beyond CIs, or matched by the bit-only allocator"*.

**The confirmatory H4 cell — the assignment's "Tier-2 H4 cell", listed in `preregistration.md`
section 8 row 7 on the `CLOUD-COLAB` substrate (Tier 1) — is NOT RUN.** It requires ≥5 seeds on the
cloud notebook substrate and a validated `run_manifest.json`; none exists. No Tier-1/2 number is
implied anywhere in this document.

**What the fixture evidence does say.** On this single-seed Tier-0 fixture, the exact mixed
rank+bit allocation beats uniform at **equal measured bytes** at two predeclared ladder points
(2.73x at 16928 B, 1.63x at 9496 B), and the proxy's ranking agrees with the measured damage
(rho +0.97). That is *consistent with* H4's premise, but:

* it is one seed, 13 modules, one option grid, class-2 damage — no confidence interval, no
  beyond-CI claim, and the "predeclared budgets" here are fixture byte totals, not the Tier-1 sweep's;
* the logits-damage trade (section 3) shows the proxy targets the hidden state, not the logits;
* the **HAWQ-V2-style bit-only comparator is NOT RUN** (it is a confirmatory-cell arm), so the
  falsifier's second clause cannot be evaluated here;
* the falsifier's own metric is a Pareto-hypervolume comparison with a paired bootstrap CI over
  seeds (section 7.3), which a one-seed fixture cannot produce.

So the fixture **informs H4's premise only**; it neither supports nor refutes the confirmatory H4
claim. The confirmatory cell remains **not run**.

## 8. Determinism and reproducibility

* The per-point manifests are byte-deterministic: a second run produces **identical** files
  (`diff -rq` clean). `solve_ortools` records `ortools_wall_time_s`, which is stripped from the written
  manifest (it is telemetry, not part of the allocation) by `deterministic_manifest`.
* `frontier.json` is identical across runs apart from `fixture.training.wall_time_s` (the fixture-fit
  wall clock, retained as provenance per the repository's existing fixture artifacts).
* Every manifest replays: `reproduce_allocation` re-solves the reconstructed problem and reproduces the
  recorded per-layer map, accounted bytes and predicted error; the committed manifests are replayed by
  `test_committed_frontier_manifests_replay`.
* Single-threaded (`torch.set_num_threads(1)`), CPU-only, no CUDA import.

## 9. Limits (quote with any use of the tables)

* **Tier-0 fixture only.** One seed, 13 modules, a synthetic next-token task, a tiny model. Not a
  language model and not the Tier-1/2 H4 cell (NOT RUN).
* **Class 2 quality.** Damage is fake-quantization quality, not packed-storage quality and not
  kernel-backed inference; the class-3 number here is a *byte* count, never a quality claim.
* **Hidden-state target.** The proxy unit is the final hidden state; logits damage is reported
  separately and behaves differently (section 3).
* **Greedy is not a safe approximation** on this fixture (section 3).
* **Integer byte tolerance.** The 0.5 % tolerance becomes an integer byte budget via `ceil`, so a
  relative difference marginally above 0.5 % can still pass (section 5); the relative difference is
  always reported.
* **No kernel/latency claim.** Nothing here is class 4 or 5.
