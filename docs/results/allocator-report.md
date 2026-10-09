# Allocator report (Milestone 2, slice H)

Owner: allocator slice. Evidence: `src/spectraquant/allocation/**`, `tests/unit/test_allocation_*.py`,
and the machine-readable artifact `artifacts/sample-results/allocator/oracle-comparison.json`
(regenerate with `uv run python scripts/experiments/allocator_oracle_comparison.py`: per-budget solver
objectives, relative excess versus the exhaustive oracle, oracle-vs-CP-SAT agreement, and the
metadata-overhead examples). Every figure below comes from that artifact.

## 1. What was measured

Problem: four layers with shapes `a=(64,64)`, `b=(128,64)`, `c=(64,128)`, `d=(256,128)`; allowed ranks
`{2,4,8,16,32}`; allowed bit widths `{4,8}`; per-group symmetric cost model with `group_size=32`
(every byte figure comes from `spectraquant.quantization.accounting`, never re-derived); analytic
per-layer error model `error = layer_scale/rank + 0.01*bits`. Minimum feasible budget: **3 504 B**.

| Budget (B) | min feasible | uniform | greedy | exhaustive (oracle) | OR-Tools CP-SAT |
|---|---|---|---|---|---|
| 4 000 | 3 504 | 1.7850 (**+7.5 %**) / 3 936 B / 0.8 ms | 3.2225 (**+94.1 %**) / 3 936 B / 1.4 ms | 1.6600 / 3 936 B / 5.5 ms | **1.6600 / 3 936 B / 9.2 ms** |
| 8 000 | 3 504 | 0.9725 (**+62.8 %**) / 4 800 B / 0.6 ms | 0.8631 (**+44.5 %**) / 7 840 B / 2.4 ms | 0.5975 / 7 744 B / 8.8 ms | **0.5975 / 7 744 B / 2.7 ms** |
| 16 000 | 3 504 | 0.5663 (**+49.5 %**) / 8 576 B / 0.6 ms | 0.4569 (**+20.6 %**) / 12 928 B / 2.4 ms | 0.3788 / 14 464 B / 12.8 ms | **0.3788 / 14 464 B / 2.6 ms** |

Percentages are relative to the oracle objective (lower error is better).

**Findings**

1. **OR-Tools CP-SAT reproduces the exhaustive oracle exactly** at all three budgets
   (`|Δobjective| < 1e-9`), on instances small enough to enumerate — this is the Milestone-2 gate
   requirement, and it is met. CP-SAT is also 3-5× faster than enumeration at these sizes.
2. **Uniform allocation is 7.5–62.8 % worse than the oracle** and **greedy is 20.6–94.1 % worse**.
   Greedy is *not* a safe approximation: at the tightest budget it is **13× worse than uniform**
   (+94.1 % vs +7.5 %) because its marginal-error-per-byte heuristic is myopic — it spends the budget
   on the first large marginal gain it finds and cannot revisit the choice.
3. **Greedy leaves budget unused** (3 936 B of 8 000 B, 12 928 B of 16 000 B), so its loss is not
   merely a scheduling artefact: the constraint is slack and it still cannot find the better point.
4. Every solver respects the byte budget in all three configurations (`accounted_bytes ≤ budget`).

## 2. Metadata overhead is not negligible

Cost of one layer `(out, in) = (128, 64)` under the per-group symmetric model, `group_size=32`
(figures from `make_accounting_cost_fn`, i.e. the same function the constraint uses):

| rank | bits | accounted bytes | nominal payload bytes | overhead | overhead share |
|---|---|---|---|---|---|
| 8 | 4 | 1 056 | 768 | 288 | **37.5 %** |
| 32 | 4 | 3 456 | 3 072 | 384 | 12.5 % |
| 64 | 4 | 6 912 | 6 144 | 768 | 12.5 % |
| 8 | 8 | 1 824 | 1 536 | 288 | **18.8 %** |

Per-block scales alone cost 12.5–37.5 % of the payload, and the share is largest exactly where the
allocator is most likely to look for savings (small ranks). This is why the constrained quantity and
the reported quantity must be the same function (`design-m2-interfaces.md` §7.6) — an allocator that
constrains nominal bits while the report quotes measured bytes would silently overspend.

## 3. Reproducibility

* An allocation is emitted as a deterministic JSON manifest carrying the git SHA, the cost-model
  descriptor, the solver and its parameters, the byte total and the per-layer map.
* **Replaying the manifest reproduces the identical assignment** (verified: same per-layer map, same
  9 216 B total in the first probe; the replay path rebuilds the problem from the manifest's evaluated
  tables rather than trusting the recorded solution).
* Infeasible budgets raise `InfeasibleBudget` carrying the minimum feasible budget (verified: minimum
  1 824 B reported for a 3-layer instance whose budget was 10 B).

## 4. What this slice does NOT do

* The `error_fn` used here is **analytic**. Wiring the real proxy variants into `error_fn` is the
  Milestone-6 integration; until then no allocation claim may be made about real compression quality.
* Byte figures are **class 1** (analytical accounting). Class-3 measured serialization of a full
  allocation has not been run yet; the reconciliation identity exists
  (`accounted_bytes == measure_serialized_bytes`) and is tested in the quantization slice, but the
  end-to-end check on an allocated model is Milestone 6 work.
* The measured latency figures in the table are solver wall-clock on this workstation, not a
  deployment property, and are not comparable to any other machine.
* Solver runs are single-threaded CPU; no timeout behaviour under a binding `time_limit_s` is
  characterised beyond the 10 s limit used here (the CP-SAT optimum was found in ≤ 9.2 ms).
