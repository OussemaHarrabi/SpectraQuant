# Result-registry validation (Milestone-7 gate)

Owner: registry-validator slice. Tool: `spectraquant registry-validate`
(`src/spectraquant/reporting/run_registry.py`). Tests: `tests/unit/test_run_registry.py`.
Machine-readable index for the committed manifests: `artifacts/sample-results/registry-validation.json`
(regenerate with the command in §2). Every figure below is a **count of files, records or pairs**,
not a scientific measurement: the AGENTS.md §5 measurement taxonomy (classes 1–5) labels model
quality / memory / latency numbers, and no such number appears in this report.

Reproduce everything in this file with:

```bash
uv run spectraquant registry-validate --runs artifacts/sample-results --out artifacts/sample-results/registry-validation.json
uv run spectraquant registry-validate --runs outputs/dry      # local plan dry-run bundle (uncommitted)
uv run spectraquant registry-validate --runs outputs           # local smoke gate runs (uncommitted)
uv run pytest -q tests/unit/test_run_registry.py
```

(`COLUMNS=200` was exported for the captures below so the tables are not folded by the terminal.)

## 1. What the tool is

The gate turns a directory of run manifests into a **comparability cell matrix**. It ingests every
manifest through `reporting/manifests.py` (`*.manifest.json` — the committed sample layout and
`smoke-manifest.json` — and `run_manifest.json`, the cloud-bundle layout of
`design-cloud-adapter.md` §5 and the `run-plan` layout `<out>/<arm>/seed-<n>/run_manifest.json`),
indexes each run by the comparability key the frozen protocol declares (`eval-protocol.md` §8), and
reports per cell: expected vs present seeds and the missing ones, duplicate `(cell, seed)` runs,
interrupted/aborted runs, schema-invalid or unparseable files, cells that differ in a frozen
comparability field (incomparable — never poolable), and the equal-memory verdict of every arm pair,
obtained from `reporting/comparability.py`.

Three keys are kept distinct, because "the same cell", "the arms we may compare" and "the cells we
may tabulate together" are three different sets:

| key | contents | used for |
|---|---|---|
| **cell** | substrate, hardware generation, runtime backend, model id + revision, dataset ids/revisions/checksums, split, arm, compression settings | the seed matrix: expected vs present vs missing seeds, duplicates |
| **comparison group** | cell minus `(arm, compression)` | which arms are gated against each other at equal memory |
| **family** | cell minus the axes (substrate, hardware, runtime backend, model revision, dataset revisions) | incomparability: two cells of one family that differ in an axis must not be pooled |

Two design choices are worth naming, because they decide what the tool can see:

* **The compression *settings* are in the cell key; the byte *outcomes* are not.** `accounted_bytes`
  and `measured_bytes` are results, not configuration: two runs of one arm at one grid point that
  report different stored bytes are the *same* cell with conflicting measurements (reported as a
  duplicate), while whether either may be compared with another arm is exactly what the equal-memory
  gate decides.
* **A missing declaration is recorded as missing, never guessed.** A run that records no plan
  substrate is `substrate="undeclared"`; a run that records no expected seed set has
  `expected="?"` and no missing-seed check is invented for it. The declared seed set and floor are
  read from what `spectraquant run-plan` records (`metrics["plan.reported_seeds"]`,
  `metrics["plan.seed_floor"]`), and a cell whose runs disagree about them is reported
  (`conflicting_seed_declaration`) rather than resolved.

**Exit code.** Non-zero (1) exactly when a *blocking* problem exists: a missing seed for a declared
cell, a duplicate run, a schema-invalid/unparseable manifest, or an unequal-memory arm pair — the
last meaning the gate **resolved both byte figures, applied the predeclared tolerance and found the
gap wider than it**. A pair the gate refuses *earlier* (no declared `bytes_source`, no declared
`measured_bytes_tolerance`, or two different byte sources/serializers) is reported verbatim as
`memory_gate_refused` at **warning** severity: such a pair is not a declared equal-memory comparison
at all, which is a finding but not evidence about memory. Interrupted runs, incomparable cells,
orphan `metrics.json` files, seed-declaration conflicts and an absent runs directory are warnings.

**Fixtures.** A manifest that declares itself one (`resolved_config.fixture.not_a_real_run == true`)
is a *fixture*, not a run: it is never a cell, never contributes a seed, a duplicate or an
incomparability, and never fails the gate. Its arm pairs are still gated, so the mechanism is
exercised on committed data, but every finding about it is reported at warning severity
(`fixture_only=true`) — a fixture is never a result (`preregistration.md` §8). A directory whose
manifests are all fixtures is printed as `fixture-only … not a missing cell`.

## 2. Run over the committed manifests — `artifacts/sample-results/**`

```bash
$ uv run spectraquant registry-validate --runs artifacts/sample-results --out artifacts/sample-results/registry-validation.json
registry: artifacts/sample-results
1 cell(s), 1 run(s), 5 fixture(s): 0 blocking, 3 warning(s)
┌──────────────────────────────────────────┬────────────────────────┬───────────┬─────────┬──────┬───────┬──────────┐
│ cell                                     │ substrate/hardware     │ seeds p/e │ missing │ runs │ dupes │ problems │
├──────────────────────────────────────────┼────────────────────────┼───────────┼─────────┼──────┼───────┼──────────┤
│ train/<no-arm>/<local>@sha256:fcf32/none │ undeclared/cpu(no-gpu) │ 1/?       │ -       │ 1    │ 0     │ -        │
└──────────────────────────────────────────┴────────────────────────┴───────────┴─────────┴──────┴───────┴──────────┘
fixtures (declared non-runs; excluded from the cell matrix, their arm pairs are gated at warning severity):
┌────────────────────────────────────────────────┬────────────────────────────┬──────────────────┐
│ fixture                                        │ run_id                     │ cell key         │
├────────────────────────────────────────────────┼────────────────────────────┼──────────────────┤
│ comparability/accounted-arm-a.manifest.json    │ fixture-accounted-arm-a    │ c4ea4ab88bfee813 │
│ comparability/accounted-arm-b.manifest.json    │ fixture-accounted-arm-b    │ c4ea4ab88bfee813 │
│ comparability/int4-arm-a.manifest.json         │ fixture-int4-arm-a         │ 73ef05bcda2b6f16 │
│ comparability/int4-arm-b.manifest.json         │ fixture-int4-arm-b         │ 73ef05bcda2b6f16 │
│ comparability/int4-arm-oversized.manifest.json │ fixture-int4-arm-oversized │ 73ef05bcda2b6f16 │
└────────────────────────────────────────────────┴────────────────────────────┴──────────────────┘
comparability: fixture-only (5 declared fixture(s), no run) - not a missing cell
┌──────────┬─────────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ severity │ code                │ detail                                                                                                                                                             │
├──────────┼─────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ warning  │ memory_gate_refused │ 6 arm pair(s) cannot be gated at equal memory (AGENTS.md section 4.5): 6x unequal-memory comparison rejected: the two runs use different byte sources [e.g.        │
│          │                     │ fixture-accounted-arm-a vs fixture-int4-arm-a]                                                                                                                     │
│ warning  │ unequal_memory      │ seed 1234: comparability/int4-arm-a.manifest.json vs comparability/int4-arm-oversized.manifest.json: the two runs differ by 10012 bytes, more than the 200-byte    │
│          │                     │ tolerance (bytes_a=1001200, bytes_b=1011212, source='measured')                                                                                                    │
│ warning  │ unequal_memory      │ seed 1234: comparability/int4-arm-b.manifest.json vs comparability/int4-arm-oversized.manifest.json: the two runs differ by 9912 bytes, more than the 200-byte     │
│          │                     │ tolerance (bytes_a=1001300, bytes_b=1011212, source='measured')                                                                                                    │
└──────────┴─────────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
json: artifacts\sample-results\registry-validation.json
$ echo $?
0
```

**What it found.** Six manifests: the committed smoke manifest and the five hand-authored
equal-memory fixtures.

1. **One cell, one run — the smoke manifest.** `train/<no-arm>/<local>@sha256:fcf32/none` on
   `undeclared/cpu(no-gpu)`: a real executed Tier-0 run whose model and dataset are local
   (`model_id=null`, `synthetic-lcg-v1`), method `none`, seed 1234. It declares **no** expected seed
   set and **no** seed floor (no plan recorded it), so the seeds cell reads `1/?` and no
   missing-seed finding is invented. This is the CI/Tier-0 smoke artifact, not a research cell: it
   carries `measurement_class: null` and `method: "none"`.
2. **Five fixtures, zero cells — `comparability/`.** The directory is printed as *fixture-only* and
   explicitly *not* a missing cell. Its five manifests are declared non-runs
   (`resolved_config.fixture.not_a_real_run = true`), so they form no cell, contribute no seed and
   cannot produce a blocking finding. They are, however, gated, which is the point of exercising the
   tool on committed data — 10 pairs:
   * **2 accepted** equal-memory verdicts recorded in the JSON index
     (`artifacts/sample-results/registry-validation.json` → `equal_memory`):
     `fixture-accounted-arm-a` = `fixture-accounted-arm-b` (900 000 B vs 900 000 B, tolerance 0 B,
     **class 1** analytical), and `fixture-int4-arm-a` = `fixture-int4-arm-b`
     (1 001 200 B vs 1 001 300 B, `|Δ| = 100 B ≤ 200 B`, **class 3** measured).
   * **6 refusals** collapsed into one warning: the accounted arms declare `bytes_source="accounted"`
     and the int4 arms `bytes_source="measured"`, and the gate refuses to mix a class-1 figure with a
     class-3 one — reported verbatim, not silently pooled.
   * **2 rejections** (`unequal_memory`): the pair against `int4-arm-oversized` exceeds the declared
     200-byte tolerance by 10 012 B and 9 912 B. This is the fixture doing its job — it is the
     *negative* comparison the equal-memory mechanism must refuse — and because it is a declared
     fixture the finding is at warning severity, so the committed registry still exits **0**.
3. **Nothing else:** no duplicates, no interrupted runs, no schema-invalid files, no incomparable
   cells, no orphan `metrics.json`.

## 3. Local, uncommitted bundles

Two scratch trees exist on the workstation (`outputs/` is git-ignored). They are reported here as
evidence that the gate catches what it will have to catch when the cloud bundles land; neither is
committed and neither is a research cell.

### 3.1 `outputs/dry` — a `run-plan` dry run (three arms, one seed)

```bash
$ uv run spectraquant registry-validate --runs outputs/dry
registry: outputs/dry
3 cell(s), 3 run(s), 0 fixture(s): 3 blocking, 1 warning(s)
┌──────────────────────────────────────────────────────────┬────────────────────┬───────────┬─────────┬──────┬───────┬──────────┐
│ cell                                                     │ substrate/hardware │ seeds p/e │ missing │ runs │ dupes │ problems │
├──────────────────────────────────────────────────────────┼────────────────────┼───────────┼─────────┼──────┼───────┼──────────┤
│ test/rank_then_quant/outputs\tiny-model@local-dir:sh/rtn │ colab/cpu(no-gpu)  │ 1/5       │ 1,2,3,4 │ 1    │ 0     │ 1        │
│ test/fp16_reference/outputs\tiny-model@local-dir:sh/none │ colab/cpu(no-gpu)  │ 1/5       │ 1,2,3,4 │ 1    │ 0     │ 1        │
│ test/ptq_uniform_4/outputs\tiny-model@local-dir:sh/rtn   │ colab/cpu(no-gpu)  │ 1/5       │ 1,2,3,4 │ 1    │ 0     │ 1        │
└──────────────────────────────────────────────────────────┴────────────────────┴───────────┴─────────┴──────┴───────┴──────────┘
┌──────────┬─────────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ severity │ code                │ detail                                                                                                                                                             │
├──────────┼─────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ blocking │ missing_seeds       │ test/rank_then_quant/outputs\tiny-model@local-dir:sh/rtn: declared seeds [0, 1, 2, 3, 4] but only [0] have a successful run; missing [1, 2, 3, 4]                  │
│ blocking │ missing_seeds       │ test/fp16_reference/outputs\tiny-model@local-dir:sh/none: declared seeds [0, 1, 2, 3, 4] but only [0] have a successful run; missing [1, 2, 3, 4]                  │
│ blocking │ missing_seeds       │ test/ptq_uniform_4/outputs\tiny-model@local-dir:sh/rtn: declared seeds [0, 1, 2, 3, 4] but only [0] have a successful run; missing [1, 2, 3, 4]                    │
│ warning  │ memory_gate_refused │ 3 arm pair(s) cannot be gated at equal memory (AGENTS.md section 4.5): 1x unequal-memory comparison rejected: no tolerance: pass an explicit tolerance or declare  │
│          │                     │ compression.measured_bytes_tolerance in the manifests [e.g. tier1_smollm2_135m-ptq_uniform_4-seed0-20261009T053221Z-9b77df39730a-plan vs                           │
│          │                     │ tier1_smollm2_135m-rank_then_quant-seed0-20261009T053221Z-9b77df39730a-plan]; 2x unequal-memory comparison rejected: the two runs use different byte sources [e.g. │
│          │                     │ tier1_smollm2_135m-fp16_reference-seed0-20261009T053220Z-9b77df39730a-plan vs tier1_smollm2_135m-ptq_uniform_4-seed0-20261009T053221Z-9b77df39730a-plan]           │
└──────────┴─────────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
$ echo $?
1
```

Read correctly, these are three real findings about a bundle that is **not** a plan measurement
(each manifest carries `substitution.is_plan_measurement = false`: a local stand-in model and corpus
were substituted):

* **3 blocking `missing_seeds`.** Every arm declares the plan's seed set `[0, 1, 2, 3, 4]` and its
  1-of-5 seed count. `run-plan` defaults to the first reported seed, so a default dry run is — by
  construction — an incomplete cell. The gate says so instead of letting one seed stand in for five.
* **3 arm pairs that cannot be gated**, each refused for a structural reason the tool prints
  verbatim: `fp16_reference` offers a **class-1 `accounted`** figure while the quantized arms offer
  **class-3 `measured`** ones (two pairs), and no manifest records the preregistration's
  `measured_bytes_tolerance` at all (one pair, `ptq_uniform_4` vs `rank_then_quant`). A warning, not
  a failure: these arms are different Pareto points, not a declared equal-memory comparison; the
  finding is that the manifests cannot yet be gated.
* The three arms are **not** reported incomparable — they are three arms of one family, which is
  exactly what a family is allowed to contain.

### 3.2 `outputs` — the two smoke gate replications plus `outputs/dry`

```bash
$ uv run spectraquant registry-validate --runs outputs
registry: outputs
4 cell(s), 5 run(s), 0 fixture(s): 5 blocking, 2 warning(s)
┌──────────────────────────────────────────────────────────┬────────────────────────┬───────────┬─────────┬──────┬───────┬──────────┐
│ cell                                                     │ substrate/hardware     │ seeds p/e │ missing │ runs │ dupes │ problems │
├──────────────────────────────────────────────────────────┼────────────────────────┼───────────┼─────────┼──────┼───────┼──────────┤
│ test/rank_then_quant/outputs\tiny-model@local-dir:sh/rtn │ colab/cpu(no-gpu)      │ 1/5       │ 1,2,3,4 │ 1    │ 0     │ 1        │
│ test/fp16_reference/outputs\tiny-model@local-dir:sh/none │ colab/cpu(no-gpu)      │ 1/5       │ 1,2,3,4 │ 1    │ 0     │ 1        │
│ train/<no-arm>/<local>@sha256:fcf32/none                 │ undeclared/cpu(no-gpu) │ 1/?       │ -       │ 2    │ 1     │ 1        │
│ test/ptq_uniform_4/outputs\tiny-model@local-dir:sh/rtn   │ colab/cpu(no-gpu)      │ 1/5       │ 1,2,3,4 │ 1    │ 0     │ 1        │
└──────────────────────────────────────────────────────────┴────────────────────────┴───────────┴─────────┴──────┴───────┴──────────┘
┌──────────┬─────────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ severity │ code                │ detail                                                                                                                                                             │
├──────────┼─────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ blocking │ duplicate_run       │ train/<no-arm>/<local>@sha256:fcf32/none seed 1234: 2 successful runs of the same cell and seed (smoke-20261008T204703Z-f77fc9a5199d-10d3c7,                       │
│          │                     │ smoke-20261008T204722Z-f77fc9a5199d-85214a); their metric payloads are byte-identical (a replication)                                                              │
│ blocking │ missing_seeds       │ test/rank_then_quant/outputs\tiny-model@local-dir:sh/rtn: declared seeds [0, 1, 2, 3, 4] but only [0] have a successful run; missing [1, 2, 3, 4]                  │
│ blocking │ missing_seeds       │ test/fp16_reference/outputs\tiny-model@local-dir:sh/none: declared seeds [0, 1, 2, 3, 4] but only [0] have a successful run; missing [1, 2, 3, 4]                  │
│ blocking │ missing_seeds       │ test/ptq_uniform_4/outputs\tiny-model@local-dir:sh/rtn: declared seeds [0, 1, 2, 3, 4] but only [0] have a successful run; missing [1, 2, 3, 4]                    │
│ blocking │ schema_invalid      │ alloc-manifest.json: <root>: Additional properties are not allowed ('accounted_bytes', 'allowed_bits', 'allowed_ranks', 'budget_bytes', 'cost_model',              │
│          │                     │ 'cost_model_version', 'diagnostics', 'error_table', 'feasible', 'git', 'layer_shapes', 'overhead_table', 'per_layer', 'predicted_error', 'schema', 'solver',       │
│          │                     │ 'solver_params' were unexpected); <root>: 'schema_version' is a required property; <root>: 'run_id' is a required property (+25 more)                              │
│ warning  │ memory_gate_refused │ 3 arm pair(s) cannot be gated at equal memory (AGENTS.md section 4.5): 1x unequal-memory comparison rejected: no tolerance: pass an explicit tolerance or declare  │
│          │                     │ compression.measured_bytes_tolerance in the manifests [e.g. tier1_smollm2_135m-ptq_uniform_4-seed0-20261009T053221Z-9b77df39730a-plan vs                           │
│          │                     │ tier1_smollm2_135m-rank_then_quant-seed0-20261009T053221Z-9b77df39730a-plan]; 2x unequal-memory comparison rejected: the two runs use different byte sources [e.g. │
│          │                     │ tier1_smollm2_135m-fp16_reference-seed0-20261009T053220Z-9b77df39730a-plan vs tier1_smollm2_135m-ptq_uniform_4-seed0-20261009T053221Z-9b77df39730a-plan]           │
│ warning  │ orphan_metrics      │ dry: metrics.json present without a run manifest. An interrupted or truncated bundle, or the invocation-level aggregate of a plan run (design-cloud-adapter.md     │
│          │                     │ section 11.3) - check before treating it as a run.                                                                                                                 │
└──────────┴─────────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
$ echo $?
1
```

Two of these deserve a note:

* **The duplicate is a determinism-gate replication, and it is still a duplicate.** `outputs/gate-a`
  and `outputs/gate-b` each hold one successful smoke run of the *same cell at the same seed*, with
  byte-identical metric payloads (the manifest records `loss_sequence_sha256`). The tool reports it as
  a blocking `duplicate_run` and *says which kind it is* — "byte-identical (a replication)" rather
  than "metrics DIFFER". The registry contract has one run per `(cell, seed)`; a deliberate
  replication belongs in a gate log, not in the results directory, so the correct resolution is to
  keep one and record the other's purpose, not to silence the check. (The replication is itself
  useful evidence: the smoke run is deterministic.)
* **`alloc-manifest.json` is reported schema-invalid, correctly.** The gate treats every
  `*manifest.json` as a run-manifest candidate, because a runs directory must contain run manifests.
  This file is an allocation manifest written by another slice into the scratch tree, so the run
  schema rejects it (17 unexpected properties, 3 missing ones). Pointed at a real runs directory —
  where the only manifests are run manifests — the finding does not arise. The same rule is what
  picks up the committed `smoke-manifest.json`.

## 4. What the tool will report once the cloud bundles exist

When `spectraquant cloud fetch --dest artifacts/runs/<run_id>/` and `run-plan` bundles land, the
default invocation (`spectraquant registry-validate`, `--runs` defaulting to
`$SPECTRAQUANT_RUNS_DIR` or `<repo>/artifacts/runs`) will, per `(arm × grid point × substrate ×
hardware generation × model/dataset revision)` cell, print:

* the **seed matrix** — expected seeds from `metrics["plan.reported_seeds"]`/`plan.seed_floor`, the
  seeds that actually produced a **successful** run, and the missing ones (blocking);
* **duplicates**: more than one successful record of one cell at one seed, flagged as a replication
  when the metric payloads are byte-identical and as a conflicting measurement when they are not;
* **interrupted/aborted** runs (`status` `failed`/`aborted`, with the recorded failure reason) and
  orphan `metrics.json` files left by a truncated bundle;
* **schema-invalid or unparseable** manifests, one finding each, without stopping the pass;
* **incomparable** cells: cells of one family that differ in substrate, hardware, runtime backend,
  model revision or dataset revisions/checksums, reported per axis with the instruction that they
  must not be pooled, averaged or differenced;
* the **equal-memory verdict** of every matched-seed arm pair in every comparison group, with the
  accepted verdicts (class 1 or class 3, byte figures, tolerance) in the JSON index and every
  refusal/rejection printed verbatim;
* **exit 1** if any of the four blocking classes fired, **exit 0** for warnings only.

**No cloud cell is currently present.** The committed registry holds exactly one run cell — the
Tier-0 smoke manifest, a local-CPU run that records no substrate label (the tool prints
`undeclared`) — and no cell from Colab, Kaggle or any GPU substrate; the three cloud tiers remain
**NOT RUN**, not "in progress". The gate is therefore exercised end-to-end on local data only, and
its first real input will be the first validated bundle (`eval-protocol.md` §6.5).

## 5. Tests

`tests/unit/test_run_registry.py` — every test writes real manifests to a temporary runs directory
and drives the real pass, so the behaviour under test is the behaviour the gate has:

```
tests/unit/test_run_registry.py::test_cell_key_is_stable_and_order_independent
tests/unit/test_run_registry.py::test_cell_key_separates_every_frozen_field
tests/unit/test_run_registry.py::test_hardware_change_alone_changes_the_cell_key
tests/unit/test_run_registry.py::test_missing_seed_is_detected
tests/unit/test_run_registry.py::test_complete_seed_matrix_is_not_reported
tests/unit/test_run_registry.py::test_zero_seed_is_a_real_seed
tests/unit/test_run_registry.py::test_aborted_seed_does_not_count_as_present
tests/unit/test_run_registry.py::test_interrupted_run_alone_is_only_a_warning
tests/unit/test_run_registry.py::test_conflicting_seed_declaration_is_reported
tests/unit/test_run_registry.py::test_duplicate_cell_seed_is_detected
tests/unit/test_run_registry.py::test_duplicate_with_conflicting_metrics_is_marked_as_a_conflict
tests/unit/test_run_registry.py::test_failed_run_beside_a_success_is_not_a_duplicate
tests/unit/test_run_registry.py::test_schema_invalid_file_is_reported_without_crashing_the_pass
tests/unit/test_run_registry.py::test_unparseable_file_is_reported_without_crashing_the_pass
tests/unit/test_run_registry.py::test_orphan_metrics_file_is_reported
tests/unit/test_run_registry.py::test_missing_runs_dir_is_a_warning
tests/unit/test_run_registry.py::test_cells_differing_only_in_substrate_are_incomparable
tests/unit/test_run_registry.py::test_cells_differing_only_in_hardware_are_incomparable
tests/unit/test_run_registry.py::test_different_arms_of_one_plan_are_not_incomparable
tests/unit/test_run_registry.py::test_unequal_memory_arm_pair_is_flagged
tests/unit/test_run_registry.py::test_pair_within_tolerance_is_recorded_as_an_equal_memory_verdict
tests/unit/test_run_registry.py::test_pair_the_gate_cannot_gate_is_a_warning_not_a_failure
tests/unit/test_run_registry.py::test_runs_of_one_cell_are_not_gated_against_each_other
tests/unit/test_run_registry.py::test_fixture_is_not_a_cell_and_never_blocks
tests/unit/test_run_registry.py::test_fixture_only_directory_is_reported_as_such
tests/unit/test_run_registry.py::test_fixture_gate_is_per_comparison_group
tests/unit/test_run_registry.py::test_json_index_is_deterministic
tests/unit/test_run_registry.py::test_report_paths_are_relative_to_the_runs_dir
tests/unit/test_run_registry.py::test_cli_exits_non_zero_on_a_blocking_problem
tests/unit/test_run_registry.py::test_cli_exits_zero_for_warnings_only
tests/unit/test_run_registry.py::test_cli_exits_zero_on_the_committed_fixtures
tests/unit/test_run_registry.py::test_cli_json_output_and_out_file
tests/unit/test_run_registry.py::test_cli_json_exit_code_follows_the_blocking_findings
```

```bash
$ uv run pytest -q tests/unit/test_run_registry.py
.................................                                        [100%]
33 passed in 2.17s

$ uv run pytest -q
966 passed, 1 warning in 64.18s (0:01:04)
```

The negative case that matters is covered twice over: `test_unequal_memory_arm_pair_is_flagged`
proves a pair outside the tolerance is blocking, and its companion
`test_pair_within_tolerance_is_recorded_as_an_equal_memory_verdict` proves the *same* pair inside the
tolerance is recorded as equal and does not fail — so the finding cannot come from an unrelated
error path. `test_cli_exits_zero_on_the_committed_fixtures` pins the fixture rule against the real
committed fixtures, and `test_cli_exits_non_zero_on_a_blocking_problem` /
`test_cli_exits_zero_for_warnings_only` pin the exit-code contract in both directions.

## 6. Assumptions, limits and unresolved questions

* **Assumptions.** (1) The comparability key is read from recorded manifest fields only; the
  hardware fingerprint is `platform/system/release/machine/processor/gpu_devices/torch_device/
  torch_threads` and excludes `ram_total_mb` and `cpu_count_logical`, which describe the host
  instance rather than the hardware generation. (2) The "substrate" axis is the plan substrate the
  manifest records (`metrics["plan.substrate"]`, e.g. `colab`), not a derived
  `CLOUD-COLAB`/`CLOUD-GPU` label: the manifest schema carries no substrate field, so deriving a
  protocol label would be an inference the tool refuses to make. (3) Device lists and compression
  exclusions are treated as sets (sorted), so enumeration order does not change a cell key; dataset
  ids/revisions keep their recorded order because they are parallel arrays. (4) A file whose name
  ends in `manifest.json` is a run-manifest candidate.
* **Limits.** The gate cannot see inside a bundle: a `run_manifest.json` that validates is trusted
  for the fields it declares, and checksum validation of the payloads remains `cloud collect`'s job
  (`design-cloud-adapter.md` §5). It does not verify the frozen protocol values of
  `eval-protocol.md` §8 one by one (harness commit, few-shot counts, dtype, batch size, `--limit`):
  those are recorded in `metrics["harness.*"]`/`resolved_config` by the runner and are not yet
  cross-checked against the frozen constants. It does not merge `cloud/registry.py`'s JSONL
  transitions — a run that was submitted but never collected is invisible to it, which is the
  intended division of labour.
* **Unresolved.** (a) The plan runner records `compression.measured_bytes_tolerance: null`, so
  plan arms cannot be gated without an explicit tolerance; whether the plan runner should record the
  preregistration's 0.5 % parity tolerance is a decision for its owning slice, not this one — the
  gate reports the gap rather than filling it. (b) The `fp16_reference` arm declares a class-1
  `accounted` figure, so it can never be gated against a class-3 arm; whether the reference arm
  should measure a serialized fp16 container (class 3) is the same open question. (c) A non-run
  manifest in a runs directory (e.g. the allocation slice's `alloc-manifest.json` in the scratch
  `outputs/` tree) is reported schema-invalid; if that becomes normal, the candidate rule needs an
  explicit allow-list rather than silence.
