# Coordination status

Last updated: 2026-10-08 by the orchestrator. Update this file at every gate; it is the single
source of truth for "what is done, what is claimed, what is blocked".

## 1. Repository state

| Item | Value |
|---|---|
| Remote | `https://github.com/OussemaHarrabi/SpectraQuant.git` (PUBLIC, created 2026-10-08T19:55:34Z) |
| Default branch | `main` (created with the first commit on this branch set; initially empty) |
| Active branch | `infra/bootstrap` |
| Branch strategy | one branch per stream: `research/`, `feat/`, `infra/`, `docs/`, `fix/`, `eval/`, `perf/` |
| Local path | `C:\Users\oussa\oussema\SpectraQuant` |
| Git identity | `Oussema Harrabi <189178358+OussemaHarrabi@users.noreply.github.com>` (local repo config only) |

## 2. Compute

CPU-only Windows workstation; **no CUDA**, AMD iGPU unusable, 16.23 GB RAM. Full record:
`docs/research/environment.md`. Consequence: Tiers 0–1 executable locally; Tiers 2–5 are
**planned/GPU-gated** and may not be reported as completed. Measurement classes 4–5 unavailable.

Compute consumed so far: negligible (bootstrap + documentation only). No GPU spend, no cloud cost.

## 3. Completed work

| # | Work | Evidence | Status |
|---|---|---|---|
| 1 | Non-destructive bootstrap: verified empty local dir + empty remote, `git init`, remote, `infra/bootstrap`, identity, line-ending policy | `git remote -v`, `gh repo view`, `git config --local --list` | done |
| 2 | Hardware/software audit | `docs/research/environment.md` (command transcript embedded) | done |
| 3 | `AGENTS.md` operating contract (compute rules, scientific boundaries, measurement taxonomy, commands, git protocol, agent report contract) | `AGENTS.md` | done |
| 4 | Path ownership map for all specialist streams | `docs/coordination/ownership.md` | done |
| 5 | Python 3.11.16 installed via uv | `uv python list` | done |
| 6 | Scientific foundation: charter, literature review + 50-row matrix, novelty/prior-art risk, preregistration with traceability, risk register, upstream lockfile with 8 verified SHAs + licenses, BibTeX (50 verified entries), method-card template | `docs/research/**`, `references/**`; SHAs re-verified independently by the orchestrator (LR-QAT `8795afe0…`, LoftQ `ae33fd4f…`, lq-lora `c2424b3a…`, pytorch/ao `cff77b46…`, lm-eval `ddd67220…` peeled from tag `v0.4.13`); CSV parses 50×19, 0 duplicate keys | delivered, pending M1 gate |
| 7 | Backend capability + measurement protocols: taxonomy operations, 23-row live compatibility matrix, benchmark protocol, byte-accounting rules with worked examples | `docs/research/backend-capability.md`, `docs/protocols/{measurement-taxonomy,benchmark-protocol,memory-accounting}.md` | delivered (revision in flight) |
| 8 | Model/dataset licence + revision inventory and frozen evaluation protocol (5-role split separation, 6-task harness suite, pinned harness commit, cost per local cell) | `docs/research/model-dataset-licenses.md`, `docs/protocols/eval-protocol.md`; revisions/licences re-verified independently (TinyLlama-Chat `fe8a4ea1…`, Qwen2.5-0.5B-Instruct `7ae55760…`, SmolLM2-360M-Instruct `a10cc151…`, deit-base `a0fc9b37…`, all apache-2.0; SlimPajama-6B licence genuinely absent → `[UNRESOLVED]`) | delivered |
| 10 | **Milestone 2 systems core** (quantization, factorization, cloud adapter, comparability gate) | 715-test suite green; orchestrator's independent probes: `accounted_bytes == measured == len(blob) == file_size` with zero residual across 16 bit/granularity/group configurations; no-CUDA scan over 51 files clean; cloud `collect` validates an intact bundle (3 artifacts), rejects a tampered artifact on sha256 and rejects a commit mismatch; `compare-manifests` accepts an equal-byte pair and rejects an oversized pair | done |
| 11 | **Proxy slice** (base contract, 8 variants, gain estimator, toy fixtures + exact float64 ground truth) | `tests/unit/test_proxy_*.py`, `tests/unit/test_toy_ground_truth.py`; artifact `artifacts/sample-results/proxy-fixture/proxy-fixture.json`; report `docs/results/proxy-fixture-report.md`. Measured: naive proxy ρ = −0.060 under LayerNorm vs gain-aware ρ = **+0.830**; joint/Σproxy 0.069 (naive) vs 0.85 (gain-aware); joint/Σsingle = 1.375 | done (M4 gate extends it) |
| 12 | **Allocator slice** (uniform/greedy/exhaustive/CP-SAT, deterministic manifests) | `tests/unit/test_allocation_*.py`; report `docs/results/allocator-report.md`. Independent verification: **CP-SAT equals the exhaustive oracle exactly** at three budgets (Δ < 1e-9); uniform +7.5–62.8 % and greedy +20.6–94.1 % worse than the oracle; manifest replay reproduces the identical assignment; infeasibility reports the minimum budget | done |
| 13 | Adversarial review of the M1 foundation + 17 blocking issues closed | `docs/results/verification/m1-novelty-review.md`; novelty re-verdicts, H2 unit redesign with a predeclared MDE, byte-parity/serializer freeze, equal-memory mechanism, cloud-substrate amendment A-0004 and correction amendment A-0005 | done |
| 14 | Cloud execution adapter (RunSpec, generated thin notebooks, Kaggle/Colab/Colab-Enterprise adapters, registry, collection, budget guard, secret redaction) + run book | `src/spectraquant/cloud/**`, `scripts/cloud/README.md`, `notebooks/generated/`; 173 slice tests | done |
| 15 | **M1 freeze**: strict cloud-plan schema + three committed plans (Tier-1 SmolLM2-135M, Tier-2 TinyLlama-1.1B, M3 reproduction), preregistration §13 checklist completed and the document marked FROZEN, amendment A-0006 | `configs/{tier1,tier2,repro}/*.yaml`, `src/spectraquant/experiment_plan.py`, `tests/unit/test_experiment_plan.py` (15 tests); freeze commit `6281c56`; 731-test suite green, pyright 0 errors | done |
| 16 | Predeclared H2 comparator set implemented (`weight_magnitude`, `activation_magnitude`, `hessian_diag`, `weight_frobenius`) and the fixture re-measured | `artifacts/sample-results/proxy-fixture/proxy-fixture.json`; on the LayerNorm fixture the candidate reaches ρ = +0.830 against +0.412 / +0.357 / +0.121 / −0.044 for the four comparators and −0.060 for the naive baseline | done (M4 confirms across seeds) |
| 17 | **M4 local-fixture half**: multi-seed, multi-cell proxy validation with the model as the statistical unit | `docs/results/proxy-validation-report.md`, `artifacts/sample-results/proxy-validation/proxy-validation.json`; 10 trained models, 12 cells, 5 seeds; the candidate `gain_aware_composed` beats every predeclared comparator in **15/15 aggregate contrasts** (Fisher-z CI excluding 0); achieved MDE(z) 0.476-0.715 inside the predeclared 0.33-0.79 band; the untuned `combined` variant is *worse* than two comparators; cloud Tier-1 cells NOT RUN | done (local half) |
| 18 | **Class-4-CPU fixture**: our own int4/int8 ONNX containers executed by a real CPU kernel under the benchmark-protocol section 8 rules | `src/spectraquant/quantization/onnx_export.py`, `src/spectraquant/benchmarking/kernel_cpu.py`, `artifacts/sample-results/class4cpu/class4cpu.json`; int4 1662 B (638 B graph + 1024 B sidecar), int8 2727 B, int4 relative error 0.0771 with the weight distribution stated; the orchestrator re-ran the script and reproduced every figure | done |
| 19 | Candidate proxy declared and the freeze checklist closed | amendment **A-0007** (candidate = `gain_aware_composed`; `combined` is an ablation with untuned weights); section 13 class-4-CPU item flipped to satisfied; the M2 report's stale fixture description corrected | done |
| 20 | **M6 allocator integration**: real proxy as `error_fn`, measured-byte validation, first frontier | `docs/results/allocation-integration-report.md`, `artifacts/sample-results/allocation-frontier/frontier.json`; 18 arms, byte parity 18/18 within the 0.5% tolerance (0.0% relative, headerless container), 9 equal-memory pairs gated with 3 refused; **mixed allocation cuts hidden-state damage 2.73x versus uniform at identical measured bytes (10.48 vs 28.57 at 16 928 B)**, 4 equal-memory wins, predicted-vs-measured rank agreement +0.971; caveat published: the unit is the final hidden state, so logits damage worsens at mixed points | done (fixture) |
| 21 | **M5 rounding-aware spectral preparation**: objective, training loop, arms, exploratory sweep | `docs/results/regularizer-report.md`, `artifacts/sample-results/regularizer/sweep.json`; the predeclared exact ratio term has a degenerate STE gradient, replaced by a smooth surrogate (A-0008); sweep reproduced independently: lambda_round drives the rounding residual down -90.6% and improves dense-fidelity -36%, but dev NLL worsens (+638% full arm) — **H3 not supported on the fixture**, reported mixed/inconclusive | done (fixture); gate not met locally |
| 22 | Gap found: the notebook generator could not consume the frozen plans and no remote `run-plan` entry point existed | `spectraquant cloud notebook --config configs/repro/...` fails on a `PlanConfig`; delegated as the PlanRunner slice | in flight |

## 4. Streams

In flight: **M8** (`Ablations`, component/calibration/outlier/seed-count ablations), **M10** (`ReleasePackage`, paper source, generated tables+figures, cards, career ledger, reproducibility file) and **L** (`ClaimAudit`, independent claim-to-evidence audit from a clean checkout). Completed streams: **A** literature/preregistration (incl. the adversarial
review's blocking issues), **B** infra/scaffold (incl. the `cloud`/`alloc`/`onnx` extras and the
comparability gate), **C** reproduction planning, **D** quantization core, **D-lite** backend
capability + measurement protocols, **E** factorization/spectral, **F** proxies, **H** allocator,
**I-lite** licences + evaluation protocol, **K** cloud adapter, **L** adversarial verification.

Next streams (not started, in dependency order):

| Stream | Scope | Gate |
|---|---|---|
| M1 freeze | complete the preregistration freeze checklist (Tier-1 configs, seed list, byte-budget ladder, comparator implementations, `run_manifest.json` schema, notebook generator, cost ceiling) | M1 |
| M3 reproduction | bounded LR-QAT / LoftQ reproduction **on the cloud substrate** via the cloud adapter (Colab is the developer's vehicle) | M3 |
| M4 proxy validation | multi-seed, multi-configuration proxy validation with the model as the statistical unit and the predeclared MDE | M4 |
| M5/M6 | rounding-aware preparation + allocator integration with the real proxy as `error_fn` | M5, M6 |

## 5. Gates

| Gate | Requirement | Status |
|---|---|---|
| M0 | clean checkout passes lint, type checks, unit tests, tiny experiment | **PASSED.** Fresh clone of the pushed branch (`6ed13b16`) into a temp directory: `uv sync --all-extras` from scratch, `ruff check` clean, `ruff format --check` 53 files, `pyright` 0 errors, `pytest -q` **94 passed**, `spectraquant smoke` reproduced the identical loss-sequence digest `sha256:1b2e0622…` (matches the working-tree runs and the scaffold's own run), `validate-manifest` OK. |
| M1 | literature matrix + novelty audit + frozen preregistration + pinned upstreams; **gate** = independent review confirms the candidate extension is differentiated (or revises the question) | **PASSED, with narrowed scope.** `preregistration.md` is **FROZEN 2026-10-09 at commit `6281c56`** (amendment A-0006). The independent adversarial review (`docs/results/verification/m1-novelty-review.md`) returned verdicts **A: differentiated (narrow, empirical)**, **B: differentiated (weak–moderate, empirical)**, **C: previously-known as a concept** — i.e. differentiated only through A — and raised 17 blocking issues, all closed (novelty corrections incl. the two uncited neighbours SVDQuant/JoLT, H2 unit redesign with a predeclared MDE, byte-parity/serializer freeze, equal-memory mechanism, cloud budget line). Claim wording is constrained to those verdicts. |
| M2 | math fixtures pass; allocator matches exhaustive search on tiny cases | **PASSED for the implemented slices.** Exact float64 toy agreement for the unit-defining variants (`rtol=1e-9`); `accounted_bytes == measured` with zero residual across 16 configurations; CP-SAT == exhaustive oracle at three budgets; no-CUDA invariant test; equal-memory gate exercised by a negative test. The *proxy-quality* gate is M4 (proxy must beat the predeclared comparator) and is **not** yet decided. |
| M4 | proposed proxy must beat the simplest baseline proxy on the predeclared criteria | **LOCAL-FIXTURE HALF PASSED; CLOUD HALF NOT RUN.** On the Tier-0 fixture (5 seeds x 6 (rank, bits) cells x 2 fixtures) the declared candidate `gain_aware_composed` has a positive Fisher-z advantage over every predeclared comparator in 15/15 aggregate contrasts, with the achieved MDE inside the predeclared band, while the naive per-layer baseline loses its ranking ability under LayerNorm. The confirmatory H2/H4 cells are `CLOUD-COLAB` and remain **NOT RUN** until a validated Colab bundle exists. |
| M5 | method beats its ablations on development outcomes within the memory budget | **NOT MET on the fixture (honest negative).** The regularizer reduces the rounding residual and improves fidelity to the dense reference, but worsens dev NLL; the two quality axes disagree in sign, so H3 is unsupported locally (A-0008). The confirmatory H3 cell is `CLOUD-COLAB` and is NOT RUN. |
| M6 | every allocation respects real accounting and is reproducible from a manifest | **PASSED on the fixture.** 18/18 arms reconcile class-3 measured bytes with class-1 accounting inside the 0.5% tolerance; manifests replay deterministically; the equal-memory gate refuses byte-unequal pairs. Tier-2 allocation remains NOT RUN. |

## 6. Blockers and limits

- **No GPU.** Tiers 2–5 blocked until an external GPU host is documented (driver/CUDA version,
  supported quantization formats). No cloud GPU budget has been agreed yet.
- **Measurement classes 4–5 unavailable.** Any kernel-backed latency or service-level claim is
  deferred; nothing will be estimated to fill the gap.
- Upstream LR-QAT / LoftQ scripts rely on CUDA-only stacks; a CPU-executable bounded reproduction
  path must be designed explicitly (Milestone 3) or reported as infeasible with a reason.

## 7. Next actions

1. **M3 reproduction (needs the developer to run it).** The cloud adapter is ready and the reproduction
   plan is frozen, but no reproduction number exists until a Colab run is executed and collected:
   ```
   uv run spectraquant cloud notebook --config configs/repro/lr_qat_smollm2_135m.yaml --platform colab
   # run the generated notebook in Colab, export its artifacts, then locally:
   uv run spectraquant cloud collect --run-id <id> --source <downloaded dir>
   ```
   Free tier only; no paid resource may be started without explicit authorization (AGENTS.md section 2b rule 8).
2. **M5/M6** are in flight locally (regularizer, allocator integration) on the Tier-0 fixture.
3. Once M3 collects, run the Tier-1 pilot plan, then the confirmatory M4/M7 cells.
4. Keep pushing coherent commits; every result manifest records its measurement class and substrate.

## 8. Recorded deviation: single worktree during bootstrap

The specification calls for one git worktree and branch per independently editable stream. During
waves 1–2 all streams instead share **one working tree on `infra/bootstrap`** with strict path
ownership (`docs/coordination/ownership.md`), because subagents on a single Windows workstation share
one `uv` environment and one checkout; separate worktrees would multiply a 200 MB+ torch install and
make the CPU smoke gate non-comparable.

Consequences: (a) two agents must never touch the same file — enforced by the ownership map; (b)
per-stream branches are still used for *integration* of accepted changes (e.g. `research/*`, `feat/*`),
so the public history keeps one branch per reviewed change; (c) the M0 gate is verified on a clean
clone of the pushed branch, not on the working tree. To be revisited if a second machine or GPU host
becomes available.

## 9. Discipline reminders

- One writer per file; check `docs/coordination/ownership.md` before editing.
- Every result carries a measurement class; fake quantization is never low-bit storage.
- Equal-memory comparison required for every compression comparison.
- Amendments to the preregistration are append-only and timestamped.

## 10. Incident log

| When | What | Impact | Resolution |
|---|---|---|---|
| 2026-10-09 | The model provider returned HTTP 429 (`GoUsageLimitError`, ~1 h retry-after) while two subagent slices were running. `ProxyCore` had written only `proxies/base.py`; `Allocator` had finished code and tests but not its report. | Two slices stalled mid-flight; no work was lost and no result was fabricated. | The orchestrator completed both slices directly (variants, gain estimator, toy fixtures, tests, measurement artifact, both slice reports), removed the dead agent's scratch file `_bench_toy.py`, and re-ran the full verification. Delegation was unavailable, not abandoned. |
| 2026-10-09 | A sibling slice added an `onnx` extra to `pyproject.toml` while the infra stream owned that file. | None: the infra stream kept the change, refreshed the lock over it and documented it. | Recorded here because two writers touched one file; the ownership rule still holds. |
