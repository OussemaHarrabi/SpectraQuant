# Phase 0 state report — 2026-10-10 (cloud recovery pass)

Produced before any new GPU launch, per the continuation brief. Every line is checked against the
repository, the registry or the Kaggle API; nothing is recalled from prose.

## 1. Repository

| Item | Value |
|---|---|
| Branch | `research/tier1-closure` |
| HEAD | `8ff7a9e` (32 commits ahead of the reviewed `b0b6daa` → 34 after this pass) |
| Working tree | clean |
| Remote branches | `infra/bootstrap`, `research/tier1-closure` |
| Tags | `method-v0.1` (never moved, points at the earlier local-fixture checkpoint) |
| Latest CI | `ci`, `docs`, `security` green on the previous push; the `8ff7a9e` run is in flight |
| Local gate | ruff check, ruff format --check, pyright 0 errors, **1155 tests collected**; the full suite passed at `deef…`-era commits (1153 passed, 1 skipped); one unreproduced flake remains recorded |

## 2. Kaggle kernels and their state

| Kernel | Last run (UTC) | Status now | Artifacts |
|---|---|---|---|
| `m3-tier1-smollm2-135m-cloud` | 2026-10-10 17:27 | **RUNNING** (seed-0 M3 pilot, 8 frozen arms) | not yet collectable |
| `tier1-smollm2-135m-cloud` | 2026-10-10 20:20 | COMPLETE | **collected this pass** (manifest `failed`, see §4) |
| `repro-lr-qat-loftq-smollm2-135m-cloud` | 2026-10-10 01:35 | COMPLETE | collected earlier, `validated` (trainable-arm diagnostic) |
| `spectraquant-tier1-smollm2-135m-cloud` | 2026-10-09 16:16 | superseded kernel (pre-rename) | — |

Unrelated kernels on the same account (`aegisgraph-*`, `skills-gap-*`) belong to another project and
consume their own quota; they are not part of this ledger.

## 3. Registry (append-only, `artifacts/runs/registry.jsonl`)

| Run id | State history (tail) | What it means |
|---|---|---|
| `tier1_smollm2_135m-cloud` | … → `validated` → `resubmitted` → `collected` → `rejected` → `rejected` → `resubmitted` | the CPU comparator slice was `validated`; the GPU method attempts failed twice (ortools extra, then CPU-only serialization) |
| `repro_lr_qat_loftq_smollm2_135m-cloud` | … → `collected` → `validated` | the trainable-arm diagnostic, twice validated |
| `m3_tier1_smollm2_135m-cloud` | `submitted` → `failed` (artifacts expired) → `resubmitted` | the M3 seed-0 pilot; one attempt lost to output expiry, one running |

## 4. What the collected-but-`failed` method run contains (recoverable evidence)

The run is **not** a valid result — its manifest says `failed` and the collector rejected it — but its
log preserves two real measurements that the next run will re-measure cleanly:

* **The CP-SAT allocation solved on the real model**: `solver=ortools bytes=59719488 <= budget
  59719680` over 210 layers with **mixed effective ranks [2, 4, 8, 32] and bits [2, 3, 4, 8]** — i.e.
  the allocator chose a genuinely mixed configuration and landed 192 bytes under the budget.
* **`proxy_allocated` scored 64.5477 perplexity** at `accounted_bytes = measured_bytes = 59 719 488`
  (class-1 == class-3 exactly), against int4 PTQ's 18.6190 at 59 719 680 bytes and fp16's 14.0181 at
  212 336 640. So the untrained mixed allocation is **worse than uniform int4 at equal memory** — an
  honest, important H4 data point, and the first equal-memory comparison the project has.
* **`spectraquant_regularized` trained its full 300 steps** (loss 3.93 → 3.39, noisy) and then died in
  the container serialization (`serialization runs on CPU only`), which is now fixed.

## 5. Validated artifacts and their provenance

| Artifact | Run | Class | Substrate | Seed | Tokens |
|---|---|---|---|---|---|
| `artifacts/runs/tier1_smollm2_135m-cloud/download/**` | comparator slice | 1–2 | CLOUD-CPU | 0 | 21 758 |
| `artifacts/runs/repro_lr_qat_loftq_smollm2_135m-cloud/download/**` | trainable-arm diagnostic | 1–2 | CLOUD-GPU (T4) | 0 | 21 758 |

Both carry a schema-valid `run_manifest.json` with the pinned model revision
(`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`), the pinned dataset revision
(`647234772b9554e208af6c826f23b99e3cac88c8`), the git commit, the seed, the step count, the sequence
length, the measured/accounted bytes and the teardown marker.

## 6. Missing artifacts

* The M3 seed-0 pilot's bundle (run in flight).
* Any *successful* Tier-1 method bundle (two attempts, both `failed`; the fixes are committed).
* Seeds 1–4 of the M3 pilot; the Tier-2 campaign; any class-4-GPU measurement.

## 7. GPU-hours consumed and remaining

Kaggle does not expose the free-tier GPU quota through its API or CLI, so the following is an
**estimate from session wall-clock**, and it is the ledger's opening balance:

| Session | Approx. GPU wall-clock |
|---|---|
| `repro_lr_qat_loftq_smollm2_135m-cloud` (2 collected attempts + 1 operator-stopped) | ≈ 8.5 h |
| `m3-tier1-smollm2-135m-cloud` (expired attempt ≈ 6 h + the running attempt) | ≈ 6 h + in flight |
| `tier1-smollm2-135m-cloud` (two failed method attempts, ~30 min each) | ≈ 1 h |
| **Estimated consumed** | **≈ 15.5 h** |

If the weekly allowance is 30 h, the **estimated remaining budget is ≈ 14.5 h, less whatever the
running M3 pilot consumes** (it started at 17:27 UTC). The brief's ceiling (30 h total) is therefore
the binding number, and the ledger below is written against a **planning budget of 12 h with a 3 h
untouched reserve** — i.e. at most 9 h of new work — until the running pilot reports.

## 8. Known failures (all recorded, none hidden)

1. `ORToolsNotInstalledError` — the plan install spec omitted the `alloc` extra. Fixed by deriving the
   extras from the declared arm kinds.
2. `serialization runs on CPU only` — the container measurement ran on CUDA tensors. Fixed at the
   measurement boundary.
3. Artifact expiry — a 6-hour M3 attempt completed and its session output was gone when requested
   ~7 h later. Lesson recorded: collect promptly.
4. The operator-stopped C4 run (materialised the dataset instead of streaming). Fixed.
5. One unreproduced full-suite flake (1 failure in 4 runs, name not captured).

## 9. Exact next operation

1. **Wait for and collect the M3 seed-0 pilot** (running); it is the first protocol cell and the only
   thing that can produce an M3 verdict input. Collect within minutes of completion.
2. **Resubmit the Tier-1 method run** with the serialization fix (≈ 30–40 min of GPU), which yields the
   H3 (method) and H4 (allocation) cells at equal memory.
3. **No other GPU work** until 1 and 2 report.

The compute ledger (`docs/results/compute-budget-ledger.md`) is written against exactly these two
operations plus the confirmatory seeds, and is committed with this report.
