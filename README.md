# SpectraQuant

**Budget-Aware Joint Low-Rank and Low-Bit Training for Efficient Transformers**

**Completed bounded study, 12 October:** read the
[coherent research synthesis](docs/report/spectraquant-research-synthesis.md).
The original allocator remains negative. Fixed weight-only clipping improves
2/4 new model/corpus cells and fails its frozen consistency gate, despite a
0.575% geometric mean reduction. Predeclared activation-weighted controls lower
C4 perplexity by **7.39–7.70% on SmolLM2-360M** and **5.75–6.09% on Pythia-410M**
versus matched group32/axis1 RTN on reserved C4 prefixes
at identical checkpoint bytes, but regress on Pythia PG19. All 42 checkpoint
files are checksum-verified. These are conventional controls and calibration
subsets, not a new allocation-method gain or independent training replications.
See [all primary and secondary results](docs/results/phase4-clipping-transfer.md),
[generated notebooks](notebooks/final/README.md) and
[objective diagnostics](reports/tables/clipping-proxy-diagnostics.md).

SpectraQuant studies one question with two halves:

1. When a transformer is *prepared* by low-rank factorization, does that preparation help or hurt
   the quantization that follows it?
2. Can an output-aware sensitivity proxy allocate per-layer ranks and bit widths so that a
   compressed model beats **uniform** compression at **equal memory**?

The repository provides a measured-storage compression/evaluation pipeline,
fixture frontiers, partial literature reproduction and a transparent record
of positive, negative and mixed real-model outcomes.

**11 October continuation:** a separate exploratory Phase2 study now scores the actual
serialized/reloaded `Q(W)+Q(B)Q(A)` representation using marginal calibration NLL in a uniform-INT4
background. It permits rank zero (plain quantization), compares bits-only with residual-inclusive
allocation, and selects on validation before reading test. Completed at `fe39413`, with25 verified
artifacts and21 checkpoint files: validation retained uniform INT4. Alternative test perplexities
were5.5–14.8% worse; [full results](docs/results/phase2-global-damage.md) preserve the negative result.
Phase3 is now complete at `71b2e2b`: validation selected a conventional weights-only clipping
control, which reduced perplexity **28.0056 → 27.4656 (1.93%)** on **49,951 new C4 holdout
tokens**, with exactly **59,719,680 target-payload bytes** and **116,522,698 full-checkpoint
bytes** for both arms. All six checkpoint files were verified. This is one deterministic
comparison on SmolLM2-135M, not a novel allocation-method or low-bit GPU speed result.
The prior allocations improved their own calibration blocks less than predicted and worsened
loss on disjoint training blocks. Bounded allocations were compared only on development;
their held-out efficacy is not established. See the [completed Phase3 report](docs/results/phase3-controlled-followup.md),
[raw numerical evidence](artifacts/sample-results/phase3-controlled-followup/metrics.json),
and [ADR-0024](docs/decisions/ADR-0024-phase3-controlled-followup.md).
See [ADR-0023](docs/decisions/ADR-0023-phase2-global-damage.md) and
[takeover record](docs/coordination/phase2-takeover-2026-10-11.md). A legacy exploratory H2 artifact
also exists, but it uses a different compression operator, one checkpoint, CPU float32 and a dirty
checkout; it does not complete the frozen confirmatory H2 protocol.

> **Original training-study status: local evidence measured; its cloud Tier-1 comparison is negative; original Tier-2 NOT RUN.**
> The CPU harness runs, validates its own outputs and is reproducible, and the compression machinery
> (quantization, factorization, proxies, allocator, regularizers, class-4-CPU measurement) has
> produced **Tier-0 fixture** evidence on this workstation. On the cloud substrate, three runs are
> collected and checksum-validated, and **the original method has a bounded negative result in the tested configuration**:
> on SmolLM2-135M at **equal measured bytes** (59 719 488 vs 59 719 680, a 0.0003 % match) the trained
> allocated arm scores **23.3131** perplexity (regularizer on) / **23.1698** (off) against uniform
> int4 PTQ's **18.6193** — **the method does not beat equal-memory uniform quantization**, and the
> preparation regularizer's own contribution is **negative** (+0.62 %). The frozen M3 protocol ran as a
> resource-constrained Tier-1 subset (amendment A-0018): the LR-QAT *direction* reproduces
> (24.1265 vs 26.9432) at **21.8 %** of the predeclared 25 % magnitude — **partially reproduced, not
> confirmed**. Tier-2, the H2 real-model proxy cell, the multi-seed H3 comparison and the H4 frontier
> remain **NOT RUN**. The honest boundary is the status table below,
> [`docs/results/tier1-method-comparison.md`](docs/results/tier1-method-comparison.md),
> [`docs/results/m3-verdict.md`](docs/results/m3-verdict.md) and
> [`docs/results/experiment-index.md`](docs/results/experiment-index.md).

The publication package lives in [`reports/paper/`](reports/paper/paper.md) (the research report),
[`reports/tables/`](reports/tables/) and [`reports/figures/`](reports/figures/) (generated from the
committed artifacts), [`docs/cards/`](docs/cards/) (model/dataset cards), [`CAREER_EVIDENCE.md`](CAREER_EVIDENCE.md)
(claim ledger) and [`reports/REPRODUCIBILITY.md`](reports/REPRODUCIBILITY.md).

---

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11 (uv installs it if missing). CPU only; no
CUDA, no GPU, no cloud.

```bash
uv sync --all-extras                        # create/refresh the environment (CPU PyTorch + extras)
uv run spectraquant --help                  # CLI surface
uv run spectraquant env                     # resolved device, hardware, software versions (JSON)
uv run spectraquant smoke --config configs/experiment/smoke.yaml   # tiny end-to-end experiment
uv run spectraquant validate-manifest artifacts/sample-results/smoke-manifest.json
uv run spectraquant compare-manifests \
  artifacts/sample-results/comparability/int4-arm-a.manifest.json \
  artifacts/sample-results/comparability/int4-arm-b.manifest.json   # equal-memory gate (AGENTS.md §4.5)
uv run pytest -q                            # full CPU test suite
make check                                  # ruff + pyright + pytest (equivalent to CI)
```

A smoke run trains a small character-level transformer for 50 steps on a deterministic synthetic
sequence task, prints the loss sequence, and writes a schema-valid run manifest to
`artifacts/sample-results/smoke-manifest.json`. Two runs at the same seed produce identical loss
values bit-for-bit; the run takes seconds.

This smoke run is the sanctioned local **CI smoke fixture** (AGENTS.md §2.3, Tier 0). Research
training does **not** happen on this workstation: Tiers 1–5 execute on the cloud notebook substrate
(AGENTS.md §2b) via the `cloud` adapter, which generates thin notebooks from the frozen plans.

---

## How to reproduce the local evidence

The exact commands (clean checkout → tests → artifact regeneration → table/figure regeneration),
the expected test count, and the stable digests are in
[`reports/REPRODUCIBILITY.md`](reports/REPRODUCIBILITY.md). The short form:

```bash
uv sync --all-extras
uv run pytest -q            # 1155 collected; 1153 passed / 1 skipped at 8ff7a9e (the skip is the
                            # PEFT cross-check, whose API the frozen gate cannot use - see A-0013).
                            # Counts are per commit: earlier snapshots report 908 (M10 bootstrap,
                            # clean checkout) and 1,034 (Phase 0 baseline) because the suite grew.
uv run python scripts/experiments/proxy_validation_sweep.py        # artifacts/sample-results/proxy-validation/
uv run python scripts/experiments/allocation_frontier.py           # artifacts/sample-results/allocation-frontier/
uv run python scripts/experiments/regularizer_sweep.py             # artifacts/sample-results/regularizer/
uv run python scripts/experiments/class4cpu_measurement.py         # artifacts/sample-results/class4cpu/ (onnx extra)
uv run python scripts/reproduce/generate_release_artifacts.py      # reports/tables/ + reports/figures/
```

The published tables and the Pareto figure are generated **only** by
`scripts/reproduce/generate_release_artifacts.py` from the committed artifacts;
`tests/unit/test_release_artifacts.py` fails if they drift.

---

## Status — what is measured, what is NOT RUN

This table is deliberately unflattering. It is the honest boundary of the repository as committed.

| Area | Delivered now | Status / Not done |
|---|---|---|
| Environment & packaging | `pyproject.toml`, committed `uv.lock`, CPU torch pin, ruff/pyright/pytest config | — |
| CLI | `env`, `smoke`, `run-plan`, `validate-manifest`, `compare-manifests`, `cloud` | — |
| Config | Hydra composition + Pydantic validation for `model`/`data`/`method`/`experiment`; frozen cloud plans under `configs/{tier1,tier2,repro}/` | — |
| Data | deterministic synthetic LCG corpus (Tier 0 fixture) | WikiText-2 / C4 loaders are cloud-only; contamination audit requires a cloud run |
| Model | tiny fixture transformers (`configs/model/tiny.yaml`; `TinyConfig`) | SmolLM2-135M @ `93efa2f0…` **is loaded, trained and evaluated** on the cloud GPU substrate (validated runs plus the M3 pilot and the method arm's 300 training steps); TinyLlama-1.1B is pinned but not yet loaded |
| Quantization | fake quantize/dequantize, generic 2/3/4/8-bit packing, accounting, serialization/reload, ONNX export | Packed storage does not imply a supported low-bit kernel; H5 kernel chain remains $b\in\{4,8\}$ |
| Factorization | truncated/randomized SVD, spectral summaries, factor bytes | — |
| Proxies | 10 variants incl. the declared candidate `gain_aware_composed` and the predeclared comparators; gain estimator | confirmatory H2 cloud cell **NOT RUN** |
| Allocation | uniform / greedy / CP-SAT exact + proxy adapter; deterministic manifests | confirmatory H4 cloud cell **NOT RUN** |
| Regularizers | rounding-aware spectral preparation objective + training loop + Tier-0 sweep | confirmatory H3 cloud cell **NOT RUN**; local quality signal is **mixed (negative)** |
| Training | seeding, smoke loop, factorized linear layers, preparation loop with checkpoint/resume + manifests | Tier-1+ training is cloud-only (AGENTS.md §2b) |
| Evaluation | Tier-0 toy fixtures + exact float64 ground truth; harness API surface | LM Evaluation Harness integration and downstream metrics await a cloud run |
| Benchmarking | analytical byte accounting + **class-4-CPU** kernel runner (ONNX Runtime `MatMulNBits`/`MatMulInteger`) on self-serialized containers | class 4-GPU and class 5 are unavailable locally (cloud-only); no latency/throughput claim |
| Reporting | run manifests + JSON Schema, git provenance, environment capture, equal-memory gate, **artifact-sourced tables/figures** | — |
| Cloud execution | RunSpec, generated thin notebooks, Kaggle/Colab adapters, registry, collection, budget guard | **executed end to end** on Kaggle: three runs `validated` (a CPU comparator slice, the T4 Tier-1 equal-memory comparison incl. the regularizer on/off pair, and the M3 Tier-1 subset), plus one unclaimable bundle that failed validation at export (A-0018). Tier-2 and the frozen protocol at full scale are NOT RUN |

### Measured (local, with the substrate label)

| Result | Substrate / class | Artifact |
|---|---|---|
| Proxy fixture: naive proxy fails under LayerNorm (ρ = −0.060), gain-aware ρ = +0.830 | `LOCAL-FIXTURE` / 2 | `artifacts/sample-results/proxy-fixture/proxy-fixture.json` |
| Proxy validation (M4 local half): candidate beats every predeclared comparator in 15/15 aggregate Fisher-z contrasts | `LOCAL-FIXTURE` / 2 | `artifacts/sample-results/proxy-validation/proxy-validation.json` |
| Allocator frontier (M6): mixed rank+bit cuts measured hidden-state damage **2.73×** vs uniform at identical measured bytes (16 928 B); 18/18 byte reconciliations at 0.0 relative difference | `LOCAL-FIXTURE` / 1–3 | `artifacts/sample-results/allocation-frontier/frontier.json` |
| Regularizer sweep (M5): rounding-grid term drives the measured rounding residual −93.98 %, but dev NLL does **not** improve — H3 **not supported locally** | `LOCAL-FIXTURE` / 1–2 | `artifacts/sample-results/regularizer/sweep.json` |
| Class-4-CPU: our own int4/int8 ONNX containers execute through real CPU kernels (int4 relative max error 0.0771 at (4, 64)) | `LOCAL-FIXTURE` / 4-CPU, 3 | `artifacts/sample-results/class4cpu/class4cpu.json` |

### NOT RUN — never to be reported as measured

Tier-2; the H2 real-model proxy-ranking cell; the multi-seed H3 comparison; the H4 frontier; the
**frozen M3
reproduction protocol** — its eight specified arms (2-bit NF-codebook block-64 LoftQ at ranks 16/64;
4-bit g128 LR-QAT with rank 32 and a learned step size; `R2-RTN-4bit-g128`; `R2-fullQAT-4bit-g128`;
the validation-split learning-rate search) are **not fully executed and validated**. The
resource-constrained subset is reported separately in `docs/results/m3-verdict.md`, without claiming
the full protocol; the Tier-2 TinyLlama campaign (`CLOUD-GPU`); class 4-GPU
kernel inference and class 5 service latency/throughput. They become results only once a validated
`run_manifest.json` and checksum-validated artifacts exist (`AGENTS.md` §2b).

---

## Measurement classes (label every number)

From `AGENTS.md` section 5; these labels are mandatory in every manifest and every report.

| # | Class | Meaning | Locally producible |
|---|---|---|---|
| 1 | Analytical estimate | derived from shapes and bit widths only | yes |
| 2 | Fake-quantization quality | float execution simulating quantization numerics | yes |
| 3 | Packed storage | serialized low-bit weights, bytes actually measured | yes (for formats we serialize) |
| 4-CPU | Kernel-backed inference (CPU) | a real CPU low-bit kernel (ONNX Runtime `MatMulNBits`/`MatMulInteger`, torchao intx) executing an artifact **we serialized**, against an fp32 baseline measured in the same session | **yes** — available and measured (`artifacts/sample-results/class4cpu/`) |
| 4-GPU | Kernel-backed inference (GPU) | CUDA-only kernels (bitsandbytes, GPTQ/AWQ/Marlin, TorchAO CUDA) | **no** — cloud substrate only; NOT RUN |
| 5 | End-to-end service | request-level latency/throughput | **no** — cloud substrate only; NOT RUN |

Classes must never be mixed inside a single claim. A float execution is never "low-bit storage";
fake quantization is never "accelerated inference"; a third-party-format kernel (llama.cpp GGUF,
someone else's ONNX) is engineering telemetry and never class 4 for our artifacts. Class 4-CPU is
measured only on containers SpectraQuant serializes itself, and its numbers are **not** comparable to
published GPU latency/throughput figures.

---

## Repository map

```
src/spectraquant/
  cli/            Typer CLI: env, smoke, run-plan, validate-manifest, compare-manifests, cloud
  config.py       typed Hydra/Pydantic experiment config
  data/           deterministic synthetic corpora (Tier 0/1)
  factorization/  truncated/randomized SVD, spectral summaries
  quantization/   fake-quantize / pack / dequantize / accounting / ONNX export
  proxies/        10 sensitivity proxy variants incl. the declared candidate
  regularizers/   rounding-aware spectral preparation term
  allocation/     uniform / greedy / CP-SAT allocator + proxy adapter
  training/       seeding, smoke loop, factorized linear layers, preparation loop
  evaluation/     Tier-0 toy fixture + exact float64 ground truth
  benchmarking/   memory accounting + class-4-CPU kernel runner (4-GPU/5 cloud-only)
  cloud/          cloud notebook adapter (Colab/Kaggle) + run-plan
  reporting/      logging, git provenance, environment capture, run manifests, equal-memory gate
configs/{model,data,method,experiment,tier1,tier2,repro}/   Hydra composition + frozen plans
artifacts/{schemas,sample-results}/       JSON Schema + committed sample results
reports/{paper,tables,figures}/           generated publication package
docs/cards/                               model & dataset cards
scripts/{data,reproduce,experiments,cloud,benchmark}/
tests/{unit,integration,regression}/      CPU, deterministic
docker/                                   CPU-only smoke image
```

Research code and infrastructure are separated on purpose: `src/spectraquant/*` produces numbers and
manifests, `docs/research/*` fixes the claims *before* the numbers exist, and
`artifacts/sample-results/*` records what was actually executed. See `docs/architecture/system.md`.

---

## Reproducibility contract

* Every run seeds Python, NumPy and PyTorch from one integer and enables
  `torch.use_deterministic_algorithms(True)`; CPU threads are pinned for bit-reproducible reductions.
* Every run writes a manifest validated against `artifacts/schemas/run-manifest.schema.json`,
  recording the git commit, a dirty flag, the resolved config, dataset checksums, hardware, software
  versions and the measurement class.
* Determinism is enforced by tests (`tests/unit/test_seeding.py`, `tests/integration/test_smoke.py`),
  and the published tables/figures are drift-checked against their artifacts
  (`tests/unit/test_release_artifacts.py`).

## License

Apache-2.0 — see [LICENSE](LICENSE). Third-party model, dataset and repository terms are tracked in
`docs/research/upstream-lockfile.md` and `docs/research/model-dataset-licenses.md`.
