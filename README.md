# SpectraQuant

**Budget-Aware Joint Low-Rank and Low-Bit Training for Efficient Transformers**

SpectraQuant studies one question with two halves:

1. When a transformer is *prepared* by low-rank factorization, does that preparation help or hurt
   the quantization that follows it?
2. Can an output-aware sensitivity proxy allocate per-layer ranks and bit widths so that a
   compressed model beats **uniform** compression at **equal memory**?

The deliverable is a quality–memory Pareto frontier plus the sensitivity proxy that produces it —
not a chatbot, not an agent framework, not a RAG pipeline, and not "LoRA fine-tuning" dressed up as
compression.

> **Status: pre-alpha research scaffold.** The harness runs, validates its own outputs and is
> reproducible on CPU. The science has not started yet: no compression method, proxy, allocator or
> Tier-2 model exists in this repository (see the status table below).

---

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11 (uv installs it if missing). CPU only; no
CUDA, no GPU, no cloud.

```bash
uv sync --all-extras                        # create/refresh the environment (CPU PyTorch)
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

A smoke run trains a ~0.1 M-parameter character-level transformer for 50 steps on a deterministic
synthetic sequence task, prints the loss sequence, and writes a schema-valid run manifest to
`artifacts/sample-results/smoke-manifest.json`. Two runs at the same seed produce identical loss
values bit-for-bit; the run takes seconds, not minutes.

This smoke run is the sanctioned local **CI smoke fixture** (AGENTS.md §2.3, Tier 0). Research
training does **not** happen on this workstation: Tiers 1–5 execute on the cloud notebook substrate
(AGENTS.md §2b), and the adapter that drives it (`src/spectraquant/cloud/**`) is a later slice that
does not exist yet.

---

## What is *not* done yet

This table is deliberately unflattering. It is the honest boundary of the repository as committed.

| Area | Delivered now | Not done (owner / gate) |
|---|---|---|
| Environment & packaging | `pyproject.toml`, committed `uv.lock`, CPU torch pin, ruff/pyright/pytest config | — |
| CLI | `env`, `smoke`, `validate-manifest`, `compare-manifests` | `run`, `report`, `benchmark` subcommands |
| Config | Hydra composition + Pydantic validation for `model`/`data`/`method`/`experiment` | method/sweep configs beyond `none` |
| Data | deterministic synthetic LCG sequence corpus (Tier 0 fixture) | WikiText-2 pipeline, tokenizers, contamination audit (Milestone 2) |
| Model | one hand-written tiny char transformer (<= 200 k params, CI smoke fixture) | TinyLlama-1.1B and any pretrained checkpoint (Tier 2, cloud-gated per AGENTS.md §2b) |
| Quantization | package + typed API surface only (`NotImplementedError`); the `onnx` extra pins the class 4-CPU kernel path (`onnx`, `onnxruntime`, `onnx-ir`, `torchao`), imported lazily | int8/int4 fake quant, group-size edge cases, packing (Milestone 2) |
| Factorization | package + typed API surface only (`NotImplementedError`) | truncated SVD / LoftQ-style init (Milestone 2) |
| Proxies | package + typed API surface only (`NotImplementedError`) | output-aware sensitivity proxy, ranking validation (Milestone 3) |
| Allocation | package + typed API surface only (`NotImplementedError`); the `alloc` extra pins the constrained optimizer (`ortools>=9.11`) the allocator will import lazily | layer-wise rank/bit allocator, budget feasibility (Milestone 4) |
| Training | deterministic seeding, tiny smoke loop, factorized linear layers, the preparation loop with checkpoint/resume + per-term regularizer diagnostics + schema-valid manifests (Milestone 5, Tier 0) | Tier-1+ training and QAT/LoRA fine-tuning: cloud substrate only (AGENTS.md §2b), and `evaluate_language_model` awaits the Milestone-6 corpus loaders |
| Evaluation | package + typed API surface only | LM Evaluation Harness integration, downstream metrics (Milestone 6) |
| Benchmarking | package + typed API surface only | latency/throughput harness — 4-GPU and class 5 are unavailable locally (cloud substrate only); 4-CPU is planned (self-serialized int4/int8 container executed by a real CPU kernel, same-session fp32 baseline) and must never be phrased as latency or as GPU-comparable |
| Reporting | run manifests + JSON Schema, logging, git provenance, environment capture, sample result, **equal-memory comparability gate** (`assert_equal_memory` + `spectraquant compare-manifests`, AGENTS.md §4.5) | result registry, figure/table generation |
| Tracking | `track` extra pins `mlflow-skinny` (ADR-0003); no run logs to it yet | wiring runs to MLflow (Milestone 5) |
| Research docs | charter, literature review, preregistration, ADRs, risk register (other agents' paths) | preregistration frozen, method selection |
| Compute tiers | Tier 0 locally only: fixtures, unit/property tests, config validation, the CI smoke fixture, analysis | **No research training runs locally.** Tiers 1–5 execute on the cloud notebook substrate (AGENTS.md §2b); none of them has run yet |
| Cloud execution | `cloud` extra declared (`nbformat>=5.10`, `kaggle>=1.7`) so `uv.lock` is stable; contract frozen in `docs/coordination/design-cloud-adapter.md` | the adapter itself (`src/spectraquant/cloud/**` — not created here), `spectraquant cloud …` subcommands, registry, budget guard: all owned by a wave-2 slice |

Class 4-GPU (CUDA kernels) and class 5 (end-to-end service) are **not available** here and are never
estimated — on the cloud substrate they become claimable only under AGENTS.md §2b's manifest and
cost rules. Class 4-CPU is locally available in principle (§5) but nothing in this scaffold produces
it yet, so no class 4 number is claimed anywhere in this repository.

---

## Measurement classes (label every number)

From `AGENTS.md` section 5; these labels are mandatory in every manifest and every report.

| # | Class | Meaning | Locally producible |
|---|---|---|---|
| 1 | Analytical estimate | derived from shapes and bit widths only | yes |
| 2 | Fake-quantization quality | float execution simulating quantization numerics | yes |
| 3 | Packed storage | serialized low-bit weights, bytes actually measured | yes (only for formats we can really serialize) |
| 4-CPU | Kernel-backed inference (CPU) | a real CPU low-bit kernel (ONNX Runtime `MatMulNBits`/`MatMulInteger`, torchao intx) executing an artifact **we serialized**, against an fp32 baseline measured in the same session | available in principle (AGENTS.md §5) — **no such path exists in this scaffold yet** |
| 4-GPU | Kernel-backed inference (GPU) | CUDA-only kernels (bitsandbytes, GPTQ/AWQ/Marlin, TorchAO CUDA) | **no** — cloud substrate only |
| 5 | End-to-end service | request-level latency/throughput | **no** — cloud substrate only |

Classes must never be mixed inside a single claim. A float execution is never "low-bit storage";
fake quantization is never "accelerated inference"; a third-party-format kernel (llama.cpp GGUF,
someone else's ONNX) is engineering telemetry and never class 4 for our artifacts. The committed
sample manifest carries `measurement_class: null` because it reports a *training* diagnostic of an
*uncompressed* toy model — the validator rejects `null` as soon as the run declares a compression
method.

---

## Repository map

```
src/spectraquant/
  cli/            Typer CLI: env, smoke, validate-manifest
  config.py       typed Hydra/Pydantic experiment config
  data/           deterministic synthetic corpora (Tier 0/1)
  factorization/  low-rank decomposition API            (Milestone 2, NotImplementedError)
  quantization/   fake-quantize / pack / dequantize API  (Milestone 2, NotImplementedError)
  proxies/        output-aware sensitivity proxies       (Milestone 3, NotImplementedError)
  regularizers/   rounding-aware spectral preparation   (Milestone 5, implemented:
                  objective + ablation terms + Tier-0 coefficient sweep)
  allocation/     rank & bit-width allocator             (Milestone 4, NotImplementedError)
  training/       seeding, tiny smoke loop, training loop API
  evaluation/     evaluation harness API                 (Milestone 6, NotImplementedError)
  benchmarking/   memory + latency harness API           (kernel paths gated; 4-CPU planned, 4-GPU/5 cloud-only)
  cloud/          cloud notebook adapter (Colab/Kaggle)   (wave 2 — not present in this scaffold)
  reporting/      logging, git provenance, environment capture, run manifests,
                  equal-memory comparability gate
configs/{model,data,method,experiment}/   Hydra composition groups
artifacts/{schemas,sample-results}/       JSON Schema + committed sample manifest
docker/                                   CPU-only smoke image
docs/{architecture,decisions,...}/        system design + ADRs
scripts/reproduce/environment_probe.py    hardware/software audit (stdlib + optional torch)
tests/{unit,integration,regression}/      CPU, deterministic
```

Research code and infrastructure are separated on purpose: `src/spectraquant/*` produces numbers and
manifests, `docs/research/*` fixes the claims *before* the numbers exist, and
`artifacts/manifests/*` records what was actually executed. See `docs/architecture/system.md`.

---

## Reproducibility contract

* Every run seeds Python, NumPy and PyTorch from one integer and enables
  `torch.use_deterministic_algorithms(True)`; CPU threads are pinned to 1 so reductions are
  bit-reproducible.
* Every run writes a manifest validated against `artifacts/schemas/run-manifest.schema.json`,
  recording the git commit, a dirty flag, the resolved config, dataset checksums, hardware, software
  versions and the measurement class.
* Determinism is enforced by tests: `tests/unit/test_seeding.py` and
  `tests/integration/test_smoke.py` run the same experiment twice and compare loss sequences.

## License

Apache-2.0 — see [LICENSE](LICENSE). Third-party model, dataset and repository terms are tracked in
`docs/research/upstream-lockfile.md`.
