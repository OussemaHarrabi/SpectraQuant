# AGENTS.md — SpectraQuant operating contract

Authoritative instructions for every human or agent contributing to this repository.
Read this file before editing anything. If a task conflicts with this file, this file wins.

---

## 1. Project identity

**SpectraQuant: Budget-Aware Joint Low-Rank and Low-Bit Training for Efficient Transformers**

Research question (primary): does preparing a transformer for low-rank factorization improve or
damage its subsequent quantization, and can an output-aware sensitivity proxy select layer-wise
ranks and bit widths that produce a better quality–memory Pareto frontier than uniform compression?

Non-goals: this is **not** an agent/RAG/chatbot/orchestration project. It is **not** a robotics
project. It does **not** treat LoRA fine-tuning alone as compression.

Governing specification: the master orchestrator prompt delivered 2026-10-08 (see
`docs/research/charter.md` for the distilled charter derived from it).

---

## 2. Compute reality (binding constraint)

Recorded 2026-10-08 on the development workstation; full detail in `docs/research/environment.md`.

| Resource | Value | Consequence |
|---|---|---|
| CPU | AMD Ryzen AI 7 350, 8 cores / 16 threads | CPU-only execution; all code must run on CPU |
| RAM | 16.2 GB (≈15.1 GiB usable) | Model ≤ ~1B params only with 8-bit/4-bit CPU or tiny models; no 7B |
| GPU | AMD Radeon 860M (integrated, shared memory) | **No CUDA. No ROCm on Windows for this iGPU.** |
| Disk | C: 1 TB, ≈695 GB free | WikiText-2 + two small checkpoints fine; 7B weights not |
| OS | Windows 11 (10.0.26200) | Linux-only tooling must run via Docker |
| Toolchain | Python 3.11.16 (uv-managed), uv 0.12.15, git 2.55, Docker Desktop, gh CLI | uv is the only supported dependency workflow |

**Binding rules derived from the above**

1. CUDA-dependent components (bitsandbytes CUDA kernels, TorchAO CUDA kernels, `vllm` CUDA,
   GPTQ/AWQ GPU kernels, flash-attention) MUST NOT run on the local machine. They are executed only
   on the cloud GPU substrate (§2b) and MUST fail loudly locally rather than silently fall back.
2. Any code path that requires CUDA MUST fail loudly (`NotImplementedError`/explicit config error),
   never silently fall back to a different numeric result.
3. **Local execution scope (binding, supersedes earlier wording).** The local machine is used ONLY
   for: repository management, CPU unit/property tests, tiny synthetic fixtures, static analysis and
   type checking, configuration validation, notebook generation, result analysis, figure/table/report
   production, and the CI smoke fixture. **No research training, no quantization-aware training, no
   large-scale inference, and no GPU evaluation runs locally** — including the Tier-1 tiny-transformer
   experiments, which move to the cloud substrate.
4. Every result carries a measurement-class label (§5). Class 5 and class 4-GPU are **unavailable
   locally** and must be reported as "not measured" locally, never estimated or faked; on the cloud
   substrate they become claimable only under §2b's manifest and cost rules. Class 4-CPU (a real CPU
   low-bit kernel executing an artifact we serialized ourselves) **is** available locally and is the
   only kernel-backed class that may be claimed from local execution — under the scope rules in §5.

---

## 2b. Cloud execution policy (binding, added 2026-10-08)

The developer's workstation has no usable CUDA GPU. Every GPU-dependent experiment therefore runs on
a **cloud notebook substrate**; the local machine orchestrates, validates, and analyses.

**Platforms**

| Platform | Role | Automation |
|---|---|---|
| **Google Colab** (chosen by the developer) | primary execution vehicle: interactive debugging and the declared experiment runs | consumer Colab has **no official submission API** — notebooks are generated from versioned configs and executed by the developer (or by Colab Enterprise when a GCP project is authorized); artifacts are exported to Drive/HF and validated locally by `spectraquant cloud collect` |
| **Kaggle Notebooks** | preferred **unattended** backend: upload → start → poll → download via the official CLI | fully programmatic |
| **Colab Enterprise** (optional) | programmatic one-off/scheduled runs via GCP CLI/SDK/REST | requires an authorized GCP project + billing |

**Rules**

1. **Thin notebooks, thick modules.** All scientific logic lives in importable `src/spectraquant/**`
   modules. A notebook MAY only: install the pinned environment, fetch versioned data, call repo
   code, record the git commit, capture hardware/dependency metadata, run the declared experiment,
   export machine-readable results. Logic that exists only in a notebook cell is a defect.
2. **Generated, not hand-edited.** Notebooks are produced from version-controlled experiment configs
   by a generator; a hand-edited notebook that cannot be regenerated is not accepted as evidence.
3. **Every remote run writes `run_manifest.json`** with: run id, resolved config, dependency lock
   information, git commit SHA, dataset versions/checksums, seeds, hardware metadata, start/end times,
   GPU-hours, status, metrics, artifact checksums. It is validated against `artifacts/schemas/`.
4. **Failures are preserved.** A failed run keeps its logs and enters the research record with its
   failure reason; it is never silently retried into a success.
5. **No fabricated results.** The registry is updated only from downloaded, checksum-validated
   artifacts. A run whose artifacts fail validation is recorded as failed.
6. **Resumability.** Submission is idempotent and re-runnable: the remote run id is persisted so a
   collection step can be resumed after interruption without re-submitting.
7. **Credentials never touch the repo.** Kaggle/GCP/HF/W&B/storage credentials live in platform
   secrets or environment variables only — never in notebooks, configs, commits, logs, or reports.
8. **No paid resource without prior authorization.** Do not start a paid cloud resource without the
   user's explicit authorization and a stated maximum estimated cost. Free tiers are the default.
9. **Measurement honesty is unchanged.** A cloud GPU result is class 4-GPU/5 only if it uses a real
   supported kernel and the controlled protocol in `docs/protocols/benchmark-protocol.md`; the
   hardware, driver, kernel, batch/sequence shape, warmups and repeats are recorded in the manifest.

---

## 3. Repository layout and ownership

Canonical layout (do not restructure without an ADR):

```
src/spectraquant/{data,factorization,quantization,proxies,regularizers,allocation,training,
                   evaluation,benchmarking,reporting,cli}
tests/{unit,integration,regression}
configs/{model,data,method,experiment}
scripts/{data,reproduce,experiments,benchmark}
docs/{architecture,research,protocols,decisions,results,coordination}
artifacts/{manifests,schemas,sample-results}
reports/{paper,figures,tables}
notebooks/{exploratory,final}
docker/
```

Path ownership is assigned in `docs/coordination/ownership.md`. **Two agents must never edit the
same file concurrently.** Shared interfaces require a design note under `docs/decisions/` before
implementation.

---

## 4. Scientific boundaries (MUST)

1. No agent/RAG/chatbot/orchestration framing.
2. Distinguish three different memory numbers at all times: training-time memory, stored checkpoint
   size, deployed inference representation.
3. Fake quantization MUST NOT be reported as low-bit storage or accelerated inference.
4. No latency claim without a real supported kernel and a controlled, documented protocol.
5. Comparisons are only valid at equal memory; every compression comparison MUST also report an
   equal-memory counterpart.
6. No tuning on test labels; no repeated test-set peeking to steer method decisions.
7. No training on LM Evaluation Harness evaluation questions.
8. Record license + terms for every model, dataset and upstream repository used
   (`docs/research/upstream-lockfile.md`).
9. No state-of-the-art claims from narrow or incomparable experiments.
10. Negative and inconclusive results are published, not hidden.
11. Core experiments must be reproducible with local open models; external APIs stay out of the
    core path.
12. No Kubernetes/Kafka/distributed infrastructure without a measured requirement.
13. Derived quantities that are approximate MUST be labelled approximate in code and in prose.
    Never present an approximation or estimator as exact.

---

## 5. Measurement taxonomy (label every number)

| # | Class | Definition |
|---|---|---|
| 1 | Analytical estimate | derived from shapes/bit widths only |
| 2 | Fake-quantization quality | float execution simulating quantization numerics |
| 3 | Packed storage | serialized low-bit weights, measured bytes |
| 4 | Kernel-backed inference | real supported low-bit computation |
| 5 | End-to-end service | request-level latency/throughput |

Classes MUST NOT be mixed inside a single claim.

**Class 4 is split by hardware, because a CPU low-bit kernel path was verified on this workstation on
2026-10-08** (evidence: `docs/research/backend-capability.md` §2.4, independently reproduced by the
orchestrator: an int4 weight-only ONNX artifact we serialize ourselves executes through ONNX Runtime's
`MatMulNBits` CPU kernel — payload 1024 B + scales 256 B inside a 1501 B artifact — and dynamic int8
executes through `MatMulInteger`, 2627 B artifact):

- **Class 4-CPU** — *available*. A real CPU kernel executes a low-bit artifact **that SpectraQuant
  itself serialized**. Permitted backends: ONNX Runtime CPU (`MatMulNBits` int4 weight-only,
  `MatMulInteger` int8) and torchao intx weight-only (`IntxWeightOnlyConfig(torch.int4, PerGroup(g))`).
  A class 4-CPU claim MUST name the backend, the kernel/op, the container format, the thread count and
  the CPU model, MUST compare against an fp32 CPU baseline measured on the same machine in the same
  session, and MUST NOT be presented as comparable to published GPU latency/throughput numbers.
- **Class 4-GPU** — *unavailable/deferred*. CUDA-only kernels (bitsandbytes, GPTQ/AWQ/Marlin,
  TorchAO CUDA tinygemm, FlashAttention, vLLM CUDA) cannot run here.
- Third-party-format kernels (llama.cpp GGUF, ONNX artifacts produced by other pipelines) are
  **engineering telemetry only** and may never be reported as class 4 for a SpectraQuant artifact.
- **Class 5** — *unavailable* on this workstation (vLLM has no Windows wheel; service measurement is
  out of scope until a Linux GPU host is documented).

Locally available classes: 1, 2, 3, and **4-CPU**. This makes hypothesis H5 ("fake-quantization gains
do not survive conversion to real packed weights and supported kernels") locally testable in the CPU
scope, and it does not license any latency claim beyond that scope.

---

## 6. Scope ladder and execution substrate

- **Tier 0 (local CPU, required)** correctness: synthetic matrices, tiny linear nets, exact
  enumeration, unit/property tests, deterministic fixtures, the CI smoke fixture.
- **Tier 1 (cloud, required)** tiny-transformer proof: small LM, full-factorial where feasible,
  ≥5 seeds for cheap cells, proxy-ranking validation. Trains on the cloud substrate (§2b) — **not**
  locally, per §2.3.
- **Tier 2 (cloud GPU, required for the primary claim)** TinyLlama-1.1B-class + WikiText-2,
  FP16/PTQ/QLoRA/LoftQ/LR-QAT vs SpectraQuant, ≥3 seeds.
- **Tier 3 (cloud GPU)** second 0.6–1.7B model.
- **Tier 4 (cloud GPU, optional)** ViT/DeiT CIFAR-100/ImageNet-100 cross-architecture test.
- **Tier 5 (cloud GPU, optional)** 3B–7B; core validity MUST NOT depend on it.

Tier 0 runs locally; Tiers 1–5 run on the cloud substrate. Local work for every cloud tier is
limited to fixture construction, configuration validation, submission, collection, checksum
validation, statistics, figures and report text. Scope reduction is only allowed via a timestamped
amendment in `docs/research/preregistration-amendments.md`.

---

## 7. Engineering standards

- Python 3.11, managed exclusively with `uv` (`pyproject.toml` + committed `uv.lock`).
- PyTorch (CPU wheel index locally). Hugging Face `transformers` for models/tokenizers.
- Ruff (lint + format), Pyright or mypy on core modules, pytest, Hypothesis for math properties.
- Hydra/OmegaConf for experiment composition; one tracker only (MLflow chosen, see ADR-0003).
- Every run writes a machine-readable manifest validated against `artifacts/schemas/`.
- **Cloud execution adapters** live in `src/spectraquant/cloud/**` and MUST support: generate/update
  `.ipynb` from a versioned config; submit (Kaggle / Colab Enterprise) or emit an
  execution-ready Colab notebook; persist the remote run id; poll with bounded retries; download
  logs/metrics/checkpoints/plots/executed notebook; validate expected artifacts and checksums;
  update the run registry from validated artifacts only; resume safely after interruption.
  Credentials come from environment variables/platform secrets, never from the repository.
- Core math functions MUST document shapes, dtypes, devices, assumptions, numerical limitations.
- Deterministic by default: seed everything; tests must be reproducible bit-for-bit on CPU.
- Never commit: model weights, datasets, credentials, tokens, caches, large raw outputs,
  machine-specific paths.

### Required test coverage (minimum)

quantize/dequantize round trip; zero-range/constant/non-divisible-group edge cases; memory-accounting
formula vs serialized bytes; factorization reconstruction; proxy vs exact toy computation; allocator
feasibility and budget compliance; seed determinism; config validation; metrics vs hand-computed
fixtures; train/eval overlap detection; one tiny end-to-end CI experiment.

---

## 8. Commands

```bash
uv sync --all-extras                 # create/refresh env
uv run pytest -q                     # full test suite (CPU)
uv run ruff check . && uv run ruff format --check .
uv run spectraquant --help           # CLI
uv run spectraquant smoke --config configs/experiment/smoke.yaml   # tiny e2e
make check                           # lint + type + tests (see Makefile)
```

Never run repo-wide lint/format/test loops mid-task; run them once at the end of a change.

---

## 9. Git protocol

1. Never develop directly on `main`.
2. Branch prefixes: `research/`, `feat/`, `infra/`, `docs/`, `fix/`, `eval/`, `perf/`.
3. Conventional commit prefixes: `feat:`, `fix:`, `research:`, `eval:`, `infra:`, `docs:`, `test:`,
   `chore:`.
4. Each commit is a coherent unit and passes the checks relevant to it.
5. Push accepted commits to the remote branch; never force-push `main`; never rewrite public release
   history.
6. Every result manifest records the exact commit SHA and a dirty-state flag.
7. Tag reproducible checkpoints: `repro-v0.1`, `method-v0.1`, then semantic releases.

---

## 10. Agent reporting contract

Every agent (subagent or human) returns exactly:

1. Scope completed.
2. Files changed.
3. Commands executed.
4. Tests and results.
5. Evidence created.
6. Assumptions.
7. Risks / unresolved questions.
8. Recommended next action.

Report failures and negative findings directly. Never weaken a test, skip a baseline, or replace an
unavailable measurement with an estimate.
