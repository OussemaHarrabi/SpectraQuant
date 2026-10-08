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
   GPTQ/AWQ GPU kernels, flash-attention) MUST NOT enter the core experiment path. They MAY be
   referenced, pinned, and documented as *deferred* behind a documented GPU prerequisite.
2. Any code path that requires CUDA MUST fail loudly (`NotImplementedError`/explicit config error),
   never silently fall back to a different numeric result.
3. Tier 0 and Tier 1 (see §6) are the locally executable scope. Tier 2+ runs remain *planned* until
   an external GPU is documented; they may not be reported as completed.
4. Every result carries a measurement-class label (§5). Classes 4 and 5 are **unavailable on this
   workstation**; they must be reported as "not measured", never estimated or faked.

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

Classes MUST NOT be mixed inside a single claim. Locally available: 1, 2, 3 (3 only for formats we
can actually serialize, e.g. int8/int4 packing routines we own). Classes 4–5: unavailable here.

---

## 6. Scope ladder

- **Tier 0 (local, required)** CPU correctness: synthetic matrices, tiny linear nets, exact
  enumeration, unit/property tests, deterministic fixtures.
- **Tier 1 (local, required)** tiny transformer proof: small LM, full-factorial where feasible,
  ≥5 seeds for cheap experiments, proxy ranking validation.
- **Tier 2 (GPU required, planned)** TinyLlama-1.1B-class + WikiText-2, FP16/PTQ/QLoRA/LoftQ/LR-QAT
  vs SpectraQuant, ≥3 seeds.
- **Tier 3 (GPU, planned)** second 0.6–1.7B model.
- **Tier 4 (optional)** ViT/DeiT CIFAR-100/ImageNet-100 cross-architecture test.
- **Tier 5 (optional)** 3B–7B; core validity MUST NOT depend on it.

Scope reduction is only allowed via a timestamped amendment in
`docs/research/preregistration-amendments.md`.

---

## 7. Engineering standards

- Python 3.11, managed exclusively with `uv` (`pyproject.toml` + committed `uv.lock`).
- PyTorch (CPU wheel index locally). Hugging Face `transformers` for models/tokenizers.
- Ruff (lint + format), Pyright or mypy on core modules, pytest, Hypothesis for math properties.
- Hydra/OmegaConf for experiment composition; one tracker only (MLflow chosen, see ADR-0003).
- Every run writes a machine-readable manifest validated against `artifacts/schemas/`.
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
