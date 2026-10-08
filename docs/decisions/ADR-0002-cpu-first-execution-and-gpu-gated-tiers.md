# ADR-0002 — CPU-first execution with GPU-gated tiers

* **Status:** Accepted
* **Date:** 2026-10-08
* **Deciders:** repository bootstrap (infra stream)
* **Supersedes:** none

## Context

The only compute available is a CPU-only Windows workstation: AMD Ryzen AI 7 350 (8 cores /
16 threads), 16.2 GB RAM, and an AMD Radeon 860M integrated GPU with no CUDA, no ROCm support on
Windows for that part, and no maintained research-grade alternative (`docs/research/environment.md`).
There is no `nvcc`, no NVIDIA device, and no external GPU secured.

The project's own scope ladder (`AGENTS.md` section 6) already reflects this: Tier 0 (synthetic
correctness) and Tier 1 (tiny transformer) are local; Tier 2 (TinyLlama-1.1B + WikiText-2), Tier 3
(second 0.6-1.7B model), Tier 4 (ViT) and Tier 5 (3B-7B) require a GPU that does not exist yet.

The measurement taxonomy (`AGENTS.md` section 5) makes the consequence concrete: classes 1-3 are
locally obtainable (analytical estimates, fake-quantization quality, packed storage for formats we
can actually serialize); classes 4 (kernel-backed inference) and 5 (end-to-end service) are not,
and section 4.4 forbids a latency claim without a real supported kernel and a controlled protocol.

Options:

1. **CPU-first, GPU-gated** — build and prove the harness locally; every GPU-dependent path raises
   until the hardware is documented.
2. Install a CUDA-emulating or vendor-specific backend (DirectML, ROCm for Windows) — DirectML's
   PyTorch backend is unmaintained; ROCm does not support this iGPU on Windows. Both would produce
   numbers that look like class 4 while being nothing of the sort.
3. Rent cloud GPU time immediately — the cost policy is explicitly undecided in
   `docs/research/environment.md` section 5, and paying before the harness is validated would buy
   nothing but noise.
4. Drop the GPU tiers from the plan — dishonest: the research question genuinely needs them, and
   the preregistration is required to scope them as *planned*, not to pretend they do not exist.

## Decision

1. **CPU is the default and only currently executable backend.** `torch` is pinned to the CPU wheel
   index; no CUDA-only package (bitsandbytes, CUDA TorchAO kernels, vLLM, GPTQ/AWQ GPU kernels,
   flash-attention) may enter the core dependency set.
2. **Every GPU-requiring code path raises.** `NotImplementedError` with an explicit reason is the
   contract, e.g. `spectraquant.benchmarking.measurement.measure_inference_latency`, which refuses
   to emit a latency number because class 4/5 evidence is unobtainable here. Silently falling back
   to a different numeric path is forbidden.
3. **Tier 2-5 are reported as planned, never completed.** Manifests describe what actually ran;
   documents and README status tables mark Tier 2-5 as gated on an external GPU.
4. **Classes 4 and 5 are written as "not measured".** No estimate, no surrogate, no proxy number is
   substituted for them.
5. **The preregistration scopes honestly.** Only Tier 0/1 claims may be made from local runs; the
   amendment log (`docs/research/preregistration-amendments.md`) records any hardware-driven scope
   change with a timestamp.
6. **Docker is Linux, not GPU.** `docker/Dockerfile` is a CPU image; it exists for tooling parity,
   not for acceleration.

## Amendment — 2026-10-08 (later the same day): cloud execution substrate

`AGENTS.md` §2.3 (local execution scope) and the new §2b (cloud execution policy) supersede parts of
this ADR:

* **Decision point 3 is superseded.** Tier 1 no longer runs locally. The scope ladder puts Tiers 1-5
  on a cloud notebook substrate (Colab primary, Kaggle unattended); local execution is limited to
  repository management, CPU unit/property tests, tiny synthetic fixtures, static analysis and type
  checking, configuration validation, notebook generation, result analysis, report production and the
  CI smoke fixture.
* **Decision point 4 is refined.** Class 4 is now split by hardware (`AGENTS.md` §5): **class 4-CPU is
  locally available** for a real CPU low-bit kernel (ONNX Runtime `MatMulNBits`/`MatMulInteger`,
  torchao intx) executing an artifact SpectraQuant itself serialized, compared against an fp32 CPU
  baseline measured in the same session — evidence in `docs/research/backend-capability.md` §2.4.
  Class 4-GPU and class 5 remain unavailable locally and move to the cloud substrate.
* **Unchanged.** No CUDA-only package enters the core dependency set; every GPU-dependent code path
  still raises `NotImplementedError` rather than approximating; the CPU-only Docker image, the
  determinism rules and the manifest contract all stand.

The cloud adapter itself (`src/spectraquant/cloud/**`, contract frozen in
`docs/coordination/design-cloud-adapter.md`) is owned by a later slice and is **not** part of this
scaffold. Its optional dependencies are declared now as the `cloud` extra
(`nbformat>=5.10`, `kaggle>=1.7`) so `uv.lock` is stable and the adapter can import them lazily.

## Consequences

**Positive**

* Every committed number is reproducible on the machine that produced it, today.
* The harness (config, seeding, manifests, schema, CI, smoke) is validated before expensive compute
  is ever requested, so a future GPU run starts from a working measurement pipeline.
* No dependency on vendor libraries that would have to be swapped later; the code that runs on a
  GPU later is the same code that runs on CPU now.

**Negative / accepted costs**

* The repository cannot currently answer its own headline question at scale; Tier 0/1 results are
  evidence about the harness and about toy-model behaviour only, and must be labelled as such.
* Anyone expecting "QLoRA on TinyLlama" numbers will not find them until a GPU is secured.
* Keeping CUDA paths unimplemented means the GPU migration is real work (kernels, memory accounting,
  batching), not a flag flip.

## Alternatives considered (summary)

* DirectML / ROCm-on-Windows — rejected: unsupported or unmaintained, and would produce numbers that
  masquerade as class 4.
* Immediate cloud GPU — rejected for now: cost policy undecided, harness unvalidated.
* Dropping Tier 2-5 from the plan — rejected: dishonest scoping; they stay in the ladder as planned.

## References

* `AGENTS.md` sections 2, 4.4, 5, 6
* `docs/research/environment.md` sections 1, 2, 5
* `src/spectraquant/benchmarking/measurement.py`, `src/spectraquant/training/loop.py`
* `docker/Dockerfile`
