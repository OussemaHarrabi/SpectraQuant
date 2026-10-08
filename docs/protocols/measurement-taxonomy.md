# Measurement taxonomy — operational rules

**Status:** normative. Expands `AGENTS.md` §5 into rules that a reviewer can enforce against a
manifest. **Owner:** D-lite. **Created:** 2026-10-08. **Companion documents:**
`docs/protocols/benchmark-protocol.md` (classes 4–5 procedure),
`docs/protocols/memory-accounting.md` (classes 1 and 3 arithmetic),
`docs/research/backend-capability.md` (which backends exist on this host).

`AGENTS.md` §5 defines five measurement classes and forbids mixing them. That is necessary but not
sufficient: it does not say what evidence each class needs, what it may claim, or what it must
record. This document does. Where this document and `AGENTS.md` disagree, `AGENTS.md` wins and this
document MUST be amended.

---

## 1. The five classes, operationally

Every number in a manifest, result table, figure or README line carries exactly one class label.

### Class 1 — Analytical estimate

* **Is:** a number derived from shapes, bit widths and declared metadata only. Nothing is executed,
  nothing is serialized.
* **May claim:** bit costs, parameter counts, memory upper bounds, FLOP estimates, budget
  feasibility. Wording MUST include *analytical* / *estimated*.
* **Required evidence:** the formula, its inputs (shapes, dtypes, `group_size`, symmetric flag,
  alignment rule) and a hand-checkable arithmetic trace of the kind in
  `memory-accounting.md` §7.
* **Required metadata:** `analytical.formula_id`, `analytical.inputs`, and
  `analytical.approximate: true` unless the formula is exact for the modelled object.
* **Forbidden:**
  * reporting an analytical number as a **measured** byte count or a measured time;
  * reporting analytical `bits_per_param` as serialized size (`memory-accounting.md` §6);
  * presenting an estimate without the word *estimate* in prose and `analytical` in the manifest.
* **Naming:** keys prefixed `est.` or suffixed `.analytical_estimate_bytes`.

### Class 2 — Fake-quantization quality

* **Is:** float execution that *simulates* quantization numerics (quantize → dequantize kept in
  float), used to measure **quality** (loss, perplexity, accuracy, sensitivity ranking).
* **May claim:** quality deltas against an fp baseline and against an **equal-memory** counterpart
  (`AGENTS.md` §4.5); relative ordering of layer sensitivities; proxy-vs-exact agreement.
* **Required evidence:** the exact fake-quant recipe (bit width, granularity/group size,
  symmetric vs asymmetric, scale dtype, clipping/percentile, whether a straight-through estimator
  was used, and the compute dtype), the eval harness and its config, the dataset split identifier
  and hash, and seeds. Quality numbers MUST come with the memory number of the configuration they
  describe.
* **Required metadata:**
  `fake_quant.scheme`, `bits`, `group_size`, `symmetric`, `scale_dtype`, `compute_dtype`,
  `ste_used`, `harness.name`, `harness.version`, `harness.commit`, `tasks`, `num_fewshot`, `seed`,
  `split_sha256`.
* **Forbidden:**
  * reporting fake-quantized quality as evidence of **speed** or of low-bit inference;
  * reporting fake-quantized tensors as **low-bit storage** (a float tensor is fp32/fp16 bytes —
    storage claims must come from class 1 or class 3);
  * claiming "measured memory" from a model whose weights are fp32/fp16 during the run;
  * reporting a fake-quant result as an integer-kernel result.
* **Note:** class 2 is the *only* quality class available on this workstation. All Tier-0/1 quality
  claims here are class 2 by construction.
* **Procedure:** *how* a class-2 evaluation is run — split separation, the predeclared calibration
  slice, harness pin, the bounded task suite, the contamination audit and the byte-identity freeze
  (`eval_config_hash`) — is normative in `docs/protocols/eval-protocol.md`. This section defines only
  the class label and the evidence/metadata a class-2 number must carry; a class-2 run that does not
  follow `eval-protocol.md` is exploratory and MUST NOT be presented as confirmatory.

### Class 3 — Packed storage

* **Is:** bytes of a serialized artifact that we actually write and can re-read (our own int8/int4
  packers, `safetensors`, an ONNX/GGUF-like container we produce).
* **May claim:** `mem.checkpoint.*` and `mem.deployed.stored_bytes`
  (`memory-accounting.md` §1), measured `bits_per_param`, and honest compression ratios with stated
  numerator/denominator.
* **Required evidence:** artifact path; `os.path.getsize` of **every** file in the artifact
  directory (payload, sidecar `.data`, scale/zero-point files, config, tokenizer); sha256; a
  load-back round trip (deserialize → dequantize → compare) reporting max absolute error and dtype;
  and a byte-accounting table that reconciles with `memory-accounting.md` §3.
* **Required metadata:** `artifact.format`, `artifact.files[]` (`name`, `bytes`, `sha256`),
  `artifact.payload_bytes`, `artifact.metadata_bytes`, `artifact.total_bytes`, `packing`,
  `group_size`, `scale_dtype`, `zero_point_dtype`, `alignment_bytes`, `packer.commit`.
* **Forbidden:**
  * counting only the weight file and omitting scale/zero-point/metadata bytes (understates size);
  * reporting a fake-quantized **float checkpoint** as low-bit storage;
  * reporting theoretical `bits_per_param` as the measured serialized figure;
  * reporting `stored_bytes` under the name `resident_bytes` (see `memory-accounting.md` §1).
* **Availability here:** yes — PyTorch 2.14.1+cpu packs int8/int4/int3 weights, and ONNX Runtime
  writes int4 containers whose bit packing we verified byte-exact (`backend-capability.md` §2).

### Class 4 — Kernel-backed inference (split by hardware: `4-CPU` / `4-GPU`)

`AGENTS.md` §5 splits class 4 by hardware, because a CPU low-bit kernel path was verified on this
workstation on 2026-10-08 (`backend-capability.md` §2.4b). Every class-4 claim MUST carry its
sub-class label; an unlabelled "class 4" is a protocol violation.

**Common definition (both sub-classes).**

* **Is:** a forward/generation executed by a real library kernel that computes (not merely stores) in
  reduced precision — int8/int4 GEMM/GEMV — rather than dequantize-to-float.
* **May claim:** latency, throughput, memory bandwidth; **only** under
  `docs/protocols/benchmark-protocol.md` (common block §5; `4-CPU` additionally §8).
* **Required evidence (both):** the identity of the kernel that actually ran (library, version,
  kernel name **and domain**, e.g. `MatMulNBits` in `com.microsoft`), the execution
  provider/backend **as selected at runtime and logged**, proof the weights were not silently
  upcast, the **thread count**, and the full benchmark metadata block.
* **Required metadata (both):** `sub_class` (`"4-CPU"` | `"4-GPU"`), the entire `bench` block of
  `benchmark-protocol.md` §5, plus `backend.name`, `backend.version`, `backend.commit`,
  `kernel.name`, `kernel.domain`, `weights_dtype`, `activations_dtype`, `dequantized_path: false`.

**`4-CPU` — available here.** A real CPU kernel executes a low-bit artifact **that SpectraQuant
itself serialized**. Permitted backends (`AGENTS.md` §5): ONNX Runtime CPU (`MatMulNBits` int4
weight-only, `MatMulInteger` int8) and torchao intx weight-only
(`IntxWeightOnlyConfig(torch.int4, PerGroup(g))`). Additional requirements:

* name the **container format** and the **CPU model** in every result;
* compare against an **fp32 CPU baseline measured on the same machine in the same session**;
* state explicitly that the figure is **not comparable to published GPU latency/throughput numbers**.

**`4-GPU` — unavailable here.** CUDA-only kernels (bitsandbytes, GPTQ/AWQ/Marlin, TorchAO CUDA
tinygemm, FlashAttention, vLLM CUDA) cannot run on this host.

**Forbidden (both):**

* reporting a **dequantized fallback** as kernel acceleration (the likeliest error on this host:
  bitsandbytes CPU 4-bit loads and runs here, but computes in fp32 after dequantizing — its timings
  are class 2/3, never class 4);
* reporting class-2 numbers as class 4;
* reporting a speedup measured on other hardware as if measured here;
* any latency claim without `benchmark-protocol.md` metadata;
* **third-party-format kernels** (llama.cpp GGUF, ONNX artifacts produced by another pipeline)
  reported as class 4 — they are engineering telemetry, never class 4 for a SpectraQuant artifact;
* a `4-CPU` number presented without its fp32 same-session baseline, or presented as comparable to
  GPU numbers.

* **Availability here:** **`4-CPU` available; `4-GPU` not available** (see §4, §5).

### Class 5 — End-to-end service

* **Is:** request-level measurement against a serving endpoint (HTTP/gRPC), including queueing,
  batching, tokenisation and scheduling.
* **May claim:** p50/p95/p99 request latency, throughput at a declared concurrency, goodput,
  time-to-first-token.
* **Required evidence:** server name + version + commit; model artifact sha256; the server config;
  concurrency and arrival process; **prompt tokens and generated tokens per request**; warm-up
  requests (count, discarded?); client host recorded separately from server host; the raw
  per-request records (not just aggregates).
* **Required metadata:** `bench` block **plus** `service.endpoint`, `service.server_version`,
  `service.concurrency`, `service.arrival_process`, `service.num_requests`,
  `service.prompt_tokens`, `service.generated_tokens`, `service.warmup_requests`,
  `service.client_host`.
* **Forbidden:**
  * presenting single-request wall-clock as service throughput;
  * mixing client-side and server-side timing in one number;
  * publishing a service number without the raw per-request record;
  * reporting an unmeasured client/server topology.
* **Availability here:** **not available** (§4, §5).

---

## 2. Universal metadata block (required in every manifest)

```jsonc
"measurement": {
  "class": 2,                                   // 1..5, exactly one
  "class_name": "fake_quant_quality",
  "claim": "one sentence stating precisely what is claimed",
  "not_measured": ["class4", "class5"],          // always list the classes NOT measured here
  "tool":    { "name": "torch", "version": "2.14.1+cpu", "commit": null },
  "host": {
    "os": "Windows 11 10.0.26200",
    "cpu": "AMD Ryzen AI 7 350", "physical_cores": 8, "logical_cores": 16,
    "ram_bytes": 16233910272,
    "gpu": "AMD Radeon 860M (integrated, no CUDA)",
    "cuda_available": false,
    "torch_num_threads": 8,
    "quantized_engine": "onednn"
  },
  "repro": { "commit": "<repo sha>", "dirty": false, "seed": 0, "config_sha256": "..." }
}
```

Class-specific blocks are given in §1 and, for classes 4–5, in `benchmark-protocol.md` §5.

---

## 3. Cross-class invariants

1. **One claim, one class.** A sentence mixing classes (e.g. "4-bit, 4× smaller, 2× faster") MUST be
   split, with each part labelled. Every class-4 claim MUST additionally carry its **sub-class**
   label (`4-CPU` or `4-GPU`): an unlabelled "class 4" is a protocol violation, because the two
   sub-classes have different availability, evidence requirements and comparability limits here.
2. **Equal memory.** Every compression comparison also reports an equal-memory counterpart
   (`AGENTS.md` §4.5). A quality table without the memory column is invalid.
3. **Consistency.** Where an analytical and a measured number describe the same artifact, both are
   published and their discrepancy is reported (`memory-accounting.md` §6).
4. **Honest absence.** Any class not measured is written as `not_measured` / "not measured"; it is
   never estimated, imputed, or borrowed from a paper.
5. **No silent fallback.** If an execution path degrades to a different backend than requested, the
   run is recorded as a **backend-capability failure** (`benchmark-protocol.md` §6) and MUST NOT be
   reported under the originally requested class.
6. **Approximation labelling.** Anything approximate is labelled approximate in both code and prose
   (`AGENTS.md` §4.13).
7. **Evaluation procedure.** How a class-2 evaluation is executed — split separation (§2 of
   `docs/protocols/eval-protocol.md`), the predeclared calibration slice, the harness pin, the
   bounded task suite, the contamination audit and the byte-identity freeze (`eval_config_hash`) — is
   normative in `docs/protocols/eval-protocol.md`. The class definitions and metadata requirements in
   this document are the labels those runs must carry; the two documents are complementary, and where
   they appear to differ, this document governs labelling and `eval-protocol.md` governs procedure.

---

## 4. Availability on this workstation (2026-10-08)

Host: AMD Ryzen AI 7 350, 8C/16T, 16.23 GB RAM, no CUDA device, Windows 11 10.0.26200.
Evidence for each verdict: `docs/research/backend-capability.md`.

| # | Class | Status here | Deferred to | Basis |
|---|---|---|---|---|
| 1 | Analytical estimate | **AVAILABLE** | — | Pure arithmetic; no host dependency |
| 2 | Fake-quantization quality | **AVAILABLE** | Tier 2+ for ≥1B models | CPU fp32/fp16 execution; `torch 2.14.1+cpu`, `mkldnn=True`; tiny / Tier-0–1 models only |
| 3 | Packed storage | **AVAILABLE** | Tier 2+ for ≥1B artifacts | int8/int4/int3 packing verified in torch 2.14.1+cpu; byte-exact int4 container verified via ONNX Runtime 1.30.0 |
| 4-CPU | Kernel-backed inference, CPU, **SpectraQuant-serialized artifact** | **AVAILABLE** | Tier 2+ for ≥1B artifacts | ONNX Runtime 1.30.0 `MatMulNBits` (int4) / `MatMulInteger` (int8) execute our own container on this CPU; torchao intx int4/int3 weight-only forwards on CPU — protocol in `benchmark-protocol.md` §8 |
| 4-GPU | Kernel-backed inference, CUDA | **UNAVAILABLE** | GPU host (`backend-capability.md` §5) | No NVIDIA device; all CUDA kernels absent |
| 5 | End-to-end service | **UNAVAILABLE** | GPU host with a Linux serving stack | See §5 |

"Deferred" here means *planned behind the external-GPU gate* (`AGENTS.md` §2.3, §6), never
"approximated in the meantime". Per `AGENTS.md` §2.4 a deferred sub-class is reported
**"not measured"**, and per §6 of this document no estimate may stand in for it.

**Any `4-GPU` or class-5 value, and any unlabelled "class 4" value, appearing in a result produced
on this host is a protocol violation.** A `4-CPU` value is valid only when it carries the §1
sub-class requirements and the `benchmark-protocol.md` §8 procedure (thread pinning, same-session
fp32 CPU baseline, not-comparable-to-GPU scope line).

---

## 5. Capability vs authorization (applied per class-4 sub-class)

These are two different questions and conflating them is how a project ends up publishing an
unmeasurable claim. `AGENTS.md` §5 resolves them differently for `4-CPU` than for `4-GPU`/5.

* **Capability** — can *some* low-bit kernel run on this host at all? *Yes.* Four verified CPU kernel
  surfaces exist: ONNX Runtime int4 weight-only (`MatMulNBits`, domain `com.microsoft`) and int8
  (`MatMulInteger`), torchao intx weight-only (int4/int3), PyTorch/oneDNN int8
  (`quantize_dynamic`; engine `onednn`), and llama.cpp k-quant/GGUF CPU kernels.
* **Authorization** — may a number so obtained be reported as a SpectraQuant measurement? This is now
  split, not binary:

  1. **`4-CPU` — authorized, with conditions.** A real CPU kernel executing a low-bit artifact
     **SpectraQuant serialized itself** through a *permitted* backend (ONNX Runtime CPU, torchao
     intx) may be reported as `4-CPU`, provided `benchmark-protocol.md` §8 is followed: pinned
     thread count, named CPU model, named container format, an fp32 CPU baseline measured in the same
     session on the same machine, and an explicit "not comparable to GPU" scope line. `AGENTS.md`
     §5 authorizes exactly this, which is what makes hypothesis H5 locally testable.
  2. **`4-GPU` — not authorized.** Every toolkit that executes low-bit weights as *kernels* on our
     kind of artifact is CUDA-gated here (torchao `Int4WeightOnlyConfig` fails with
     `Requires mslk ≥ 1.0.0`, a CUDA-index package; GPTQ/AWQ/exllama/marlin kernels are CUDA; vLLM's
     quantized paths are Linux/CUDA). The disqualifier for GPU-class claims is unchanged: no device,
     and no number measured here would be comparable with the GPU baselines the literature reports
     (`AGENTS.md` §4.4, §4.9).
  3. **Third-party formats — telemetry only.** llama.cpp GGUF kernels, externally produced ONNX
     artifacts and the PyTorch/oneDNN int8 path (not on the §5 permitted list) run fine, but they
     execute someone else's artifact or a non-permitted backend. Their timings MAY be logged as
     engineering telemetry and MUST NOT be reported under any class-4 label.
  4. **Class 5 — not authorized.** The only serving stack reachable here is vLLM inside the Linux
     container (CPU support is Linux-only; no Windows wheel), and it would serve a non-SpectraQuant
     model. `AGENTS.md` §2.4 keeps service measurement out of scope on this workstation.

**Escalation path (still required, now only for the excluded cases).** Promoting a *third-party
format*, a non-permitted backend, a `4-GPU` configuration, or any class-5 measurement to a reported
result requires (a) an ADR under `docs/decisions/`, (b) a timestamped amendment in
`docs/research/preregistration-amendments.md`, and (c) a benchmark-protocol entry naming the kernel
and the container format. `4-CPU` on a SpectraQuant artifact needs none of that — it is already
authorized — but it MUST follow `benchmark-protocol.md` §8.

---

## 6. Forbidden-claim quick reference

| Wrong (forbidden) | Why | Correct |
|---|---|---|
| "Our int4 model is 4× smaller" (measured from an fp16 fake-quant checkpoint) | fake-quant floats are fp16 bytes | "analytical int4 payload is 4.125 bits/param; measured artifact is X bytes (`mem.checkpoint.total_bytes`)" |
| "int4 checkpoint = 4 bits/param" | scales/zero-points/container exist | analytical 4.125 vs measured **5.86–6.42** for the §7.5 fixture (container term is build-dependent — §6.1) |
| "4-bit inference on CPU is 1.8× faster" (bitsandbytes CPU / any dequant fallback) | dequantize-then-fp32 is not a low-bit kernel | report as class 2/3 behaviour; no speed claim |
| "We ran our int4 container through the CPU int4 kernel, so we are 1.7× faster than fp32" **without a same-session fp32 baseline and thread/CPU disclosure** | class `4-CPU` is conditional, not free | a full `4-CPU` claim per §8 of `benchmark-protocol.md`: named kernel/domain, container format, thread count, CPU model, same-session fp32 baseline, "not comparable to GPU" line |
| "1.7× faster" on a `4-CPU` table with no scope line | CPU int8/int4 GEMM on an 8-core laptop is not GPU-comparable | add the mandatory scope line; never compare with published GPU speedups |
| "llama.cpp q4_k runs at N tok/s, so our method's kernels are fast" | GGUF is a third-party artifact | engineering telemetry only; never a class-4 claim for a SpectraQuant artifact |
| "Matches the 2.1× speedup reported by [method]" | measured on other hardware | "not measured here; literature figure from [cite]" |
| "End-to-end latency 42 ms" (single `model.generate` wall clock) | that is not a service measurement | `4-CPU` per-forward latency under `benchmark-protocol.md` §5 + §8, or class 5 with a real server and raw records |
| A single bare "memory" column | three distinct numbers exist | name `mem.training.*`, `mem.checkpoint.*`, `mem.deployed.*` |
