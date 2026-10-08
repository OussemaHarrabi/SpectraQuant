# Benchmark protocol — latency, throughput and memory measurement

**Status:** normative. **Class `4-CPU` (available on this workstation) is live** — §2 plus §8 govern
it. Class `4-GPU` and class 5 remain unavailable on the workstation of record
(`docs/protocols/measurement-taxonomy.md` §4–§5), so those parts of the protocol stay dormant until a
GPU host exists; they are written now so that the first GPU run is protocol-clean, and so that local
engineering timings are clearly separated from published claims.

**Owner:** D-lite. **Created:** 2026-10-08. **Companions:**
`docs/protocols/measurement-taxonomy.md` (class labels + forbidden claims),
`docs/protocols/memory-accounting.md` (byte accounting),
`docs/research/backend-capability.md` (which kernels exist where).

---

## 1. Scope

Applies to **every** number that is a time, a rate or a resident-memory size:

* **class 4 (kernel-backed inference)**: per-forward / per-token latency, prefill and decode
  throughput, memory bandwidth, peak resident memory of the inference process. Class 4 is split by
  hardware (`AGENTS.md` §5): **`4-CPU` is available here and uses the additional protocol in §8**;
  `4-GPU` is unavailable. Every class-4 row MUST state its sub-class;
* class 5 (end-to-end service): request latency percentiles, throughput, TTFT, goodpath/goodput;
* any local "how slow is this" engineering timing, including Tier-0/1 CPU timings.

Does **not** apply to class 1 (analytical), class 2 (quality; quality metrics are not timings) or
class 3 (stored bytes). Those are governed by `memory-accounting.md`.

---

## 2. Non-negotiable rules

1. **Fixed workload shape.** Every reported run declares and holds constant: `batch_size`,
   `sequence_length` (input), `generation_length` (output), `dtype`, and the exact tensor shapes
   fed to the measured op/layer. Shape sweeps are reported as a table, one row per shape — never as
   a single averaged number.
2. **Warm-up is mandatory and disclosed.** Discarded iterations are counted and reported. The
   discarded warm-up count MUST be ≥ the library's documented warm-up needs (kernel autotuning,
   JIT/`torch.compile` compilation, allocator pool growth) and ≥ 10 % of measured iterations, with a
   floor of 3 for CPU-only runs. Warm-up iterations are NEVER included in the statistics.
3. **Synchronisation is explicit.** Every timed region ends with the documented synchronisation
   primitive for the backend actually used, and the primitive is recorded:
   * CUDA → `torch.cuda.synchronize()`;
   * CPU eager PyTorch → `sync_method: "cpu_blocking"` (CPU eager execution is synchronous; the
     value MUST still be recorded so a future async/compiled path cannot inherit a false claim);
   * `torch.compile` / lazy backends → the backend's documented sync or a full result
     materialisation (`.cpu()` / `float(...)`) inside the timed region;
   * ONNX Runtime → `sync_method: "ort_run_blocking"` (`InferenceSession.run` is blocking);
   * vLLM/service → response completion, not request submission.
   Timing at submission without synchronisation is a protocol violation, not a caveat.
4. **Timer.** `time.perf_counter_ns` for wall clock; `torch.cpu`/`torch.cuda` events additionally if
   they exist for the backend. The timer name is recorded.
5. **Repeats.** `repeats ≥ 5` for published kernel numbers and `≥ 3` for service numbers; the
   **median of repeats** is the point estimate, and every repeat is stored. Repeats that were
   discarded are listed with their reason (§4).
6. **Hardware and clock recorded.** CPU/GPU model, physical/logical core count, RAM, total VRAM,
   driver version, CUDA/ROCm runtime version, library versions, and the clock/power context:
   Windows power plan or Linux `scaling_governor`, and whether AC power and thermal throttling
   state were recorded. Without these, a timing number is not publishable.
7. **Percentiles with sample counts.** p50/p95/p99 MAY only be reported with the number of samples
   behind them. Rule: `p95` and `p99` REQUIRE ≥ 100 samples; with fewer samples report
   `median` + `min` + `max` + `IQR` and label the percentiles *indicative*. Never quote a p99 from
   5 repeats.
8. **Baseline on the same host.** Every speedup is a ratio of two runs on the same host, same
   shapes, same protocol, same session; the fp baseline is measured in the same script.
9. **Both memory numbers.** Peak resident memory is reported as `stored_bytes` and `resident_bytes`
   separately (`memory-accounting.md` §1) plus `peak_rss_bytes`, sampled with a stated method
   (e.g. `psutil.Process().memory_info().rss` polled between iterations, or the platform equivalent).
10. **Raw records are retained.** Per-iteration/per-request records are written to the artifact
    directory and referenced by hash from the manifest; only aggregates may be published, but the
    records must exist.
11. **Threads are pinned and reported.** For any CPU timing the thread count is set explicitly
    (`torch.set_num_threads`, ONNX Runtime `SessionOptions.intra_op_num_threads` and
    `inter_op_num_threads`, `OMP_NUM_THREADS`) — never left at the library default — then **read
    back**, recorded, and the CPU model stated alongside. The `4-CPU` variant of this rule is
    normative in §8.

---

## 3. Repository defaults (must be overridden explicitly, never silently)

| Field | Default |
|---|---|
| `warmup_iters` | 10 (discarded) |
| `measured_iters` | 50 |
| `repeats` | 5 |
| `batch_size` | 1 |
| `sequence_length` | 512 |
| `generation_length` | 128 |
| `dtype` | fp32 on CPU, bf16 on GPU |
| `torch_num_threads` | 8 (recorded; not left to the library default) |
| `quantize_dynamic`-style int8 | group/per-channel as declared in the artifact manifest |

---

## 4. Discarded-run and retry logging

Every run that is started and not used in the statistics is recorded, with a reason code:

| Reason code | Meaning | Handling |
|---|---|---|
| `warmup` | declared warm-up | expected; count reported |
| `thermal` | throttling detected (clock drop, temperature rise) | re-run on a cool host; if ≥ 1/3 of repeats are thermal, the whole measurement is flagged `degraded: true` |
| `oom` | out-of-memory of any kind | record the attempted setting; retry once at a smaller `batch_size` ONLY as a separate labelled run |
| `backend_capability_failure` | the requested kernel/format was not available or silently substituted | see §6 — the run is invalid, not retried silently |
| `nondeterministic_op` | NaN/Inf or op-level nondeterminism | record; do not average over it |
| `interference` | another process consumed the CPU/GPU | re-run; record what was running |

Rules:

* Retries MUST NOT be used to select the best sample. The reported statistic is computed over the
  *first* `repeats` usable runs after warm-up; later re-runs replace the degraded batch wholesale,
  not individual samples.
* `degraded: true` measurements MAY be reported as diagnostics but MUST NOT be used for a headline
  claim or a Pareto frontier point.
* OOM retries change `batch_size`, which changes the workload: they are a *new* row in the results
  table, never a substitution inside an existing row.

---

## 5. Required metadata block

Every timing number carries this block (extends the universal block in
`measurement-taxonomy.md` §2):

```jsonc
"bench": {
  "sub_class": "4-CPU",                      // null | "4-CPU" | "4-GPU"; required for every class-4 row
  "warmup_iters": 10,
  "measured_iters": 50,
  "repeats": 5,
  "discarded": [ { "reason": "warmup", "count": 10 }, { "reason": "thermal", "count": 1 } ],
  "batch_size": 1,
  "sequence_length": 512,
  "generation_length": 128,
  "weights_dtype": "int4",
  "activations_dtype": "bf16",
  "layer_shapes": [[1, 4096, 4096]],
  "timer": "time.perf_counter_ns",
  "sync_method": "cpu_blocking",            // cuda_synchronize | cpu_blocking | ort_run_blocking | response_completion
  "threads": {                               // required whenever any measured path runs on CPU
    "requested": 8,
    "effective": 8,                          // READ BACK, never assumed
    "knobs": ["torch.set_num_threads(8)", "OMP_NUM_THREADS=8"],
    "cpu_model": "AMD Ryzen AI 7 350",
    "physical_cores": 8, "logical_cores": 16
  },
  "statistics": ["p50", "p95", "p99", "median", "min", "max", "iqr"],
  "samples": 250,                            // raw sample count behind the percentiles
  "percentiles_indicative": false,           // true when samples < 100
  "host": {
    "cpu": "AMD Ryzen AI 7 350", "physical_cores": 8, "logical_cores": 16, "ram_bytes": 16233910272,
    "gpu": null, "driver": null, "cuda_runtime": null, "vram_total_bytes": null,
    "power_context": "Windows power plan: Balanced", "thermal_notes": "none observed"
  },
  "backend": { "name": "onnxruntime", "version": "1.30.0", "commit": null,
               "execution_provider": "CPUExecutionProvider",
               "kernel_name": "MatMulNBits", "kernel_domain": "com.microsoft",
               "dequantized_path": false },
  "degraded": false,
  "raw_records_sha256": "...",
  "backend_capability_failures": []
}
```

`dequantized_path` is the single most important field on this host. **Any path that dequantizes to
float before computing MUST set it to `true`, and such a run carries no class-4 claim.**

---

## 6. Fallback = backend-capability failure (mandatory rule)

If the requested backend/format cannot be executed and the code either (a) falls back to another
backend, or (b) dequantizes and computes in float, then:

1. the run is recorded with reason code `backend_capability_failure` and the run is **invalid for
   the requested class**;
2. the manifest lists the failure in `bench.backend_capability_failures` with
   `{ "requested": "...", "actual": "...", "error": "<exception text>" }`;
3. the substitution MUST NOT be silently accepted, averaged in, or described as the requested
   configuration;
4. per `AGENTS.md` §2.2, the code path MUST fail loudly (`NotImplementedError` / explicit config
   error) rather than return a different numeric result.

Worked instance from this host (see `backend-capability.md` §2/§3): bitsandbytes 0.50.2 runs
`Linear4bit` forward **on CPU**. It dequantizes NF4 → fp32 and computes in fp32. A timing of that
path is a valid class-2/class-3 observation about *our* model, and a hard `dequantized_path: true`
case; it is NOT a 4-bit kernel measurement, and it MUST NOT be presented as one.

A second instance: torchao `Int4WeightOnlyConfig` on Windows/CPU raises
`ImportError: Requires mslk >= 1.0.0`. That is the *correct* behaviour (loud failure); the run is
invalid, and it is recorded rather than substituted with `IntxWeightOnlyConfig` output.

---

## 7. Reporting template (results tables)

Kernel/layer-level:

| config | shape (B,S,H) | dtype w/a | kernel | median | p50 | p95 | p99 | samples | speedup vs fp (same host) | peak RSS | degraded |
|---|---|---|---|---|---|---|---|---|---|---|---|

Service-level:

| model@sha8 | server@version | concurrency | arrival | prompt tok | gen tok | reqs | TTFT p50 | latency p50/p95/p99 | throughput | raw sha |

Hard rules for both: one workload per row; the fp baseline row is always present; `not measured` is
written explicitly for every class not measured in that table. Any table containing a class-4 row
MUST state the sub-class and, for `4-CPU`, MUST carry the mandatory scope line of §8.3
("not comparable to published GPU latency or throughput figures").

---

## 8. `4-CPU` measurement protocol

The only class-4 measurement available on this workstation. `AGENTS.md` §5 authorizes class `4-CPU`:
a real CPU kernel executing a low-bit artifact **that SpectraQuant serialized itself**. This section
is additive to §2 and makes such a number publishable; without it, the number is telemetry.

### 8.1 Permitted backends and artifacts

| Backend | Kernel / op | Container | Must also record |
|---|---|---|---|
| ONNX Runtime CPU (`CPUExecutionProvider`) | `MatMulNBits`, domain `com.microsoft` (int4 weight-only) | our own ONNX int4 container | `block_size`, `accuracy_level`, `bits` |
| ONNX Runtime CPU | `MatMulInteger` (int8) | our own ONNX int8 container | dynamic vs static quantization |
| torchao | `IntxWeightOnlyConfig(torch.int4 \| torch.int3, PerGroup(g))` weight-only forward | in-process packed tensor | `g`, `intx_packing_format` |

Anything outside this table — PyTorch/oneDNN dynamic int8, llama.cpp GGUF k-quants, ONNX artifacts
produced by another pipeline — is **engineering telemetry** and MUST NOT be reported under a class-4
label (`AGENTS.md` §5).

### 8.2 Required configuration

1. **Pin threads and read them back.** `torch.set_num_threads(n)`; for ONNX Runtime set
   `SessionOptions.intra_op_num_threads = n` and `inter_op_num_threads = 1` explicitly (the library
   default is not a protocol), and set `OMP_NUM_THREADS` to the same value for the process. Record the
   **effective** value as read back, the CPU model, and physical/logical core counts (`bench.threads`).
2. **Fix the shapes.** `batch_size`, `sequence_length` and `layer_shapes` are identical in the
   quantized and baseline runs. A shape sweep is one row per shape.
3. **Warm up and disclose.** ≥ 10 discarded iterations (first touch, allocator pool growth, thread-pool
   spin-up), listed in `bench.discarded`.
4. **Repeat.** ≥ 5 repeats, every repeat stored, median of repeats as the point estimate.
5. **Same-session fp32 baseline.** Measure the fp32 (or declared bf16) baseline in the **same
   process, same session, same shapes, same thread count** as the `4-CPU` run. A baseline taken from
   another run or from a paper is invalid — this is the rule that makes a speedup a measurement
   rather than a comparison of two unrelated numbers.
6. **Report median and p95.** p95 requires ≥ 100 samples; otherwise report `median`/`min`/`max`/`IQR`
   and set `percentiles_indicative: true`.
7. **Prove no silent upcast.** Set `dequantized_path: false` and record the kernel as actually
   selected at run time (for ONNX Runtime, the graph node/op in the executed model). If the path
   dequantizes or substitutes a backend, §6 applies and the run is **not** a `4-CPU` result.

### 8.3 Mandatory scope line

Every table, figure or manifest row containing a `4-CPU` number MUST carry, verbatim or equivalent:

> CPU-only measurement on `<CPU model>` with `<n>` threads; **not comparable to published GPU latency
> or throughput figures.**

A `4-CPU` number without this line is treated as a violation of `AGENTS.md` §4.9.

### 8.4 What a `4-CPU` result may and may not support

* **May:** assert that a kernel-backed path executed our serialized container at a measured cost;
  rank our own configurations against each other **at equal memory on this machine**; provide the
  local evidence for H5 ("do fake-quantization gains survive conversion to real packed weights and
  supported kernels?") within the CPU scope.
* **May not:** support any Pareto claim against GPU-measured literature numbers; support a headline
  "X× faster" claim without the same-session baseline; be mixed into one table with `4-GPU` or
  class-5 numbers without separating them by hardware column.
