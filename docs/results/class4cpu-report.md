# Class-4-CPU kernel fixture report (Milestone 2/6 evidence)

Owner: class-4-CPU slice. Evidence: `scripts/experiments/class4cpu_measurement.py` and the committed
artifact `artifacts/sample-results/class4cpu/class4cpu.json` (regenerate with
`uv run python scripts/experiments/class4cpu_measurement.py`).

This closes the deferred freeze item (`docs/research/preregistration.md` §13: *"Class-4-CPU ONNX
fixture (int4/int8 container written by us, its runner, the same-session fp32 baseline)"*). It shows
that the H5 CPU-kernel cell is **runnable and measurable** with real kernels executing containers
SpectraQuant serialized itself. It is a fixture, not the H5 result: the quality half and the GPU half
of H5 remain deferred (§6).

## 1. Setup

* **Fixture** (`backend-capability.md` §2.4c / `memory-accounting.md` §7.5.1): one weight-only linear
  map, `W ~ N(0, 0.1²)` i.i.d. with shape `(K, N) = (64, 32)` (`|W|.max = 0.3171`), exported as the
  ONNX `MatMul(X, W)` orientation with a symbolic batch dimension. Inputs `X ~ N(0, 1)` i.i.d.,
  `(4, 64)` and `(512, 64)`.
* **Containers (written by us)**: int4 weight-only `MatMulNBits` (domain `com.microsoft`,
  `block_size = 32`, symmetric, `accuracy_level = 4`, IR version 10) and dynamic int8
  `MatMulInteger`; plus the fp32 `MatMul` reference graph. All exported by
  `spectraquant.quantization.onnx_export`.
* **Runner**: `spectraquant.benchmarking.kernel_cpu`, protocol §8 — threads pinned and read back
  (`torch.set_num_threads`, `OMP_NUM_THREADS`, ORT `intra_op_num_threads = 8` / `inter_op_num_threads
  = 1`), 10 disclosed warm-ups, 5 repeats, median + p95, and an fp32 CPU baseline measured in the
  **same process, same session, identical inputs and identical thread configuration**
  (`inputs_identical = true`, `threads_identical = true` in the artifact).
* **Host**: AMD Ryzen AI 7 350 w/ Radeon 860M, 8 physical / 16 logical cores, 16 233 910 272 B RAM;
  onnxruntime 1.30.0; provider `CPUExecutionProvider`; `dequantized_path = false`. Substrate
  `LOCAL-FIXTURE`, measurement class **4-CPU**. The ONNX containers are git-ignored (`*.onnx`); the
  artifact records the per-file byte listing and the SHA-256 of each container.

## 2. Measured container bytes (class 3)

| backend | op / domain | graph B | sidecar B | payload B | scales B | zero-point B | overhead B | total B | bits/param measured |
|---|---|---|---|---|---|---|---|---|---|
| fp32 `MatMul` | `MatMul` | 8 321 | 0 | 8 192 | 0 | 0 | 129 | **8 321** | 32.504 |
| int4 `MatMulNBits` | `MatMulNBits` / `com.microsoft` | 638 | 1 024 | 1 024 | 256 | 0 | 382 | **1 662** | 6.492 |
| int8 dynamic | `MatMulInteger` | 2 727 | 0 | 2 048 | 4 | 1 | 674 | **2 727** | 10.652 |

Every file the save produced is summed (graph + `.data` sidecar). The int4 payload `W_Q4` = 1 024 B
(= 2 048 four-bit values) and fp32 `W_scales` = 256 B (64 blocks) match `memory-accounting.md` §7.5
exactly; the container term is measured per artifact, never assumed.

**Reconciliation against `spectraquant.quantization.accounting`** (class 1 vs class 3):

| artifact | analytical B (class 1) | measured B (class 3) | residual B | of which scale-dtype | of which container |
|---|---|---|---|---|---|
| int4 | 1 152 | 1 662 | **510** | 128 | 382 |
| int8 | 2 053 | 2 727 | **674** | 0 | 674 |
| fp32 | 8 192 | 8 321 | **129** | 0 | 129 |

The residual is **expected non-zero and is decomposed, not forced to zero**: the analytical model
contains no graph/file overhead, and it stores per-group scales at fp16 while ONNX Runtime writes
fp32 (a 128 B difference at 64 blocks). The measured int4 container overhead is 382 B on a 1 280 B
tensor payload, i.e. the same order as the 221–363 B spread documented for this fixture
(`memory-accounting.md` §6.1, §7.5) — the container term is a property of the serializer invocation.

## 3. Numerical error vs the fp32 reference (class 4-CPU)

Reference: `numpy float32 X @ W` on the fp32 ONNX weight `W (K, N)`; the ORT fp32 session on the same
graph is checked against it and agrees **bit-for-bit** (`max|diff| = 0.000e+00`) at both shapes.
The weight distribution (`W ~ N(0, 0.1²)` i.i.d., `(64, 32)`) and the input distribution
(`X ~ N(0, 1)` i.i.d., `(batch, 64)`) are stated with every figure, and the absolute error is
**scale-dependent** — the relative error is the scale-invariant number
(`backend-capability.md` §2.4c).

| `X` shape | backend | `max\|y−ref\|` | `mean\|y−ref\|` | `\|ref\|.max` | relative max |
|---|---|---|---|---|---|
| (4, 64) | fp32 baseline | 0.000000 | 0.000000 | 2.6457 | 0.000000 |
| (4, 64) | int4 `MatMulNBits` | **0.204059** | 0.0523427 | 2.6457 | **0.077129** |
| (4, 64) | int8 `MatMulInteger` | **0.0207714** | 0.00625642 | 2.6457 | 0.00785104 |
| (512, 64) | fp32 baseline | 0.000000 | 0.000000 | 2.9404 | 0.000000 |
| (512, 64) | int4 `MatMulNBits` | **0.277936** | 0.053518 | 2.9404 | 0.0945227 |
| (512, 64) | int8 `MatMulInteger` | **0.0372313** | 0.00741851 | 2.9404 | 0.0126619 |

The `(4, 64)` int4 figure reproduces the published `0.2041` / relative `0.0771` and the int8
`0.0207714` (published `0.0208`) from `backend-capability.md` §2.4c — independent confirmation that
the containers execute through the real kernels. The larger absolute errors at `(512, 64)` are a max
over 8× more outputs of the same distribution, as documented for the int8 pair.

## 4. Latency table (class 4-CPU)

`timer = time.perf_counter_ns`, `sync_method = ort_run_blocking`, 10 warm-ups, 5 repeats. p95 is
**indicative** (5 samples < 100; protocol §2.7) — read the median and the min/max/IQR.

> CPU-only measurement on AMD Ryzen AI 7 350 w/ Radeon 860M with 8 threads; **not comparable to
> published GPU latency or throughput figures.**

| `X` shape | backend | median ns | p95 ns | min ns | max ns | IQR ns | samples | threads |
|---|---|---|---|---|---|---|---|---|
| (4, 64) | fp32 baseline | 8 900 | 9 820 | 8 700 | 10 000 | 400 | 5 | 8 |
| (4, 64) | int4 `MatMulNBits` | 9 400 | 9 780 | 9 300 | 9 800 | 400 | 5 | 8 |
| (4, 64) | int8 `MatMulInteger` | 9 400 | 9 660 | 9 000 | 9 700 | 400 | 5 | 8 |
| (512, 64) | fp32 baseline | 13 900 | 14 420 | 13 300 | 14 500 | 600 | 5 | 8 |
| (512, 64) | int4 `MatMulNBits` | 26 800 | 27 300 | 26 300 | 27 300 | 800 | 5 | 8 |
| (512, 64) | int8 `MatMulInteger` | 13 700 | 13 960 | 13 600 | 14 000 | 100 | 5 | 8 |

At these shapes the absolute latency is dominated by ONNX Runtime session-call overhead, so the
latency table's value is **protocol compliance**, not a performance claim: it demonstrates that the
kernels execute our containers under a controlled, disclosed protocol with a same-session baseline.
No speedup is claimed or implied.

## 5. Claim rules honoured

* Class 4-CPU naming: backend (`onnxruntime` 1.30.0), kernel/op (`MatMulNBits` / `MatMulInteger`),
  container format (our own ONNX int4/int8), thread count (8, read back) and CPU model are all in the
  artifact.
* Same-session fp32 CPU baseline with identical inputs and threads (verified by input SHA-256 and the
  thread block).
* No silent fallback: the provider list must equal the requested CPU provider and the expected
  kernel op must be present, else a `BackendCapabilityError` is raised (protocol §6, §8.2.7).
* `dequantized_path = false`; the executed graph nodes are recorded.
* The mandatory scope line is present verbatim in the artifact and above.
* Every error figure carries its `W`/`X` distributions, its reference definition and the
  scale-dependence caveat.

## 6. What this result does NOT support

* **No GPU comparison.** It is a CPU-only, single-machine number and may not be placed in a table,
  figure or claim alongside published GPU latency/throughput figures (protocol §8.4).
* **No H5 confirmation by itself.** H5 has three parts (`preregistration.md` §11.1): (a) packed-storage
  survival (class 2 → class 3), (b) CPU-kernel survival (class 4-CPU, this fixture builds the tool),
  (c) GPU scope. This report establishes only that the class-4-CPU cell **runs**; it does not test
  whether a fake-quantization quality gain survives the packed round-trip, and it makes no GPU claim.
  H5 is therefore reported **partially supported / inconclusive at GPU scope**, never confirmed.
* **No performance claim.** Absolute latencies here are overhead-dominated at fixture scale; no
  speedup against the baseline is asserted.
* **Fixture scope.** One weight matrix, two batch sizes, one seed; not the Tier-1 model. It is
  measurement class 4-CPU and 3 only — no class-2 quality number is reported here.
