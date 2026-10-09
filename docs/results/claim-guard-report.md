# Claim guard — measurement-integrity scan of the committed prose

Owner: `ClaimGuard` slice. Tool: `src/spectraquant/reporting/claim_guard.py`; CLI:
`spectraquant claim-guard [PATHS...] [--json]`. Gate automated: *no claim may rely only on fake
quantization for a deployment property, and no number may mix measurement classes* (`AGENTS.md`
§4.3, §4.4, §4.5, §4.9, §5). Every count in this report is measurement class **1** (analytical: a
deterministic scan over committed text — nothing was executed to produce a number).

## 1. Command and result

```
$ uv run spectraquant claim-guard
61 documents scanned: 0 violation(s), 123 quote(s), 51 skipped
$ echo $?
0
```

The default scan set is `README.md`, `AGENTS.md`, `docs/**/*.md`, `reports/**/*.md` (the 61
documents present at scan time; this report was written afterwards, and
`tests/unit/test_claim_guard.py::test_committed_repository_has_zero_violations` re-scans the whole
set including it). **Zero violations; 123 matches were classified as quotes of the rules, and 51
as skipped with a reason.**

* **Quotes (123)** — a forbidden phrase inside an explicitly negated/marked context: a prohibition
  bullet, a forbidden-claims table, a scope line ("not comparable to published GPU …"), or an
  attributed literature figure. See §4.
* **Skipped (51)** — a real match whose document-level gate lifts the prohibition:
  `document_declares_real_kernel_backend` (47) and `analytical_estimate_not_measured` (4). See §5.

## 2. What each pattern forbids

| Pattern (forbidden unless marked) | Forbidden claim it matches | Source rule |
|---|---|---|
| `latency_without_kernel_backend` | a latency/throughput/speed figure for a compressed artifact in a document that names no real low-bit kernel | `AGENTS.md` §4.4 |
| `fake_quant_as_storage` | a fake-quantization result presented as low-bit storage, a size, or measured bytes | `AGENTS.md` §4.3 |
| `analytical_as_measured` | a class-1 analytical figure presented as a measured value | `AGENTS.md` §5 |
| `gpu_service_asserted_locally` | a class-4-GPU or class-5 number asserted from local execution | `AGENTS.md` §2.4, §5 |
| `cpu_vs_published_gpu` | a comparison of our CPU numbers with published GPU numbers | `AGENTS.md` §4.4, §4.9 |
| `unqualified_novelty` | an unqualified novelty claim not backed by the frozen verdict wording | `AGENTS.md` §4.9 |

## 3. Counts by pattern and by document

By pattern (all quotes; no violations):

| Pattern | quotes | violations |
|---|---|---|
| `gpu_service_asserted_locally` | 33 | 0 |
| `cpu_vs_published_gpu` | 23 | 0 |
| `latency_without_kernel_backend` | 23 | 0 |
| `fake_quant_as_storage` | 17 | 0 |
| `unqualified_novelty` | 15 | 0 |
| `analytical_as_measured` | 12 | 0 |

By document (the 31 documents with at least one finding; the other 30 documents scanned clean):

| Documents | findings | | Documents | findings |
|---|---|---|---|---|
| `docs/protocols/measurement-taxonomy.md` | 12 | | `AGENTS.md` | 4 |
| `docs/results/verification/snapshot/measurement-taxonomy.md` | 12 | | `docs/results/verification/snapshot/AGENTS.md` | 4 |
| `docs/research/preregistration.md` | 9 | | `docs/research/novelty-risk.md` | 4 |
| `docs/results/verification/snapshot/preregistration.md` | 9 | | `docs/results/verification/snapshot/novelty-risk.md` | 4 |
| `docs/research/charter.md` | 8 | | `docs/results/verification/claim-audit.md` | 3 |
| `docs/results/verification/snapshot/charter.md` | 8 | | `docs/research/upstream-lockfile.md` | 3 |
| `docs/protocols/eval-protocol.md` | 6 | | `docs/protocols/benchmark-protocol.md` | 2 |
| `docs/research/preregistration-amendments.md` | 6 | | `docs/protocols/memory-accounting.md` | 2 |
| `docs/results/verification/snapshot/preregistration-amendments.md` | 6 | | `docs/results/verification/snapshot/memory-accounting.md` | 2 |
| `docs/results/verification/m1-novelty-review.md` | 5 | | `docs/research/risk-register.md` | 2 |
| | | | `reports/paper/paper.md` | 2 |
| | | | 10 further docs (`design-m2-interfaces`, `status`, `literature-review`, `reproduction-plan`, `proxy-validation-report`, `regularizer-report`, their snapshots, `REPRODUCIBILITY`) | 1 each |

## 4. Raw findings

Format: `document:line [pattern] verdict :: context` (context collapsed to one line). All 123 are
quotes; the verdict column is `quote` throughout.

```
AGENTS.md:51 [gpu_service_asserted_locally] quote :: locally** and must be reported as "not measured" locally, never estimated or faked; on the cloud Class 5 and class 4-GPU
AGENTS.md:126 [fake_quant_as_storage] quote :: 3. Fake quantization MUST NOT be reported as low-bit storage or accelerated inference.
AGENTS.md:134 [unqualified_novelty] quote :: 9. No state-of-the-art claims from narrow or incomparable experiments.
AGENTS.md:167 [cpu_vs_published_gpu] quote :: session, and MUST NOT be presented as comparable to published GPU latency/throughput numbers. the CPU model, MUST compar
docs/coordination/design-m2-interfaces.md:53 [analytical_as_measured] quote :: Rules: `theoretical_bits` is class 1 (analytical) and MUST NOT be presented as measured storage.
docs/coordination/status.md:120 [fake_quant_as_storage] quote :: - Every result carries a measurement class; fake quantization is never low-bit storage. ## 9. Discipline reminders
docs/protocols/benchmark-protocol.md:211 [cpu_vs_published_gpu] quote :: ("not comparable to published GPU latency or throughput figures"). MUST state the sub-class and, for `4-CPU`, MUST carry
docs/protocols/benchmark-protocol.md:258 [cpu_vs_published_gpu] quote :: > CPU-only measurement on `<CPU model>` with `<n>` threads; **not comparable to published GPU latency Every table, figur
docs/protocols/eval-protocol.md:35 [gpu_service_asserted_locally] quote :: presented as measured bytes or measured time. Class **4-GPU** and class **5** numbers are unavailable every analytical c
docs/protocols/eval-protocol.md:36 [gpu_service_asserted_locally] quote :: on this host and are reported as "not measured" (`measurement-taxonomy.md` §4); class **4-CPU** *is* Class **4-GPU** and
docs/protocols/eval-protocol.md:40 [cpu_vs_published_gpu] quote :: figure is never presented as comparable to published GPU latency/throughput and supports no latency model are named, an 
docs/protocols/eval-protocol.md:405 [latency_without_kernel_backend] quote :: **Cloud constants** [INFERENCE, class-1 analytical] — effective fp32 throughput for a batch-1 now bound only `LOCAL-FIXT
docs/protocols/eval-protocol.md:476 [gpu_service_asserted_locally] quote :: cell, because evaluation cells do not run locally. No cloud figure is ever presented as a class-4-GPU it is never quoted
docs/protocols/eval-protocol.md:477 [gpu_service_asserted_locally] quote :: or class-5 claim unless it used a real supported kernel under `benchmark-protocol.md`. No cloud figure is ever presented
docs/protocols/measurement-taxonomy.md:32 [analytical_as_measured] quote :: * reporting an analytical number as a **measured** byte count or a measured time;
docs/protocols/measurement-taxonomy.md:54 [fake_quant_as_storage] quote :: * reporting fake-quantized tensors as **low-bit storage** (a float tensor is fp32/fp16 bytes —
docs/protocols/measurement-taxonomy.md:55 [fake_quant_as_storage] quote :: storage claims must come from class 1 or class 3); * reporting fake-quantized tensors as **low-bit storage** (a float te
docs/protocols/measurement-taxonomy.md:82 [fake_quant_as_storage] quote :: * reporting a fake-quantized **float checkpoint** as low-bit storage;
docs/protocols/measurement-taxonomy.md:83 [analytical_as_measured] quote :: * reporting theoretical `bits_per_param` as the measured serialized figure;
docs/protocols/measurement-taxonomy.md:280 [fake_quant_as_storage] quote :: | "Our int4 model is 4× smaller" (measured from an fp16 fake-quant checkpoint) | fake-quant floats are fp16 bytes | "ana
docs/protocols/measurement-taxonomy.md:282 [latency_without_kernel_backend] quote :: | "4-bit inference on CPU is 1.8× faster" (bitsandbytes CPU / any dequant fallback) | dequantize-then-fp32 is not a low-
docs/protocols/measurement-taxonomy.md:283 [latency_without_kernel_backend] quote :: | "We ran our int4 container through the CPU int4 kernel, so we are 1.7× faster than fp32" **without a same-session fp32
docs/protocols/measurement-taxonomy.md:284 [cpu_vs_published_gpu] quote :: | "1.7× faster" on a `4-CPU` table with no scope line | CPU int8/int4 GEMM on an 8-core laptop is not GPU-comparable | a
docs/protocols/measurement-taxonomy.md:284 [latency_without_kernel_backend] quote :: | "1.7× faster" on a `4-CPU` table with no scope line | CPU int8/int4 GEMM on an 8-core laptop is not GPU-comparable | a
docs/protocols/measurement-taxonomy.md:287 [gpu_service_asserted_locally] quote :: | "End-to-end latency 42 ms" (single `model.generate` wall clock) | that is not a service measurement | `4-CPU` per-forw
docs/protocols/measurement-taxonomy.md:287 [latency_without_kernel_backend] quote :: | "End-to-end latency 42 ms" (single `model.generate` wall clock) | that is not a service measurement | `4-CPU` per-forw
docs/protocols/memory-accounting.md:145 [analytical_as_measured] quote :: * **Training state** (class-1 analytical estimate only; never presented as measured):
docs/protocols/memory-accounting.md:294 [analytical_as_measured] quote :: quoting analytical bits as measured bytes, and why §6.1 requires reconciliation. This is exactly why §6 forbids
docs/research/charter.md:94 [gpu_service_asserted_locally] quote :: "not measured", never estimated; it is **claimable only from a validated cloud run** with a real *GPU-kernel / service h
docs/research/charter.md:114 [latency_without_kernel_backend] quote :: gives no basis for latency/throughput claims; class 4-GPU and class 5 are out of scope here **No latency or throughput c
docs/research/charter.md:116 [unqualified_novelty] quote :: 4. **No state-of-the-art claims** from narrow or incomparable runs (`AGENTS.md` §4.9).
docs/research/charter.md:120 [fake_quant_as_storage] quote :: 7. **No fake quantization reported as low-bit storage or speedup** (`AGENTS.md` §4.3).
docs/research/charter.md:140 [cpu_vs_published_gpu] quote :: **not** comparable to published GPU numbers. Class 4-GPU and class 5 are always reported "not measured", MUST compare ag
docs/research/charter.md:140 [gpu_service_asserted_locally] quote :: **not** comparable to published GPU numbers. Class 4-GPU and class 5 are always reported "not measured", MUST compare ag
docs/research/charter.md:141 [gpu_service_asserted_locally] quote :: never estimated — they are **claimable only from a validated cloud run** with a real supported kernel Class 4-GPU and cl
docs/research/charter.md:156 [latency_without_kernel_backend] quote :: unavailable locally, and no throughput claim follows from the CPU kernel path. Latency/throughput claims (class 4-GPU, c
docs/research/literature-review.md:165 [latency_without_kernel_backend] quote :: - **Model scale:** 12B FLUX.1-class diffusion models (reported 3.5× lower memory; 3.0× speedup over a 4-bit weight-only 
docs/research/novelty-risk.md:8 [unqualified_novelty] quote :: component is novel unless this file gives **a cited closest neighbour** and **an explicit No claim in this repository, i
docs/research/novelty-risk.md:9 [unqualified_novelty] quote :: difference statement** that a reader can check. component is novel unless this file gives **a cited closest neighbour** 
docs/research/novelty-risk.md:196 [unqualified_novelty] quote :: Re-check is mandatory before: any submission, any `reports/**` draft that uses the word "novel", and ## 5. Re-check log 
docs/research/novelty-risk.md:197 [unqualified_novelty] quote :: any public README claim. A re-check that changes a verdict MUST be recorded here and, if it changes any submission, any 
docs/research/preregistration-amendments.md:61 [gpu_service_asserted_locally] quote :: measurable** on this workstation and are always reported "not measured" (never estimated). Measurement classes 4 (kernel
docs/research/preregistration-amendments.md:109 [gpu_service_asserted_locally] quote :: half is **not measurable** and is reported "not measured"; H5 is therefore *partially* testable Its kernel-backed (class
docs/research/preregistration-amendments.md:109 [latency_without_kernel_backend] quote :: half is **not measurable** and is reported "not measured"; H5 is therefore *partially* testable Its kernel-backed (class
docs/research/preregistration-amendments.md:155 [cpu_vs_published_gpu] quote :: on the same machine, and MUST NOT be presented as comparable to published GPU latency/throughput thread count and CPU mo
docs/research/preregistration-amendments.md:156 [gpu_service_asserted_locally] quote :: numbers or used to make any latency/throughput claim. **4-GPU unavailable/deferred**. Third-party on the same machine, a
docs/research/preregistration-amendments.md:161 [gpu_service_asserted_locally] quote :: (c) GPU-kernel (class 4-GPU) and service (class 5) halves — **not runnable here, "not measured"**.
docs/research/preregistration.md:39 [cpu_vs_published_gpu] quote :: latency/throughput claim and is never compared to published GPU numbers; class 4-GPU and class 5 are Class 4-CPU carries
docs/research/preregistration.md:40 [gpu_service_asserted_locally] quote :: "not measured" here. class 4-GPU and class 5 are
docs/research/preregistration.md:287 [cpu_vs_published_gpu] quote :: | **CPU-kernel survival:** our own serialized **int4/int8** container ($b \in \{4,8\}$ only, §5) executed by a real **CP
docs/research/preregistration.md:292 [gpu_service_asserted_locally] quote :: | **H5 GPU half:** class 4-GPU kernel-backed inference and class 5 service latency/throughput | **H5** | any | `CLOUD-GP
docs/research/preregistration.md:292 [latency_without_kernel_backend] quote :: | **H5 GPU half:** class 4-GPU kernel-backed inference and class 5 service latency/throughput | **H5** | any | `CLOUD-GP
docs/research/preregistration.md:301 [cpu_vs_published_gpu] quote :: and those results are **not** comparable to published GPU latency/throughput; its class 4-GPU and CPU scope** — packed-s
docs/research/preregistration.md:301 [latency_without_kernel_backend] quote :: and those results are **not** comparable to published GPU latency/throughput; its class 4-GPU and CPU scope** — packed-s
docs/research/preregistration.md:387 [cpu_vs_published_gpu] quote :: | **H5 (fake quantization → real gains)** | Three-part, canonical: (a) **falsified** if the fake-quantization gain inver
docs/research/preregistration.md:387 [gpu_service_asserted_locally] quote :: | **H5 (fake quantization → real gains)** | Three-part, canonical: (a) **falsified** if the fake-quantization gain inver
docs/research/reproduction-plan.md:430 [gpu_service_asserted_locally] quote :: kernel measurement on a self-serialized artifact, not training). A class-4-GPU label requires a available is **class 4-C
docs/research/risk-register.md:18 [fake_quant_as_storage] quote :: | T2 | **Fake-vs-real quantization confusion.** Float execution of quantization numerics (Class 2) is mistaken for low-b
docs/research/risk-register.md:18 [gpu_service_asserted_locally] quote :: | T2 | **Fake-vs-real quantization confusion.** Float execution of quantization numerics (Class 2) is mistaken for low-b
docs/research/upstream-lockfile.md:35 [cpu_vs_published_gpu] quote :: | U1 | `Qualcomm-AI-research/LR-QAT` | arXiv:2406.06385 | `8795afe054cf951b714299e01083a1b354721829` | `BSD-3-Clause-Cle
docs/research/upstream-lockfile.md:125 [cpu_vs_published_gpu] quote :: an fp32 CPU baseline in the same session. Class 4-CPU results are **not** comparable to published GPU only when the clai
docs/research/upstream-lockfile.md:126 [cpu_vs_published_gpu] quote :: numbers and carry **no** latency/throughput claim. Class 4-GPU and class 5 remain deferred Class 4-CPU results are **not
docs/results/proxy-validation-report.md:325 [latency_without_kernel_backend] quote :: * Nothing here is a storage, latency or kernel measurement: class 1/2 only (`AGENTS.md` section 5). The equal-memory gat
docs/results/regularizer-report.md:93 [gpu_service_asserted_locally] quote :: - **not measured here**: class 3 (packed storage), class 4-CPU/4-GPU (kernel execution), class 5 (service). No latency, 
docs/results/verification/claim-audit.md:175 [analytical_as_measured] quote :: - **Analytical not presented as measured:** the frontier's classification labels the equal-memory
docs/results/verification/claim-audit.md:176 [analytical_as_measured] quote :: gate as on **measured class-3** bytes; `regularizer-report.md` §5 states its equal-memory check is on - **Analytical not
docs/results/verification/claim-audit.md:191 [unqualified_novelty] quote :: | Any novelty claim stronger than the frozen verdicts (A narrow/empirical, B weak–moderate, C previously-known)? | **No.
docs/results/verification/m1-novelty-review.md:474 [latency_without_kernel_backend] quote :: `AGENTS.md` §2b rule 8"* with no number. Scaling my measured CPU throughput by a conservative 20–50× GPU-h per model per
docs/results/verification/m1-novelty-review.md:524 [gpu_service_asserted_locally] quote :: 'not measured' here"*), §4 (substrate tags), §5 (H5 arm), §8 (rows 8–9 `LOCAL-CPU-MEASUREMENT`, §1 (*"H5 uses additional
docs/results/verification/m1-novelty-review.md:545 [gpu_service_asserted_locally] quote :: *"Class 4-GPU and class 5 numbers are unavailable here and are reported as 'not measured'; class **Minimal fix:** replac
docs/results/verification/m1-novelty-review.md:698 [fake_quant_as_storage] quote :: consistently, and no text presents a fake-quantized float checkpoint as low-bit storage (the **The measurement-class sep
docs/results/verification/m1-novelty-review.md:699 [fake_quant_as_storage] quote :: `memory-accounting.md` §6 rule and the taxonomy's forbidden-claim table are correct and consistent consistently, and no 
docs/results/verification/snapshot/AGENTS.md:51 [gpu_service_asserted_locally] quote :: locally** and must be reported as "not measured" locally, never estimated or faked; on the cloud Class 5 and class 4-GPU
docs/results/verification/snapshot/AGENTS.md:126 [fake_quant_as_storage] quote :: 3. Fake quantization MUST NOT be reported as low-bit storage or accelerated inference.
docs/results/verification/snapshot/AGENTS.md:134 [unqualified_novelty] quote :: 9. No state-of-the-art claims from narrow or incomparable experiments.
docs/results/verification/snapshot/AGENTS.md:167 [cpu_vs_published_gpu] quote :: session, and MUST NOT be presented as comparable to published GPU latency/throughput numbers. the CPU model, MUST compar
docs/results/verification/snapshot/charter.md:94 [gpu_service_asserted_locally] quote :: "not measured", never estimated; it is **claimable only from a validated cloud run** with a real *GPU-kernel / service h
docs/results/verification/snapshot/charter.md:114 [latency_without_kernel_backend] quote :: gives no basis for latency/throughput claims; class 4-GPU and class 5 are out of scope here **No latency or throughput c
docs/results/verification/snapshot/charter.md:116 [unqualified_novelty] quote :: 4. **No state-of-the-art claims** from narrow or incomparable runs (`AGENTS.md` §4.9).
docs/results/verification/snapshot/charter.md:120 [fake_quant_as_storage] quote :: 7. **No fake quantization reported as low-bit storage or speedup** (`AGENTS.md` §4.3).
docs/results/verification/snapshot/charter.md:140 [cpu_vs_published_gpu] quote :: **not** comparable to published GPU numbers. Class 4-GPU and class 5 are always reported "not measured", MUST compare ag
docs/results/verification/snapshot/charter.md:140 [gpu_service_asserted_locally] quote :: **not** comparable to published GPU numbers. Class 4-GPU and class 5 are always reported "not measured", MUST compare ag
docs/results/verification/snapshot/charter.md:141 [gpu_service_asserted_locally] quote :: never estimated — they are **claimable only from a validated cloud run** with a real supported kernel Class 4-GPU and cl
docs/results/verification/snapshot/charter.md:156 [latency_without_kernel_backend] quote :: unavailable locally, and no throughput claim follows from the CPU kernel path. Latency/throughput claims (class 4-GPU, c
docs/results/verification/snapshot/design-m2-interfaces.md:51 [analytical_as_measured] quote :: Rules: `theoretical_bits` is class 1 (analytical) and MUST NOT be presented as measured storage.
docs/results/verification/snapshot/eval-protocol.md:400 [latency_without_kernel_backend] quote :: **Cloud constants** [INFERENCE, class-1 analytical] — effective fp32 throughput for a batch-1 now bound only `LOCAL-FIXT
docs/results/verification/snapshot/measurement-taxonomy.md:32 [analytical_as_measured] quote :: * reporting an analytical number as a **measured** byte count or a measured time;
docs/results/verification/snapshot/measurement-taxonomy.md:54 [fake_quant_as_storage] quote :: * reporting fake-quantized tensors as **low-bit storage** (a float tensor is fp32/fp16 bytes —
docs/results/verification/snapshot/measurement-taxonomy.md:55 [fake_quant_as_storage] quote :: storage claims must come from class 1 or class 3); * reporting fake-quantized tensors as **low-bit storage** (a float te
docs/results/verification/snapshot/measurement-taxonomy.md:82 [fake_quant_as_storage] quote :: * reporting a fake-quantized **float checkpoint** as low-bit storage;
docs/results/verification/snapshot/measurement-taxonomy.md:83 [analytical_as_measured] quote :: * reporting theoretical `bits_per_param` as the measured serialized figure;
docs/results/verification/snapshot/measurement-taxonomy.md:280 [fake_quant_as_storage] quote :: | "Our int4 model is 4× smaller" (measured from an fp16 fake-quant checkpoint) | fake-quant floats are fp16 bytes | "ana
docs/results/verification/snapshot/measurement-taxonomy.md:282 [latency_without_kernel_backend] quote :: | "4-bit inference on CPU is 1.8× faster" (bitsandbytes CPU / any dequant fallback) | dequantize-then-fp32 is not a low-
docs/results/verification/snapshot/measurement-taxonomy.md:283 [latency_without_kernel_backend] quote :: | "We ran our int4 container through the CPU int4 kernel, so we are 1.7× faster than fp32" **without a same-session fp32
docs/results/verification/snapshot/measurement-taxonomy.md:284 [cpu_vs_published_gpu] quote :: | "1.7× faster" on a `4-CPU` table with no scope line | CPU int8/int4 GEMM on an 8-core laptop is not GPU-comparable | a
docs/results/verification/snapshot/measurement-taxonomy.md:284 [latency_without_kernel_backend] quote :: | "1.7× faster" on a `4-CPU` table with no scope line | CPU int8/int4 GEMM on an 8-core laptop is not GPU-comparable | a
docs/results/verification/snapshot/measurement-taxonomy.md:287 [gpu_service_asserted_locally] quote :: | "End-to-end latency 42 ms" (single `model.generate` wall clock) | that is not a service measurement | `4-CPU` per-forw
docs/results/verification/snapshot/measurement-taxonomy.md:287 [latency_without_kernel_backend] quote :: | "End-to-end latency 42 ms" (single `model.generate` wall clock) | that is not a service measurement | `4-CPU` per-forw
docs/results/verification/snapshot/memory-accounting.md:145 [analytical_as_measured] quote :: * **Training state** (class-1 analytical estimate only; never presented as measured):
docs/results/verification/snapshot/memory-accounting.md:294 [analytical_as_measured] quote :: quoting analytical bits as measured bytes, and why §6.1 requires reconciliation. This is exactly why §6 forbids
docs/results/verification/snapshot/novelty-risk.md:8 [unqualified_novelty] quote :: component is novel unless this file gives **a cited closest neighbour** and **an explicit No claim in this repository, i
docs/results/verification/snapshot/novelty-risk.md:9 [unqualified_novelty] quote :: difference statement** that a reader can check. component is novel unless this file gives **a cited closest neighbour** 
docs/results/verification/snapshot/novelty-risk.md:153 [unqualified_novelty] quote :: Re-check is mandatory before: any submission, any `reports/**` draft that uses the word "novel", and ## 5. Re-check log 
docs/results/verification/snapshot/novelty-risk.md:154 [unqualified_novelty] quote :: any public README claim. A re-check that changes a verdict MUST be recorded here and, if it changes any submission, any 
docs/results/verification/snapshot/preregistration-amendments.md:61 [gpu_service_asserted_locally] quote :: measurable** on this workstation and are always reported "not measured" (never estimated). Measurement classes 4 (kernel
docs/results/verification/snapshot/preregistration-amendments.md:109 [gpu_service_asserted_locally] quote :: half is **not measurable** and is reported "not measured"; H5 is therefore *partially* testable Its kernel-backed (class
docs/results/verification/snapshot/preregistration-amendments.md:109 [latency_without_kernel_backend] quote :: half is **not measurable** and is reported "not measured"; H5 is therefore *partially* testable Its kernel-backed (class
docs/results/verification/snapshot/preregistration-amendments.md:155 [cpu_vs_published_gpu] quote :: on the same machine, and MUST NOT be presented as comparable to published GPU latency/throughput thread count and CPU mo
docs/results/verification/snapshot/preregistration-amendments.md:156 [gpu_service_asserted_locally] quote :: numbers or used to make any latency/throughput claim. **4-GPU unavailable/deferred**. Third-party on the same machine, a
docs/results/verification/snapshot/preregistration-amendments.md:161 [gpu_service_asserted_locally] quote :: (c) GPU-kernel (class 4-GPU) and service (class 5) halves — **not runnable here, "not measured"**.
docs/results/verification/snapshot/preregistration.md:39 [cpu_vs_published_gpu] quote :: latency/throughput claim and is never compared to published GPU numbers; class 4-GPU and class 5 are Class 4-CPU carries
docs/results/verification/snapshot/preregistration.md:40 [gpu_service_asserted_locally] quote :: "not measured" here. class 4-GPU and class 5 are
docs/results/verification/snapshot/preregistration.md:189 [cpu_vs_published_gpu] quote :: | **CPU-kernel survival:** our own serialized int4/int8 container executed by a real **CPU** low-bit kernel (ONNX Runtim
docs/results/verification/snapshot/preregistration.md:194 [gpu_service_asserted_locally] quote :: | **H5 GPU half:** class 4-GPU kernel-backed inference and class 5 service latency/throughput | **H5** | any | `CLOUD-GP
docs/results/verification/snapshot/preregistration.md:194 [latency_without_kernel_backend] quote :: | **H5 GPU half:** class 4-GPU kernel-backed inference and class 5 service latency/throughput | **H5** | any | `CLOUD-GP
docs/results/verification/snapshot/preregistration.md:203 [cpu_vs_published_gpu] quote :: and those results are **not** comparable to published GPU latency/throughput; its class 4-GPU and CPU scope** — packed-s
docs/results/verification/snapshot/preregistration.md:203 [latency_without_kernel_backend] quote :: and those results are **not** comparable to published GPU latency/throughput; its class 4-GPU and CPU scope** — packed-s
docs/results/verification/snapshot/preregistration.md:265 [cpu_vs_published_gpu] quote :: | **H5 (fake quantization → real gains)** | Three-part, canonical: (a) **falsified** if the fake-quantization gain inver
docs/results/verification/snapshot/preregistration.md:265 [gpu_service_asserted_locally] quote :: | **H5 (fake quantization → real gains)** | Three-part, canonical: (a) **falsified** if the fake-quantization gain inver
docs/results/verification/snapshot/status.md:92 [fake_quant_as_storage] quote :: - Every result carries a measurement class; fake quantization is never low-bit storage. ## 9. Discipline reminders
reports/REPRODUCIBILITY.md:129 [cpu_vs_published_gpu] quote :: model evaluation) but is **not** comparable to published GPU latency or throughput figures, and makes The class-4-CPU me
reports/paper/paper.md:7 [unqualified_novelty] quote :: Nothing here is a state-of-the-art claim. It contains no cloud,
reports/paper/paper.md:422 [unqualified_novelty] quote :: "no state-of-the-art claim" rule are binding here. No dual-use code (e.g. kernels, datasets) is this is why measurement-
```

## 5. Skipped matches and why

| Skip reason | count | meaning |
|---|---|---|
| `document_declares_real_kernel_backend` | 47 | a performance figure in a document that names a real low-bit kernel (ONNX Runtime `MatMulNBits`/`MatMulInteger`, torchao intx, class 4-CPU) — the prohibition lifted by `AGENTS.md` §4.4 |
| `analytical_estimate_not_measured` | 4 | a figure explicitly marked `analytical` / `INFERENCE` / class-1, so it is an estimate, not a measured claim |

(47 + 4 = 51, matching §1.) A third reason exists in the code — `classes_explicitly_labelled`, which
skips a sentence that names its classes (`class 1` … `class 3`) and so separates rather than mixes
them — but no committed sentence currently hits it; `tests/unit/test_claim_guard.py` pins it.

A skipped match is neither a quote nor a violation: it is a real pattern hit that a document-level
gate (or an explicit label) permits. They are counted, with their reason, so a reviewer can see
exactly what the gate waved through.

## 6. Crafted-violation and crafted-quote demos

A violating sentence injected into a throwaway file is caught and exits non-zero:

```
$ printf 'Our int4 model runs at 2.5 ms per token on this laptop.\n' > .cg-demo/violation.md
$ uv run spectraquant claim-guard .cg-demo/violation.md
┌───────────────────┬──────┬───────────────────┬───────────┬──────────────────┐
│ document          │ line │ pattern           │ verdict   │ context          │
├───────────────────┼──────┼───────────────────┼───────────┼──────────────────┤
│ .cg-demo/violati… │ 1    │ latency_without_… │ violation │ Our int4 model   │
│                   │      │                   │           │ runs at 2.5 ms   │
│                   │      │                   │           │ per token on     │
│                   │      │                   │           │ this laptop.     │
└───────────────────┴──────┴───────────────────┴───────────┴──────────────────┘
1 documents scanned: 1 violation(s), 0 quote(s), 0 skipped
exit=1
```

The same phrase inside a prohibition is a quote, and the gate passes:

```
$ printf 'We never claim our int4 model is 2.5 ms per token.\n' > .cg-demo/quote.md
$ uv run spectraquant claim-guard .cg-demo/quote.md
┌───────────────────┬──────┬────────────────────┬─────────┬───────────────────┐
│ document          │ line │ pattern            │ verdict │ context           │
├───────────────────┼──────┼────────────────────┼─────────┼───────────────────┤
│ .cg-demo/quote.md │ 1    │ latency_without_k… │ quote   │ We never claim    │
│                   │      │                    │         │ our int4 model is │
│                   │      │                    │         │ 2.5 ms per token. │
└───────────────────┴──────┴────────────────────┴─────────┴───────────────────┘
1 documents scanned: 0 violation(s), 1 quote(s), 0 skipped
exit=0
```

## 7. Zero-violation test

```
$ uv run pytest tests/unit/test_claim_guard.py -q
24 passed in 2.91s
```

`test_committed_repository_has_zero_violations` scans `default_document_paths()` (≥ 50 documents)
and asserts `report.violations == ()`, and asserts the scan produced findings (a broken pattern set
that matched nothing would make the gate vacuous). The other tests pin one positive and one
negative case per pattern, the forbidden-claims-table and prohibition-list rules, determinism,
order-independence, and the CLI exit-code contract.

## 8. Limits (what the scanner cannot catch)

The scanner is a lexical gate with a negation heuristic, not a proof. Its honest limits:

1. **A wrong number that is correctly labelled passes.** A figure that carries the right class,
   names a real kernel and sits under no forbidden context is invisible here. The scan cannot check
   the *value* against the artifact — that is `ClaimAudit`'s re-derivation work, and a wrong-but-
   labelled number is exactly the case this tool does not cover.
2. **A stray negation hides a real violation.** The context heuristic treats *any* `not`/`no`/
   `never`/`without` in the match's sentence (or its block label) as a prohibition, so a genuine
   claim that happens to sit near an unrelated negation is misread as a quote. The false-negative
   direction is the safe one, but it is a false negative.
3. **Fenced code blocks are not scanned.** Raw transcripts inside ``` fences are skipped, so a
   violation would have to appear in prose or a table to be caught. This is what keeps this report's
   own raw-output blocks from re-triggering the gate.
4. **Two-part claims split across sentences are missed.** The context is the line plus its wrapped
   continuation; a class-mixing claim whose halves are in different paragraphs is not matched.
5. **Semantic co-occurrence is required, not meaning.** `latency_without_kernel_backend` needs a
   low-bit/inference cue in the same context; a bare "the model took 40 ms" with no such cue is not
   flagged (the cue-requirement is what keeps solver wall-clock and host GEMM constants out).
6. **No understanding of the frozen verdict wording beyond a regex.** `unqualified_novelty` passes
   when the *document* contains the frozen verdict strings; a document could in principle carry
   them and still overclaim elsewhere in a way this pattern does not model.
7. **No positive false-positives are known against the committed repository**, but the design means
   a genuinely new claim placed under a `Forbidden:` label would be reported as a quote.
