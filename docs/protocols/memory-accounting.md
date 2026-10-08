# Memory accounting — byte-exact rules

**Status:** normative. Expands `AGENTS.md` §4.2 (three distinct memory numbers) and §4.13
(approximations must be labelled approximate) into arithmetic that code and tests can be held to.
**Owner:** D-lite. **Created:** 2026-10-08. **Verified arithmetic:** `docs/protocols/` worked
examples in §7 are machine-checked (see §7.6 reproduction).

Every number produced by this project that ends in `_bytes` or `bits_per_param` MUST be derivable
from this document. If a formula here disagrees with an implementation, the implementation is wrong
until this document is amended.

---

## 1. The three memory numbers (exact naming)

The project reports **three** memory numbers and never conflates them (`AGENTS.md` §4.2). The names
below are the **only** accepted keys. Snake-case, byte units, SI-free (bytes, not MB/GB).

| Published key | Meaning | How it is obtained | Class (see `measurement-taxonomy.md`) |
|---|---|---|---|
| `mem.training.peak_rss_bytes` | Peak process resident set during a training step (fwd+bwd+optimizer step) | measured from a real run | measured (class-2 run) — MAY NOT be derived from shapes |
| `mem.training.analytical_estimate_bytes` | Shape-derived upper bound on the same quantity | §5 formula | class 1 (analytical) |
| `mem.checkpoint.payload_bytes` | Bytes of quantized/factorized **tensor payload** written to disk | `os.path.getsize` summed over payload files | class 3 (measured) or class 1 (analytical) |
| `mem.checkpoint.metadata_bytes` | Bytes of scales, zero-points, headers, graph/protobuf, config, tokenizer | `os.path.getsize` of the remaining artifact files | class 3 |
| `mem.checkpoint.total_bytes` | `payload_bytes + metadata_bytes` | measured, or class-1 estimate | class 3 |
| `mem.deployed.stored_bytes` | Bytes of the representation that is *stored* for inference (may equal `checkpoint.payload_bytes`) | measured file bytes | class 3 |
| `mem.deployed.resident_bytes` | Bytes resident in the process once the model is loaded and ready to serve, **including** runtime-expanded buffers | measured (RSS delta) or analytical | class 3 + measured |

**Naming rule.** Any ratio published as a compression factor MUST state which numerator and which
denominator it used, e.g. `compression_ratio(checkpoint.total_bytes / fp16_total_bytes)`. A bare
"4× compression" is a report-contract violation.

**`resident_bytes` rule.** Dequantize-then-compute backends (bitsandbytes CPU 4-bit, and any
"dequantized fallback") hold a full-precision working copy per layer at runtime. `resident_bytes`
MUST therefore be reported separately from `stored_bytes`; a 4-bit *stored* model that materialises
fp32 weights at load has `resident_bytes ≈ 2× stored_bytes` for the weight tensors (`fp32` vs
`int4`), not `1×`. Never publish `stored_bytes` under the name `resident_bytes`.

---

## 2. Symbols

| Symbol | Meaning |
|---|---|
| `out`, `in` | output / input feature dimensions of a linear weight `W ∈ R^{out×in}` |
| `N = out · in` | number of weight elements |
| `b_w` | weight bit width (`8`, `4`, `3`, `…`) |
| `g` | group size along the quantized axis (number of elements sharing one scale) |
| `s_b` | bytes per scale element (`4` for fp32, `2` for fp16/bf16) |
| `z_b` | bytes per zero-point element (`0` symmetric, `1` if stored as int8, `0.5` if packed 4-bit) |
| `P` | bytes per element of a float tensor (`4` fp32, `2` fp16/bf16) |

**Grouping rule.** Groups are contiguous along one declared axis. Let `A` be the number of elements
along that axis and `axis_len` the length of the other axis:

```
n_blocks = axis_len · ceil(A / g)          # ceil, because a trailing short group still needs a scale
```

For a weight quantized along the **input** axis with group size `g`: `n_blocks = out · ceil(in / g)`.
For per-channel (row-wise) quantization: `n_blocks = out` (equivalently `g = in`).
For per-tensor: `n_blocks = 1`.

---

## 3. Weight payload and metadata

```
payload_bytes        = ceil(N · b_w / 8)                       # ideal packing
payload_bytes_padded = bytes_per_row_padded · out             # real packers pad per row
scales_bytes         = n_blocks · s_b
zero_points_bytes    = n_blocks · z_b
tensor_total_bytes   = payload_bytes_padded + scales_bytes + zero_points_bytes
bits_per_param       = 8 · tensor_total_bytes / N             # NOT b_w
```

### 3.1 Row alignment / packing overhead

Most packers store a row as a whole number of 32-bit words (four int8 or eight int4 values).
For `w` bits and a row length `in`:

```
bytes_per_row_padded = ceil(in · w / 32) · 4          # 4-byte alignment
alignment_bytes      = payload_bytes_padded − ceil(N · w / 8)
```

Alignment is zero for `in` a multiple of 8 at 4 bits, or a multiple of 4 at 8 bits. Worked example
in §7.1b: `in = 5`, 4-bit, `out = 4` pads `10 B → 16 B` (60 % overhead).

### 3.2 Asymmetric quantization

Asymmetric requires a zero-point per block. Zero-points are NOT free and MUST be counted, at their
declared storage width — the width is part of the claim. Two legitimate conventions:

* `z_b = 1` — zero-point stored as int8 (the safe default; what ONNX Runtime's dynamic int8 path
  does: measured `W_zero_point` dtype `int8`);
* `z_b = 0.5` — zero-point packed to 4 bits alongside the weights (cheaper, requires a packer that
  supports it; MUST be verified against the serialized file, not assumed).

### 3.3 Nested / double quantization

If scales themselves are quantized (e.g. QLoRA double quant, `bnb_4bit_use_double_quant=True`), the
scale term becomes a second-level payload plus second-level scales:

```
scale_payload = ceil(n_blocks · b_s / 8)          # b_s = 8 for uint8 scale codebooks
scale_scales  = ceil(n_blocks / g2) · s_b2
scales_bytes  = scale_payload + scale_scales
```

Report the effective `bits_per_param`; do not reuse the plain formula.

---

## 4. Low-rank factor bytes

A rank-`r` factorization replaces `W ∈ R^{out×in}` with `W ≈ B·A`, `A ∈ R^{r×in}`, `B ∈ R^{out×r}`:

```
lr_params      = r · in + out · r = r · (in + out)
lr_bytes       = lr_params · P                      # float factors
lr_bytes_intk  = ceil(lr_params · b / 8) + (blocks_A + blocks_B) · s_b + zp_bytes
   blocks_A    = r   · ceil(in / g)                 # A is r × in
   blocks_B    = out · ceil(r  / g)                 # B is out × r
```

**Important:** `blocks_B` uses `r`, not `in`. With `r < g` every row of `B` is one block.

Low-rank bytes scale as `r(in+out)` while the dense tensor is `in·out`; the reduction is only real
for `r < in·out/(in+out)` — i.e. `r < in/2` when `in = out`. Always report `r`, `in`, `out` and the
factor dtype together, because `lr_bytes` at `fp32` is **2×** its value at `fp16`.

---

## 5. Outlier / residual channels and training-time state

* **Outlier channels kept in fp16** (LLM.int8-style mixed-precision): if `k` feature channels are
  excluded from quantization and stored at `P` bytes instead of `b_w` bits:

  ```
  outlier_bytes = k · axis_len · P
  mixed_total   = (N − k·axis_len)·b_w/8 + outlier_bytes + scales_bytes
  ```

* **Training state** (class-1 analytical estimate only; never presented as measured):

  ```
  mem.training.analytical_estimate_bytes ≈
        bytes(weights in compute dtype)
      + bytes(gradients, same dtype; 0 for frozen/quantized base weights)
      + bytes(optimizer state per trainable tensor)     # Adam fp32: 8 B/param (m, v)
      + activation_estimate_bytes                       # declare the formula used
      + k_bit_scratch                                   # dequant buffers, per layer
  ```

  For 4-bit frozen base + adapters, quantized base weights are parameters of the *stored* model, so
  they appear as `mem.deployed.stored_bytes`, and the fp32 **working copy** appears in
  `mem.deployed.resident_bytes` and in training RSS — never double-count them in one number.

---

## 6. Analytical bits vs measured serialized bytes

`AGENTS.md` §4.13 forbids presenting an estimate as exact. The two are different quantities:

| | `bits_per_param_analytical` | `bits_per_param_measured` |
|---|---|---|
| Source | §3 formula from shapes | `8 · mem.checkpoint.total_bytes / N` |
| Includes graph/protobuf/header overhead? | no | **yes** |
| Labelled | `analytical` | `measured` |
| Legal phrase | "the analytical cost is 4.125 bits/param" | "the serialized artifact is 6.352 bits/param" |

Rule: when both exist, publish the pair and their difference, e.g.
`overhead_frac = measured/analytical − 1`. For small tensors this overhead dominates (§7.5: nominal
4.0 → measured **5.86–6.42 bits/param** depending on the serializer invocation, for a 2 048-element
matrix — see the container reconciliation rule in §6.1). Analytical numbers MUST NOT be reported as
measured bytes anywhere in a results table.

---

## 7. Worked examples (arithmetic shown; machine-verified)

All examples below were computed and checked; reproduction command in §7.6. Later unit tests MUST
assert these exact values.

### 7.1 Tiny tensor, `out = 4`, `in = 8`, `N = 32`

Baselines: `fp32 = 32·4 = 128 B`, `fp16 = 32·2 = 64 B`.

| Configuration | Blocks | Payload | Scales | ZP | Total | bits/param |
|---|---|---|---|---|---|---|
| int8 per-tensor, fp32 scale | 1 | `32·8/8 = 32` | `1·4 = 4` | 0 | **36 B** | `36·8/32 = 9.0000` |
| int8 group=4, fp16 scale | `4·(8/4)=8` | 32 | `8·2 = 16` | 0 | **48 B** | `48·8/32 = 12.0000` |
| int8 group=8, fp16 scale | `4·1 = 4` | 32 | `4·2 = 8` | 0 | **40 B** | `40·8/32 = 10.0000` |
| int8 group=8, fp16 scale, int8 zp | 4 | 32 | 8 | 4 | **44 B** | `44·8/32 = 11.0000` |
| int4 per-tensor, fp32 scale | 1 | `32·4/8 = 16` | 4 | 0 | **20 B** | `20·8/32 = 5.0000` |
| int4 group=4, fp16 scale | 8 | 16 | 16 | 0 | **32 B** | `32·8/32 = 8.0000` |
| int4 group=8, fp16 scale | 4 | 16 | 8 | 0 | **24 B** | `24·8/32 = 6.0000` |
| int4 group=8, fp16 scale, 4-bit zp | 4 | 16 | 8 | `4·0.5=2` | **26 B** | `26·8/32 = 6.5000` |

Read the table as the core lesson: **grouping can cost more than the bit-width saves.** At `N = 32`,
int8 group=4 (12.0 bits/param) is worse than plain fp32 (32 bits/param) is not, but it is worse than
int8 per-tensor (9.0) — and int4 group=4 (8.0 bits/param) is *no better* than int8 per-tensor.

### 7.1b Row alignment, `in = 5`, 4-bit, `out = 4`

```
row bits            = 5 · 4 = 20
unaligned row bytes = 20/8 = 2.5      → unaligned payload = 4 · 2.5 = 10 B
padded row bytes    = ceil(20/32)·4 = 1·4 = 4 B
aligned payload     = 4 · 4 = 16 B        (alignment overhead = 6 B, +60 %)
```

### 7.2 Transformer projection, `out = in = 4096`, `N = 16 777 216`

`fp16 baseline = 16 777 216 · 2 = 33 554 432 B = 32 MiB`.

| Configuration | Payload | Scales | ZP | Total | bits/param | ratio to fp16 |
|---|---|---|---|---|---|---|
| int8 symmetric per-channel, fp32 scales | 16 777 216 | `4096·4 = 16 384` | 0 | 16 793 600 | 8.0078 | 0.500488 |
| int4 group=128, fp16 scales (symmetric) | 8 388 608 | `131 072·2 = 262 144` | 0 | 8 650 752 | **4.1250** | **0.257812** |
| int4 group=128, fp16 scales, 4-bit zp | 8 388 608 | 262 144 | `131 072·0.5 = 65 536` | 8 716 288 | 4.1562 | 0.259766 |
| int4 group=128, fp16 scales, int8 zp | 8 388 608 | 262 144 | 131 072 | 8 781 824 | 4.1875 | 0.261719 |
| *uniform int4, no metadata (ideal)* | 8 388 608 | — | — | 8 388 608 | 4.0000 | 0.250000 |

`n_blocks = 4096 · ceil(4096/128) = 4096 · 32 = 131 072`.
Group-wise int4 with fp16 scales costs **+3.125 %** over the ideal 4.0000 bits/param.

### 7.3 Low-rank factors, `out = in = 4096`

`lr_params(r) = r·(4096+4096) = 8192·r`.

| `r` | `lr_params` | fp16 bytes | fp32 bytes | ratio to fp16 dense (33 554 432 B) |
|---|---|---|---|---|
| 16 | 131 072 | 262 144 (0.25 MiB) | 524 288 | 0.007812 |
| 64 | 524 288 | 1 048 576 (1.00 MiB) | 2 097 152 | 0.031250 |
| 128 | 1 048 576 | 2 097 152 (2.00 MiB) | 4 194 304 | 0.062500 |
| 256 | 2 097 152 | 4 194 304 (4.00 MiB) | 8 388 608 | 0.125000 |

Int4 factors at `r = 64`, group 128, fp16 scales:

```
payload   = ceil(524 288 · 4 / 8)                 = 262 144 B
blocks_A  = 64  · ceil(4096/128) = 64  · 32 =  2 048
blocks_B  = 4096 · ceil(64/128)  = 4096 ·  1 =  4 096      # r=64 < g=128 ⇒ one block per row
scales    = (2 048 + 4 096) · 2                   =  12 288 B
total     = 262 144 + 12 288                      = 274 432 B   (ratio to fp16 dense = 0.008179)
```

### 7.4 Outlier channels kept in fp16, `out = 4096`

`outlier_bytes = k · 4096 · 2`:

| `k` | extra bytes | % of fp16 dense (33 554 432 B) |
|---|---|---|
| 16 | 131 072 | 0.391 % |
| 32 | 262 144 | 0.781 % |
| 64 | 524 288 | 1.562 % |

### 7.5 Measured verification against a real serialized artifact

Fixture: ONNX Runtime 1.30.0, int4 blockwise weight-only quantization of a single `MatMul`
with `W ∈ R^{64×32}` (`N = 2 048`), `bits = 4`, `block_size = 32`, `is_symmetric = True`,
`accuracy_level = 4`; `X (4, 64)` so that `MatMul(X, W) = X·W` (see `backend-capability.md` §2.4a
for the API and the orientation requirement). Files written by
`MatMulNBitsQuantizer.model.save_model_to_file`. Re-measured 2026-10-08.

| Quantity | Analytical (§3) | Measured |
|---|---|---|
| Packed weight payload | `ceil(2048·4/8) = 1 024 B` | `W_Q4` initializer `uint8[32,2,16]` = **1 024 B** ✓ (invariant across every build tried) |
| Scales: `n_blocks = 32 · ceil(64/32) = 64`, fp32 | `64·4 = 256 B` | `W_scales` `float32[32,2]` = **256 B** ✓ (invariant) |
| Payload + scales (the transferable part) | **1 280 B** | 1 280 B |
| `mem.checkpoint.payload_bytes` (external data sidecar) | — | **1 024 B** in the `.data` file (`W_Q4`) |
| `mem.checkpoint.metadata_bytes` (protobuf: scales + graph + node + names) | — | **603 B** (scales 256 B inline + 347 B of graph/node metadata) |
| `mem.checkpoint.total_bytes` | — | **1 627 B** |
| `bits_per_param_measured` | — | `1 627 · 8 / 2 048 = 6.3555` |
| Same graph at fp32 (measured) | — | 8 289 B ⇒ int4/fp32 = **0.1963** |

**The container term is build-dependent — do not treat it as a constant.** Same payload, different
serializer invocation (`backend-capability.md` §2.4b):

| Build | Container | Total | Overhead | `bits_per_param_measured` |
|---|---|---|---|---|
| `algo_config` + `load_model_with_shape_infer` | 603 B | 1 627 B | 347 B | 6.3555 |
| raw `onnx.load`, no pre-processing | 566 B | 1 590 B | 310 B | 6.2109 |
| constructor kwargs `bits=4, block_size=32, is_symmetric=True` | 569 B | 1 593 B | 313 B | 6.2227 |
| producer/doc strings cleared | 575 B | 1 599 B | 319 B | 6.2461 |
| orchestrator's independent fixture | 477 B | 1 501 B | 221 B | 5.8633 |

Interpretation: the *packing* is bit-exact (1 024 B = 2 048 nibbles, 256 B = 64 fp32 scales), so the
analytical model in §3 is correct for the tensor payload; the *container* adds 221–363 B of real
bytes for a 2 048-element matrix, which is 17–27 % on top of the 1 280 B payload. For a small tensor
this dominates — nominal 4.0 bits/param becomes 5.86–6.42 measured. This is exactly why §6 forbids
quoting analytical bits as measured bytes, and why §6.1 requires reconciliation.

### 6.1 Container reconciliation rule

For any SpectraQuant-serialized low-bit container:

```
mem.checkpoint.total_bytes  ==  payload_bytes + scales_bytes + zero_point_bytes + container_overhead_bytes
```

`container_overhead_bytes` (graph protobuf, node attributes, tensor names, alignment, sidecar
headers) MUST be **measured per artifact and reported** — it is real storage, not an accounting
detail, and for small tensors it can exceed the payload saving. A `bits_per_param` claim that omits
it is invalid. Concretely, when we export our own int4 container, the artifact total MUST be
reconciled against the sum of its parts, and the residual reported as
`mem.checkpoint.metadata_bytes` minus the scale/zero-point bytes it contains.

**External-data saves MUST be summed, not spot-checked.** When a serializer writes large tensors to a
sidecar (`.data`) file, `mem.checkpoint.total_bytes` MUST sum the graph file **and** every sidecar,
and any other file the save produced. Counting only the graph file understates the container by an
order of magnitude and can even make the overhead look *negative*: in the orchestrator's fixture the
graph file alone is 530 B for 1 280 B of tensor payload, while the true artifact is 530 + 1 024 =
**1 554 B** (and 1 501 B when the same tensors are written inline, without a sidecar). Our own saves
show the same shape: 597–619 B graph + 1 024 B `.data`. A per-file listing with sizes is therefore
mandatory evidence in §8 item 4, not optional detail.

Int8 comparison, same fixture via `quantize_dynamic(..., weight_type=QInt8)`: payload `W_quantized`
`int8[64,32]` = **2 048 B** ✓, per-tensor `W_scale` fp32 = 4 B, `W_zero_point` int8 = 1 B
⇒ tensor total 2 053 B; artifact `tiny_i8.onnx` = **2 667 B** ⇒ 614 B container overhead
(`bits_per_param_measured = 10.4180`). An independently built fixture by the orchestrator measured
2 627 B for the same nominal tensors — same payload, slightly leaner graph.

### 7.5.1 Reconstruction error for the same fixture (required to read the byte saving honestly)

The byte saving above is only meaningful alongside what it costs in numerics:

* `W` = `RandomState(0).randn(64, 32) * 0.1`, shape **(K, N) = (64, 32)**; `|W|.max = 0.3171`;
* `X` = `RandomState(1).randn(4, 64)`, shape **(·, K) = (4, 64)**; `X.std = 0.9680`;
* reference = `X @ W` in numpy on the fp32 weights; an ORT fp32 session on the same graph agrees
  bit-for-bit (`max|diff| = 0.000e+00`); `intra_op_num_threads = 8`;
* int4 `MatMulNBits`: `max|y − ref| = **0.2041**`, `mean|y − ref| = 0.0523`, `|ref|.max = 2.6457`
  ⇒ relative error **0.0771**.

The absolute figure is **scale-dependent**: the identical configuration with `σ_w = 1.0` gives
`max|y − ref| = 2.0406` and the *same* relative error 0.0771 (`backend-capability.md` §2.4c has the
full table). An error number MUST therefore be published with (a) the `W`/`X` distributions and
shapes, (b) the reference definition, and (c) this caveat; without them it is not reproducible.

### 7.6 Reproduction

The arithmetic in §7 was verified by a scratch script; the checked output is reproduced verbatim in
`docs/research/backend-capability.md` §2 (probe `verify_mem.py`). Re-derive with:

```bash
"$TEMP/sq-probe/Scripts/python.exe" "$TEMP/sq-probe/verify_mem.py"   # probe-only venv, outside the repo
"$TEMP/sq-probe/Scripts/python.exe" "$TEMP/sq-probe/probe_err.py"    # §7.5.1 error figures + scale dependence
"$TEMP/sq-probe/Scripts/python.exe" "$TEMP/sq-probe/probe_remeasure.py"  # container bytes (§7.5)
```

---

## 8. Test contract (what later code must satisfy)

1. `bits_per_param` returned by any packer equals `8·total_bytes/N` for the bytes it actually wrote.
2. `payload_bytes` from a packer equals `ceil(N·b_w/8)` before padding and the documented padded
   value after (§3.1).
3. `scales_bytes` equals `n_blocks·s_b` with `n_blocks` from §2's `ceil` formula (assert the
   trailing-short-group case explicitly).
4. `checkpoint.total_bytes` equals the sum of measured file sizes of **every** file in the artifact
   directory, including sidecar `.data`, scale, config and tokenizer files.
5. The three names in §1 are produced together; a compression ratio without a stated numerator and
   denominator fails validation.
6. `mem.deployed.resident_bytes ≥ mem.deployed.stored_bytes` for every dequantize-then-compute
   backend; equality only if the implementation proves no fp working copy exists.
7. **Container reconciliation (§6.1).** For a SpectraQuant-serialized container,
   `total_bytes == payload_bytes + scales_bytes + zero_point_bytes + container_overhead_bytes`, with
   `container_overhead_bytes` measured per artifact and reported, and **every** file the save
   produced (graph + sidecars) summed. A `bits_per_param` claim whose container term is unmeasured or
   omitted fails validation (measured range on this host for a 2 048-element int4 matrix: **221–363 B**,
   i.e. 5.86–6.42 bits/param against a nominal 4.0).
8. **Error figures are specified or absent.** Any reconstruction-error value emitted by tests or
   reports carries (a) the `W`/`X` distributions and shapes, (b) the reference definition, and
   (c) the scale-dependence caveat (§7.5.1). A bare `max_abs_err` assertion is invalid; tests must
   assert against a fixed fixture with a stated distribution.
