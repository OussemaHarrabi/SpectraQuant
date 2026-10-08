# Backend capability matrix — live state as of 2026-10-08

**Status:** evidence document. Every row resolves to exactly one of **usable-now**,
**usable-in-container**, **CUDA-gated**, **not-viable**. **Owner:** D-lite.
**Host:** AMD Ryzen AI 7 350 (8C/16T), 16 233 910 272 B RAM, AMD Radeon 860M iGPU, no CUDA device,
Windows 11 10.0.26200, Docker Desktop 4.85.0 (Linux containers).

Claims below are grounded either in a **probe actually executed on this host** (raw output in §2) or
in a **specific upstream document that was read** (URL given in the row). Anything not verified is
tagged `[UNVERIFIED]` in the row and listed in §7. Nothing here was inferred from a blog post or a
memory of how a library used to behave.

---

## 1. Method

* Probes ran in a **scratch virtual environment outside the repository**, so no dependency was added
  to the project environment (owned by another stream): `uv venv --python 3.11 "$TEMP/sq-probe"`,
  i.e. `C:/Users/oussa/AppData/Local/Temp/sq-probe`, with `torch` from the CPU wheel index
  (`--index-url https://download.pytorch.org/whl/cpu`) plus `torchao`, `onnxruntime`, `onnx`,
  `onnx_ir`, `numpy`, `bitsandbytes`, `transformers`, `accelerate`, `peft`.
* Wheel availability was checked against the PyPI JSON API (`https://pypi.org/pypi/<pkg>/json`,
  filename tags of the latest release), not from a mirror summary.
* Repository commits were pinned with `gh api repos/<owner>/<repo>/commits/<default-branch>
  --jq .sha`; no SHA in this document was guessed.
* Upstream documentation claims are cited by the exact URL that was read in this session.

---

## 2. Raw probe evidence (verbatim)

Only progress-bar and blank lines were removed. Nothing else was edited.

### 2.1 Toolchain and container runtime

```
$ uv --version
uv 0.12.15 (d35f1f270 2026-09-15 x86_64-pc-windows-msvc)
$ git --version
git version 2.55.0.windows.3
$ docker version
Client:  Version: 29.6.2  API version: 1.55  Go version: go1.26.5
         OS/Arch: windows/amd64   Context: desktop-linux
Server: Docker Desktop 4.85.0 (235549)
 Engine: Version: 29.6.2  API version: 1.55 (minimum version 1.40)  OS/Arch: linux/amd64
 containerd: v2.2.5   runc: 1.3.6   docker-init: 0.19.0
$ docker info | grep -iE "operating system|architecture|cpus|total memory|kernel|default runtime"
 Default Runtime: runc
 Kernel Version: 6.18.33.2-microsoft-standard-WSL2
 Operating System: Docker Desktop
 Architecture: x86_64
 CPUs: 16
 Total Memory: 7.318GiB
 Storage Driver: overlayfs / io.containerd.snapshotter.v1
$ docker run --rm alpine:3.20 uname -a
Linux 3fde355e2d9c 6.18.33.2-microsoft-standard-WSL2 #1 SMP PREEMPT_DYNAMIC Thu Jun 18 21:54:43 UTC 2026 x86_64 Linux
=== exit:0 ===
$ docker run --rm python:3.11-slim python -c "import platform,sys;print(platform.platform());print(sys.version)"
Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.41
3.11.17 (main, Oct  6 2026, 02:02:21) [GCC 14.2.0]
```

**Reading:** Linux containers are genuinely usable here (real `uname`, `exit:0`), the container glibc
is **2.41**, and the container budget is 16 CPUs / 7.318 GiB — considerably less RAM than the host.

### 2.2 PyTorch (CPU wheel) — scratch venv

```
$ "$TEMP/sq-probe/Scripts/python.exe" -c "import torch; <probe body, see §8>"
python          : 3.11.16 AMD64
torch           : 2.14.1+cpu
torch.version.cuda: None
mkldnn          : True
mkl             : True
cuda avail      : False
cuda devcount   : 0
quant engines   : ['onednn']
quant engine    : onednn
num_threads     : 8
ao.quantization : OK True
ao.nn.qat avail : True
nn.quantized    : OK True
```

Machine: `AMD64 Family 26 Model 96 Stepping 0, AuthenticAMD` (Zen 5); `torch.backends.mkldnn.enabled
= True`; `interop_threads = 8`.

**Reading:** `torch.cuda.is_available() == False`; the only registered quantized engine is
**`onednn`** (no `fbgemm`, no `qnnpack` — those are absent from Windows CPU wheels). `mkl`/`mkldnn`
are present, so CPU GEMM paths are the accelerated ones.

### 2.3 Quantization capability probe (`probe_quant.py`)

```
A torch.ao.quantize_dynamic(int8) : OK  out (4, 8) | packed params dtype: torch.qint8
...
torchao     : 0.18.0
B torchao Int8WeightOnlyConfig              : OK  layer=Linear out=(4, 32)
B torchao Int4WeightOnlyConfig(gs=32)       : FAIL ImportError: Requires mslk >= 1.0.0
B torchao Int8DynActInt8WeightConfig        : OK  layer=Linear out=(4, 32)
B torchao IntxWeightOnlyConfig(4)           : FAIL AssertionError: weight_dtype must be torch.intx,
                                                     where 1 <= x <= 8, but got 4
onnxruntime : 1.30.0 | providers: ['AzureExecutionProvider', 'CPUExecutionProvider']
```

(The `IntxWeightOnlyConfig(4)` failure is a probe-authoring error — the real signature takes a
`torch.dtype`; it was re-run correctly in §2.4, where it succeeds.)

### 2.4 torchao intx on Windows CPU + ONNX Runtime int8/int4 kernels (`probe_ort2.py`, `probe_remeasure.py`)

```
G IntxWeightOnlyConfig(int4, PerGroup(32))    : OK layer=Linear out=(4, 32)
    warnings: none
G IntxWeightOnlyConfig(int3, PerGroup(32))    : OK layer=Linear out=(4, 32)
    warnings: none
G Int8DynActIntxWeight(int4, PerGroup(32))    : FAIL TypeError: ... unexpected keyword argument 'granularity'

H0 fp32 .onnx bytes            : 8289
H1 ORT dynamic int8            : OK bytes=2667 nodes=[('DynamicQuantizeLinear', ''), ('Mul', ''),
                                  ('MatMulInteger', ''), ('Cast', ''), ('Mul', '')] max_abs_err=1.568e-02
H2 ORT int4 MatMulNBits        : OK nodes=[('MatMulNBits', 'com.microsoft')] max_abs_err=2.041e-01
```

H2 required `pip install onnx_ir` (see §3 caveat). Every `max_abs_err` figure quoted anywhere in this
document is **scale-dependent** and is fully specified — inputs, reference, caveat — in §2.4c.

#### 2.4a API surface actually verified (`probe_remeasure.py`)

An earlier probe used the **wrong module name**. The verified API in ONNX Runtime **1.30.0** is:

```
onnxruntime.quantization.matmul_nbits_quantizer.MatMulNBitsQuantizer    # NOT "matmul_4bits_quantizer"
  __init__(self, model: ModelProto | str, bits: int = 4, block_size: int = 128,
           is_symmetric: bool = False, accuracy_level: int | None = None,
           nodes_to_exclude=None, nodes_to_include: list[str] | None = None,
           quant_format=QuantFormat.QOperator, op_types_to_quantize: tuple[str, ...] | None = None,
           quant_axes: tuple[tuple[str, int], ...] | None = None,
           channel_wised_quantize: bool = False,
           algo_config: WeightOnlyQuantConfig | None = None)
onnxruntime.quantization.matmul_bnb4_quantizer.MatMulBnB4Quantizer      # legacy variant
onnxruntime.quantization.quant_utils.load_model_with_shape_infer
```

`onnxruntime.quantization.matmul_4bits_quantizer` **does not exist** in 1.30.0 — the upstream doc
page still shows that name, which is a documentation lag, not a real module.

After `process()`, `quantizer.model` is an **`ONNXModel` wrapper**, not a `ModelProto`:

```
  [i4_shapeinfer] type(quantizer.model) = ONNXModel; has save_model_to_file = True
  [i4_shapeinfer] type(quantizer.model.model) = ModelProto
  [i4_shapeinfer] onnx.save(q.model) fails as reported: AttributeError: 'function' object has no attribute 'initializer'
```

so persistence MUST go through `quantizer.model.save_model_to_file(path, use_external_data=True)`.
(`onnx.save(quantizer.model, …)` raises the `AttributeError` above; `onnx.save(quantizer.model.model, …)`
works but bypasses external-data sidecar management.)

**Orientation requirement.** ONNX `MatMul(X, W)` computes `Y = X · W` with `X` shaped `(·, K)` and `W`
shaped `(K, N)`. A PyTorch `x @ w.T` is **not** that operator — `W` must be transposed at export
time. The fixture below is `X (4, 64)`, `W (64, 32)`, `Y (4, 32)`, reference `X @ W`. Feeding a
`(64, 4)` activation into this graph is a different, silently wrong op (and would also produce a
different serialized graph, hence different artifact bytes).

#### 2.4b Re-measured artifact bytes

```
== fp32 sources ==
fp32 direct            : 8289 B
== int4 MatMulNBits ==
  [i4_shapeinfer] files: [('i4_shapeinfer', 603), ('i4_shapeinfer.data', 1024)] -> TOTAL 1627 B
  [i4_direct]     files: [('i4_direct', 566), ('i4_direct.data', 1024)]         -> TOTAL 1590 B
  initializers: [('W_Q4', 2, [32, 2, 16], 0), ('W_scales', 1, [32, 2], 256)]
  nodes: [('MatMulNBits', 'com.microsoft')]
== int8 dynamic ==
  bytes=2667 nodes=[('DynamicQuantizeLinear',''), ('Mul',''), ('MatMulInteger',''), ('Cast',''), ('Mul','')]
  initializers: [('W_scale', 1, [], 0), ('W_zero_point', 3, [], 0), ('W_quantized', 3, [64, 32], 2048)]
== execution + thread pinning ==
  SessionOptions.intra_op_num_threads = 8 | OMP_NUM_THREADS env = None
  i4_shapeinfer  run OK  out=(4, 32)  max_abs_err=0.2041
  i4_direct      run OK  out=(4, 32)  max_abs_err=0.2041
  int8_dynamic   run OK  out=(4, 32)  max_abs_err=0.0208
```

**Reading (corrected).** ONNX Runtime 1.30.0 executes **real** int8 (`MatMulInteger`) and int4
weight-only (`MatMulNBits`, domain `com.microsoft`) kernels on this CPU, and the int4 packing is
**bit-exact and construction-invariant**: `W_Q4` is `uint8[32,2,16]` = **1 024 B** exactly (= 2 048
four-bit values) and `W_scales` is `float32[32,2]` = **256 B** (64 blocks) in every build tried.
Note `W_Q4` lands in the external `.data` sidecar while the 256 B of scales stay inline in the
protobuf (`W_Q4.raw_data` is empty when loaded with `load_external_data=False`).

The **container** term is *not* invariant: it is graph metadata, and it moves with the build and save
path (pre-processing/shape inference, producer fields, external-data policy). Measured spread for the
identical 1 280 B tensor payload:

| Construction | Payload | Scales | Container | Total | Container overhead |
|---|---|---|---|---|---|
| `algo_config` + `load_model_with_shape_infer` (pre-processed) | 1 024 B | 256 B | 603 B | **1 627 B** | 347 B |
| `algo_config`, raw `onnx.load`, no pre-processing | 1 024 B | 256 B | 566 B | **1 590 B** | 310 B |
| constructor kwargs `bits=4, block_size=32, is_symmetric=True` | 1 024 B | 256 B | 569 B | **1 593 B** | 313 B |
| producer/doc strings cleared before quantizing | 1 024 B | 256 B | 575 B | **1 599 B** | 319 B |
| output path `a.onnx` (sidecar `a.data`) | 1 024 B | 256 B | 591 B | **1 615 B** | 335 B |
| output path `a_much_longer_output_filename.onnx` | 1 024 B | 256 B | 619 B | **1 643 B** | 363 B |
| raw `onnx.save(q.model.model, …)`, tensors inline, no sidecar | 1 280 B | — | (inside protobuf) | **1 568 B** | 288 B |
| orchestrator's independent fixture, 2026-10-08 | 1 024 B | 256 B | 477 B | **1 501 B** | 221 B |

**Caveat on reading that table:** its rows differ in more than one uncontrolled factor (output
filename length, pre-processing, producer fields, save API), so it demonstrates *variance*, not a
clean single-factor effect. The filename effect was isolated separately: the same tensors saved as
`a` and as `a_much_longer_output_filename` produced 1 615 B and 1 643 B — **28 B apart** — because the
external-data sidecar's *name* is serialized into the protobuf as the tensor's
`external_data.location`. Consequences are the same either way: the container term MUST be measured
for the actual artifact and quoted with any `bits/param` figure, and `mem.checkpoint.total_bytes`
MUST sum every file the save produced (§`memory-accounting.md` §6.1).

It is also 1-byte sensitive to tensor *shapes*: the first run used `X (2, 64)` instead of `X (4, 64)`
and the protobuf came out 602 B rather than 603 B. The orchestrator's independently built fixture
measured 1 501 B (int4) and 2 627 B (int8) for the same nominal tensors; our 1 568–1 643 B and
2 667 B are equally real. Both are published rather than reconciled to a single "official" number,
because the honest statement is that the container term is a property of the serializer invocation.

Reproducible end-to-end snippet (verified on this host):

```python
import numpy as np, onnx
from onnx import helper, TensorProto
from onnxruntime.quantization import matmul_nbits_quantizer as mnq, quant_utils
from pathlib import Path

W = (np.random.RandomState(0).randn(64, 32).astype(np.float32) * 0.1)        # (K, N)
g = helper.make_graph([helper.make_node("MatMul", ["X", "W"], ["Y"], name="mm")], "tiny",
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [4, 64])],         # (., K)
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 32])],
    [helper.make_tensor("W", TensorProto.FLOAT, [64, 32], W.ravel().tolist())])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
m.ir_version = 10            # onnx 1.23.2 defaults to IR 14; ORT 1.30 accepts <= 13
onnx.save(m, "tiny_fp32.onnx")

cfg = mnq.DefaultWeightOnlyQuantConfig(
    block_size=32, is_symmetric=True, accuracy_level=4,
    quant_format=quant_utils.QuantFormat.QOperator, op_types_to_quantize=("MatMul",))
q = mnq.MatMulNBitsQuantizer(
        quant_utils.load_model_with_shape_infer(Path("tiny_fp32.onnx")), algo_config=cfg)
q.process()
q.model.save_model_to_file("tiny_i4.onnx", True)     # NOT onnx.save(q.model, ...)
```

Two ordering traps: `pip install onnx_ir` is required by the 4-bit quantizer, and the emitted IR
version must stay ≤ 13 for ORT 1.30, which is why `ir_version` is pinned above.

#### 2.4c Error figures: distribution, reference, and scale dependence

Every `max_abs_err` value in this document is **scale-dependent** and is meaningless without its
inputs. The exact fixture behind all of them (`probe_err.py`):

* `W` = `numpy.random.RandomState(0).randn(64, 32) * σ_w`, shape **(K, N) = (64, 32)**;
* `X` = `numpy.random.RandomState(1).randn(4, 64)`, shape **(·, K) = (4, 64)**;
* **reference** = `X @ W` in numpy on the fp32 weights. The alternative reference — running the fp32
  ONNX graph through an ORT `CPUExecutionProvider` session — was checked and agrees **bit-for-bit**
  (`max|ref_numpy − ref_ort_fp32| = 0.000e+00`), so at this size the two reference definitions are
  interchangeable; both are recorded here rather than assumed;
* int4 config: `bits = 4`, `block_size = 32`, `is_symmetric = True`, `accuracy_level = 4`,
  kernel `MatMulNBits` (domain `com.microsoft`), `intra_op_num_threads = 8`.

Measured with identical RNG draws, varying only the weight scale:

| `σ_w` | `|W|.max` | `X.std` | ref std | ref max | int4 `max|y−ref|` | mean abs err | **relative (max err / ref max)** |
|---|---|---|---|---|---|---|---|
| 1.0 | 3.1710 | 0.9680 | 8.5899 | 26.4568 | **2.0406** | 0.5234 | **0.0771** |
| 0.1 | 0.3171 | 0.9680 | 0.8590 | 2.6457 | **0.2041** | 0.0523 | **0.0771** |

The absolute error scales linearly with the weight magnitude — the relative error is identical to four
decimals — which is the expected behaviour of blockwise int4 weight-only quantization: the per-block
scale is proportional to `|W|`, so quantized weights and output error scale with it. The
same configuration on a differently drawn `W ~ N(0,1)` (the orchestrator's independent run) gives
2.3686 instead of 2.0406; that is the max over 128 outputs of a different draw, **not** a different
code path, save format or `accuracy_level` — both save paths (raw `onnx.save` and
`save_model_to_file(…, True)`) reproduced 2.0406 in the same measurement.

Int8 figures (dynamic `quantize_dynamic`, `weight_type = QInt8`, per-tensor scale), same `W`
(`σ_w = 0.1`), the two rows differing only in `X`:

| `X` shape | int8 `max|y−ref|` | mean abs err | ref max | artifact |
|---|---|---|---|---|---|
| (2, 64) | **1.5681e-02** | 4.1057e-03 | 1.5940 | 2 667 B |
| (4, 64) | **2.0771e-02** | 6.2564e-03 | 2.6457 | 2 667 B |

The two int8 numbers (§2.4 `H1` = 1.568e-02, §2.4b = 0.0208) are therefore the same configuration
sampled over a different number of outputs — a max over 64 versus 128 values of the same distribution
— and the artifact size is identical (2 667 B) because only `W` is quantized. The 0.4537 figure in
§2.5 is a different quantity entirely: a bitsandbytes NF4 **quantize→dequantize round-trip** on a
64×64 `N(0,1)` tensor, not a model-level `MatMul` error.

**Rule (binding for any result we publish):** an error figure MUST be reported together with
(a) the `W` and `X` distributions and shapes, (b) the definition of the reference tensor, and
(c) the statement that the value is scale-dependent and only meaningful with (a) and (b). Two
unexplained error numbers for "the same" configuration are a reproducibility defect.

### 2.5 bitsandbytes on Windows CPU (`probe_bnb.py`)

```
python: 3.11.16 AMD64 | torch: 2.14.1+cpu | cuda: False
bitsandbytes: 0.50.2
OK   | bnb.nn.Linear8bitLt on CPU forward               | (4, 32)
OK   | bnb.nn.Linear4bit on CPU forward                 | (4, 32)
OK   | bnb.functional.quantize_4bit(nf4) CPU            | packed dtype=torch.uint8 shape=(2048, 1)
                                                           roundtrip_maxerr=0.4537
OK   | bnb.optim.Adam8bit on CPU params                 | loss=0.32147
FAIL | transformers BitsAndBytesConfig construction      | ModuleNotFoundError: No module named 'transformers'
```

The last failure was only a missing package in the probe venv; it was re-run in §2.6.

### 2.6 HF transformers + PEFT + bitsandbytes, 4-bit load on CPU (`probe_hf.py`)

```
torch: 2.14.1+cpu | cuda: False
transformers: 5.19.0 | peft: 0.21.2 | accelerate: 1.15.0
bitsandbytes: 0.50.2
int8_vectorwise_quant returns: tuple len 3 [('torch.int8', (64, 64)), ('torch.float32', (64,)), None]
BitsAndBytesConfig OK: nf4 True True
peft.prepare_model_for_kbit_training: True
peft.replace_lora_weights_loftq   : True
4-bit from_pretrained on CPU : OK device=cpu linear_types=['Linear', 'Linear4bit'] logits=(1, 3, 1000)
```

**Reading:** a real NF4 4-bit model (`hf-internal-testing/tiny-random-gpt2`) loads **and runs a
forward pass on CPU** with `Linear4bit` modules. This is 4-bit *storage* with dequantize-to-fp32
compute — it is **not** a 4-bit kernel measurement (see `benchmark-protocol.md` §6).

### 2.7 PyPI wheel availability (latest release, Windows-relevant tags)

```
bitsandbytes    0.50.2           win_amd64 wheel present  [bitsandbytes-0.50.2-py3-none-win_amd64.whl]
vllm            0.31.0           NO win wheel   [manylinux_2_28/2_39 x86_64, aarch64, macosx, + sdist]
auto-gptq       0.7.1            win_amd64 wheels present  (cp38–cp311) — but CUDA-only kernels
gptqmodel       7.5.0            NO win wheel   [sdist only]
autoawq         0.2.9            NO win wheel   [sdist only]
onnxruntime     1.30.0           win_amd64 wheels present  (cp311–cp314)
onnx            1.23.2           [installed; IR_VERSION 14]
onnx-ir         1.0.0            [installed; required by ORT 4-bit quantizer]
torchao         0.18.0           pure-py3 (py3-none-any)
transformers    5.19.0           pure-py3
peft            0.21.2           pure-py3
accelerate      1.15.0           pure-py3
datasets        5.1.0            pure-py3
mlflow          3.17.0           pure-py3
dvc             3.67.1           pure-py3
lm-eval         0.4.13           pure-py3
evaluate        0.4.6            pure-py3
safetensors     0.8.0            win_amd64 wheel present
quanto          0.2.0            pure-py3
hqq             0.2.8.post1      sdist only
torch-directml  0.2.5.dev240914  win_amd64 wheels present (dev build, dated 2024-09-14)
vllm v0.31.0 release assets: vllm-0.31.0+cpu-cp38-abi3-manylinux_2_39_x86_64.whl (150 877 005 B)
```

### 2.8 Memory-accounting arithmetic verifier (`verify_mem.py`)

```
EX2  transformer projection out=in=4096
N params                    : 16777216
fp16 bytes                  : 33554432
int8 sym per-channel: w=16777216 scales=16384 total=16793600 bits/param=8.0078 ratio_fp16=0.500488
int4 gs=128 sym fp16 scales : blocks=131072 w=8388608 scales=262144 zp=0 total=8650752 bits/param=4.1250 ratio_fp16=0.257812
int4 gs=128 asym 4-bit zp   : total=8716288 bits/param=4.1562 ratio_fp16=0.259766
int4 gs=128 asym int8 zp    : total=8781824 bits/param=4.1875 ratio_fp16=0.261719
uniform-int4 theoretical N*4/8 : 8388608 bits/param 4.0000 ratio_fp16=0.250000

EX5  measured ONNX Runtime artifacts vs analytical (re-measured 2026-10-08)
params: 2048
int4 payload=1024 B (INVARIANT, uint8[32,2,16]) ; scales=256 B (INVARIANT, float32[32,2]) ; payload+scales=1280 B
int4 container term is BUILD-DEPENDENT (graph protobuf metadata):
   load_model_with_shape_infer    container=603 B -> total=1627 B  overhead=347 B  bits/param=6.3555
   raw onnx.load                  container=566 B -> total=1590 B  overhead=310 B  bits/param=6.2109
   ctor kwargs bits=4,bs=32,sym   container=569 B -> total=1593 B  overhead=313 B  bits/param=6.2227
   producer/doc cleared           container=575 B -> total=1599 B  overhead=319 B  bits/param=6.2461
   orchestrator independent       container=477 B -> total=1501 B  overhead=221 B  bits/param=5.8633
   invariant part of the split: payload 1024 + scales 256 = 1280 B in every build
int8 dynamic: payload=2048 scale=4 zp=1 payload+meta=2053 ; measured artifact=2667 B ; overhead=614 B ; bits/param=10.4180
fp32 same graph (measured) 8289 B; int4/fp32 = 0.1963 (shape-infer build) ; int4/theoretical = 0.1986
```

---

## 3. Compatibility matrix

Legend: **usable-now** = works on this Windows/CPU host today, verified; **usable-in-container** =
needs the Linux container path; **CUDA-gated** = needs an NVIDIA GPU that this host does not have;
**not-viable** = no supported path on any hardware available to this project.

| # | Backend / feature | Verdict | Evidence | Verified version · pinned upstream SHA | Blocker / caveat | Legitimate use in SpectraQuant |
|---|---|---|---|---|---|---|
| 1 | **PyTorch CPU quantized ops** (`torch.ao.quantization.quantize_dynamic`, engine `onednn`) | usable-now | probe §2.2, §2.3 (`OK`, `dtype: torch.qint8`) | torch **2.14.1+cpu**; `pytorch/pytorch@79f81f9e6d3b7c60549a35569dcbf1b8e295f9b3` | `supported_engines == ['onednn']` only (no fbgemm/qnnpack on Windows). Deprecated: docs state `torch.ao.quantization` is planned for deletion in favour of torchao (the page names 2.10; it still exists and runs in 2.14.1, emitting a `DeprecationWarning`). [https://docs.pytorch.org/docs/2.14/quantization.html] | Class 3 int8 packing + class 2 numerics reference. Its onednn int8 kernels are **not** on the `AGENTS.md` §5 permitted class 4-CPU backend list (only ONNX Runtime and torchao intx are), so any timing from this path stays engineering telemetry. |
| 2 | **TorchAO — int8 / intx weight-only + QAT prepare/convert on CPU** | usable-now | probe §2.4 (`OK`, no warnings) | torchao **0.18.0**; `pytorch/ao@cff77b46ef85ba8472b4a286598073d6c01e18e5` | Not listed as a supported target in TorchAO's own hardware matrix (only NVIDIA CUDA / Edge-ARM / ROCm / Intel columns); int3/int4 intx here are CPU-supported but not benchmarked upstream. [https://docs.pytorch.org/ao/stable/workflows/index.html] | Class 2 QAT/fake-quant research and class 3 int8/int4 factor packing. |
| 3 | **TorchAO — `Int4WeightOnlyConfig` (int4 tinygemm)** | CUDA-gated | probe §2.3 (`ImportError: Requires mslk >= 1.0.0`) | torchao 0.18.0 | `mslk` ships only via `--index-url https://download.pytorch.org/whl/cu130`; int4 tinygemm targets A100/H100. [https://docs.pytorch.org/ao/stable/workflows/inference.html] | Nothing locally. Deferred Tier 2+ (this is the LR-QAT-style int4 path we care about). |
| 4 | **bitsandbytes CPU backend** (NF4 4-bit storage, `Linear8bitLt`, `Linear4bit`, 8-bit optimizers) | usable-now | probes §2.5, §2.6 (4-bit model forward on CPU) | bitsandbytes **0.50.2**; `bitsandbytes-foundation/bitsandbytes@833649043474794b8fe7a4136e0c40faf077b2e0` | CPU is **dequantize-then-fp32**: no low-bit compute. [https://huggingface.co/docs/bitsandbytes/main/en/installation] (CPU + Windows x86-64 support, min AVX2); [https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes] ("CPU … Windows x86-64") | Real 4-bit **checkpoint format** to compare against (class 3) and QLoRA/LoftQ-style class-2 experiments. Never a speed claim. |
| 5 | **bitsandbytes CUDA kernels** (LLM.int8, NF4/FP4 matmul) | CUDA-gated | host has no CUDA device (§2.2); install table lists CUDA targets | bnb 0.50.2 | Requires NVIDIA CC ≥ 6.0 (NF4) / ≥ 7.5 (LLM.int8) and CUDA 11.8–13.x. [https://huggingface.co/docs/bitsandbytes/main/en/installation] | Deferred: the QLoRA baseline for Tier 2. |
| 6 | **GPTQ** (AutoGPTQ / GPTQModel) | not-viable | PyPI §2.7 (auto-gptq win wheels exist, but kernels are CUDA); docs | auto-gptq 0.7.1 · `AutoGPTQ/AutoGPTQ@9f7d37072917ab3a7545835f23e808294a542153`; GPTQModel 7.5.0 · `ModelCloud/GPTQModel@d0e59f892b77228e6e9774fc4c43850410e362bb` | All `qlinear_*` kernels are `cuda`/`exllama`/`marlin`/`triton`/`hpu`; install table is CUDA 11.8 / CUDA 12.1 / ROCm only; `BUILD_CUDA_EXT=0` falls back to a "slow python implementation" the docs "strongly discourage"; AutoGPTQ is unmaintained. [https://github.com/AutoGPTQ/AutoGPTQ] | Nothing locally. GPTQ int4 checkpoints are a Tier-2 comparison target only. |
| 7 | **AWQ** (AutoAWQ / llm-awq) | not-viable | PyPI §2.7 (no Windows wheel, sdist only); docs | AutoAWQ 0.2.9 · `casper-hansen/AutoAWQ@88e4c76b20755db275574e6a03c83c84ba3bece5`; `mit-han-lab/llm-awq@d6e797a42b9ef7778de8ee2352116e0f48a78d61` | Deprecated ("officially deprecated and will no longer be maintained"); default install relies on **Triton** kernels (CUDA); the Intel-x86 CPU path needs `intel_extension_for_pytorch` + torch ≥ 2.4 and is "optimized for Intel"; FasterTransformer fused path is Linux-only. [https://github.com/casper-hansen/AutoAWQ] | Nothing locally. Tier-2 comparison target; upstream successor is vLLM's llm-compressor. |
| 8 | **SmoothQuant — fake-quant + PPL evaluation path** | usable-now | docs (pure PyTorch `fake_quant.py`, `ppl_eval.py`) `[UNVERIFIED: not executed here]` | `mit-han-lab/smoothquant@c61476d728e42ae0d8a35e7e78494edcac3237b5` | Repo install pins torch 1.12.1+cu113 / python 3.8 — we would reuse only the algorithm, not its pinned environment. [https://github.com/mit-han-lab/smoothquant] | Class-2 W8A8 activation-smoothing baseline for the sensitivity proxy. |
| 9 | **SmoothQuant — INT8 inference** (CUTLASS INT8 GEMM via `torch-int`) | CUDA-gated | docs | as above | Its real INT8 inference is "implemented … with CUTLASS INT8 GEMM kernels, wrapped … in torch-int"; NVIDIA-only. [https://github.com/mit-han-lab/smoothquant] | Deferred Tier 2+. |
| 10 | **llama.cpp / GGUF CPU k-quant kernels** | usable-now | docs `[UNVERIFIED: not executed here]` | `ggml-org/llama.cpp@de7fa0a3c6a2e1b4cd9f22eb8d6bf5b12dbdb63b` | None for running; Windows prebuilt binaries + winget + `powershell irm https://llama.app/install.ps1`. AVX/AVX2/AVX512 support, 1.5–8-bit integer quantization. [https://github.com/ggml-org/llama.cpp] | Real **CPU** low-bit kernels, but they execute *GGUF produced by llama.cpp*, not our artifacts — per `AGENTS.md` §5 that is **engineering telemetry only**, never a class 4 claim for a SpectraQuant artifact. Still the most practical local Tier-1 sanity check of a compressed model's text quality. |
| 11 | **ONNX Runtime CPU** (int8 dynamic `MatMulInteger`, int4 weight-only `MatMulNBits`) | usable-now | probe §2.4a/§2.4b (both `OK`, real kernels, bit-exact int4 packing, bytes re-measured) | onnxruntime **1.30.0**; `microsoft/onnxruntime@17be9c28b14dcd175c4dec16c46255886ad6cda0` | Three integration caveats, all measured here: (a) `onnx 1.23.2` writes **IR version 14** while ORT 1.30 accepts **≤ 13** → the fixture must pin `model.ir_version = 10`; (b) the 4-bit quantizer needs the extra `onnx_ir` package (`No module named 'onnx_ir'` until installed); (c) the 4-bit quantizer lives at `onnxruntime.quantization.matmul_nbits_quantizer` (the doc page's `matmul_4bits_quantizer` does not exist) and returns an `ONNXModel` wrapper that must be saved with `quantizer.model.save_model_to_file(path, True)` — `onnx.save(q.model, …)` fails with `AttributeError: 'function' object has no attribute 'initializer'` (§2.4a). [https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html] | Class 3 int4/int8 **container** writing with byte-exact accounting; **the primary class 4-CPU backend** for SpectraQuant-serialized int4/int8 artifacts (`AGENTS.md` §5; protocol in `benchmark-protocol.md` §8). |
| 12 | **ONNX Runtime GPU (TensorRT EP)** | CUDA-gated | docs | — | "You need a device that supports Tensor Core int8 computation, like T4 or A100." [https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html] | Deferred Tier 2+. |
| 13 | **vLLM — CPU serving (BF16/FP16)** | usable-in-container | container capability verified §2.1; CPU wheel asset verified §2.7 `[UNVERIFIED: vLLM itself not installed here]` | vllm **0.31.0**; `vllm-project/vllm@72d59adc2c76de8f77aad640c7a04d663a970735` | **No Windows wheel, ever**: CPU support is "OS: Linux", x86 requires `avx2` (limited) or `avx512f` (recommended); engine wheels exist only for `manylinux_2_39`. [https://docs.vllm.ai/en/latest/getting_started/installation/cpu/] | Nothing in the core path; class 5 is unavailable by contract (`AGENTS.md` §2.4). Would require a preregistration amendment even to be timed. |
| 14 | **vLLM — quantized kernels** (AWQ / GPTQ / bitsandbytes / FP8) | CUDA-gated | docs | vllm 0.31.0 | bitsandbytes ❌ / GGUF ❌ / Marlin ❌ on x86 CPU; x86-CPU column shows AWQ ✅ and GPTQ ✅ upstream, but the local path is Linux-container-only and out of scope here. NVIDIA support is CC-gated (Marlin needs Turing+). [https://docs.vllm.ai/en/latest/features/quantization/] | Deferred Tier 2+. |
| 15 | **HF transformers + accelerate** | usable-now | probe §2.6 (imports + real 4-bit CPU forward) | transformers **5.19.0**, accelerate **1.15.0**; `huggingface/transformers@4cc2aa84301c9aa210b5513dab9fefae03981f6e` | None blocking. The quantization support table marks many methods CPU-capable, but "CPU" there means *storage + dequantized execution*. [https://huggingface.co/docs/transformers/main/en/quantization/overview] | Model loading, tokenization, class-2 evaluation plumbing for Tier 0/1. |
| 16 | **HF PEFT** | usable-now | probe §2.6 (`prepare_model_for_kbit_training`, `replace_lora_weights_loftq` present) | peft **0.21.2**; `huggingface/peft@8bb3e702976b48f16c1995c97b4cb32b038e1e15` | Documented limits: `replace_lora_weights_loftq` supports **only bitsandbytes 4-bit** and requires safetensors; PEFT+torchao merging only works for `int8_weight_only`. [https://huggingface.co/docs/peft/developer_guides/quantization] | LoRA/LoftQ-style adapter experiments (class 2) on the CPU-4-bit base model. |
| 17 | **lm-evaluation-harness — `hf` backend** | usable-now | PyPI §2.7 (pure-py3) `[UNVERIFIED: not installed/run here]` | lm-eval **0.4.13**; `EleutherAI/lm-evaluation-harness@d6de81643928d653435c431bae19945d41d32520` | Base package no longer includes torch/transformers — needs `pip install "lm_eval[hf]"`; README examples target CUDA but the backend takes any torch device. [https://github.com/EleutherAI/lm-evaluation-harness] | Tier-1 quality evaluation (class 2) on tiny models; must keep evaluation questions out of training data (`AGENTS.md` §4.7). |
| 18 | **lm-evaluation-harness — `vllm` backend** | usable-in-container | as #13 | lm-eval 0.4.13 · vllm 0.31.0 | Needs the Linux container + vLLM CPU wheel; class 5 unavailable here. | Only if a service-measurement amendment is ever approved. |
| 19 | **DVC** | usable-now | PyPI §2.7 (pure-py3); install docs | dvc **3.67.1**; `iterative/dvc@56e59829512ff134aa269099a2099587b810b4dd` | None. Documented Windows paths: winget / choco / scoop / conda / pip / self-contained installer. Note remote-storage extras are separate packages. [https://dvc.org/doc/install/windows] | Data/artifact versioning (WikiText-2, seed fixtures, run outputs). |
| 20 | **MLflow** | usable-now | PyPI §2.7 (pure-py3) | mlflow **3.17.0**; `mlflow/mlflow@9c30f9c57b80b6a0af74a457333e7c12be081f26` | None blocking. Chosen tracker in ADR-0003. [https://mlflow.org/docs/latest/index.html] | The single experiment tracker (`AGENTS.md` §7: one tracker only). |
| 21 | **torch-directml** (DirectML iGPU path) | not-viable | PyPI §2.7: last release `0.2.5.dev240914` | 0.2.5.dev240914 (dated **2024-09-14**) | Unmaintained developer build on a *different* torch fork; not a research-grade backend; the Radeon 860M shares system RAM so there is no memory win. Consistent with `docs/research/environment.md` ("GPU acceleration is treated as absent"). | Nothing. |
| 22 | **flash-attention** | CUDA-gated | `AGENTS.md` §2.1 (binding); probe §2.2 (`cuda avail: False`) | — | CUDA-only; SM80+ in practice. | Deferred Tier 2+. |
| 23 | **Docker Linux containers** | usable-now | probe §2.1 (alpine + python:3.11-slim both ran, `exit:0`) | Docker Desktop **4.85.0**, Engine **29.6.2**, containerd 2.2.5, kernel 6.18.33.2-WSL2 | Container limits: 16 CPUs, **7.318 GiB** RAM. Container must be started explicitly; the repository path is not the container path. | Only route for Linux-only tooling (vLLM, some CUDA-adjacent wheels). Not a substitute for a GPU. |

**What immediately follows from the matrix.** Two beliefs are worth correcting early, because both
would otherwise turn into a bad claim later:

* *"4-bit on this laptop means fake quantization only."* False as stated: real 4-bit **storage** is
  reachable here through three independent routes (bitsandbytes NF4 including a working
  `transformers` load, torchao intx, ONNX Runtime int4 containers), and two of them are **class-3
  measured bytes**, not estimates.
* *"…therefore we can measure 4-bit speed."* **Partly true, in a strictly bounded way.** `AGENTS.md`
  §5 was amended on 2026-10-08 to split class 4 by hardware, and **class 4-CPU is available**: a real
  CPU kernel executing an artifact **SpectraQuant serialized itself** — ONNX Runtime `MatMulNBits`
  (int4 weight-only) / `MatMulInteger` (int8) on our own container, and torchao intx weight-only
  int4/int8. Two things stay out of scope: the dequantize-to-fp32 routes (bitsandbytes CPU 4-bit
  computes in fp32) and **third-party-format** kernels (llama.cpp GGUF, ONNX artifacts produced by
  other pipelines), which are engineering telemetry only. Class 4-GPU and class 5 remain unavailable
  — see §4.

---

## 4. Definitive statement on class 4 and class 5

`AGENTS.md` §5 (orchestrator-owned) was amended on 2026-10-08 to split **class 4 by hardware**, on the
strength of the evidence in §2.4a/§2.4b. The position for this workstation is therefore:

| Class | Status on this host | Basis |
|---|---|---|
| 1 analytical estimate | **available** | pure arithmetic, no host dependency |
| 2 fake-quantization quality | **available** | CPU fp32/fp16 execution (`torch 2.14.1+cpu`) |
| 3 packed storage | **available** | byte-exact int4 container verified (§2.4b) |
| **4-CPU — kernel-backed inference on a SpectraQuant-serialized artifact** | **available** | ONNX Runtime 1.30.0 `MatMulNBits` (int4) and `MatMulInteger` (int8) execute our own container on this CPU (§2.4b); torchao `IntxWeightOnlyConfig(torch.int4/int3, PerGroup(g))` forwards on CPU (§2.4) |
| **4-GPU — kernel-backed inference on CUDA** | **unavailable / deferred** | no NVIDIA device; bitsandbytes, GPTQ, AWQ/Marlin/exllama, TorchAO CUDA tinygemm (`mslk`), FlashAttention and vLLM-CUDA are all CUDA-gated (§3) |
| **5 end-to-end service** | **unavailable** | vLLM ships no Windows wheel and its CPU support is Linux-only; service measurement stays out of scope until a Linux GPU host is documented (`AGENTS.md` §2.4) |

**What a class 4-CPU claim requires** (mirrors `AGENTS.md` §5 and `benchmark-protocol.md` §8):

1. the artifact executed MUST be one **SpectraQuant serialized itself** — an artifact produced by
   another pipeline (llama.cpp GGUF, an externally produced ONNX file) is telemetry, never class 4
   for our artifact;
2. name the backend + version, the **kernel/op and its domain** (e.g. `MatMulNBits` in
   `com.microsoft`), the container format, the **thread count**, and the CPU model;
3. compare against an **fp32 CPU baseline measured in the same session on the same machine**;
4. state explicitly that the number is **not comparable to published GPU latency/throughput** figures.

**What remains unavailable, and why** (the capability-vs-authorization argument, now applied per
sub-class — see `docs/protocols/measurement-taxonomy.md` §5):

* **4-GPU.** torchao's int4 tinygemm fails loudly here (`Requires mslk >= 1.0.0`, a CUDA-index
  package); GPTQ/AWQ/Marlin/exllama and bitsandbytes' real kernels are CUDA-only; vLLM's quantized
  paths are Linux + NVIDIA with no Windows wheel at all. Nothing short of a GPU host changes this
  (§5), so the disqualifier for GPU-class claims stands exactly as before.
* **5.** The only serving stack reachable here is vLLM inside the Linux container, which would serve
  a non-SpectraQuant model; `AGENTS.md` §2.4 keeps service measurement out of scope on this
  workstation.
* **Third-party-format CPU kernels.** Runnable (llama.cpp k-quant/GGUF, externally produced ONNX
  artifacts) but they execute someone else's artifact and cannot be compared with the GPU baselines
  the literature reports — telemetry only.

Escalating anything from telemetry to a reported 4-GPU measurement still requires a superseding ADR
plus a timestamped amendment in `docs/research/preregistration-amendments.md`. Class 4-CPU needs no
such escalation — `AGENTS.md` §5 already authorizes it — but every class 4-CPU number MUST carry the
protocol in `docs/protocols/benchmark-protocol.md` §8.

---

## 5. What a future GPU host would need

Recorded so the Tier-2 gate (`AGENTS.md` §2.3, §6) can be checked mechanically. Requirements marked
`[INFERENCE]` are derived from the cited upstream tables rather than read verbatim.

**Platform and driver**

* One NVIDIA discrete GPU; **Linux host strongly preferred** — vLLM has no Windows wheel (§2.7), and
  the CUDA wheel ecosystem is Linux-first. A Windows GPU host is workable only for the PyTorch-native
  parts (bitsandbytes, torchao) and loses the vLLM serving path.
* Driver/CUDA: the bitsandbytes install table for `Windows x86-64` / `Linux x86-64` covers CUDA
  toolkits **11.8 – 13.2**; install-time minimum is CUDA **11.8**. A CUDA 12.8+ driver is the
  pragmatic floor because it unlocks the widest wheel set (`sm70…sm120`). [bnb installation doc]
* Compute capability floor per feature: **CC ≥ 6.0** for NF4/FP4 and 8-bit optimizers,
  **CC ≥ 7.5** for `LLM.int8()` [bnb doc]; TorchAO float8 dynamic activation requires
  **CUDA ≥ 8.9** (or AMD MI350+/Intel XPU); `mxfp4`/`nvfp4` require **SM100+ (Blackwell)**;
  vLLM Marlin needs Turing+ (**SM 7.5**), with a MXFP4 exception on Turing. [torchao workflows +
  inference docs; vLLM quantization doc]
* CUDA toolkit **13.0+** only if we intend to use an Ada/Hopper-era wheel set with the newest
  targets; the safe default is the vendor-recommended driver for CUDA 12.8 and the CPU-side toolchain
  left untouched.
* **VRAM** `[INFERENCE]`, sizing from `memory-accounting.md` §7.2/§7.3 plus activations and optimizer
  state:
  * Tier 2 (TinyLlama-1.1B, ≥3 seeds, QLoRA/LR-QAT-class training + full evaluation): **≥ 24 GB**
    (7B-class headroom; a 1.1B model in bf16 is ~2.2 GB of weights, but QAT/LR-QAT keeps fp32
    master weights plus optimizer state and a dequantized compute path).
  * Tier 3 (0.6–1.7B second model): fits in the same 24 GB envelope.
  * Tier 4 (ViT/DeiT on CIFAR-100/ImageNet-100): **≥ 16 GB**.
  * Tier 5 (3B–7B, optional): **≥ 40 GB** for 7B bf16 QAT; a 24 GB card handles 3B and
    7B QLoRA-style work with reduced batch size.
  * For class-5 service numbers add headroom above model size for KV cache and concurrent batches.

**Software we would additionally unlock**

| Unlocked | Why it was blocked here |
|---|---|
| bitsandbytes 4-bit/8-bit CUDA kernels, NF4/FP4 | no NVIDIA device |
| torchao `Int4WeightOnlyConfig` (int4 tinygemm) via `mslk` | CUDA-only index package |
| GPTQ (GPTQModel) int4, AWQ int4, Marlin kernels | CUDA-only kernels |
| TorchAO float8 / nvfp4 / mxfp8 | CC-gated (≥ 8.9 / SM100+) |
| vLLM serving incl. quantized models; lm-eval `vllm` backend | no Windows wheel + CUDA |
| SmoothQuant CUTLASS int8 | CUDA-only |

**Before any Tier-2 run:** re-run §2's probes on the new host and re-issue this document with the
observed driver, CUDA runtime, VRAM and per-package versions; `docs/research/environment.md` (owned
by the orchestrator) must be re-audited first, per its own §5 policy.

---

## 6. Verified upstream pins (obtained 2026-10-08 via `gh api`)

| Upstream | Default branch commit | Used for |
|---|---|---|
| `Qualcomm-AI-research/LR-QAT` | `8795afe054cf951b714299e01083a1b354721829` | method baseline (deferred Tier 2) |
| `yxli2123/LoftQ` | `ae33fd4fd05fd4ba146555cd77c13d307eb4e9b3` | method baseline |
| `HanGuo97/lq-lora` | `c2424b3adc27197815da1ac9e1304565168d824d` | method baseline |
| `pytorch/ao` | `cff77b46ef85ba8472b4a286598073d6c01e18e5` | torchao 0.18.0 behaviour |
| `pytorch/pytorch` | `79f81f9e6d3b7c60549a35569dcbf1b8e295f9b3` | torch 2.14.1+cpu behaviour |
| `bitsandbytes-foundation/bitsandbytes` | `833649043474794b8fe7a4136e0c40faf077b2e0` | bnb 0.50.2 behaviour |
| `microsoft/onnxruntime` | `17be9c28b14dcd175c4dec16c46255886ad6cda0` | ORT 1.30.0 quantized CPUs ops |
| `ggml-org/llama.cpp` | `de7fa0a3c6a2e1b4cd9f22eb8d6bf5b12dbdb63b` | GGUF/k-quant CPU path |
| `huggingface/transformers` | `4cc2aa84301c9aa210b5513dab9fefae03981f6e` | transformers 5.19.0 |
| `huggingface/peft` | `8bb3e702976b48f16c1995c97b4cb32b038e1e15` | peft 0.21.2 |
| `EleutherAI/lm-evaluation-harness` | `d6de81643928d653435c431bae19945d41d32520` | harness 0.4.13 |
| `vllm-project/vllm` | `72d59adc2c76de8f77aad640c7a04d663a970735` | vLLM 0.31.0 |
| `mit-han-lab/smoothquant` | `c61476d728e42ae0d8a35e7e78494edcac3237b5` | SmoothQuant |
| `mit-han-lab/llm-awq` | `d6e797a42b9ef7778de8ee2352116e0f48a78d61` | AWQ (original) |
| `casper-hansen/AutoAWQ` | `88e4c76b20755db275574e6a03c83c84ba3bece5` | AutoAWQ (deprecated) |
| `ModelCloud/GPTQModel` | `d0e59f892b77228e6e9774fc4c43850410e362bb` | GPTQModel |
| `AutoGPTQ/AutoGPTQ` | `9f7d37072917ab3a7545835f23e808294a542153` | AutoGPTQ (unmaintained) |
| `iterative/dvc` | `56e59829512ff134aa269099a2099587b810b4dd` | DVC |
| `mlflow/mlflow` | `9c30f9c57b80b6a0af74a457333e7c12be081f26` | MLflow |
| `Vahe1994/AQLM` | `e79a896ed6656fe4ed06193d42d004e7d0bbdbb2` | 2-bit comparison context |

Where these overlap with `docs/research/upstream-lockfile.md` (LR-QAT, LoftQ, lq-lora, `pytorch/ao`,
`lm-evaluation-harness`, `peft`, `vllm`) the SHAs agree **exactly** — verified by string comparison
on 2026-10-08. The remaining pins (bitsandbytes, onnxruntime, llama.cpp, transformers, SmoothQuant,
llm-awq, AutoAWQ, GPTQModel, AutoGPTQ, DVC, MLflow, `pytorch/pytorch`, AQLM) are newly recorded
here and should be folded into the lockfile by its owner. They are *current* as of the date above
and MUST be re-verified (not assumed) if a lockfile is regenerated later.

---

## 7. `[UNVERIFIED]` register

Claims in this document that rest on documentation alone, plus the exact check that would settle
each. None of them is load-bearing for the Tier-0/1 scope.

| Claim | Source | How to settle |
|---|---|---|
| llama.cpp prebuilt Windows binaries run k-quant models on this CPU | upstream README | `winget install` / download release, `llama-cli` a tiny GGUF, record output |
| SmoothQuant's `fake_quant.py` / `ppl_eval.py` run on torch 2.14 CPU | upstream repo layout | port the module into a scratch venv, run the demo on OPT-125M |
| lm-evaluation-harness `hf` backend runs on torch CPU | upstream README | `pip install "lm_eval[hf]"` in the scratch venv, evaluate `hf-internal-testing/tiny-random-gpt2` on `hellaswag` (limit 10) |
| vLLM `0.31.0+cpu` wheel installs and serves BF16 in the Linux container | vLLM CPU install doc + release asset listing | run the documented `uv pip install <wheel> --torch-backend cpu`, then `vllm serve` a ≤1B model |
| ONNX Runtime `GatherBlockQuantized` (int4 embeddings) works on this CPU | ORT quantization doc (needs ORT ≥ 1.20) | build a tiny GatherBlockQuantized model and run it |
| vLLM's x86-CPU AWQ/GPTQ support applies to the `+cpu` wheel | vLLM quantization hardware table vs CPU install doc | install the CPU wheel and load a 4-bit AWQ model |
| `hqq`/`quanto`/`evaluate`/`datasets`/`safetensors` behave on Windows CPU | PyPI wheel tags only | install in the scratch venv and import |

---

## 8. Re-running these probes

```bash
# scratch env OUTSIDE the repository; nothing is written into the project environment
uv venv --python 3.11 "$TEMP/sq-probe"
uv pip install --python "$TEMP/sq-probe/Scripts/python.exe" torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$TEMP/sq-probe/Scripts/python.exe" torchao onnxruntime onnx onnx_ir numpy \
    bitsandbytes transformers accelerate peft
"$TEMP/sq-probe/Scripts/python.exe" "$TEMP/sq-probe/probe_quant.py"    # torch + torchao + ORT providers
"$TEMP/sq-probe/Scripts/python.exe" "$TEMP/sq-probe/probe_ort2.py"     # torchao intx + ORT int8/int4 kernels
"$TEMP/sq-probe/Scripts/python.exe" "$TEMP/sq-probe/probe_bnb.py"      # bitsandbytes on CPU
"$TEMP/sq-probe/Scripts/python.exe" "$TEMP/sq-probe/probe_hf.py"       # transformers + PEFT + 4-bit CPU load
"$TEMP/sq-probe/Scripts/python.exe" "$TEMP/sq-probe/probe_remeasure.py" # corrected ORT API + artifact bytes (results valid)
"$TEMP/sq-probe/Scripts/python.exe" "$TEMP/sq-probe/verify_mem.py"     # memory-accounting arithmetic

uv --version && docker version && docker info
docker run --rm alpine:3.20 uname -a && docker run --rm python:3.11-slim python -V
for pkg in bitsandbytes vllm auto-gptq onnxruntime torchao; do
  curl -s "https://pypi.org/pypi/$pkg/json" | python -c "import sys,json;d=json.load(sys.stdin);print(d['info']['version'], sorted({u['filename'] for u in d['releases'][d['info']['version']]}))"
done
```

The probe scripts live only in `$TEMP/sq-probe` and are intentionally **not** committed: they probe
third-party backends, and `AGENTS.md` §7 forbids committing machine-specific paths or caches. If a
permanent capability regression test is wanted later, it belongs in
`tests/integration/test_backend_capability.py` with the environment-dependent parts skipped when the
backend is absent.
