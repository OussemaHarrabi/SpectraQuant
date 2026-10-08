# Environment audit — development workstation

Measured 2026-10-08 by the orchestrator on the machine of record (`C:\Users\oussa\oussema\SpectraQuant`).
Values below are **observed**, not inferred. Anything not directly observed is marked `[UNKNOWN]`.

## 1. Hardware

| Item | Observed value | Source |
|---|---|---|
| CPU | AMD Ryzen AI 7 350 w/ Radeon 860M | `Win32_Processor.Name` |
| Cores / threads | 8 physical / 16 logical | `Win32_Processor` |
| RAM | 16,233,910,272 bytes (16.23 GB ≈ 15.1 GiB) | `Win32_ComputerSystem.TotalPhysicalMemory` |
| Discrete GPU | none detected (no `nvidia-smi` on PATH; single display adapter reported) | `Win32_VideoController` |
| Display adapter | AMD Radeon(TM) 860M Graphics, `AdapterRAM` 536,870,912 (reported iGPU carve-out) | `Win32_VideoController` |
| Driver | 32.0.22032.3003 | `Win32_VideoController.DriverVersion` |
| Disk C: | 1,021,623,734,272 bytes total, 695,586,541,568 free (≈ 648 GiB free) | `Win32_LogicalDisk` |
| CUDA toolkit (`nvcc`) | not installed | `Get-Command nvcc` → empty |

### Interpretation (binding for scope)

- **No NVIDIA GPU ⇒ no CUDA.** PyTorch CUDA builds, bitsandbytes 4-bit CUDA kernels, TorchAO CUDA
  kernels, GPTQ/AWQ GPU kernels, vLLM CUDA, FlashAttention are **unavailable on this machine**.
- The AMD Radeon 860M is an integrated GPU sharing system RAM. ROCm has no Windows support for this
  part; DirectML (`torch-directml`) is unmaintained and cannot be a research-grade backend.
  GPU acceleration is therefore treated as **absent** rather than emulated.
- 16.23 GB RAM caps local work at: Tier 0, Tier 1, and int8/int4 *fake-quantization* + *CPU packed
  storage* experiments. A 1.1B-parameter model in fp16 needs ≈2.2 GB of weights alone, plus
  activations/optimizer state — not viable for training locally, and only marginally viable for
  fp32-free CPU forward evaluation.

### Corrected on 2026-10-08: a real CPU low-bit kernel path exists

The initial reading ("no CUDA ⇒ only fake quantization and analytical estimates") was **too strong**
and was corrected by measurement (`docs/research/backend-capability.md` §2.4, reproduced independently
by the orchestrator):

- `torch` 2.14.1+cpu, `torch.cuda.is_available() == False`, `mkldnn`/`mkl` present, quantized engine
  `onednn` only (no fbgemm/qnnpack on Windows).
- **ONNX Runtime 1.30.0 CPU executes real low-bit kernels**: `MatMulNBits` (int4 weight-only, domain
  `com.microsoft`) and `MatMulInteger` (dynamic int8). Verified: an int4 artifact we serialize holds a
  1024 B payload (2048 four-bit values, bit-exact) plus 256 B of fp32 scales inside a 1501 B file; the
  int4 session runs and returns results (max abs error 2.37 on N(0,1) weights), int8 gives 0.216.
- **torchao 0.18.0 `IntxWeightOnlyConfig(torch.int4, PerGroup(32))` works on CPU**;
  `Int4WeightOnlyConfig` (int4 tinygemm) fails with `Requires mslk >= 1.0.0` (CUDA-index package).
- **bitsandbytes 0.50.2 has a Windows CPU backend** (NF4 storage + dequantized fp32 compute).
- **Docker Linux containers work** (kernel 6.18.33.2-WSL2, 16 CPUs, 7.318 GiB container RAM).

Consequence: measurement class **4-CPU** is available for artifacts SpectraQuant serializes itself
(`AGENTS.md` §5). Class 4-GPU and class 5 remain unavailable. This is a scope *expansion*, recorded
in `docs/research/preregistration-amendments.md`; nothing in the Tier 2–5 plan changes.

## 2. Software

| Item | Observed value |
|---|---|
| OS | Microsoft Windows 11 Famille, version 10.0.26200 |
| Shell | POSIX-style shell provided by the agent runtime (git 2.55.0.windows.3) |
| System Python | 3.13.14 (`C:\Users\oussa\AppData\Local\Programs\Python\Python313\python.exe`), no torch installed |
| Project Python | 3.11.16, installed by uv (`%APPDATA%\uv\python\cpython-3.11.16-windows-x86_64-none`) |
| uv | 0.12.15 (x86_64-pc-windows-msvc) |
| git | 2.55.0.windows.3; local identity `Oussema Harrabi <189178358+OussemaHarrabi@users.noreply.github.com>`; `core.autocrlf=false`, `pull.rebase=true`, `push.autoSetupRemote=true` |
| GitHub CLI | present (`gh`), authenticated as `OussemaHarrabi`, token scopes `gist, read:org, repo, workflow` |
| Docker | Docker Desktop CLI present (`docker.exe`); Linux container execution to be verified per use |
| PyTorch | not installed (project env pins the CPU wheel index) |
| CUDA | [UNKNOWN]/absent — no toolkit, no driver |
| sqlite3 | present (winget SQLite) |

## 3. Network reachability

Verified by HTTP status: `pypi.org/simple` 200, `download.pytorch.org/whl/cpu` 200, `github.com` 200,
`arxiv.org/abs/2406.06385` 200. Outbound HTTPS is open; no proxy observed.

## 4. Remote repository state (verified before bootstrap)

- `gh repo view OussemaHarrabi/SpectraQuant` → `visibility: PUBLIC`, `isEmpty: true`, `createdAt:
  2026-10-08T19:55:34Z`, `defaultBranchRef: ""` (no commits, no default branch yet),
  `licenseInfo: null`.
- `git ls-remote https://github.com/OussemaHarrabi/SpectraQuant.git` → empty output (consistent with
  an empty repository), no authentication error ⇒ HTTPS access with the `gh` credential works.
- Local directory contained only `.repowise/` (agent tooling index), no user work. Bootstrap was
  therefore non-destructive: `git init -b main`, remote added, branch `infra/bootstrap` created.

## 5. Compute policy

**Superseded 2026-10-08 by the cloud-compute execution policy (`AGENTS.md` §2b).** The local machine is
an orchestration and correctness host only:

- local: repository management, CPU unit/property tests, tiny synthetic fixtures, static analysis and
  type checking, configuration validation, notebook generation, result analysis, figures, tables,
  reports, and the CI smoke fixture;
- cloud (Google Colab as the developer's chosen vehicle; Kaggle Notebooks preferred for unattended
  runs; Colab Enterprise optional with an authorized GCP project): **all** model training,
  quantization-aware training, large-scale inference, and GPU evaluation, i.e. Tiers 1–5 of §6;
- every cloud run produces `run_manifest.json` (config, lock, git SHA, dataset versions, seeds,
  hardware, GPU-hours, status, metrics, artifact checksums) and is collected + checksum-validated
  locally before it enters the research record;
- no paid cloud resource may be started without the user's prior authorization and a stated maximum
  estimated cost; free tiers are the default;
- credentials live only in platform secrets/environment variables.

Consequence for the earlier plan: Tier 1 (tiny-transformer experiments) is **not** a local workload.
Locally we still hold: Tier 0 exactness, the class 4-CPU kernel path (a measurement, not training),
the reproduction *fixtures*, and all analysis. Tier 2+ remains gated on cloud GPU availability and
must never be reported as completed from local execution.

## 6. Reproduce this audit

```bash
git --version && uv --version && python --version && gh auth status
gh repo view OussemaHarrabi/SpectraQuant --json name,visibility,isEmpty,defaultBranchRef
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Processor | Select Name,NumberOfCores,NumberOfLogicalProcessors; Get-CimInstance Win32_ComputerSystem | Select TotalPhysicalMemory; Get-CimInstance Win32_VideoController | Select Name,AdapterRAM,DriverVersion; Get-CimInstance Win32_LogicalDisk -Filter \"DeviceID='C:'\" | Select Size,FreeSpace"
uv run python scripts/reproduce/environment_probe.py   # added with the scaffold
```
