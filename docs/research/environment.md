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

- Local compute is the default and the only compute currently available.
- Any Tier 2+ run requires an external GPU. That resource is **not yet secured**; until it is,
  Tier 2–5 results are *planned*, never *completed*.
- Cost policy for cloud GPU: not yet decided — must be documented here before the first paid run.
- Scope changes forced by hardware MUST be recorded as timestamped amendments in
  `docs/research/preregistration-amendments.md`.

## 6. Reproduce this audit

```bash
git --version && uv --version && python --version && gh auth status
gh repo view OussemaHarrabi/SpectraQuant --json name,visibility,isEmpty,defaultBranchRef
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Processor | Select Name,NumberOfCores,NumberOfLogicalProcessors; Get-CimInstance Win32_ComputerSystem | Select TotalPhysicalMemory; Get-CimInstance Win32_VideoController | Select Name,AdapterRAM,DriverVersion; Get-CimInstance Win32_LogicalDisk -Filter \"DeviceID='C:'\" | Select Size,FreeSpace"
uv run python scripts/reproduce/environment_probe.py   # added with the scaffold
```
