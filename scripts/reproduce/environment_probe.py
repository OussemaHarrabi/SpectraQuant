#!/usr/bin/env python
"""Standalone hardware/software probe for the SpectraQuant workstation of record.

Referenced by ``docs/research/environment.md`` section 6. It is deliberately self-contained: the
standard library only, plus an *optional* ``torch`` import (used to report the resolved device and
the CUDA build, if any). It never fails because a probe is unavailable — the field becomes ``null``
with a note instead.

Usage::

    python scripts/reproduce/environment_probe.py          # human-readable
    python scripts/reproduce/environment_probe.py --json   # machine-readable

Nothing is written to disk; the output is meant to be pasted into the environment audit.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import subprocess
import sys
from typing import Any

TIMEOUT_S = 20.0
_IS_WINDOWS = sys.platform == "win32"
_IS_MACOS = sys.platform == "darwin"


def _run(command: list[str]) -> str | None:
    """Run a command and return stripped stdout, or ``None`` if it is unavailable/failing."""
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    output = completed.stdout.strip()
    return output or None


def _powershell(script: str) -> str | None:
    """Run a PowerShell snippet on Windows; ``None`` elsewhere or on failure."""
    if not _IS_WINDOWS:
        return None
    return _run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script])


def cpu_info() -> dict[str, Any]:
    """CPU model, physical/logical core counts."""
    name: str | None = None
    if _IS_WINDOWS:
        name = _powershell("(Get-CimInstance Win32_Processor).Name")
    elif _IS_MACOS:
        name = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    else:
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.lower().startswith("model name"):
                        name = line.split(":", 1)[1].strip()
                        break
        except OSError:
            name = None

    physical: int | None = None
    if _IS_WINDOWS:
        raw = _powershell("(Get-CimInstance Win32_Processor).NumberOfCores")
        physical = int(raw) if raw and raw.isdigit() else None
    elif not _IS_MACOS:
        try:
            import re

            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                text = handle.read()
            cores = set(re.findall(r"core id\s*:\s*(\d+)", text))
            physical = len(cores) or None
        except OSError:
            physical = None

    return {
        "name": name or platform.processor() or None,
        "architecture": platform.machine(),
        "physical_cores": physical,
        "logical_cores": os.cpu_count(),
    }


def memory_info() -> dict[str, Any]:
    """Total physical memory in MiB (and GiB), or ``None`` when unmeasurable."""
    total_bytes: int | None = None

    if _IS_WINDOWS:

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        win_dll = getattr(ctypes, "WinDLL", None)
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if win_dll is not None:
            try:
                kernel32 = win_dll("kernel32", use_last_error=True)
                function = getattr(kernel32, "GlobalMemoryStatusEx", None)
                if function is not None:
                    function.argtypes = [ctypes.POINTER(_MemoryStatusEx)]
                    function.restype = ctypes.c_int
                    if function(ctypes.byref(status)):
                        total_bytes = int(status.ullTotalPhys)
            except OSError:
                total_bytes = None
    else:
        try:
            total_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (AttributeError, ValueError, OSError):
            total_bytes = None

    if total_bytes is None:
        return {"total_mb": None, "total_gib": None}
    total_mb = total_bytes / (1024.0 * 1024.0)
    return {"total_mb": round(total_mb, 1), "total_gib": round(total_mb / 1024.0, 2)}


def gpu_info() -> dict[str, Any]:
    """Discrete GPU inventory (NVIDIA via ``nvidia-smi``, Windows via WMI, Linux via ``lspci``)."""
    nvidia = _run(["nvidia-smi", "-L"])
    devices: list[dict[str, Any]] = []

    if _IS_WINDOWS:
        raw = _powershell(
            "Get-CimInstance Win32_VideoController | "
            "Select-Object Name,DriverVersion,AdapterRAM | ConvertTo-Json -Compress"
        )
        if raw:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = None
            for item in parsed if isinstance(parsed, list) else [parsed]:
                if isinstance(item, dict) and item.get("Name"):
                    devices.append(
                        {
                            "name": item.get("Name"),
                            "driver": item.get("DriverVersion"),
                            "adapter_ram_bytes": item.get("AdapterRAM"),
                            "source": "Win32_VideoController",
                        }
                    )
    elif not _IS_MACOS:
        raw = _run(["lspci"])
        if raw:
            for line in raw.splitlines():
                lowered = line.lower()
                if "vga compatible controller" in lowered or "3d controller" in lowered:
                    devices.append({"name": line.split(":", 2)[-1].strip(), "source": "lspci"})

    cuda_available: bool | None = None
    torch_version: str | None = None
    torch_cuda_build: str | None = None
    try:
        import torch

        torch_version = torch.__version__
        torch_cuda_build = torch.version.cuda
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = None

    return {
        "nvidia_smi": nvidia,
        "devices": devices,
        "cuda_available": cuda_available,
        "torch": torch_version,
        "torch_cuda_build": torch_cuda_build,
        "note": (
            "No CUDA device means measurement classes 4-5 are unavailable; see "
            "docs/research/environment.md"
        ),
    }


def os_info() -> dict[str, Any]:
    """Operating system identification."""
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
    }


def toolchain_info() -> dict[str, Any]:
    """Versions of the tools the workflow depends on (``null`` when not installed)."""
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "python_implementation": platform.python_implementation(),
        "uv": _run(["uv", "--version"]),
        "git": _run(["git", "--version"]),
        "docker": _run(["docker", "--version"]),
        "gh": _run(["gh", "--version"]),
        "nvidia_smi_present": _run(["nvidia-smi", "--version"]) is not None,
        "nvcc": _run(["nvcc", "--version"]),
    }


def probe() -> dict[str, Any]:
    """Collect every probe into one report."""
    return {
        "os": os_info(),
        "cpu": cpu_info(),
        "memory": memory_info(),
        "gpu": gpu_info(),
        "toolchain": toolchain_info(),
    }


def _print_human(report: dict[str, Any]) -> None:
    for section, values in report.items():
        print(f"[{section}]")
        for key, value in values.items():
            print(f"  {key}: {value}")
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="emit JSON instead of plain text")
    args = parser.parse_args(argv)

    report = probe()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    else:
        _print_human(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
