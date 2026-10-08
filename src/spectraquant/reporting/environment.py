"""Hardware, software and device capture for run manifests and ``spectraquant env``.

Everything here is *measured*, not inferred: nothing is estimated, and an unavailable value is
returned as ``None`` rather than as a plausible-looking number. The workstation has no CUDA device
(see ``docs/research/environment.md``), so ``gpu_available`` is ``False`` and no CUDA-only backend
is ever reported as usable.
"""

from __future__ import annotations

import ctypes
import os
import platform
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from typing import Any

import torch

__all__ = [
    "collect_hardware",
    "collect_software",
    "environment_report",
    "process_peak_rss_mb",
    "resolve_device",
    "total_ram_mb",
]

#: Runtime distributions recorded in every manifest (missing ones become ``None``).
TRACKED_DISTRIBUTIONS: tuple[str, ...] = (
    "spectraquant",
    "torch",
    "numpy",
    "hydra-core",
    "omegaconf",
    "pydantic",
    "typer",
    "rich",
    "jsonschema",
    "mlflow-skinny",
)


def resolve_device() -> str:
    """Return the torch device this process must use: ``"cuda:0"`` if usable, else ``"cpu"``."""
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _dist_version_or_none(name: str) -> str | None:
    try:
        return _dist_version(name)
    except PackageNotFoundError:
        return None
    except Exception:  # pragma: no cover - corrupt metadata must not break a run
        return None


def _windows_kernel32() -> Any | None:
    """Return a ``kernel32`` handle on Windows, else ``None`` (kept pyright-safe on Linux CI)."""
    win_dll = getattr(ctypes, "WinDLL", None)
    if win_dll is None:  # pragma: no cover - non-Windows
        return None
    try:
        return win_dll("kernel32", use_last_error=True)
    except OSError:  # pragma: no cover - unusual Windows install
        return None


def total_ram_mb() -> float | None:
    """Total physical memory in MiB, or ``None`` when the platform does not expose it."""
    if sys.platform == "win32":

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

        kernel32 = _windows_kernel32()
        if kernel32 is None:  # pragma: no cover
            return None
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        function = getattr(kernel32, "GlobalMemoryStatusEx", None)
        if function is None:  # pragma: no cover
            return None
        function.argtypes = [ctypes.POINTER(_MemoryStatusEx)]
        function.restype = ctypes.c_int
        if not function(ctypes.byref(status)):
            return None
        return float(status.ullTotalPhys) / (1024.0 * 1024.0)

    try:  # POSIX
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):  # pragma: no cover - exotic POSIX
        return None
    if not isinstance(page_size, int) or not isinstance(pages, int):
        return None
    return float(page_size * pages) / (1024.0 * 1024.0)


def process_peak_rss_mb() -> float | None:
    """Peak resident memory of *this process* in MiB, or ``None`` when unavailable.

    Windows: ``K32GetProcessMemoryInfo`` → ``PeakWorkingSetSize`` (falling back to the legacy
    ``psapi.GetProcessMemoryInfo`` export).
    POSIX: ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` (KiB on Linux, bytes on macOS).

    This measures the whole process, not only tensors, and is therefore an upper bound on what the
    experiment itself used. It is a measurement (never an estimate), reported verbatim.
    """
    if sys.platform == "win32":

        class _ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = _windows_kernel32()
        if kernel32 is None:  # pragma: no cover
            return None
        function = getattr(kernel32, "K32GetProcessMemoryInfo", None)
        if function is None:  # pragma: no cover - pre-Windows 7
            psapi_dll = getattr(ctypes, "WinDLL", None)
            if psapi_dll is None:
                return None
            try:
                function = psapi_dll("psapi", use_last_error=True).GetProcessMemoryInfo
            except (OSError, AttributeError):
                return None
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        function.restype = ctypes.c_int

        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
        if not function(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return None
        return float(counters.PeakWorkingSetSize) / (1024.0 * 1024.0)

    try:
        import resource
    except ImportError:  # pragma: no cover - Windows already handled above
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return float(usage) / divisor


def _gpu_devices() -> list[str]:
    if not torch.cuda.is_available():  # pragma: no cover - no CUDA on the workstation of record
        return []
    return [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]


def collect_hardware() -> dict[str, Any]:
    """Return the manifest ``hardware`` block."""
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count_logical": os.cpu_count(),
        "ram_total_mb": total_ram_mb(),
        "gpu_available": bool(torch.cuda.is_available()),
        "gpu_devices": _gpu_devices(),
        "torch_device": resolve_device(),
        "torch_threads": torch.get_num_threads(),
    }


def collect_software() -> dict[str, str | None]:
    """Return the manifest ``software`` block (versions of every runtime dependency)."""
    software: dict[str, str | None] = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "torch_cuda_build": torch.version.cuda,
        "torch_hip_build": getattr(torch.version, "hip", None),
        "operator_system": platform.system(),
    }
    for name in TRACKED_DISTRIBUTIONS:
        software[name] = _dist_version_or_none(name)
    return software


def environment_report() -> dict[str, Any]:
    """Aggregate report printed by ``spectraquant env``.

    Includes the measurement classes this workstation can actually produce, so that the CLI never
    implies class 4/5 evidence is obtainable locally (AGENTS.md sections 2 and 5).
    """
    from spectraquant import __version__

    return {
        "spectraquant_version": __version__,
        "device": resolve_device(),
        "cuda_available": bool(torch.cuda.is_available()),
        "measurement_classes": {
            # Produced by this repository today.
            "locally_producible": [1, 2, 3],
            # Class 4 is split by hardware (AGENTS.md section 5): 4-CPU needs a real CPU low-bit
            # kernel executing an artifact SpectraQuant serialized itself, against an fp32 baseline
            # measured in the same session. The kernel path does not exist in this scaffold.
            "locally_producible_with_conditions": ["4-CPU"],
            "not_producible_locally": ["4-GPU", 5],
            "class_4_cpu_available_in_principle": True,
            "class_4_cpu_implemented": False,
        },
        "measurement_classes_note": (
            "Classes 1-3 are producible locally today (class 3 only for formats this repository can "
            "actually serialize). Class 4-CPU is available in principle under AGENTS.md section 5 "
            "(permitted backends: ONNX Runtime MatMulNBits/MatMulInteger, torchao intx weight-only) "
            "but no such kernel path exists in this scaffold yet, so no class 4 number is produced. "
            "Class 4-GPU and class 5 require the cloud substrate (AGENTS.md section 2b)."
        ),
        "hardware": collect_hardware(),
        "software": collect_software(),
        "peak_process_rss_mb": process_peak_rss_mb(),
    }
