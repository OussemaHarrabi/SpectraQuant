"""Class-4-CPU kernel measurement protocol (``docs/protocols/benchmark-protocol.md`` §8).

The only class-4 measurement available on this workstation: a real CPU low-bit kernel executing a
low-bit container **SpectraQuant serialized itself** (``AGENTS.md`` §5). This module implements the
protocol's mandatory mechanics — thread pinning with read-back, a same-process/same-session fp32
baseline, disclosed warm-ups, repeats with a median point estimate, and a **loud** failure when the
runtime did not actually select the low-bit kernel (no silent dequantized fallback; §6, §8.2.7).

Every number produced here is measurement class **4-CPU** and MUST be reported with the §8.3 scope
line (see :data:`CPU_SCOPE_LINE`); it is never comparable to published GPU latency/throughput figures.

Dependencies. ``onnxruntime`` is the optional ``onnx`` extra; it is imported lazily and a missing
extra raises :class:`~spectraquant.quantization.onnx_export.OnnxExtraMissingError` naming it.

Numerical / resource limitations.
    * Thread pinning sets ``torch.set_num_threads``, ``OMP_NUM_THREADS`` and the ONNX Runtime
      ``intra_op_num_threads`` / ``inter_op_num_threads`` explicitly and **reads each back**; the
      effective values, not the requested ones, are what a claim carries.
    * ``p95`` requires ≥ 100 samples (protocol §2.7); below that the returned statistics set
      ``percentiles_indicative`` and the report must quote ``median``/``min``/``max``/``IQR``.
    * Core counts and RAM are read from the platform (Windows registry + ``GetLogicalProcessorInformationEx``
      + ``GlobalMemoryStatusEx``); where a value cannot be read it is ``None`` and is never guessed.
"""

from __future__ import annotations

import hashlib
import os
import platform
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from spectraquant.quantization.onnx_export import OnnxExtraMissingError

__all__ = [
    "CPU_SCOPE_LINE",
    "BackendCapabilityError",
    "CpuSession",
    "LatencyStats",
    "ThreadPinning",
    "WorkloadResult",
    "assert_lowbit_kernel_selected",
    "core_counts",
    "cpu_model",
    "error_statistics",
    "measure_latency",
    "open_cpu_session",
    "physical_memory_bytes",
    "pin_cpu_threads",
    "run_cpu_workload",
]

#: The mandatory §8.3 scope line, formatted with the CPU model and thread count.
CPU_SCOPE_LINE = (
    "CPU-only measurement on {cpu_model} with {threads} threads; not comparable to published GPU "
    "latency or throughput figures."
)

_DEFAULT_PROVIDERS: tuple[str, ...] = ("CPUExecutionProvider",)


class BackendCapabilityError(RuntimeError):
    """The requested backend/format was not executed — a run-invalidating failure (protocol §6).

    Raised when the runtime did not select the requested execution provider or the expected low-bit
    kernel, i.e. when a silent substitution or a dequantized fallback would otherwise be reported as
    the requested configuration.
    """


def cpu_model() -> str:
    """Return the CPU model string, preferring the Windows registry marketing name.

    Returns:
        A human-readable model string; an empty string is never returned.
    """
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            name = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
            if name:
                return name
        except OSError:  # pragma: no cover - registry access can fail on locked-down hosts
            pass
    return platform.processor() or platform.machine() or "unknown"


def _windows_physical_cores() -> int | None:
    """Physical core count via ``GetLogicalProcessorInformationEx`` (Windows only).

    ``ctypes.windll`` exists only on Windows, so it is fetched with ``getattr`` rather than as an
    attribute: a direct access is a static error on any other platform (CI runs on Linux, where
    Pyright flags it even though the call site is guarded), and the guard here makes the platform
    requirement explicit instead of relying on the caller.
    """
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        windll = getattr(ctypes, "windll", None)
        if windll is None:  # pragma: no cover - win32 always provides windll
            return None
        kernel32 = windll.kernel32
        length = wintypes.DWORD(0)
        # First call is expected to fail with ERROR_INSUFFICIENT_BUFFER and fill ``length``.
        kernel32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(length))
        if length.value == 0:
            return None
        buffer = ctypes.create_string_buffer(length.value)
        if not kernel32.GetLogicalProcessorInformationEx(0, buffer, ctypes.byref(length)):
            return None
        data = bytes(buffer)
        count = 0
        offset = 0
        while offset + 8 <= length.value:
            relationship = int.from_bytes(data[offset : offset + 4], "little")
            size = int.from_bytes(data[offset + 4 : offset + 8], "little")
            if relationship == 0:  # RelationProcessorCore
                count += 1
            if size <= 0:
                break
            offset += size
        return count or None
    except Exception:  # pragma: no cover - platform-specific best effort
        return None


def core_counts() -> tuple[int | None, int | None]:
    """Return ``(physical_cores, logical_cores)``; an unreadable value is ``None``, never guessed."""
    logical = os.cpu_count()
    physical = _windows_physical_cores() if sys.platform == "win32" else None
    return physical, logical


def physical_memory_bytes() -> int | None:
    """Total physical RAM in bytes, or ``None`` when the platform call is unavailable."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", wintypes.DWORD),
                    ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys)
            return None
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except Exception:  # pragma: no cover - platform-specific best effort
        return None


@dataclass(frozen=True)
class ThreadPinning:
    """The thread configuration of a class-4-CPU run, with every value read back.

    ``requested`` is what we set; ``torch_effective`` / ``ort_intra_effective`` /
    ``ort_inter_effective`` / ``omp_num_threads`` are read back from the libraries and the process
    environment. A claim carries the effective values (protocol §2.11, §8.2.1).
    """

    requested: int
    inter_op_threads: int
    torch_effective: int
    ort_intra_effective: int | None
    ort_inter_effective: int | None
    omp_num_threads: str | None
    knobs: tuple[str, ...]
    cpu_model: str
    physical_cores: int | None
    logical_cores: int | None

    def to_dict(self) -> dict[str, object]:
        """Return the thread block as a JSON-serializable dict (protocol §5 ``bench.threads``)."""
        return {
            "requested": self.requested,
            "inter_op_threads": self.inter_op_threads,
            "effective": self.ort_intra_effective
            if self.ort_intra_effective is not None
            else self.torch_effective,
            "torch_effective": self.torch_effective,
            "ort_intra_effective": self.ort_intra_effective,
            "ort_inter_effective": self.ort_inter_effective,
            "omp_num_threads": self.omp_num_threads,
            "knobs": list(self.knobs),
            "cpu_model": self.cpu_model,
            "physical_cores": self.physical_cores,
            "logical_cores": self.logical_cores,
        }


def pin_cpu_threads(num_threads: int, *, inter_op_threads: int = 1) -> ThreadPinning:
    """Pin the CPU thread count for torch and the process environment, then read it back.

    Args:
        num_threads: Threads for the measured path (``torch.set_num_threads`` and
            ``OMP_NUM_THREADS``).
        inter_op_threads: ONNX Runtime inter-op threads (recorded; applied by
            :func:`open_cpu_session`).

    Returns:
        A :class:`ThreadPinning` whose ``torch_effective`` and ``omp_num_threads`` are read back
        after setting.

    Raises:
        ValueError: If ``num_threads`` is not positive.
    """
    import torch

    if num_threads < 1:
        raise ValueError(f"num_threads must be >= 1, got {num_threads}")
    torch.set_num_threads(int(num_threads))
    os.environ["OMP_NUM_THREADS"] = str(int(num_threads))
    physical, logical = core_counts()
    knobs = (
        f"torch.set_num_threads({int(num_threads)})",
        f"OMP_NUM_THREADS={int(num_threads)}",
        f"onnxruntime.SessionOptions.intra_op_num_threads={int(num_threads)}",
        f"onnxruntime.SessionOptions.inter_op_num_threads={int(inter_op_threads)}",
    )
    return ThreadPinning(
        requested=int(num_threads),
        inter_op_threads=int(inter_op_threads),
        torch_effective=int(torch.get_num_threads()),
        ort_intra_effective=None,
        ort_inter_effective=None,
        omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
        knobs=knobs,
        cpu_model=cpu_model(),
        physical_cores=physical,
        logical_cores=logical,
    )


def assert_lowbit_kernel_selected(
    model_path: str | Path, *, op_type: str, domain: str
) -> tuple[tuple[str, str], ...]:
    """Assert the serialized graph actually contains the expected kernel op (protocol §8.2.7).

    Args:
        model_path: The ONNX container that will be executed.
        op_type: Expected run-time op, e.g. ``"MatMulNBits"`` or ``"MatMulInteger"``.
        domain: Expected op domain, e.g. ``"com.microsoft"`` (empty string for the default domain).

    Returns:
        All ``(op_type, domain)`` pairs in the graph, for the record.

    Raises:
        OnnxExtraMissingError: If the optional ``onnx`` extra is not installed.
        BackendCapabilityError: If the expected op/domain is absent — the run would not execute the
            requested kernel and is invalid for a class-4 claim.
    """
    try:
        import onnx
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise OnnxExtraMissingError(
            "kernel verification requires the optional 'onnx' extra (`uv sync --extra onnx`)"
        ) from exc
    graph = Path(model_path)
    if not graph.is_file():
        raise FileNotFoundError(f"no such ONNX container: {graph}")
    model = onnx.load(str(graph))
    nodes = tuple((node.op_type, node.domain) for node in model.graph.node)
    if (op_type, domain) not in nodes:
        raise BackendCapabilityError(
            f"{graph.name} does not contain the expected low-bit kernel ({op_type!r}, {domain!r}); "
            f"nodes present: {nodes}. The requested kernel was not selected — the run is invalid for "
            "a class-4 claim (silent dequantized fallback is forbidden: protocol sections 2.2/6)."
        )
    return nodes


@dataclass(frozen=True)
class CpuSession:
    """An ONNX Runtime CPU session whose provider and kernel were verified."""

    model_path: str
    session: object
    providers: tuple[str, ...]
    threads: ThreadPinning
    nodes: tuple[tuple[str, str], ...]
    dequantized_path: bool = False

    def to_dict(self) -> dict[str, object]:
        """Return the session record as a JSON-serializable dict."""
        return {
            "model_path": self.model_path,
            "providers": list(self.providers),
            "nodes": [{"op_type": op, "domain": domain} for op, domain in self.nodes],
            "dequantized_path": self.dequantized_path,
        }


def open_cpu_session(
    model_path: str | Path,
    *,
    pinning: ThreadPinning,
    op_type: str,
    domain: str,
    providers: tuple[str, ...] = _DEFAULT_PROVIDERS,
) -> CpuSession:
    """Open a CPU-only ONNX Runtime session with pinned threads and a verified kernel.

    The session options set ``intra_op_num_threads`` / ``inter_op_num_threads`` explicitly and read
    them back. The **effective** provider list must equal the requested one and be CPU-only, and the
    graph must contain ``(op_type, domain)`` — otherwise :class:`BackendCapabilityError` is raised,
    because a silently substituted provider or a dequantized path is not a class-4 result (§6, §8.2).

    Raises:
        OnnxExtraMissingError: If the optional ``onnx`` / ``onnxruntime`` extra is missing.
        BackendCapabilityError: On provider substitution or a missing kernel op.
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise OnnxExtraMissingError(
            "class-4-CPU execution requires the optional 'onnx' extra "
            "(`uv sync --extra onnx`, onnxruntime~=1.30.0)"
        ) from exc
    graph = Path(model_path)
    if not graph.is_file():
        raise FileNotFoundError(f"no such ONNX container: {graph}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = int(pinning.requested)
    options.inter_op_num_threads = int(pinning.inter_op_threads)
    session = ort.InferenceSession(str(graph), options, providers=list(providers))
    effective = tuple(session.get_providers())
    if effective != tuple(providers) or "CPUExecutionProvider" not in effective:
        raise BackendCapabilityError(
            f"requested providers {tuple(providers)} but the session selected {effective}; a "
            "substituted provider is a backend-capability failure, not a class-4-CPU result "
            "(protocol section 6)."
        )
    nodes = assert_lowbit_kernel_selected(graph, op_type=op_type, domain=domain)
    threads = replace(
        pinning,
        ort_intra_effective=int(options.intra_op_num_threads),
        ort_inter_effective=int(options.inter_op_num_threads),
    )
    return CpuSession(
        model_path=str(graph),
        session=session,
        providers=effective,
        threads=threads,
        nodes=nodes,
    )


@dataclass(frozen=True)
class LatencyStats:
    """Latency statistics over the measured repeats of one workload (class 4-CPU)."""

    warmup_iters: int
    repeats: int
    samples: int
    median_ns: float
    min_ns: int
    max_ns: int
    p50_ns: float
    p95_ns: float
    iqr_ns: float
    percentiles_indicative: bool
    raw_ns: tuple[int, ...]
    timer: str
    sync_method: str

    def to_dict(self) -> dict[str, object]:
        """Return the latency record as a JSON-serializable dict."""
        return {
            "warmup_iters": self.warmup_iters,
            "repeats": self.repeats,
            "samples": self.samples,
            "median_ns": self.median_ns,
            "p50_ns": self.p50_ns,
            "p95_ns": self.p95_ns,
            "min_ns": self.min_ns,
            "max_ns": self.max_ns,
            "iqr_ns": self.iqr_ns,
            "percentiles_indicative": self.percentiles_indicative,
            "raw_ns": list(self.raw_ns),
            "timer": self.timer,
            "sync_method": self.sync_method,
        }


def measure_latency(
    run_once: Callable[[], object],
    *,
    warmup_iters: int = 10,
    repeats: int = 5,
    timer: str = "time.perf_counter_ns",
    sync_method: str = "ort_run_blocking",
) -> LatencyStats:
    """Time ``run_once`` with disclosed warm-ups and stored per-repeat samples.

    Args:
        run_once: A zero-argument callable performing one complete, synchronised execution
            (``InferenceSession.run`` is blocking, so the timing region is genuinely complete).
        warmup_iters: Discarded iterations, executed before any timing (protocol §2.2 requires
            ``>= 10`` for a class-4-CPU number).
        repeats: Measured iterations (protocol §2.5 requires ``>= 5``).
        timer: The wall-clock timer name recorded with the number.
        sync_method: The synchronisation primitive recorded with the number (protocol §2.3).

    Returns:
        :class:`LatencyStats` with the median point estimate plus per-repeat records.

    Raises:
        ValueError: If ``warmup_iters < 0`` or ``repeats < 1``.
    """
    if warmup_iters < 0:
        raise ValueError(f"warmup_iters must be >= 0, got {warmup_iters}")
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    for _ in range(int(warmup_iters)):
        run_once()
    samples: list[int] = []
    for _ in range(int(repeats)):
        start = time.perf_counter_ns()
        run_once()
        samples.append(time.perf_counter_ns() - start)
    values = np.asarray(samples, dtype=np.float64)
    p50, p95, p25, p75 = (float(v) for v in np.percentile(values, [50, 95, 25, 75]))
    return LatencyStats(
        warmup_iters=int(warmup_iters),
        repeats=int(repeats),
        samples=len(samples),
        median_ns=float(np.median(values)),
        min_ns=int(values.min()),
        max_ns=int(values.max()),
        p50_ns=p50,
        p95_ns=p95,
        iqr_ns=p75 - p25,
        percentiles_indicative=len(samples) < 100,
        raw_ns=tuple(int(v) for v in samples),
        timer=timer,
        sync_method=sync_method,
    )


def _hash_inputs(inputs: Mapping[str, np.ndarray]) -> str:
    """Stable SHA-256 over the (name, dtype, shape, bytes) of every network input."""
    digest = hashlib.sha256()
    for name in sorted(inputs):
        array = np.ascontiguousarray(inputs[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class WorkloadResult:
    """One class-4-CPU workload: verified session, latency table, and the executed output."""

    backend: str
    model_path: str
    op_type: str
    domain: str
    providers: tuple[str, ...]
    threads: ThreadPinning
    input_sha256: str
    input_shape: tuple[int, ...]
    output_sha256: str
    latency: LatencyStats
    output: np.ndarray
    dequantized_path: bool = False

    def to_dict(self) -> dict[str, object]:
        """Return the workload record as a JSON-serializable dict (the output array is hashed)."""
        return {
            "backend": self.backend,
            "model_path": self.model_path,
            "op_type": self.op_type,
            "domain": self.domain,
            "providers": list(self.providers),
            "dequantized_path": self.dequantized_path,
            "input_sha256": self.input_sha256,
            "input_shape": list(self.input_shape),
            "output_sha256": self.output_sha256,
            "latency": self.latency.to_dict(),
        }


def run_cpu_workload(
    model_path: str | Path,
    inputs: Mapping[str, np.ndarray],
    *,
    pinning: ThreadPinning,
    op_type: str,
    domain: str,
    backend: str,
    warmup_iters: int = 10,
    repeats: int = 5,
    providers: tuple[str, ...] = _DEFAULT_PROVIDERS,
) -> WorkloadResult:
    """Execute one container on CPU and measure it per the class-4-CPU protocol.

    The session is opened with :func:`open_cpu_session` (provider + kernel verified), one
    materialising run produces the output used for the numerical comparison, and
    :func:`measure_latency` times the same inputs. The recorded ``input_sha256`` makes it checkable
    that the fp32 baseline and the low-bit run consumed **identical inputs** (protocol §8.2.2, §8.2.5).

    Returns:
        A :class:`WorkloadResult` carrying the output tensor, its hash and the latency table.

    Raises:
        BackendCapabilityError: If the provider was substituted or the kernel is missing.
    """
    feed = {name: np.ascontiguousarray(array) for name, array in inputs.items()}
    session = open_cpu_session(
        model_path, pinning=pinning, op_type=op_type, domain=domain, providers=providers
    )

    def run_once() -> None:
        session.session.run(None, feed)  # type: ignore[attr-defined]

    first = session.session.run(None, feed)  # type: ignore[attr-defined]
    output = np.asarray(first[0])
    latency = measure_latency(run_once, warmup_iters=warmup_iters, repeats=repeats)
    primary = next(iter(feed.values()))
    return WorkloadResult(
        backend=backend,
        model_path=session.model_path,
        op_type=op_type,
        domain=domain,
        providers=session.providers,
        threads=session.threads,
        input_sha256=_hash_inputs(feed),
        input_shape=tuple(int(d) for d in primary.shape),
        output_sha256=hashlib.sha256(np.ascontiguousarray(output).tobytes()).hexdigest(),
        latency=latency,
        output=output,
    )


def error_statistics(measured: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Absolute-error statistics of ``measured`` against ``reference``.

    The caller MUST publish these together with (a) the ``W``/``X`` distributions and shapes,
    (b) the reference definition, and (c) the scale-dependence caveat
    (``backend-capability.md`` §2.4c; ``memory-accounting.md`` §8.8). A bare error number is a
    reproducibility defect.
    """
    diff = np.abs(np.asarray(measured, dtype=np.float64) - np.asarray(reference, dtype=np.float64))
    ref = np.abs(np.asarray(reference, dtype=np.float64))
    ref_max = float(ref.max()) if ref.size else 0.0
    return {
        "max_abs_err": float(diff.max()) if diff.size else 0.0,
        "mean_abs_err": float(diff.mean()) if diff.size else 0.0,
        "ref_max_abs": ref_max,
        "relative_max_abs_err": float(diff.max() / ref_max) if ref_max else 0.0,
    }
