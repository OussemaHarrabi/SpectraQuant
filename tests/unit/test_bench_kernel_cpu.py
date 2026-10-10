"""Class-4-CPU kernel protocol: pinning read-back, latency statistics, and loud-failure guards.

The fixtures use the ``backend-capability.md`` §2.4c / ``memory-accounting.md`` §7.5.1 weight
(``RandomState(0)``, ``(K, N) = (64, 32)``) so the int4/int8 error figures reproduce the published
ones. Every number here is class 4-CPU; the §8.3 scope line is asserted present.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from spectraquant.benchmarking.kernel_cpu import (
    CPU_SCOPE_LINE,
    BackendCapabilityError,
    assert_lowbit_kernel_selected,
    error_statistics,
    measure_latency,
    open_cpu_session,
    pin_cpu_threads,
    run_cpu_workload,
)
from spectraquant.quantization.onnx_export import (
    MATMULINTEGER_DOMAIN,
    MATMULINTEGER_OP,
    MATMULNBITS_DOMAIN,
    MATMULNBITS_OP,
    export_fp32_matmul,
    export_int4_matmulnbits,
    export_int8_matmulinteger,
)

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

K, N = 64, 32
GROUP_SIZE = 32
THREADS = 8
WARMUP = 10
REPEATS = 5


@pytest.fixture(autouse=True)
def _restore_torch_threads():
    """Restore the process thread count after each test (pinning is process-global)."""
    previous = torch.get_num_threads()
    yield
    torch.set_num_threads(previous)


def _w_onnx() -> np.ndarray:
    """The documentation fixture weight in ONNX orientation ``(K, N)``, C-contiguous."""
    return np.ascontiguousarray(np.random.RandomState(0).randn(K, N).astype(np.float32) * 0.1)


def _weight() -> torch.Tensor:
    """The documentation fixture weight, in PyTorch ``(out, in)`` orientation."""
    return torch.tensor(_w_onnx().T.copy())


def _input(batch: int) -> np.ndarray:
    """The documentation fixture input ``X ~ N(0, 1)``, shape ``(batch, K)``."""
    return np.random.RandomState(1).randn(batch, K).astype(np.float32)


@pytest.fixture
def containers(tmp_path):
    """Export the fp32 reference and the two low-bit containers into a per-test directory."""
    weight = _weight()
    return {
        "fp32": export_fp32_matmul(weight, tmp_path / "linear_fp32.onnx"),
        "int4": export_int4_matmulnbits(
            weight, tmp_path / "linear_int4.onnx", block_size=GROUP_SIZE, symmetric=True
        ),
        "int8": export_int8_matmulinteger(weight, tmp_path / "linear_int8.onnx"),
    }


def test_pin_cpu_threads_reads_back_effective_values(monkeypatch):
    """Requested threads are set for torch and the environment and then read back (protocol §2.11)."""
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    pinning = pin_cpu_threads(3, inter_op_threads=1)
    assert pinning.requested == 3
    assert pinning.torch_effective == torch.get_num_threads() == 3
    assert pinning.omp_num_threads == "3"
    assert pinning.ort_intra_effective is None  # filled when a session is opened
    assert pinning.cpu_model
    assert pinning.logical_cores is None or pinning.logical_cores >= 1
    with pytest.raises(ValueError):
        pin_cpu_threads(0)


def test_measure_latency_discloses_warmups_and_stores_every_repeat():
    """Warm-ups are discarded; every measured repeat is stored; percentiles are indicative."""
    calls = {"n": 0}

    def run_once() -> None:
        calls["n"] += 1

    stats = measure_latency(run_once, warmup_iters=3, repeats=4)
    assert calls["n"] == 3 + 4
    assert stats.warmup_iters == 3
    assert stats.samples == 4
    assert len(stats.raw_ns) == 4
    assert stats.min_ns <= stats.median_ns <= stats.max_ns
    assert stats.p95_ns >= stats.p50_ns
    assert stats.percentiles_indicative is True  # 4 < 100 samples (protocol §2.7)
    assert stats.sync_method == "ort_run_blocking"
    with pytest.raises(ValueError):
        measure_latency(run_once, warmup_iters=0, repeats=0)


def test_missing_lowbit_kernel_op_fails_loudly(containers):
    """A graph without the expected op raises rather than acting as a dequantized fallback (§6)."""
    with pytest.raises(BackendCapabilityError):
        assert_lowbit_kernel_selected(
            containers["fp32"].path, op_type=MATMULNBITS_OP, domain=MATMULNBITS_DOMAIN
        )
    nodes = assert_lowbit_kernel_selected(
        containers["int4"].path, op_type=MATMULNBITS_OP, domain=MATMULNBITS_DOMAIN
    )
    assert (MATMULNBITS_OP, MATMULNBITS_DOMAIN) in nodes


def test_substituted_provider_fails_loudly(containers):
    """A provider the runtime silently substitutes is a backend-capability failure (§6)."""
    pinning = pin_cpu_threads(THREADS)
    with pytest.raises(BackendCapabilityError):
        open_cpu_session(
            containers["int4"].path,
            pinning=pinning,
            op_type=MATMULNBITS_OP,
            domain=MATMULNBITS_DOMAIN,
            providers=("CUDAExecutionProvider",),
        )


def test_baseline_and_lowbit_share_inputs_and_threads(containers):
    """The fp32 baseline and the int4 run use identical inputs and an identical thread block."""
    pinning = pin_cpu_threads(THREADS)
    x = _input(4)
    reference = (x @ _w_onnx()).astype(np.float32)
    fp32 = run_cpu_workload(
        containers["fp32"].path,
        {"X": x},
        pinning=pinning,
        op_type="MatMul",
        domain="",
        backend="fp32",
        warmup_iters=WARMUP,
        repeats=REPEATS,
    )
    int4 = run_cpu_workload(
        containers["int4"].path,
        {"X": x},
        pinning=pinning,
        op_type=MATMULNBITS_OP,
        domain=MATMULNBITS_DOMAIN,
        backend="int4",
        warmup_iters=WARMUP,
        repeats=REPEATS,
    )
    assert fp32.input_sha256 == int4.input_sha256
    assert fp32.threads.to_dict() == int4.threads.to_dict()
    assert int4.threads.ort_intra_effective == THREADS
    assert fp32.providers == ("CPUExecutionProvider",)
    assert int4.dequantized_path is False
    # The fp32 baseline reproduces the numpy reference; the int4 path does not (it is lossy).
    # The fp32 bound is a *few ulps*, not zero: ONNX Runtime and numpy may reduce in different orders,
    # and on the Linux CI runner the observed difference is exactly one ulp of fp32 (5.96e-07 = 2^-24)
    # while on Windows it is 0.0. A bitwise assertion would be a platform assertion, not a
    # numerical-agreement assertion. A convention error (a transposed weight, a wrong block axis) is
    # O(1) here, so 1e-5 still catches it -- and it is 100x above the fp32 round-off floor.
    assert error_statistics(fp32.output, reference)["max_abs_err"] <= 1e-5
    assert error_statistics(int4.output, reference)["max_abs_err"] > 0.0
    assert CPU_SCOPE_LINE.format(cpu_model=fp32.threads.cpu_model, threads=THREADS)


def test_lowbit_error_reproduces_the_documented_fixture(containers):
    """int4/int8 error reproduces the published fixture (W ~ N(0, 0.1^2), (64,32); X ~ N(0,1), (4,64)).

    The reference is ``numpy float32 X @ W`` on the fp32 weights. The absolute figures are
    scale-dependent; the relative figures (0.07713 int4, 0.00785 int8) are scale-invariant
    (``backend-capability.md`` §2.4c). This is proof the containers execute through real kernels.
    """
    pinning = pin_cpu_threads(THREADS)
    x = _input(4)
    reference = (x @ _w_onnx()).astype(np.float32)
    int4 = run_cpu_workload(
        containers["int4"].path,
        {"X": x},
        pinning=pinning,
        op_type=MATMULNBITS_OP,
        domain=MATMULNBITS_DOMAIN,
        backend="int4",
        warmup_iters=WARMUP,
        repeats=REPEATS,
    )
    int8 = run_cpu_workload(
        containers["int8"].path,
        {"X": x},
        pinning=pinning,
        op_type=MATMULINTEGER_OP,
        domain=MATMULINTEGER_DOMAIN,
        backend="int8",
        warmup_iters=WARMUP,
        repeats=REPEATS,
    )
    int4_err = error_statistics(int4.output, reference)
    int8_err = error_statistics(int8.output, reference)
    assert int4_err["max_abs_err"] == pytest.approx(0.2041, abs=5e-4)
    assert int4_err["relative_max_abs_err"] == pytest.approx(0.0771, abs=5e-5)
    assert int8_err["max_abs_err"] == pytest.approx(0.0208, abs=5e-5)
    assert int8_err["relative_max_abs_err"] == pytest.approx(0.00785, abs=5e-5)
