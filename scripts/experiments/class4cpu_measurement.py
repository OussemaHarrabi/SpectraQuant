"""Class-4-CPU kernel fixture: export our own int4/int8 ONNX containers and measure them.

Produces the deferred freeze item (``preregistration.md`` §13: "Class-4-CPU ONNX fixture (int4/int8
container written by us, its runner, the same-session fp32 baseline)"). It serializes a linear layer
with :mod:`spectraquant.quantization.onnx_export`, executes the containers on ONNX Runtime's CPU
``MatMulNBits`` (int4 weight-only) and ``MatMulInteger`` (int8) kernels, and measures both against an
fp32 CPU baseline taken **in the same process, session and thread configuration**
(``docs/protocols/benchmark-protocol.md`` §8).

Usage:
    uv run python scripts/experiments/class4cpu_measurement.py [--out DIR] [--work-dir DIR] [--threads 8]

Writes ``class4cpu.json`` into ``artifacts/sample-results/class4cpu/`` by default; the ONNX
containers themselves go to ``outputs/class4cpu/containers`` (git-ignored — model weights are never
committed). CPU-only, deterministic, a few seconds.

Every number is measurement class 4-CPU, substrate ``LOCAL-FIXTURE``, and carries the §8.3 scope
line; it is never comparable to published GPU latency/throughput figures.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from spectraquant.benchmarking.kernel_cpu import (
    CPU_SCOPE_LINE,
    error_statistics,
    physical_memory_bytes,
    pin_cpu_threads,
    run_cpu_workload,
)
from spectraquant.quantization.fake_quant import QuantSpec
from spectraquant.quantization.onnx_export import (
    export_fp32_matmul,
    export_int4_matmulnbits,
    export_int8_matmulinteger,
    reconcile_with_accounting,
)

# Fixture constants — the exact fixture of `backend-capability.md` §2.4c and
# `memory-accounting.md` §7.5.1, so the measured error reproduces the published numbers.
SEED_W, SEED_X, SIGMA_W = 0, 1, 0.1
K, N = 64, 32  # ONNX weight (in_features, out_features)
GROUP_SIZE = 32
THREADS = 8
WARMUP_ITERS = 10
REPEATS = 5
# Two declared workloads: the documentation fixture and a batch large enough for the GEMM to
# dominate the ORT session-call overhead. One row per shape (protocol §2.1).
BATCHES = (4, 512)

BACKENDS = (
    ("fp32", "MatMul", "", "onnx-fp32-matmul"),
    ("int4", "MatMulNBits", "com.microsoft", "onnx-int4-matmulnbits"),
    ("int8", "MatMulInteger", "", "onnx-int8-matmulinteger"),
)


def git_commit() -> str:
    """Current commit SHA, or ``unknown`` outside a checkout."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    """Whether the working tree has uncommitted changes."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout
        return bool(out.strip())
    except Exception:
        return False


def build_fixture() -> tuple[np.ndarray, torch.Tensor, dict[int, np.ndarray]]:
    """Return ``(W_onnx (K, N), W_torch (out, in), {batch: X (batch, K)})`` with fixed RNG seeds."""
    w_onnx = np.random.RandomState(SEED_W).randn(K, N).astype(np.float32) * SIGMA_W
    w_torch = torch.tensor(w_onnx.T.copy())  # (out_features, in_features)
    inputs = {
        batch: np.random.RandomState(SEED_X).randn(batch, K).astype(np.float32) for batch in BATCHES
    }
    return w_onnx, w_torch, inputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/sample-results/class4cpu")
    parser.add_argument("--work-dir", default="outputs/class4cpu/containers")
    parser.add_argument("--threads", type=int, default=THREADS)
    args = parser.parse_args()

    w_onnx, w_torch, inputs = build_fixture()
    containers = Path(args.work_dir)
    containers.mkdir(parents=True, exist_ok=True)

    fp32 = export_fp32_matmul(w_torch, containers / "linear_fp32.onnx")
    int4 = export_int4_matmulnbits(
        w_torch, containers / "linear_int4.onnx", block_size=GROUP_SIZE, symmetric=True
    )
    int8 = export_int8_matmulinteger(w_torch, containers / "linear_int8.onnx")
    artifacts = {"fp32": fp32, "int4": int4, "int8": int8}

    spec4 = QuantSpec(
        bits=4, granularity="per_group", group_size=GROUP_SIZE, symmetric=True, axis=0
    )
    spec8 = QuantSpec(bits=8, granularity="per_tensor", group_size=None, symmetric=False, axis=0)
    shape = (N, K)  # PyTorch (out_features, in_features)
    reconciliation = {
        "int4": reconcile_with_accounting(int4, shape=shape, spec=spec4).to_dict(),
        "int8": reconcile_with_accounting(int8, shape=shape, spec=spec8).to_dict(),
        "fp32": reconcile_with_accounting(fp32, shape=shape, spec=None).to_dict(),
    }

    pinning = pin_cpu_threads(args.threads)
    scope_line = CPU_SCOPE_LINE.format(cpu_model=pinning.cpu_model, threads=pinning.requested)

    workloads: list[dict] = []
    reference_checks: list[dict] = []
    threads_seen: list[dict] = []
    for batch in BATCHES:
        x = inputs[batch]
        reference = (x @ w_onnx).astype(np.float32)
        reference_definition = (
            "numpy float32 `X @ W` on the fp32 ONNX weight W (K,N); the ORT fp32 session on the "
            "same graph is checked against it and the max difference is recorded"
        )
        reference_agreement: float | None = None
        for name, op, domain, backend in BACKENDS:
            result = run_cpu_workload(
                artifacts[name].path,
                {"X": x},
                pinning=pinning,
                op_type=op,
                domain=domain,
                backend=backend,
                warmup_iters=WARMUP_ITERS,
                repeats=REPEATS,
            )
            threads_seen.append(result.threads.to_dict())
            errors = error_statistics(result.output, reference)
            if name == "fp32":
                reference_agreement = errors["max_abs_err"]
            workloads.append(
                {
                    "shape_batch_x_K": [batch, K],
                    "backend": backend,
                    "op_type": op,
                    "domain": domain,
                    "threads": result.threads.to_dict(),
                    "input_sha256": result.input_sha256,
                    "output_sha256": result.output_sha256,
                    "latency": result.latency.to_dict(),
                    "error": {
                        **errors,
                        "weight_distribution": (
                            f"W ~ N(0, {SIGMA_W}^2) i.i.d., shape (K,N)=({K},{N}), "
                            f"|W|.max={float(np.abs(w_onnx).max()):.4f}"
                        ),
                        "input_distribution": (
                            f"X ~ N(0, 1) i.i.d., shape ({batch},{K}), X.std={float(x.std()):.4f}"
                        ),
                        "reference_definition": reference_definition,
                        "scale_dependence": (
                            "absolute error is scale-dependent and only meaningful with the "
                            "distributions above; the relative error is scale-invariant "
                            "(backend-capability.md section 2.4c)"
                        ),
                    },
                }
            )
        reference_checks.append(
            {
                "shape_batch_x_K": [batch, K],
                "numpy_vs_ort_fp32_max_abs": reference_agreement,
                "definition": reference_definition,
            }
        )

    input_hashes = {
        row["input_sha256"] for row in workloads if row["shape_batch_x_K"][0] == BATCHES[0]
    }
    same_inputs = len(input_hashes) == 1
    same_threads = all(t == threads_seen[0] for t in threads_seen)
    if not (same_inputs and same_threads):
        raise RuntimeError(
            "protocol violation: the fp32 baseline and the low-bit runs must use identical inputs "
            "and an identical thread configuration (benchmark-protocol.md section 8.2)"
        )

    document = {
        "schema_version": 1,
        "run_id": "class4cpu-linear-fixture",
        "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "measurement_class": "4-CPU",
        "substrate": "LOCAL-FIXTURE",
        "scope_line": scope_line,
        "scope_line_required_verbatim": (
            "not comparable to published GPU latency or throughput figures"
        ),
        "fixture": {
            "description": "single weight-only linear map (ONNX MatMul), symbolic batch dimension",
            "onnx_weight_shape_K_N": [K, N],
            "torch_weight_shape_out_in": [N, K],
            "group_size": GROUP_SIZE,
            "seeds": {"W": SEED_W, "X": SEED_X},
            "weight_distribution": (
                f"W ~ N(0, {SIGMA_W}^2) i.i.d., shape (K,N)=({K},{N}), "
                f"|W|.max={float(np.abs(w_onnx).max()):.4f}"
            ),
            "input_distribution": (
                f"X ~ N(0, 1) i.i.d., shape (batch,{K}), X.std={float(inputs[BATCHES[0]].std()):.4f}"
            ),
        },
        "threads": threads_seen[0] if threads_seen else pinning.to_dict(),
        "host": {
            "cpu": pinning.cpu_model,
            "physical_cores": pinning.physical_cores,
            "logical_cores": pinning.logical_cores,
            "ram_bytes": physical_memory_bytes(),
            "os": sys.platform,
        },
        "backend": {
            "name": "onnxruntime",
            "execution_provider": "CPUExecutionProvider",
            "dequantized_path": False,
            "extra": "onnx",
            "kernels": {"int4": "MatMulNBits (com.microsoft)", "int8": "MatMulInteger"},
        },
        "artifacts": {name: art.to_dict() for name, art in artifacts.items()},
        "reconciliation": reconciliation,
        "protocol": {
            "warmup_iters": WARMUP_ITERS,
            "repeats": REPEATS,
            "timer": "time.perf_counter_ns",
            "sync_method": "ort_run_blocking",
            "same_process_baseline": True,
            "baseline_backend": "onnx-fp32-matmul",
            "inputs_identical": same_inputs,
            "threads_identical": same_threads,
        },
        "workloads": workloads,
        "reference_checks": reference_checks,
        "limitations": [
            "CPU-only, single machine: not comparable to published GPU latency or throughput figures.",
            "This is a fixture (one weight matrix, two batch sizes), not the Tier-1 H5 chain; it "
            "confirms the class-4-CPU cell is runnable, not that H5 holds.",
            "The int4/int8 containers are SpectraQuant-serialized ONNX artifacts executed by ONNX "
            "Runtime CPU kernels; the scales are fp32 and the payload is weight-only (int4) or "
            "dynamically quantized (int8).",
            "Latency at these shapes is dominated by ONNX Runtime session-call overhead; the value "
            "is protocol compliance, not a performance claim.",
            "Absolute error is scale-dependent; the relative error is the scale-invariant figure.",
        ],
        "does_not_support": [
            "any Pareto or speedup claim against GPU-measured literature numbers",
            "any headline 'X times faster' claim without the same-session baseline",
            "any H5 confirmation by itself (the quality half and the GPU half are deferred)",
        ],
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "class4cpu.json"
    path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {path}")
    print(f"scope: {scope_line}")
    for name in ("fp32", "int4", "int8"):
        record = artifacts[name].bytes
        print(
            f"[{name}] files={dict(record.files)} total={record.total_bytes} "
            f"payload={record.payload_bytes} scales={record.scales_bytes} "
            f"overhead={record.container_overhead_bytes}"
        )
    for row in workloads:
        lat = row["latency"]
        err = row["error"]
        print(
            f"[{row['shape_batch_x_K']} {row['backend']}] median={lat['median_ns']}ns "
            f"p95={lat['p95_ns']}ns max_abs_err={err['max_abs_err']:.6g} "
            f"rel={err['relative_max_abs_err']:.6g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
