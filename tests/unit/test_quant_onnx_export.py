"""ONNX container export: determinism, kernels, and measured-byte reconciliation (class 3).

The fixture is the one of ``docs/protocols/memory-accounting.md`` §7.5 / §7.5.1 (``W`` drawn from
``RandomState(0)``, ``(K, N) = (64, 32)``), so the payload/scale byte figures asserted here are the
documented ones. Every byte number is class 3 (measured) and every reconciliation states its residual
rather than forcing equality (``memory-accounting.md`` §6.1).
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

from spectraquant.quantization.accounting import accounted_bytes
from spectraquant.quantization.fake_quant import QuantSpec
from spectraquant.quantization.onnx_export import (
    MATMULINTEGER_DOMAIN,
    MATMULINTEGER_OP,
    MATMULNBITS_DOMAIN,
    MATMULNBITS_OP,
    OnnxExtraMissingError,
    export_fp32_matmul,
    export_int4_matmulnbits,
    export_int8_matmulinteger,
    measure_artifact_bytes,
    reconcile_with_accounting,
)

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

K, N = 64, 32  # ONNX weight (in_features, out_features)
GROUP_SIZE = 32
SHAPE_OUT_IN = (N, K)  # PyTorch weight (out_features, in_features)
SPEC4 = QuantSpec(bits=4, granularity="per_group", group_size=GROUP_SIZE, symmetric=True, axis=0)
SPEC8 = QuantSpec(bits=8, granularity="per_tensor", group_size=None, symmetric=False, axis=0)


def _weights() -> torch.Tensor:
    """The documentation fixture weight, in PyTorch ``(out, in)`` orientation."""
    w_onnx = np.random.RandomState(0).randn(K, N).astype(np.float32) * 0.1
    return torch.tensor(w_onnx.T.copy())


@pytest.fixture
def exported(tmp_path):
    """Export one of each container into a per-test directory."""
    weight = _weights()
    return {
        "fp32": export_fp32_matmul(weight, tmp_path / "linear_fp32.onnx"),
        "int4": export_int4_matmulnbits(
            weight, tmp_path / "linear_int4.onnx", block_size=GROUP_SIZE, symmetric=True
        ),
        "int8": export_int8_matmulinteger(weight, tmp_path / "linear_int8.onnx"),
    }


def test_export_is_deterministic_for_the_same_file_name(tmp_path):
    """Two exports of the same weights to the same file name are byte-identical (pinned versions)."""
    weight = _weights()
    first = export_int4_matmulnbits(weight, tmp_path / "a" / "w.onnx", block_size=GROUP_SIZE)
    second = export_int4_matmulnbits(weight, tmp_path / "b" / "w.onnx", block_size=GROUP_SIZE)
    assert first.bytes.sha256 == second.bytes.sha256
    assert first.bytes.files == second.bytes.files
    assert first.bytes.total_bytes == second.bytes.total_bytes


def test_exported_graphs_contain_the_expected_op_and_domain(exported):
    """The int4 graph must be ``MatMulNBits`` (``com.microsoft``); int8 must use ``MatMulInteger``."""
    assert (MATMULNBITS_OP, MATMULNBITS_DOMAIN) in exported["int4"].nodes
    assert (MATMULINTEGER_OP, MATMULINTEGER_DOMAIN) in exported["int8"].nodes
    assert ("MatMul", "") in exported["fp32"].nodes
    for artifact in exported.values():
        assert artifact.ir_version <= 13  # onnxruntime 1.30 rejects IR > 13


def test_int4_bytes_match_the_documented_payload_and_scales(exported):
    """int4 payload 1024 B, fp32 scales 256 B, external sidecar present (memory-accounting §7.5)."""
    report = exported["int4"].bytes
    assert report.payload_bytes == 1024  # ceil(2048 * 4 / 8)
    assert report.scales_bytes == 256  # 64 blocks * fp32
    assert report.zero_point_bytes == 0  # symmetric
    assert report.sidecar_bytes == 1024  # W_Q4 lives in the .data sidecar
    assert report.graph_bytes > 0


def test_int8_bytes_match_the_documented_payload_scale_and_zero_point(exported):
    """int8 payload 2048 B, per-tensor fp32 scale 4 B, int8 zero-point 1 B (memory-accounting §7.5)."""
    report = exported["int8"].bytes
    assert report.payload_bytes == 2048  # int8[64, 32]
    assert report.scales_bytes == 4
    assert report.zero_point_bytes == 1
    assert report.sidecar_bytes == 0


def test_measured_byte_reconciliation_identity_holds_for_every_artifact(exported):
    """``total == graph + sidecar == payload + scales + zp + other + overhead`` (every file summed)."""
    for artifact in exported.values():
        report = artifact.bytes
        assert report.total_bytes == sum(size for _, size in report.files)
        assert report.total_bytes == report.graph_bytes + report.sidecar_bytes
        assert report.total_bytes == (
            report.payload_bytes
            + report.scales_bytes
            + report.zero_point_bytes
            + report.other_bytes
            + report.container_overhead_bytes
        )


def test_reconciliation_residual_is_decomposed_not_forced_to_zero(exported):
    """The residual is scale-dtype difference + container overhead; equality is never forced."""
    int4 = reconcile_with_accounting(exported["int4"], shape=SHAPE_OUT_IN, spec=SPEC4)
    assert int4.analytical_total_bytes == accounted_bytes(SHAPE_OUT_IN, SPEC4) == 1152
    assert int4.scale_dtype_delta_bytes == 128  # fp32 on disk (256) vs fp16 model convention (128)
    assert int4.residual_bytes == (int4.scale_dtype_delta_bytes + int4.container_overhead_bytes)
    assert int4.residual_bytes > 0  # graph/file overhead is not in the analytical model

    int8 = reconcile_with_accounting(exported["int8"], shape=SHAPE_OUT_IN, spec=SPEC8)
    assert int8.analytical_total_bytes == 2053  # 2048 + 4 + 1
    assert int8.scale_dtype_delta_bytes == 0
    assert int8.residual_bytes == int8.container_overhead_bytes > 0


def test_analytical_model_reproduces_the_documentation_breakdown(exported):
    """The accounting model is the source of truth; measured payload+scales match it up to dtypes."""
    int4 = reconcile_with_accounting(exported["int4"], shape=SHAPE_OUT_IN, spec=SPEC4)
    assert int4.analytical is not None
    assert int4.analytical.payload_bytes_padded == 1024
    assert int4.analytical.n_blocks == 64
    assert int4.analytical.scales_bytes == 128  # fp16 convention
    measured = exported["int4"].bytes
    assert measured.payload_bytes == int4.analytical.payload_bytes_padded


def test_export_is_idempotent_when_the_target_directory_is_populated(tmp_path):
    """Re-exporting into a directory that already holds the sidecar must not accumulate bytes.

    ONNX Runtime's external-data writer extends an existing sidecar rather than replacing it, so a
    naive re-export would grow ``.data`` (and the measured total) on every run. The exporter clears
    its own target first, so the second export matches the first.
    """
    weight = _weights()
    target = tmp_path / "linear_int4.onnx"
    first = export_int4_matmulnbits(weight, target, block_size=GROUP_SIZE)
    second = export_int4_matmulnbits(weight, target, block_size=GROUP_SIZE)
    assert second.bytes.files == first.bytes.files
    assert second.bytes.total_bytes == first.bytes.total_bytes
    assert second.bytes.sidecar_bytes == 1024


def test_measure_artifact_bytes_rejects_a_missing_file(tmp_path):
    """A missing container is an error, not an empty report."""
    with pytest.raises(FileNotFoundError):
        measure_artifact_bytes(tmp_path / "absent.onnx")


def test_missing_onnx_extra_raises_a_named_error(monkeypatch):
    """A missing optional dependency raises :class:`OnnxExtraMissingError` naming the ``onnx`` extra."""
    monkeypatch.setitem(sys.modules, "onnx", None)
    with pytest.raises(OnnxExtraMissingError, match="onnx"):
        export_fp32_matmul(_weights(), "unused.onnx")
