"""SpectraQuant-serialized low-bit ONNX containers (class 3) for the class-4-CPU kernel.

This module exports a linear map ``W`` to ONNX and quantizes it **with our own controlled
invocation**, producing the two containers the class-4-CPU cell is allowed to execute
(``AGENTS.md`` §5; ``docs/protocols/benchmark-protocol.md`` §8.1):

* **int4 weight-only** — ``MatMulNBits`` (domain ``com.microsoft``) via ONNX Runtime's
  ``MatMulNBitsQuantizer``, ``block_size`` = the quantizer group size, symmetric;
* **int8 dynamic** — ``MatMulInteger`` via ``quantize_dynamic(weight_type=QInt8)``;
* an **fp32** container, the same-graph reference the class-4-CPU baseline runs.

Orientation (``docs/research/backend-capability.md`` §2.4a). ONNX ``MatMul(X, W)`` computes
``Y = X·W`` with ``X`` shaped ``(·, K)`` and ``W`` shaped ``(K, N)``. A PyTorch ``nn.Linear`` weight
is ``(out_features, in_features)``, so the exported ONNX initializer is its **transpose**,
``(in_features, out_features) = (K, N)``. The serialized graph therefore differs from (and is not
interchangeable with) a graph fed a transposed activation.

Byte accounting is **measured** (class 3): :func:`measure_artifact_bytes` sums the graph file and
every external-data sidecar, classifies each initializer as payload / scales / zero-points, and
reports the container overhead. ``memory-accounting.md`` §6.1 requires exactly this — the container
term is a property of the serializer invocation, not a constant, so it is measured per artifact and
never assumed. :func:`reconcile_with_accounting` states the residual against
:mod:`spectraquant.quantization.accounting` **honestly**: our analytical model contains no graph/file
overhead, and it stores per-group scales at fp16 while ONNX Runtime writes fp32, so the residual is
expected to be non-zero and is decomposed rather than forced to zero.

Dependencies. ``onnx``, ``onnxruntime`` and ``onnx-ir`` are the optional ``onnx`` extra
(``pyproject.toml``; verified versions ``onnx 1.23.2`` / ``onnxruntime 1.30.0`` / ``onnx-ir 1.0.0``
per ``backend-capability.md`` §2.4). They are imported **lazily** inside functions and a missing
extra raises :class:`OnnxExtraMissingError` naming it, so the core package stays free of them.

Numerical / format limitations.
    * ``onnx`` 1.23.x defaults to IR version 14; onnxruntime 1.30 accepts ``<= 13``, so the exported
      ``ir_version`` is pinned (default :data:`DEFAULT_IR_VERSION`).
    * The int4 export needs ``onnx-ir`` installed (the quantizer imports it); the int4 sidecar holds
      the packed payload while the scales stay inline in the protobuf.
    * Everything is deterministic on CPU for the pinned versions: two exports of the same weights to
      the same file name produce byte-identical files (asserted by test).
"""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from spectraquant.quantization.accounting import ByteBreakdown, byte_breakdown
from spectraquant.quantization.fake_quant import QuantSpec

__all__ = [
    "DEFAULT_IR_VERSION",
    "DEFAULT_OPSET",
    "MATMULINTEGER_DOMAIN",
    "MATMULINTEGER_OP",
    "MATMULNBITS_DOMAIN",
    "MATMULNBITS_OP",
    "ArtifactBytes",
    "ArtifactTensor",
    "ByteReconciliation",
    "ExportedArtifact",
    "OnnxExtraMissingError",
    "export_fp32_matmul",
    "export_int4_matmulnbits",
    "export_int8_matmulinteger",
    "measure_artifact_bytes",
    "reconcile_with_accounting",
]

#: Name of the optional dependency extra that provides onnx / onnxruntime / onnx-ir.
ONNX_EXTRA = "onnx"

#: IR version written by default. ``onnx`` 1.23.x defaults to 14; onnxruntime 1.30 accepts ``<= 13``.
DEFAULT_IR_VERSION = 10
#: Opset written by default for the source fp32 graph.
DEFAULT_OPSET = 17

MATMULNBITS_OP = "MatMulNBits"
MATMULNBITS_DOMAIN = "com.microsoft"
MATMULINTEGER_OP = "MatMulInteger"
MATMULINTEGER_DOMAIN = ""

_ROLE_PAYLOAD = "payload"
_ROLE_SCALES = "scales"
_ROLE_ZERO_POINT = "zero_point"


class OnnxExtraMissingError(ImportError):
    """Raised when the optional ``onnx`` extra (onnx / onnxruntime / onnx-ir) is absent."""


def _import_onnx():
    """Return the ``onnx`` module, or raise :class:`OnnxExtraMissingError` naming the extra."""
    try:
        import onnx  # optional dependency, imported lazily by design
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise OnnxExtraMissingError(
            "ONNX container export requires the optional 'onnx' extra: install it with "
            "`uv sync --extra onnx` (provides onnx~=1.23.2, onnxruntime~=1.30.0, onnx-ir~=1.0.0, "
            "torchao~=0.18.0). See docs/research/backend-capability.md section 2.4."
        ) from exc
    return onnx


def _import_ort_quantization():
    """Return ``(matmul_nbits_quantizer, quant_utils, quantize_dynamic, QuantType)`` lazily."""
    try:
        from onnxruntime.quantization import (
            QuantType,
            matmul_nbits_quantizer,
            quant_utils,
            quantize_dynamic,
        )
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise OnnxExtraMissingError(
            "ONNX quantization requires the optional 'onnx' extra: install it with "
            "`uv sync --extra onnx` (onnxruntime~=1.30.0). See "
            "docs/research/backend-capability.md section 2.4."
        ) from exc
    return matmul_nbits_quantizer, quant_utils, quantize_dynamic, QuantType


@dataclass(frozen=True)
class ArtifactTensor:
    """One ONNX initializer with its measured (class 3) logical byte size and role."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    role: str
    external: bool

    def to_dict(self) -> dict[str, object]:
        """Return the tensor record as a JSON-serializable dict."""
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "nbytes": self.nbytes,
            "role": self.role,
            "external": self.external,
        }


@dataclass(frozen=True)
class ArtifactBytes:
    """Measured (class 3) byte breakdown of one serialized ONNX container.

    ``total_bytes`` sums the graph file **and every external-data sidecar** it wrote
    (``memory-accounting.md`` §6.1). ``container_overhead_bytes`` is
    ``total_bytes - payload - scales - zero_points - other`` — the graph protobuf, node
    attributes, tensor names and alignment, i.e. graph overhead that the analytical model does not
    contain.
    """

    path: str
    files: tuple[tuple[str, int], ...]
    graph_bytes: int
    sidecar_bytes: int
    total_bytes: int
    payload_bytes: int
    scales_bytes: int
    zero_point_bytes: int
    other_bytes: int
    container_overhead_bytes: int
    tensors: tuple[ArtifactTensor, ...]
    sha256: str

    def to_dict(self) -> dict[str, object]:
        """Return the artifacts report as a JSON-serializable dict."""
        return {
            "path": self.path,
            "files": [{"name": name, "bytes": size} for name, size in self.files],
            "graph_bytes": self.graph_bytes,
            "sidecar_bytes": self.sidecar_bytes,
            "total_bytes": self.total_bytes,
            "payload_bytes": self.payload_bytes,
            "scales_bytes": self.scales_bytes,
            "zero_point_bytes": self.zero_point_bytes,
            "other_bytes": self.other_bytes,
            "container_overhead_bytes": self.container_overhead_bytes,
            "tensors": [tensor.to_dict() for tensor in self.tensors],
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ExportedArtifact:
    """One serialized ONNX container plus the nodes that make it what it is."""

    backend: str
    path: str
    ir_version: int
    opset: int
    nodes: tuple[tuple[str, str], ...]
    bytes: ArtifactBytes

    def to_dict(self) -> dict[str, object]:
        """Return the export record as a JSON-serializable dict."""
        return {
            "backend": self.backend,
            "path": self.path,
            "ir_version": self.ir_version,
            "opset": self.opset,
            "nodes": [{"op_type": op, "domain": domain} for op, domain in self.nodes],
            "bytes": self.bytes.to_dict(),
        }


@dataclass(frozen=True)
class ByteReconciliation:
    """Measured container vs the analytical accounting model, with the residual decomposed.

    ``residual_bytes = measured_total - analytical_total`` is **expected to be non-zero**: the
    analytical model (``accounting.byte_breakdown``) covers the padded payload + scales + zero-points
    of our own container and contains no graph/file overhead. It is decomposed into
    ``scale_dtype_delta_bytes`` (ONNX Runtime writes per-group scales at fp32; our model uses fp16)
    and ``container_overhead_bytes`` (graph protobuf, node attributes, names, the sidecar header).
    """

    artifact: str
    shape: tuple[int, int]
    spec: QuantSpec | None
    analytical_total_bytes: int
    measured_total_bytes: int
    residual_bytes: int
    scale_dtype_delta_bytes: int
    container_overhead_bytes: int
    analytical: ByteBreakdown | None
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return the reconciliation as a JSON-serializable dict."""
        return {
            "artifact": self.artifact,
            "shape_out_in": list(self.shape),
            "spec": None
            if self.spec is None
            else {
                "bits": self.spec.bits,
                "granularity": self.spec.granularity,
                "group_size": self.spec.group_size,
                "symmetric": self.spec.symmetric,
                "axis": self.spec.axis,
            },
            "analytical_total_bytes": self.analytical_total_bytes,
            "measured_total_bytes": self.measured_total_bytes,
            "residual_bytes": self.residual_bytes,
            "scale_dtype_delta_bytes": self.scale_dtype_delta_bytes,
            "container_overhead_bytes": self.container_overhead_bytes,
            "analytical_breakdown": None if self.analytical is None else self.analytical.to_dict(),
            "analytical_bits_per_param": (
                None if self.analytical is None else self.analytical.bits_per_param
            ),
            "measured_bits_per_param": (
                None
                if self.analytical is None or self.analytical.n_params == 0
                else 8.0 * self.measured_total_bytes / self.analytical.n_params
            ),
            "notes": list(self.notes),
        }


def _weight_to_onnx(weight: torch.Tensor) -> np.ndarray:
    """Return the ONNX-orientation weight ``(K, N) = (in_features, out_features)`` as float32."""
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2-D (out, in), got shape {tuple(weight.shape)}")
    if weight.is_cuda:  # pragma: no cover - no CUDA device exists on the target host
        raise ValueError("ONNX export runs on CPU only: CUDA tensors are rejected by design")
    return weight.detach().to("cpu", torch.float32).numpy().T.copy()


def _build_fp32_model(weight: torch.Tensor, *, ir_version: int, opset: int):
    """Build the fp32 ``MatMul`` ModelProto for ``weight`` with a symbolic batch dimension."""
    onnx = _import_onnx()
    w_onnx = _weight_to_onnx(weight)
    in_features, out_features = int(w_onnx.shape[0]), int(w_onnx.shape[1])
    node = onnx.helper.make_node("MatMul", ["X", "W"], ["Y"], name="sq_matmul")
    graph = onnx.helper.make_graph(
        [node],
        "spectraquant_linear",
        [onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, ["batch", in_features])],
        [onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, ["batch", out_features])],
        [
            onnx.helper.make_tensor(
                "W", onnx.TensorProto.FLOAT, [in_features, out_features], w_onnx.ravel()
            )
        ],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", opset)])
    model.ir_version = int(ir_version)
    return model


def _graph_nodes(path: Path) -> tuple[tuple[str, str], ...]:
    """Return ``(op_type, domain)`` for every node in the serialized graph at ``path``."""
    onnx = _import_onnx()
    model = onnx.load(str(path))
    return tuple((node.op_type, node.domain) for node in model.graph.node)


def _assert_ir_version_supported(path: Path) -> int:
    """Read back the serialized IR version and reject one onnxruntime cannot load."""
    onnx = _import_onnx()
    ir_version = int(onnx.load(str(path)).ir_version)
    if ir_version > 13:
        raise RuntimeError(
            f"serialized {path.name} has IR version {ir_version}; onnxruntime 1.30 accepts <= 13 "
            "(set ir_version explicitly, see docs/research/backend-capability.md section 2.4)"
        )
    return ir_version


def _classify_role(name: str) -> str:
    """Classify an initializer by name: zero-point, scales, or payload."""
    lowered = name.lower()
    if "zero_point" in lowered:
        return _ROLE_ZERO_POINT
    if "scale" in lowered:
        return _ROLE_SCALES
    return _ROLE_PAYLOAD


def _artifact_files(path: Path) -> list[Path]:
    """Every file the save produced: the graph and any ``<graph>.data*`` sidecar."""
    prefix = path.name + "."
    entries = [
        entry
        for entry in sorted(path.parent.iterdir())
        if entry.is_file() and (entry.name == path.name or entry.name.startswith(prefix))
    ]
    return entries


def _clear_target(target: Path) -> None:
    """Delete the target graph and any ``<target>.*`` sidecar so a save cannot append to leftovers.

    ONNX Runtime's external-data writer extends an existing sidecar rather than replacing it, so a
    repeated export into a populated directory would accumulate bytes. Clearing first makes the
    artifact exactly what this invocation produced and keeps the export deterministic.
    """
    if target.is_file():
        target.unlink()
    for stale in target.parent.glob(target.name + ".*"):
        if stale.is_file():
            stale.unlink()


def _sha256_files(entries: list[Path]) -> str:
    """SHA-256 over the concatenation of the artifact's files in sorted-name order."""
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(entry.name.encode("utf-8"))
        digest.update(entry.read_bytes())
    return digest.hexdigest()


def measure_artifact_bytes(path: str | Path) -> ArtifactBytes:
    """Measure one ONNX container: graph + sidecars summed, tensors classified (class 3).

    Shapes / dtypes / device:
        ``path`` is a serialized ``.onnx`` file written by one of the export functions; its sidecar
        (if any) sits beside it as ``<name>.data``. Returns host-side byte counts only.

    Assumptions / limitations:
        * ``total_bytes`` is the sum of **every** file the save produced (``memory-accounting.md``
          §6.1), so it is the honest artifact size; counting the graph alone can understate it an
          order of magnitude.
        * ``payload_bytes`` / ``scales_bytes`` / ``zero_point_bytes`` are **logical** initializer
          sizes (``numpy_helper.to_array(...).nbytes``), not a per-region file split. For an
          external-data save the packed payload physically lives in the sidecar while the scales stay
          inline — the logical sizes are what the accounting model reconciles against.
        * ``container_overhead_bytes`` is a residual: graph protobuf, node attributes, tensor names
          and any padding. It is real storage, not an accounting detail.
    """
    graph = Path(path)
    if not graph.is_file():
        raise FileNotFoundError(f"no such ONNX container: {graph}")
    entries = _artifact_files(graph)
    files = tuple((entry.name, entry.stat().st_size) for entry in entries)
    graph_bytes = graph.stat().st_size
    total_bytes = sum(size for _, size in files)
    sidecar_bytes = total_bytes - graph_bytes

    onnx = _import_onnx()
    from onnx import numpy_helper

    model = onnx.load(str(graph))
    payload = scales = zero_points = other = 0
    tensors: list[ArtifactTensor] = []
    for tensor in model.graph.initializer:
        array = numpy_helper.to_array(tensor)
        role = _classify_role(tensor.name)
        if role == _ROLE_PAYLOAD:
            payload += array.nbytes
        elif role == _ROLE_SCALES:
            scales += array.nbytes
        elif role == _ROLE_ZERO_POINT:
            zero_points += array.nbytes
        else:  # pragma: no cover - no such initializer is emitted today
            other += array.nbytes
        tensors.append(
            ArtifactTensor(
                name=tensor.name,
                dtype=str(array.dtype),
                shape=tuple(int(d) for d in array.shape),
                nbytes=int(array.nbytes),
                role=role,
                external=bool(tensor.data_location),
            )
        )
    overhead = total_bytes - payload - scales - zero_points - other
    return ArtifactBytes(
        path=str(graph),
        files=files,
        graph_bytes=graph_bytes,
        sidecar_bytes=sidecar_bytes,
        total_bytes=total_bytes,
        payload_bytes=payload,
        scales_bytes=scales,
        zero_point_bytes=zero_points,
        other_bytes=other,
        container_overhead_bytes=overhead,
        tensors=tuple(tensors),
        sha256=_sha256_files(entries),
    )


def _exported_artifact(backend: str, path: Path, ir_version: int, opset: int) -> ExportedArtifact:
    """Assemble an :class:`ExportedArtifact` from a serialized graph on disk."""
    return ExportedArtifact(
        backend=backend,
        path=str(path),
        ir_version=ir_version,
        opset=opset,
        nodes=_graph_nodes(path),
        bytes=measure_artifact_bytes(path),
    )


def export_fp32_matmul(
    weight: torch.Tensor,
    path: str | Path,
    *,
    input_shape: tuple[int, int] | None = None,
    ir_version: int = DEFAULT_IR_VERSION,
    opset: int = DEFAULT_OPSET,
) -> ExportedArtifact:
    """Serialize ``weight`` as an fp32 ONNX ``MatMul`` (the class-4-CPU reference graph).

    Shapes:
        ``weight``: PyTorch ``(out_features, in_features)``; the ONNX initializer is its transpose
        ``(in_features, out_features)``. ``input_shape`` is accepted for symmetry with the quantized
        exports but is not needed (the graph uses a symbolic ``batch`` dimension).

    Dtypes / device:
        CPU float32 only; a CUDA tensor is rejected.

    Raises:
        OnnxExtraMissingError: If the optional ``onnx`` extra is not installed.
        ValueError: If ``weight`` is not 2-D or is a CUDA tensor.
        OSError: If the file cannot be written.
    """
    del input_shape  # the graph declares a symbolic batch dimension
    onnx = _import_onnx()
    model = _build_fp32_model(weight, ir_version=ir_version, opset=opset)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _clear_target(target)
    onnx.save(model, str(target))
    return _exported_artifact("onnx-fp32-matmul", target, ir_version, opset)


def export_int4_matmulnbits(
    weight: torch.Tensor,
    path: str | Path,
    *,
    block_size: int,
    symmetric: bool = True,
    accuracy_level: int = 4,
    input_shape: tuple[int, int] | None = None,
    ir_version: int = DEFAULT_IR_VERSION,
    opset: int = DEFAULT_OPSET,
) -> ExportedArtifact:
    """Serialize ``weight`` as an int4 weight-only ``MatMulNBits`` container (domain ``com.microsoft``).

    The int4 quantization uses ONNX Runtime's ``MatMulNBitsQuantizer`` with ``block_size`` = the
    group size and the requested symmetry, on a shape-inferred fp32 source graph, and is persisted
    through ``save_model_to_file(path, use_external_data_format=True)`` so the packed payload lands in
    the ``.data`` sidecar (``backend-capability.md`` §2.4a). The scales remain fp32 and inline.

    Raises:
        OnnxExtraMissingError: If the optional ``onnx`` / ``onnx-ir`` extra is not installed.
        RuntimeError: If the serialized graph's IR version exceeds 13 (onnxruntime would reject it).
    """
    del input_shape  # the graph declares a symbolic batch dimension
    matmul_nbits_quantizer, quant_utils, _, _ = _import_ort_quantization()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _clear_target(target)
    with tempfile.TemporaryDirectory(prefix="spectraquant-onnx-src-") as tmp:
        source = Path(tmp) / "fp32_source.onnx"
        _import_onnx().save(
            _build_fp32_model(weight, ir_version=ir_version, opset=opset), str(source)
        )
        config = matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
            block_size=int(block_size),
            is_symmetric=bool(symmetric),
            accuracy_level=int(accuracy_level),
            quant_format=quant_utils.QuantFormat.QOperator,
            op_types_to_quantize=("MatMul",),
        )
        quantizer = matmul_nbits_quantizer.MatMulNBitsQuantizer(
            quant_utils.load_model_with_shape_infer(source), algo_config=config
        )
        quantizer.process()
        wrapper = quantizer.model
        # ``quantizer.model`` is an ONNXModel wrapper; persistence MUST go through it.
        inner = getattr(wrapper, "model", None)
        if inner is not None:
            inner.ir_version = int(ir_version)
        wrapper.save_model_to_file(str(target), True)
    resolved = _assert_ir_version_supported(target)
    return _exported_artifact("onnx-int4-matmulnbits", target, resolved, opset)


def export_int8_matmulinteger(
    weight: torch.Tensor,
    path: str | Path,
    *,
    input_shape: tuple[int, int] | None = None,
    ir_version: int = DEFAULT_IR_VERSION,
    opset: int = DEFAULT_OPSET,
) -> ExportedArtifact:
    """Serialize ``weight`` as a dynamic-int8 ``MatMulInteger`` container.

    Uses ``onnxruntime.quantization.quantize_dynamic(..., weight_type=QInt8)`` on a temporary fp32
    source graph: only the weight is quantized (per-tensor, asymmetric — one fp32 scale and one int8
    zero-point), and the activations are quantized at run time by ``DynamicQuantizeLinear``.
    """
    del input_shape  # the graph declares a symbolic batch dimension
    _, _, quantize_dynamic, quant_type = _import_ort_quantization()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _clear_target(target)
    with tempfile.TemporaryDirectory(prefix="spectraquant-onnx-src-") as tmp:
        source = Path(tmp) / "fp32_source.onnx"
        _import_onnx().save(
            _build_fp32_model(weight, ir_version=ir_version, opset=opset), str(source)
        )
        quantize_dynamic(
            str(source),
            str(target),
            weight_type=quant_type.QInt8,
            op_types_to_quantize=["MatMul"],
        )
    resolved = _assert_ir_version_supported(target)
    return _exported_artifact("onnx-int8-matmulinteger", target, resolved, opset)


def reconcile_with_accounting(
    artifact: ExportedArtifact,
    *,
    shape: tuple[int, int],
    spec: QuantSpec | None,
) -> ByteReconciliation:
    """Reconcile a measured container against :mod:`spectraquant.quantization.accounting`.

    Shapes:
        ``shape`` is the **PyTorch** weight shape ``(out_features, in_features)`` and ``spec`` the
        quantizer spec whose axis matches the ONNX quantization axis (for ``MatMulNBits`` that is
        ``axis=0`` of the torch weight, i.e. the ONNX last axis ``N = out_features``).

    Assumptions / limitations:
        The residual ``measured_total - analytical_total`` is **expected to be non-zero** and is
        decomposed into the scale-dtype difference (fp32 on disk vs the model's fp16 per-group
        convention) and the measured container overhead. This function never forces equality: the
        analytical model does not contain graph/file overhead (``memory-accounting.md`` §6).
    """
    breakdown = None if spec is None else byte_breakdown(shape, spec)
    if breakdown is None:
        analytical_total = int(np.prod(shape)) * 4
        analytical_scales = 0
    else:
        analytical_total = breakdown.total_bytes
        analytical_scales = breakdown.scales_bytes
    measured = artifact.bytes
    residual = measured.total_bytes - analytical_total
    scale_dtype_delta = measured.scales_bytes - analytical_scales
    notes = [
        "residual is expected non-zero: the analytical model has no graph/file overhead",
        "scale_dtype_delta: ONNX Runtime writes scales at fp32; the accounting model uses fp16 "
        "for per-group scales",
        "container_overhead is measured (graph protobuf + node attributes + tensor names + padding)",
    ]
    return ByteReconciliation(
        artifact=artifact.path,
        shape=shape,
        spec=spec,
        analytical_total_bytes=analytical_total,
        measured_total_bytes=measured.total_bytes,
        residual_bytes=residual,
        scale_dtype_delta_bytes=scale_dtype_delta,
        container_overhead_bytes=measured.container_overhead_bytes,
        analytical=breakdown,
        notes=tuple(notes),
    )
