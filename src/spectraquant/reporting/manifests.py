"""Run manifests: construction, JSON-Schema validation and writing.

The JSON Schema in ``artifacts/schemas/run-manifest.schema.json`` is the contract; the Pydantic
model in this module is the typed constructor for it. :func:`write_manifest` always validates
against the schema before touching the filesystem, so a manifest on disk is schema-valid by
construction.

Three rules cannot be expressed in JSON Schema and are enforced in code:

1. ``measurement_class`` may be ``null`` only when ``compression.method == "none"`` (AGENTS.md
   section 5: every *result* carries a class; the uncompressed smoke harness produces no
   compression number).
2. ``timestamp_utc`` must be a real UTC timestamp (the schema's ``format`` keyword is advisory and
   is not checked by default).
3. The equal-memory byte fields must be internally consistent (AGENTS.md section 4.5): declaring
   ``bytes_source == "measured"`` requires a non-null ``measured_bytes`` and the
   ``serializer`` that produced it; declaring ``accounted_bytes`` may not coexist with a
   class-3 ``measured_bytes`` figure, so a measurement can never be silently downgraded to the
   analytical estimate. The comparison itself lives in
   :mod:`spectraquant.reporting.comparability`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict, Field

from spectraquant.paths import schemas_dir

__all__ = [
    "SCHEMA_FILENAME",
    "SCHEMA_VERSION",
    "CompressionBlock",
    "HardwareBlock",
    "ManifestValidationError",
    "RunManifest",
    "TrainingBlock",
    "build_run_id",
    "load_manifest",
    "load_schema",
    "manifest_path",
    "utc_timestamp",
    "validate_manifest_dict",
    "validate_manifest_file",
    "write_manifest",
]

SCHEMA_FILENAME = "run-manifest.schema.json"
SCHEMA_VERSION = 1


class ManifestValidationError(ValueError):
    """Raised when a manifest document violates the published schema or the semantic rules."""

    def __init__(self, errors: Iterable[str], *, source: str | None = None) -> None:
        self.errors: list[str] = list(errors)
        self.source = source
        location = f" ({source})" if source else ""
        detail = "\n  - ".join(self.errors) if self.errors else "unknown error"
        super().__init__(f"invalid run manifest{location}:\n  - {detail}")


# --------------------------------------------------------------------------------------
# Typed blocks
# --------------------------------------------------------------------------------------
class HardwareBlock(BaseModel):
    """Machine the run executed on. Values are measured, never inferred."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    platform: str
    system: str
    release: str
    machine: str
    processor: str | None = None
    cpu_count_logical: int | None = Field(default=None, ge=1)
    ram_total_mb: float | None = Field(default=None, ge=0.0)
    gpu_available: bool = False
    gpu_devices: list[str] = Field(default_factory=list)
    torch_device: str = "cpu"
    torch_threads: int = Field(default=1, ge=1)


class CompressionBlock(BaseModel):
    """Compression plan actually applied by the run.

    The byte fields are the equal-memory comparison contract: ``accounted_bytes`` is the
    class-1 analytical figure, ``measured_bytes`` the class-3 serialized figure, and
    ``bytes_source`` declares which of the two this run offers for comparison (see
    :mod:`spectraquant.reporting.comparability`).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: str = "none"
    ranks: list[int] | None = None
    bits: int | None = None
    group_size: int | None = None
    exclusions: list[str] = Field(default_factory=list)
    accounted_bytes: int | None = Field(default=None, ge=0)
    measured_bytes: int | None = Field(default=None, ge=0)
    measured_bytes_tolerance: int | float | None = None
    serializer: str | None = None
    nominal_bits_per_param: float | None = Field(default=None, ge=0.0)
    measured_bits_per_param: float | None = Field(default=None, ge=0.0)
    bytes_source: Literal["accounted", "measured"] | None = None

    @property
    def is_uncompressed(self) -> bool:
        """True when this block describes an uncompressed baseline."""
        return self.method == "none"


class TrainingBlock(BaseModel):
    """Training budget consumed by the run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    steps: int = Field(ge=0)
    tokens: int = Field(ge=0)
    wall_time_s: float = Field(ge=0.0)
    peak_mem_mb: float | None = Field(default=None, ge=0.0)


class RunManifest(BaseModel):
    """One executed run, mirroring ``run-manifest.schema.json`` field for field."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = SCHEMA_VERSION
    run_id: str = Field(min_length=1, max_length=128)
    timestamp_utc: str
    git_commit: str | None = None
    git_dirty: bool | None = None
    config_path: str
    resolved_config: dict[str, Any]
    model_id: str | None = None
    model_revision: str | None = None
    dataset_ids: list[str] = Field(default_factory=list)
    dataset_revisions: list[str | None] = Field(default_factory=list)
    dataset_checksums: list[str] = Field(default_factory=list)
    split: str = "train"
    seed: int = Field(ge=0)
    hardware: HardwareBlock
    software: dict[str, str | None] = Field(default_factory=dict)
    compression: CompressionBlock = CompressionBlock()
    theoretical_bits: float | None = Field(default=None, ge=0.0)
    packed_bytes: int | None = Field(default=None, ge=0)
    runtime_backend: str = "torch-cpu-fp32"
    measurement_class: int | None = Field(default=None, ge=1, le=5)
    training: TrainingBlock
    metrics: dict[str, Any] = Field(default_factory=dict)
    status: str = "success"
    failure_reason: str | None = None
    log_path: str | None = None
    artifact_paths: list[str] = Field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable document (no Path/Enum/datetime objects)."""
        return self.model_dump(mode="json")


# --------------------------------------------------------------------------------------
# Schema loading and validation
# --------------------------------------------------------------------------------------
def schema_path() -> Path:
    """Absolute path of the published manifest schema."""
    return schemas_dir() / SCHEMA_FILENAME


def load_schema(path: Path | str | None = None) -> dict[str, Any]:
    """Load the JSON Schema from disk (raises ``FileNotFoundError`` when absent)."""
    target = Path(path) if path is not None else schema_path()
    with target.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    if not isinstance(schema, dict):  # pragma: no cover - a broken schema file
        raise ManifestValidationError(
            ["schema file does not contain a JSON object"], source=str(target)
        )
    return schema


def _jsonable(value: Any) -> Any:
    """Recursively coerce a value into something ``json.dumps`` accepts."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    return str(value)


def _compression_byte_errors(compression: Any) -> list[str]:
    """Equal-memory byte-field consistency rules (AGENTS.md section 4.5)."""
    if not isinstance(compression, Mapping):
        return []

    errors: list[str] = []
    source = compression.get("bytes_source")
    accounted_bytes = compression.get("accounted_bytes")
    measured_bytes = compression.get("measured_bytes")
    serializer = compression.get("serializer")

    if source == "measured" and measured_bytes is None:
        errors.append(
            "compression.bytes_source is 'measured' but compression.measured_bytes is null "
            "(a class-3 comparison needs a measured byte count)"
        )
    if source == "accounted" and accounted_bytes is None:
        errors.append(
            "compression.bytes_source is 'accounted' but compression.accounted_bytes is null "
            "(a class-1 comparison needs the analytical byte count)"
        )
    if source == "accounted" and measured_bytes is not None:
        errors.append(
            "compression.bytes_source is 'accounted' while compression.measured_bytes is "
            f"{measured_bytes}; a class-3 measurement may not be downgraded to the class-1 "
            "estimate - declare bytes_source 'measured' instead"
        )
    if measured_bytes is not None and not serializer:
        errors.append(
            "compression.serializer (the frozen invocation that produced measured_bytes) is "
            "required when compression.measured_bytes is reported"
        )

    return errors


def _semantic_errors(document: Mapping[str, Any]) -> list[str]:
    """Cross-field rules that JSON Schema cannot express."""
    errors: list[str] = []

    compression = document.get("compression")
    method = compression.get("method") if isinstance(compression, Mapping) else None
    measurement_class = document.get("measurement_class")
    if measurement_class is None and method not in (None, "none"):
        errors.append(
            "measurement_class must be 1-5 when compression.method is "
            f"{method!r} (only method='none' may omit a class)"
        )
    errors.extend(_compression_byte_errors(compression))

    timestamp = document.get("timestamp_utc")
    if isinstance(timestamp, str):
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            errors.append(f"timestamp_utc is not ISO 8601: {timestamp!r}")
        else:
            if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
                errors.append(f"timestamp_utc must be UTC: {timestamp!r}")

    status = document.get("status")
    failure_reason = document.get("failure_reason")
    if status == "success" and failure_reason is not None:
        errors.append("failure_reason must be null when status == 'success'")
    if status in {"failed", "aborted"} and not failure_reason:
        errors.append(f"failure_reason is required when status == {status!r}")

    return errors


def validate_manifest_dict(
    document: Mapping[str, Any],
    *,
    schema: dict[str, Any] | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """Validate a manifest document against the schema and the semantic rules.

    Args:
        document: the manifest mapping (already JSON-like).
        schema: pre-loaded schema (loaded from disk when omitted).
        source: label used in the error message (e.g. the file path).

    Returns:
        The validated document, unchanged.

    Raises:
        ManifestValidationError: schema violations or semantic rule violations.
    """
    active_schema = schema if schema is not None else load_schema()
    validator = Draft202012Validator(active_schema)
    errors = [
        f"{'/'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
        for error in sorted(
            validator.iter_errors(document), key=lambda err: list(err.absolute_path)
        )
    ]
    errors.extend(_semantic_errors(document))
    if errors:
        raise ManifestValidationError(errors, source=source)
    return dict(document)


def validate_manifest_file(
    path: Path | str, *, schema: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Load a manifest file from disk and validate it."""
    target = Path(path)
    try:
        document = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestValidationError([f"file not found: {target}"], source=str(target)) from exc
    except json.JSONDecodeError as exc:
        raise ManifestValidationError([f"invalid JSON: {exc}"], source=str(target)) from exc
    if not isinstance(document, dict):
        raise ManifestValidationError(["manifest must be a JSON object"], source=str(target))
    return validate_manifest_dict(document, schema=schema, source=str(target))


def load_manifest(path: Path | str, *, validate: bool = True) -> RunManifest:
    """Read a manifest from disk into the typed model, validating by default."""
    target = Path(path)
    document = json.loads(target.read_text(encoding="utf-8"))
    if validate:
        validate_manifest_dict(document, source=str(target))
    return RunManifest.model_validate(document)


# --------------------------------------------------------------------------------------
# Construction helpers
# --------------------------------------------------------------------------------------
def utc_timestamp(moment: datetime | None = None) -> str:
    """Return an ISO 8601 UTC timestamp with a ``Z`` suffix (second precision or finer)."""
    current = (moment or datetime.now(UTC)).astimezone(UTC)
    return current.isoformat(timespec="seconds").replace("+00:00", "Z")


def build_run_id(
    experiment: str,
    *,
    moment: datetime | None = None,
    git_commit: str | None = None,
    suffix: str | None = None,
) -> str:
    """Build a unique, filesystem-safe run id: ``<experiment>-<utcstamp>-<sha12|nogit>-<suffix>``."""
    stamp = (moment or datetime.now(UTC)).astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    sha = (git_commit or "nogit")[:12]
    tail = suffix if suffix is not None else f"{_random_suffix()}"
    raw = f"{experiment}-{stamp}-{sha}-{tail}"
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in raw)
    return cleaned[:128]


def _random_suffix() -> str:
    import uuid

    return uuid.uuid4().hex[:6]


def manifest_path(run_id: str, results_dir: Path | str) -> Path:
    """Path of a run manifest inside ``results_dir``."""
    return Path(results_dir) / f"{run_id}.manifest.json"


def write_manifest(
    manifest: RunManifest | Mapping[str, Any],
    path: Path | str,
    *,
    schema: dict[str, Any] | None = None,
) -> Path:
    """Validate ``manifest`` against the schema and write it as pretty-printed JSON.

    Args:
        manifest: typed model or mapping.
        path: destination file (parent directories are created).
        schema: pre-loaded schema (loaded from disk when omitted).

    Returns:
        The path written.

    Raises:
        ManifestValidationError: the document is not schema-valid.
    """
    document = manifest.to_json_dict() if isinstance(manifest, RunManifest) else _jsonable(manifest)
    validate_manifest_dict(document, schema=schema, source=str(path))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target
