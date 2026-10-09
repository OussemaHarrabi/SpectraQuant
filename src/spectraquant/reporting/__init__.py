"""Reporting package: logging, git provenance, environment capture, run manifests.

Every run of the harness is described by a JSON manifest validated against
``artifacts/schemas/run-manifest.schema.json`` (see :mod:`spectraquant.reporting.manifests`).
"""

from __future__ import annotations

from spectraquant.reporting.claim_guard import (
    ClaimPattern,
    DocumentReport,
    Finding,
    ScanReport,
    default_document_paths,
    scan_document,
    scan_paths,
    scan_text,
)
from spectraquant.reporting.comparability import (
    ComparisonVerdict,
    UnequalMemoryComparison,
    assert_equal_memory,
    compare_manifest_files,
    load_comparable_manifests,
)
from spectraquant.reporting.environment import (
    collect_hardware,
    collect_software,
    environment_report,
    process_peak_rss_mb,
    resolve_device,
)
from spectraquant.reporting.gitinfo import GitInfo, git_info
from spectraquant.reporting.logging import configure_logging, get_logger, log_event
from spectraquant.reporting.manifests import (
    ManifestValidationError,
    RunManifest,
    load_schema,
    validate_manifest_dict,
    write_manifest,
)

__all__ = [
    "ClaimPattern",
    "ComparisonVerdict",
    "DocumentReport",
    "Finding",
    "GitInfo",
    "ManifestValidationError",
    "RunManifest",
    "ScanReport",
    "UnequalMemoryComparison",
    "assert_equal_memory",
    "collect_hardware",
    "collect_software",
    "compare_manifest_files",
    "configure_logging",
    "default_document_paths",
    "environment_report",
    "get_logger",
    "git_info",
    "load_comparable_manifests",
    "load_schema",
    "log_event",
    "process_peak_rss_mb",
    "resolve_device",
    "scan_document",
    "scan_paths",
    "scan_text",
    "validate_manifest_dict",
    "write_manifest",
]
