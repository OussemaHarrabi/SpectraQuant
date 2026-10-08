"""Reporting package: logging, git provenance, environment capture, run manifests.

Every run of the harness is described by a JSON manifest validated against
``artifacts/schemas/run-manifest.schema.json`` (see :mod:`spectraquant.reporting.manifests`).
"""

from __future__ import annotations

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
    "GitInfo",
    "ManifestValidationError",
    "RunManifest",
    "collect_hardware",
    "collect_software",
    "configure_logging",
    "environment_report",
    "get_logger",
    "git_info",
    "load_schema",
    "log_event",
    "process_peak_rss_mb",
    "resolve_device",
    "validate_manifest_dict",
    "write_manifest",
]
