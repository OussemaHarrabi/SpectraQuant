"""SpectraQuant: budget-aware joint low-rank and low-bit training for efficient transformers.

This package implements the research programme described in ``AGENTS.md`` and
``docs/research/charter.md``. Every numeric result produced here carries a measurement-class
label (``AGENTS.md`` section 5) and is written to a schema-validated run manifest
(``artifacts/schemas/run-manifest.schema.json``).

Execution scope (AGENTS.md sections 2.3, 2b and 6): the local workstation runs Tier 0 only —
synthetic fixtures, unit/property tests, configuration validation, the CI smoke fixture, and result
analysis. Tiers 1-5 execute on the cloud notebook substrate and are **not implemented here**; every
GPU-dependent code path raises ``NotImplementedError`` rather than approximating (see ADR-0002).
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0.dev0"
