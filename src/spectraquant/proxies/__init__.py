"""Sensitivity-proxy package (Milestone 3 — not implemented).

Planned responsibilities: output-aware sensitivity estimation, proxy ranking of layers, and
validation of that ranking against exact toy computations.

Every entry point raises ``NotImplementedError``; see :mod:`spectraquant.proxies.sensitivity`.
"""

from __future__ import annotations

from spectraquant.proxies.sensitivity import (
    output_aware_sensitivity,
    rank_layers,
    sensitivity_report,
)

__all__ = ["output_aware_sensitivity", "rank_layers", "sensitivity_report"]
