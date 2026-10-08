"""Cloud execution substrate (``AGENTS.md`` §2b, ``design-cloud-adapter.md``).

This package is the only sanctioned path between the repository and the cloud GPU substrate. It
never executes scientific logic: it builds versioned notebooks from a frozen :class:`RunSpec`,
submits them through an official platform API where one exists, validates the returned artifact
bundle byte-for-byte, and records the outcome in an append-only registry.

Nothing here imports CUDA, and nothing here fabricates a result: the registry only records metrics
that came from a downloaded, checksum-validated ``run_manifest.json`` (``design-cloud-adapter.md``
§5).
"""

from __future__ import annotations

__doc__ = __doc__  # keep the module docstring explicit for tooling
