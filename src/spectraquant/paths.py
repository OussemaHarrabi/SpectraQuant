"""Filesystem locations for the repository.

The repository root is discovered by walking up from this file until a directory containing
``pyproject.toml`` (and preferably ``AGENTS.md``) is found. That keeps the package usable when it
is installed editable, when it is imported from a checkout, and when tests run from any cwd.

Set ``SPECTRAQUANT_REPO_ROOT`` to override discovery (useful for CI, Docker and throwaway
scripts).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

__all__ = [
    "artifacts_dir",
    "configs_dir",
    "repo_root",
    "sample_results_dir",
    "schemas_dir",
]

_ENV_OVERRIDE = "SPECTRAQUANT_REPO_ROOT"


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Return the absolute path of the repository root."""
    override = os.environ.get(_ENV_OVERRIDE)
    if override:
        return Path(override).expanduser().resolve()

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "AGENTS.md").is_file():
            return parent
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd().resolve()


def configs_dir() -> Path:
    """Return ``<repo>/configs`` (Hydra composition root)."""
    return repo_root() / "configs"


def artifacts_dir() -> Path:
    """Return ``<repo>/artifacts`` (schemas, manifests, sample results)."""
    return repo_root() / "artifacts"


def schemas_dir() -> Path:
    """Return ``<repo>/artifacts/schemas``."""
    return artifacts_dir() / "schemas"


def sample_results_dir() -> Path:
    """Return ``<repo>/artifacts/sample-results`` (committed smoke output)."""
    return artifacts_dir() / "sample-results"
