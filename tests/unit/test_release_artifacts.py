"""Regression test: the committed release tables/figures match the generator output.

The generated files under ``reports/tables/`` and ``reports/figures/`` must be byte-identical to what
``scripts/reproduce/generate_release_artifacts.py`` produces from the committed artifacts; otherwise
a table or figure has drifted from its source artifact. This test is the drift guard.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR_PATH = REPO_ROOT / "scripts" / "reproduce" / "generate_release_artifacts.py"


def _load_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("release_artifact_generator", GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator() -> ModuleType:
    return _load_generator()


def test_committed_tables_and_figures_match_generator(generator: ModuleType) -> None:
    committed = REPO_ROOT / "reports"
    mismatches: list[str] = []
    for filename, builder in generator.TABLES.items():
        produced = builder()
        path = committed / "tables" / filename
        if not path.exists() or path.read_text(encoding="utf-8") != produced:
            mismatches.append(f"tables/{filename}")
    for filename, builder in generator.FIGURES.items():
        produced = builder()
        path = committed / "figures" / filename
        if not path.exists() or path.read_text(encoding="utf-8") != produced:
            mismatches.append(f"figures/{filename}")
    assert not mismatches, (
        "committed release artifacts drifted from the generator output: "
        f"{mismatches}; regenerate with "
        "`uv run python scripts/reproduce/generate_release_artifacts.py`"
    )


def test_generate_is_deterministic(generator: ModuleType, tmp_path: Path) -> None:
    first = generator.generate(tmp_path / "a")
    second = generator.generate(tmp_path / "b")
    assert {p.relative_to(tmp_path / "a") for p in first} == {
        p.relative_to(tmp_path / "b") for p in second
    }
    for left, right in zip(sorted(first), sorted(second), strict=True):
        assert left.read_bytes() == right.read_bytes()


def test_generate_writes_required_artifacts(generator: ModuleType, tmp_path: Path) -> None:
    written = {p.name for p in generator.generate(tmp_path)}
    assert {
        "proxy-ranking.md",
        "allocator-frontier.md",
        "class4cpu-bytes.md",
        "class4cpu-latency.md",
        "pareto-damage-vs-bytes.svg",
    } <= written
