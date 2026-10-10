"""The release generator must work with and without the git-ignored cloud bundles.

The collected bundles live under `artifacts/runs/<run>/download/`, which is runtime state: CI has
none, a developer's tree has some. Both cases must produce a *truthful* artifact - a table that says
"no bundle is present" rather than an empty table that reads as a measured zero.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path("scripts/reproduce/generate_release_artifacts.py")


@pytest.fixture(scope="module")
def generator():
    spec = importlib.util.spec_from_file_location("generate_release_artifacts", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_cloud_table_is_truthful_without_a_bundle(generator) -> None:
    """With no bundle present the table must say so, not print zeros."""
    if generator._cloud_bundles():
        pytest.skip(
            "collected bundles are present; the absent-bundle path cannot be exercised here"
        )
    text = generator.table_cloud_arms()
    assert "No collected bundle is present" in text
    assert "not a measured zero" in text


def test_the_cloud_figure_renders_without_a_bundle(generator) -> None:
    svg = generator.figure_cloud_frontier()
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    if not generator._cloud_bundles():
        assert "no successful collected bundle" in svg


def test_the_cloud_table_carries_provenance_for_every_row(generator) -> None:
    """A row must never be readable without its commit, model revision and split."""
    bundles = generator._cloud_bundles()
    if not bundles:
        pytest.skip("no collected bundle present")
    text = generator.table_cloud_arms()
    for run_id, _bundle in bundles:
        assert f"`{run_id}`" in text
    assert "commit `" in text and "@ `" in text and "split `" in text


def test_a_failed_bundle_is_labelled_not_a_result(generator) -> None:
    failed = [
        run_id
        for run_id, bundle in generator._cloud_bundles()
        if generator._cloud_provenance(bundle)["status"] != "success"
    ]
    if not failed:
        pytest.skip("no failed bundle present")
    text = generator.table_cloud_arms()
    assert "**Not a result.**" in text


def test_generate_writes_every_table_and_figure(generator, tmp_path: Path) -> None:
    written = generator.generate(tmp_path)
    names = {path.name for path in written}
    assert "cloud-arms.md" in names
    assert "cloud-quality-memory.svg" in names
    for path in written:
        assert path.is_file() and path.stat().st_size > 0
