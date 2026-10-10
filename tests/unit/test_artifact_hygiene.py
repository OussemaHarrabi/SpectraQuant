"""Repository hygiene: no generated artifact may dominate the tree.

Regression for the pinned oracle fixture, which was committed at 85 290 746 bytes and 4 129 559 lines
because a 768x768 capture was stored as JSON number lists. It was compacted 151x (to 563 249 bytes) by
regenerating the input from its seed and keeping norms, spectra, hashes and a 64x64 convention-detector
submatrix instead of the full arrays (amendment A-0019).
"""

from __future__ import annotations

import json
from pathlib import Path

#: No committed artifact may exceed this. The limit is deliberately far above every legitimate fixture
#: (the largest is the compact oracle fixture at ~0.55 MB) and far below the 85 MB that slipped in.
MAX_ARTIFACT_BYTES = 4 * 1024 * 1024

SAMPLE_RESULTS = Path("artifacts/sample-results")


def _artifact_files() -> list[Path]:
    if not SAMPLE_RESULTS.is_dir():
        return []
    return sorted(path for path in SAMPLE_RESULTS.rglob("*") if path.is_file())


def test_no_committed_sample_artifact_exceeds_the_size_limit() -> None:
    offenders = [
        (path, path.stat().st_size)
        for path in _artifact_files()
        if path.stat().st_size > MAX_ARTIFACT_BYTES
    ]
    assert not offenders, (
        "committed artifacts over the size limit: "
        + ", ".join(f"{path} ({size} B)" for path, size in offenders)
        + f"; the limit is {MAX_ARTIFACT_BYTES} B (A-0019). Compact the artifact, record its sha256 "
        "and its retrieval instructions, and keep only the summary in Git."
    )


def test_the_oracle_fixture_records_its_raw_artifacts_hash_and_retrieval() -> None:
    """A compacted fixture must still say how to rebuild what it dropped."""
    fixture = json.loads((SAMPLE_RESULTS / "m3-oracle" / "loftq-nf2-block64.json").read_text("utf-8"))
    raw = fixture["loftq_768"]["raw_artifact"]
    assert len(raw["sha256"]) == 64
    assert raw["bytes"] > 0 and raw["lines"] > 0
    assert "Regenerate" in raw["retrieval"] or "regenerate" in raw["retrieval"]
    # The input is regenerated rather than stored, so its generator must be fully specified.
    generator = fixture["loftq_768"]["input"]
    assert generator["regenerate"].startswith("torch.manual_seed(")
    assert generator["shape"] == [768, 768]
    assert len(generator["sha256"]) == 64
