"""Tests for the measurement-integrity scanner (``spectraquant.reporting.claim_guard``).

The scanner is only useful if three things hold: every forbidden pattern fires on a crafted
violation, the same phrase inside a prohibition (or a forbidden-claims table) is *not* a violation,
and the committed repository is clean. This module pins all three, plus determinism and the CLI
exit-code contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from spectraquant.cli.main import app
from spectraquant.paths import repo_root
from spectraquant.reporting.claim_guard import (
    CLAIM_PATTERNS,
    default_document_paths,
    scan_paths,
    scan_text,
)

runner = CliRunner()

#: A crafted violation per pattern: the forbidden claim with no negating or attributing context.
VIOLATIONS: dict[str, str] = {
    "latency_without_kernel_backend": ("Our int4 model runs at 2.5 ms per token on this laptop."),
    "fake_quant_as_storage": (
        "Our fake-quantized 4-bit model is 2x smaller than the fp16 checkpoint."
    ),
    "analytical_as_measured": (
        "We present the analytical 11680 B figure as the measured storage size."
    ),
    "gpu_service_asserted_locally": (
        "Our local vLLM server achieved 42 ms p99 end-to-end service latency."
    ),
    "cpu_vs_published_gpu": (
        "Our CPU int4 kernel is 1.7x faster than the published A100 GPU result."
    ),
    "unqualified_novelty": (
        "Our method is the first to combine low-rank preparation with low-bit training."
    ),
}

#: The same forbidden phrasing inside a prohibition: these must be quotes, never violations.
QUOTES: dict[str, str] = {
    "latency_without_kernel_backend": ("We never claim our int4 model is 2.5 ms per token."),
    "fake_quant_as_storage": (
        "A fake-quantized float checkpoint must not be reported as 2x smaller storage."
    ),
    "analytical_as_measured": ("The analytical estimate is never presented as measured."),
    "gpu_service_asserted_locally": (
        "A class-5 service latency number must not be asserted from a local 42 ms measurement."
    ),
    "cpu_vs_published_gpu": ("CPU figures must never be compared with published GPU numbers."),
    "unqualified_novelty": (
        "We must not claim the method is novel or the first to combine the two axes."
    ),
}


def test_every_pattern_is_covered_by_a_crafted_case() -> None:
    """A new pattern with no test case would silently widen the gate; forbid that."""
    names = {pattern.name for pattern in CLAIM_PATTERNS}
    assert set(VIOLATIONS) == names
    assert set(QUOTES) == names


@pytest.mark.parametrize("pattern_name", sorted(VIOLATIONS))
def test_crafted_violation_is_caught(pattern_name: str) -> None:
    report = scan_text(VIOLATIONS[pattern_name] + "\n", "crafted.md")

    matched = [f for f in report.findings if f.pattern == pattern_name]
    assert matched, f"{pattern_name} did not fire at all"
    assert all(f.verdict == "violation" for f in matched)
    assert report.violations


@pytest.mark.parametrize("pattern_name", sorted(QUOTES))
def test_crafted_quote_is_not_a_violation(pattern_name: str) -> None:
    report = scan_text(QUOTES[pattern_name] + "\n", "crafted.md")

    assert report.violations == (), f"{pattern_name} misread a prohibition as a violation"
    matched = [f for f in report.findings if f.pattern == pattern_name]
    assert matched and all(f.verdict == "quote" for f in matched)


def test_forbidden_claims_table_row_is_a_quote() -> None:
    """A match inside a table whose header says 'forbidden' is a quote (the taxonomy's own table)."""
    text = (
        "| Wrong (forbidden) | Why |\n"
        "|---|---|\n"
        '| "Our int4 model is 1.8x faster" | no real kernel |\n'
    )

    report = scan_text(text, "table.md")

    assert report.violations == ()
    assert any(f.verdict == "quote" for f in report.findings)


def test_prohibition_list_label_is_a_quote() -> None:
    """A bullet under a ``**Forbidden:**`` label is a rule, not a claim."""
    text = "Some section\n\n* **Forbidden:**\n  * reporting an analytical number as a measured byte count;\n"

    report = scan_text(text, "list.md")

    assert report.violations == ()
    assert any(f.verdict == "quote" for f in report.findings)


def test_explicitly_labelled_sentence_is_separated_not_mixed() -> None:
    """A sentence that labels its classes separates them, so it is skipped, not a violation."""
    text = "The report gives stored bytes (class 3) and a fake-quantization round-trip (class 1).\n"

    report = scan_text(text, "labelled.md")

    assert report.violations == ()
    assert report.skipped.get("classes_explicitly_labelled", 0) >= 1


def test_matching_run_without_deployment_cue_is_not_flagged() -> None:
    """Solver wall-clock is not a deployment property: no low-bit/inference cue, no match."""
    text = "CP-SAT is also 3-5x faster than enumeration at these sizes.\n"

    report = scan_text(text, "solver.md")

    assert [f for f in report.findings if f.pattern == "latency_without_kernel_backend"] == []


def test_fenced_code_block_is_not_scanned() -> None:
    """Documented limitation: fenced code (raw transcripts) is skipped, so it is not a violation."""
    text = "```text\nOur int4 model runs at 2.5 ms per token.\n```\n"

    report = scan_text(text, "code.md")

    assert report.findings == ()


def test_scan_is_deterministic() -> None:
    text = "\n".join(
        [VIOLATIONS["unqualified_novelty"], "# heading", QUOTES["cpu_vs_published_gpu"], ""]
    )

    first = scan_text(text, "det.md").to_json_dict()
    second = scan_text(text, "det.md").to_json_dict()

    assert first == second


def test_scan_paths_is_order_independent(tmp_path: Path) -> None:
    (tmp_path / "b.md").write_text(VIOLATIONS["unqualified_novelty"] + "\n", encoding="utf-8")
    (tmp_path / "a.md").write_text(
        VIOLATIONS["latency_without_kernel_backend"] + "\n", encoding="utf-8"
    )

    forward = scan_paths([tmp_path / "a.md", tmp_path / "b.md"])
    reverse = scan_paths([tmp_path / "b.md", tmp_path / "a.md"])

    assert [d.path for d in forward.documents] == [d.path for d in reverse.documents]
    assert forward.to_json_dict() == reverse.to_json_dict()


def test_cli_exits_non_zero_on_a_violation(tmp_path: Path) -> None:
    target = tmp_path / "bad.md"
    target.write_text(VIOLATIONS["fake_quant_as_storage"] + "\n", encoding="utf-8")

    result = runner.invoke(app, ["claim-guard", str(target)])

    assert result.exit_code == 1
    assert "violation" in result.output


def test_cli_exits_zero_on_a_clean_document(tmp_path: Path) -> None:
    target = tmp_path / "ok.md"
    target.write_text(QUOTES["fake_quant_as_storage"] + "\n", encoding="utf-8")

    result = runner.invoke(app, ["claim-guard", str(target)])

    assert result.exit_code == 0
    assert "0 violation(s)" in result.output


def test_cli_json_report_shape(tmp_path: Path) -> None:
    target = tmp_path / "bad.md"
    target.write_text(VIOLATIONS["unqualified_novelty"] + "\n", encoding="utf-8")

    result = runner.invoke(app, ["claim-guard", str(target), "--json"])
    payload = json.loads(result.output)

    assert result.exit_code == 1
    assert payload["totals"]["violations"] == 1
    assert payload["documents"][0]["findings"][0]["pattern"] == "unqualified_novelty"


def test_committed_repository_has_zero_violations() -> None:
    """The gate itself: the committed prose must not contain an unmarked forbidden claim."""
    paths = default_document_paths(repo_root())
    report = scan_paths(paths)

    assert len(report.documents) >= 50, "the default scan set is suspiciously small"
    assert report.violations == (), [
        f"{f.document}:{f.line} [{f.pattern}] {f.context!r}" for f in report.violations
    ]
    # The scan must be doing real work, not skipping everything.
    assert report.findings, "no findings at all: the patterns may have stopped matching"
