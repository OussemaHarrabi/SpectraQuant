"""Offline integrity checks and restrained interpretation of A-0023 evidence."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from spectraquant.cloud.collect import ValidationReport


def verify_collected_study(
    validation: ValidationReport, root: str | Path, *, study_name: str, study_key: str
) -> tuple[ValidationReport, dict[str, Any] | None, dict[str, Any] | None]:
    """Reject corrupt dynamic checkpoints *before* any validated registry entry."""
    if not validation.ok:
        return validation, None, None
    root = Path(root)
    try:
        matches = list(root.rglob(f"{study_name}/metrics.json"))
        if len(matches) != 1 or validation.manifest is None:
            raise ValueError("exactly one scientific report and its cloud manifest required")
        path = matches[0]
        report = json.loads(path.read_text(encoding="utf-8"))
        verify_study_manifest(path, validation.manifest, study_key=study_key)
        integrity = verify_checkpoints(report, path.parent)
        summarize(report)  # successful status, same token protocol and valid byte budget
        checksums = dict(validation.artifact_checksums)
        for arm, result in report["arms"].items():
            for name, info in result["storage"]["files"].items():
                relative = (path.parent / "checkpoints" / arm / name).relative_to(root).as_posix()
                logical = f"artifacts/runs/{study_name}/checkpoints/{arm}/{name}"
                checksums[logical if logical in checksums else relative] = (
                    "sha256:" + info["sha256"]
                )
        return replace(validation, artifact_checksums=checksums), report, integrity
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return (
            replace(
                validation,
                state="rejected",
                checksum_verified=False,
                reasons=[*validation.reasons, f"scientific integrity: {exc}"],
            ),
            None,
            None,
        )


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_study_manifest(
    metrics_path: str | Path, manifest: Mapping[str, Any], *, study_key: str = "phase2"
) -> None:
    """Anchor the metrics containing checkpoint hashes to the remote manifest."""
    expected = manifest.get("metrics", {}).get(f"{study_key}.full_report_sha256")
    if not expected or expected.removeprefix("sha256:") != _digest(Path(metrics_path)):
        raise ValueError("scientific metrics checksum differs from remote manifest")


def verify_checkpoints(report: Mapping[str, Any], root: str | Path) -> dict[str, Any]:
    """Verify measured files, payload lengths and full checkpoint totals.

    File checksums come from the downloaded metrics, which must separately be
    matched to the immutable cloud manifest. This checks self-consistency;
    it does not establish the integrity of an untrusted metrics file by itself.
    """
    root = Path(root).resolve()
    count, total = 0, 0
    for arm, result in report["arms"].items():
        if "storage" not in result:
            raise ValueError(f"{arm}: stored checkpoint evidence missing")
        directory = (root / "checkpoints" / arm).resolve()
        if not directory.is_relative_to(root):
            raise ValueError("checkpoint path escapes result directory")
        storage = result["storage"]
        measured = 0
        for name, expected in storage["files"].items():
            path = (directory / name).resolve()
            if not path.is_relative_to(directory):
                raise ValueError("artifact path escapes checkpoint directory")
            if not path.is_file() or path.stat().st_size != expected["bytes"]:
                raise ValueError(f"{arm}/{name}: missing file or size mismatch")
            if _digest(path) != expected["sha256"]:
                raise ValueError(f"{arm}/{name}: checksum mismatch")
            count += 1
            measured += path.stat().st_size
        if measured != storage["full_checkpoint_bytes"]:
            raise ValueError(f"{arm}: full checkpoint byte total differs")
        if (directory / "weights.sqpack").stat().st_size != storage["target_payload_bytes"]:
            raise ValueError(f"{arm}: target payload size differs")
        layout_path = directory / "layout.json"
        if layout_path.exists():
            layout = json.loads(layout_path.read_text(encoding="utf-8"))
            offset = 0
            for row in layout["targets"]:
                if row["offset"] != offset:
                    raise ValueError(f"{arm}: noncontiguous packed layout")
                offset += row["bytes"]
            if offset != storage["target_payload_bytes"]:
                raise ValueError(f"{arm}: packed layout does not cover payload")
        total += measured
    return {"verified_files": count, "total_downloaded_checkpoint_bytes": total}


def summarize(report: Mapping[str, Any]) -> dict[str, Any]:
    """A successful run can still be a negative scientific result."""
    if report["status"] != "success":
        raise ValueError(
            f"experiment is incomplete: {report.get('failure_reason', report['status'])}"
        )
    baseline = report["arms"]["uniform4"]
    base_ppl = baseline["test"]["perplexity"]
    if not math.isfinite(base_ppl) or base_ppl <= 0:
        raise ValueError("invalid uniform4 perplexity")
    budget = report["target_payload_budget_bytes"]
    if any(
        result["test"]["tokens"] != baseline["test"]["tokens"] for result in report["arms"].values()
    ):
        raise ValueError("different test token protocols")
    rows = []
    for arm, result in report["arms"].items():
        if result["test"]["tokens"] != baseline["test"]["tokens"]:
            raise ValueError("different test token protocols")
        payload = result["storage"]["target_payload_bytes"]
        if payload > budget:
            raise ValueError(f"{arm}: exceeds declared byte budget")
        ppl = result["test"]["perplexity"]
        if not math.isfinite(ppl) or ppl <= 0:
            raise ValueError("invalid treatment perplexity")
        rows.append(
            {
                "arm": arm,
                "perplexity": ppl,
                "perplexity_reduction_percent": 100 * (1 - ppl / base_ppl),
                "target_payload_bytes": payload,
                "full_checkpoint_bytes": result["storage"]["full_checkpoint_bytes"],
                "development_nll": result.get("development", report["uniform4_development"])["nll"],
                "delta_nll_ci95": result.get("paired_vs_uniform4", {}).get("ci95"),
                "surrogate_delta_nll": result.get("surrogate_delta_nll"),
            }
        )
    rows.sort(key=lambda x: (x["arm"] != "uniform4", x["arm"]))
    return {
        "selected_family": report["selection_frozen_before_test"]["family"],
        "rows": rows,
        "scope": (
            "one pretrained checkpoint; one deterministic clipping comparison; class-2 quality and class-3 stored bytes"
            if report["selection_frozen_before_test"]["family"] == "mseclip"
            else "one checkpoint; calibration-subset variation; class-2 quality and class-3 stored bytes"
        ),
        "limitations": [
            (
                "One pretrained checkpoint; new C4 validation holdout, with pretraining exposure unknown."
                if report.get("study_key") == "phase3"
                else "One pretrained checkpoint and one previously used test benchmark."
            ),
            "Three calibration subsets are not independent training replicates.",
            "Paired contiguous-block bootstrap is descriptive; adjacent blocks depend.",
            "No low-bit inference kernel, acceleration, downstream accuracy or generalization claim.",
            "Equal target-payload ceiling; full checkpoint bytes are separately measured.",
            "The marginal-NLL knapsack objective is approximate; interactions and dataset shift can break it.",
        ],
    }


def write_report(
    report: Mapping[str, Any], integrity: Mapping[str, Any], target: str | Path
) -> Path:
    summary = summarize(report)
    lines = [
        (
            "# SpectraQuant Phase 3: joint-loss diagnosis and controlled follow-up"
            if report.get("study_key") == "phase3"
            else "# SpectraQuant Phase 2: deployment-matched global-damage allocation"
        ),
        "",
        f"Source commit: `{report.get('git', {}).get('commit', 'unknown')}`.",
        "",
        f"Validation selected **{summary['selected_family']}** before the test split was read.",
        "",
        "Quality is float16 execution of reconstructed packed weights (class 2). Stored bytes are measured files (class 3).",
        f"Held-out evaluation: {report.get('data', {}).get('test', {}).get('id', 'unknown')} / {report.get('data', {}).get('test', {}).get('split', 'unknown')}; {report['arms']['uniform4']['test']['tokens']:,} predicted tokens.",
        "",
        "| Arm | Test perplexity ↓ | Reduction vs INT4 | Target payload B | Full checkpoint B | Validation NLL |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {row['arm']} | {row['perplexity']:.4f} | {row['perplexity_reduction_percent']:+.2f}% | {row['target_payload_bytes']:,} | {row['full_checkpoint_bytes']:,} | {row['development_nll']:.6f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        summary["scope"] + ".",
        "",
        f"Verified {integrity['verified_files']} checkpoint files against the collected report's checksums.",
        "",
    ]
    selected = [
        r
        for r in summary["rows"]
        if r["arm"].startswith(summary["selected_family"] + "_seed")
        or r["arm"] == summary["selected_family"]
    ]
    if summary["selected_family"] == "uniform4":
        lines.append(
            "No proposed family met the prespecified validation improvement threshold. Retain the baseline; do not claim a method improvement."
        )
    elif summary["selected_family"] == "mseclip" and selected:
        reduction = selected[0]["perplexity_reduction_percent"]
        lines.append(
            f"The validation-selected conventional clipping control changes held-out perplexity by {reduction:+.2f}% reduction over INT4 at the measured bytes above. This is one deterministic comparison on one pretrained checkpoint, not a replicated allocation-method gain."
        )
    elif selected:
        reductions = [r["perplexity_reduction_percent"] for r in selected]
        lines.append(
            f"The validation-selected family's test perplexity changes range from {min(reductions):+.2f}% to {max(reductions):+.2f}% reduction over INT4 across calibration subsets. This is a bounded exploratory result, not a universal gain."
        )
    lines += [
        "",
        "## Limits",
        "",
        *["- " + x for x in summary["limitations"]],
        "",
        "## Resource record",
        "",
        f"Scientific runner wall time: {report.get('wall_seconds', 0):.1f} s. Visible-device time upper bound through runner finish: {report.get('gpu_device_hours_upper_bound', 0):.3f} h. The outer cloud interval includes installation through runner finish but excludes final export and platform overhead; neither interval is exact Kaggle quota billing.",
        "",
        "## What this proves for a research CV",
        "",
        "A reproducible PyTorch/Hugging Face compression experiment with constrained OR-Tools allocation, actual serialization/reload, isolated calibration/validation/test logic, cloud execution, uncertainty reporting and honest negative-result analysis. Quantitative gain wording must use the selected-family table and this scope.",
        "",
    ]
    if report.get("study_key") == "phase3":
        lines += [
            "",
            "## Joint interaction diagnosis",
            "",
            "Negative delta NLL means better. Nonadditivity = actual joint minus summed marginal delta on the same calibration blocks.",
            "",
            "| Prior arm | Sum of marginal deltas | Actual joint delta | Nonadditivity gap | Disjoint training-block delta |",
            "|---|---:|---:|---:|---:|",
        ]
        for arm, row in report["diagnostics"].items():
            lines.append(
                f"| {arm} | {row['summed_marginal_delta']:+.6f} | {row['actual_joint_delta']:+.6f} | {row['nonadditivity_gap']:+.6f} | {row['disjoint_train_delta']:+.6f} |"
            )
        lines += [
            "",
            "## Development selection",
            "",
            "All family/cap choices were compared on WikiText validation before C4 validation was read. Selection is exploratory and may overfit development; there is no tuning on the new heldout.",
            "",
            "| Family | Mean development NLL |",
            "|---|---:|",
        ]
        for family, nll in report["selection_frozen_before_test"]["mean_development_nll"].items():
            lines.append(f"| {family} | {nll:.6f} |")
        lines += [
            "",
            "Clipping is a conventional weights-only control, not a novel SpectraQuant method. Its one deterministic checkpoint is not three independent replicates.",
            "",
            "The fresh holdout is not the prior WikiText test. Full Tier 2 reproduction, cross-model transfer and trainable-method gains remain unproved.",
            "",
        ]
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
    return target
