"""Offline integrity, fixed comparisons and cross-run transfer conclusions."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from spectraquant.evaluation.phase2_report import verify_checkpoints, verify_study_manifest


def authenticate_layer_scores(root: Path, manifest: Mapping[str, Any], model_key: str) -> str:
    """Anchor layer diagnostics to the downloaded remote manifest, not local hashes."""
    name = f"artifacts/runs/phase4-transfer-{model_key}/scores.json"
    entries = manifest.get("metrics", {}).get("cloud.artifact_checksums", [])
    declared = [
        entry.split("=", 1)[1]
        for entry in entries
        if isinstance(entry, str) and entry.startswith(name + "=")
    ]
    if len(declared) != 1:
        raise ValueError("layer-score checksum provenance missing or ambiguous")
    measured = "sha256:" + hashlib.sha256((root / "scores.json").read_bytes()).hexdigest()
    if measured != declared[0]:
        raise ValueError("layer-score checksum differs from remote manifest")
    return measured


def verify_transfer(
    report: Mapping[str, Any], root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify frozen identity/files/protocol before accepting a scientific run."""
    verify_study_manifest(root / "metrics.json", manifest, study_key="phase4")
    score_hash = authenticate_layer_scores(root, manifest, report["model_key"])
    if report["status"] != "success" or report["git"]["dirty"]:
        raise ValueError("incomplete or dirty scientific execution")
    frozen = json.loads((root / "frozen.json").read_text())
    if frozen != report["arms_frozen_before_holdout"]:
        raise ValueError("frozen checkpoint descriptor differs from scientific report")
    expected = {
        "rtn_axis0",
        "mse_axis0",
        "rtn_axis1",
        "mse_axis1",
        "act_axis1_seed0",
        "act_axis1_seed1",
        "act_axis1_seed2",
    }
    if set(report["arms"]) != expected or set(frozen["arms"]) != expected:
        raise ValueError("seven predeclared arms required")
    cfg = report["config"]
    if frozen["primary_comparison"] != cfg["primary_comparison"]:
        raise ValueError("primary comparison changed")
    from spectraquant.experiment_plan import load_plan

    model = load_plan(cfg["models"][report["model_key"]]["plan"]).models[0]
    if report["model"]["id"] != model.id or report["model"]["revision"] != model.revision:
        raise ValueError("scientific model differs from frozen pin carrier")
    integrity = verify_checkpoints(report, root)
    for arm, evidence in report["arms"].items():
        if evidence["storage"] != frozen["arms"][arm]:
            raise ValueError("evaluated checkpoint differs from frozen descriptor")
        axis = 0 if "axis0" in arm else 1
        layout = json.loads((root / "checkpoints" / arm / "layout.json").read_text())
        if len(layout["targets"]) != report["target_count"] or any(
            x["bits"] != 4
            or x["rank"] != 0
            or x["quant_spec"]["axis"] != axis
            or x["quant_spec"]["group_size"] != 32
            for x in layout["targets"]
        ):
            raise ValueError("checkpoint compression protocol differs")
        baseline = report["arms"][f"rtn_axis{axis}"]
        if any(
            evidence["storage"][k] != baseline["storage"][k]
            for k in ("target_payload_bytes", "full_checkpoint_bytes")
        ):
            raise ValueError("same-axis byte equality failed")
        if set(evidence["tests"]) != set(cfg["evaluation"]):
            raise ValueError("incomplete corpus coverage")
        for corpus, result in evidence["tests"].items():
            if (
                result["tokens"] != cfg["predicted_tokens"]
                or len(result["documents"]) < cfg["minimum_documents"]
            ):
                raise ValueError("evaluation token/document protocol differs")
            reference = baseline["tests"][corpus]
            if sum(x["tokens"] for x in result["documents"]) != result[
                "tokens"
            ] or not math.isclose(
                sum(x["nll_sum"] for x in result["documents"]) / result["tokens"],
                result["nll"],
                rel_tol=1e-10,
            ):
                raise ValueError("document aggregates disagree with corpus metrics")
            if [(x["document_id"], x["tokens"]) for x in result["documents"]] != [
                (x["document_id"], x["tokens"]) for x in reference["documents"]
            ]:
                raise ValueError("paired document protocol differs")
            if not math.isfinite(result["nll"]) or not math.isclose(
                math.exp(result["nll"]), result["perplexity"], rel_tol=1e-10
            ):
                raise ValueError("invalid perplexity/NLL")
    return {**integrity, "authenticated_scores_sha256": score_hash}


def aggregate_transfer(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """No raw-PPL averaging across tokenizers; four fixed paired log ratios."""
    if len(reports) != 2 or {r["model_key"] for r in reports} != {"smol360", "pythia410"}:
        raise ValueError("both models required before a transfer conclusion")
    first, second = reports
    if first["git"]["commit"] != second["git"]["commit"]:
        raise ValueError("scientific source commits differ")
    if any(
        first["data"][key]["sha256"] != second["data"][key]["sha256"]
        for key in ("calibration", "c4_reserved", "pg19")
    ):
        raise ValueError("cross-model corpus/calibration fingerprint differs")
    configs = [{k: v for k, v in r["config"].items() if k != "selected_model_key"} for r in reports]
    if configs[0] != configs[1] or any(r["status"] != "success" for r in reports):
        raise ValueError("scientific configuration/status differs")
    cells = []
    for report in reports:
        for corpus in ("c4_reserved", "pg19"):
            base = report["arms"]["rtn_axis0"]["tests"][corpus]
            treatment = report["arms"]["mse_axis0"]["tests"][corpus]
            delta = treatment["nll"] - base["nll"]
            cells.append(
                {
                    "model": report["model_key"],
                    "corpus": corpus,
                    "rtn_perplexity": base["perplexity"],
                    "clipping_perplexity": treatment["perplexity"],
                    "delta_nll": delta,
                    "reduction_percent": 100 * (1 - math.exp(delta)),
                    "delta_nll_ci95": treatment.get("paired_vs_same_axis_rtn", {}).get("ci95"),
                    "document_clusters": len(treatment.get("documents", [])),
                    "fp16_reference_perplexity": report.get("fp16_tests", {})
                    .get(corpus, {})
                    .get("perplexity"),
                }
            )
    positive = sum(c["delta_nll"] < 0 for c in cells)
    reduction = 100 * (1 - math.exp(sum(c["delta_nll"] for c in cells) / len(cells)))
    gate = configs[0]["transfer_gate"]
    return {
        "primary_cells": cells,
        "positive_primary_cells": positive,
        "geometric_mean_reduction_percent": reduction,
        "transfer_gate_passed": positive >= gate["positive_cells_required"]
        and reduction >= gate["geometric_mean_reduction_percent_minimum"],
        "gate_frozen_before_results": gate,
        "scope": "two pretrained checkpoints, two reserved corpus slices; conventional clipping transfer, not original allocator or training replication",
    }


def write_transfer_report(reports: Sequence[Mapping[str, Any]], target: Path) -> dict[str, Any]:
    summary = aggregate_transfer(reports)
    lines = [
        "# SpectraQuant: fixed clipping transfer across two models and text domains",
        "",
        "The fixed primary comparison is MSE clipping vs RTN with group32/axis0. All arms were stored and frozen before heldout loading.",
        "",
        "| Model | Corpus | RTN PPL | Clipped PPL | Reduction | Documents | Descriptive delta-NLL CI |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for cell in summary["primary_cells"]:
        interval = cell["delta_nll_ci95"]
        ci = f"[{interval[0]:+.5f}, {interval[1]:+.5f}]" if interval else "not available"
        lines.append(
            f"| {cell['model']} | {cell['corpus']} | {cell['rtn_perplexity']:.4f} | {cell['clipping_perplexity']:.4f} | {cell['reduction_percent']:+.2f}% | {cell['document_clusters']} | {ci} |"
        )
    lines += [
        "",
        f"Positive primary cells: {summary['positive_primary_cells']}/4. Geometric mean reduction: {summary['geometric_mean_reduction_percent']:.3f}%. Frozen transfer gate passed: **{summary['transfer_gate_passed']}**.",
        "",
        "Perplexities are tokenizer-dependent; do not compare or average raw values between models. The summary combines within-cell log ratios only.",
        "",
        "## Predeclared secondary comparisons",
        "",
        "Axis1 MSE/activation-weighted clipping compare with their own axis1 RTN baseline. Calibration subsets are not training seeds. These are conventional criteria, not full AWQ or a novel SpectraQuant algorithm.",
        "",
        "| Model | Corpus | Arm | PPL | Reduction vs same-axis RTN | Payload B | Full checkpoint B |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for report in reports:
        for arm, evidence in report["arms"].items():
            axis = 0 if "axis0" in arm else 1
            for corpus, result in evidence["tests"].items():
                base = report["arms"][f"rtn_axis{axis}"]["tests"][corpus]
                change = 100 * (1 - result["perplexity"] / base["perplexity"])
                storage = evidence["storage"]
                lines.append(
                    f"| {report['model_key']} | {corpus} | {arm} | {result['perplexity']:.4f} | {change:+.2f}% | {storage['target_payload_bytes']:,} | {storage['full_checkpoint_bytes']:,} |"
                )
    lines += [
        "",
        "## Scope and reproducibility",
        "",
        "Class2: dense FP16 execution of reconstructed packed checkpoints. Class3: actual stored target, layout and unchanged-weight files, tokenizer/runtime excluded. Same-axis comparisons have exact target and full-checkpoint byte equality. No low-bit GPU kernel or acceleration claim.",
        "",
        "51,200 predicted tokens per corpus/model with document boundaries. Paired document bootstrap is descriptive corpus uncertainty; no model-seed or publication significance claim. Pretraining exposure is unknown. Original Tier2, original allocation efficacy and complete LR-QAT reproduction remain unproved.",
        "",
        "The narrative progresses from a failed allocator, through measured nonadditivity/calibration-transfer failure, to a fixed simpler criterion and this external transfer test. Report all negative cells. Do not select or retune an arm on these exposed holdouts.",
        "",
    ]
    for report in reports:
        lines.append(
            f"- {report['model_key']}: scientific `{report['git']['commit']}`, clean={not report['git']['dirty']}; runner {report['wall_seconds']:.1f}s, approximate visible-device time {report['gpu_device_hours_upper_bound']:.3f}h, excluding installation/export and platform overhead; not exact quota billing."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    target.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
