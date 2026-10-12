# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["matplotlib==3.10.8", "numpy==2.2.6"]
# ///
"""Rebuild scientific figures from verified public numerical evidence only.

No model execution, dataset downloads, tests on new data or method selection.
Use uv run scripts/reproduce/research_figures.py --repo . --out reports/figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, RED = "#2474a8", "#d78628", "#b44b4b"
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.hashsalt": "spectraquant-figures-v1",
    }
)


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, default=Path("reports/figures"))
    parser.add_argument("--require-transfer", action="store_true")
    args = parser.parse_args()
    root, out = args.repo.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    inputs, outputs = [], []

    def load(relative: str) -> dict:
        path = root / relative
        inputs.append({"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        return read(path)

    def save(fig, name: str) -> None:
        for extension in ("svg", "png"):
            path = out / f"{name}.{extension}"
            fig.savefig(
                path,
                dpi=180,
                bbox_inches="tight",
                metadata={"Date": None}
                if extension == "svg"
                else {"Software": "SpectraQuant / Matplotlib"},
            )
            if extension == "svg":
                # Matplotlib emits trailing spaces and platform text newlines.
                # Canonicalize these before recording the published file hash.
                path.write_text(
                    "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
            outputs.append(
                {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            )
        plt.close(fig)

    phase3 = load("artifacts/sample-results/phase3-controlled-followup/metrics.json")
    if phase3["status"] != "success" or phase3["git"]["dirty"]:
        raise ValueError("Phase3 evidence is not a successful clean scientific run")
    names = sorted(phase3["diagnostics"])
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 4.6), layout="constrained")
    for offset, field, label, color in (
        (-0.25, "summed_marginal_delta", "Predicted: summed local gains", BLUE),
        (0, "actual_joint_delta", "Measured: joint calibration", ORANGE),
        (0.25, "disjoint_train_delta", "Measured: disjoint training blocks", RED),
    ):
        ax.bar(
            x + offset,
            [-phase3["diagnostics"][n][field] for n in names],
            width=0.24,
            label=label,
            color=color,
        )
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set(
        xticks=x,
        xticklabels=[n.replace("_seed", " / subset ") for n in names],
        ylabel="NLL improvement vs uniform INT4 (positive is better)",
        title="Local predicted gains overstate joint gains and fail to transfer",
    )
    ax.tick_params(axis="x", labelrotation=18)
    highest = max(-phase3["diagnostics"][n]["summed_marginal_delta"] for n in names)
    ax.set_ylim(top=highest * 1.35)
    ax.legend(fontsize=8, loc="upper center", ncol=3)
    fig.supxlabel("SmolLM2-135M; three calibration subsets, not training replicates", fontsize=9)
    save(fig, "allocation-prediction-vs-joint")

    fig, ax = plt.subplots(figsize=(6.5, 4), layout="constrained")
    labels = ["Uniform INT4", "Conventional MSE clipping"]
    values = [phase3["arms"][a]["test"]["perplexity"] for a in ("uniform4", "mseclip")]
    bars = ax.bar(labels, values, color=[BLUE, ORANGE], width=0.5)
    ax.bar_label(bars, labels=[f"{v:.4f}" for v in values], padding=4)
    ax.set(
        ylim=(0, max(values) * 1.2),
        ylabel="C4 holdout perplexity (lower is better)",
        title=f"Fixed control: {100 * (1 - values[1] / values[0]):.2f}% lower perplexity",
    )
    storage = phase3["arms"]["uniform4"]["storage"]
    if (
        storage["full_checkpoint_bytes"]
        != phase3["arms"]["mseclip"]["storage"]["full_checkpoint_bytes"]
    ):
        raise ValueError("Phase3 full checkpoint parity failed")
    token_count = phase3["arms"]["uniform4"]["test"]["tokens"]
    if token_count != phase3["arms"]["mseclip"]["test"]["tokens"]:
        raise ValueError("Phase3 evaluated token counts differ")
    fig.supxlabel(
        f"{token_count:,} predicted tokens; {storage['full_checkpoint_bytes']:,} B per checkpoint\n"
        "One deterministic comparison; dense reconstruction quality, no speed claim",
        fontsize=9,
    )
    save(fig, "fixed-clipping-first-holdout")

    transfer = root / "docs/results/phase4-clipping-transfer.json"
    if not transfer.exists():
        if args.require_transfer:
            raise FileNotFoundError("Verified two-model transfer evidence is required")
    else:
        summary = load("docs/results/phase4-clipping-transfer.json")
        reports = {}
        objective_rows = []
        for key in ("smol360", "pythia410"):
            report = load(f"artifacts/sample-results/phase4-transfer/{key}/metrics.json")
            attestation = load(f"artifacts/sample-results/phase4-transfer/{key}/collection.json")
            if report["status"] != "success" or report["git"]["dirty"]:
                raise ValueError("Invalid transfer scientific run")
            if inputs[-2]["sha256"] != attestation["metrics_sha256"].removeprefix("sha256:"):
                raise ValueError("Transfer metrics do not match extracted collection attestation")
            reports[key] = report
            scores = load(f"artifacts/sample-results/phase4-transfer/{key}/scores.json")
            if inputs[-1]["sha256"] != attestation["scores_sha256"].removeprefix("sha256:"):
                raise ValueError("Layer scores do not match verified collection attestation")
            aggregates = {}
            for arm in report["arms"]:
                squared_error = sum(
                    row["arms"][arm]["weight_mse"] * math.prod(row["shape"]) for row in scores
                )
                diagonal_proxy = sum(
                    row["arms"][arm]["diagonal_output_error_proxy"] * row["shape"][0]
                    for row in scores
                )
                groups = sum(sum(row["arms"][arm]["alpha_counts"].values()) for row in scores)
                clipped = sum(
                    count
                    for row in scores
                    for alpha, count in row["arms"][arm]["alpha_counts"].items()
                    if float(alpha) < 1
                )
                aggregates[arm] = {
                    "weight_squared_error_sum": squared_error,
                    "diagonal_proxy_sum": diagonal_proxy,
                    "clipped_group_percent": 100 * clipped / groups,
                }
            for arm, measured in aggregates.items():
                axis = 0 if "axis0" in arm else 1
                baseline = aggregates[f"rtn_axis{axis}"]
                objective_rows.append(
                    {
                        "model": key,
                        "arm": arm,
                        **measured,
                        "weight_error_reduction_percent": 100
                        * (
                            1
                            - measured["weight_squared_error_sum"]
                            / baseline["weight_squared_error_sum"]
                        ),
                        "diagonal_proxy_reduction_percent": 100
                        * (1 - measured["diagonal_proxy_sum"] / baseline["diagonal_proxy_sum"]),
                        **{
                            f"{corpus}_ppl_reduction_percent": 100
                            * (
                                1
                                - math.exp(
                                    report["arms"][arm]["tests"][corpus]["nll"]
                                    - report["arms"][f"rtn_axis{axis}"]["tests"][corpus]["nll"]
                                )
                            )
                            for corpus in ("c4_reserved", "pg19")
                        },
                    }
                )
        cells = summary["primary_cells"]
        if len(cells) != 4:
            raise ValueError("Four primary cells required")
        fig, ax = plt.subplots(figsize=(9, 4.6), layout="constrained")
        for i, cell in enumerate(cells):
            report = reports[cell["model"]]
            base = report["arms"]["rtn_axis0"]
            treat = report["arms"]["mse_axis0"]
            if any(
                base["storage"][k] != treat["storage"][k]
                for k in ("target_payload_bytes", "full_checkpoint_bytes")
            ):
                raise ValueError("Transfer primary storage parity failed")
            corpus = cell["corpus"]
            delta = treat["tests"][corpus]["nll"] - base["tests"][corpus]["nll"]
            value = 100 * (1 - math.exp(delta))
            if not math.isclose(value, cell["reduction_percent"], abs_tol=1e-10):
                raise ValueError("Published primary summary differs from measured loss")
            ci = cell["delta_nll_ci95"]
            if ci:
                low, high = 100 * (1 - math.exp(ci[1])), 100 * (1 - math.exp(ci[0]))
                ax.hlines(i, low, high, color=BLUE, linewidth=2)
                ax.plot([low, high], [i, i], "|", color=BLUE)
            ax.plot(value, i, "o", color=BLUE if value > 0 else RED)
            ax.annotate(
                f"{value:+.2f}%",
                (value, i),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set(
            yticks=range(4),
            yticklabels=[f"{c['model']} / {c['corpus']}" for c in cells],
            xlabel="Perplexity reduction vs same-axis RTN (%) — positive is better",
            title="Frozen primary endpoint: clipping transfer at identical stored bytes",
        )
        ax.invert_yaxis()
        fig.supxlabel(
            f"Geometric mean reduction {summary['geometric_mean_reduction_percent']:.3f}%; "
            f"positive cells {summary['positive_primary_cells']}/4; gate {summary['transfer_gate_passed']}\n"
            "51,200 predicted tokens/cell; descriptive document-bootstrap intervals",
            fontsize=9,
        )
        save(fig, "clipping-transfer-primary")

        fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
        arms = ["mse_axis0", "mse_axis1", "act_axis1_seed0", "act_axis1_seed1", "act_axis1_seed2"]
        for ax, cell in zip(axes.flat, cells, strict=True):
            report, corpus = reports[cell["model"]], cell["corpus"]
            reductions = []
            for arm in arms:
                axis = 0 if arm == "mse_axis0" else 1
                baseline = report["arms"][f"rtn_axis{axis}"]["tests"][corpus]["nll"]
                reductions.append(
                    100 * (1 - math.exp(report["arms"][arm]["tests"][corpus]["nll"] - baseline))
                )
            ax.barh(range(5), reductions, color=[ORANGE, BLUE, BLUE, BLUE, BLUE])
            ax.axvline(0, color="#333333", linewidth=0.8)
            ax.set(
                yticks=range(5),
                yticklabels=[
                    "MSE axis0 (primary)",
                    "MSE axis1",
                    "Weighted subset0",
                    "Weighted subset1",
                    "Weighted subset2",
                ],
                title=f"{cell['model']} / {corpus}",
                xlabel="Reduction vs own-axis RTN (%)",
            )
            ax.invert_yaxis()
        fig.suptitle("Grouping and activation-weighted controls — all predeclared arms")
        fig.supxlabel(
            "Diagonal second-moment weighting is not full AWQ; subsets are not training seeds",
            fontsize=9,
        )
        save(fig, "clipping-transfer-controls")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
        for ax, key in zip(axes, ("smol360", "pythia410"), strict=True):
            selected = {row["arm"]: row for row in objective_rows if row["model"] == key}
            x = np.arange(len(arms))
            for offset, field, label, color in (
                (-0.24, "weight_error_reduction_percent", "Squared weight error", BLUE),
                (0, "c4_reserved_ppl_reduction_percent", "C4 perplexity", ORANGE),
                (0.24, "pg19_ppl_reduction_percent", "PG19 perplexity", RED),
            ):
                ax.bar(
                    x + offset,
                    [selected[a][field] for a in arms],
                    width=0.23,
                    label=label,
                    color=color,
                )
            ax.axhline(0, color="#333333", linewidth=0.8)
            ax.set(
                xticks=x,
                xticklabels=[
                    "MSE / axis0",
                    "MSE / axis1",
                    "Weighted / 0",
                    "Weighted / 1",
                    "Weighted / 2",
                ],
                title=key,
                ylabel="Reduction vs same-axis RTN (%)",
            )
            ax.tick_params(axis="x", labelrotation=28)
            ax.legend(fontsize=8)
        fig.suptitle("Lower weight reconstruction error need not improve language-model quality")
        fig.supxlabel(
            "Descriptive analysis of all predeclared controls; no causal or general predictor claim",
            fontsize=9,
        )
        save(fig, "clipping-objective-vs-quality")

        table = [
            "# Clipping objectives and whole-model quality",
            "",
            "Descriptive analysis of saved pre-holdout layer scores and verified model metrics. All reductions compare with each arm's own-axis RTN baseline.",
            "",
            "Weight SSE sums layer MSE times the number of elements. The diagonal proxy sums each layer's per-output mean times its output dimension, using calibration subset0's input second moments. It omits cross-feature covariance and is not measured output MSE or a causal explanation. Calibration subsets are not training seeds.",
            "",
            "| Model | Arm | Weight SSE reduction | Diagonal proxy reduction | Clipped groups | C4 PPL reduction | PG19 PPL reduction |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for row in objective_rows:
            table.append(
                f"| {row['model']} | {row['arm']} | {row['weight_error_reduction_percent']:+.2f}% | {row['diagonal_proxy_reduction_percent']:+.2f}% | {row['clipped_group_percent']:.2f}% | {row['c4_reserved_ppl_reduction_percent']:+.2f}% | {row['pg19_ppl_reduction_percent']:+.2f}% |"
            )
        table_dir = out.parent / "tables"
        table_dir.mkdir(parents=True, exist_ok=True)
        table_path = table_dir / "clipping-proxy-diagnostics.md"
        table_path.write_text("\n".join(table) + "\n", encoding="utf-8", newline="\n")
        outputs.append(
            {
                "path": "../tables/clipping-proxy-diagnostics.md",
                "sha256": hashlib.sha256(table_path.read_bytes()).hexdigest(),
            }
        )

    (out / "research-figures-provenance.json").write_text(
        json.dumps(
            {
                "inputs": inputs,
                "outputs": outputs,
                "scope": "Figures summarize existing verified evidence; no additional experiments or holdout decisions",
                "matplotlib": matplotlib.__version__,
                "numpy": np.__version__,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {"figures": [x["path"] for x in outputs], "transfer_included": transfer.exists()}
        )
    )


if __name__ == "__main__":
    main()
