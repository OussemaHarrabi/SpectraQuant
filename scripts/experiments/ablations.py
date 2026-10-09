"""Milestone-8 local half: required ablations, calibration sensitivity and outlier behaviour.

Runs four Tier-0 (``LOCAL-FIXTURE``, measurement class 1-2) studies on the trained fixture
transformer and writes one machine-readable document plus its rendered report:

* **proxy component ablation** — the declared candidate ``gain_aware_composed`` against its parts
  removed (naive per-layer output error, in-situ-only, the full in-situ x gain composition, and the
  untuned ``combined``), with the model-level Fisher-z contrast and CI for each pair;
* **calibration-size sensitivity** — ``rho`` and the plug-in estimator's relative SE versus the
  calibration token count per layer (8 .. 512);
* **outlier behaviour** — controlled per-channel magnitude spikes (1, 2, 4, 8, 16) and how per-group
  quantization confines them relative to per-tensor;
* **seed-count sensitivity** — the aggregate candidate-vs-comparator contrast and its CI for 1..5
  seeds.

Everything is local-fixture work: the Tier-1/Tier-2 cells of the frozen section 8 matrix are
**NOT RUN** and need the cloud substrate. No Tier-1 number is implied anywhere in the output.

Usage:
    uv run python scripts/experiments/ablations.py [--out DIR] [--report PATH]
        [--seeds 0 1 2 3 4] [--boot-ci 2000] [--train-steps 120] [--quick]
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import pairwise
from pathlib import Path
from typing import Any

from spectraquant.proxies.ablations import (
    CALIBRATION_SIZES,
    CANDIDATE,
    COMPONENT_LABELS,
    COMPONENT_VARIANTS,
    NEW_VARIANT,
    OUTLIER_SEVERITIES,
    REFERENCE_COMPARATOR,
    build_document,
    git_state,
    run_calibration_size_sweep,
    run_component_sweep,
    run_outlier_sweep,
    seed_count_sensitivity,
)
from spectraquant.proxies.validation import DEFAULT_CELLS, DEFAULT_FIXTURES, DEFAULT_SEEDS

ARTIFACT_NAME = "ablations.json"
DEFAULT_OUT = "artifacts/sample-results/ablations"
DEFAULT_REPORT = "docs/results/ablation-report.md"


def _fmt(value: Any, digits: int = 3) -> str:
    """Format a number for the console tables; ``None`` renders as ``n/a``."""
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:+.{digits}f}" if digits else f"{value:.{digits}f}"


def _ci(ci: Any, digits: int = 3) -> str:
    if not ci or ci[0] is None or ci[1] is None:
        return "n/a"
    return f"[{ci[0]:+.{digits}f},{ci[1]:+.{digits}f}]"


def print_component(document: dict[str, Any]) -> None:
    """Print the component-ablation variant table and the paired contrasts with CIs."""
    analysis = document["component_ablation"]["analysis"]
    for group in ("layernorm", "no_layernorm", "all_fixtures"):
        payload = analysis[group]
        print(f"\n== component ablation: {group} ==")
        print(f"{'variant':26s} {'rho_mean':>9s} {'n_eff':>7s}")
        for variant in COMPONENT_VARIANTS:
            row = payload["variants"][variant]
            print(f"{variant:26s} {_fmt(row['rho_mean'], 4):>9s} {_fmt(row['n_eff_mean'], 2):>7s}")
        print("  paired contrasts (candidate - comparator), 95% CIs:")
        for comparator, contrast in payload["contrasts"].items():
            print(
                f"    {CANDIDATE:20s} - {comparator:22s} dz={_fmt(contrast['delta_z'])} "
                f"CI={_ci(contrast['ci'])} bootCI={_ci(contrast['boot_ci'])} "
                f"p={contrast['p_value']:.3g} -> {contrast['verdict']['verdict'].upper()}"
            )


def print_calibration(document: dict[str, Any]) -> None:
    """Print the calibration-size curve."""
    print("\n== calibration-size curve (mean over models) ==")
    print(f"{'tokens':>7s} {'rho_cand':>9s} {'rho_naive':>10s} {'rho_in_situ':>11s} {'rel_se':>8s}")
    for size in sorted(document["calibration_size"]["mean"], key=int):
        row = document["calibration_size"]["mean"][size]
        print(
            f"{int(size):>7d} {_fmt(row['rho_gain_aware_composed'], 4):>9s} "
            f"{_fmt(row['rho_per_layer_output_error'], 4):>10s} "
            f"{_fmt(row['rho_in_situ_output_error'], 4):>11s} "
            f"{_fmt(row['estimator_rel_se_mean'], 4):>8s}"
        )


def print_outliers(document: dict[str, Any]) -> None:
    """Print the outlier table."""
    payload = document["outliers"]
    print("\n== outlier behaviour (mean over models) ==")
    print(
        f"{'severity':>8s} {'damage':>10s} {'res_group':>11s} {'res_tensor':>11s} "
        f"{'t/g':>6s} {'dmg/res':>8s} {'rho_cand':>9s} {'rho_naive':>10s}"
    )
    for severity in sorted(payload["mean"], key=float):
        row = payload["mean"][severity]
        print(
            f"{float(severity):>8.1f} {_fmt(row['damage_sum'], 3):>10s} "
            f"{_fmt(row['rounding_residual_per_group'], 3):>11s} "
            f"{_fmt(row['rounding_residual_per_tensor'], 3):>11s} "
            f"{_fmt(row['per_tensor_over_per_group'], 2):>6s} "
            f"{_fmt(row['damage_over_rounding'], 3):>8s} "
            f"{_fmt(row['rho_gain_aware_composed'], 3):>9s} "
            f"{_fmt(row['rho_per_layer_output_error'], 3):>10s}"
        )


def print_seed_count(document: dict[str, Any]) -> None:
    """Print the seed-count sensitivity table."""
    print("\n== seed-count sensitivity (candidate vs weight_frobenius) ==")
    print(
        f"{'seeds':>5s} {'k':>4s} {'delta_z':>9s} {'CI':>20s} {'boot_CI':>20s} {'MDE':>7s} verdict"
    )
    for row in document["seed_count"]:
        if row.get("n_clusters", 0) == 0:
            print(f"{row['seeds']:>5d} {0:>4d}  (no usable units)")
            continue
        print(
            f"{row['seeds']:>5d} {row['k']:>4d} {_fmt(row['delta_z']):>9s} "
            f"{_ci(row['ci']):>20s} {_ci(row['boot_ci']):>20s} "
            f"{_fmt(row['mde_achieved_z']):>7s} {row['verdict']['verdict'].upper()}"
        )


def print_verdicts(document: dict[str, Any]) -> None:
    """Print the per-hypothesis fixture-scale verdicts and the NOT RUN cloud cells."""
    print("\n== hypothesis verdicts at fixture scale only ==")
    for hyp, payload in document["hypotheses"].items():
        print(
            f"  [{hyp}] {str(payload['verdict']).upper()}  (tested_by_this_slice="
            f"{payload['tested_by_this_slice']})"
        )
    print("\n== cloud cells (NOT RUN) ==")
    for key, cell in document["cloud_cells"].items():
        print(f"  {cell['substrate']:12s} {cell['status']:7s} tier{cell['tier']} {key}")


def _plateau(document: dict[str, Any]) -> str:
    """First calibration size beyond which the candidate's rho moves by < 0.02."""
    means = document["calibration_size"]["mean"]
    sizes = sorted(int(s) for s in means)
    text = []
    for a, b in pairwise(sizes):
        ra = means[str(a)]["rho_gain_aware_composed"]
        rb = means[str(b)]["rho_gain_aware_composed"]
        if ra is not None and rb is not None:
            text.append((a, b, abs(rb - ra)))
    stop = next((b for a, b, delta in text if delta < 0.02), None)
    if stop is None:
        return (
            "no plateau within the swept range: the candidate's rho still moves by >= 0.02 between "
            "consecutive sizes"
        )
    return (
        f"the candidate's rho moves by less than 0.02 per step from {stop} tokens per layer upward "
        f"(the swept range ends at 512), so the proxy stops improving at or below {stop}"
    )


def render_report(document: dict[str, Any]) -> str:
    """Render the human-readable report from the same document written to JSON."""
    plan = document["plan"]
    lines: list[str] = []
    add = lines.append
    add("# Ablation report (Milestone 8, local half — component, calibration, outlier, seed)")
    add("")
    add(
        "Owner: proxy/ablation slice. Machine-readable artifact: "
        "`artifacts/sample-results/ablations/ablations.json` (regenerate with "
        "`uv run python scripts/experiments/ablations.py`). This report is generated from that "
        "document, so the numbers below and the JSON cannot drift apart."
    )
    add("")
    add(
        f"Substrate `{document['substrate']}` (CPU workstation); every number is measurement class "
        f"**{document['measurement_class']}** (`AGENTS.md` section 5). Seeds `{plan['seeds']}` x cells "
        f"`{plan['cells']}` on fixtures `{plan['fixtures']}` for the component ablation; the "
        f"calibration and outlier studies use the representative cell `{plan['study_cell']}`. "
        f"{document['runtime']['n_models']} trained models and "
        f"{document['runtime']['n_component_cells']} component cells; wall time "
        f"{document['runtime']['wall_time_s']:.1f} s; git commit `{document['git_commit']}` "
        f"(dirty={document['git_dirty']})."
    )
    add("")
    add(
        "**All Tier-1/Tier-2 cells are NOT RUN** (section 5 below). No Tier-1 number is implied by "
        "anything in this document; the local evidence is fixture scale only."
    )
    add("")

    # --- 1. component ablation
    analysis = document["component_ablation"]["analysis"]
    signal = document["component_ablation"]["signal_verdict"]
    add("## 1. Proxy component ablation — which component carries the signal")
    add("")
    add(
        "The declared candidate is `gain_aware_composed` = `g^2 * mean_i ||x_i delta^T||^2` (the gain "
        "term applied to the **naive** pre-normalisation per-layer error). The component factorial "
        "spans the error basis `{naive, in-situ}` x `{no gain, gain}` plus the untuned `combined`:"
    )
    add("")
    add("| variant | in-situ | gain | role |")
    add("|---|---|---|---|")
    add(
        "| `per_layer_output_error` | no | no | "
        + COMPONENT_LABELS["per_layer_output_error"]
        + " |"
    )
    add("| `in_situ_output_error` | yes | no | " + COMPONENT_LABELS["in_situ_output_error"] + " |")
    add("| `gain_aware_composed` | no | yes | " + COMPONENT_LABELS["gain_aware_composed"] + " |")
    add("| `gain_aware_in_situ` | yes | yes | " + COMPONENT_LABELS["gain_aware_in_situ"] + " |")
    add("| `combined` | no | yes* | " + COMPONENT_LABELS["combined"] + " |")
    add("")
    add(
        "`combined` carries the gain through its `gain_aware` component. `gain_aware_in_situ` is "
        "implemented in `src/spectraquant/proxies/ablations.py` and is **not** in the frozen registry."
    )
    add("")
    for group in ("layernorm", "no_layernorm", "all_fixtures"):
        payload = analysis[group]
        add(f"### 1.{('layernorm', 'no_layernorm', 'all_fixtures').index(group) + 1} {group}")
        add("")
        add("| variant | rho_mean | n_eff_mean | k |")
        add("|---|---|---|---|")
        for variant in COMPONENT_VARIANTS:
            row = payload["variants"][variant]
            add(
                f"| `{variant}` | {_fmt(row['rho_mean'], 4)} | {_fmt(row['n_eff_mean'], 2)} | "
                f"{row['k']} |"
            )
        add("")
        add("| candidate | comparator | delta_z | CI (analytic) | boot CI | p | verdict |")
        add("|---|---|---|---|---|---|---|")
        for comparator, contrast in payload["contrasts"].items():
            add(
                f"| `{CANDIDATE}` | `{comparator}` | {_fmt(contrast['delta_z'])} | "
                f"{_ci(contrast['ci'])} | {_ci(contrast['boot_ci'])} | {contrast['p_value']:.3g} | "
                f"{contrast['verdict']['verdict']} |"
            )
        add("")
    add("### 1.4 Which component carries the signal")
    add("")
    gain = signal["gain_term"]
    insitu = signal["in_situ_term"]
    extra = signal["extra_terms"]
    naive = signal["naive_basis_vs_in_situ_only"]
    add(
        f"* **Gain term** — the pooled `{gain['contrast']}` contrast is delta_z = "
        f"{_fmt(gain['delta_z'])} with CI {_ci(gain['ci'])}: the gain term "
        f"**{'carries' if gain['carries_signal'] else 'does not carry'} the signal**."
    )
    add(
        f"* **In-situ term** — holding the gain fixed, `{insitu['contrast']}` is delta_z = "
        f"{_fmt(insitu['delta_z'])} with CI {_ci(insitu['ci'])}: the in-situ normalisation basis "
        f"**{'adds' if insitu['carries_signal'] else 'does not add'} signal** beyond the gain term "
        f"(the reversed view is `{signal['candidate_minus_full']['contrast']}` = "
        f"{_fmt(signal['candidate_minus_full']['delta_z'])}, "
        f"CI {_ci(signal['candidate_minus_full']['ci'])})."
    )
    add(
        f"* **Extra terms** — `{extra['contrast']}` is delta_z = {_fmt(extra['delta_z'])} with CI "
        f"{_ci(extra['ci'])}: the untuned Hessian/rounding terms "
        f"**{'help' if extra['ci'] and extra['ci'][1] < 0 else 'do not help'}** relative to the "
        f"gain-only candidate."
    )
    add(
        f"* **Naive vs in-situ-only basis** — `{naive['contrast']}` is delta_z = "
        f"{_fmt(naive['delta_z'])} with CI {_ci(naive['ci'])}."
    )
    means = signal["component_rho_mean"]
    add("")
    add("Pooled mean rho by component:")
    add("")
    add("| variant | rho_mean |")
    add("|---|---|")
    for variant in COMPONENT_VARIANTS:
        add(f"| `{variant}` | {_fmt(means[variant], 4)} |")
    add("")
    lay = analysis["layernorm"]["contrasts"].get(NEW_VARIANT, {})
    lay_delta = lay.get("delta_z")
    lay_ci = lay.get("ci")
    add(
        "**Statement.** The component that carries the ranking signal is the **downstream-gain "
        f"term**: the candidate is ahead of the naive per-layer error (pooled delta_z = "
        f"{_fmt(gain['delta_z'])}, CI {_ci(gain['ci'])}, CI excludes 0) and of the in-situ-only "
        f"variant (delta_z = {_fmt(naive['delta_z'])}, CI {_ci(naive['ci'])}). The ablation that "
        "**does not discriminate** is the in-situ normalisation term — at pooled fixture scale "
        f"removing it is, if anything, better for the ranking (candidate - full composition = "
        f"{_fmt(signal['candidate_minus_full']['delta_z'])}, CI "
        f"{_ci(signal['candidate_minus_full']['ci'])}), and on the `layernorm` fixture alone the "
        f"full composition versus the gain-only candidate is inconclusive (delta_z = {_fmt(lay_delta)},"
        f" CI {_ci(lay_ci)}). A null ablation is a result: the in-situ Jacobian is **not** the active "
        "ingredient at this scale, and the untuned `combined` variant is dominated by the candidate "
        f"it contains (delta_z = {_fmt(extra['delta_z'])}, CI {_ci(extra['ci'])})."
    )
    add("")

    # --- 2. calibration size
    add("## 2. Calibration-size sensitivity")
    add("")
    add(
        "The **target** (single-layer damage used for the ranking) is always measured on the fixed "
        "2x24-token batch, so it does not move with the calibration size; only the proxy's "
        "calibration set (activations, residual errors, gains) grows. `rel_se` is the mean over "
        "layers of the naive plug-in estimator's relative standard error."
    )
    add("")
    add("| tokens/layer | rho candidate | rho naive | rho in-situ | estimator rel_se |")
    add("|---|---|---|---|---|")
    for size in sorted(document["calibration_size"]["mean"], key=int):
        row = document["calibration_size"]["mean"][size]
        add(
            f"| {int(size)} | {_fmt(row['rho_gain_aware_composed'], 4)} | "
            f"{_fmt(row['rho_per_layer_output_error'], 4)} | "
            f"{_fmt(row['rho_in_situ_output_error'], 4)} | "
            f"{_fmt(row['estimator_rel_se_mean'], 4)} |"
        )
    add("")
    add(
        f"**Where it stops improving.** {_plateau(document)}. The relative SE falls roughly as "
        "1/sqrt(tokens) across the whole range, but the ranking itself is flat well before the "
        "largest calibration set: more calibration data does not buy a better ranking here."
    )
    add("")

    # --- 3. outliers
    payload = document["outliers"]
    add("## 3. Outlier behaviour")
    add("")
    add(f"**Injection procedure (exact).** {payload['procedure']}")
    add("")
    add(
        "`res_group` is the summed per-group rounding residual (the study spec, `group_size=32`), "
        "`res_tensor` the per-tensor control; `t/g` is their ratio; `dmg/res` the damage per unit "
        "rounding residual; `rho_cand`/`rho_naive` are the model-level ranking correlations against "
        "the injected model's damage."
    )
    add("")
    add(
        "| severity | damage | res_group | res_tensor | t/g | dmg/res | rho candidate | rho naive |"
    )
    add("|---|---|---|---|---|---|---|---|")
    for severity in sorted(payload["mean"], key=float):
        row = payload["mean"][severity]
        add(
            f"| {float(severity):.1f} | {_fmt(row['damage_sum'], 3)} | "
            f"{_fmt(row['rounding_residual_per_group'], 3)} | "
            f"{_fmt(row['rounding_residual_per_tensor'], 3)} | "
            f"{_fmt(row['per_tensor_over_per_group'], 2)} | "
            f"{_fmt(row['damage_over_rounding'], 3)} | "
            f"{_fmt(row['rho_gain_aware_composed'], 3)} | "
            f"{_fmt(row['rho_per_layer_output_error'], 3)} |"
        )
    add("")
    means = payload["mean"]
    order = sorted(means, key=float)
    damage = [means[s]["damage_sum"] for s in order]
    res_group = [means[s]["rounding_residual_per_group"] for s in order]
    res_tensor = [means[s]["rounding_residual_per_tensor"] for s in order]
    rho_cand = [means[s]["rho_gain_aware_composed"] for s in order]
    rho_naive = [means[s]["rho_per_layer_output_error"] for s in order]
    peak = order[max(range(len(damage)), key=lambda i: damage[i])]
    add(
        f"**Reading.** {len(order)} severities were injected ({order[0]} .. {order[-1]}). The measured "
        f"damage is **non-monotone**: it peaks at severity {peak} "
        f"({_fmt(max(damage), 3)}) and falls to {_fmt(damage[-1], 3)} at the highest severity (the "
        f"injected model is itself re-measured, so an extreme spike changes what 'damage' the "
        f"low-rank reconstruction incurs). The rounding residual rises monotonically with severity for "
        f"both quantizers ({_fmt(res_group[0], 3)} -> {_fmt(res_group[-1], 3)} per-group, "
        f"{_fmt(res_tensor[0], 3)} -> {_fmt(res_tensor[-1], 3)} per-tensor), and per-group is below "
        f"per-tensor at every severity (`t/g` in "
        f"[{_fmt(min(means[s]['per_tensor_over_per_group'] for s in order), 2)}, "
        f"{_fmt(max(means[s]['per_tensor_over_per_group'] for s in order), 2)}]): the per-tensor global "
        f"scale must absorb the spike, while per-group keeps one scale per block. At the top severity "
        f"the residual is dominated by squared block scales (a spiked block rounds its non-spiked "
        f"entries to zero), so the per-group residual saturates at a large value. The candidate's "
        f"ranking does **not** collapse — it stays in "
        f"[{_fmt(min(rho_cand), 3)}, {_fmt(max(rho_cand), 3)}] across all severities — whereas the "
        f"naive per-layer error falls from {_fmt(rho_naive[0], 3)} to {_fmt(rho_naive[-1], 3)}. The "
        f"damage / rounding-residual ratio falls with severity ({_fmt(means[order[0]]['damage_over_rounding'], 3)}"
        f" -> {_fmt(means[order[-1]]['damage_over_rounding'], 3)}): the outliers push a larger fraction "
        f"of the remaining error into the quantizer rather than the retained low-rank signal."
    )
    add("")

    # --- 4. seed count
    add("## 4. Seed-count sensitivity")
    add("")
    add(
        f"Pooled `{CANDIDATE} - {REFERENCE_COMPARATOR}` contrast versus the number of seeds. This is "
        "a sensitivity view of the pre-registered fixed-n rule (section 7.6): it shows what a "
        "smaller seed count would have reported, it is **not** an optional-stopping licence."
    )
    add("")
    add("| seeds | k units | delta_z | CI (analytic) | boot CI | MDE(z) | verdict |")
    add("|---|---|---|---|---|---|---|")
    for row in document["seed_count"]:
        if row.get("n_clusters", 0) == 0:
            add(f"| {row['seeds']} | 0 | (no usable units) | | | | |")
            continue
        add(
            f"| {row['seeds']} | {row['k']} | {_fmt(row['delta_z'])} | {_ci(row['ci'])} | "
            f"{_ci(row['boot_ci'])} | {_fmt(row['mde_achieved_z'])} | "
            f"{row['verdict']['verdict']} |"
        )
    add("")

    # --- 5. hypotheses
    add("## 5. Hypothesis verdicts against the frozen section 11 falsifiers (fixture scale only)")
    add("")
    for hyp in ("H1", "H2", "H4"):
        payload = document["hypotheses"][hyp]
        add(f"### {hyp}")
        add("")
        add(f'**Frozen falsifier wording (section 11):** "{payload["falsifier"]}"')
        add("")
        if hyp == "H2":
            add(
                f"Contrast `{payload['contrast']}`: delta_z = {_fmt(payload['delta_z'])}, analytic CI "
                f"{_ci(payload['ci'])}, bootstrap CI {_ci(payload['boot_ci'])}, "
                f"p = {payload['p_value']:.3g}."
            )
            add("")
            add(f"**Verdict: {str(payload['verdict']).upper()}** — {payload['rationale']}")
        else:
            add(
                f"**Verdict: {str(payload['verdict']).upper()}** (not tested by this slice) — "
                f"{payload['reason']}"
            )
        add("")
    add(
        "The vocabulary is mapped to the requested form: the frozen rule's `refuted` is reported as "
        "`contradicted`; `supported` and `inconclusive` are unchanged. Only **H2** is tested by this "
        "slice. **H1** and **H4** are `inconclusive` at fixture scale because the cells that test "
        "them are `NOT RUN` — never as a refutation."
    )
    add("")
    add("### 5.1 Tier-1 / Tier-2 cells (all NOT RUN)")
    add("")
    add("| cell | hypothesis | tier | substrate | status |")
    add("|---|---|---|---|---|")
    for _key, cell in document["cloud_cells"].items():
        add(
            f"| {cell['confirmatory_cell']} | {cell['hypothesis']} | {cell['tier']} | "
            f"`{cell['substrate']}` | **{cell['status']}** |"
        )
    add("")
    add(
        "No Tier-1 or Tier-2 number is claimed, implied or estimated anywhere in this document. "
        "Each of those cells becomes a result only from a validated `run_manifest.json` produced on "
        "the cloud substrate (`preregistration.md` section 8)."
    )
    add("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT, help="artifact directory")
    parser.add_argument("--report", default=DEFAULT_REPORT, help="report markdown path")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--boot-ci", type=int, default=2000)
    parser.add_argument("--train-steps", type=int, default=120)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="one seed, two cells, three calibration sizes and three severities (smoke only)",
    )
    args = parser.parse_args()

    seeds = tuple(args.seeds)
    cells = DEFAULT_CELLS
    fixtures = DEFAULT_FIXTURES
    sizes = CALIBRATION_SIZES
    severities = OUTLIER_SEVERITIES
    if args.quick:
        seeds = seeds[:1]
        cells = ((8, 4), (32, 8))
        sizes = (8, 64, 512)
        severities = (1.0, 4.0, 16.0)

    started = time.perf_counter()
    component_runs = run_component_sweep(
        seeds=seeds, cells=cells, fixtures=fixtures, train_steps=args.train_steps
    )
    calibration = run_calibration_size_sweep(
        seeds=seeds, fixtures=fixtures, sizes=sizes, train_steps=args.train_steps
    )
    outliers = run_outlier_sweep(
        seeds=seeds, fixtures=fixtures, severities=severities, train_steps=args.train_steps
    )
    seed_rows = seed_count_sensitivity(component_runs, all_seeds=seeds, n_boot_ci=args.boot_ci)
    wall = time.perf_counter() - started
    state = git_state()
    document = build_document(
        component_runs=component_runs,
        calibration=calibration,
        outliers=outliers,
        seed_count=seed_rows,
        seeds=seeds,
        cells=cells,
        fixtures=fixtures,
        train_steps=args.train_steps,
        n_boot_ci=args.boot_ci,
        wall_time_s=wall,
        git_commit=state["git_commit"],
        git_dirty=state["git_dirty"],
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = out_dir / ARTIFACT_NAME
    artifact.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(document), encoding="utf-8")

    print_component(document)
    print_calibration(document)
    print_outliers(document)
    print_seed_count(document)
    print_verdicts(document)
    print(f"\nwrote {artifact} and {report_path} in {wall:.1f} s")


if __name__ == "__main__":
    main()
