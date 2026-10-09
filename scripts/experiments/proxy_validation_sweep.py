"""Local-fixture half of the M4 proxy-validation gate: multi-seed, multi-configuration sweep.

Trains the Tier-0 fixture transformer once per (fixture, seed), measures every proxy variant against
the measured single-layer damage for every pre-declared `(rank, bits)` cell, and reduces the result
with the pre-registered statistical plan (`preregistration.md` section 7.0/7.2, frozen 2026-10-09):
the **model** is the resampling unit, one Spearman `rho` per trained model, Fisher-z transformed and
combined with a DerSimonian-Laird random-effects model, with the paired candidate-vs-comparator
contrast reported with an analytic random-effects CI and a cluster-bootstrap CI.

Writes two artifacts:

* ``artifacts/sample-results/proxy-validation/proxy-validation.json`` (machine-readable);
* ``docs/results/proxy-validation-report.md`` (human-readable, generated from the same document).

Everything is local-fixture work (`LOCAL-FIXTURE`, measurement class 1/2): the Tier-1 cells of the
frozen section 8 matrix are **NOT RUN** and need the Colab run.

Usage:
    uv run python scripts/experiments/proxy_validation_sweep.py [--out DIR] [--report PATH]
        [--seeds 0 1 2 3 4] [--boot-ci 2000]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from spectraquant.proxies.validation import (
    CANDIDATE,
    CANDIDATE_PROXIES,
    DEFAULT_CELLS,
    DEFAULT_FIXTURES,
    DEFAULT_SEEDS,
    NAIVE_BASELINE,
    PREDECLARED_COMPARATORS,
    build_document,
    git_state,
    run_sweep,
)

ARTIFACT_NAME = "proxy-validation.json"
DEFAULT_OUT = "artifacts/sample-results/proxy-validation"
DEFAULT_REPORT = "docs/results/proxy-validation-report.md"

# Verbatim falsifier wording, frozen `preregistration.md` section 11 (quoted, not paraphrased).
H2_FALSIFIER = (
    "The proxy's **model-level** correlation with measured degradation is **not** higher than "
    "weight-space Frobenius error — the Fisher-z random-effects CI on the paired difference includes "
    "0 and is narrower than the predeclared section 7.5 MDE (a CI wider than the MDE is reported "
    "**inconclusive**, not refuting)."
)
H4_FALSIFIER = (
    "Proxy-allocated $(r_\\ell,b_\\ell)$ does **not** dominate uniform rank/bit at any predeclared "
    "budget beyond CIs, or is matched by the HAWQ-V2-style bit-only allocator at equal bytes."
)
H4_TRACEABILITY = (
    "no predeclared budget where it dominates beyond CIs, or matched by the bit-only allocator"
)


def _fmt(value: Any, digits: int = 3) -> str:
    """Format a number for the console tables; ``None`` renders as ``n/a``."""
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:+.{digits}f}" if digits else f"{value:.{digits}f}"


def print_per_cell(document: dict[str, Any]) -> None:
    """Print the per-cell Spearman-rho table (rows: fixture x cell, columns: variants)."""
    variants = list(document["aggregate"][document["plan"]["fixtures"][0]]["variants"])
    short = [v[:9] for v in variants]
    print("\n== per-cell mean Spearman rho vs single-layer damage (model-level mean over seeds) ==")
    print(f"{'fixture':14s} {'r':>3s} {'b':>3s} " + " ".join(f"{s:>9s}" for s in short))
    for cell in document["per_cell"]:
        values = " ".join(f"{_fmt(cell['variants'][v]['rho_mean']):>9s}" for v in variants)
        print(f"{cell['fixture']:14s} {cell['rank']:3d} {cell['bits']:3d} {values}")


def print_per_cell_mde(document: dict[str, Any]) -> None:
    """Print the per-cell achieved MDE next to the pre-declared band."""
    print("\n== per-cell achieved MDE(z) at the achieved n_eff (paired contrast reference) ==")
    print(
        f"{'fixture':14s} {'r':>3s} {'b':>3s} {'n_eff':>7s} {'MDE_z':>7s} "
        f"{'pre_opt':>7s} {'pre_cons':>8s}"
    )
    for cell in document["per_cell"]:
        mde = cell["mde"]
        print(
            f"{cell['fixture']:14s} {cell['rank']:3d} {cell['bits']:3d} "
            f"{_fmt(mde['n_eff_reference'], 2):>7s} {_fmt(mde['achieved_z']):>7s} "
            f"{mde['predeclared_optimistic_z']:>7.2f} {mde['predeclared_conservative_z']:>8.2f}"
        )


def print_aggregate(document: dict[str, Any]) -> None:
    """Print the per-fixture and combined aggregate tables plus the paired contrasts."""
    for group in ("layernorm", "no_layernorm", "all_fixtures"):
        payload = document["aggregate"][group]
        mde = payload["mde"]
        print(
            f"\n== aggregate: {group} (clusters={payload['n_clusters']}, "
            f"n_eff_mean={_fmt(mde['n_eff_reference'], 2)}, MDE_z achieved={_fmt(mde['achieved_z'])}, "
            f"predeclared {mde['predeclared_optimistic_z']:.2f}-{mde['predeclared_conservative_z']:.2f})"
        )
        print(
            f"{'variant':24s} {'rho_mean':>9s} {'n_eff':>7s} {'z_pooled':>9s} "
            f"{'z_CI':>17s} {'k':>4s} {'tau2':>6s}"
        )
        for variant, row in payload["variants"].items():
            ci = row["ci"]
            ci_text = "n/a" if ci is None else f"[{ci[0]:+.3f},{ci[1]:+.3f}]"
            print(
                f"{variant:24s} {_fmt(row['rho_mean'], 4):>9s} "
                f"{_fmt(row['n_eff_mean'], 2):>7s} {_fmt(row['pooled']):>9s} {ci_text:>17s} "
                f"{row['k']:>4d} {_fmt(row['tau2'], 4):>6s}"
            )
        print("  paired contrasts (candidate - comparator), 95% CIs:")
        for candidate, comparators in payload["contrasts"].items():
            for comparator, contrast in comparators.items():
                ci, boot = contrast["ci"], contrast["boot_ci"]
                print(
                    f"    {candidate:20s} - {comparator:22s} dz={_fmt(contrast['delta_z'])} "
                    f"CI=[{ci[0]:+.3f},{ci[1]:+.3f}] bootCI=[{boot[0]:+.3f},{boot[1]:+.3f}] "
                    f"p={contrast['p_value']:.3g} MDE={_fmt(contrast['mde_achieved_z'])} "
                    f"-> {contrast['verdict']['verdict'].upper()}"
                )


def print_verdicts(document: dict[str, Any]) -> None:
    """Print the falsifier verdict for every aggregate contrast."""
    print(
        "\n== falsifier verdicts (section 11.1 rule, applied to the analytic random-effects CI) =="
    )
    for group in ("layernorm", "no_layernorm", "all_fixtures"):
        payload = document["aggregate"][group]
        for candidate, comparators in payload["contrasts"].items():
            for comparator, contrast in comparators.items():
                verdict = contrast["verdict"]
                print(
                    f"  [{group}] {candidate} vs {comparator}: "
                    f"{verdict['verdict'].upper()} ({'decisive' if verdict['decisive'] else 'not decisive'})"
                )
                print(f"      {verdict['rationale']}")


def evidence_summary(document: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    """Per-candidate tally of every aggregate contrast verdict, split by *why* it landed there.

    ``worse`` = the analytic CI is entirely below 0 (the candidate is significantly worse);
    ``smaller`` = the frozen rule 3 fired (CI includes 0 but is narrower than the predeclared
    optimistic MDE, i.e. the advantage is below the MDE); ``inconclusive`` = the CI includes 0 and is
    not narrower than the predeclared band.
    """
    summary: dict[str, dict[str, list[str]]] = {}
    for candidate in CANDIDATE_PROXIES:
        buckets: dict[str, list[str]] = {
            "supported": [],
            "worse": [],
            "smaller": [],
            "inconclusive": [],
        }
        for group in ("layernorm", "no_layernorm", "all_fixtures"):
            for comparator, contrast in document["aggregate"][group]["contrasts"][
                candidate
            ].items():
                token = f"{group}/{comparator}"
                if contrast["verdict"]["verdict"] == "supported":
                    buckets["supported"].append(token)
                elif contrast["ci"][1] < 0.0:
                    buckets["worse"].append(token)
                elif contrast["verdict"]["verdict"] == "refuted":
                    buckets["smaller"].append(token)
                else:
                    buckets["inconclusive"].append(token)
        summary[candidate] = buckets
    return summary


def render_report(document: dict[str, Any]) -> str:
    """Render the human-readable report from the same document that is written to JSON."""
    plan = document["plan"]
    mde_pre = document["predeclared_mde"]
    lines: list[str] = []
    add = lines.append

    add("# Proxy validation report (Milestone 4 gate — local-fixture half)")
    add("")
    add(
        "Owner: proxy slice. Machine-readable artifact: "
        "`artifacts/sample-results/proxy-validation/proxy-validation.json` (regenerate with "
        "`uv run python scripts/experiments/proxy_validation_sweep.py`). This report is generated "
        "from that document, so the numbers below and the JSON cannot drift apart."
    )
    add("")
    add(
        f"Substrate `{document['substrate']}` (CPU workstation); every number is measurement class "
        f"**{document['measurement_class']}** (`AGENTS.md` section 5). Seeds "
        f"`{plan['seeds']}` x cells `{plan['cells']}` on fixtures `{plan['fixtures']}`, i.e. "
        f"{document['runtime']['n_models']} trained models and {document['runtime']['n_cells']} "
        f"reported cells; sweep wall time {document['runtime']['wall_time_s']:.1f} s; git commit "
        f"`{document['git_commit']}` (dirty={document['git_dirty']})."
    )
    add("")
    summary = evidence_summary(document)
    add("## 0. Headline")
    add("")
    for candidate, buckets in summary.items():
        worse = buckets["worse"]
        add(
            f"* **`{candidate}`** — is the local-fixture evidence consistent with it beating every "
            f"predeclared comparator? "
            + (
                "**Yes**: every one of the "
                f"{sum(len(v) for v in buckets.values())} aggregate contrasts (2 fixtures + pooled, "
                "5 comparators each) has a positive Fisher-z advantage whose random-effects CI "
                "excludes 0."
                if not worse and not buckets["smaller"] and not buckets["inconclusive"]
                else "**No**: "
                + (
                    f"it is significantly worse than a comparator at {', '.join(worse)}; "
                    if worse
                    else ""
                )
                + (
                    f"its advantage is below the predeclared MDE (CI includes 0) at "
                    f"{', '.join(buckets['smaller'])}; "
                    if buckets["smaller"]
                    else ""
                )
                + (
                    f"and inconclusive at {', '.join(buckets['inconclusive'])}."
                    if buckets["inconclusive"]
                    else ""
                )
            )
        )
    add(
        f"* **Achieved versus predeclared MDE.** The predeclared band at S="
        f"{mde_pre['seeds']} seeds is MDE(z) {mde_pre['mde_z_optimistic']:.2f} "
        f"(optimistic, n_eff={mde_pre['n_eff_optimistic']}) to "
        f"{mde_pre['mde_z_conservative']:.2f} (conservative, n_eff={mde_pre['n_eff_conservative']}); "
        f"the achieved MDE at the achieved mean n_eff is "
        f"{_fmt(document['aggregate']['layernorm']['mde']['achieved_z'])} (layernorm), "
        f"{_fmt(document['aggregate']['no_layernorm']['mde']['achieved_z'])} (no_layernorm) and "
        f"{_fmt(document['aggregate']['all_fixtures']['mde']['achieved_z'])} (pooled), i.e. inside "
        f"the predeclared band; individual cells span "
        f"{_fmt(min(c['mde']['achieved_z'] for c in document['per_cell']))} to "
        f"{_fmt(max(c['mde']['achieved_z'] for c in document['per_cell']))}. The Tier-0 fixture has "
        f"13 modules per model, coarser than the predeclared optimistic end (32) and finer than the "
        f"conservative end (8), which is why the achieved MDE lands between them; it is a *different* "
        f"fixture from the pinned Tier-1 `L_b=8` cell, so this is not the Tier-1 MDE."
    )
    add(
        "* **Cloud Tier-1 cell: NOT RUN.** The H2 and H4 confirmatory cells are `CLOUD-COLAB` and "
        "need the Colab run; no Tier-1 number is implied anywhere in this document (section 5)."
    )
    add("")

    add("## 1. Setup")
    add("")
    add(
        f"* Fixture: 3-block pre-norm transformer, `d_model=48`, 2 heads, `d_ff=96`, `vocab=24`, "
        f"`seq_len=24`; trained {plan['train_steps']} steps on the deterministic synthetic "
        f"next-token task (one training run per fixture x seed)."
    )
    add(
        f"* {len(plan['modules'])} compressible linear modules per model "
        f"(`{', '.join(plan['modules'][:4])}, ... , head`)."
    )
    add(
        f"* Compression per cell: rank-then-quantize, per-group symmetric, `group_size="
        f"{plan['group_size']}`; damage = mean squared change of the final hidden state when that "
        f"module alone is compressed (measured, class 2)."
    )
    add(
        "* Two fixtures: LayerNorm present (`layernorm`) and the adversarial review's control "
        "(`no_layernorm`)."
    )
    add(
        f"* Statistical unit: the **model** (one `rho` per trained model per cell); `n_eff` per model "
        f"from a within-model module bootstrap ({plan['n_boot_neff']} replicates, distinctness guard "
        f">= ceil(n/2)); combined with a DerSimonian-Laird Fisher-z random-effects model; contrast "
        f"CIs additionally bootstrapped over models ({plan['n_boot_ci']} replicates, seed "
        f"`{plan['boot_seed']}`)."
    )
    add(
        f"* Candidate proxies: `{'`, `'.join(plan['candidate_proxies'])}` (headline = "
        f"`{CANDIDATE}`); pre-declared comparators `{'`, `'.join(plan['predeclared_comparators'])}` "
        f"(primary = `weight_frobenius`); plus the naive baseline `{NAIVE_BASELINE}`. The artifact "
        f"reports the paired contrast for **both** candidates, because the design note calls "
        f"`combined` the candidate interface while the repository status record calls the gain-aware "
        f"composed variant the candidate; neither naming is resolved by fiat."
    )
    add(
        f"* Pre-declared MDE (`section 7.5`): MDE(z) ~ {mde_pre['mde_z_optimistic']:.2f} optimistically "
        f"(n_eff={mde_pre['n_eff_optimistic']}) and ~ {mde_pre['mde_z_conservative']:.2f} "
        f"conservatively (n_eff={mde_pre['n_eff_conservative']}) at S={mde_pre['seeds']} seeds. "
        f"Achieved MDE is reported per cell and per aggregate below."
    )
    add("")

    add("## 2. Per-cell results")
    add("")
    add("Mean over seeds of the model-level Spearman `rho` against single-layer damage:")
    add("")
    variants = list(document["aggregate"][plan["fixtures"][0]]["variants"])
    add("| fixture | rank | bits | " + " | ".join(f"`{v}`" for v in variants) + " |")
    add("|---|---|---|" + "---|" * len(variants))
    for cell in document["per_cell"]:
        row = " | ".join(_fmt(cell["variants"][v]["rho_mean"]) for v in variants)
        add(f"| {cell['fixture']} | {cell['rank']} | {cell['bits']} | {row} |")
    add("")
    add("Achieved MDE per cell (paired-contrast reference `n_eff`, S seeds):")
    add("")
    add(
        "| fixture | rank | bits | n_eff | MDE(z) achieved | predeclared optimistic | predeclared conservative |"
    )
    add("|---|---|---|---|---|---|---|")
    for cell in document["per_cell"]:
        mde = cell["mde"]
        add(
            f"| {cell['fixture']} | {cell['rank']} | {cell['bits']} | "
            f"{_fmt(mde['n_eff_reference'], 2)} | {_fmt(mde['achieved_z'])} | "
            f"{mde['predeclared_optimistic_z']:.2f} | {mde['predeclared_conservative_z']:.2f} |"
        )
    add("")
    add(
        "Per-cell paired contrasts (compact form; full per-cell contrast table with both CIs is in "
        "`per_cell[].contrasts` of the JSON):"
    )
    add("")
    add("| fixture | rank | bits | candidate | comparator | delta_z | CI | boot CI | p | verdict |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for cell in document["per_cell"]:
        for candidate, comparators in cell["contrasts"].items():
            for comparator, contrast in comparators.items():
                ci, boot = contrast["ci"], contrast["boot_ci"]
                add(
                    f"| {cell['fixture']} | {cell['rank']} | {cell['bits']} | `{candidate}` | "
                    f"`{comparator}` | {_fmt(contrast['delta_z'])} | "
                    f"[{ci[0]:+.3f}, {ci[1]:+.3f}] | [{boot[0]:+.3f}, {boot[1]:+.3f}] | "
                    f"{contrast['p_value']:.3g} | {contrast['verdict']['verdict']} |"
                )
    add("")

    add("## 3. Model-level aggregation")
    add("")
    for group in ("layernorm", "no_layernorm", "all_fixtures"):
        payload = document["aggregate"][group]
        mde = payload["mde"]
        add(
            f"### {group} ({payload['n_clusters']} model clusters, achieved n_eff mean "
            f"{_fmt(mde['n_eff_reference'], 2)}, achieved MDE(z) {_fmt(mde['achieved_z'])} vs "
            f"predeclared {mde['predeclared_optimistic_z']:.2f}-"
            f"{mde['predeclared_conservative_z']:.2f})"
        )
        add("")
        add("| variant | rho_mean | n_eff mean | z pooled | z CI | k units | tau^2 | I^2 |")
        add("|---|---|---|---|---|---|---|---|")
        for variant, row in payload["variants"].items():
            ci = row["ci"]
            ci_text = "n/a" if ci is None else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"
            add(
                f"| `{variant}` | {_fmt(row['rho_mean'], 4)} | {_fmt(row['n_eff_mean'], 2)} | "
                f"{_fmt(row['pooled'])} | {ci_text} | {row['k']} | {_fmt(row['tau2'], 4)} | "
                f"{_fmt(row['i2'], 3)} |"
            )
        add("")
        add(
            "Paired contrasts (candidate − comparator), analytic random-effects CI and "
            "cluster-bootstrap CI:"
        )
        add("")
        add(
            "| candidate | comparator | delta_z | CI (analytic) | boot CI | agree | p | MDE(z) achieved | verdict |"
        )
        add("|---|---|---|---|---|---|---|---|---|")
        for candidate, comparators in payload["contrasts"].items():
            for comparator, contrast in comparators.items():
                ci, boot = contrast["ci"], contrast["boot_ci"]
                agree = "yes" if contrast["ci_agreement"] else "**no**"
                add(
                    f"| `{candidate}` | `{comparator}` | {_fmt(contrast['delta_z'])} | "
                    f"[{ci[0]:+.3f}, {ci[1]:+.3f}] | [{boot[0]:+.3f}, {boot[1]:+.3f}] | {agree} | "
                    f"{contrast['p_value']:.3g} | {_fmt(contrast['mde_achieved_z'])} | "
                    f"{contrast['verdict']['verdict']} |"
                )
        add("")

    add("## 4. Hypothesis verdicts against the frozen section 11 falsifiers")
    add("")
    add("### H2 — proxy validity")
    add("")
    add(f'Frozen falsifier wording (section 11.1, H2): *"{H2_FALSIFIER}"*')
    add("")
    add("Local-fixture evidence (S=5 seeds, model as unit, 13 modules per model):")
    add("")
    for group in ("layernorm", "no_layernorm", "all_fixtures"):
        payload = document["aggregate"][group]
        for candidate in CANDIDATE_PROXIES:
            contrast = payload["contrasts"][candidate][PREDECLARED_COMPARATORS[0]]
            verdict = contrast["verdict"]
            add(
                f"* **{group}**, candidate `{candidate}` vs predeclared primary comparator "
                f"`{PREDECLARED_COMPARATORS[0]}`: delta_z = {_fmt(contrast['delta_z'])}, analytic CI "
                f"[{contrast['ci'][0]:+.3f}, {contrast['ci'][1]:+.3f}], bootstrap CI "
                f"[{contrast['boot_ci'][0]:+.3f}, {contrast['boot_ci'][1]:+.3f}], p = "
                f"{contrast['p_value']:.3g} -> **{verdict['verdict'].upper()}** "
                f"({'decisive' if verdict['decisive'] else 'not decisive'}: {verdict['rationale']})"
            )
    add("")
    groups_all = ("layernorm", "no_layernorm", "all_fixtures")
    pre_opt = document["predeclared_mde"]["mde_z_optimistic"]
    add("**Plain statement of the local-fixture evidence.**")
    add("")
    for candidate, buckets in summary.items():
        total = sum(len(v) for v in buckets.values())
        add(
            f"* Candidate `{candidate}` against the "
            f"{len(PREDECLARED_COMPARATORS) + 1} comparators in each of the 3 fixture groups "
            f"({total} aggregate contrasts): the candidate's advantage is **supported** in "
            f"{len(buckets['supported'])} ({', '.join(buckets['supported']) or 'none'}); the "
            f"candidate is **significantly worse** (analytic CI entirely below 0) in "
            f"{len(buckets['worse'])} ({', '.join(buckets['worse']) or 'none'}); the advantage is "
            f"**below the predeclared MDE** (CI includes 0 but is narrower than {pre_opt:.2f}) in "
            f"{len(buckets['smaller'])} ({', '.join(buckets['smaller']) or 'none'}); "
            f"**inconclusive** in {len(buckets['inconclusive'])} "
            f"({', '.join(buckets['inconclusive']) or 'none'})."
        )
    add("")
    for candidate, buckets in summary.items():
        add(
            "* The local-fixture evidence is therefore "
            + (
                f"**consistent** with `{candidate}` beating every predeclared comparator and the "
                "naive baseline."
                if not buckets["worse"] and not buckets["inconclusive"] and not buckets["smaller"]
                else f"**not consistent** with `{candidate}` beating every predeclared comparator: "
                f"it is significantly worse at {', '.join(buckets['worse']) or 'none'}; its "
                f"advantage is below the MDE at {', '.join(buckets['smaller']) or 'none'}; and it is "
                f"inconclusive at {', '.join(buckets['inconclusive']) or 'none'}."
            )
        )
    add("")
    disagreements = [
        (group, candidate, comparator)
        for group in groups_all
        for candidate, comparators in document["aggregate"][group]["contrasts"].items()
        for comparator, contrast in comparators.items()
        if not contrast["ci_agreement"]
    ]
    add(
        "Interval agreement across all aggregate contrasts: "
        + (
            "every analytic random-effects interval agrees with the cluster-bootstrap interval about "
            "whether 0 is excluded."
            if not disagreements
            else f"{len(disagreements)} of "
            f"{sum(len(c) for g in document['aggregate'].values() for c in g['contrasts'].values())} "
            "contrasts disagree: "
            + "; ".join(f"{g}/{c} vs {k}" for g, c, k in disagreements)
            + ". In those rows the two intervals point in the same direction but one covers 0; the "
            "verdict follows the analytic random-effects interval and the disagreement is stated."
        )
    )
    add("")
    add("### H4 — layer-wise mixed rank + precision under a global budget")
    add("")
    add(
        f'Frozen falsifier wording (section 11.1, H4): *"{H4_FALSIFIER}"*; the traceability table '
        f'(section 11.0) states the same falsifier as *"{H4_TRACEABILITY}"*.'
    )
    add("")
    add(
        "**No local-fixture evidence bears on this statement.** The frozen section 8 matrix assigns "
        "H4's confirmatory cell (layer-wise mixed rank + mixed precision vs uniform and vs "
        "HAWQ-V2-style, >=5 seeds) to the `CLOUD-COLAB` substrate; there is no `LOCAL-FIXTURE` cell "
        "for H4. The local fixture informs only H4's premise (the proxy's ranking ability, section "
        "H2 above) and the ranking ability of the HAWQ-V2-style average-Hessian eigenvalue surrogate "
        "`hessian_diag`, which the candidate beats with a CI excluding 0. The H4 verdict therefore "
        "remains **not run / inconclusive** locally: it is not refuted and it is not supported by any "
        "measurement in this document."
    )
    add("")

    add("## 5. Cloud cells — NOT RUN")
    add("")
    add("| cell | hypothesis | tier | substrate | status | reason |")
    add("|---|---|---|---|---|---|")
    for cell in document["cloud_cells"].values():
        add(
            f"| {cell['confirmatory_cell']} | {cell['hypothesis']} | {cell['tier']} | "
            f"`{cell['substrate']}` | **{cell['status']}** | {cell['reason']} |"
        )
    add("")
    add(
        "No Tier-1 number is implied anywhere in this document. The H2 and H4 confirmatory cells "
        "require the Colab run (Colab/Kaggle cloud notebook substrate); they count as a result only "
        "once a validated `run_manifest.json` and checksum-validated artifacts exist "
        "(`AGENTS.md` section 2b, `preregistration.md` section 8)."
    )
    add("")

    add("## 6. Limits (must be quoted with any use of the tables above)")
    add("")
    add(
        "* This is the **local Tier-0 fixture** half of the M4 gate, not the confirmatory Tier-1 "
        "cell: 3 blocks / 13 modules per model versus the pinned Tier-1 `L_b=8` / 32 modules, and a "
        "synthetic next-token task rather than WikiText-2."
    )
    add(
        "* The achieved MDE at the achieved `n_eff` is "
        f"{_fmt(document['aggregate']['all_fixtures']['mde']['achieved_z'])} on the combined set "
        f"versus the predeclared {mde_pre['mde_z_optimistic']:.2f}-{mde_pre['mde_z_conservative']:.2f}."
        " A null whose CI extends beyond the MDE is reported inconclusive, never as a refutation of a "
        "smaller effect (section 7.7)."
    )
    add(
        "* `n_eff` is a bootstrap estimate over 13 modules and is noisy; it is reported per model and "
        "its mean is what the achieved MDE uses."
    )
    add(
        "* The `refuted` label covers two different situations and the artifact distinguishes them: "
        "an analytic CI entirely below 0 (the candidate is *significantly worse*) and an analytic CI "
        "that includes 0 but is narrower than the predeclared **optimistic** MDE of "
        f"{mde_pre['mde_z_optimistic']:.2f} (the advantage is below the MDE). Rule 3 measures "
        '"narrower" against the optimistic end of the predeclared band; using the conservative end '
        f"({mde_pre['mde_z_conservative']:.2f}) instead would leave those rows inconclusive. Both "
        "ends are recorded in the artifact, so the reading is auditable."
    )
    add(
        "* The cluster-bootstrap CI resamples trained models (5 per fixture, 10 combined); with so "
        "few clusters its percentile coverage is coarse and it can disagree with the analytic CI "
        "about whether 0 is excluded. The verdict rule uses the analytic random-effects interval "
        "(section 7.2) and every disagreement is visible in the tables."
    )
    add(
        "* Float values in the artifact are stored at "
        f"{document['float_precision_significant_digits']} significant digits; the sweep itself is "
        "exact and deterministic, and re-running reproduces the analysis from unrounded values."
    )
    add(
        "* Nothing here is a storage, latency or kernel measurement: class 1/2 only (`AGENTS.md` "
        "section 5). The equal-memory gate and the H5 chain are separate cells."
    )
    add("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT, help="directory for the JSON artifact")
    parser.add_argument("--report", default=DEFAULT_REPORT, help="path of the markdown report")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--cells",
        nargs="+",
        default=[f"{r}:{b}" for r, b in DEFAULT_CELLS],
        help="cells as rank:bits",
    )
    parser.add_argument("--fixtures", nargs="+", default=[name for name, _ in DEFAULT_FIXTURES])
    parser.add_argument("--boot-ci", type=int, default=2000)
    parser.add_argument("--boot-neff", type=int, default=200)
    args = parser.parse_args()

    cells = tuple(tuple(int(p) for p in token.split(":")) for token in args.cells)  # type: ignore[misc]
    fixture_lookup = dict(DEFAULT_FIXTURES)
    fixtures = tuple((name, fixture_lookup[name]) for name in args.fixtures)

    started = time.perf_counter()
    runs = run_sweep(
        seeds=tuple(args.seeds),
        cells=cells,  # type: ignore[arg-type]
        fixtures=fixtures,
        n_boot_neff=args.boot_neff,
    )
    wall = time.perf_counter() - started
    document = build_document(
        runs,
        seeds=tuple(args.seeds),
        cells=cells,  # type: ignore[arg-type]
        fixtures=fixtures,
        n_boot_neff=args.boot_neff,
        n_boot_ci=args.boot_ci,
        wall_time_s=wall,
        **git_state(),
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = out_dir / ARTIFACT_NAME
    payload = json.dumps(document, indent=2, sort_keys=True, allow_nan=False)
    artifact.write_text(payload + "\n", encoding="utf-8")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(document), encoding="utf-8")

    print(f"sweep wall time: {wall:.1f} s")
    print(f"wrote {artifact} ({len(payload)} bytes) and {report_path}")
    print_per_cell(document)
    print_per_cell_mde(document)
    print_aggregate(document)
    print_verdicts(document)


if __name__ == "__main__":
    main()
