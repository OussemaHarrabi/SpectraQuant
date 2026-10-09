"""Allocation frontier on the Tier-0 fixture with the real proxy and measured serialization (M6).

This is the Milestone-6 integration experiment. It wires the **declared candidate proxy**
(``gain_aware_composed``) into the allocator's ``error_fn``, allocates at every point of a
predeclared byte-budget ladder with the uniform / greedy / CP-SAT (exact) solvers, and then:

* serializes each allocated configuration for real and reconciles the **class-3** measured container
  bytes against the **class-1** accounted bytes under the preregistered 0.5 % tolerance;
* measures the **class-2** end-to-end damage of each allocation against the uncompressed fixture;
* gates every quality comparison through ``spectraquant.reporting.comparability.assert_equal_memory``;
* reports the rank agreement between the proxy's predicted error and the measured damage.

Everything here is a **Tier-0 fixture** result: one seed, 13 modules, synthetic task, class 1/2/3
numbers only. It is exploratory and does **not** speak to the confirmatory H4 cell, which is a
Tier-1/2 cloud cell that has not been run (see the report and ``preregistration.md`` sections 8/11).

Usage:
    uv run python scripts/experiments/allocation_frontier.py [--out DIR]

Writes ``frontier.json`` plus one deterministic manifest per (budget, solver) into ``DIR/manifests/``.
CPU-only and deterministic (single-threaded); well under a minute.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from spectraquant.allocation import (
    BYTE_PARITY_TOLERANCE,
    ORToolsNotInstalledError,
    QuantCostModel,
    assert_byte_parity,
    build_manifest,
    equal_memory_manifest,
    gate_equal_memory,
    make_accounting_cost_fn,
    make_proxy_problem,
    ortools_available,
    reconcile_allocation_bytes,
    solve_greedy,
    solve_ortools,
    solve_uniform,
    write_manifest,
)
from spectraquant.evaluation.toy import (
    TinyConfig,
    layer_inputs,
    layer_names,
    network_single_layer_damage,
    norm_preimages,
    train_tiny,
)
from spectraquant.factorization import truncated_svd
from spectraquant.proxies.analysis import following_norm_container, spearman
from spectraquant.proxies.base import LayerInputs
from spectraquant.proxies.gain import estimate_downstream_gains
from spectraquant.proxies.variants import get_proxy
from spectraquant.quantization import SQ_CONTAINER_FORMAT_ID, QuantSpec, fake_quantize
from spectraquant.reporting.comparability import UnequalMemoryComparison
from spectraquant.reporting.gitinfo import git_info

# --------------------------------------------------------------------------------------
# Predeclared configuration (fixed before any quality number is computed).
# --------------------------------------------------------------------------------------

SCHEMA_ID = "spectraquant.allocation-frontier/1"
SUBSTRATE = "LOCAL-FIXTURE"
TRAIN_STEPS = 120
SEED = 0
RANKS: tuple[int, ...] = (4, 8, 16)
BITS: tuple[int, ...] = (4, 8)
GROUP_SIZE = 32
CANDIDATE_PROXY = "gain_aware_composed"
BYTE_PARITY_TOLERANCE_LOCAL = BYTE_PARITY_TOLERANCE  # 0.005, preregistration.md section 5
LADDER_RULE = (
    "sorted set of the accounted byte totals of the uniform (rank, bit) configurations, so every "
    "ladder point has a uniform arm at exactly its own bytes; fixed before any quality number"
)
SOLVERS = ("uniform", "greedy", "ortools")

PROXY_UNIT = (
    "sum over layers of the gain-aware per-layer mean squared layer-output error "
    "g^2 * mean_i ||x_i W^T - x_i (Q(B)Q(A))^T||^2"
)


def _git_block() -> dict[str, Any]:
    info = git_info()
    return {
        "commit": info.commit,
        "short_commit": info.short_commit,
        "branch": info.branch,
        "describe": info.describe,
        "dirty": info.dirty,
        "error": info.error,
    }


def _spec(cost_model: QuantCostModel, bits: int) -> QuantSpec:
    return cost_model.spec_for_bits(bits)


def deterministic_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return a manifest with the solver's wall-clock diagnostic removed.

    ``solve_ortools`` records ``ortools_wall_time_s`` in its diagnostics. That is real telemetry but
    not part of the allocation, and it makes the written manifest differ run to run. The per-point
    manifest files are therefore stripped of ``*wall_time_s`` diagnostics so that a manifest is
    byte-deterministic; the timing is kept in ``frontier.json`` on the arm record instead.
    """
    diagnostics = {
        key: value
        for key, value in manifest.get("diagnostics", {}).items()
        if not str(key).endswith("wall_time_s")
    }
    return {**manifest, "diagnostics": diagnostics}


def measure_joint_damage(
    model: Any,
    ids: torch.Tensor,
    names: tuple[str, ...],
    per_layer: dict[str, tuple[int, int]],
    cost_model: QuantCostModel,
) -> tuple[float, float]:
    """Compress every layer to its allocated ``(rank, bits)`` and measure end-to-end damage (class 2).

    Returns ``(hidden_damage, logits_damage)``: mean squared change of the final hidden state and of
    the logits against the uncompressed forward pass, both in float64.
    """
    originals = {n: layer.weight.detach().clone() for n, layer in model.linear_layers().items()}
    with torch.no_grad():
        base_logits, base_hidden = model(ids)
        base_flat = base_hidden.reshape(-1, base_hidden.shape[-1])
        for name in names:
            rank, bits = per_layer[name]
            spec = _spec(cost_model, int(bits))
            factors = truncated_svd(originals[name], int(rank))
            compressed = fake_quantize(factors.B, spec) @ fake_quantize(factors.A, spec)
            model.linear_layers()[name].weight.copy_(compressed)
        logits, hidden = model(ids)
        for name, layer in model.linear_layers().items():
            layer.weight.copy_(originals[name])
    diff = hidden.reshape(-1, hidden.shape[-1]) - base_flat
    hidden_damage = float(torch.mean(torch.sum(diff.to(torch.float64) ** 2, dim=-1)))
    logit_diff = (logits - base_logits).to(torch.float64)
    logits_damage = float(torch.mean(torch.sum(logit_diff * logit_diff, dim=-1)))
    return hidden_damage, logits_damage


def build_context(
    model: Any,
    cfg: TinyConfig,
    ids: torch.Tensor,
    names: tuple[str, ...],
    options: tuple[tuple[int, int], ...],
    cost_model: QuantCostModel,
    activations: dict[str, torch.Tensor],
    preimages: dict[str, torch.Tensor],
    weights: dict[str, torch.Tensor],
) -> tuple[dict[tuple[int, int], Any], dict[tuple[int, int], Any]]:
    """Measure the option-specific proxy context and the per-layer single-layer damage.

    Returns ``(single_by_option, gains_by_option)``. ``single_by_option[(r, b)]`` is the
    :class:`DamageReport` of compressing each layer alone at ``(r, b)`` (its ``per_layer`` map is the
    measured class-2 single-layer damage used for the per-layer rank agreement), and
    ``gains_by_option`` is the downstream-gain estimate at that option.
    """
    single_by_option: dict[tuple[int, int], Any] = {}
    gains_by_option: dict[tuple[int, int], Any] = {}

    def forward(name: str, perturbation: torch.Tensor) -> torch.Tensor:
        hidden = model(ids, inject=(name, perturbation))[1]
        return hidden.reshape(-1, cfg.d_model)

    for rank, bits in options:
        spec = _spec(cost_model, bits)
        report = network_single_layer_damage(model, ids, names, rank, spec)
        single_by_option[(rank, bits)] = report
        gains_by_option[(rank, bits)] = estimate_downstream_gains(forward, report.layer_errors)
    return single_by_option, gains_by_option


def make_context_fn(
    model: Any,
    activations: dict[str, torch.Tensor],
    preimages: dict[str, torch.Tensor],
    weights: dict[str, torch.Tensor],
    gains_by_option: dict[tuple[int, int], Any],
) -> Any:
    """Return the option-specific ``(layer, rank, bits) -> LayerInputs`` context callable."""

    def context(layer: str, rank: int, bits: int) -> LayerInputs:
        return LayerInputs(
            weight=weights[layer],
            activations=activations[layer],
            following_norm=following_norm_container(model, layer, preimages),
            downstream_gain=gains_by_option[(int(rank), int(bits))].per_layer[layer],
        )

    return context


def _arm_record(
    allocation: Any,
    problem: Any,
    weights: dict[str, torch.Tensor],
    cost_model: QuantCostModel,
    names: tuple[str, ...],
    manifest_path: str,
    single_by_option: dict[tuple[int, int], Any],
    hidden_damage: float,
    logits_damage: float,
) -> dict[str, Any]:
    """Assemble the JSON record of one solved arm, including its byte reconciliation."""
    reconciliation = reconcile_allocation_bytes(
        weights=weights, per_layer=allocation.per_layer, cost_model=cost_model
    )
    assert_byte_parity(reconciliation)  # requirement: class-3 vs class-1 parity, hard assert
    per_layer_pred = {
        name: float(problem.error_fn(name, *allocation.per_layer[name])) for name in names
    }
    per_layer_measured = {
        name: float(single_by_option[allocation.per_layer[name]].per_layer[name]) for name in names
    }
    ranks = [allocation.per_layer[n][0] for n in names]
    bits = [allocation.per_layer[n][1] for n in names]
    return {
        "solver": allocation.solver,
        "manifest_path": manifest_path,
        "per_layer": {n: [int(r), int(b)] for n, (r, b) in sorted(allocation.per_layer.items())},
        "accounted_bytes": int(allocation.accounted_bytes),
        "measured_bytes": int(reconciliation["measured_bytes"]),
        "byte_parity": reconciliation,
        "predicted_error": float(allocation.predicted_error),
        "measured_hidden_damage": float(hidden_damage),
        "measured_logits_damage": float(logits_damage),
        "distinct_options": len(set(allocation.per_layer.values())),
        "mean_rank": sum(ranks) / len(ranks),
        "mean_bits": sum(bits) / len(bits),
        "per_layer_predicted_error": per_layer_pred,
        "per_layer_measured_damage": per_layer_measured,
        "per_layer_rank_agreement": spearman(
            [per_layer_pred[n] for n in names], [per_layer_measured[n] for n in names]
        ),
    }


def _comparison(
    reference_arm: dict[str, Any],
    candidate_arm: dict[str, Any],
    *,
    reference_name: str,
    candidate_name: str,
    budget: int,
    seed: int,
) -> dict[str, Any]:
    """Gate a pair of arms at equal measured bytes and, if it passes, compare quality."""
    ref_manifest = equal_memory_manifest(
        f"frontier-{budget}-{reference_name}",
        measured_bytes=reference_arm["measured_bytes"],
        accounted_bytes=reference_arm["accounted_bytes"],
        seed=seed,
    )
    cand_manifest = equal_memory_manifest(
        f"frontier-{budget}-{candidate_name}",
        measured_bytes=candidate_arm["measured_bytes"],
        accounted_bytes=candidate_arm["accounted_bytes"],
        seed=seed,
    )
    a = reference_arm["measured_bytes"]
    b = candidate_arm["measured_bytes"]
    relative = abs(a - b) / max(a, b) if max(a, b) else 0.0
    record: dict[str, Any] = {
        "reference": reference_name,
        "candidate": candidate_name,
        "reference_measured_bytes": a,
        "candidate_measured_bytes": b,
        "relative_byte_difference": relative,
    }
    try:
        verdict = gate_equal_memory(ref_manifest, cand_manifest)
    except UnequalMemoryComparison as exc:
        record["equal_memory"] = False
        record["rejected"] = str(exc)
        return record
    record["equal_memory"] = True
    record["verdict"] = verdict.to_json_dict()
    ref_damage = reference_arm["measured_hidden_damage"]
    cand_damage = candidate_arm["measured_hidden_damage"]
    record["hidden_damage_reference"] = ref_damage
    record["hidden_damage_candidate"] = cand_damage
    record["hidden_damage_ratio"] = cand_damage / ref_damage if ref_damage else None
    record["candidate_beats_reference_hidden"] = cand_damage < ref_damage
    record["predicted_error_ratio"] = (
        candidate_arm["predicted_error"] / reference_arm["predicted_error"]
        if reference_arm["predicted_error"]
        else None
    )
    return record


def run(out_dir: Path) -> dict[str, Any]:
    """Run the whole experiment and write ``frontier.json`` plus per-point manifests."""
    torch.set_num_threads(1)
    torch.manual_seed(SEED)

    cfg = TinyConfig(seed=SEED)
    model, training = train_tiny(cfg, steps=TRAIN_STEPS)
    names = tuple(layer_names(model))
    ids = torch.arange(24)[None].repeat(2, 1) % 16
    cost_model = QuantCostModel(
        granularity="per_group", group_size=GROUP_SIZE, symmetric=True, axis=1
    )
    options = tuple((r, b) for r in RANKS for b in BITS)
    weights = {n: model.linear_layers()[n].weight.detach() for n in names}
    layer_shapes = {n: (int(w.shape[0]), int(w.shape[1])) for n, w in weights.items()}
    activations = dict(layer_inputs(model, ids, names))
    preimages = dict(norm_preimages(model, ids))

    # Refuse ranks the factorization would clamp (that would make the class-1 cost vacuous).
    for n, (out, in_features) in layer_shapes.items():
        for rank in RANKS:
            if rank > min(out, in_features):
                raise SystemExit(
                    f"rank {rank} exceeds layer {n!r} intrinsic rank {min(out, in_features)}"
                )

    single_by_option, gains_by_option = build_context(
        model, cfg, ids, names, options, cost_model, activations, preimages, weights
    )
    context = make_context_fn(model, activations, preimages, weights, gains_by_option)

    cost_fn = make_accounting_cost_fn(layer_shapes, cost_model)
    uniform_totals = {(r, b): sum(cost_fn(n, r, b) for n in names) for (r, b) in options}
    ladder = sorted(set(uniform_totals.values()))
    min_feasible = sum(cost_fn(n, *min(options, key=lambda o: cost_fn(n, *o))) for n in names)
    max_bytes = sum(max(cost_fn(n, *o) for o in options) for n in names)

    proxy = get_proxy(CANDIDATE_PROXY)

    manifest_dir = out_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    points: list[dict[str, Any]] = []
    all_reconciliations: list[dict[str, Any]] = []
    all_arms: list[dict[str, Any]] = []
    for budget in ladder:
        problem = make_proxy_problem(
            proxy=proxy,
            layer_inputs=context,
            layer_shapes=layer_shapes,
            ranks=RANKS,
            bits=BITS,
            budget_bytes=budget,
            cost_model=cost_model,
        )
        solvers = {
            "uniform": lambda p: solve_uniform(p),
            "greedy": lambda p: solve_greedy(p),
            "ortools": lambda p: solve_ortools(p, time_limit_s=20.0, seed=SEED),
        }
        arms: dict[str, dict[str, Any]] = {}
        for name in SOLVERS:
            allocation = solvers[name](problem)
            manifest = build_manifest(allocation, problem, solver_params={"seed": SEED})
            manifest_path = manifest_dir / f"{budget}-{name}.json"
            write_manifest(deterministic_manifest(manifest), manifest_path)
            hidden, logits = measure_joint_damage(
                model, ids, names, allocation.as_assignment(), cost_model
            )
            record = _arm_record(
                allocation,
                problem,
                weights,
                cost_model,
                names,
                str(manifest_path.relative_to(out_dir)),
                single_by_option,
                hidden,
                logits,
            )
            arms[name] = record
            all_reconciliations.append(record["byte_parity"])
            all_arms.append(record)
        comparisons = [
            _comparison(
                arms["uniform"],
                arms["ortools"],
                reference_name="uniform",
                candidate_name="ortools",
                budget=budget,
                seed=SEED,
            ),
            _comparison(
                arms["uniform"],
                arms["greedy"],
                reference_name="uniform",
                candidate_name="greedy",
                budget=budget,
                seed=SEED,
            ),
        ]
        points.append(
            {
                "budget_bytes": int(budget),
                "arms": arms,
                "comparisons": comparisons,
            }
        )

    max_relative = max(float(r["relative_difference"]) for r in all_reconciliations)
    total_overhead = sum(int(r["container_overhead_bytes"]) for r in all_reconciliations)
    per_solver_agreement = {
        solver: spearman(
            [a["predicted_error"] for a in all_arms if a["solver"] == solver],
            [a["measured_hidden_damage"] for a in all_arms if a["solver"] == solver],
        )
        for solver in SOLVERS
    }
    equal_memory_wins = [
        {
            "budget_bytes": point["budget_bytes"],
            "candidate": comp["candidate"],
            "hidden_damage_reference": comp["hidden_damage_reference"],
            "hidden_damage_candidate": comp["hidden_damage_candidate"],
            "hidden_damage_ratio": comp["hidden_damage_ratio"],
        }
        for point in points
        for comp in point["comparisons"]
        if comp.get("equal_memory") and comp.get("candidate_beats_reference_hidden")
    ]

    document: dict[str, Any] = {
        "schema": SCHEMA_ID,
        "substrate": SUBSTRATE,
        "scope": (
            "exploratory Tier-0 fixture (one seed, 13 modules, synthetic next-token task); "
            "not a Tier-1/2 result and not the confirmatory H4 cell"
        ),
        "measurement_classes": {
            "accounted_bytes": 1,
            "predicted_error": 2,
            "measured_hidden_damage": 2,
            "measured_logits_damage": 2,
            "measured_bytes": 3,
            "equal_memory_gate": "class 1 (accounted) / class 3 (measured)",
        },
        "git": _git_block(),
        "fixture": {
            "config": {
                "n_blocks": cfg.n_blocks,
                "d_model": cfg.d_model,
                "n_heads": cfg.n_heads,
                "d_ff": cfg.d_ff,
                "vocab": cfg.vocab,
                "seq_len": cfg.seq_len,
                "use_norm": cfg.use_norm,
                "final_norm": cfg.final_norm,
                "seed": cfg.seed,
            },
            "training": training,
            "n_layers": len(names),
            "layer_shapes": {n: list(layer_shapes[n]) for n in names},
        },
        "compression": {
            "ranks": list(RANKS),
            "bits": list(BITS),
            "group_size": GROUP_SIZE,
            "cost_model": cost_model.to_dict(),
            "serializer": SQ_CONTAINER_FORMAT_ID,
            "byte_parity_tolerance": BYTE_PARITY_TOLERANCE_LOCAL,
        },
        "proxy": {
            "name": CANDIDATE_PROXY,
            "measurement_class": 2,
            "unit": PROXY_UNIT,
            "context": "option-specific LayerInputs (activations + measured downstream gain per option)",
        },
        "budgets": {
            "ladder_bytes": [int(b) for b in ladder],
            "ladder_rule": LADDER_RULE,
            "uniform_totals": {f"{r}:{b}": int(v) for (r, b), v in sorted(uniform_totals.items())},
            "min_feasible_bytes": int(min_feasible),
            "max_bytes": int(max_bytes),
        },
        "points": points,
        "byte_reconciliation": {
            "all_within_tolerance": bool(max_relative <= BYTE_PARITY_TOLERANCE_LOCAL),
            "max_relative_difference": max_relative,
            "total_container_overhead_bytes": total_overhead,
            "n_reconciliations": len(all_reconciliations),
            "note": (
                "the SpectraQuant container is headerless and stores exactly the accounted terms "
                "(padded payload + per-block scales + zero-points), so measured class-3 bytes equal "
                "accounted class-1 bytes; the container overhead is 0 B by construction. A "
                "third-party container's overhead would appear in 'container_overhead_bytes'."
            ),
        },
        "rank_agreement": {
            "across_all_arms_spearman": spearman(
                [a["predicted_error"] for a in all_arms],
                [a["measured_hidden_damage"] for a in all_arms],
            ),
            "per_solver_spearman": per_solver_agreement,
            "per_arm_layer_spearman_mean": sum(
                float(a["per_layer_rank_agreement"]) for a in all_arms
            )
            / len(all_arms),
            "note": (
                "rank agreement is between the proxy's predicted error (class 2, hidden-state unit) "
                "and the measured hidden-state damage (class 2); logits damage is reported "
                "separately and is not the proxy's target"
            ),
        },
        "equal_memory_wins": equal_memory_wins,
        "h4": {
            "falsifier": (
                "proxy-allocated (r_l, b_l) does not dominate uniform rank/bit at any predeclared "
                "budget beyond CIs, or is matched by the HAWQ-V2-style bit-only allocator at equal "
                "bytes (preregistration.md section 11.1)"
            ),
            "confirmatory_cell": "NOT RUN",
            "confirmatory_cell_detail": (
                "preregistration.md section 8 row 7 (substrate CLOUD-COLAB, Tier 1; referred to in "
                "the M6 assignment as the Tier-2 H4 cell); no validated run_manifest.json exists"
            ),
            "fixture_evidence": (
                "exploratory Tier-0 only: the fixture numbers below bear on H4's premise (proxy "
                "ranking ability) and not on the confirmatory claim"
            ),
            "bit_only_comparator": "NOT RUN (HAWQ-V2-style comparator is a confirmatory-cell arm)",
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "frontier.json"
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/sample-results/allocation-frontier", type=Path)
    args = parser.parse_args()
    if not ortools_available():
        raise SystemExit(
            "solve_ortools requires the optional 'alloc' extra: run `uv sync --extra alloc`"
        )
    try:
        document = run(args.out)
    except ORToolsNotInstalledError as exc:  # pragma: no cover - guarded above
        raise SystemExit(str(exc)) from exc
    path = args.out / "frontier.json"
    print(f"wrote {path}")
    print(
        f"  ladder: {document['budgets']['ladder_bytes']} B; "
        f"byte-parity max rel diff "
        f"{document['byte_reconciliation']['max_relative_difference']:.3e}"
    )
    print(
        f"  rank agreement across arms: "
        f"{document['rank_agreement']['across_all_arms_spearman']:+.4f}"
    )
    print(f"  equal-memory wins (mixed beats uniform): {len(document['equal_memory_wins'])}")
    for win in document["equal_memory_wins"]:
        print(
            f"    budget {win['budget_bytes']} B: hidden damage "
            f"{win['hidden_damage_candidate']:.4e} vs uniform "
            f"{win['hidden_damage_reference']:.4e} (ratio {win['hidden_damage_ratio']:.3f})"
        )


if __name__ == "__main__":
    main()
