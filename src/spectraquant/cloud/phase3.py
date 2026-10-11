"""A-0024: diagnose joint interactions; test bounded allocation and clipping.

All real-model execution is cloud CUDA only. Helpers accept tiny CPU fixtures.
No WikiText test is read. New C4 validation is read after selection freeze.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import signal
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from spectraquant.cloud.phase2 import (
    Candidate,
    Choice,
    _blocks,
    _evaluate,
    _save,
    _write_study_manifest,
    build_candidates,
    choose_allocation,
    paired_nll_summary,
)
from spectraquant.evaluation.phase2_report import verify_checkpoints
from spectraquant.quantization import QuantSpec, deserialize_state, serialize_state


def _arm_wall_timer(seconds: float) -> Callable[[], None]:
    """Unix cloud watchdog interrupts networking, solves and checkpoint writes.

    Cancel before failure evidence/manifest export. Kaggle's separate platform
    timeout remains the final ceiling even if a native extension delays signals.
    """
    sigalrm = getattr(signal, "SIGALRM", None)
    setitimer = getattr(signal, "setitimer", None)
    itimer_real = getattr(signal, "ITIMER_REAL", None)
    if sigalrm is None or setitimer is None or itimer_real is None:
        raise RuntimeError("real phase3 execution requires a Unix cloud watchdog")
    previous = signal.getsignal(sigalrm)

    def timeout(_signum: int, _frame: Any) -> None:
        raise TimeoutError("phase3 scientific wall timer expired; preserve partial evidence")

    signal.signal(sigalrm, timeout)
    setitimer(itimer_real, seconds)

    def cancel() -> None:
        setitimer(itimer_real, 0)
        signal.signal(sigalrm, previous)

    return cancel


def clipped_candidate(weight: Tensor, factors: Sequence[float]) -> tuple[Candidate, Tensor]:
    """Select group32/axis0 thresholds by exact reloaded FP16-weight MSE.

    Input (out,in) float matrix; CPU FP32 packing, CPU FP16-rounded scoring.
    Independent groups are rows of <=32 entries in each input column. Alpha=1
    guarantees the ordinary RTN option is present. No data labels are consulted.
    Return one actual packed candidate and alpha matrix (ceil(out/32),in).
    """
    if 1.0 not in factors or any(not math.isfinite(x) or not 0 < x <= 1 for x in factors):
        raise ValueError("clipping factors must include 1 and lie in (0,1]")
    w = weight.detach().cpu().float()
    if w.ndim != 2 or not w.numel():
        raise ValueError("nonempty matrix required")
    spec = QuantSpec(4, "per_group", 32, True, axis=0)
    n = math.ceil(w.shape[0] / 32)
    bounds = torch.stack([w[i * 32 : (i + 1) * 32].abs().amax(0) for i in range(n)])
    best = torch.full_like(bounds, float("inf"))
    selected = torch.ones_like(bounds)
    # Score the stored representation, including scale serialization and FP16 cast.
    for alpha in factors:
        threshold = (bounds * alpha).repeat_interleave(32, 0)[: w.shape[0]]
        source = torch.maximum(torch.minimum(w, threshold), -threshold)
        blob = serialize_state({"base": source}, {"base": spec})
        q = deserialize_state(blob, {"base": tuple(w.shape)}, {"base": spec})["base"]
        error = (w - q.half().float()).square()
        loss = torch.stack([error[i * 32 : (i + 1) * 32].sum(0) for i in range(n)])
        improve = loss < best
        best = torch.where(improve, loss, best)
        selected = torch.where(improve, alpha, selected)
    threshold = (bounds * selected).repeat_interleave(32, 0)[: w.shape[0]]
    source = torch.maximum(torch.minimum(w, threshold), -threshold)
    blob = serialize_state({"base": source}, {"base": spec})
    shapes = {"base": tuple(w.shape)}
    q = deserialize_state(blob, shapes, {"base": spec})["base"]
    return Candidate(q, blob, shapes, spec, 0.0), selected


def read_checkpoint(directory: Path) -> dict[str, Candidate]:
    """Decode the exact concatenated SQPACK layout; hashes checked by caller."""
    layout = json.loads((directory / "layout.json").read_text(encoding="utf-8"))
    blob = (directory / "weights.sqpack").read_bytes()
    result: dict[str, Candidate] = {}
    offset = 0
    for row in layout["targets"]:
        if row["offset"] != offset or row["bytes"] <= 0 or row["layer"] in result:
            raise ValueError("invalid checkpoint layout")
        segment = blob[offset : offset + row["bytes"]]
        if len(segment) != row["bytes"]:
            raise ValueError("truncated checkpoint layout")
        shapes = {name: tuple(shape) for name, shape in row["shapes"].items()}
        spec = QuantSpec(**row["quant_spec"])
        parts = deserialize_state(segment, shapes, dict.fromkeys(shapes, spec))
        effective = parts["base"]
        if row["rank"]:
            effective = effective + parts["B"] @ parts["A"]
        result[row["layer"]] = Candidate(effective, segment, shapes, spec, 0.0)
        offset += row["bytes"]
    if offset != len(blob):
        raise ValueError("layout does not cover checkpoint payload")
    return result


def save_checkpoint(
    directory: Path,
    candidates: Mapping[str, Candidate],
    untouched: Mapping[str, Tensor],
    model: Mapping[str, str],
) -> dict[str, Any]:
    """Save actual candidates used in evaluation, plus unique untouched weights."""
    directory.mkdir(parents=True, exist_ok=True)
    layout, offset = [], 0
    with (directory / "weights.sqpack").open("wb") as stream:
        for name, candidate in candidates.items():
            stream.write(candidate.blob)
            layout.append(
                {
                    "layer": name,
                    "offset": offset,
                    "bytes": candidate.payload_bytes,
                    "shapes": candidate.shapes,
                    "quant_spec": asdict(candidate.spec),
                    "rank": candidate.shapes.get("A", (0,))[0],
                    "bits": candidate.spec.bits,
                }
            )
            offset += candidate.payload_bytes
    torch.save(dict(untouched), directory / "unchanged.pt")
    _save(
        directory / "layout.json",
        {
            "model": model,
            "targets": layout,
            "tied_weights": "restore pinned architecture first; unique named parameters preserve tied embeddings",
        },
    )
    files = {
        p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in directory.iterdir()
        if p.is_file()
    }
    return {
        "target_payload_bytes": offset,
        "full_checkpoint_bytes": sum(x["bytes"] for x in files.values()),
        "files": files,
        "scope": "packed targets + layout + unique unchanged parameters; excludes tokenizer/runtime",
    }


def run_phase3(config_path: str | Path) -> dict[str, Any]:
    from spectraquant.cloud.plan_data import load_plan_texts, plan_dataset_ref
    from spectraquant.cloud.plan_runner import compressible_linear_names, load_model
    from spectraquant.experiment_plan import load_plan
    from spectraquant.reporting.gitinfo import git_info

    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + cfg["max_wall_seconds"]
    report: dict[str, Any] = {
        "study_key": "phase3",
        "status": "running",
        "config": cfg,
        "started_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git": asdict(git_info(Path.cwd())),
        "quality_class": 2,
        "storage_class": 3,
        "runtime": "dense FP16 reconstruction of packed weights; no low-bit kernel",
        "seeds_interpretation": "three prior calibration subsets, one checkpoint; not training replicates",
        "arms": {},
        "diagnostics": {},
        "development_candidates": {},
    }
    rows: list[dict[str, Any]] = []
    cancel_wall: Callable[[], None] | None = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("phase3 research execution requires cloud CUDA; no CPU fallback")
        cancel_wall = _arm_wall_timer(max(0.01, deadline - time.monotonic()))
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        report["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "visible_count": torch.cuda.device_count(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        }
        if shutil.disk_usage(out).free < 3 * 1024**3:
            raise RuntimeError("less than 3 GiB free for final checkpoint/export copies")
        sources = list(Path(cfg["input_root"]).rglob("phase2-global-damage/metrics.json"))
        if len(sources) != 1:
            raise ValueError(
                f"expected exactly one prior kernel metrics file, found {len(sources)}"
            )
        source = sources[0]
        if hashlib.sha256(source.read_bytes()).hexdigest() != cfg["source_metrics_sha256"]:
            raise ValueError("prior source report checksum changed")
        old = json.loads(source.read_text(encoding="utf-8"))
        report["source_checkpoint_integrity"] = verify_checkpoints(old, source.parent)
        report["source_metrics_sha256"] = cfg["source_metrics_sha256"]
        scores_path = source.parent / "scores.json"
        if hashlib.sha256(scores_path.read_bytes()).hexdigest() != cfg["source_scores_sha256"]:
            raise ValueError("prior source scores checksum changed")
        rows = json.loads(scores_path.read_text(encoding="utf-8"))
        plan = load_plan(cfg["plan"])
        model, tokenizer, model_id, revision, _ = load_model(plan, device="cuda", dtype="float16")
        report["model"] = {"id": model_id, "revision": revision}
        if report["model"] != old["model"]:
            raise ValueError("source and current model pins differ")
        names, excluded = compressible_linear_names(model)
        params = dict(model.named_parameters())
        originals = {name: params[name].detach().cpu().clone() for name in names}
        untouched = {
            name: p.detach().cpu().clone() for name, p in params.items() if name not in names
        }
        report["excluded"] = excluded
        c4, wiki = plan_dataset_ref(plan, "calibration"), plan_dataset_ref(plan, "development")
        train_docs = load_plan_texts(
            c4,
            split="train",
            dataset_config="en",
            max_documents=old["config"]["calibration_documents"],
            stream=True,
        )
        dev_docs = load_plan_texts(wiki, split="validation", dataset_config="wikitext-2-raw-v1")

        def digest(texts: Sequence[str]) -> str:
            return hashlib.sha256("\n\n".join(texts).encode()).hexdigest()

        if digest(train_docs) != old["data"]["calibration"]["sha256"]:
            raise ValueError("calibration text fingerprint changed")
        if digest(dev_docs) != old["data"]["development"]["sha256"]:
            raise ValueError("development text fingerprint changed")
        pool = _blocks(
            tokenizer,
            train_docs,
            length=old["config"]["calibration_sequence"],
            cap=old["config"]["calibration_pool_tokens"],
        )
        dev = _blocks(
            tokenizer,
            dev_docs,
            length=old["config"]["evaluation_sequence"],
            cap=old["config"]["development_tokens"],
        )
        selected_indices = {
            s: random.Random(s).sample(range(len(pool)), old["config"]["calibration_blocks"])
            for s in cfg["seeds"]
        }
        used = {i for indices in selected_indices.values() for i in indices}
        diagnostic_indices = random.Random(99).sample(
            [i for i in range(len(pool)) if i not in used], cfg["diagnostic_train_blocks"]
        )
        diagnostic_blocks = [pool[i] for i in diagnostic_indices]
        report["data"] = {
            "calibration": old["data"]["calibration"],
            "development": old["data"]["development"],
            "train_diagnostic_indices": diagnostic_indices,
            "train_diagnostic_note": "disjoint token blocks, same training documents; not independent documents",
        }
        report["fp16_development"] = _evaluate(model, dev, deadline=deadline)
        cache: dict[str, dict[tuple[int, int], Candidate]] = {}
        clipping: dict[str, Candidate] = {}
        histogram: dict[str, int] = {}
        build_started = time.monotonic()
        for i, name in enumerate(names):
            if time.monotonic() > deadline - 600:
                raise TimeoutError("reserve for evaluation and export exhausted")
            cache[name] = build_candidates(
                originals[name], bits=cfg["bits"], ranks=cfg["ranks"], svd_device="cuda"
            )
            clipped, alphas = clipped_candidate(originals[name], cfg["clip_factors"])
            clipping[name] = clipped
            for alpha in cfg["clip_factors"]:
                key = str(alpha)
                histogram[key] = histogram.get(key, 0) + int((alphas == alpha).sum())
            for option in cache[name].values():
                option.weight = option.weight.half()
            clipping[name].weight = clipping[name].weight.half()
            if (
                i == 1
                and (time.monotonic() - build_started) / 2 * len(names)
                > deadline - time.monotonic() - 600
            ):
                raise TimeoutError("candidate timing projects beyond budget")
            if i % 35 == 0:
                print(
                    f"phase3 candidates {i + 1}/{len(names)} elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )
        report["build_seconds"] = time.monotonic() - build_started
        report["clipping_alpha_group_counts"] = histogram
        baseline = {name: cache[name][4, 0] for name in names}
        budget = sum(c.payload_bytes for c in baseline.values())
        report["target_payload_budget_bytes"] = budget
        if (
            budget != old["target_payload_budget_bytes"]
            or sum(c.payload_bytes for c in clipping.values()) != budget
        ):
            raise ValueError("source/current/clipping payload bytes differ")

        def install(candidates: Mapping[str, Candidate]) -> None:
            with torch.no_grad():
                for name in names:
                    params[name].copy_(candidates[name].weight.to(params[name]))

        install(baseline)
        report["uniform4_development"] = _evaluate(model, dev, deadline=deadline)
        if abs(report["uniform4_development"]["nll"] - old["uniform4_development"]["nll"]) > 1e-6:
            raise ValueError("source uniform4 development NLL did not reproduce")
        baseline_diag = _evaluate(model, diagnostic_blocks, deadline=deadline)
        report["baseline_train_diagnostic"] = baseline_diag
        score_map = {(r["seed"], r["layer"], r["bits"], r["rank"]): r for r in rows}
        for arm, evidence in old["arms"].items():
            loaded = read_checkpoint(source.parent / "checkpoints" / arm)
            if set(loaded) != set(names):
                raise ValueError("source target set changed")
            allocation = evidence.get(
                "allocation", {name: {"bits": 4, "rank": 0} for name in names}
            )
            for name in names:
                c = allocation[name]
                if loaded[name].blob != cache[name][c["bits"], c["rank"]].blob:
                    raise ValueError(
                        f"candidate bytes changed: {arm}/{name}; refuse stale score reuse"
                    )
            if arm == "uniform4":
                continue
            seed = int(arm.rsplit("seed", 1)[1])
            marginal_sum = sum(
                score_map[seed, name, c["bits"], c["rank"]]["damage"]
                for name, c in allocation.items()
            )
            if abs(marginal_sum - evidence["surrogate_delta_nll"]) > 1e-10:
                raise ValueError("source score objective checksum consistency failed")
            install(baseline)
            calibration = [pool[j] for j in selected_indices[seed]]
            base_cal = _evaluate(model, calibration, deadline=deadline)
            install(loaded)
            joint = _evaluate(model, calibration, deadline=deadline)
            independent = _evaluate(model, diagnostic_blocks, deadline=deadline)
            delta = joint["nll"] - base_cal["nll"]
            report["diagnostics"][arm] = {
                "calibration_indices": selected_indices[seed],
                "baseline_calibration": base_cal,
                "joint_calibration": joint,
                "summed_marginal_delta": marginal_sum,
                "actual_joint_delta": delta,
                "nonadditivity_gap": delta - marginal_sum,
                "disjoint_train_delta": independent["nll"] - baseline_diag["nll"],
                "disjoint_train": independent,
            }
            print(f"diagnostic {arm}: marginal={marginal_sum:+.5f} joint={delta:+.5f}", flush=True)
            _save(out / "metrics.json", report)
        all_allocations: dict[str, dict[str, Candidate]] = {}
        for seed in cfg["seeds"]:
            choices = {
                name: [
                    Choice(r["bits"], r["rank"], r["payload_bytes"], r["damage"])
                    for r in rows
                    if r["seed"] == seed and r["layer"] == name
                ]
                for name in names
            }
            for name, xs in choices.items():
                if len(xs) != len(cache[name]) or {(c.bits, c.rank) for c in xs} != set(
                    cache[name]
                ):
                    raise ValueError("source score grid incomplete or duplicated")
                if any(c.payload_bytes != cache[name][c.bits, c.rank].payload_bytes for c in xs):
                    raise ValueError("source score payload inconsistent")
                if any(
                    score_map[seed, name, c.bits, c.rank]["calibration_indices"]
                    != selected_indices[seed]
                    for c in xs
                ):
                    raise ValueError("source score calibration subset changed")
            for family in ("bits", "residual"):
                allowed = {
                    name: [c for c in xs if family == "residual" or c.rank == 0]
                    for name, xs in choices.items()
                }
                for cap in cfg["change_caps"]:
                    allocation = choose_allocation(allowed, budget, max_changes=cap)
                    arm = f"{family}_k{cap}_seed{seed}"
                    candidates = {
                        name: cache[name][c.bits, c.rank] for name, c in allocation.items()
                    }
                    all_allocations[arm] = candidates
                    install(candidates)
                    report["development_candidates"][arm] = {
                        "development": _evaluate(model, dev, deadline=deadline),
                        "changed_targets": sum(
                            (c.bits, c.rank) != (4, 0) for c in allocation.values()
                        ),
                        "surrogate_delta_nll": sum(c.damage for c in allocation.values()),
                        "target_payload_bytes": sum(c.payload_bytes for c in allocation.values()),
                        "allocation": {name: asdict(c) for name, c in allocation.items()},
                    }
            _save(out / "metrics.json", report)
        install(clipping)
        clip_dev = _evaluate(model, dev, deadline=deadline)
        report["development_candidates"]["mseclip"] = {"development": clip_dev}
        means = {
            f"{family}_k{cap}": sum(
                report["development_candidates"][f"{family}_k{cap}_seed{s}"]["development"]["nll"]
                for s in cfg["seeds"]
            )
            / len(cfg["seeds"])
            for family in ("bits", "residual")
            for cap in cfg["change_caps"]
        }
        means.update(uniform4=report["uniform4_development"]["nll"], mseclip=clip_dev["nll"])
        winner = min(means, key=lambda key: means[key])
        if means["uniform4"] - means[winner] < cfg["minimum_development_nll_gain"]:
            winner = "uniform4"
        report["selection_frozen_before_test"] = {
            "family": winner,
            "mean_development_nll": means,
            "minimum_gain": cfg["minimum_development_nll_gain"],
            "frozen_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        _save(out / "selection.json", report["selection_frozen_before_test"])
        _save(out / "metrics.json", report)
        # First read of the new project holdout; no tuning after this point.
        heldout_docs = load_plan_texts(
            c4,
            split="validation",
            dataset_config="en",
            max_documents=cfg["heldout_documents"],
            stream=True,
        )
        known_hashes = {hashlib.sha256(t.encode()).hexdigest() for t in train_docs + dev_docs}
        if any(hashlib.sha256(t.encode()).hexdigest() in known_hashes for t in heldout_docs):
            raise ValueError("exact heldout document overlap")
        test = _blocks(
            tokenizer,
            heldout_docs,
            length=old["config"]["evaluation_sequence"],
            cap=cfg["heldout_tokens"],
        )
        if sum(block.numel() for block in test) != cfg["heldout_tokens"]:
            raise ValueError("heldout token cap not reached; refuse a silently smaller evaluation")
        report["data"]["test"] = {
            "id": c4.name,
            "revision": c4.revision,
            "split": "validation",
            "sha256": digest(heldout_docs),
            "prior_exposure": "first use for project method development; pretraining exposure unknown",
        }
        with torch.no_grad():
            for name in names:
                params[name].copy_(originals[name].to(params[name]))
        report["fp16_test"] = _evaluate(model, test, deadline=deadline)
        finals = {"uniform4": baseline, "mseclip": clipping}
        if winner not in finals:
            finals.update(
                {f"{winner}_seed{s}": all_allocations[f"{winner}_seed{s}"] for s in cfg["seeds"]}
            )
        for arm, candidates in finals.items():
            install(candidates)
            report["arms"][arm] = {
                **report["development_candidates"].get(arm, {}),
                "test": _evaluate(model, test, deadline=deadline),
                "storage": save_checkpoint(
                    out / "checkpoints" / arm, candidates, untouched, report["model"]
                ),
            }
            if report["arms"][arm]["storage"]["target_payload_bytes"] > budget:
                raise ValueError("final checkpoint exceeds budget")
            _save(out / "metrics.json", report)
        for arm, evidence in report["arms"].items():
            if arm != "uniform4":
                evidence["paired_vs_uniform4"] = paired_nll_summary(
                    report["arms"]["uniform4"]["test"]["blocks"],
                    evidence["test"]["blocks"],
                    seed=20261011,
                )
        report["status"] = "success"
    except Exception as exc:
        report["status"] = "failed"
        report["failure_reason"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if cancel_wall is not None:
            cancel_wall()
        report["wall_seconds"] = time.monotonic() - started
        report["gpu_device_hours_upper_bound"] = (
            report["wall_seconds"] / 3600 * report.get("gpu", {}).get("visible_count", 0)
        )
        _save(out / "scores.json", rows)
        _save(out / "metrics.json", report)
        _write_study_manifest(report, config_path, out / "run_manifest.json")
    return report
