"""A-0023: exploratory deployment-matched global-damage allocation.

Quality uses float16 dense execution of reloaded SQPACK weights (class 2),
while payload sizes come from actual packed files (class 3). No speed claim.
Real-model execution is CUDA-only; synthetic correctness helpers support CPU.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from spectraquant.quantization import QuantSpec, deserialize_state, serialize_state


@dataclass(frozen=True)
class Choice:
    bits: int
    rank: int
    payload_bytes: int
    damage: float


@dataclass
class Candidate:
    weight: Tensor
    blob: bytes
    shapes: dict[str, tuple[int, ...]]
    spec: QuantSpec
    reload_max_abs: float

    @property
    def payload_bytes(self) -> int:
        return len(self.blob)


def build_candidates(
    weight: Tensor,
    *,
    bits: Sequence[int] = (3, 4, 8),
    ranks: Sequence[int] = (0, 8, 16),
    svd_device: str = "cpu",
) -> dict[tuple[int, int], Candidate]:
    """CPU packing, optional CUDA fp32 SVD; shape (out,in), CPU fp32 outputs.

    Residual is defined against the *reloaded* base, not the in-memory scale
    approximation. Each positive-rank candidate is Q(W)+Q(B)Q(A), with all
    three components stored at its declared bit width. Rank zero is plain Q(W).
    Full SVD is deterministic subject to backend floating-point differences.
    """
    w = weight.detach().cpu().float()
    if w.ndim != 2 or not ranks or 0 not in ranks or any(r < 0 for r in ranks):
        raise ValueError("a matrix and a rank grid containing zero are required")
    result: dict[tuple[int, int], Candidate] = {}
    for b in bits:
        spec = QuantSpec(int(b), "per_group", 32, True, axis=0)
        base_blob = serialize_state({"base": w}, {"base": spec})
        q = deserialize_state(base_blob, {"base": tuple(w.shape)}, {"base": spec})["base"]
        u, s, vh = torch.empty(0), torch.empty(0), torch.empty(0)
        if any(r > 0 for r in ranks):
            residual = (w - q).to(svd_device)
            u, s, vh = torch.linalg.svd(residual, full_matrices=False)
            u, s, vh = u.cpu(), s.cpu(), vh.cpu()
        for r in ranks:
            state = {"base": w}
            if r:
                rank = min(r, min(w.shape))
                root = s[:rank].sqrt()
                state["B"] = u[:, :rank] * root
                state["A"] = root[:, None] * vh[:rank]
            shapes = {key: tuple(value.shape) for key, value in state.items()}
            specs = dict.fromkeys(state, spec)
            blob = base_blob if r == 0 else serialize_state(state, specs)
            loaded = deserialize_state(blob, shapes, specs)
            effective = loaded["base"]
            if r:
                effective = effective + loaded["B"] @ loaded["A"]
            # Re-read bytes as a separate integrity check. Both proxy and final
            # evaluator receive this exact reconstruction, rounded to model dtype.
            again = deserialize_state(blob, shapes, specs)
            reconstructed = again["base"]
            if r:
                reconstructed = reconstructed + again["B"] @ again["A"]
            delta = float((effective - reconstructed).abs().max())
            result[int(b), int(r)] = Candidate(effective, blob, shapes, spec, delta)
    return result


def choose_allocation(
    choices: Mapping[str, Sequence[Choice]], budget_bytes: int, *, seconds: float = 30.0
) -> dict[str, Choice]:
    """Multiple-choice CP-SAT knapsack; negative damages are preserved.

    Loss deltas are rounded to 1e-8 NLL units for the integer objective. This
    additive surrogate is approximate: joint interactions require validation.
    """
    from ortools.sat.python import cp_model

    if not choices or any(not xs for xs in choices.values()):
        raise ValueError("nonempty choices required for every target")
    if sum(min(x.payload_bytes for x in xs) for xs in choices.values()) > budget_bytes:
        raise ValueError("infeasible byte budget")
    model = cp_model.CpModel()
    variables = {
        name: [model.new_bool_var(f"{name}:{i}") for i in range(len(xs))]
        for name, xs in choices.items()
    }
    for vs in variables.values():
        model.add_exactly_one(vs)
    model.add(
        sum(
            x.payload_bytes * v
            for name, xs in choices.items()
            for x, v in zip(xs, variables[name], strict=True)
        )
        <= budget_bytes
    )
    if any(not math.isfinite(x.damage) for xs in choices.values() for x in xs):
        raise ValueError("nonfinite candidate damage")
    model.minimize(
        sum(
            round(x.damage * 1e8) * v
            for name, xs in choices.items()
            for x, v in zip(xs, variables[name], strict=True)
        )
    )
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise ValueError(f"allocation infeasible or unsolved: {solver.status_name(status)}")
    result = {
        name: next(x for x, v in zip(xs, variables[name], strict=True) if solver.value(v))
        for name, xs in choices.items()
    }
    assert sum(x.payload_bytes for x in result.values()) <= budget_bytes
    return result


@contextmanager
def temporary_weight(parameter: Tensor, replacement: Tensor) -> Iterator[None]:
    """Restore the actual original parameter even on a failed intervention."""
    original = parameter.detach().clone()
    with torch.no_grad():
        parameter.copy_(replacement.to(parameter))
    try:
        yield
    finally:
        with torch.no_grad():
            parameter.copy_(original)


def paired_nll_summary(
    reference: Sequence[Mapping[str, Any]],
    treatment: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    repeats: int = 2500,
) -> dict[str, Any]:
    """Paired token-weighted block bootstrap; descriptive, not i.i.d. seeds."""
    if len(reference) != len(treatment) or not reference:
        raise ValueError("aligned nonempty evaluation blocks required")
    tokens = np.array([x["tokens"] for x in reference], dtype=np.float64)
    if any(a["tokens"] != b["tokens"] for a, b in zip(reference, treatment, strict=True)):
        raise ValueError("block token counts differ")
    delta = np.array(
        [b["nll_sum"] - a["nll_sum"] for a, b in zip(reference, treatment, strict=True)]
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(tokens), size=(repeats, len(tokens)))
    estimates = delta[indices].sum(axis=1) / tokens[indices].sum(axis=1)
    return {
        "delta_nll": float(delta.sum() / tokens.sum()),
        "ci95": np.quantile(estimates, [0.025, 0.975]).tolist(),
        "unit": "paired contiguous token blocks; not independent model seeds",
        "limitations": "adjacent blocks are dependent; exploratory interval, not a publication-level significance test",
    }


def _save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _blocks(tokenizer: Any, texts: Sequence[str], *, length: int, cap: int) -> list[Tensor]:
    ids = tokenizer("\n\n".join(texts), return_tensors="pt", truncation=False).input_ids[0][:cap]
    return [
        ids[i : i + length][None]
        for i in range(0, len(ids), length)
        if len(ids[i : i + length]) >= 2
    ]


def _evaluate(model: Any, blocks: Sequence[Tensor], *, deadline: float) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for block in blocks:
            if time.monotonic() >= deadline:
                raise TimeoutError("phase2 wall-clock limit reached; partial evidence preserved")
            ids = block.to("cuda")
            logits = model(input_ids=ids, use_cache=False).logits[:, :-1].float()
            n = int(ids.shape[1] - 1)
            nll = float(
                F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction="sum"
                )
            )
            if not math.isfinite(nll):
                raise ValueError("nonfinite model loss")
            records.append({"nll_sum": nll, "tokens": n})
    tokens = sum(x["tokens"] for x in records)
    if not tokens:
        raise ValueError("evaluation has no predicted tokens")
    nll = sum(x["nll_sum"] for x in records) / tokens
    return {"nll": nll, "perplexity": math.exp(nll), "tokens": tokens, "blocks": records}


def run_phase2(config_path: str | Path) -> dict[str, Any]:
    """Cloud-only execution, checkpointing before/after every major stage.

    Choices are frozen by JSON before submission. Only C4 calibration scores
    guide the solver; WikiText validation chooses a family; test is read after
    all choices are frozen. Prior WikiText-2 exposure is disclosed in A-0023.
    """
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
        "started_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "running",
        "config": cfg,
        "quality_class": 2,
        "storage_class": 3,
        "runtime": "float16 dense reconstruction of packed weights; no low-bit kernel",
        "seeds_interpretation": "three calibration subsets of one pretrained checkpoint, not training replicates",
        "git": asdict(git_info(Path.cwd())),
        "arms": {},
        "stages": [],
    }
    score_rows: list[dict[str, Any]] = []
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("phase2 research execution requires cloud CUDA; no CPU fallback")
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        report["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "visible_count": torch.cuda.device_count(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        }
        plan = load_plan(cfg["plan"])
        model, tokenizer, model_id, revision, _ = load_model(plan, device="cuda", dtype="float16")
        report["model"] = {"id": model_id, "revision": revision}
        names, excluded = compressible_linear_names(model)
        params = dict(model.named_parameters())
        originals = {name: params[name].detach().cpu().clone() for name in names}
        untouched = {
            name: p.detach().cpu().clone() for name, p in params.items() if name not in names
        }
        report["excluded"] = excluded
        report["untouched_unique_tensor_bytes"] = sum(
            x.numel() * x.element_size() for x in untouched.values()
        )
        c4 = plan_dataset_ref(plan, "calibration")
        wiki = plan_dataset_ref(plan, "development")
        calibration_docs = load_plan_texts(
            c4,
            split="train",
            dataset_config="en",
            max_documents=cfg["calibration_documents"],
            stream=True,
        )
        development_docs = load_plan_texts(
            wiki, split="validation", dataset_config="wikitext-2-raw-v1"
        )

        def hashes(texts: Sequence[str]) -> set[str]:
            return {hashlib.sha256(x.encode()).hexdigest() for x in texts}

        if hashes(calibration_docs) & hashes(development_docs):
            raise ValueError("exact document overlap in calibration and development")
        pool = _blocks(
            tokenizer,
            calibration_docs,
            length=cfg["calibration_sequence"],
            cap=cfg["calibration_pool_tokens"],
        )
        dev = _blocks(
            tokenizer,
            development_docs,
            length=cfg["evaluation_sequence"],
            cap=cfg["development_tokens"],
        )
        report["data"] = {
            "calibration": {
                "id": c4.name,
                "revision": c4.revision,
                "split": "train",
                "sha256": hashlib.sha256("\n\n".join(calibration_docs).encode()).hexdigest(),
            },
            "development": {
                "id": wiki.name,
                "revision": wiki.revision,
                "split": "validation",
                "sha256": hashlib.sha256("\n\n".join(development_docs).encode()).hexdigest(),
            },
            "calibration_pool_blocks": len(pool),
            "exact_calibration_development_overlap": 0,
        }
        report["fp16_development"] = _evaluate(model, dev, deadline=deadline)
        cache: dict[str, dict[tuple[int, int], Candidate]] = {}
        build_started = time.monotonic()
        for i, name in enumerate(names):
            if time.monotonic() >= deadline - 1200:
                raise TimeoutError("insufficient reserved time for score/evaluation/export")
            cache[name] = build_candidates(
                originals[name], bits=cfg["bits"], ranks=cfg["ranks"], svd_device="cuda"
            )
            for option in cache[name].values():
                option.weight = option.weight.half()
            if i == 1:
                build_projection = (time.monotonic() - build_started) / 2 * len(names)
                report["projected_build_seconds_approximate"] = build_projection
                if build_projection > (deadline - time.monotonic() - 1200) / 2:
                    raise TimeoutError("early build projection leaves insufficient scoring reserve")
            if i % 14 == 0:
                print(
                    f"candidate build {i + 1}/{len(names)} elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )
        report["build_seconds"] = time.monotonic() - build_started
        budget = sum(cache[name][4, 0].payload_bytes for name in names)
        report["target_payload_budget_bytes"] = budget

        def install(allocation: Mapping[str, Choice] | None) -> None:
            with torch.no_grad():
                for name in names:
                    key = (
                        (4, 0)
                        if allocation is None
                        else (allocation[name].bits, allocation[name].rank)
                    )
                    params[name].copy_(cache[name][key].weight.to(params[name]))

        install(None)
        report["uniform4_development"] = _evaluate(model, dev, deadline=deadline)
        allocations: dict[str, dict[str, Choice]] = {}
        uniform = {name: Choice(4, 0, cache[name][4, 0].payload_bytes, 0.0) for name in names}
        allocations["uniform4"] = uniform
        for seed_index, seed in enumerate(cfg["seeds"]):
            install(None)
            rng = random.Random(seed)
            indices = rng.sample(range(len(pool)), cfg["calibration_blocks"])
            blocks = [pool[i] for i in indices]
            baseline = _evaluate(model, blocks, deadline=deadline)
            choices: dict[str, list[Choice]] = {}
            score_started = time.monotonic()
            for i, name in enumerate(names):
                choices[name] = []
                for (bits, rank), candidate in cache[name].items():
                    if (bits, rank) == (4, 0):
                        measured = baseline
                    else:
                        with temporary_weight(params[name], candidate.weight):
                            measured = _evaluate(model, blocks, deadline=deadline)
                    damage = measured["nll"] - baseline["nll"]
                    choice = Choice(bits, rank, candidate.payload_bytes, damage)
                    choices[name].append(choice)
                    fro = float(
                        (originals[name].float() - candidate.weight.float()).square().sum()
                        / originals[name].float().square().sum()
                    )
                    score_rows.append(
                        {
                            "seed": seed,
                            "layer": name,
                            **asdict(choice),
                            "relative_fro_squared": fro,
                            "calibration_indices": indices,
                        }
                    )
                if i == 1:
                    projected = (
                        (time.monotonic() - score_started)
                        / 2
                        * len(names)
                        * (len(cfg["seeds"]) - seed_index)
                    )
                    report["projected_scoring_seconds"] = projected
                    if projected > deadline - time.monotonic() - 1200:
                        raise TimeoutError(
                            "preflight projects scoring beyond budget; do not spend remaining quota"
                        )
                if i % 14 == 0:
                    print(
                        f"seed={seed} scored={i + 1}/{len(names)} elapsed={time.monotonic() - started:.1f}s",
                        flush=True,
                    )
                    _save(out / "scores.json", score_rows)
            for family in ("bits", "residual"):
                available = {
                    name: [x for x in xs if family == "residual" or x.rank == 0]
                    for name, xs in choices.items()
                }
                allocation = choose_allocation(available, budget)
                arm = f"{family}_seed{seed}"
                allocations[arm] = allocation
                install(allocation)
                report["arms"][arm] = {
                    "development": _evaluate(model, dev, deadline=deadline),
                    "surrogate_delta_nll": sum(x.damage for x in allocation.values()),
                    "target_payload_bytes": sum(x.payload_bytes for x in allocation.values()),
                    "allocation": {name: asdict(x) for name, x in allocation.items()},
                }
            report["stages"].append(
                {"calibration_seed": seed, "seconds": time.monotonic() - score_started}
            )
            _save(out / "metrics.json", report)
        _save(out / "scores.json", score_rows)
        means = {
            family: float(
                np.mean(
                    [
                        report["arms"][f"{family}_seed{s}"]["development"]["nll"]
                        for s in cfg["seeds"]
                    ]
                )
            )
            for family in ("bits", "residual")
        }
        means["uniform4"] = report["uniform4_development"]["nll"]
        winner = min(means, key=lambda key: means[key])
        if means["uniform4"] - means[winner] < cfg["minimum_development_nll_gain"]:
            winner = "uniform4"
        report["selection_frozen_before_test"] = {
            "family": winner,
            "mean_development_nll": means,
            "minimum_gain": cfg["minimum_development_nll_gain"],
        }
        _save(out / "metrics.json", report)
        # First and only test read in this experiment, after allocation and selection freeze.
        test_docs = load_plan_texts(wiki, split="test", dataset_config="wikitext-2-raw-v1")
        if hashes(test_docs) & (hashes(calibration_docs) | hashes(development_docs)):
            raise ValueError("exact test document overlap detected")
        test = _blocks(
            tokenizer, test_docs, length=cfg["evaluation_sequence"], cap=cfg["test_tokens"]
        )
        report["data"]["test"] = {
            "id": wiki.name,
            "revision": wiki.revision,
            "split": "test",
            "sha256": hashlib.sha256("\n\n".join(test_docs).encode()).hexdigest(),
            "prior_exposure": "WikiText-2 test used in previous project phases; phase2 is exploratory, not a new blind benchmark",
        }
        with torch.no_grad():
            for name in names:
                params[name].copy_(originals[name].to(params[name]))
        report["fp16_test"] = _evaluate(model, test, deadline=deadline)
        for arm, allocation in allocations.items():
            install(allocation)
            evaluated = _evaluate(model, test, deadline=deadline)
            report["arms"].setdefault(arm, {})["test"] = evaluated
            # Full reproducible checkpoint: actual packed payload + layout + all
            # unique unchanged parameters. Evaluated weights came from these blobs.
            checkpoint = out / "checkpoints" / arm
            checkpoint.mkdir(parents=True, exist_ok=True)
            layout: list[dict[str, Any]] = []
            offset = 0
            with (checkpoint / "weights.sqpack").open("wb") as stream:
                for name, choice in allocation.items():
                    candidate = cache[name][choice.bits, choice.rank]
                    stream.write(candidate.blob)
                    layout.append(
                        {
                            "layer": name,
                            "bits": choice.bits,
                            "rank": choice.rank,
                            "offset": offset,
                            "bytes": candidate.payload_bytes,
                            "shapes": candidate.shapes,
                            "quant_spec": asdict(candidate.spec),
                        }
                    )
                    offset += candidate.payload_bytes
            torch.save(untouched, checkpoint / "unchanged.pt")
            _save(
                checkpoint / "layout.json",
                {
                    "model": report["model"],
                    "targets": layout,
                    "tied_weights": "restore model architecture from pinned config before loading unique named_parameters; tied lm_head remains tied",
                },
            )
            files = {
                p.name: {
                    "bytes": p.stat().st_size,
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                }
                for p in checkpoint.iterdir()
                if p.is_file()
            }
            report["arms"][arm]["storage"] = {
                "target_payload_bytes": offset,
                "full_checkpoint_bytes": sum(v["bytes"] for v in files.values()),
                "files": files,
                "scope": "weights + layout + unique unchanged parameters; tokenizer/runtime environment excluded",
            }
            assert offset <= budget
            _save(out / "metrics.json", report)
        baseline_blocks = report["arms"]["uniform4"]["test"]["blocks"]
        for arm in allocations:
            if arm != "uniform4":
                report["arms"][arm]["paired_vs_uniform4"] = paired_nll_summary(
                    baseline_blocks, report["arms"][arm]["test"]["blocks"], seed=20261011
                )
        report["status"] = "success"
        report["stages"].append({"test_frozen": True, "no_training_performed": True})
    except Exception as exc:
        report["status"] = "failed"
        report["failure_reason"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        report["gpu_device_hours_upper_bound"] = (
            report["wall_seconds"] / 3600 * report.get("gpu", {}).get("visible_count", 0)
        )
        _save(out / "metrics.json", report)
        _save(out / "scores.json", score_rows)
        _write_study_manifest(report, config_path, out / "run_manifest.json")
    return report


def _write_study_manifest(report: Mapping[str, Any], config_path: str | Path, target: Path) -> None:
    """Schema-valid aggregate provenance; class labels remain inside per-arm evidence."""
    from spectraquant.reporting.environment import collect_hardware, collect_software
    from spectraquant.reporting.manifests import (
        HardwareBlock,
        RunManifest,
        TrainingBlock,
        utc_timestamp,
        write_manifest,
    )

    data = list(report.get("data", {}).values())
    refs = [x for x in data if isinstance(x, dict) and "revision" in x]
    document = RunManifest(
        run_id="phase2-global-damage",
        timestamp_utc=report.get("started_utc", utc_timestamp()),
        git_commit=report["git"]["commit"],
        git_dirty=report["git"]["dirty"],
        config_path=str(Path(config_path).resolve()),
        resolved_config={"phase2": report["config"]},
        model_id=report.get("model", {}).get("id"),
        model_revision=report.get("model", {}).get("revision"),
        dataset_ids=[x["id"] for x in refs],
        dataset_revisions=[x["revision"] for x in refs],
        dataset_checksums=["sha256:" + x["sha256"] for x in refs],
        split="calibration/train, development/validation, evaluation/test",
        seed=0,
        hardware=HardwareBlock.model_validate(collect_hardware()),
        software=collect_software(),
        runtime_backend="torch-cuda-fp16" if "gpu" in report else "torch-cpu-fp32",
        training=TrainingBlock(steps=0, tokens=0, wall_time_s=0.0, peak_mem_mb=None),
        metrics={
            "phase2.status": report["status"],
            "phase2.quality_class": 2,
            "phase2.storage_class": 3,
            "phase2.full_report_path": "metrics.json",
            "phase2.full_report_sha256": "sha256:"
            + hashlib.sha256((target.parent / "metrics.json").read_bytes()).hexdigest(),
            "phase2.runtime": report["runtime"],
            "phase2.wall_seconds": report["wall_seconds"],
            "phase2.gpu_device_hours_upper_bound": report["gpu_device_hours_upper_bound"],
            "phase2.seeds_interpretation": report["seeds_interpretation"],
            "phase2.arm_names": list(report["arms"]),
        },
        status=report["status"],
        failure_reason=report.get("failure_reason"),
        artifact_paths=["metrics.json", "scores.json", "checkpoints/"],
    )
    write_manifest(document, target)
