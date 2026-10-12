"""A-0025: fixed clipping criteria, grouping controls and heldout transfer.

Real-model execution is cloud CUDA only. Mathematical helpers support CPU fixtures.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from spectraquant.cloud.phase2 import Candidate, _blocks, _evaluate, _save, _write_study_manifest
from spectraquant.cloud.phase3 import _arm_wall_timer, read_checkpoint, save_checkpoint
from spectraquant.quantization import QuantSpec, deserialize_state, serialize_state


def clip_candidate(
    weight: Tensor,
    factors: Sequence[float],
    *,
    axis: int,
    second_moment: Tensor | None = None,
) -> tuple[Candidate, Tensor]:
    """Group32 INT4 clipping scored after exact reload and FP16 rounding.

    W:(out,in). Axis0 groups output rows per input column; axis1 groups inputs
    per output row. Diagonal input moments(in,) only affect axis1 minimizers.
    Packing/scoring is CPU FP32/FP64; return CPU effective weights and alphas.
    """
    if axis not in (0, 1) or not factors or factors[0] != 1.0:
        raise ValueError("axis0/1 and alpha1 first are required")
    if any(not math.isfinite(x) or not 0 < x <= 1 for x in factors):
        raise ValueError("clipping factors must lie in (0,1]")
    w = weight.detach().cpu().float()
    if w.ndim != 2 or not w.numel() or not bool(torch.isfinite(w).all()):
        raise ValueError("nonempty finite matrix required")
    if second_moment is not None:
        if axis != 1:
            raise ValueError("diagonal activation weighting requires axis1")
        second_moment = second_moment.detach().cpu().double()
        if (
            second_moment.shape != (w.shape[1],)
            or not bool(torch.isfinite(second_moment).all())
            or bool((second_moment < 0).any())
        ):
            raise ValueError("finite nonnegative input second moment required")
    oriented = w if axis == 0 else w.T
    groups = math.ceil(oriented.shape[0] / 32)
    bounds = torch.stack([oriented[i * 32 : (i + 1) * 32].abs().amax(0) for i in range(groups)])
    selected = torch.ones_like(bounds)
    best = torch.full_like(bounds, float("inf"), dtype=torch.float64)
    spec = QuantSpec(4, "per_group", 32, True, axis=axis)
    shapes = {"base": tuple(w.shape)}

    def pack(alphas: Tensor) -> Candidate:
        threshold = (bounds * alphas).repeat_interleave(32, 0)[: oriented.shape[0]]
        clipped = torch.maximum(torch.minimum(oriented, threshold), -threshold)
        source = clipped if axis == 0 else clipped.T.contiguous()
        blob = serialize_state({"base": source}, {"base": spec})
        q = deserialize_state(blob, shapes, {"base": spec})["base"]
        return Candidate(q, blob, shapes, spec, 0.0)

    for alpha in factors:
        candidate = pack(torch.full_like(bounds, alpha))
        error = (w - candidate.weight.half().float()).square()
        # Keep the exact Phase3 unweighted FP32 summation convention for axis0.
        if second_moment is not None:
            error = error.double() * second_moment[None, :]
        oriented_error = error if axis == 0 else error.T
        losses = torch.stack(
            [oriented_error[i * 32 : (i + 1) * 32].sum(0) for i in range(groups)]
        ).double()
        improve = losses < best
        selected = torch.where(improve, alpha, selected)
        best = torch.where(improve, losses, best)
    return pack(selected), selected


def activation_moments(
    model: Any, names: Sequence[str], blocks: Sequence[Tensor], *, device: str, deadline: float
) -> dict[str, Tensor]:
    """FP16 model inputs, exact streamed FP64 sum of squared input features.

    No labels, gradients or evaluation text. Hooks are always removed. Integer
    blocks call HF causal-LM forward; floating blocks are synthetic fixture inputs.
    """
    modules = dict(model.named_modules())
    sums: dict[str, Tensor] = {}
    counts = dict.fromkeys(names, 0)
    handles = []

    def hook_for(name: str):
        def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            x = inputs[0].detach().cpu().double().reshape(-1, inputs[0].shape[-1])
            contribution = x.square().sum(0)
            sums[name] = sums.get(name, torch.zeros_like(contribution)) + contribution
            counts[name] += x.shape[0]

        return hook

    try:
        for name in names:
            module_name = name.removesuffix(".weight") if name != "weight" else ""
            handles.append(modules[module_name].register_forward_pre_hook(hook_for(name)))
        with torch.inference_mode():
            for block in blocks:
                if time.monotonic() >= deadline:
                    raise TimeoutError("activation capture deadline")
                x = block.to(device)
                if x.is_floating_point():
                    model(x)
                else:
                    model(input_ids=x, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if any(not counts[name] for name in names):
        raise ValueError("every target must receive calibration inputs")
    return {name: sums[name] / counts[name] for name in names}


def document_blocks(
    tokenizer: Any, texts: Sequence[str], *, length: int, predicted_cap: int, document_cap: int
) -> list[dict[str, Any]]:
    """Deterministic bounded windows within documents, exact shifted-token cap."""
    if length < 2 or predicted_cap < 1 or document_cap < 2:
        raise ValueError("positive token budget and nontrivial windows required")
    blocks = []
    remaining = predicted_cap
    for index, text in enumerate(texts):
        ids = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=document_cap
        ).input_ids[0][:document_cap]
        for offset in range(0, len(ids), length):
            part = ids[offset : offset + min(length, remaining + 1)]
            if len(part) < 2:
                continue
            blocks.append({"document_id": index, "input_ids": part[None]})
            remaining -= len(part) - 1
            if remaining == 0:
                return blocks
    raise ValueError(f"insufficient document tokens: {remaining} predicted tokens missing")


def paired_documents(
    reference: Sequence[Mapping[str, Any]],
    treatment: Sequence[Mapping[str, Any]],
    *,
    seed: int = 20261012,
) -> dict[str, Any]:
    """Paired document-cluster bootstrap, descriptive corpus uncertainty only."""
    if (
        not reference
        or len(reference) != len(treatment)
        or any(
            a["document_id"] != b["document_id"] or a["tokens"] != b["tokens"]
            for a, b in zip(reference, treatment, strict=True)
        )
    ):
        raise ValueError("aligned nonempty document clusters required")
    tokens = np.array([r["tokens"] for r in reference], dtype=np.float64)
    if np.any(tokens <= 0):
        raise ValueError("positive document token counts required")
    deltas = np.array(
        [b["nll_sum"] - a["nll_sum"] for a, b in zip(reference, treatment, strict=True)]
    )
    indices = np.random.default_rng(seed).integers(0, len(tokens), size=(2500, len(tokens)))
    samples = deltas[indices].sum(1) / tokens[indices].sum(1)
    return {
        "delta_nll": float(deltas.sum() / tokens.sum()),
        "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "document_clusters": len(tokens),
        "unit": "paired document clusters, not model/training seeds",
        "limitations": "descriptive corpus sampling; documents/authors may depend; no publication significance claim",
    }


def _load_documents(ref: Any, *, split: str, count: int, skip: int = 0) -> list[str]:
    """Cloud-only bounded payload loading from immutable declarations.

    PG19 reads only the requested test books from the publisher's storage,
    avoiding the legacy loader's full train download. Payload hashes are recorded.
    """
    from itertools import islice
    from urllib.request import urlopen

    from spectraquant.cloud.plan_data import require_datasets, verify_pinned_revision

    verify_pinned_revision(ref.name, ref.revision, repo_type="dataset")
    if ref.name == "deepmind/pg19":
        import re

        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            ref.name, f"data/{split}_files.txt", repo_type="dataset", revision=ref.revision
        )
        files = Path(path).read_text().splitlines()[skip : skip + count]
        if len(files) != count or any(not re.fullmatch(r"test/\d+\.txt", x) for x in files):
            raise ValueError("invalid or insufficient pinned PG19 test file list")
        result = []
        for relative in files:
            with urlopen(
                "https://storage.googleapis.com/deepmind-gutenberg/" + relative, timeout=45
            ) as response:
                result.append(response.read().decode("utf-8"))
        return result
    dataset = require_datasets().load_dataset(
        ref.name,
        name=ref.config,
        revision=ref.revision,
        split=split,
        streaming=True,
        trust_remote_code=True,
    )
    texts = [str(row["text"]) for row in islice(dataset, skip, skip + count)]
    if len(texts) != count or any(not t.strip() for t in texts):
        raise ValueError("insufficient or empty frozen documents")
    return texts


def _document_eval(
    model: Any, blocks: Sequence[Mapping[str, Any]], *, deadline: float
) -> dict[str, Any]:
    evaluated = _evaluate(model, [x["input_ids"] for x in blocks], deadline=deadline)
    documents: dict[int, dict[str, Any]] = {}
    for block, value in zip(blocks, evaluated["blocks"], strict=True):
        identifier = block["document_id"]
        row = documents.setdefault(
            identifier, {"document_id": identifier, "tokens": 0, "nll_sum": 0.0}
        )
        row["tokens"] += value["tokens"]
        row["nll_sum"] += value["nll_sum"]
    evaluated["documents"] = list(documents.values())
    return evaluated


def _fingerprint(texts: Sequence[str]) -> str:
    # Length framing avoids ambiguous document-concatenation hashes.
    digest = hashlib.sha256()
    for text in texts:
        raw = text.encode("utf-8")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def run_phase4(config_path: str | Path, model_key: str) -> dict[str, Any]:
    """Freeze all stored arms before test loading, preserving partial failures."""
    from spectraquant.cloud.plan_data import plan_dataset_ref
    from spectraquant.cloud.plan_runner import compressible_linear_names, load_model
    from spectraquant.experiment_plan import load_plan
    from spectraquant.reporting.gitinfo import git_info

    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    out = Path(cfg["out"]) / f"phase4-transfer-{model_key}"
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + cfg["max_wall_seconds"]
    report: dict[str, Any] = {
        "study_key": "phase4",
        "status": "running",
        "model_key": model_key,
        "config": {**cfg, "selected_model_key": model_key},
        "started_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git": asdict(git_info(Path.cwd())),
        "quality_class": 2,
        "storage_class": 3,
        "runtime": "dense FP16 reconstruction of actual packed files; no low-bit kernel",
        "seeds_interpretation": "three calibration subsets only; one deterministic primary clipping comparison",
        "arms": {},
        "data": {},
        "stages": [],
    }
    cancel = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("phase4 research execution requires cloud CUDA; no CPU fallback")
        cancel = _arm_wall_timer(max(0.01, deadline - time.monotonic()))
        torch.set_num_threads(1)
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        report["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "visible_count": torch.cuda.device_count(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        }
        plan = load_plan(cfg["models"][model_key]["plan"])
        model, tokenizer, model_id, revision, _ = load_model(plan, device="cuda", dtype="float16")
        report["model"] = {
            "id": model_id,
            "revision": revision,
            "parameter_count": sum(p.numel() for p in model.parameters()),
        }
        names, excluded = compressible_linear_names(model)
        report["excluded"] = excluded
        params = dict(model.named_parameters())
        originals = {name: params[name].detach().cpu().clone() for name in names}
        untouched = {
            name: p.detach().cpu().clone() for name, p in params.items() if name not in names
        }
        estimated_payload = sum(
            (w.numel() + 1) // 2
            + 4
            * max(math.ceil(w.shape[0] / 32) * w.shape[1], math.ceil(w.shape[1] / 32) * w.shape[0])
            + 256
            for w in originals.values()
        )
        estimated_full = (
            estimated_payload
            + sum(w.numel() * w.element_size() for w in untouched.values())
            + 2_000_000
        )
        # Runner, outer-manifest staging and final export coexist as three copies.
        required_free = 3 * 7 * estimated_full + 1024**3
        if shutil.disk_usage(out).free < required_free:
            raise RuntimeError(
                f"insufficient disk for seven checkpoints and export: need approximately{required_free}B"
            )
        c4 = plan_dataset_ref(plan, "calibration")
        train_docs = _load_documents(c4, split="train", count=cfg["calibration_documents"])
        report["data"]["calibration"] = {
            "id": c4.name,
            "revision": c4.revision,
            "split": "train",
            "sha256": _fingerprint(train_docs),
        }
        pool = _blocks(
            tokenizer,
            train_docs,
            length=cfg["calibration_sequence"],
            cap=cfg["calibration_pool_tokens"],
        )
        if sum(block.numel() for block in pool) != cfg["calibration_pool_tokens"]:
            raise ValueError("calibration pool does not reach its frozen token cap")
        moments = {}
        for seed in cfg["seeds"]:
            indices = random.Random(seed).sample(range(len(pool)), cfg["calibration_blocks"])
            moments[seed] = activation_moments(
                model, names, [pool[i] for i in indices], device="cuda", deadline=deadline
            )
            report.setdefault("calibration_indices", {})[str(seed)] = indices
        factors = cfg["clip_factors"]
        arms = [
            "rtn_axis0",
            "mse_axis0",
            "rtn_axis1",
            "mse_axis1",
            *[f"act_axis1_seed{s}" for s in cfg["seeds"]],
        ]
        cache: dict[str, dict[str, Candidate]] = {arm: {} for arm in arms}
        layer_records: list[dict[str, Any]] = []
        build_started = time.monotonic()
        for i, name in enumerate(names):
            if time.monotonic() >= deadline - 900:
                raise TimeoutError(
                    "insufficient evaluation/export reserve during candidate building"
                )
            weight = originals[name]
            configs = [
                ("rtn_axis0", 0, (1.0,), None),
                ("mse_axis0", 0, factors, None),
                ("rtn_axis1", 1, (1.0,), None),
                ("mse_axis1", 1, factors, None),
                *[(f"act_axis1_seed{s}", 1, factors, moments[s][name]) for s in cfg["seeds"]],
            ]
            centered = weight.float() - weight.float().mean()
            variance = float(centered.square().mean())
            row: dict[str, Any] = {
                "layer": name,
                "shape": list(weight.shape),
                "weight_kurtosis": float(centered.pow(4).mean()) / max(variance**2, 1e-30),
                "max_abs_over_rms": float(weight.abs().max())
                / max(float(weight.float().square().mean().sqrt()), 1e-15),
                "arms": {},
            }
            for arm, axis, grid, moment in configs:
                candidate, alpha = clip_candidate(weight, grid, axis=axis, second_moment=moment)
                candidate.weight = candidate.weight.half()
                cache[arm][name] = candidate
                error = (weight.float() - candidate.weight.float()).square()
                row["arms"][arm] = {
                    "payload_bytes": candidate.payload_bytes,
                    "weight_mse": float(error.mean()),
                    "diagonal_output_error_proxy": float(
                        (error.double() * moments[0][name][None, :]).sum(1).mean()
                    ),
                    "alpha_counts": {str(a): int((alpha == a).sum()) for a in factors},
                }
            layer_records.append(row)
            if (
                i == 1
                and (time.monotonic() - build_started) / 2 * len(names)
                > deadline - time.monotonic() - 900
            ):
                raise TimeoutError("candidate throughput projects beyond frozen compute cap")
            if i % 25 == 0:
                print(
                    f"phase4 {model_key} build{i + 1}/{len(names)} elapsed{time.monotonic() - started:.1f}s",
                    flush=True,
                )
                _save(out / "scores.json", layer_records)
        report["target_count"] = len(names)
        report["build_seconds"] = time.monotonic() - build_started
        for arm in arms:
            storage = save_checkpoint(
                out / "checkpoints" / arm, cache[arm], untouched, report["model"]
            )
            report["arms"][arm] = {"storage": storage, "tests": {}}
            _save(out / "metrics.json", report)
        for axis in (0, 1):
            base = report["arms"][f"rtn_axis{axis}"]["storage"]
            for arm in arms:
                if f"axis{axis}" in arm:
                    storage = report["arms"][arm]["storage"]
                    if any(
                        storage[k] != base[k]
                        for k in ("target_payload_bytes", "full_checkpoint_bytes")
                    ):
                        raise ValueError("same-axis checkpoint byte parity failed")
        report["arms_frozen_before_holdout"] = {
            "frozen_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "primary_comparison": ["rtn_axis0", "mse_axis0"],
            "arms": {arm: evidence["storage"] for arm, evidence in report["arms"].items()},
        }
        _save(out / "frozen.json", report["arms_frozen_before_holdout"])
        _save(out / "metrics.json", report)
        del cache, moments
        blocked_hashes = {hashlib.sha256(t.encode()).hexdigest() for t in train_docs}
        tests = {}
        refs = {d.name: d for d in plan.datasets}
        for corpus, definition in cfg["evaluation"].items():
            texts = _load_documents(
                refs[definition["dataset"]],
                split=definition["split"],
                count=definition["documents"],
                skip=definition["skip"],
            )
            hashes = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
            if len(set(hashes)) != len(hashes) or blocked_hashes.intersection(hashes):
                raise ValueError("duplicate or calibration-overlapping holdout document")
            blocked_hashes.update(hashes)
            blocks = document_blocks(
                tokenizer,
                texts,
                length=cfg["evaluation_sequence"],
                predicted_cap=cfg["predicted_tokens"],
                document_cap=cfg["document_input_cap"],
            )
            if len({b["document_id"] for b in blocks}) < cfg["minimum_documents"]:
                raise ValueError("too few heldout document clusters")
            tests[corpus] = blocks
            report["data"][corpus] = {
                "id": definition["dataset"],
                "revision": refs[definition["dataset"]].revision,
                "split": definition["split"],
                "sha256": _fingerprint(texts),
                "document_hashes": hashes,
                "row_selection": {"skip": definition["skip"], "count": definition["documents"]},
                "prior_exposure": "new project holdout slice; model pretraining exposure unknown",
            }
        report["fp16_tests"] = {
            corpus: _document_eval(model, blocks, deadline=deadline)
            for corpus, blocks in tests.items()
        }
        evaluation_started = time.monotonic()
        for i, arm in enumerate(arms):
            loaded = read_checkpoint(out / "checkpoints" / arm)
            with torch.no_grad():
                for name in names:
                    params[name].copy_(loaded[name].weight.to(params[name]))
            del loaded
            for corpus, blocks in tests.items():
                report["arms"][arm]["tests"][corpus] = _document_eval(
                    model, blocks, deadline=deadline
                )
                _save(out / "metrics.json", report)
            if (
                i == 0
                and (time.monotonic() - evaluation_started) * (len(arms) - 1)
                > deadline - time.monotonic() - 180
            ):
                raise TimeoutError("evaluation throughput exceeds remaining cap")
            print(f"phase4 {model_key} evaluated{arm}", flush=True)
        for arm, evidence in report["arms"].items():
            axis = 0 if "axis0" in arm else 1
            for corpus, result in evidence["tests"].items():
                reference = report["arms"][f"rtn_axis{axis}"]["tests"][corpus]
                result["paired_vs_same_axis_rtn"] = paired_documents(
                    reference["documents"], result["documents"]
                )
        report["status"] = "success"
    except Exception as exc:
        report["status"] = "failed"
        report["failure_reason"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if cancel is not None:
            cancel()
        report["wall_seconds"] = time.monotonic() - started
        report["gpu_device_hours_upper_bound"] = (
            report["wall_seconds"] / 3600 * report.get("gpu", {}).get("visible_count", 0)
        )
        _save(out / "metrics.json", report)
        _save(out / "scores.json", locals().get("layer_records", []))
        _write_study_manifest(report, config_path, out / "run_manifest.json")
    return report
