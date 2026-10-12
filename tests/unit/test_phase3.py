"""Bounded follow-up invariants; synthetic CPU weights only."""

import json

import pytest
import torch

from spectraquant.cloud.phase2 import Choice, build_candidates, choose_allocation


def test_change_cap_prevents_coupled_budget_swaps():
    options = {
        "a": [Choice(4, 0, 10, 0), Choice(3, 0, 6, 0.1)],
        "b": [Choice(4, 0, 10, 0), Choice(4, 2, 14, -0.7)],
    }
    for cap in (0, 1):
        selected = choose_allocation(options, 20, max_changes=cap)
        assert all((c.bits, c.rank) == (4, 0) for c in selected.values())
    selected = choose_allocation(options, 20, max_changes=2)
    assert sum(c.damage for c in selected.values()) == pytest.approx(-0.6)
    with pytest.raises(ValueError, match="nonnegative"):
        choose_allocation(options, 20, max_changes=-1)


def test_group_clipping_has_exact_payload_parity_and_no_worse_reload_mse():
    from spectraquant.cloud.phase3 import clipped_candidate

    weight = torch.linspace(-0.8, 0.8, 70 * 7).reshape(70, 7)
    weight[0, :] = 2.5
    plain = build_candidates(weight, bits=(4,), ranks=(0,))[4, 0]
    clipped, choices = clipped_candidate(weight, (1.0, 0.95, 0.9, 0.85, 0.8, 0.75))
    assert clipped.payload_bytes == plain.payload_bytes
    assert choices.shape == (3, 7)
    assert (
        float((weight - clipped.weight.half().float()).square().sum())
        <= float((weight - plain.weight.half().float()).square().sum()) + 1e-6
    )
    assert float((weight - clipped.weight).square().sum()) < float(
        (weight - plain.weight).square().sum()
    )
    with pytest.raises(ValueError, match="include 1"):
        clipped_candidate(weight, (0.8,))


def test_packed_checkpoint_reload_rejects_corrupt_input(tmp_path):
    from spectraquant.cloud.phase3 import read_checkpoint

    candidate = build_candidates(torch.eye(5), bits=(4,), ranks=(0,))[4, 0]
    layout = {
        "targets": [
            {
                "layer": "a",
                "offset": 0,
                "bytes": candidate.payload_bytes,
                "shapes": candidate.shapes,
                "quant_spec": {
                    "bits": 4,
                    "granularity": "per_group",
                    "group_size": 32,
                    "symmetric": True,
                    "axis": 0,
                },
                "rank": 0,
                "bits": 4,
            }
        ]
    }
    (tmp_path / "layout.json").write_text(json.dumps(layout))
    (tmp_path / "weights.sqpack").write_bytes(candidate.blob)
    loaded = read_checkpoint(tmp_path)
    assert torch.equal(loaded["a"].weight, candidate.weight)
    (tmp_path / "weights.sqpack").write_bytes(candidate.blob[:-1])
    with pytest.raises(ValueError, match="layout"):
        read_checkpoint(tmp_path)


def test_followup_refuses_cpu_and_preserves_manifest(tmp_path, monkeypatch):
    from spectraquant.cloud.phase3 import run_phase3

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"out": str(tmp_path / "out"), "max_wall_seconds": 30}))
    with pytest.raises(RuntimeError, match="cloud CUDA"):
        run_phase3(config)
    metrics = json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert metrics["status"] == "failed"
    manifest = json.loads((tmp_path / "out" / "run_manifest.json").read_text())
    assert "phase3" in manifest["resolved_config"]
    assert manifest["metrics"]["phase3.status"] == "failed"


def test_generic_collection_cannot_validate_missing_dynamic_checkpoint(tmp_path):
    import hashlib

    from spectraquant.cloud.collect import ValidationReport
    from spectraquant.evaluation.phase2_report import verify_collected_study

    study = tmp_path / "phase3-controlled-followup"
    study.mkdir()
    metrics = {
        "arms": {
            "residual_k8_seed0": {
                "storage": {"files": {"weights.sqpack": {"bytes": 10, "sha256": "f" * 64}}}
            }
        }
    }
    path = study / "metrics.json"
    path.write_text(json.dumps(metrics))
    validation = ValidationReport(
        run_id="fixture",
        state="validated",
        source_dir=str(tmp_path),
        checksum_verified=True,
        manifest={
            "metrics": {"phase3.full_report_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        },
    )
    result, report, integrity = verify_collected_study(
        validation, tmp_path, study_name="phase3-controlled-followup", study_key="phase3"
    )
    assert result.state == "rejected"
    assert result.checksum_verified is False
    assert "missing file" in result.reasons[0]
    assert report is None and integrity is None


def test_synthetic_followup_freezes_before_holdout_and_saves_reloadable_final_arms(
    tmp_path, monkeypatch
):
    """Exercise orchestration with a tiny synthetic layer; no research model/GPU."""
    import hashlib
    import math
    import random
    from dataclasses import asdict
    from types import SimpleNamespace

    from spectraquant import experiment_plan
    from spectraquant.cloud import phase3, plan_data, plan_runner
    from spectraquant.evaluation.phase2_report import verify_checkpoints

    layer = torch.nn.Linear(8, 8, bias=False).half()
    with torch.no_grad():
        layer.weight.copy_(torch.arange(64).reshape(8, 8).sin())
    original = layer.weight.detach().clone()
    candidates = build_candidates(original, bits=(3, 4, 8), ranks=(0,))
    source_dir = tmp_path / "input/phase2-global-damage"
    pin = {"id": "fixture/model", "revision": "a" * 40}
    storage = phase3.save_checkpoint(
        source_dir / "checkpoints/uniform4", {"weight": candidates[4, 0]}, {}, pin
    )
    other_storage = phase3.save_checkpoint(
        source_dir / "checkpoints/bits_seed0", {"weight": candidates[4, 0]}, {}, pin
    )
    train_docs, dev_docs, holdout_docs = (
        ["training document"],
        ["development document"],
        ["new heldout document"],
    )

    def digest(texts):
        return hashlib.sha256("\n\n".join(texts).encode()).hexdigest()

    source_report = {
        "model": pin,
        "target_payload_budget_bytes": candidates[4, 0].payload_bytes,
        "config": {
            "calibration_documents": 1,
            "calibration_sequence": 4,
            "calibration_pool_tokens": 128,
            "calibration_blocks": 4,
            "evaluation_sequence": 8,
            "development_tokens": 64,
        },
        "data": {
            "calibration": {
                "id": "fixture/c4",
                "revision": "b" * 40,
                "split": "train",
                "sha256": digest(train_docs),
            },
            "development": {
                "id": "fixture/wiki",
                "revision": "c" * 40,
                "split": "validation",
                "sha256": digest(dev_docs),
            },
        },
        "uniform4_development": {"nll": 2.0},
        "arms": {
            "uniform4": {"storage": storage},
            "bits_seed0": {
                "storage": other_storage,
                "surrogate_delta_nll": 0.0,
                "allocation": {"weight": asdict(Choice(4, 0, candidates[4, 0].payload_bytes, 0.0))},
            },
        },
    }
    metrics_path = source_dir / "metrics.json"
    metrics_path.write_text(json.dumps(source_report))
    rows = [
        {
            "seed": seed,
            "layer": "weight",
            **asdict(Choice(bits, 0, c.payload_bytes, 0.0)),
            "calibration_indices": random.Random(seed).sample(range(32), 4),
        }
        for seed in (0, 1, 2)
        for (bits, _), c in candidates.items()
    ]
    scores_path = source_dir / "scores.json"
    scores_path.write_text(json.dumps(rows))
    out = tmp_path / "out"
    config = {
        "plan": "fixture",
        "out": str(out),
        "max_wall_seconds": 1000,
        "input_root": str(tmp_path / "input"),
        "bits": [3, 4, 8],
        "ranks": [0],
        "seeds": [0, 1, 2],
        "change_caps": [1],
        "clip_factors": [1.0, 0.9],
        "diagnostic_train_blocks": 16,
        "heldout_documents": 1,
        "heldout_tokens": 64,
        "minimum_development_nll_gain": 0.005,
        "source_metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
        "source_scores_sha256": hashlib.sha256(scores_path.read_bytes()).hexdigest(),
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "synthetic-no-GPU")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(phase3, "_arm_wall_timer", lambda _: lambda: None)
    monkeypatch.setattr(experiment_plan, "load_plan", lambda _: object())
    monkeypatch.setattr(
        plan_data,
        "plan_dataset_ref",
        lambda _, role: SimpleNamespace(
            name="fixture/c4" if role == "calibration" else "fixture/wiki",
            revision="b" * 40 if role == "calibration" else "c" * 40,
        ),
    )
    reads = []

    def load_texts(ref, *, split, **_kwargs):
        reads.append((ref.name, split))
        if ref.name == "fixture/c4" and split == "validation":
            selection = json.loads((out / "selection.json").read_text())
            assert selection["family"] == "uniform4"
            return holdout_docs
        return train_docs if split == "train" else dev_docs

    monkeypatch.setattr(plan_data, "load_plan_texts", load_texts)

    def tokenizer(*_args, **_kwargs):
        return SimpleNamespace(input_ids=torch.arange(128)[None])

    monkeypatch.setattr(
        plan_runner,
        "load_model",
        lambda *_args, **_kwargs: (layer, tokenizer, pin["id"], pin["revision"], ""),
    )
    monkeypatch.setattr(plan_runner, "compressible_linear_names", lambda _: (["weight"], []))
    monkeypatch.setattr(
        phase3,
        "build_candidates",
        lambda w, **kwargs: build_candidates(w, bits=kwargs["bits"], ranks=kwargs["ranks"]),
    )

    def evaluate(_model, blocks, **_kwargs):
        records = [{"tokens": b.numel() - 1, "nll_sum": 2.0 * (b.numel() - 1)} for b in blocks]
        return {
            "nll": 2.0,
            "perplexity": math.exp(2.0),
            "tokens": sum(r["tokens"] for r in records),
            "blocks": records,
        }

    monkeypatch.setattr(phase3, "_evaluate", evaluate)
    monkeypatch.setattr(phase3, "_write_study_manifest", lambda *_args: None)
    result = phase3.run_phase3(config_path)
    assert result["status"] == "success"
    assert set(result["arms"]) == {"uniform4", "mseclip"}
    assert result["data"]["test"]["split"] == "validation"
    assert reads[-1] == ("fixture/c4", "validation")
    assert not any(split == "test" for _, split in reads)
    assert verify_checkpoints(result, out)["verified_files"] == 6
