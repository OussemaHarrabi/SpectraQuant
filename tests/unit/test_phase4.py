"""Synthetic fixtures for the frozen transfer study; no research model loading."""

from types import SimpleNamespace

import pytest
import torch


def test_axis0_clipping_preserves_prior_packed_representation():
    from spectraquant.cloud.phase3 import clipped_candidate
    from spectraquant.cloud.phase4 import clip_candidate

    weight = torch.arange(70 * 8).reshape(70, 8).sin().half()
    old, old_alpha = clipped_candidate(weight, (1.0, 0.95, 0.9))
    new, new_alpha = clip_candidate(weight, (1.0, 0.95, 0.9), axis=0)
    assert new.blob == old.blob
    assert torch.equal(new_alpha, old_alpha)


def test_axis1_weighted_clipping_is_nontrivial_and_keeps_exact_bytes():
    from spectraquant.cloud.phase4 import clip_candidate

    weight = torch.linspace(-1, 1, 64).reshape(1, 64)
    weight[0, 0] = 3
    factors = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75)
    plain, _ = clip_candidate(weight, (1.0,), axis=1)
    mse, alpha = clip_candidate(weight, factors, axis=1)
    equal, equal_alpha = clip_candidate(weight, factors, axis=1, second_moment=torch.ones(64))
    moments = torch.ones(64)
    moments[0] = 1e6
    weighted, weighted_alpha = clip_candidate(weight, factors, axis=1, second_moment=moments)
    assert mse.blob == equal.blob and torch.equal(alpha, equal_alpha)
    assert not torch.equal(alpha, weighted_alpha)
    assert plain.payload_bytes == mse.payload_bytes == weighted.payload_bytes
    assert ((weight.float() - mse.weight.half().float()) ** 2).sum() <= (
        (weight.float() - plain.weight.half().float()) ** 2
    ).sum()
    for invalid in (torch.full((64,), float("nan")), torch.full((64,), -1.0)):
        with pytest.raises(ValueError, match="moment"):
            clip_candidate(weight, factors, axis=1, second_moment=invalid)
    with pytest.raises(ValueError, match="axis1"):
        clip_candidate(weight, factors, axis=0, second_moment=moments)


def test_second_moment_hooks_match_direct_math_and_are_removed():
    from spectraquant.cloud.phase4 import activation_moments

    model = torch.nn.Sequential(torch.nn.Linear(3, 4, bias=False))
    x = torch.arange(24).reshape(2, 4, 3).float() / 5
    result = activation_moments(model, ["0.weight"], [x], device="cpu", deadline=float("inf"))
    assert torch.allclose(result["0.weight"], x.double().square().reshape(-1, 3).mean(0))
    assert not model[0]._forward_pre_hooks


def test_document_blocks_preserve_boundaries_and_exact_predicted_budget():
    from spectraquant.cloud.phase4 import document_blocks

    class Tokenizer:
        def __call__(self, text, **kwargs):
            return SimpleNamespace(input_ids=torch.tensor([[int(x) for x in text.split()]]))

    docs = [" ".join(map(str, range(1, 18))), " ".join(map(str, range(101, 118)))]
    blocks = document_blocks(Tokenizer(), docs, length=8, predicted_cap=20, document_cap=16)
    assert sum(x["input_ids"].numel() - 1 for x in blocks) == 20
    assert all(x["input_ids"].numel() <= 8 for x in blocks)
    for block in blocks:
        assert bool((block["input_ids"] < 100).all()) == (block["document_id"] == 0)
    with pytest.raises(ValueError, match="insufficient"):
        document_blocks(Tokenizer(), docs[:1], length=8, predicted_cap=30, document_cap=16)


def test_document_paired_summary_uses_clusters_and_rejects_misalignment():
    from spectraquant.cloud.phase4 import paired_documents

    baseline = [{"document_id": i, "tokens": 10, "nll_sum": 20.0} for i in range(5)]
    treatment = [{"document_id": i, "tokens": 10, "nll_sum": 19.0} for i in range(5)]
    result = paired_documents(baseline, treatment)
    assert result["delta_nll"] == pytest.approx(-0.1)
    assert result["document_clusters"] == 5
    assert result["ci95"] == pytest.approx([-0.1, -0.1])
    treatment[0]["tokens"] = 9
    with pytest.raises(ValueError, match="aligned"):
        paired_documents(baseline, treatment)


def test_transfer_refuses_local_research_and_saves_failure(tmp_path, monkeypatch):
    import json

    from spectraquant.cloud.phase4 import run_phase4

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"out": str(tmp_path), "max_wall_seconds": 30}))
    with pytest.raises(RuntimeError, match="cloud CUDA"):
        run_phase4(config, "fixture")
    metrics = json.loads((tmp_path / "phase4-transfer-fixture/metrics.json").read_text())
    assert metrics["status"] == "failed"
    manifest = json.loads((tmp_path / "phase4-transfer-fixture/run_manifest.json").read_text())
    assert manifest["metrics"]["phase4.status"] == "failed"


def test_synthetic_transfer_freezes_all_files_before_holdout_and_reloads(tmp_path, monkeypatch):
    import json
    import math

    from spectraquant import experiment_plan
    from spectraquant.cloud import phase4, plan_runner
    from spectraquant.evaluation.phase2_report import verify_checkpoints

    model = torch.nn.Linear(8, 8, bias=False).half()
    original = model.weight.detach().clone()
    out = tmp_path / "phase4-transfer-fixture"
    ref = SimpleNamespace(name="fixture/c4", revision="a" * 40, roles=["calibration"])
    pg = SimpleNamespace(name="fixture/pg", revision="b" * 40, roles=["test_perplexity"])
    plan = SimpleNamespace(datasets=[ref, pg])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "synthetic")
    monkeypatch.setattr(experiment_plan, "load_plan", lambda _: plan)
    monkeypatch.setattr(
        plan_runner,
        "load_model",
        lambda *a, **k: (model, Tokenizer(), "fixture/model", "c" * 40, "fixture"),
    )
    monkeypatch.setattr(phase4, "_arm_wall_timer", lambda _: lambda: None)
    monkeypatch.setattr(phase4.shutil, "disk_usage", lambda _: SimpleNamespace(free=10**12))
    monkeypatch.setattr(phase4, "activation_moments", lambda *a, **k: {"weight": torch.ones(8)})

    class Tokenizer:
        def __call__(self, text, **kwargs):
            cap = kwargs.get("max_length", 128)
            return SimpleNamespace(input_ids=torch.arange(cap)[None])

    def texts(ref, *, split, count, skip=0):
        if split != "train":
            frozen = json.loads((out / "frozen.json").read_text())
            assert len(frozen["arms"]) == 7
            assert (
                verify_checkpoints(
                    {"arms": {a: {"storage": s} for a, s in frozen["arms"].items()}}, out
                )["verified_files"]
                == 21
            )
        return [f"{ref.name}-{split}-document{i}" for i in range(count)]

    monkeypatch.setattr(phase4, "_load_documents", texts)
    eval_calls = []

    def evaluate(model, blocks, *, deadline):
        if len(eval_calls) < 2:
            assert torch.equal(model.weight, original)
        else:
            # Quantized inference has actually installed reloaded weights.
            assert not torch.equal(model.weight, original)
        eval_calls.append(1)
        records = [{"tokens": x.numel() - 1, "nll_sum": (x.numel() - 1) * 2.0} for x in blocks]
        return {
            "nll": 2.0,
            "perplexity": math.exp(2),
            "tokens": sum(x["tokens"] for x in records),
            "blocks": records,
        }

    monkeypatch.setattr(phase4, "_evaluate", evaluate)
    config = {
        "out": str(tmp_path),
        "max_wall_seconds": 2000,
        "models": {"fixture": {"plan": "fixture"}},
        "calibration_documents": 2,
        "calibration_sequence": 4,
        "calibration_pool_tokens": 128,
        "calibration_blocks": 4,
        "seeds": [0, 1, 2],
        "clip_factors": [1.0, 0.9],
        "evaluation_sequence": 8,
        "predicted_tokens": 30,
        "document_input_cap": 16,
        "minimum_documents": 2,
        "evaluation": {
            "c4": {"dataset": ref.name, "split": "validation", "documents": 4, "skip": 0},
            "pg": {"dataset": pg.name, "split": "test", "documents": 4, "skip": 0},
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    report = phase4.run_phase4(config_path, "fixture")
    assert report["status"] == "success"
    assert verify_checkpoints(report, out)["verified_files"] == 21
    assert len(eval_calls) == 16
    assert all(r["tokens"] == 30 for a in report["arms"].values() for r in a["tests"].values())
    assert "phase4" in json.loads((out / "run_manifest.json").read_text())["resolved_config"]
