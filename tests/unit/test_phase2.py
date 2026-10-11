"""Phase-2 invariants, exercised only on tiny synthetic CPU fixtures."""

import json

import pytest
import torch

from spectraquant.cloud.phase2 import (
    Choice,
    build_candidates,
    choose_allocation,
    paired_nll_summary,
    run_phase2,
    temporary_weight,
)
from spectraquant.quantization import QuantSpec, accounted_bytes


def test_dense_option_and_residual_are_reloaded_from_real_storage():
    weight = torch.arange(64, dtype=torch.float32).reshape(8, 8).sin()
    options = build_candidates(weight, bits=(3, 4, 8), ranks=(0, 2), svd_device="cpu")
    assert set(options) == {(b, r) for b in (3, 4, 8) for r in (0, 2)}
    q = options[4, 0]
    residual = options[4, 2]
    assert q.payload_bytes == accounted_bytes(weight.shape, QuantSpec(4, "per_group", 32, True))
    assert residual.payload_bytes > q.payload_bytes
    assert q.reload_max_abs == 0.0
    assert residual.reload_max_abs == 0.0
    assert torch.isfinite(residual.weight).all()
    assert not torch.equal(q.weight, residual.weight)


def test_solver_matches_exhaustive_choices_with_negative_damages():
    choices = {
        "a": [Choice(4, 0, 10, 0.0), Choice(3, 0, 6, 0.4), Choice(4, 2, 14, -0.7)],
        "b": [Choice(4, 0, 10, 0.0), Choice(3, 0, 6, 0.2), Choice(8, 0, 20, -0.1)],
    }
    chosen = choose_allocation(choices, 20)
    assert sum(x.payload_bytes for x in chosen.values()) <= 20
    assert sum(x.damage for x in chosen.values()) == pytest.approx(-0.5)
    assert chosen["a"].rank == 2


def test_infeasible_budget_fails_instead_of_falling_back():
    with pytest.raises(ValueError, match="infeasible"):
        choose_allocation({"a": [Choice(4, 0, 10, 0.0)]}, 9)


def test_weight_restores_even_when_measurement_raises():
    layer = torch.nn.Linear(3, 2, bias=False)
    original = layer.weight.detach().clone()
    with pytest.raises(RuntimeError), temporary_weight(layer.weight, torch.zeros_like(original)):
        assert layer.weight.count_nonzero() == 0
        raise RuntimeError("fixture failure")
    assert torch.equal(layer.weight, original)


def test_paired_interval_is_over_blocks_and_accounts_for_token_counts():
    a = [{"nll_sum": 4.0, "tokens": 2}, {"nll_sum": 12.0, "tokens": 4}]
    b = [{"nll_sum": 2.0, "tokens": 2}, {"nll_sum": 8.0, "tokens": 4}]
    result = paired_nll_summary(a, b, seed=5)
    assert result["delta_nll"] == pytest.approx(-1.0)
    assert result["ci95"] == pytest.approx([-1.0, -1.0])
    assert result["unit"] == "paired contiguous token blocks; not independent model seeds"


def test_cloud_runner_refuses_cpu_and_preserves_failure_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = tmp_path / "run.json"
    config.write_text(json.dumps({"out": str(tmp_path / "out"), "max_wall_seconds": 30}))
    with pytest.raises(RuntimeError, match="cloud CUDA"):
        run_phase2(config)
    metrics = json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert metrics["status"] == "failed"
    assert "no CPU fallback" in metrics["failure_reason"]
    manifest = json.loads((tmp_path / "out" / "run_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["resolved_config"]["phase2"]["max_wall_seconds"] == 30
    assert json.loads((tmp_path / "out" / "scores.json").read_text()) == []
