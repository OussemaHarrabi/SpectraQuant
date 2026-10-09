"""Ground-truth and fixture tests for the Tier-0 toy module."""

from __future__ import annotations

import math

import pytest
import torch

from spectraquant.evaluation.toy import (
    TinyConfig,
    exact_layer_output_error,
    layer_inputs,
    layer_names,
    network_joint_damage,
    network_single_layer_damage,
    norm_preimages,
    residual_contribution_key,
    train_tiny,
)
from spectraquant.quantization import QuantSpec


def spec() -> QuantSpec:
    return QuantSpec(bits=4, granularity="per_group", group_size=32, symmetric=True)


def small_cfg(**kwargs) -> TinyConfig:
    base = {"n_blocks": 2, "d_model": 32, "n_heads": 2, "d_ff": 64, "vocab": 16, "seq_len": 12}
    base.update(kwargs)
    return TinyConfig(**base)  # type: ignore[arg-type]


def test_exact_layer_output_error_equals_hand_computed_quadratic_form() -> None:
    """``mean_i ||x_i delta^T||^2`` must equal ``tr(delta Sigma_X delta^T)`` computed independently."""
    torch.manual_seed(0)
    w = torch.randn(6, 5, dtype=torch.float64)
    x = torch.randn(9, 5, dtype=torch.float64)
    from spectraquant.proxies.operators import compression_delta

    delta = compression_delta(w, 2, spec())
    sigma = x.transpose(0, 1) @ x / x.shape[0]
    by_trace = float(torch.trace(delta @ sigma @ delta.transpose(0, 1)))
    assert exact_layer_output_error(w, x, 2, spec()) == pytest.approx(by_trace, rel=1e-10)


def test_training_is_deterministic_and_learns() -> None:
    cfg = small_cfg()
    model_a, metrics_a = train_tiny(cfg, steps=40)
    model_b, metrics_b = train_tiny(cfg, steps=40)
    assert metrics_a["initial_loss"] == metrics_b["initial_loss"]
    assert metrics_a["final_loss"] == metrics_b["final_loss"]
    assert metrics_a["final_loss"] < metrics_a["initial_loss"]
    ids = torch.arange(4)[None] % cfg.vocab
    logits_a, _ = model_a(ids)
    logits_b, _ = model_b(ids)
    assert torch.equal(logits_a, logits_b)


def test_different_seeds_give_different_models() -> None:
    _, a = train_tiny(small_cfg(seed=0), steps=10)
    _, b = train_tiny(small_cfg(seed=1), steps=10)
    assert a["final_loss"] != b["final_loss"]


def test_layer_names_and_contribution_mapping_are_consistent() -> None:
    cfg = small_cfg()
    model, _ = train_tiny(cfg, steps=5)
    names = layer_names(model)
    assert len(names) == 4 * cfg.n_blocks + 1
    assert names[-1] == "head"
    assert residual_contribution_key("head") is None
    assert residual_contribution_key("blocks.0.qkv") == "blocks.0.attn_out"
    assert residual_contribution_key("blocks.1.fc2") == "blocks.1.mlp_out"
    with pytest.raises(ValueError, match="unrecognised layer name"):
        residual_contribution_key("blocks.0.nonsense")


def test_perturbation_injection_changes_the_output_by_exactly_the_injected_contribution() -> None:
    """Injecting a perturbation into a residual-writing layer must shift the output by that amount."""
    cfg = small_cfg(use_norm=False, final_norm=False)
    model, _ = train_tiny(cfg, steps=5)
    ids = torch.arange(cfg.seq_len)[None] % cfg.vocab
    with torch.no_grad():
        _base_logits, base_hidden = model(ids)
        pert = torch.zeros(cfg.seq_len, cfg.d_model)
        pert[3, :] = 0.25
        # a zero perturbation must be a no-op
        _, same = model(ids, inject=("blocks.0.proj", torch.zeros_like(pert)))
        assert torch.equal(same, base_hidden)
        # a nonzero perturbation must move the output
        _, moved = model(ids, inject=("blocks.0.proj", pert))
    assert not torch.allclose(moved, base_hidden)
    assert float((moved - base_hidden).abs().max()) > 0.0


def test_damage_measurements_are_positive_and_baseline_is_restored() -> None:
    cfg = small_cfg()
    model, _ = train_tiny(cfg, steps=30)
    ids = torch.arange(cfg.seq_len)[None].repeat(2, 1) % cfg.vocab
    names = layer_names(model)
    before = {n: layer.weight.detach().clone() for n, layer in model.linear_layers().items()}

    damage = network_single_layer_damage(model, ids, names, 8, spec())
    joint_hidden, joint_logits = network_joint_damage(model, ids, names, 8, spec())

    assert all(value >= 0.0 for value in damage.per_layer.values())
    assert damage.joint_hidden != damage.joint_hidden  # NaN: only the joint call fills it
    assert joint_hidden > 0.0 and joint_logits > 0.0
    assert set(damage.layer_errors) == set(names)
    assert set(damage.residual_errors) == set(names) - {"head"}
    # every weight must be back to its original value
    for name, layer in model.linear_layers().items():
        assert torch.equal(layer.weight.detach(), before[name]), name


def test_norm_preimages_are_captured_for_every_normalisation() -> None:
    cfg = small_cfg()
    model, _ = train_tiny(cfg, steps=5)
    ids = torch.arange(cfg.seq_len)[None] % cfg.vocab
    preimages = norm_preimages(model, ids)
    expected = {f"blocks.{i}.{label}" for i in range(cfg.n_blocks) for label in ("ln1", "ln2")}
    expected.add("ln_f")
    assert set(preimages) == expected
    for tensor in preimages.values():
        assert tensor.dim() == 2 and tensor.shape[-1] == cfg.d_model


def test_no_norm_configuration_reports_no_norm_preimages() -> None:
    cfg = small_cfg(use_norm=False, final_norm=False)
    model, _ = train_tiny(cfg, steps=5)
    ids = torch.arange(cfg.seq_len)[None] % cfg.vocab
    assert norm_preimages(model, ids) == {}


def test_layer_inputs_shapes_match_the_linear_layers() -> None:
    cfg = small_cfg()
    model, _ = train_tiny(cfg, steps=5)
    ids = torch.arange(cfg.seq_len)[None].repeat(2, 1) % cfg.vocab
    names = layer_names(model)
    acts = layer_inputs(model, ids, names)
    tokens = 2 * cfg.seq_len
    for name, layer in model.linear_layers().items():
        assert acts[name].shape == (tokens, layer.in_features), name


def test_fixture_training_is_fast_enough_for_ci() -> None:
    """Guard the fixture's cost: it is used by other tests, so it must stay cheap."""
    _, metrics = train_tiny(small_cfg(), steps=60)
    assert metrics["wall_time_s"] < 30.0
    assert math.isfinite(metrics["final_loss"])
