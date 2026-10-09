"""Measure every proxy variant against measured network damage on the Tier-0 fixture.

This is the Milestone-2 evidence artifact for the proxy slice: for a LayerNorm and a no-LayerNorm
fixture it records, per variant, the Spearman rank correlation with single-layer end-to-end damage,
the joint-damage ratios, and the per-layer gain spread. It exists so the adversarial review's central
measurement is reproducible from the repository, and so the naive baseline's failure is published
rather than hidden.

Usage:
    uv run python scripts/experiments/proxy_fixture_measurement.py [--out DIR]

Writes ``proxy-fixture.json`` into ``artifacts/sample-results/proxy-fixture/`` by default. CPU-only,
deterministic, well under a minute.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch

from spectraquant.evaluation.toy import (
    TinyConfig,
    layer_inputs,
    layer_names,
    network_joint_damage,
    network_single_layer_damage,
    norm_preimages,
    train_tiny,
)
from spectraquant.proxies.analysis import following_norm_container, spearman
from spectraquant.proxies.base import LayerInputs
from spectraquant.proxies.gain import estimate_downstream_gains
from spectraquant.proxies.variants import get_proxy, proxy_names
from spectraquant.quantization import QuantSpec

RANK = 8
BITS = 4
GROUP_SIZE = 32
TRAIN_STEPS = 120


def measure(cfg: TinyConfig, ids: torch.Tensor, spec: QuantSpec) -> dict:
    """Run the full measurement for one fixture configuration."""
    model, training = train_tiny(cfg, steps=TRAIN_STEPS)
    names = layer_names(model)
    damage = network_single_layer_damage(model, ids, names, RANK, spec)
    joint_hidden, joint_logits = network_joint_damage(model, ids, names, RANK, spec)
    activations = layer_inputs(model, ids, names)
    preimages = norm_preimages(model, ids)
    gains = estimate_downstream_gains(
        lambda nm, pert: model(ids, inject=(nm, pert))[1].reshape(-1, cfg.d_model),
        damage.layer_errors,
    )

    proxies: dict[str, dict] = {}
    for variant in proxy_names():
        proxy = get_proxy(variant)
        per_layer: dict[str, float] = {}
        for name in names:
            layer = LayerInputs(
                weight=model.linear_layers()[name].weight.detach(),
                activations=activations[name],
                following_norm=following_norm_container(model, name, preimages),
                residual_error=damage.residual_errors.get(name),
                downstream_gain=gains.per_layer[name],
            )
            per_layer[name] = proxy._context_layer(layer, RANK, spec).value
        proxy_values = [per_layer[n] for n in names]
        damages = [damage.per_layer[n] for n in names]
        ratios = [d / p for d, p in zip(damages, proxy_values, strict=True) if p > 0]
        total = sum(proxy_values)
        proxies[variant] = {
            "per_layer": per_layer,
            "spearman_vs_single_layer_damage": spearman(proxy_values, damages),
            "sum_proxy": total,
            "joint_over_sum_proxy": (joint_hidden / total) if total else None,
            "gain_spread": (max(ratios) / min(ratios)) if ratios and min(ratios) > 0 else None,
        }

    sum_single = sum(damage.per_layer.values())
    return {
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
        "compression": {"rank": RANK, "bits": BITS, "group_size": GROUP_SIZE},
        "training": training,
        "damage": {
            "per_layer": damage.per_layer,
            "per_layer_logits": damage.per_layer_logits,
            "joint_hidden": joint_hidden,
            "joint_logits": joint_logits,
            "sum_single_layer": sum_single,
            "joint_over_sum_single": joint_hidden / sum_single,
        },
        "gains": {"per_layer": gains.per_layer, "forward_calls": gains.forward_calls},
        "proxies": proxies,
    }


def git_commit() -> str:
    """Current commit SHA, or ``unknown`` outside a checkout."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/sample-results/proxy-fixture")
    args = parser.parse_args()

    ids = torch.arange(24)[None].repeat(2, 1) % 16
    spec = QuantSpec(bits=BITS, granularity="per_group", group_size=GROUP_SIZE, symmetric=True)
    document = {
        "git_commit": git_commit(),
        "fixtures": {
            "layernorm": measure(TinyConfig(use_norm=True, seed=0), ids, spec),
            "no_layernorm": measure(TinyConfig(use_norm=False, seed=0), ids, spec),
        },
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "proxy-fixture.json"
    path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {path}")
    for fixture, payload in document["fixtures"].items():
        print(f"\n[{fixture}] joint/sum_single = {payload['damage']['joint_over_sum_single']:.4f}")
        for variant, values in sorted(payload["proxies"].items()):
            spread = values["gain_spread"]
            print(
                f"  {variant:24s} rho={values['spearman_vs_single_layer_damage']:+.3f} "
                f"joint/sum_proxy={values['joint_over_sum_proxy']:.3e} "
                f"gain_spread={'n/a' if spread is None else f'{spread:.1f}x'}"
            )


if __name__ == "__main__":
    main()
