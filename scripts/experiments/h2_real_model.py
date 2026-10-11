"""Real-model H2 ranking measurement on the CPU substrate (measurement class 2).

This is the decisive experiment the project never ran: does the output-aware sensitivity proxy — the
core of the proxy-guided allocator — actually rank a **real** model's layers by the damage a
compression variant causes? The proxy's ranking ability was previously measured only on the
13-module synthetic Tier-0 fixture.

What the script does
--------------------

1. Loads a real pretrained model (default ``HuggingFaceTB/SmolLM2-135M`` at its pinned revision) in
   float32 on the CPU, and a small calibration batch drawn **only** from the training-role split of
   the plan's calibration dataset (``allenai/c4`` ``en`` ``train``, revision-pinned, streamed and
   capped). The test/validation split is never read.
2. For each compressible ``nn.Linear`` layer and each predeclared :class:`CompressionVariant`
   (rank-dominant at two ranks, quantization-dominant at two bit widths, two rank+quantized
   combinations), measures the damage the variant causes on the real activations, on two targets:
   the layer's own output error (the frozen proxy unit) and the end-to-end change of the final hidden
   state when that layer alone is compressed (the Tier-0 fixture's target).
3. Scores every registered proxy through its public ``score_model`` entry point, and computes the
   pre-registered model-level statistic (Spearman ``rho`` per proxy per cell, Fisher-``z``,
   ``n_eff`` by a within-model module bootstrap, paired DerSimonian-Laird contrasts against each
   predeclared comparator, achieved MDE).
4. Writes a machine-readable artifact and prints a readable summary.

Nothing is trained. No GPU is used. Every number is **measurement class 2** (fake quantization: float
execution that simulates quantization numerics) — never packed storage (class 3) and never a
kernel-backed measurement (class 4). No latency is claimed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, cast

import torch

from spectraquant.cloud.plan_data import load_plan_texts, verify_pinned_revision
from spectraquant.experiment_plan import DatasetRole
from spectraquant.proxies.base import LayerInputs
from spectraquant.proxies.gain import rms
from spectraquant.proxies.h2_ranking import (
    CALIBRATION_ROLE,
    DEFAULT_CALIBRATION_SEED,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    DEFAULT_PROXIES,
    DEFAULT_VARIANTS,
    CompressionVariant,
    build_document,
    compressed_weight,
    layer_output_damage,
    measure_cell,
)
from spectraquant.proxies.validation import git_state

ARTIFACT_NAME = "h2-real-model.json"
DEFAULT_OUT = "artifacts/sample-results/h2-real-model"

#: The plan's calibration dataset pin (`configs/tier1/smollm2_135m.yaml`): C4 `en`, role
#: ``calibration`` (whose split is the dataset's ``train`` split, never the test split).
CALIBRATION_DATASET = DatasetRole(
    name="allenai/c4",
    revision="1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
    license="odc-by",
    config="en",
    roles=["train", "calibration"],
)

#: The two damage targets, in report order. The first is the frozen proxy unit (the task's target);
#: the second is the Tier-0 fixture's end-to-end target.
TARGETS: tuple[str, ...] = ("layer_output", "model_output")
TARGET_NOTES: dict[str, str] = {
    "layer_output": (
        "mean_i ||x_i W^T - x_i W_hat^T||^2 on the real calibration activations: the frozen proxy "
        "unit, i.e. the damage the variant causes to the layer's own output"
    ),
    "model_output": (
        "mean_i ||h_i - h_hat_i||^2 of the final hidden state when that layer alone is compressed "
        "(one extra forward per layer and variant): the Tier-0 fixture's end-to-end target"
    ),
}


def _linear_names(model: torch.nn.Module) -> list[str]:
    """Every compressible ``nn.Linear`` module, in stable name order."""
    return sorted(
        name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)
    )


def _contribution_module(name: str) -> str | None:
    """The module whose output **is** the residual-stream contribution of ``name``, or ``None``.

    SmolLM2 is Llama-shaped: the attention sub-layer output added to the residual stream is
    ``o_proj``'s output, and the MLP sub-layer output is ``down_proj``'s output. ``q/k/v`` and
    ``gate/up`` feed those sub-layers, so compressing them changes the sub-layer output indirectly;
    ``lm_head`` writes logits and has no residual contribution.
    """
    parts = name.split(".")
    if len(parts) >= 4 and parts[-1] in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        return ".".join([*parts[:-1], "o_proj"])
    if len(parts) >= 4 and parts[-1] in {"gate_proj", "up_proj", "down_proj"}:
        return ".".join([*parts[:-1], "down_proj"])
    return None


def _calibration_ids(
    tokenizer: Any,
    *,
    documents: int,
    tokens: int,
    batch_size: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """A deterministic calibration batch from the training-role split of the pinned dataset."""
    verify_pinned_revision(
        CALIBRATION_DATASET.name, CALIBRATION_DATASET.revision, repo_type="dataset"
    )
    texts = load_plan_texts(
        CALIBRATION_DATASET,
        split=CALIBRATION_ROLE,
        dataset_config=CALIBRATION_DATASET.config,
        max_documents=documents,
        stream=True,
    )
    joined = "\n\n".join(texts)
    encoded = tokenizer(joined, return_tensors="pt").input_ids[0]
    if encoded.numel() < tokens:
        raise ValueError(
            f"calibration corpus produced {encoded.numel()} tokens, fewer than the requested {tokens}"
        )
    usable = (tokens // batch_size) * batch_size
    ids = encoded[:usable].reshape(batch_size, usable // batch_size)
    provenance = {
        "dataset": CALIBRATION_DATASET.name,
        "revision": CALIBRATION_DATASET.revision,
        "config": CALIBRATION_DATASET.config,
        "role": "calibration",
        "split": CALIBRATION_ROLE,
        "documents": len(texts),
        "tokens": int(encoded.numel()),
        "tokens_used": int(usable),
        "batch_size": batch_size,
        "sequence_length": usable // batch_size,
        "text_sha256": hashlib.sha256(joined.encode("utf-8")).hexdigest(),
        "seed": DEFAULT_CALIBRATION_SEED,
        "test_split_read": False,
    }
    return ids, provenance


def _capture(
    model: Any,
    names: list[str],
    ids: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """One uncompressed forward, capturing layer inputs, residual contributions and the output.

    Returns ``(activations, contributions, hidden, logits)`` flattened to 2-D (tokens, width) in
    float64 for the statistics. ``contributions`` is keyed by the **contribution module** name
    (:func:`_contribution_module`).
    """
    activations: dict[str, torch.Tensor] = {}
    contributions: dict[str, torch.Tensor] = {}
    handles = []
    for name in names:
        module = model.get_submodule(name)

        def pre_hook(_module: Any, inputs: Any, *, _name: str = name) -> None:
            activations[_name] = inputs[0].detach()

        handles.append(module.register_forward_pre_hook(pre_hook))
        key = _contribution_module(name)
        if key is not None and key not in contributions:
            contribution_module = model.get_submodule(key)

            def post_hook(_module: Any, _inputs: Any, output: Any, *, _key: str = key) -> None:
                contributions[_key] = output.detach()

            handles.append(contribution_module.register_forward_hook(post_hook))
    try:
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
    finally:
        for handle in handles:
            handle.remove()
    hidden = out.hidden_states[-1].reshape(-1, out.hidden_states[-1].shape[-1]).to(torch.float64)
    logits = out.logits.reshape(-1, out.logits.shape[-1]).to(torch.float64)
    acts = {
        name: value.reshape(-1, value.shape[-1]).to(torch.float64)
        for name, value in activations.items()
    }
    contribs = {
        key: value.reshape(-1, value.shape[-1]).to(torch.float64)
        for key, value in contributions.items()
    }
    return acts, contribs, hidden, logits


def _measure(
    model: Any,
    names: list[str],
    ids: torch.Tensor,
    variants: tuple[CompressionVariant, ...],
    *,
    hidden_width: int,
    progress: bool,
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
    dict[str, torch.Tensor],
]:
    """Measure both damage targets and the downstream gain for every (variant, layer).

    Returns ``(layer_output, model_output, gain, activations)``: the first three are
    ``variant name -> layer name -> value``; ``activations`` is ``layer name -> (tokens, in)`` from
    the uncompressed forward, reused to score the proxies.
    """
    activations, base_contrib, base_hidden, _ = _capture(model, names, ids)
    layer_output: dict[str, dict[str, float]] = {v.name: {} for v in variants}
    model_output: dict[str, dict[str, float]] = {v.name: {} for v in variants}
    gain: dict[str, dict[str, float]] = {v.name: {} for v in variants}

    contribution_modules = {name: _contribution_module(name) for name in names}
    originals = {name: model.get_submodule(name).weight.detach().clone() for name in names}

    for variant in variants:
        weights = {name: compressed_weight(originals[name], variant) for name in names}
        for index, name in enumerate(names):
            module = model.get_submodule(name)
            layer_output[variant.name][name] = layer_output_damage(
                originals[name], activations[name], variant
            )
            key = contribution_modules[name]
            captured: dict[str, torch.Tensor] = {}
            handle = None
            if key is not None:

                def post_hook(
                    _module: Any,
                    _inputs: Any,
                    output: Any,
                    *,
                    _key: str = key,
                    _captured: dict[str, torch.Tensor] = captured,
                ) -> None:
                    _captured[_key] = output.detach()

                handle = model.get_submodule(key).register_forward_hook(post_hook)
            try:
                module.weight.copy_(weights[name])
                with torch.no_grad():
                    out = model(ids, output_hidden_states=True)
            finally:
                module.weight.copy_(originals[name])
                if handle is not None:
                    handle.remove()
            hidden = out.hidden_states[-1].reshape(-1, hidden_width).to(torch.float64)
            change = hidden - base_hidden
            model_output[variant.name][name] = float(torch.mean(torch.sum(change * change, dim=-1)))
            if key is not None:
                residual = (
                    captured[key].reshape(-1, hidden_width).to(torch.float64) - base_contrib[key]
                )
                injected = rms(residual)
                gain[variant.name][name] = rms(change) / injected if injected > 0.0 else 0.0
            if progress and index % 25 == 0:
                print(f"  [{variant.name}] {index + 1}/{len(names)} layers", flush=True)
    return layer_output, model_output, gain, activations


def _layer_inputs(
    model: Any,
    names: list[str],
    activations: dict[str, torch.Tensor],
    gains: dict[str, float],
) -> dict[str, LayerInputs]:
    """Build the ``LayerInputs`` handed to the proxies, carrying the measured downstream gain.

    The weight is the model's own (float32) tensor, so the proxy's operator error is bit-identical to
    the one the measured damage was computed from.
    """
    inputs: dict[str, LayerInputs] = {}
    for name in names:
        module = model.get_submodule(name)
        inputs[name] = LayerInputs(
            weight=module.weight.detach(),
            activations=activations[name],
            downstream_gain=gains.get(name),
        )
    return inputs


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):+.{digits}f}"


def print_summary(document: dict[str, Any]) -> None:
    """Print the readable summary: per-cell rhos, aggregate z, contrasts, MDE and wall clock."""
    plan = document["plan"]
    proxies = plan["proxies"]
    for target, section in document["targets"].items():
        print("=" * 110)
        print(f"TARGET {target}: {section['description']}")
        print("=" * 110)
        header = f"{'variant':<14}" + "".join(f"{name[:13]:>14}" for name in proxies)
        print(header)
        for cell in section["per_cell"]:
            row = f"{cell['variant']:<14}"
            for name in proxies:
                row += f"{_fmt(cell['variants'][name]['rho']):>14}"
            print(row)
        print()
        print("Model-level aggregate (pooled Fisher-z over cells):")
        print(f"{'proxy':<24}{'rho_mean':>10}{'n_eff_mean':>12}{'pooled_z':>10}{'k':>4}{'I2':>8}")
        for name in proxies:
            entry = section["aggregate"]["variants"][name]
            print(
                f"{name:<24}{_fmt(entry['rho_mean']):>10}{_fmt(entry['n_eff_mean']):>12}"
                f"{_fmt(entry['pooled_z']):>10}{entry['k']:>4}{_fmt(entry['i2']):>8}"
            )
        print()
        print("Paired contrasts (candidate - comparator), analytic random-effects CI:")
        for candidate, comparators in section["aggregate"]["contrasts"].items():
            for comparator, payload in comparators.items():
                if not payload.get("usable"):
                    print(f"  {candidate:<22} - {comparator:<22} n/a (no usable cell)")
                    continue
                verdict = payload["verdict"]["verdict"]
                print(
                    f"  {candidate:<22} - {comparator:<22} delta_z={_fmt(payload['delta_z'])} "
                    f"CI=[{_fmt(payload['ci'][0])}, {_fmt(payload['ci'][1])}] "
                    f"cell_boot=[{_fmt(payload['cell_boot_ci'][0])}, "
                    f"{_fmt(payload['cell_boot_ci'][1])}] p={payload['p_value']:.4g} "
                    f"k={payload['k']} MDE={_fmt(payload['mde_achieved_z'])} -> {verdict}"
                )
        mde = section["aggregate"]["mde"]
        print(
            f"Achieved MDE(z) reference n_eff={_fmt(mde['n_eff_reference'])} "
            f"n_models={mde['n_models']} -> MDE(z)={_fmt(mde['achieved_z'])} "
            f"(predeclared 0.33 optimistic / 0.79 conservative at S=5)"
        )
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--calibration-documents", type=int, default=16)
    parser.add_argument("--calibration-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-layers", type=int, default=None, help="cap the layer count (smoke)")
    parser.add_argument(
        "--variants",
        default=",".join(v.name for v in DEFAULT_VARIANTS),
        help="comma-separated variant names to run",
    )
    parser.add_argument("--proxies", default=",".join(DEFAULT_PROXIES))
    parser.add_argument("--n-boot-neff", type=int, default=200)
    parser.add_argument("--n-boot-ci", type=int, default=2000)
    parser.add_argument("--boot-seed", type=int, default=20261009)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(DEFAULT_CALIBRATION_SEED)
    torch.set_num_threads(args.threads)
    start = time.perf_counter()

    variants = tuple(
        v
        for v in DEFAULT_VARIANTS
        if v.name in {n.strip() for n in args.variants.split(",") if n.strip()}
    )
    if not variants:
        raise SystemExit(f"no known variant in --variants {args.variants!r}")

    verify_pinned_revision(args.model_id, args.revision, repo_type="model")
    from spectraquant.cloud.plan_data import require_transformers

    transformers = require_transformers()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_id, revision=args.revision, dtype=torch.float32
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    ids, calibration = _calibration_ids(
        tokenizer,
        documents=args.calibration_documents,
        tokens=args.calibration_tokens,
        batch_size=args.batch_size,
    )
    names = _linear_names(model)
    if args.max_layers is not None:
        names = names[: args.max_layers]
    print(
        f"model {args.model_id}@{args.revision[:12]} | {len(names)} compressible linear layers | "
        f"calibration {calibration['documents']} documents / {calibration['tokens_used']} tokens "
        f"from the {calibration['split']!r} split of {calibration['dataset']}",
        flush=True,
    )

    hidden_width = int(model.config.hidden_size)
    layer_damage, model_damage, gains, activations = _measure(
        model, names, ids, variants, hidden_width=hidden_width, progress=not args.quiet
    )

    per_cell: dict[str, list[dict[str, Any]]] = {target: [] for target in TARGETS}
    for target_index, target in enumerate(TARGETS):
        damages = layer_damage if target == "layer_output" else model_damage
        for variant_index, variant in enumerate(variants):
            inputs = _layer_inputs(model, names, activations, gains[variant.name])
            per_cell[target].append(
                measure_cell(
                    inputs,
                    damages[variant.name],
                    variant,
                    proxies=[p for p in args.proxies.split(",") if p],
                    n_boot_neff=args.n_boot_neff,
                    neff_seed=args.boot_seed + target_index * 1000 + variant_index,
                )
            )

    state = git_state()
    document = build_document(
        per_cell,
        target_notes=TARGET_NOTES,
        model={
            "id": args.model_id,
            "revision": args.revision,
            "parameters": int(sum(p.numel() for p in model.parameters())),
            "dtype": "float32",
            "device": "cpu",
            "threads": args.threads,
        },
        calibration=calibration,
        variants=variants,
        proxies=[p for p in args.proxies.split(",") if p],
        n_boot_neff=args.n_boot_neff,
        n_boot_ci=args.n_boot_ci,
        boot_seed=args.boot_seed,
        wall_time_s=time.perf_counter() - start,
        git_commit=state["git_commit"],
        git_dirty=state["git_dirty"],
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = out_dir / ARTIFACT_NAME
    artifact.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.quiet:
        print_summary(document)
    runtime = cast(dict[str, Any], document["runtime"])
    print(f"wrote {artifact} | wall {runtime['wall_time_s']} s")


if __name__ == "__main__":
    main()
