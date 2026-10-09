"""Tier-0 toy fixtures and exact ground truth for proxy validation.

Two different things live here and are deliberately named differently:

* **Exact single-layer ground truth** — :func:`exact_layer_output_error` computes
  ``mean_i ||x_i W^T - x_i W_hat^T||^2`` by brute force in **float64**. This is the quantity the
  proxy unit is defined on, so it is the reference for the M2 gate ("proxy results versus exact toy
  calculations").
* **Network damage** — :func:`network_single_layer_damage` and :func:`network_joint_damage` measure,
  in the tiny transformer, the end-to-end change of the final hidden state (and of the logits) caused
  by compressing one layer, or all layers at once. These are the quantities H2/H4 care about, and the
  adversarial review showed they are *not* the sum of the per-layer output errors.

The fixture model is a small pre-norm transformer with an optional LayerNorm control, because the
review's central measurement is that the LayerNorm control moves the naive proxy's correlation by up
to 0.7. Everything is deterministic, CPU-only and cheap enough to run in CI-adjacent scripts.

Shapes:
    ``ids``: ``(batch, seq)`` int64. Layer names are ``blocks.<i>.qkv``, ``blocks.<i>.proj``,
    ``blocks.<i>.fc1``, ``blocks.<i>.fc2`` and ``head``. A perturbation passed to
    :meth:`TinyTransformer.forward` has the shape of that layer's **output** tensor.

Numerical limitations:
    Network damage is measured in float32 (the model's dtype), so comparisons between proxies and
    network damage carry float32 noise; proxy-versus-exact comparisons use float64 and are held to a
    tight tolerance instead. Training the fixture is a fixture-quality fit, not a research result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

import torch
from torch import nn

from spectraquant.factorization import truncated_svd
from spectraquant.proxies.operators import compression_delta
from spectraquant.quantization import QuantSpec, fake_quantize

__all__ = [
    "TinyConfig",
    "TinyTransformer",
    "exact_layer_output_error",
    "layer_names",
    "network_joint_damage",
    "network_single_layer_damage",
    "norm_preimages",
    "train_tiny",
]


@dataclass(frozen=True)
class TinyConfig:
    """Configuration of the tiny fixture transformer."""

    n_blocks: int = 3
    d_model: int = 48
    n_heads: int = 2
    d_ff: int = 96
    vocab: int = 24
    seq_len: int = 24
    use_norm: bool = True
    final_norm: bool = True
    seed: int = 0


class _Identity(nn.Module):
    """Drop-in replacement for a normalisation layer, used as the review's control."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _norm(cfg: TinyConfig) -> nn.Module:
    return nn.LayerNorm(cfg.d_model) if cfg.use_norm else _Identity()


class _Block(nn.Module):
    """Pre-norm transformer block: ``x = x + attn(ln1(x)); x = x + mlp(ln2(x))``."""

    def __init__(self, cfg: TinyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.ln1: nn.Module = _norm(cfg)
        self.ln2: nn.Module = _norm(cfg)
        self.qkv: nn.Linear = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj: nn.Linear = nn.Linear(cfg.d_model, cfg.d_model)
        self.fc1: nn.Linear = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2: nn.Linear = nn.Linear(cfg.d_ff, cfg.d_model)
        self.act: nn.Module = nn.SiLU()

    def mlp(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TinyTransformer(nn.Module):
    """The fixture model, with per-layer perturbation injection for gain and damage measurement."""

    def __init__(self, cfg: TinyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb: nn.Embedding = nn.Embedding(cfg.vocab, cfg.d_model)
        self.pos_emb: nn.Embedding = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks: nn.ModuleList = nn.ModuleList(_Block(cfg) for _ in range(cfg.n_blocks))
        self.ln_f: nn.Module = nn.LayerNorm(cfg.d_model) if cfg.final_norm else _Identity()
        self.head: nn.Linear = nn.Linear(cfg.d_model, cfg.vocab, bias=False)

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        ids: torch.Tensor,
        inject: tuple[str, torch.Tensor] | None = None,
        capture: list[str] | None = None,
        capture_norms: list[str] | None = None,
        capture_contrib: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the model, optionally injecting ``(layer_name, perturbation)`` after that layer's op.

        Args:
            ids: ``(batch, seq)`` int64 token ids.
            inject: ``(name, perturbation)`` added to the named layer's output, or ``None``.
            capture: layer names whose **input activations** (the tensor fed to the linear op) should
                be collected.
            capture_norms: normalisation names (``blocks.<i>.ln1``, ``blocks.<i>.ln2``, ``ln_f``)
                whose **input** tensor should be collected; these are the preimages the in-situ proxy
                differentiates.
            capture_contrib: residual-stream contribution names (``blocks.<i>.attn_out``,
                ``blocks.<i>.mlp_out``) whose value should be collected; these are the tensors added
                into the residual stream, i.e. the basis the following normalisation consumes.

        Returns:
            ``(logits, final_hidden)`` — ``logits`` ``(batch, seq, vocab)``, ``final_hidden``
            ``(batch, seq, d_model)`` — plus ``captures`` when either capture argument is set.
        """
        _, t = ids.shape
        x = self.tok_emb(ids) + self.pos_emb(torch.arange(t, device=ids.device))[None]
        captures: dict[str, torch.Tensor] = {}
        target = inject[0] if inject is not None else None
        pert = inject[1] if inject is not None else None
        want_capture = (
            capture is not None or capture_norms is not None or capture_contrib is not None
        )
        capture = capture or []
        capture_norms = capture_norms or []
        capture_contrib = capture_contrib or []

        def maybe_add(name: str, tensor: torch.Tensor) -> torch.Tensor:
            if target == name and pert is not None:
                p = pert
                if p.dim() == 2 and tensor.dim() == 3:
                    # Accept a flattened (batch * seq, width) perturbation, which is how the proxy
                    # and gain code carries per-token errors.
                    p = p.reshape(tensor.shape[0], tensor.shape[1], -1)
                return tensor + p.to(tensor.dtype)
            return tensor

        for i in range(len(self.blocks)):
            block = cast(_Block, self.blocks[i])
            prefix = f"blocks.{i}"
            n1 = block.ln1(x)
            if f"{prefix}.ln1" in capture_norms:
                captures[f"{prefix}.ln1"] = n1.detach().reshape(-1, n1.shape[-1]).clone()
            if f"{prefix}.qkv" in capture:
                captures[f"{prefix}.qkv"] = n1.detach().reshape(-1, n1.shape[-1]).clone()
            qkv_out = maybe_add(f"{prefix}.qkv", block.qkv(n1))
            attn_in = _attention(qkv_out, self.cfg)
            if f"{prefix}.proj" in capture:
                captures[f"{prefix}.proj"] = attn_in.detach().reshape(-1, attn_in.shape[-1]).clone()
            attn_out = maybe_add(f"{prefix}.proj", block.proj(attn_in))
            if f"{prefix}.attn_out" in capture_contrib:
                captures[f"{prefix}.attn_out"] = (
                    attn_out.detach().reshape(-1, attn_out.shape[-1]).clone()
                )
            x = x + attn_out

            n2 = block.ln2(x)
            if f"{prefix}.ln2" in capture_norms:
                captures[f"{prefix}.ln2"] = n2.detach().reshape(-1, n2.shape[-1]).clone()
            if f"{prefix}.fc1" in capture:
                captures[f"{prefix}.fc1"] = n2.detach().reshape(-1, n2.shape[-1]).clone()
            h = maybe_add(f"{prefix}.fc1", block.fc1(n2))
            h_act = block.act(h)
            if f"{prefix}.fc2" in capture:
                captures[f"{prefix}.fc2"] = h_act.detach().reshape(-1, h_act.shape[-1]).clone()
            mlp_out = maybe_add(f"{prefix}.fc2", block.fc2(h_act))
            if f"{prefix}.mlp_out" in capture_contrib:
                captures[f"{prefix}.mlp_out"] = (
                    mlp_out.detach().reshape(-1, mlp_out.shape[-1]).clone()
                )
            x = x + mlp_out

        if "ln_f" in capture_norms:
            captures["ln_f"] = x.detach().reshape(-1, x.shape[-1]).clone()
        if "head" in capture:
            captures["head"] = x.detach().reshape(-1, x.shape[-1]).clone()
        h_final = self.ln_f(x)
        logits = self.head(h_final)
        if want_capture:
            return logits, h_final, captures  # type: ignore[return-value]
        return logits, h_final

    # ---------------------------------------------------------------- helpers
    def linear_layers(self) -> dict[str, nn.Linear]:
        """``name -> Linear`` for every compressible linear layer, in a stable order."""
        out: dict[str, nn.Linear] = {}
        for i, block in enumerate(self.blocks):
            for part in ("qkv", "proj", "fc1", "fc2"):
                out[f"blocks.{i}.{part}"] = getattr(block, part)
        out["head"] = self.head
        return out

    def following_norm(self, name: str) -> tuple[str, nn.Module] | None:
        """The normalisation that next consumes the residual stream this layer writes into.

        Returns ``(norm_name, module)`` or ``None`` for the output head. The residual stream that
        ``proj``/``fc2`` write into is consumed by the block's second/following norm; ``qkv``/``fc1``
        write into a sub-layer whose result is added to the residual and then normalised, so the same
        following norm applies (through attention/SiLU, which the normalisation Jacobian does not
        model — that gap is why the gain-aware variant exists).
        """
        if name == "head":
            return None
        block_index = int(name.split(".")[1])
        part = name.split(".")[2]
        block = cast(_Block, self.blocks[block_index])
        if part in ("qkv", "proj"):
            return f"blocks.{block_index}.ln2", block.ln2
        if block_index + 1 < len(self.blocks):
            return f"blocks.{block_index + 1}.ln1", cast(_Block, self.blocks[block_index + 1]).ln1
        return "ln_f", self.ln_f


def _attention(qkv: torch.Tensor, cfg: TinyConfig) -> torch.Tensor:
    """Attention from a precomputed ``qkv`` tensor of width ``3 * d_model``.

    Returns a ``(batch, seq, d_model)`` tensor, i.e. the width of the value projection, so the caller
    can inject a perturbation at the ``qkv`` or ``proj`` output without a shape change.
    """
    b, t, width = qkv.shape
    d = width // 3
    q, k, v = qkv.split(d, dim=-1)
    h = cfg.n_heads
    dh = d // h
    q = q.view(b, t, h, dh).transpose(1, 2)
    k = k.view(b, t, h, dh).transpose(1, 2)
    v = v.view(b, t, h, dh).transpose(1, 2)
    att = torch.softmax(q @ k.transpose(-2, -1) / dh**0.5, dim=-1)
    return (att @ v).transpose(1, 2).reshape(b, t, d)


def layer_names(model: TinyTransformer) -> Sequence[str]:
    """Stable ordering of the compressible layer names."""
    return tuple(model.linear_layers())


# ---------------------------------------------------------------- training


def train_tiny(
    cfg: TinyConfig | None = None,
    *,
    steps: int = 150,
    batch_size: int = 8,
    lr: float = 3e-3,
) -> tuple[TinyTransformer, dict[str, float]]:
    """Train the fixture on a deterministic synthetic next-token task.

    The task is ``token[t+1] = (token[t] + 1) mod vocab`` with a fixed offset per sequence, which is
    learnable in a few hundred steps on CPU and gives the model non-trivial attention/MLP structure.
    Determinism: the data generator, the initialization and the batch order are all seeded from
    ``cfg.seed``.

    Returns:
        ``(model, metrics)`` where ``metrics`` holds ``initial_loss``, ``final_loss`` and
        ``wall_time_s`` — fixture provenance, not a research result.
    """
    import time

    cfg = cfg or TinyConfig()
    torch.manual_seed(cfg.seed)
    model = TinyTransformer(cfg)
    gen = torch.Generator().manual_seed(cfg.seed + 1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    started = time.perf_counter()

    def batch() -> tuple[torch.Tensor, torch.Tensor]:
        offsets = torch.randint(0, cfg.vocab, (batch_size, 1), generator=gen)
        base = torch.randint(0, cfg.vocab, (batch_size, 1), generator=gen)
        seq = (base + torch.arange(cfg.seq_len + 1)[None] + offsets) % cfg.vocab
        return seq[:, :-1], seq[:, 1:]

    losses: list[float] = []
    for _ in range(steps):
        ids, target = batch()
        logits, _ = model(ids)
        loss = nn.functional.cross_entropy(logits.reshape(-1, cfg.vocab), target.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    model.eval()
    return model, {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "wall_time_s": time.perf_counter() - started,
        "steps": float(steps),
    }


# ---------------------------------------------------------------- ground truth


def exact_layer_output_error(w: torch.Tensor, x: torch.Tensor, rank: int, spec: QuantSpec) -> float:
    """Brute-force **float64** value of the proxy unit for one layer: ``mean_i ||x_i delta^T||^2``.

    Args:
        w: ``(out, in)`` weight.
        x: ``(n_samples, in)`` activations.
        rank: low-rank budget.
        spec: quantization configuration applied to both factors.

    Returns:
        The exact plug-in value of the frozen unit for this batch. This is the reference the proxy
        variants are tested against; the estimator's *sampling* error relative to the population is
        separate and reported by the variants themselves.
    """
    delta = compression_delta(w, rank, spec)
    proj = x.to(torch.float64) @ delta.to(torch.float64).transpose(0, 1)
    return float(torch.mean(torch.sum(proj * proj, dim=-1)))


@dataclass
class DamageReport:
    """Measured network damage for one compression configuration."""

    per_layer: dict[str, float]
    per_layer_logits: dict[str, float]
    joint_hidden: float
    joint_logits: float
    baseline_hidden: torch.Tensor | None = field(repr=False, default=None)
    layer_errors: dict[str, torch.Tensor] = field(repr=False, default_factory=dict)
    residual_errors: dict[str, torch.Tensor] = field(repr=False, default_factory=dict)
    diagnostics: dict[str, float] = field(default_factory=dict)


def residual_contribution_key(layer_name: str) -> str | None:
    """The residual-stream contribution a layer writes into: ``blocks.<i>.attn_out``/``mlp_out``.

    ``None`` for the output head, which writes no residual contribution. This mapping is what lets
    the in-situ proxy express a layer's error in the basis the following normalisation consumes
    (``qkv``/``fc1`` errors pass through attention/SiLU first, so their *output-space* error has a
    different width than the residual stream).
    """
    if layer_name == "head":
        return None
    parts = layer_name.split(".")
    if len(parts) != 3:
        raise ValueError(f"unrecognised layer name {layer_name!r}")
    block, part = parts[1], parts[2]
    if part in ("qkv", "proj"):
        return f"blocks.{block}.attn_out"
    if part in ("fc1", "fc2"):
        return f"blocks.{block}.mlp_out"
    raise ValueError(f"unrecognised layer name {layer_name!r}")


def _flatten_hidden(hidden: torch.Tensor) -> torch.Tensor:
    return hidden.reshape(-1, hidden.shape[-1])


def network_single_layer_damage(
    model: TinyTransformer,
    ids: torch.Tensor,
    names: Sequence[str],
    rank: int,
    spec: QuantSpec,
) -> DamageReport:
    """Compress **one layer at a time** and measure the end-to-end damage plus the layer's own error.

    Returns a :class:`DamageReport` with:
        ``per_layer[name]``: ``mean ||Δh||^2`` of the final hidden state,
        ``per_layer_logits[name]``: ``mean ||Δlogits||^2``,
        ``layer_errors[name]``: the measured change of that layer's **output tensor** (the injected
        error, used as ``residual_error`` by the in-situ variant and as the perturbation for the gain
        estimator).
    """
    originals = {
        name: layer.weight.detach().clone() for name, layer in model.linear_layers().items()
    }
    contrib_keys = sorted(
        {key for name in names if (key := residual_contribution_key(name)) is not None}
    )
    with torch.no_grad():
        base_logits, base_hidden = model(ids)
        base_flat = _flatten_hidden(base_hidden)
        _, _, base_contrib = model(ids, capture_contrib=contrib_keys)
    per_layer: dict[str, float] = {}
    per_layer_logits: dict[str, float] = {}
    layer_errors: dict[str, torch.Tensor] = {}
    residual_errors: dict[str, torch.Tensor] = {}
    for name in names:
        layer = model.linear_layers()[name]
        compressed = _compressed_weight(originals[name], rank, spec)
        delta = compressed - originals[name]
        key = residual_contribution_key(name)
        with torch.no_grad():
            x_in = _layer_input(model, ids, name)
            layer_errors[name] = (x_in @ delta.transpose(0, 1)).detach()
            layer.weight.copy_(compressed)
            logits, hidden = model(ids)
            contrib = None
            if key is not None:
                _, _, compressed_contrib = model(ids, capture_contrib=[key])
                contrib = (compressed_contrib[key] - base_contrib[key]).detach()
            layer.weight.copy_(originals[name])
        if contrib is not None:
            residual_errors[name] = contrib
        diff = _flatten_hidden(hidden) - base_flat
        per_layer[name] = float(torch.mean(torch.sum(diff.to(torch.float64) ** 2, dim=-1)))
        logit_diff = (logits - base_logits).to(torch.float64)
        per_layer_logits[name] = float(torch.mean(torch.sum(logit_diff * logit_diff, dim=-1)))
    return DamageReport(
        per_layer=per_layer,
        per_layer_logits=per_layer_logits,
        joint_hidden=float("nan"),
        joint_logits=float("nan"),
        baseline_hidden=base_flat.detach(),
        layer_errors=layer_errors,
        residual_errors=residual_errors,
    )


def network_joint_damage(
    model: TinyTransformer,
    ids: torch.Tensor,
    names: Sequence[str],
    rank: int,
    spec: QuantSpec,
) -> tuple[float, float]:
    """Compress **all** layers simultaneously and return ``(hidden damage, logits damage)``."""
    originals = {
        name: layer.weight.detach().clone() for name, layer in model.linear_layers().items()
    }
    with torch.no_grad():
        base_logits, base_hidden = model(ids)
        base_flat = _flatten_hidden(base_hidden)
        for name in names:
            layer = model.linear_layers()[name]
            layer.weight.copy_(_compressed_weight(originals[name], rank, spec))
        logits, hidden = model(ids)
        for name, layer in model.linear_layers().items():
            layer.weight.copy_(originals[name])
    diff = _flatten_hidden(hidden) - base_flat
    hidden_damage = float(torch.mean(torch.sum(diff.to(torch.float64) ** 2, dim=-1)))
    logit_diff = (logits - base_logits).to(torch.float64)
    logits_damage = float(torch.mean(torch.sum(logit_diff * logit_diff, dim=-1)))
    return hidden_damage, logits_damage


def _compressed_weight(w: torch.Tensor, rank: int, spec: QuantSpec) -> torch.Tensor:
    """``Q(B) Q(A)``: the low-rank form with both factors fake-quantized."""
    if rank <= 0:
        return torch.zeros_like(w)
    factors = truncated_svd(w, rank)
    return fake_quantize(factors.B, spec) @ fake_quantize(factors.A, spec)


def _layer_input(model: TinyTransformer, ids: torch.Tensor, name: str) -> torch.Tensor:
    """The activations fed to ``name`` on the uncompressed forward pass, flattened to 2-D."""
    with torch.no_grad():
        _, _, captures = model(ids, capture=[name])
    return captures[name]


def layer_inputs(
    model: TinyTransformer,
    ids: torch.Tensor,
    names: Sequence[str],
) -> Mapping[str, torch.Tensor]:
    """Uncompressed input activations for every requested layer, flattened to ``(tokens, in)``."""
    with torch.no_grad():
        _, _, captures = model(ids, capture=list(names))
    return {name: captures[name] for name in names}


def norm_preimages(model: TinyTransformer, ids: torch.Tensor) -> Mapping[str, torch.Tensor]:
    """Preimages (inputs) of every **real** normalisation on the uncompressed forward pass.

    Flattened to 2-D. Only ``nn.LayerNorm`` modules are reported: when the fixture is built with the
    LayerNorm control disabled, the replacement module is the identity, whose "preimage" carries no
    information and would let the in-situ variant claim context it does not have.
    """
    wanted = [
        f"blocks.{i}.{label}"
        for i in range(len(model.blocks))
        for label, module in (("ln1", model.blocks[i].ln1), ("ln2", model.blocks[i].ln2))
        if isinstance(module, nn.LayerNorm)
    ]
    if model.cfg.final_norm:
        wanted.append("ln_f")
    if not wanted:
        return {}
    with torch.no_grad():
        _, _, captures = model(ids, capture_norms=wanted)
    return {name: captures[name] for name in wanted}
