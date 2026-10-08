"""Hand-written char-level transformer for the local CI smoke fixture (Tier 0).

Pure PyTorch, CPU-only, no external model weights: the whole model is defined here so that a smoke
run has no download, no license surface and no hidden revision. Parameter count is capped by
:data:`MAX_SMOKE_PARAMS` (200 k) so the model can never silently grow into something the
workstation cannot train.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache

import torch
from torch import Tensor, nn

from spectraquant.config import ModelConfig

__all__ = [
    "MAX_SMOKE_PARAMS",
    "CausalSelfAttention",
    "TinyCharTransformer",
    "TransformerBlock",
    "count_parameters",
    "initial_state_checksum",
]

#: Hard cap for the smoke-fixture transformer (Tier 0 fixture budget, AGENTS.md section 6).
MAX_SMOKE_PARAMS = 200_000


@lru_cache(maxsize=8)
def _causal_mask(seq_len: int) -> Tensor:
    """Lower-triangular boolean mask, cached per sequence length (CPU, bool)."""
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Shapes: input ``(B, T, d_model)`` -> output ``(B, T, d_model)``; ``T <= max_seq_len``.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        batch, seq_len, channels = x.shape
        qkv = self.qkv(x).reshape(batch, seq_len, 3, self.n_heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # (B, H, T, head_dim)

        scores = (query @ key.transpose(-2, -1)) * (self.head_dim**-0.5)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        context = (weights @ value).transpose(1, 2).reshape(batch, seq_len, channels)
        return self.resid_dropout(self.proj(context))


class TransformerBlock(nn.Module):
    """Pre-LayerNorm transformer block: attention then a 4x GELU MLP, both residual."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = 4 * cfg.d_model
        self.ln_1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ln_2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x), mask)
        return x + self.mlp(self.ln_2(x))


class TinyCharTransformer(nn.Module):
    """Tiny decoder-only transformer over a small vocabulary.

    Shapes: ``forward(tokens)`` takes ``(B, T)`` int64 token ids with ``T <= max_seq_len`` and
    returns logits ``(B, T, vocab_size)`` float32. Positional information is a learned embedding;
    no RoPE, no weight tying — the point is a minimal, inspectable model for harness checks,
    not a competitive LM.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.config = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position_embedding = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, tokens: Tensor) -> Tensor:
        seq_len = tokens.shape[1]
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds model.max_seq_len {self.config.max_seq_len}"
            )
        positions = torch.arange(seq_len, device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions).unsqueeze(0)
        mask = _causal_mask(seq_len).to(tokens.device)
        for block in self.blocks:
            hidden = block(hidden, mask)
        return self.lm_head(self.ln_final(hidden))


def count_parameters(model: nn.Module) -> int:
    """Number of trainable scalar parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def initial_state_checksum(model: nn.Module) -> str:
    """``sha256:<hex>`` over parameter names and raw bytes of the *current* state.

    Used as ``model_revision`` in a manifest: for a locally defined model it identifies the exact
    architecture plus initialisation, which is deterministic given the seed.
    """
    digest = hashlib.sha256()
    for name, parameter in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return f"sha256:{digest.hexdigest()}"
