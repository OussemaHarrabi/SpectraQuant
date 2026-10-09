"""Analysis helpers for proxy validation: rank correlation and normalisation containers.

These live in the library rather than in the measurement script so the tests can import them without
depending on the script's import path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn

from spectraquant.proxies.base import FollowingNorm

__all__ = ["following_norm_container", "spearman"]


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rank correlation with average ranks for ties.

    Shapes:
        ``a`` and ``b`` must have the same length. Returns ``nan`` for fewer than two pairs or when
        either input is constant (a zero-variance rank vector has no correlation to report).

    No scipy dependency: the ranks are computed directly, which also makes tie handling explicit
    (average ranks, the standard convention).
    """
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    n = len(a)
    if n < 2:
        return float("nan")

    def ranks(values: Sequence[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: values[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = average
            i = j + 1
        return out

    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den > 0 else float("nan")


def following_norm_container(
    model: object, layer_name: str, preimages: Mapping[str, torch.Tensor]
) -> FollowingNorm | None:
    """Build the :class:`FollowingNorm` for one layer of the toy fixture.

    Args:
        model: the fixture model (anything exposing ``following_norm(name)``).
        layer_name: e.g. ``blocks.0.proj``.
        preimages: ``norm name -> input tensor`` from
            :func:`spectraquant.evaluation.toy.norm_preimages`.

    Returns:
        A container carrying the normalisation's operating point and parameters, or ``None`` when the
        layer has no following normalisation (the output head) or the preimage was not captured.
    """
    found = model.following_norm(layer_name)  # type: ignore[attr-defined]
    if found is None:
        return None
    norm_name, module = found
    preimage = preimages.get(norm_name)
    if preimage is None:
        return None
    if isinstance(module, nn.LayerNorm):
        return FollowingNorm(
            kind="layernorm",
            preimage=preimage,
            weight=module.weight.detach(),
            bias=module.bias.detach(),
            eps=float(module.eps),
        )
    return FollowingNorm(kind="identity", preimage=preimage, weight=None, eps=1e-5)
