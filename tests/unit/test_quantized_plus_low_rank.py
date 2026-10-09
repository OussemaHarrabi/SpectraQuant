"""Tests for :class:`QuantizedPlusLowRankLinear` — the LR-QAT/LoftQ training representation."""

from __future__ import annotations

import torch
from torch import nn

from spectraquant.training.low_rank import QuantizedPlusLowRankLinear


def _layer(out_features: int = 4, in_features: int = 3) -> nn.Linear:
    torch.manual_seed(0)
    return nn.Linear(in_features, out_features, bias=True)


def test_forward_is_the_base_plus_the_factor_product() -> None:
    """The forward pass must be exactly ``x @ (Wq + B @ A)^T + bias``, hand-computed."""
    torch.manual_seed(1)
    base = torch.randn(2, 3)
    layer = QuantizedPlusLowRankLinear(base, 2, bias=torch.zeros(2))
    assert layer.bias is not None
    with torch.no_grad():
        layer.A.copy_(torch.tensor([[1.0, 0.0, -1.0], [0.5, 2.0, 0.0]]))
        layer.B.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        layer.bias.copy_(torch.tensor([0.25, -0.5]))
    x = torch.tensor([[1.0, 2.0, 3.0]])

    effective = base + torch.tensor([[1.0, 0.0], [0.0, 1.0]]) @ torch.tensor(
        [[1.0, 0.0, -1.0], [0.5, 2.0, 0.0]]
    )
    expected = x @ effective.T + torch.tensor([0.25, -0.5])

    torch.testing.assert_close(layer(x), expected)
    torch.testing.assert_close(layer.effective_weight(), effective)


def test_the_base_is_frozen_and_only_the_factors_are_parameters() -> None:
    """The quantized base must not train: it is the artifact, not a free variable."""
    layer = QuantizedPlusLowRankLinear(torch.randn(3, 2), 2, bias=torch.zeros(3))
    names = {name for name, _ in layer.named_parameters()}
    assert names == {"A", "B", "bias"}
    # The bias-less constructor registers no bias at all, rather than a zero one.
    assert {n for n, _ in QuantizedPlusLowRankLinear(torch.randn(3, 2), 2).named_parameters()} == {
        "A",
        "B",
    }
    assert "base" in dict(layer.named_buffers())
    assert not layer.base.requires_grad


def test_the_factors_absorb_the_quantization_residual() -> None:
    """LR-QAT's identity: the low-rank term reduces the error the quantized base leaves behind."""
    linear = _layer()
    dense = linear.weight.detach()
    # A crude quantized base: round to two decimals, which leaves a real residual.
    base = torch.round(dense * 100.0) / 100.0

    layer = QuantizedPlusLowRankLinear.from_linear(linear, 2, base=base)

    base_error = float(torch.linalg.matrix_norm(dense - base))
    compensated = float(torch.linalg.matrix_norm(dense - layer.effective_weight()))
    assert compensated < base_error
    # The factors are the truncated SVD of the residual, so the residual itself must shrink.
    assert compensated <= float(torch.linalg.matrix_norm(dense - base - layer.B @ layer.A)) + 1e-6


def test_explicit_factors_are_used_verbatim() -> None:
    """An arm with its own schedule (LoftQ) passes factors and they must not be overwritten."""
    linear = _layer()
    a = torch.zeros(2, 3)
    b = torch.zeros(4, 2)
    layer = QuantizedPlusLowRankLinear.from_linear(
        linear, 2, base=linear.weight.detach(), factors=(a, b)
    )
    torch.testing.assert_close(layer.A, a)
    torch.testing.assert_close(layer.B, b)
    # Zero factors over the unquantized dense base: the layer is the dense layer.
    torch.testing.assert_close(layer.effective_weight(), linear.weight.detach())


def test_mismatched_shapes_are_refused() -> None:
    linear = _layer()
    try:
        QuantizedPlusLowRankLinear.from_linear(linear, 2, base=torch.randn(3, 4))
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:  # pragma: no cover - the guard must fire
        raise AssertionError("a base of the wrong shape was accepted")

    try:
        QuantizedPlusLowRankLinear.from_linear(
            linear, 2, factors=(torch.zeros(2, 4), torch.zeros(4, 2))
        )
    except ValueError as exc:
        assert "expected" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("factors of the wrong shape were accepted")
