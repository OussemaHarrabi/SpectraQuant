"""Contract tests for the frozen proxy interface (``proxies/base.py``)."""

from __future__ import annotations

import pytest
import torch

from spectraquant.proxies.base import (
    FollowingNorm,
    LayerInputs,
    ProxyInputError,
    ProxyResult,
    layer_input_error,
)
from spectraquant.quantization import QuantSpec


def _spec() -> QuantSpec:
    return QuantSpec(bits=4, granularity="per_group", group_size=32, symmetric=True)


def test_proxy_result_rejects_out_of_range_measurement_class() -> None:
    for bad in (0, 3, 5, -1):
        with pytest.raises(ValueError, match="measurement_class"):
            ProxyResult(value=1.0, exact=True, measurement_class=bad, per_layer={}, diagnostics={})


def test_proxy_result_rejects_non_finite_value() -> None:
    with pytest.raises(ValueError, match="finite float"):
        ProxyResult(
            value=float("nan"), exact=True, measurement_class=2, per_layer={}, diagnostics={}
        )


def test_proxy_result_copies_mutable_fields() -> None:
    per_layer = {"a": 1.0}
    diagnostics = {"b": 2.0}
    result = ProxyResult(
        value=1.0, exact=True, measurement_class=2, per_layer=per_layer, diagnostics=diagnostics
    )
    per_layer["a"] = 99.0
    diagnostics["b"] = 99.0
    assert result.per_layer == {"a": 1.0}
    assert result.diagnostics == {"b": 2.0}


def test_proxy_result_ratio_handles_zero_denominator() -> None:
    zero = ProxyResult(value=0.0, exact=True, measurement_class=2, per_layer={}, diagnostics={})
    nonzero = ProxyResult(value=3.0, exact=True, measurement_class=2, per_layer={}, diagnostics={})
    assert zero.ratio_to(zero) == 0.0
    assert nonzero.ratio_to(zero) == float("inf")
    assert nonzero.ratio_to(nonzero) == 1.0


@pytest.mark.parametrize("kind", ["identity", "layernorm", "rmsnorm"])
def test_following_norm_jacobian_matches_autograd(kind: str) -> None:
    """The analytic Jacobian must equal a central finite difference of the true normalisation."""
    torch.manual_seed(0)
    n, d = 5, 7
    v = torch.randn(n, d, dtype=torch.float64) + 0.5
    gamma = torch.rand(d, dtype=torch.float64) + 0.5
    beta = torch.randn(d, dtype=torch.float64) if kind == "layernorm" else None
    weight = None if kind == "identity" else gamma
    norm = FollowingNorm(kind=kind, preimage=v, weight=weight, bias=beta, eps=1e-5)

    def normalise(x: torch.Tensor) -> torch.Tensor:
        if kind == "identity":
            return x
        if kind == "rmsnorm":
            rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-5)
            return gamma * (x / rms)
        return torch.nn.functional.layer_norm(x, (d,), gamma, beta, 1e-5)

    delta = torch.randn(n, d, dtype=torch.float64)
    delta = delta / delta.norm(dim=-1, keepdim=True) * 1e-4
    analytic = norm.jacobian_apply(delta)
    h = 1e-6
    numeric = (normalise(v + h * delta) - normalise(v - h * delta)) / (2 * h)
    assert torch.allclose(analytic, numeric, rtol=1e-5, atol=1e-8)


def test_following_norm_rejects_bad_inputs() -> None:
    v = torch.randn(3, 4)
    with pytest.raises(ProxyInputError, match="kind"):
        FollowingNorm(kind="batchnorm", preimage=v, weight=torch.ones(4))
    with pytest.raises(ProxyInputError, match="2-D"):
        FollowingNorm(kind="identity", preimage=torch.randn(3))
    with pytest.raises(ProxyInputError, match="eps"):
        FollowingNorm(kind="identity", preimage=v, eps=0.0)
    with pytest.raises(ProxyInputError, match="weight"):
        FollowingNorm(kind="layernorm", preimage=v, weight=None)
    with pytest.raises(ProxyInputError, match="same 2-D shape"):
        FollowingNorm(kind="identity", preimage=v).jacobian_apply(torch.randn(3, 5))


def test_layer_inputs_validation() -> None:
    w = torch.randn(8, 4)
    with pytest.raises(ProxyInputError, match="weight must be 2-D"):
        LayerInputs(weight=torch.randn(4), activations=torch.randn(3, 4))
    with pytest.raises(ProxyInputError, match="activations must be 2-D"):
        LayerInputs(weight=w, activations=torch.randn(4))
    with pytest.raises(ProxyInputError, match="width"):
        LayerInputs(weight=w, activations=torch.randn(3, 5))
    with pytest.raises(ProxyInputError, match="joint_activations"):
        LayerInputs(weight=w, activations=torch.randn(3, 4), joint_activations=torch.randn(2, 4))
    with pytest.raises(ProxyInputError, match="downstream_gain"):
        LayerInputs(weight=w, activations=torch.randn(3, 4), downstream_gain=-1.0)


def test_layer_input_error_matches_manual_computation() -> None:
    """``x W^T - x W_hat^T`` must equal ``x (W - W_hat)^T`` for the same compression."""
    torch.manual_seed(3)
    w = torch.randn(8, 16)
    x = torch.randn(4, 16)
    layer = LayerInputs(weight=w, activations=x)
    err = layer_input_error(layer, rank=4, spec=_spec())
    from spectraquant.proxies.operators import compression_delta

    expected = x @ compression_delta(w, 4, _spec()).transpose(0, 1)
    assert torch.allclose(err, expected, atol=1e-5)
