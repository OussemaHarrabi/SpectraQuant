"""Tests for the downstream-gain estimator, including analytic cases where the true gain is known."""

from __future__ import annotations

import pytest
import torch

from spectraquant.proxies.gain import GainEstimate, estimate_downstream_gains, rms


def test_rms_handles_empty_and_grad_tensors() -> None:
    assert rms(torch.zeros(0)) == 0.0
    x = torch.tensor([3.0, 4.0], requires_grad=True)
    assert rms(x) == pytest.approx(25.0**0.5 / 2**0.5)
    assert x.grad is None  # rms must not build a graph through the tensor


def test_identity_downstream_gain_is_one() -> None:
    """If the output is the layer's own contribution, the gain is exactly 1."""
    torch.manual_seed(0)
    pert = {"layer": torch.randn(8, 4)}
    estimate = estimate_downstream_gains(lambda name, p: p, pert)
    assert isinstance(estimate, GainEstimate)
    assert estimate.per_layer["layer"] == pytest.approx(1.0, rel=1e-12)
    assert estimate.forward_calls == 2  # baseline + one layer


def test_scaling_downstream_gain_is_the_scaling_factor() -> None:
    pert = {"a": torch.full((4, 3), 2.0), "b": torch.full((4, 3), 2.0)}
    estimate = estimate_downstream_gains(lambda name, p: 3.0 * p, pert)
    assert estimate.per_layer["a"] == pytest.approx(3.0, rel=1e-12)
    assert estimate.per_layer["b"] == pytest.approx(3.0, rel=1e-12)


def test_linear_chain_gain_is_the_product_of_the_downstream_matrices() -> None:
    """For a two-layer linear chain the gain equals the product's RMS gain, which we compute directly."""
    torch.manual_seed(1)
    m1 = torch.randn(6, 6, dtype=torch.float64)
    m2 = torch.randn(5, 6, dtype=torch.float64)
    pert = {"first": torch.randn(10, 6, dtype=torch.float64)}

    def forward(name: str, p: torch.Tensor) -> torch.Tensor:
        return (p @ m1.transpose(0, 1)) @ m2.transpose(0, 1)

    estimate = estimate_downstream_gains(forward, pert)
    expected = rms(pert["first"] @ m1.transpose(0, 1) @ m2.transpose(0, 1)) / rms(pert["first"])
    assert estimate.per_layer["first"] == pytest.approx(expected, rel=1e-10)


def test_zero_injection_is_reported_as_zero_gain_not_an_error() -> None:
    pert = {"layer": torch.zeros(3, 2)}
    estimate = estimate_downstream_gains(lambda name, p: p * 5.0, pert)
    assert estimate.per_layer["layer"] == 0.0
    assert estimate.diagnostics["layer/undefined_zero_injection"] == 1.0


def test_empty_input_returns_an_empty_estimate() -> None:
    estimate = estimate_downstream_gains(lambda name, p: p, {})
    assert estimate.per_layer == {}
    assert estimate.forward_calls == 0


def test_diagnostics_record_the_inputs_of_the_estimate() -> None:
    pert = {"layer": torch.full((2, 2), 4.0)}
    estimate = estimate_downstream_gains(lambda name, p: 0.5 * p, pert)
    assert estimate.diagnostics["layer/injected_rms"] == pytest.approx(4.0)
    assert estimate.diagnostics["layer/output_rms"] == pytest.approx(2.0)
    assert estimate.forward_calls == 2
