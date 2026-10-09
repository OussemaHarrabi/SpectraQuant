"""Unit tests for the factorized linear carrier of low-rank preparation (Milestone 5)."""

from __future__ import annotations

import pytest
import torch

from spectraquant.factorization import truncated_svd
from spectraquant.quantization import QuantSpec, accounted_bytes, fake_quantize
from spectraquant.training.low_rank import (
    LowRankLinear,
    check_against_dense,
    clone_with_weights,
    deployed_weights,
    factor_storage_bytes,
    factorize_linears,
    gather_factor_pairs,
    low_rank_layer_names,
    parameter_counts,
    sequence_shape_report,
)


def _spec() -> QuantSpec:
    return QuantSpec(bits=4, granularity="per_group", group_size=4, symmetric=True)


def _dense_model(seed: int = 0) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(8, 6, bias=True),
        torch.nn.ReLU(),
        torch.nn.Linear(6, 3, bias=False),
        torch.nn.Linear(3, 3, bias=True),
    ).to(torch.float32)


def test_low_rank_linear_forward_matches_the_dense_equivalent() -> None:
    layer = LowRankLinear(5, 4, 2)
    with torch.no_grad():
        layer.A.copy_(torch.randn(2, 5, generator=torch.Generator().manual_seed(1)))
        layer.B.copy_(torch.randn(4, 2, generator=torch.Generator().manual_seed(2)))
        layer.bias.copy_(torch.randn(4, generator=torch.Generator().manual_seed(3)))
    inputs = torch.randn(3, 5, generator=torch.Generator().manual_seed(4))
    dense = torch.nn.functional.linear(inputs, layer.effective_weight(), layer.bias)
    assert torch.allclose(layer(inputs), dense, atol=1e-6)
    assert layer.rank == 2
    assert "rank=2" in layer.extra_repr()


def test_low_rank_linear_validation() -> None:
    with pytest.raises(ValueError):
        LowRankLinear(4, 4, 9)
    with pytest.raises(ValueError):
        LowRankLinear(4, 4, 0)
    with pytest.raises(ValueError):
        LowRankLinear(4, 4, 2, device="cuda")


def test_from_linear_reproduces_the_truncated_svd() -> None:
    torch.manual_seed(5)
    linear = torch.nn.Linear(9, 7)
    factorized = LowRankLinear.from_linear(linear, 3)
    expected = truncated_svd(linear.weight.detach(), 3)
    assert torch.allclose(factorized.effective_weight(), expected.B @ expected.A, atol=1e-6)
    assert factorized.bias is not None
    assert torch.allclose(factorized.bias, linear.bias.detach())


def test_factorize_linears_replaces_every_linear_and_keeps_references() -> None:
    model = _dense_model(11)
    originals = {
        name: module.weight.detach().clone()
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    inputs = torch.randn(4, 8, generator=torch.Generator().manual_seed(12))
    reference = _dense_model(11)
    with torch.no_grad():
        before = reference(inputs)

    # Per-layer ranks that capture each dense weight exactly, so the factorized forward must match
    # the dense reference; the SVD initialisation is only exact when the rank captures the weight.
    references = factorize_linears(model, rank=3, ranks={"0": 6})
    assert set(references) == set(originals)
    for name, weight in references.items():
        assert torch.allclose(weight, originals[name], atol=0.0)
    assert set(low_rank_layer_names(model)) == set(originals)
    assert check_against_dense(model, reference, inputs, atol=1e-5) < 1e-5

    aggressive = _dense_model(11)
    factorize_linears(aggressive, rank=1)
    with pytest.raises(AssertionError):
        check_against_dense(aggressive, reference, inputs, atol=1e-5)
    # The un-factorized parameters are untouched.
    with torch.no_grad():
        after = model(inputs)
    assert after.shape == before.shape


def test_factorize_linears_rank_overrides_and_exclusions() -> None:
    model = _dense_model(13)
    references = factorize_linears(model, rank=2, ranks={"2": 3}, exclude=("2",))
    assert "2" not in references
    assert "0" in references
    assert isinstance(model[0], LowRankLinear)
    assert isinstance(model[2], torch.nn.Linear)

    with pytest.raises(ValueError):
        factorize_linears(_dense_model(13), rank=2, ranks={"nope": 2})
    with pytest.raises(ValueError):
        factorize_linears(_dense_model(13), rank=99)
    with pytest.raises(ValueError):
        factorize_linears(_dense_model(13), rank=2, exclude=("0", "2", "3"))
    with pytest.raises(ValueError):
        factorize_linears(torch.nn.Sequential(torch.nn.ReLU()), rank=2)


def test_gather_factor_pairs_requires_factorized_layers() -> None:
    with pytest.raises(ValueError):
        gather_factor_pairs(_dense_model(14))
    model = _dense_model(14)
    factorize_linears(model, rank=2)
    pairs = gather_factor_pairs(model)
    assert set(pairs) == set(low_rank_layer_names(model))
    for a, b in pairs.values():
        assert a.shape[0] == b.shape[1]


def test_deployed_weights_and_storage_bytes() -> None:
    spec = _spec()
    model = _dense_model(15)
    factorize_linears(model, rank=2)
    prepared = deployed_weights(model, spec, quantize=False)
    quantized = deployed_weights(model, spec, quantize=True)
    expected = 0
    for name, module in model.named_modules():
        if not isinstance(module, LowRankLinear):
            continue
        a, b = module.factor_pair()
        assert torch.allclose(prepared[name], (b @ a).double(), atol=1e-6)
        assert torch.allclose(
            quantized[name],
            (fake_quantize(b.double(), spec) @ fake_quantize(a.double(), spec)),
            atol=1e-6,
        )
        expected += accounted_bytes(tuple(a.shape), spec) + accounted_bytes(tuple(b.shape), spec)
    assert factor_storage_bytes(model, spec) == expected

    with pytest.raises(ValueError):
        deployed_weights(_dense_model(15), spec, quantize=True)
    with pytest.raises(ValueError):
        factor_storage_bytes(torch.nn.Sequential(torch.nn.ReLU()), spec)


def test_clone_with_weights_writes_into_a_dense_copy_only() -> None:
    model = _dense_model(16)
    weights = {"0": torch.zeros(6, 8)}
    clone = clone_with_weights(model, weights)
    assert torch.equal(clone[0].weight, torch.zeros(6, 8))
    assert not torch.equal(model[0].weight, torch.zeros(6, 8))
    with pytest.raises(ValueError):
        clone_with_weights(model, {"0": torch.zeros(5, 8)})
    with pytest.raises(ValueError):
        clone_with_weights(model, {"1": torch.zeros(6, 8)})


def test_parameter_counting_helpers() -> None:
    model = _dense_model(17)
    factorize_linears(model, rank=2)
    counts = parameter_counts(model)
    assert counts["total"] == sum(p.numel() for p in model.parameters())
    assert 0 < counts["factorized"] <= counts["total"]
    rows = sequence_shape_report(model)
    assert [row[0] for row in rows] == sorted(row[0] for row in rows)
    for name, out_features, in_features, rank in rows:
        assert rank == 2
        assert (out_features, in_features) == tuple(
            model.get_submodule(name).effective_weight().shape
        )
