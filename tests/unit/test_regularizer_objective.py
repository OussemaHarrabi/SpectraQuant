"""Unit tests for the rounding-aware spectral preparation objective (Milestone 5, H3).

The tests here are the contract of ``spectraquant.regularizers.spectral``:

* every term is differentiable and gradients reach **both** factors;
* the rounding terms are exactly zero on the quantization grid;
* the coefficients scale their terms linearly;
* increasing ``lambda_round`` **reduces the measured rounding residual** (a direction test on a toy
  problem — not a threshold);
* the spectral term reduces the tail energy it claims to;
* the grid-step expansion agrees with the quantizer's own (frozen) expansion rule.
"""

from __future__ import annotations

import math
from itertools import pairwise

import pytest
import torch

from spectraquant.factorization import truncated_svd
from spectraquant.quantization import QuantSpec, fake_dequantize, fake_quantize, quant_params
from spectraquant.regularizers.spectral import (
    PreparationObjective,
    _grid_step,
    effective_rank,
    factorization_proxy,
    measured_factor_rounding_residual,
    measured_product_rounding_residual,
    orthogonality_penalty,
    rounding_grid_penalty,
    rounding_residual_ratio,
    spectral_penalty,
    spectral_tail_energy,
)

OUT, IN, RANK = 12, 10, 4


def _spec(bits: int = 4, granularity: str = "per_group", group_size: int | None = 4) -> QuantSpec:
    return QuantSpec(
        bits=bits, granularity=granularity, group_size=group_size, symmetric=True, axis=0
    )


def _factors(seed: int = 0, rank: int = RANK, dtype: torch.dtype = torch.float32):
    generator = torch.Generator().manual_seed(seed)
    a = torch.randn(rank, IN, generator=generator, dtype=dtype) * 0.4
    b = torch.randn(OUT, rank, generator=generator, dtype=dtype) * 0.4
    return {"layer": (a, b)}


def _grid_aligned(shape: tuple[int, int], spec: QuantSpec) -> torch.Tensor:
    """Integer-valued float matrix whose *every* quantization block has ``max|w| == qmax``.

    The block scale is then exactly ``1.0``, so ``round(w / 1) == w`` and every element is already on
    the code grid — the property the "exactly zero" tests rely on.
    """
    generator = torch.Generator().manual_seed(17)
    values = torch.randint(
        -(spec.qmax - 1), spec.qmax, shape, generator=generator, dtype=torch.int64
    ).to(torch.float32)
    rows, _columns = shape
    if spec.granularity == "per_tensor":
        values[0, 0] = float(spec.qmax)
    elif spec.granularity == "per_channel":
        values[:, 0] = float(spec.qmax)
    elif spec.granularity == "per_group":
        assert spec.group_size is not None
        # A per-group block spans a group of rows *and one column* (the quantizer's inner axis is the
        # trailing one), so every (row-group, column) block needs its own ``max|w| == qmax``.
        for start in range(0, rows, spec.group_size):
            values[start, :] = float(spec.qmax)
    else:  # pragma: no cover - defensive
        raise AssertionError(spec.granularity)
    return values


# --------------------------------------------------------------------------------------
# Gradients
# --------------------------------------------------------------------------------------
def test_gradients_flow_to_both_factors_and_are_finite() -> None:
    factors = _factors(seed=1)
    reference = {"layer": torch.randn(OUT, IN, generator=torch.Generator().manual_seed(3))}
    objective = PreparationObjective(
        spec=_spec(),
        lambda_factor=1.0,
        lambda_round=1.0,
        lambda_residual=1.0,
        lambda_spectrum=1.0,
        tail_rank=2,
    )
    a, b = factors["layer"]
    a.requires_grad_(True)
    b.requires_grad_(True)
    terms = objective.terms({"layer": (a, b)}, references=reference)
    terms.total.backward()

    assert a.grad is not None and b.grad is not None
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()
    assert float(a.grad.abs().sum()) > 0.0 and float(b.grad.abs().sum()) > 0.0
    for name in ("factorization", "rounding", "residual", "spectrum", "total"):
        value = getattr(terms, name)
        assert value.ndim == 0
        assert torch.isfinite(value.detach())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lambda_factor": 1.0},
        {"lambda_round": 1.0},
        {"lambda_residual": 1.0},
        {"lambda_spectrum": 1.0, "tail_rank": 2},
    ],
)
def test_every_term_alone_keeps_gradients_finite(kwargs: dict[str, float]) -> None:
    factors = _factors(seed=2)
    reference = {"layer": torch.randn(OUT, IN, generator=torch.Generator().manual_seed(5))}
    objective = PreparationObjective(spec=_spec(), **kwargs)
    a, b = factors["layer"]
    a.requires_grad_(True)
    b.requires_grad_(True)
    objective({"layer": (a, b)}, references=reference).backward()
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()


def test_repeated_singular_values_do_not_produce_nan_gradients() -> None:
    """A rank-deficient product has repeated (zero) singular values; the tail term must stay finite."""
    a = torch.randn(RANK, IN, requires_grad=True)
    b = torch.randn(OUT, RANK, requires_grad=True)
    spectral_tail_energy({"layer": (a, b)}, tail_rank=2).backward()
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()


# --------------------------------------------------------------------------------------
# Exact zeros on the grid
# --------------------------------------------------------------------------------------
def test_rounding_residual_is_exactly_zero_when_factors_are_on_the_grid() -> None:
    spec = _spec(granularity="per_tensor", group_size=None)
    a = _grid_aligned((RANK, IN), spec)
    b = _grid_aligned((OUT, RANK), spec)
    assert torch.equal(fake_quantize(a, spec), a)
    assert torch.equal(fake_quantize(b, spec), b)

    factors = {"layer": (a, b)}
    assert float(rounding_residual_ratio(factors, spec)) == 0.0
    assert measured_product_rounding_residual(factors, spec) == 0.0
    assert measured_factor_rounding_residual(factors, spec) == 0.0
    # The smooth surrogate is *not* bitwise zero: sin(pi * integer) is ~1e-16 in floating point.
    assert float(rounding_grid_penalty(factors, spec)) < 1e-24


@pytest.mark.parametrize(
    "spec",
    [
        _spec(granularity="per_tensor", group_size=None),
        _spec(granularity="per_channel", group_size=None),
        _spec(granularity="per_group", group_size=4),
        _spec(granularity="per_group", group_size=3),
    ],
)
def test_grid_alignment_zeroes_the_activation_of_the_penalty(spec: QuantSpec) -> None:
    a = _grid_aligned((RANK, IN), spec)
    b = _grid_aligned((OUT, RANK), spec)
    factors = {"layer": (a, b)}
    assert measured_factor_rounding_residual(factors, spec) == 0.0
    objective = PreparationObjective(spec=spec, lambda_round=1.0, lambda_residual=1.0)
    terms = objective.terms(factors)
    assert float(terms.residual) == 0.0
    assert float(terms.rounding) < 1e-24
    assert float(terms.total) < 1e-24


def test_grid_step_matches_the_quantizer_expansion() -> None:
    """The mirrored block expansion must agree with the quantizer's own, for every granularity."""
    for spec, weight in (
        (_spec(granularity="per_tensor", group_size=None), torch.randn(OUT, IN)),
        (_spec(granularity="per_channel", group_size=None), torch.randn(OUT, IN)),
        (_spec(granularity="per_group", group_size=4), torch.randn(OUT, IN)),
        (_spec(granularity="per_group", group_size=3), torch.randn(OUT, IN)),
        (
            QuantSpec(bits=8, granularity="per_group", group_size=2, symmetric=False, axis=1),
            torch.randn(OUT, IN),
        ),
    ):
        params = quant_params(weight, spec)
        # Probe in float64: ``(1 - zp) * s - (0 - zp) * s`` cancels at float32 precision for wide
        # asymmetric ranges, while in float64 both products are exact and the difference is the step.
        wide = weight.to(torch.float64)
        probe = fake_dequantize(torch.ones_like(wide), params, spec) - fake_dequantize(
            torch.zeros_like(wide), params, spec
        )
        _, step = _grid_step(weight, spec)
        assert torch.equal(step.reshape(weight.shape).to(torch.float64), probe), spec


# --------------------------------------------------------------------------------------
# Coefficients
# --------------------------------------------------------------------------------------
def test_terms_scale_linearly_with_their_coefficient() -> None:
    factors = _factors(seed=4)
    reference = {"layer": torch.randn(OUT, IN, generator=torch.Generator().manual_seed(6))}
    base = PreparationObjective(spec=_spec(), lambda_round=1.0)
    doubled = PreparationObjective(spec=_spec(), lambda_round=2.0)
    base_terms = base.terms(factors, references=reference)
    doubled_terms = doubled.terms(factors, references=reference)
    assert float(doubled_terms.total) == pytest.approx(2.0 * float(base_terms.total), rel=1e-6)
    assert float(doubled_terms.rounding) == pytest.approx(float(base_terms.rounding), rel=1e-6)


def test_noop_objective_is_an_exact_zero_with_gradients() -> None:
    factors = {"layer": tuple(t.requires_grad_(True) for t in _factors(seed=5)["layer"])}
    objective = PreparationObjective(spec=_spec())
    assert objective.is_noop()
    value = objective(factors)
    assert float(value.detach()) == 0.0
    value.backward()
    a, b = factors["layer"]
    assert a.grad is not None and float(a.grad.abs().sum()) == 0.0
    assert b.grad is not None and float(b.grad.abs().sum()) == 0.0


def test_total_equals_the_weighted_sum_of_components() -> None:
    factors = _factors(seed=6)
    reference = {"layer": torch.randn(OUT, IN, generator=torch.Generator().manual_seed(7))}
    objective = PreparationObjective(
        spec=_spec(),
        lambda_factor=0.5,
        lambda_round=2.0,
        lambda_residual=3.0,
        lambda_spectrum=4.0,
        tail_rank=2,
    )
    terms = objective.terms(factors, references=reference)
    expected = (
        0.5 * float(terms.factorization)
        + 2.0 * float(terms.rounding)
        + 3.0 * float(terms.residual)
        + 4.0 * float(terms.spectrum)
    )
    assert float(terms.total) == pytest.approx(expected, rel=1e-6, abs=1e-12)
    payload = terms.to_dict()
    assert payload["lambda_round"] == 2.0
    assert payload["lambda_factor"] == 0.5
    assert payload["total"] == pytest.approx(float(terms.total), rel=1e-9)


def test_monitored_residual_matches_its_measurement() -> None:
    """The monitored STE term and the no-grad measured residual are the same quantity forward."""
    factors = _factors(seed=7)
    spec = _spec()
    proxy = float(rounding_residual_ratio(factors, spec))
    measured = measured_product_rounding_residual(factors, spec)
    assert math.sqrt(proxy) == pytest.approx(measured, rel=1e-5)


# --------------------------------------------------------------------------------------
# Direction tests
# --------------------------------------------------------------------------------------
def _toy_regression(lam: float, *, seed: int = 0, steps: int = 300, lr: float = 0.01):
    """Toy fit of a random target through a rank-4 map, plus ``lam`` times the rounding penalty."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 1)
    target = torch.randn(OUT, IN, generator=generator) * 0.7
    inputs = torch.randn(24, IN, generator=generator)
    outputs = torch.randn(24, OUT, generator=generator)
    factors = truncated_svd(target, RANK)
    a = factors.A.detach().clone().requires_grad_(True)
    b = factors.B.detach().clone().requires_grad_(True)
    objective = PreparationObjective(spec=_spec(), lambda_round=lam)
    optimizer = torch.optim.SGD([a, b], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        weight = b @ a
        task = ((inputs @ weight.transpose(0, 1) - outputs) ** 2).mean()
        (task + objective({"layer": (a, b)})).backward()
        optimizer.step()
    return measured_factor_rounding_residual({"layer": (a, b)}, _spec())


def test_increasing_lambda_round_reduces_the_measured_rounding_residual() -> None:
    """Direction test: the measured (hard-rounding) residual falls as ``lambda_round`` rises."""
    residuals = [_toy_regression(lam) for lam in (0.0, 2.0, 10.0, 50.0)]
    assert all(later < earlier for earlier, later in pairwise(residuals)), residuals
    assert residuals[-1] < 0.5 * residuals[0], residuals


def test_spectral_term_reduces_the_tail_energy_it_claims() -> None:
    spec = _spec()
    torch.manual_seed(3)
    a = torch.randn(8, IN, requires_grad=True)
    b = torch.randn(OUT, 8, requires_grad=True)

    def measured_tail() -> float:
        with torch.no_grad():
            energy = torch.linalg.svdvals(b.detach() @ a.detach()).pow(2)
            return float(energy[4:].sum() / energy.sum())

    before = measured_tail()
    objective = PreparationObjective(spec=spec, lambda_spectrum=1.0, tail_rank=4)
    optimizer = torch.optim.SGD([a, b], lr=0.05)
    for _ in range(200):
        optimizer.zero_grad()
        objective({"layer": (a, b)}).backward()
        optimizer.step()
    after = measured_tail()
    assert after < 0.75 * before, (before, after)


def test_tail_energy_is_numerically_zero_when_tail_rank_covers_the_rank() -> None:
    """A rank-``r`` product carries no energy beyond ``r``; only float32 SVD noise remains."""
    factors = _factors(seed=8)
    assert float(spectral_tail_energy(factors, tail_rank=RANK)) < 1e-12
    assert float(spectral_tail_energy(factors, tail_rank=9)) < 1e-12
    smaller = {"layer": (factors["layer"][0][:2], factors["layer"][1][:, :2])}
    assert float(spectral_tail_energy(smaller, tail_rank=2)) < 1e-12


# --------------------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------------------
def test_objective_is_deterministic() -> None:
    reference = {"layer": torch.randn(OUT, IN, generator=torch.Generator().manual_seed(11))}
    objective = PreparationObjective(spec=_spec(), lambda_round=1.0, lambda_factor=1.0)
    first = objective.terms(_factors(seed=9), references=reference).to_dict()
    second = objective.terms(_factors(seed=9), references=reference).to_dict()
    assert first == second
    assert _toy_regression(10.0, seed=1, steps=40) == _toy_regression(10.0, seed=1, steps=40)


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------
def test_objective_rejects_invalid_configurations() -> None:
    with pytest.raises(TypeError):
        PreparationObjective(spec="not-a-spec")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        PreparationObjective(spec=_spec(), lambda_round=-1.0)
    with pytest.raises(ValueError):
        PreparationObjective(spec=_spec(), lambda_round=float("inf"))
    with pytest.raises(ValueError):
        PreparationObjective(spec=_spec(), lambda_spectrum=1.0)
    with pytest.raises(ValueError):
        PreparationObjective(spec=_spec(), tail_rank=2)
    with pytest.raises(ValueError):
        PreparationObjective(spec=_spec(), lambda_spectrum=1.0, tail_rank=0)
    with pytest.raises(ValueError):
        PreparationObjective(spec=_spec(), name="")


def test_objective_rejects_malformed_factors() -> None:
    objective = PreparationObjective(spec=_spec(), lambda_round=1.0)
    with pytest.raises(ValueError):
        objective({})
    with pytest.raises(ValueError):
        objective({"layer": (torch.randn(RANK, IN), torch.randn(OUT, RANK + 1))})
    with pytest.raises(ValueError):
        objective({"layer": (torch.randn(RANK), torch.randn(OUT, RANK))})
    with pytest.raises(TypeError):
        objective({"layer": (torch.zeros(RANK, IN, dtype=torch.int64), torch.randn(OUT, RANK))})
    with pytest.raises(KeyError):
        objective({"layer": _factors()["layer"]}, references={})
    with pytest.raises(ValueError):
        objective(
            _factors(),
            references={"layer": torch.randn(OUT + 1, IN)},
        )


def test_factorization_proxy_is_zero_without_a_reference_and_positive_with_one() -> None:
    factors = _factors(seed=11)
    assert float(factorization_proxy(factors)) == 0.0
    reference = {"layer": torch.randn(OUT, IN, generator=torch.Generator().manual_seed(13))}
    assert float(factorization_proxy(factors, reference)) > 0.0


def test_float64_factors_are_supported() -> None:
    factors = _factors(seed=12, dtype=torch.float64)
    objective = PreparationObjective(spec=_spec(), lambda_round=1.0)
    terms = objective.terms(factors)
    assert terms.total.dtype == torch.float64
    assert torch.isfinite(terms.total)


# --------------------------------------------------------------------------------------
# Retained model-level API
# --------------------------------------------------------------------------------------
def test_effective_rank_equals_the_participation_ratio_of_the_spectrum() -> None:
    """Entropy-based effective rank: 4 for four equal singular values, 5 for a flat 5-spectrum."""
    orthogonal, _ = torch.linalg.qr(torch.randn(6, 6, generator=torch.Generator().manual_seed(21)))
    four_equal = (
        orthogonal @ torch.diag(torch.tensor([2.0, 2.0, 2.0, 2.0, 0.0, 0.0])) @ orthogonal.T
    )
    assert effective_rank(four_equal) == pytest.approx(4.0, abs=1e-6)
    assert effective_rank(torch.eye(5)) == pytest.approx(5.0, abs=1e-6)
    assert effective_rank(torch.zeros(4, 4)) == 0.0
    with pytest.raises(ValueError):
        effective_rank(torch.randn(4))
    with pytest.raises(TypeError):
        effective_rank("nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        effective_rank(torch.randn(4, 4), tolerance=0.0)


def test_orthogonality_penalty_is_scale_invariant_and_minimal_for_orthonormal_factors() -> None:
    identity = torch.eye(4)
    zero = orthogonality_penalty((identity, identity))
    assert float(zero) == pytest.approx(0.0, abs=1e-12)
    assert float(orthogonality_penalty((identity * 1e3, identity * 1e-3))) == pytest.approx(
        0.0, abs=1e-12
    )
    generator = torch.Generator().manual_seed(22)
    assert float(orthogonality_penalty((torch.randn(4, 4, generator=generator), identity))) > 0.0
    with pytest.raises(ValueError):
        orthogonality_penalty((identity, identity), strength=-1.0)


def test_spectral_penalty_tracks_the_discarded_energy() -> None:
    model = torch.nn.Sequential(
        torch.nn.Linear(6, 4, bias=False), torch.nn.Linear(8, 5, bias=False)
    )
    with torch.no_grad():
        model[0].weight.copy_(torch.eye(4, 6))
        model[1].weight.copy_(torch.eye(5, 8))
    # Layer 0 is a rank-4 isometry over 6 inputs and layer 1 a rank-5 isometry over 8 inputs.
    assert float(spectral_penalty(model, target_rank=5).detach()) < 1e-12
    # target_rank=1 discards 3/4 of layer 0's energy and 4/5 of layer 1's.
    assert float(spectral_penalty(model, target_rank=1).detach()) == pytest.approx(
        0.75 + 0.8, abs=1e-6
    )

    random_model = torch.nn.Sequential(torch.nn.Linear(6, 8, bias=False))
    base = float(spectral_penalty(random_model, target_rank=1).detach())
    assert base > 0.0
    assert float(
        spectral_penalty(random_model, target_rank=1, strength=2.0).detach()
    ) == pytest.approx(2.0 * base, rel=1e-6)
    with pytest.raises(ValueError):
        spectral_penalty(random_model, target_rank=1, exclude=("0",))
    with pytest.raises(ValueError):
        spectral_penalty(random_model, target_rank=0)
    with pytest.raises(ValueError):
        spectral_penalty(torch.nn.Sequential(), target_rank=1)


def test_from_config_round_trips_the_coefficients() -> None:
    from spectraquant.config import RegularizerConfig

    config = RegularizerConfig(
        name="full",
        lambda_factor=1.0,
        lambda_round=0.5,
        lambda_residual=0.25,
        lambda_spectrum=2.0,
        tail_rank=2,
    )
    objective = PreparationObjective.from_config(config, spec=_spec())
    assert objective.name == "full"
    assert objective.weights() == {
        "lambda_factor": 1.0,
        "lambda_round": 0.5,
        "lambda_residual": 0.25,
        "lambda_spectrum": 2.0,
    }
    assert "tail_rank=2" in objective.extra_repr()
