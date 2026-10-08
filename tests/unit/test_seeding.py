"""Deterministic-seeding tests: identical seeds must give identical numbers."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from spectraquant.training.seeding import deterministic_context, seed_everything


def _tiny_training_losses(seed: int) -> list[float]:
    """Train a throwaway MLP for 5 steps and return the loss sequence."""
    seed_everything(seed, deterministic=True, threads=1)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    generator = torch.Generator().manual_seed(0)
    inputs = torch.randn(16, 8, generator=generator)
    targets = torch.randn(16, 4, generator=generator)

    losses: list[float] = []
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        loss = ((model(inputs) - targets) ** 2).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def test_same_seed_gives_identical_loss_sequences() -> None:
    assert _tiny_training_losses(11) == _tiny_training_losses(11)


def test_different_seed_changes_the_run() -> None:
    assert _tiny_training_losses(11) != _tiny_training_losses(12)


def test_python_and_numpy_are_seeded() -> None:
    seed_everything(5)
    first = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    seed_everything(5)
    second = (random.random(), float(np.random.rand()), float(torch.rand(1)))

    assert first == second


def test_deterministic_algorithms_enabled_and_threads_pinned() -> None:
    state = seed_everything(3)

    assert torch.are_deterministic_algorithms_enabled()
    assert torch.get_num_threads() == 1
    assert state.torch_threads == 1
    assert state.seed == 3
    assert state.numpy_seed == 3


def test_negative_seed_is_rejected() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        seed_everything(-1)


def test_non_integer_seed_is_rejected() -> None:
    with pytest.raises(TypeError, match="int"):
        seed_everything(1.5)  # type: ignore[arg-type]


def test_zero_threads_is_rejected() -> None:
    with pytest.raises(ValueError, match="threads"):
        seed_everything(1, threads=0)


def test_deterministic_context_restores_previous_settings() -> None:
    seed_everything(1, deterministic=False, threads=2)
    assert not torch.are_deterministic_algorithms_enabled()

    with deterministic_context(42, threads=1) as state:
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.get_num_threads() == 1
        assert state.seed == 42

    assert not torch.are_deterministic_algorithms_enabled()
    assert torch.get_num_threads() == 2
