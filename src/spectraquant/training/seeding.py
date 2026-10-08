"""Deterministic seeding for reproducible CPU experiments.

One integer seeds Python's ``random``, NumPy and PyTorch, and turns on
``torch.use_deterministic_algorithms(True)`` so that a nondeterministic kernel raises instead of
silently producing different numbers. CPU thread count is pinned (default 1) because parallel
floating-point reductions can reorder and break bit-for-bit reproducibility.

Reproducibility contract (verified by ``tests/unit/test_seeding.py`` and
``tests/integration/test_smoke.py``): two runs at the same seed produce identical loss sequences.
"""

from __future__ import annotations

import os
import random
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch

__all__ = ["SeedState", "deterministic_context", "seed_everything", "seed_state"]

_UINT32_MAX = 2**32 - 1


@dataclass(frozen=True)
class SeedState:
    """The state applied by :func:`seed_everything`, for logging and manifests."""

    seed: int
    deterministic_algorithms: bool
    torch_threads: int
    cuda_deterministic: bool
    numpy_seed: int


def seed_everything(
    seed: int,
    *,
    deterministic: bool = True,
    threads: int = 1,
    warn_only: bool = False,
) -> SeedState:
    """Seed every RNG this project uses and configure deterministic kernels.

    Args:
        seed: master seed; must be a non-negative integer.
        deterministic: enable ``torch.use_deterministic_algorithms``.
        threads: ``torch.set_num_threads`` value; 1 keeps reductions bit-reproducible.
        warn_only: pass through to ``torch.use_deterministic_algorithms`` (warn instead of raise).
            Only useful for diagnosis; never use it for a recorded run.

    Returns:
        The :class:`SeedState` that was applied.

    Note:
        ``PYTHONHASHSEED`` is exported for child processes; it cannot retroactively change hashing
        in the already-running interpreter.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)

    numpy_seed = seed % (_UINT32_MAX + 1)
    random.seed(seed)
    np.random.seed(numpy_seed)
    torch.manual_seed(seed)

    cuda_available = torch.cuda.is_available()
    if cuda_available:  # pragma: no cover - no CUDA on the workstation of record
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = False

    if threads < 1:
        raise ValueError(f"threads must be >= 1, got {threads}")
    torch.set_num_threads(threads)

    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)

    return SeedState(
        seed=seed,
        deterministic_algorithms=deterministic,
        torch_threads=threads,
        cuda_deterministic=deterministic and cuda_available,
        numpy_seed=numpy_seed,
    )


def seed_state() -> SeedState:
    """Return the seeding configuration currently in effect (for logging)."""
    return SeedState(
        seed=int(torch.initial_seed()),
        deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()),
        torch_threads=int(torch.get_num_threads()),
        cuda_deterministic=bool(torch.backends.cudnn.deterministic),
        numpy_seed=0,
    )


@contextmanager
def deterministic_context(seed: int, *, threads: int = 1) -> Iterator[SeedState]:
    """Context manager that seeds, then restores the previous deterministic-algorithm setting."""
    previous = torch.are_deterministic_algorithms_enabled()
    previous_threads = torch.get_num_threads()
    state = seed_everything(seed, threads=threads)
    try:
        yield state
    finally:
        torch.use_deterministic_algorithms(previous)
        torch.set_num_threads(previous_threads)
