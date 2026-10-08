"""Packed low-bit storage (measurement class 3).

Milestone 2 (owner: quantization stream). Packing routines must serialize real bytes so that
``packed_bytes`` in a run manifest is a *measured* number, not an analytical estimate. Class 3 is
available locally only for formats this repository can actually serialize.
"""

from __future__ import annotations

from pathlib import Path

import torch

__all__ = ["pack_int4", "pack_int8", "unpack"]


def pack_int8(codes: torch.Tensor) -> bytes:
    """Serialize int8 codes to bytes (one byte per element).

    Raises:
        NotImplementedError: Milestone 2 has not landed (quantization stream).
    """
    raise NotImplementedError("pack_int8 is a Milestone 2 deliverable (quantization stream)")


def pack_int4(codes: torch.Tensor) -> bytes:
    """Serialize 4-bit codes two-per-byte, plus the group metadata needed to unpack.

    Raises:
        NotImplementedError: Milestone 2 has not landed (quantization stream).
    """
    raise NotImplementedError("pack_int4 is a Milestone 2 deliverable (quantization stream)")


def unpack(path: Path | str) -> torch.Tensor:
    """Read a packed file written by :func:`pack_int8` / :func:`pack_int4` back into codes.

    Raises:
        NotImplementedError: Milestone 2 has not landed (quantization stream).
    """
    raise NotImplementedError("unpack is a Milestone 2 deliverable (quantization stream)")
