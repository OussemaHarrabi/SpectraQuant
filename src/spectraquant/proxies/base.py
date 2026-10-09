"""Frozen proxy interface for Milestone 2 (``design-m2-interfaces.md`` section 3, as amended by
section 7).

This module holds the load-bearing contract shared by every sensitivity proxy:

* :class:`ProxyResult` — the single return type, in the **one proxy unit** of invariant 3: the
  per-layer squared Frobenius norm of the *layer output* error
  ``E_X ||X W^T - X (Q(B) Q(A))^T||^2``, aggregated across layers by summation, plus the
  approximation label (``exact``) and the measurement class required by invariant 4.
* :class:`Proxy` — the frozen protocol (``name``, ``score_layer``, ``score_model``).
* :class:`LayerInputs` — what a proxy may know about one linear layer beyond its weight.
* :class:`FollowingNorm` — the normalisation that follows a layer's sub-layer, with its **exact**
  Jacobian. Section 7.1 of the review shows a proxy computed on the pre-normalisation activation is
  not the quantity that propagates, so the post-normalisation basis has to be expressible.

Measurement classes follow ``AGENTS.md`` section 5: class 1 is an analytical estimate (derived from
shapes / spectra), class 2 is fake-quantization quality (float execution that simulates quantization
numerics). Nothing here imports a CUDA kernel; the only device is CPU.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from spectraquant.quantization import QuantSpec

__all__ = [
    "NORM_KINDS",
    "FollowingNorm",
    "LayerInputs",
    "Proxy",
    "ProxyInputError",
    "ProxyResult",
    "layer_input_error",
]

#: Normalisation families :class:`FollowingNorm` knows how to differentiate exactly.
NORM_KINDS: tuple[str, ...] = ("identity", "layernorm", "rmsnorm")


class ProxyInputError(ValueError):
    """Raised when a proxy is handed inputs it cannot score (bad shapes, mismatched mappings)."""


@dataclass(frozen=True)
class ProxyResult:
    """The value of a proxy, in the common unit, for one layer or one model.

    Attributes:
        value: squared-Frobenius layer-output error in the invariant-3 unit, summed over the scored
            layers (per-layer values are in :attr:`per_layer`). For a single-layer score the
            quantity is the mean over activation samples of ``||x W^T - x (Q(B) Q(A))^T||^2``.
        exact: ``True`` only when the returned number *is* the defined quantity for the empirical
            activation distribution that was scored — i.e. no surrogate, no linearisation and no
            Monte-Carlo estimate separates it from the definition. Plug-in averages over a finite
            batch are ``exact=True`` for that batch but still carry sampling error relative to the
            population; that error is reported in ``diagnostics`` (``sampling_rel_se_*``) rather
            than hidden behind ``exact=False``.
        measurement_class: 1 (analytical estimate) or 2 (fake-quantization quality). Never 3-5: a
            proxy is a float quality estimate, never a storage or latency measurement.
        per_layer: ``name -> value`` for every scored layer. :meth:`Proxy.score_layer` uses the
            single key ``"layer"``.
        diagnostics: flat ``str -> float`` map, JSON-serializable and finite. Per-layer entries are
            namespaced ``"<layer>/<key>"``. Non-finite values are omitted rather than clamped.

    Raises:
        ValueError: if ``measurement_class`` is not 1 or 2, or ``value`` is not a finite float.
    """

    value: float
    exact: bool
    measurement_class: int
    per_layer: dict[str, float]
    diagnostics: dict[str, float]

    def __post_init__(self) -> None:
        if self.measurement_class not in (1, 2):
            raise ValueError(
                f"measurement_class must be 1 (analytical) or 2 (fake-quantization), "
                f"got {self.measurement_class}"
            )
        if not isinstance(self.value, float) or self.value != self.value:
            raise ValueError(f"value must be a finite float, got {self.value!r}")
        # Defensive copies: the frozen interface promises a plain dict, and a caller must not be
        # able to mutate a result through a shared reference.
        object.__setattr__(self, "per_layer", dict(self.per_layer))
        object.__setattr__(self, "diagnostics", {k: float(v) for k, v in self.diagnostics.items()})

    def ratio_to(self, other: ProxyResult) -> float:
        """``self.value / other.value``; ``inf`` when ``other.value`` is zero (documented, not an
        error: a zero-damage denominator is a legitimate degenerate case)."""
        if other.value == 0.0:
            return float("inf") if self.value != 0.0 else 0.0
        return self.value / other.value


@dataclass(frozen=True)
class FollowingNorm:
    """The normalisation that follows a layer's sub-layer, able to apply its **exact** Jacobian.

    Section 7.1 of the adversarial review: ``LayerNorm``/``RMSNorm`` sit between the layer whose
    error is scored and the output and rescale the residual stream, so a proxy computed on the
    pre-normalisation activation is not the quantity that propagates. This container carries the
    normalisation's operating point and its parameters so the in-situ variant can map an error from
    the residual basis into the normalisation's output basis.

    Args:
        kind: one of :data:`NORM_KINDS`.
        preimage: ``(n_samples, d)`` float tensor — the tensor the normalisation consumes,
            ``residual + sublayer_output``, evaluated on the calibration batch in the **uncompressed**
            forward pass.
        weight: ``(d,)`` gain ``gamma`` for ``"layernorm"``/``"rmsnorm"``, ``None`` for
            ``"identity"``.
        bias: ``(d,)`` offset ``beta`` for ``"layernorm"``, ``None`` otherwise.
        eps: the normalisation's numerical epsilon (must be > 0 for the Jacobian to exist).

    Dtypes / device:
        CPU tensors; ``preimage`` may be float32 or float64 and the Jacobian is applied in that
        dtype. ``weight``/``bias`` follow the same dtype as ``preimage``.

    Numerical limitations:
        The Jacobian is analytic (not autograd) and requires ``eps > 0``; with ``eps == 0`` a
        constant preimage would divide by zero. Verified against
        ``torch.autograd.functional.jacobian`` in ``tests/unit/test_proxy_base.py``.
    """

    kind: str
    preimage: torch.Tensor
    weight: torch.Tensor | None = None
    bias: torch.Tensor | None = None
    eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.kind not in NORM_KINDS:
            raise ProxyInputError(f"kind must be one of {NORM_KINDS}, got {self.kind!r}")
        if self.preimage.dim() != 2:
            raise ProxyInputError(f"preimage must be 2-D (n_samples, d), got {self.preimage.shape}")
        if self.eps <= 0.0:
            raise ProxyInputError(
                f"eps must be > 0 for a differentiable normalisation, got {self.eps}"
            )
        if self.kind != "identity" and self.weight is None:
            raise ProxyInputError(f"kind={self.kind!r} requires a (d,) weight tensor")

    @property
    def width(self) -> int:
        """Feature width ``d`` of the normalisation's input space."""
        return int(self.preimage.shape[-1])

    def jacobian_apply(self, delta: torch.Tensor) -> torch.Tensor:
        """Apply the normalisation's Jacobian to ``delta``: returns ``J_N(preimage) @ delta``.

        Shapes:
            ``delta``: ``(n_samples, d)`` with ``d == self.width``. Returns the same shape.

        Dtypes / device:
            Same dtype and device as ``preimage``; ``delta`` is cast to that dtype.

        Numerical limitations:
            First order only. LayerNorm is not linear, so ``J_N(v) @ delta`` equals
            ``N(v + delta) - N(v)`` only to first order in ``delta``; the second-order gap shrinks
            as ``O(||delta||^2)`` and is measured, not assumed, in
            ``tests/unit/test_proxy_variants.py``.
        """
        if delta.dim() != 2 or delta.shape != self.preimage.shape:
            raise ProxyInputError(
                f"delta must have the same 2-D shape as preimage {tuple(self.preimage.shape)}, "
                f"got {tuple(delta.shape)}"
            )
        v = self.preimage
        d = delta.to(dtype=v.dtype)
        if self.kind == "identity":
            return d
        if self.kind == "rmsnorm":
            # h_i = v_i / rms, rms = sqrt(mean(v^2) + eps)  =>
            # (J d)_i = gamma_i * (d_i/rms - v_i * mean_j(v_j d_j) / rms^3)
            rms = torch.sqrt(torch.mean(v * v, dim=-1, keepdim=True) + self.eps)
            proj = torch.mean(v * d, dim=-1, keepdim=True)
            out = d / rms - v * proj / (rms**3)
        else:
            # h = (v - mean(v)) / sigma  =>  (J d)_i =
            #     gamma_i/sigma * (d_i - mean_j(d_j) - h_i * mean_j(h_j d_j))
            mu = torch.mean(v, dim=-1, keepdim=True)
            centred = v - mu
            sigma = torch.sqrt(torch.mean(centred * centred, dim=-1, keepdim=True) + self.eps)
            h = centred / sigma
            centred_d = d - torch.mean(d, dim=-1, keepdim=True)
            along = torch.mean(h * d, dim=-1, keepdim=True)
            out = (centred_d - h * along) / sigma
        if self.weight is not None:
            out = out * self.weight
        return out


@dataclass(frozen=True)
class LayerInputs:
    """Everything a proxy may know about one linear layer, on one calibration batch.

    Args:
        weight: ``(out_features, in_features)`` float32/float64 CPU weight ``W``.
        activations: ``(n_samples, in_features)`` inputs actually fed to ``W`` in the **uncompressed**
            forward pass (``n_samples`` is the flattened batch x sequence-token count).
        following_norm: the normalisation that consumes the residual stream this layer's sub-layer
            writes into, or ``None`` when the layer is the last op before the measurement point.
        residual_error: ``(n_samples, d)`` residual-stream error produced by compressing **only this
            layer** (``d`` is the residual width, not necessarily ``out_features``: a fused ``qkv``
            projection writes ``3d`` values but contributes only ``d`` of them to the residual).
            Feeding ``x W^T`` through attention/ReLU is not the proxy's job, so this measured error
            is supplied by the calibration driver. Required by
            :class:`~spectraquant.proxies.variants.InSituOutputErrorProxy`.
        downstream_gain: the estimated RMS ratio between this layer's output error and the model
            output error, from :func:`spectraquant.proxies.gain.estimate_downstream_gains`.
            Required by :class:`~spectraquant.proxies.variants.GainAwareComposedProxy`.
        joint_activations: ``(n_samples, in_features)`` inputs the layer sees once **all** layers are
            compressed. Reported as the activation-distribution shift the proxy is exposed to
            (design note section 7.5) and never used to compute ``value``.

    Dtypes / device:
        CPU; ``weight``/``activations`` may be float32 or float64 and every variant computes in the
        dtype of ``weight``, promoting scalars to float64 at the reduction.
    """

    weight: torch.Tensor
    activations: torch.Tensor
    following_norm: FollowingNorm | None = None
    residual_error: torch.Tensor | None = None
    downstream_gain: float | None = None
    joint_activations: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.weight.dim() != 2:
            raise ProxyInputError(f"weight must be 2-D (out, in), got {tuple(self.weight.shape)}")
        if self.activations.dim() != 2:
            raise ProxyInputError(
                f"activations must be 2-D (n_samples, in), got {tuple(self.activations.shape)}"
            )
        if self.activations.shape[-1] != self.weight.shape[-1]:
            raise ProxyInputError(
                f"activations width {self.activations.shape[-1]} != weight in_features "
                f"{self.weight.shape[-1]}"
            )
        if (
            self.joint_activations is not None
            and self.joint_activations.shape != self.activations.shape
        ):
            raise ProxyInputError(
                "joint_activations must match the shape of activations, got "
                f"{tuple(self.joint_activations.shape)} vs {tuple(self.activations.shape)}"
            )
        if self.downstream_gain is not None and self.downstream_gain < 0.0:
            raise ProxyInputError(f"downstream_gain must be >= 0, got {self.downstream_gain}")


@runtime_checkable
class Proxy(Protocol):
    """Frozen M2 proxy protocol (``design-m2-interfaces.md`` section 3).

    Implementations live in :mod:`spectraquant.proxies.variants`. ``score_layer`` is the per-layer
    entry point of the frozen signature; variants that need context the signature cannot carry (the
    following normalisation, a measured residual error, an estimated downstream gain) document that
    ``score_layer`` degrades to the context-free form and that ``score_model`` is the context-rich
    entry point.
    """

    #: Stable variant name, used as a key in result dictionaries and measurement artifacts.
    name: str

    def score_layer(
        self, w: torch.Tensor, x: torch.Tensor, rank: int, spec: QuantSpec
    ) -> ProxyResult:
        """Score a single linear layer.

        Args:
            w: ``(out, in)`` weight.
            x: ``(n_samples, in)`` activations.
            rank: requested low-rank budget.
            spec: quantization configuration for the factors.

        Returns:
            :class:`ProxyResult` with ``per_layer == {"layer": value}``.
        """
        ...

    def score_model(
        self,
        layers: Mapping[str, LayerInputs],
        ranks: Mapping[str, int],
        specs: Mapping[str, QuantSpec],
    ) -> ProxyResult:
        """Score a whole model: one value per layer, aggregated by summation.

        Args:
            layers: ``name -> LayerInputs``.
            ranks: ``name -> rank`` (same key set as ``layers``).
            specs: ``name -> QuantSpec`` (same key set as ``layers``).

        Returns:
            :class:`ProxyResult` whose ``value`` is the sum of the per-layer values.
        """
        ...


def layer_input_error(layer: LayerInputs, rank: int, spec: QuantSpec) -> torch.Tensor:
    """Return ``x W^T - x (Q(B) Q(A))^T`` for one layer, i.e. the output-space error the proxy unit
    is defined on.

    Shapes:
        returns ``(n_samples, out_features)``, in the dtype of ``layer.weight``.

    This is a convenience for diagnostics and tests; the variant implementations do not use it (they
    work with the weight-space delta ``W - Q(B) Q(A)`` directly, which is cheaper).
    """
    from spectraquant.proxies.operators import compression_delta

    delta = compression_delta(layer.weight, rank, spec)
    return layer.activations @ delta.transpose(0, 1)
