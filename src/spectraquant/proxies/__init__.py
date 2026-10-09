"""Sensitivity-proxy package: the output-aware cost proxies of Milestone 2.

The frozen contract lives in :mod:`spectraquant.proxies.base`; the variants are in
:mod:`spectraquant.proxies.variants`; the shared compression operators are in
:mod:`spectraquant.proxies.operators`; the downstream-gain estimator is in
:mod:`spectraquant.proxies.gain`; validation helpers are in :mod:`spectraquant.proxies.analysis`.

Design (``docs/coordination/design-m2-interfaces.md`` section 3 as amended by section 7): the common
unit is the per-layer squared output error ``mean_i ||x_i W^T - x_i W_hat^T||^2``. The naive
pre-normalisation form is a **baseline whose failure is published** — it was measured to lose its
ranking ability once a normalisation layer sits between the layer and the model output — and the
candidate variants add an in-situ normalisation Jacobian and an estimated downstream gain.

Nothing here is a storage or latency measurement: every variant is measurement class 1 or 2
(``AGENTS.md`` section 5).
"""

from __future__ import annotations

from spectraquant.proxies.analysis import following_norm_container, spearman
from spectraquant.proxies.base import (
    FollowingNorm,
    LayerInputs,
    Proxy,
    ProxyInputError,
    ProxyResult,
    layer_input_error,
)
from spectraquant.proxies.gain import GainEstimate, estimate_downstream_gains
from spectraquant.proxies.operators import (
    compression_delta,
    low_rank_delta,
    low_rank_reconstruction,
    quantization_only_delta,
    quantized_factor_reconstruction,
)
from spectraquant.proxies.variants import (
    PROXY_VARIANTS,
    CombinedProxy,
    GainAwareComposedProxy,
    HessianDiagProxy,
    InSituOutputErrorProxy,
    PerLayerOutputErrorProxy,
    QuantResidualStatsProxy,
    SpectralSummaryProxy,
    WeightFrobeniusProxy,
    get_proxy,
    proxy_names,
)

__all__ = [
    "PROXY_VARIANTS",
    "CombinedProxy",
    "FollowingNorm",
    "GainAwareComposedProxy",
    "GainEstimate",
    "HessianDiagProxy",
    "InSituOutputErrorProxy",
    "LayerInputs",
    "PerLayerOutputErrorProxy",
    "Proxy",
    "ProxyInputError",
    "ProxyResult",
    "QuantResidualStatsProxy",
    "SpectralSummaryProxy",
    "WeightFrobeniusProxy",
    "compression_delta",
    "estimate_downstream_gains",
    "following_norm_container",
    "get_proxy",
    "layer_input_error",
    "low_rank_delta",
    "low_rank_reconstruction",
    "proxy_names",
    "quantization_only_delta",
    "quantized_factor_reconstruction",
    "spearman",
]
