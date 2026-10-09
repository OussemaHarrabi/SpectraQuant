"""Analysis helpers for proxy validation: rank correlation, Fisher-z aggregation, normalisation
containers.

These live in the library rather than in the measurement scripts so the tests can import them
without depending on a script's import path.

The aggregation implements the pre-registered statistical plan (`preregistration.md` section 7.0/7.2,
frozen 2026-10-09):

* the **model** is the resampling unit; one Spearman ``rho`` is computed per trained model;
* ``rho -> z = atanh(rho)`` (Fisher-z), with ``var(z_m) = 1 / (n_eff,m - 3)`` and ``n_eff`` estimated
  by a within-model bootstrap over the model's modules (resample with replacement, require at least
  ``ceil(n / 2)`` distinct indices, else redraw);
* models are combined with a **DerSimonian-Laird random-effects** model (``tau^2`` from Cochrane's
  Q); the H2 statistic is the paired Fisher-z difference between the candidate and the predeclared
  comparator, combined the same way;
* the confidence interval of a contrast is additionally bootstrapped over **clusters** (a cluster is
  one trained model / seed, carrying that model's cells), because cells within a seed are repeated
  measures of the same weights and are not independent draws.

No ``scipy`` dependency: the normal distribution comes from :mod:`statistics` and the p-value uses
``erfc``. Every function documents its shapes and its numerical limitations.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist, pvariance

import torch
from torch import nn

from spectraquant.proxies.base import FollowingNorm

__all__ = [
    "PREDECLARED_MDE_Z_CONSERVATIVE",
    "PREDECLARED_MDE_Z_OPTIMISTIC",
    "PREDECLARED_NEFF_CONSERVATIVE",
    "PREDECLARED_NEFF_OPTIMISTIC",
    "PREDECLARED_SEEDS",
    "Z_80",
    "Z_975",
    "BootstrapContrast",
    "HypothesisVerdict",
    "NeffEstimate",
    "RandomEffectsResult",
    "cluster_contrast",
    "contrast_units",
    "contrast_verdict",
    "fisher_z",
    "fisher_z_random_effects",
    "following_norm_container",
    "inverse_fisher_z",
    "mde_delta_rho",
    "mde_fisher_z",
    "paired_fisher_z_contrast",
    "spearman",
    "within_model_neff",
]

_NORMAL = NormalDist()

#: Two-sided 95% normal quantile and the 80%-power normal quantile (the `preregistration.md`
#: section 7.5 MDE constants).
Z_975: float = _NORMAL.inv_cdf(0.975)
Z_80: float = _NORMAL.inv_cdf(0.80)

#: Pre-declared MDE statement (`preregistration.md` section 7.5, frozen): with the model as unit,
#: ``n = 32`` linear modules per model and ``S = 5`` seeds, ``MDE(z) ~= 0.33`` if the modules are
#: treated as independent ("optimistic"), and ``MDE(z) ~= 0.79`` if the 8 blocks are the effective
#: sample ("conservative"). The achieved ``n_eff`` is reported per model and the H2 verdict is stated
#: against the achieved MDE as well.
PREDECLARED_NEFF_OPTIMISTIC: int = 32
PREDECLARED_NEFF_CONSERVATIVE: int = 8
PREDECLARED_SEEDS: int = 5
PREDECLARED_MDE_Z_OPTIMISTIC: float = 0.33
PREDECLARED_MDE_Z_CONSERVATIVE: float = 0.79

#: Clipping bound for ``atanh``. ``|rho| = 1`` gives an infinite Fisher-z, so the transform is
#: clipped; the clipping is reported (``clipped``) rather than silently applied.
_FISHER_CLIP: float = 1.0 - 1e-6


def fisher_z(r: float, *, clip: float = _FISHER_CLIP) -> float:
    """Fisher variance-stabilising transform ``z = atanh(r)``.

    Args:
        r: a correlation in ``[-1, 1]``; ``nan`` propagates to ``nan``.
        clip: magnitude bound applied before ``atanh`` so that ``|r| = 1`` does not return ``inf``.

    Returns:
        ``atanh(clip(r))``. Numerical limitation: for ``|r| >= 1 - 1e-6`` the value is a clipped
        approximation, not the exact transform.
    """
    if math.isnan(r):
        return float("nan")
    bounded = max(-clip, min(clip, float(r)))
    return math.atanh(bounded)


def inverse_fisher_z(z: float) -> float:
    """Inverse of :func:`fisher_z`: ``r = tanh(z)``. ``nan`` propagates."""
    if math.isnan(z):
        return float("nan")
    return math.tanh(float(z))


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rank correlation with average ranks for ties.

    Shapes:
        ``a`` and ``b`` must have the same length. Returns ``nan`` for fewer than two pairs or when
        either input is constant (a zero-variance rank vector has no correlation to report).

    No scipy dependency: the ranks are computed directly, which also makes tie handling explicit
    (average ranks, the standard convention).
    """
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    n = len(a)
    if n < 2:
        return float("nan")

    def ranks(values: Sequence[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: values[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = average
            i = j + 1
        return out

    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den > 0 else float("nan")


# ------------------------------------------------------------------ within-model effective n


@dataclass(frozen=True)
class NeffEstimate:
    """Result of a within-model bootstrap estimate of the effective module count.

    ``var(z_m) = 1 / (n_eff - 3)``, so ``n_eff = 3 + 1 / var_boot(z)`` (documented in
    `preregistration.md` section 7.0). ``n_eff`` is ``nan`` when the bootstrap variance is zero.
    """

    n_modules: int
    n_eff: float
    boot_sd_z: float
    n_boot: int
    n_draws_accepted: int
    clipped: bool


def within_model_neff(
    proxy_values: Sequence[float],
    damages: Sequence[float],
    *,
    n_boot: int = 200,
    seed: int = 0,
    min_distinct_fraction: float = 0.5,
) -> NeffEstimate:
    """Bootstrap the effective sample size of one model's modules for a proxy/damage pair.

    Procedure (pre-registered, `preregistration.md` section 7.0): resample the module indices with
    replacement, require at least ``ceil(n * min_distinct_fraction)`` **distinct** indices else
    redraw, recompute the Spearman ``rho`` on the resample, transform to Fisher-z, and take
    ``n_eff = 3 + 1 / var_boot(z)``.

    Limitations: this is a bootstrap estimate, labelled approximate; with module counts this small
    (``n = 9`` on the Tier-0 fixture) it is noisy and is reported per model, never as an exact
    quantity. Degenerate resamples (constant rank vector) are discarded and counted.
    """
    n = len(proxy_values)
    if len(damages) != n:
        raise ValueError(f"length mismatch: {n} vs {len(damages)}")
    if n < 4:
        return NeffEstimate(n, float("nan"), float("nan"), 0, 0, False)
    rng = random.Random(seed)
    needed = max(2, math.ceil(n * min_distinct_fraction))
    zs: list[float] = []
    attempts = 0
    max_attempts = 20 * n_boot
    clipped = False
    while len(zs) < n_boot and attempts < max_attempts:
        attempts += 1
        idx = [rng.randrange(n) for _ in range(n)]
        if len(set(idx)) < needed:
            continue
        rho = spearman([proxy_values[i] for i in idx], [damages[i] for i in idx])
        if math.isnan(rho):
            continue
        if abs(rho) > _FISHER_CLIP:
            clipped = True
        zs.append(fisher_z(rho))
    if len(zs) < 2:
        return NeffEstimate(n, float("nan"), float("nan"), len(zs), len(zs), clipped)
    var = pvariance(zs)
    sd = math.sqrt(var)
    n_eff = 3.0 + 1.0 / var if var > 0.0 else float("nan")
    return NeffEstimate(n, n_eff, sd, len(zs), len(zs), clipped)


# ------------------------------------------------------------------ Fisher-z random effects


@dataclass(frozen=True)
class RandomEffectsResult:
    """A DerSimonian-Laird random-effects combination of Fisher-z estimates."""

    k: int
    pooled: float
    se: float
    ci_low: float
    ci_high: float
    z_stat: float
    p_value: float
    tau2: float
    q: float
    df: int
    i2: float
    weights: tuple[float, ...]

    def to_dict(self) -> dict[str, float | int]:
        """JSON-safe mapping of every field."""
        return {
            "k": self.k,
            "pooled": self.pooled,
            "se": self.se,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "ci_width": self.ci_high - self.ci_low,
            "z_stat": self.z_stat,
            "p_value": self.p_value,
            "tau2": self.tau2,
            "q": self.q,
            "df": self.df,
            "i2": self.i2,
        }


def fisher_z_random_effects(
    effects: Sequence[float], variances: Sequence[float]
) -> RandomEffectsResult:
    """Combine Fisher-z estimates with a DerSimonian-Laird random-effects model.

    Args:
        effects: per-unit Fisher-z values; ``k = len(effects)``. Must be finite.
        variances: per-unit ``var(z)`` (``= 1 / (n_eff - 3)``); must be finite and strictly positive.

    Returns:
        :class:`RandomEffectsResult` with the pooled estimate, its normal CI, ``tau^2``, Cochrane's
        ``Q``, ``df = k - 1`` and ``I^2``. For ``k = 1`` the fixed-effect answer is returned with
        ``tau^2 = Q = df = I^2 = 0``.

    Raises:
        ValueError: on length mismatch, empty input, non-finite effects or non-positive variances.

    Limitations: ``tau^2`` from Q is approximate with very few units (the negative-``tau^2`` case is
    clamped to 0, the conventional conservative choice), and the normal approximation to the
    t-distribution overstates precision when ``k`` is small. Both are reported, not hidden.
    """
    if len(effects) != len(variances):
        raise ValueError(f"length mismatch: {len(effects)} effects vs {len(variances)} variances")
    k = len(effects)
    if k == 0:
        raise ValueError("cannot combine zero effects")
    eff = [float(e) for e in effects]
    var = [float(v) for v in variances]
    for e in eff:
        if not math.isfinite(e):
            raise ValueError(f"non-finite effect {e!r}")
    for v in var:
        if not math.isfinite(v) or v <= 0.0:
            raise ValueError(f"non-positive or non-finite variance {v!r}")
    if k == 1:
        se = math.sqrt(var[0])
        z = eff[0] / se
        return RandomEffectsResult(
            k=1,
            pooled=eff[0],
            se=se,
            ci_low=eff[0] - Z_975 * se,
            ci_high=eff[0] + Z_975 * se,
            z_stat=z,
            p_value=math.erfc(abs(z) / math.sqrt(2.0)),
            tau2=0.0,
            q=0.0,
            df=0,
            i2=0.0,
            weights=(1.0,),
        )
    w = [1.0 / v for v in var]
    sw = sum(w)
    fixed = sum(wi * e for wi, e in zip(w, eff, strict=True)) / sw
    q = sum(wi * (e - fixed) ** 2 for wi, e in zip(w, eff, strict=True))
    df = k - 1
    c = sw - sum(wi * wi for wi in w) / sw
    tau2 = max(0.0, (q - df) / c) if c > 0.0 else 0.0
    ws = [1.0 / (v + tau2) for v in var]
    sws = sum(ws)
    pooled = sum(wi * e for wi, e in zip(ws, eff, strict=True)) / sws
    se = math.sqrt(1.0 / sws)
    z = pooled / se
    i2 = max(0.0, (q - df) / q) if q > 0.0 else 0.0
    return RandomEffectsResult(
        k=k,
        pooled=pooled,
        se=se,
        ci_low=pooled - Z_975 * se,
        ci_high=pooled + Z_975 * se,
        z_stat=z,
        p_value=math.erfc(abs(z) / math.sqrt(2.0)),
        tau2=tau2,
        q=q,
        df=df,
        i2=i2,
        weights=tuple(ws),
    )


def paired_fisher_z_contrast(
    candidate_z: Sequence[float],
    comparator_z: Sequence[float],
    candidate_var: Sequence[float],
    comparator_var: Sequence[float],
) -> RandomEffectsResult:
    """Paired Fisher-z difference (candidate - comparator) combined by random effects.

    The units are models. ``d_m = z_cand,m - z_comp,m`` with ``var(d_m) = var_cand,m + var_comp,m``
    (the conservative paired form: the covariance between the two proxies' correlations on the same
    modules is not estimated, so the variance is over-stated, never under-stated). The per-unit
    variances follow the pre-registered ``1 / (n_eff - 3)`` rule.
    """
    if not (len(candidate_z) == len(comparator_z) == len(candidate_var) == len(comparator_var)):
        raise ValueError("candidate/comparator z and variance arrays must have equal length")
    diffs = [c - k for c, k in zip(candidate_z, comparator_z, strict=True)]
    var_d = [vc + vk for vc, vk in zip(candidate_var, comparator_var, strict=True)]
    return fisher_z_random_effects(diffs, var_d)


# ------------------------------------------------------------------ bootstrap over clusters


@dataclass(frozen=True)
class BootstrapContrast:
    """A paired Fisher-z contrast with both its analytic and its cluster-bootstrap interval."""

    label: str
    delta_z: float
    se: float
    ci_low: float
    ci_high: float
    boot_ci_low: float
    boot_ci_high: float
    boot_se: float
    z_stat: float
    p_value: float
    tau2: float
    i2: float
    q: float
    df: int
    k: int
    n_clusters: int
    n_boot: int
    points: int

    def to_dict(self) -> dict[str, object]:
        """JSON-safe mapping; ``ci_*`` is the analytic random-effects interval, ``boot_ci_*`` the
        cluster-bootstrap one (the conservative headline number)."""
        return {
            "label": self.label,
            "delta_z": self.delta_z,
            "se": self.se,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "ci_width": self.ci_high - self.ci_low,
            "boot_ci_low": self.boot_ci_low,
            "boot_ci_high": self.boot_ci_high,
            "boot_ci_width": self.boot_ci_high - self.boot_ci_low,
            "boot_se": self.boot_se,
            "z_stat": self.z_stat,
            "p_value": self.p_value,
            "tau2": self.tau2,
            "i2": self.i2,
            "q": self.q,
            "df": self.df,
            "k": self.k,
            "n_clusters": self.n_clusters,
            "n_boot": self.n_boot,
            "points": self.points,
        }


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile of an already sorted sequence; ``nan`` when empty."""
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return float(sorted_values[low])
    frac = pos - low
    return float(sorted_values[low]) * (1.0 - frac) + float(sorted_values[high]) * frac


def contrast_units(per_cluster: Mapping[str, Sequence[tuple[float, float]]]) -> RandomEffectsResult:
    """Pool ``(effect, variance)`` units from every cluster with the random-effects model."""
    units = [unit for key in sorted(per_cluster) for unit in per_cluster[key]]
    return fisher_z_random_effects([u[0] for u in units], [u[1] for u in units])


def cluster_contrast(
    per_cluster: Mapping[str, Sequence[tuple[float, float]]],
    *,
    label: str = "",
    n_boot: int = 2000,
    seed: int = 0,
) -> BootstrapContrast:
    """Paired Fisher-z contrast with a bootstrap CI that resamples **clusters** (models/seeds).

    Args:
        per_cluster: cluster id -> sequence of ``(effect, variance)`` units. One cluster is one
            trained model; its units are the (rank, bits) cells measured on that model, which are
            repeated measures and therefore resampled together.
        label: human-readable contrast label carried into the result.
        n_boot: bootstrap replicates (the pre-registered floor for arm comparisons is 10,000; the
            proxy study uses a smaller replicate count only where the CI is reported alongside the
            analytic interval — the value used is recorded in ``n_boot``).
        seed: RNG seed; the same seed and inputs give a bit-identical CI.

    Returns:
        :class:`BootstrapContrast`. The point estimate is the random-effects pool over all units; the
        bootstrap CI is the 2.5/97.5 percentile over cluster-resampled pools.

    Limitations: with few clusters (5 seeds) the percentile bootstrap is coarse and can be wide; it
    is reported next to the analytic interval rather than replacing it.
    """
    point = contrast_units(per_cluster)
    keys = sorted(per_cluster)
    rng = random.Random(seed)
    pools: list[float] = []
    for _ in range(n_boot):
        draws: list[tuple[float, float]] = []
        for _ in range(len(keys)):
            draws.extend(per_cluster[keys[rng.randrange(len(keys))]])
        try:
            res = fisher_z_random_effects([d[0] for d in draws], [d[1] for d in draws])
        except ValueError:  # a degenerate draw (non-finite/0 variance) is skipped, not imputed
            continue
        pools.append(res.pooled)
    pools.sort()
    boot_se = math.sqrt(pvariance(pools)) if len(pools) > 1 else float("nan")
    return BootstrapContrast(
        label=label,
        delta_z=point.pooled,
        se=point.se,
        ci_low=point.ci_low,
        ci_high=point.ci_high,
        boot_ci_low=_percentile(pools, 0.025),
        boot_ci_high=_percentile(pools, 0.975),
        boot_se=boot_se,
        z_stat=point.z_stat,
        p_value=point.p_value,
        tau2=point.tau2,
        i2=point.i2,
        q=point.q,
        df=point.df,
        k=point.k,
        n_clusters=len(keys),
        n_boot=len(pools),
        points=point.k,
    )


# ------------------------------------------------------------------ power


def mde_fisher_z(n_eff: float, n_models: int, *, power: float = 0.80, alpha: float = 0.05) -> float:
    """Minimum detectable Fisher-z difference, `preregistration.md` section 7.5.

    ``MDE(z) = (z_{1-alpha/2} + z_power) * sqrt(2 / (n_eff - 3)) / sqrt(S)`` for the paired
    model-level contrast. Returns ``nan`` when ``n_eff <= 3`` or ``n_models < 1``.
    """
    if n_eff <= 3.0 or n_models < 1:
        return float("nan")
    z_alpha = _NORMAL.inv_cdf(1.0 - alpha / 2.0)
    z_power = _NORMAL.inv_cdf(power)
    return (z_alpha + z_power) * math.sqrt(2.0 / (n_eff - 3.0)) / math.sqrt(n_models)


def mde_delta_rho(rho: float, n_eff: float, n_models: int, *, power: float = 0.80) -> float:
    """Translate an MDE(z) into an MDE in ``rho`` at correlation ``rho``: ``dz * (1 - rho^2)``."""
    dz = mde_fisher_z(n_eff, n_models, power=power)
    if math.isnan(dz):
        return float("nan")
    return dz * (1.0 - float(rho) ** 2)


# ------------------------------------------------------------------ verdicts


@dataclass(frozen=True)
class HypothesisVerdict:
    """A falsifier verdict with the rule that produced it, so it can be audited."""

    verdict: str
    decisive: bool
    rationale: str

    def to_dict(self) -> dict[str, object]:
        return {"verdict": self.verdict, "decisive": self.decisive, "rationale": self.rationale}


def _exclusion_sign(low: float, high: float) -> int:
    """1 if the interval is entirely above 0, -1 if entirely below, 0 if it covers 0."""
    if low > 0.0:
        return 1
    if high < 0.0:
        return -1
    return 0


def contrast_verdict(
    contrast: BootstrapContrast,
    *,
    mde_optimistic: float = PREDECLARED_MDE_Z_OPTIMISTIC,
    mde_conservative: float = PREDECLARED_MDE_Z_CONSERVATIVE,
) -> HypothesisVerdict:
    """Apply the frozen section 11.1 falsifier rule to one paired Fisher-z contrast.

    Frozen wording (`preregistration.md` section 11.1, H2): "The proxy's **model-level** correlation
    with measured degradation is **not** higher than weight-space Frobenius error — the Fisher-z
    random-effects CI on the paired difference includes 0 and is narrower than the predeclared
    section 7.5 MDE (a CI wider than the MDE is reported **inconclusive**, not refuting)."

    The interval the rule is applied to is the **analytic random-effects interval** prescribed by
    section 7.2; the cluster-bootstrap interval is carried alongside in :class:`BootstrapContrast`
    as a robustness check (with very few clusters it under-covers, so it is not the decision
    interval), and any disagreement between the two about excluding 0 is stated in the rationale.
    Rule:

    1. CI excludes 0 and ``delta_z > 0`` -> ``supported``;
    2. CI excludes 0 and ``delta_z < 0`` -> ``refuted`` (the candidate is significantly worse);
    3. CI includes 0 and its width is at most the predeclared optimistic MDE -> ``refuted``;
    4. otherwise -> ``inconclusive``.
    """
    low, high = contrast.ci_low, contrast.ci_high
    if low > 0.0:
        verdict = HypothesisVerdict(
            "supported",
            True,
            f"random-effects CI [{low:.3f}, {high:.3f}] excludes 0 and "
            f"delta_z = {contrast.delta_z:+.3f} > 0.",
        )
    elif high < 0.0:
        verdict = HypothesisVerdict(
            "refuted",
            True,
            f"random-effects CI [{low:.3f}, {high:.3f}] excludes 0 and "
            f"delta_z = {contrast.delta_z:+.3f} < 0.",
        )
    elif (high - low) <= mde_optimistic:
        verdict = HypothesisVerdict(
            "refuted",
            True,
            f"random-effects CI [{low:.3f}, {high:.3f}] includes 0 and is narrower "
            f"(width {high - low:.3f}) than the predeclared MDE {mde_optimistic:.2f}.",
        )
    else:
        verdict = HypothesisVerdict(
            "inconclusive",
            False,
            f"random-effects CI [{low:.3f}, {high:.3f}] includes 0 and has width {high - low:.3f}, "
            f"which is not narrower than the predeclared MDE band [{mde_optimistic:.2f}, "
            f"{mde_conservative:.2f}]; section 11.1 makes this inconclusive, not a refutation.",
        )
    if _exclusion_sign(low, high) != _exclusion_sign(contrast.boot_ci_low, contrast.boot_ci_high):
        return HypothesisVerdict(
            verdict.verdict,
            verdict.decisive,
            verdict.rationale + f" NOTE: the cluster-bootstrap CI [{contrast.boot_ci_low:.3f}, "
            f"{contrast.boot_ci_high:.3f}] disagrees about excluding 0.",
        )
    return verdict


# ------------------------------------------------------------------ normalisation containers


def following_norm_container(
    model: object, layer_name: str, preimages: Mapping[str, torch.Tensor]
) -> FollowingNorm | None:
    """Build the :class:`FollowingNorm` for one layer of the toy fixture.

    Args:
        model: the fixture model (anything exposing ``following_norm(name)``).
        layer_name: e.g. ``blocks.0.proj``.
        preimages: ``norm name -> input tensor`` from
            :func:`spectraquant.evaluation.toy.norm_preimages`.

    Returns:
        A container carrying the normalisation's operating point and parameters, or ``None`` when the
        layer has no following normalisation (the output head) or the preimage was not captured.
    """
    found = model.following_norm(layer_name)  # type: ignore[attr-defined]
    if found is None:
        return None
    norm_name, module = found
    preimage = preimages.get(norm_name)
    if preimage is None:
        return None
    if isinstance(module, nn.LayerNorm):
        return FollowingNorm(
            kind="layernorm",
            preimage=preimage,
            weight=module.weight.detach(),
            bias=module.bias.detach(),
            eps=float(module.eps),
        )
    return FollowingNorm(kind="identity", preimage=preimage, weight=None, eps=1e-5)
