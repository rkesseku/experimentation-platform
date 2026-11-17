"""Power, sample-size and minimum-detectable-effect calculators.

All sizes are returned for the control group and the treatment group (``ratio`` = n_treat /
n_control). Variance reduction (CUPED / covariate adjustment) enters through ``r2``, the squared
correlation between the metric and the covariate: the residual variance is ``(1 - r2) * sigma^2``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats


@dataclass(frozen=True)
class SampleSize:
    n_control: int
    n_treatment: int

    @property
    def total(self) -> int:
        return self.n_control + self.n_treatment


def _zs(alpha: float, power: float, two_sided: bool) -> tuple[float, float]:
    za = stats.norm.isf(alpha / 2 if two_sided else alpha)
    return float(za), float(stats.norm.ppf(power))


def _check_r2(r2: float) -> None:
    if not 0.0 <= r2 < 1.0:
        raise ValueError("r2 must lie in [0, 1)")


def sample_size_means(
    delta: float,
    sd: float,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
    two_sided: bool = True,
    r2: float = 0.0,
) -> SampleSize:
    """Sample size to detect an absolute mean difference ``delta`` (normal approximation).

    ``n_c = (z_{1-a/2} + z_{power})^2 * sigma^2 (1 - r2) (1 + 1/ratio) / delta^2``.
    """
    _check_r2(r2)
    if delta == 0 or sd <= 0:
        raise ValueError("delta must be non-zero and sd positive")
    za, zb = _zs(alpha, power, two_sided)
    n = (za + zb) ** 2 * sd**2 * (1 - r2) * (1 + 1 / ratio) / delta**2
    nc = math.ceil(n)
    return SampleSize(nc, math.ceil(nc * ratio))


def sample_size_proportions(
    p_control: float,
    p_treatment: float,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
    two_sided: bool = True,
    r2: float = 0.0,
) -> SampleSize:
    """Sample size for a two-proportion z-test (pooled variance under H0, unpooled under H1).

    Variance reduction is applied as a ``(1 - r2)`` factor on both variance terms, which is an
    approximation when a covariate is used with a binary outcome.
    """
    _check_r2(r2)
    if p_control == p_treatment:
        raise ValueError("p_control and p_treatment must differ")
    za, zb = _zs(alpha, power, two_sided)
    pbar = (p_control + ratio * p_treatment) / (1 + ratio)
    v0 = (1 + 1 / ratio) * pbar * (1 - pbar)
    v1 = p_control * (1 - p_control) + p_treatment * (1 - p_treatment) / ratio
    n = (za * math.sqrt(v0 * (1 - r2)) + zb * math.sqrt(v1 * (1 - r2))) ** 2 / (
        p_treatment - p_control
    ) ** 2
    nc = math.ceil(n)
    return SampleSize(nc, math.ceil(nc * ratio))


def power_means(
    n_control: int,
    delta: float,
    sd: float,
    alpha: float = 0.05,
    ratio: float = 1.0,
    two_sided: bool = True,
    r2: float = 0.0,
) -> float:
    """Power of the two-sample test for a mean difference (normal approximation)."""
    _check_r2(r2)
    se = sd * math.sqrt((1 - r2) * (1 + 1 / ratio) / n_control)
    za = stats.norm.isf(alpha / 2 if two_sided else alpha)
    d = abs(delta) / se
    if two_sided:
        return float(stats.norm.cdf(d - za) + stats.norm.cdf(-d - za))
    return float(stats.norm.cdf(d - za))


def mde_means(
    n_control: int,
    sd: float,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
    two_sided: bool = True,
    r2: float = 0.0,
) -> float:
    """Minimum detectable absolute difference in means at the given sample size."""
    _check_r2(r2)
    za, zb = _zs(alpha, power, two_sided)
    se = sd * math.sqrt((1 - r2) * (1 + 1 / ratio) / n_control)

    def f(d: float) -> float:
        return power_means(n_control, d, sd, alpha, ratio, two_sided, r2) - power

    return float(optimize.brentq(f, 1e-9 * se, (za + zb + 10) * se))


def power_proportions(
    n_control: int,
    p_control: float,
    p_treatment: float,
    alpha: float = 0.05,
    ratio: float = 1.0,
    two_sided: bool = True,
) -> float:
    """Power of the two-proportion z-test (pooled null variance, unpooled alternative)."""
    za = stats.norm.isf(alpha / 2 if two_sided else alpha)
    pbar = (p_control + ratio * p_treatment) / (1 + ratio)
    se0 = math.sqrt((1 + 1 / ratio) * pbar * (1 - pbar) / n_control)
    se1 = math.sqrt(
        (p_control * (1 - p_control) + p_treatment * (1 - p_treatment) / ratio) / n_control
    )
    d = abs(p_treatment - p_control)
    up = stats.norm.cdf((d - za * se0) / se1)
    if not two_sided:
        return float(up)
    return float(up + stats.norm.cdf((-d - za * se0) / se1))


def mde_proportions(
    n_control: int,
    p_control: float,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
    two_sided: bool = True,
) -> float:
    """Minimum detectable absolute lift over ``p_control`` (searches upward)."""

    def f(d: float) -> float:
        return power_proportions(n_control, p_control, p_control + d, alpha, ratio, two_sided) - power

    hi = 1 - p_control - 1e-9
    if f(hi) < 0:
        raise ValueError("no detectable lift below 1.0 at this sample size")
    return float(optimize.brentq(f, 1e-9, hi))


def ratio_metric_variance(num: np.ndarray, den: np.ndarray) -> float:
    """Per-unit variance of the delta-method linearisation of ``sum(num)/sum(den)``.

    ``Var(N - R D) / E[D]^2``. Pass its square root as ``sd`` to :func:`sample_size_means`
    to size a test on a ratio metric.
    """
    num = np.asarray(num, float)
    den = np.asarray(den, float)
    r = num.mean() / den.mean()
    return float(np.var(num - r * den, ddof=1) / den.mean() ** 2)


def sample_size_ratio_metric(
    delta: float,
    num: np.ndarray,
    den: np.ndarray,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
    r2: float = 0.0,
) -> SampleSize:
    """Sample size (in analysis units) for an absolute change ``delta`` in a ratio metric."""
    sd = math.sqrt(ratio_metric_variance(num, den))
    return sample_size_means(delta, sd, alpha, power, ratio, True, r2)


def variance_reduction_factor(r2: float) -> float:
    """Sample-size multiplier from covariate adjustment: ``1 - r2``."""
    _check_r2(r2)
    return 1.0 - r2
