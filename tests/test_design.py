import numpy as np
import pytest
from scipy import stats

from expplatform import design
from expplatform.analysis import welch_ttest


def test_sample_size_means_closed_form():
    # delta = 0.2 sigma, alpha .05, power .8 -> 392.4 per arm
    ss = design.sample_size_means(0.2, 1.0)
    assert ss.n_control == 393 and ss.n_treatment == 393 and ss.total == 786


def test_sample_size_proportions_reference_value():
    # 10% -> 12%: standard references give ~3,840 per arm
    ss = design.sample_size_proportions(0.10, 0.12)
    assert 3800 <= ss.n_control <= 3880


def test_variance_reduction_scales_sample_size():
    base = design.sample_size_means(0.1, 1.0).n_control
    red = design.sample_size_means(0.1, 1.0, r2=0.5).n_control
    assert red == pytest.approx(0.5 * base, rel=0.01)
    assert design.variance_reduction_factor(0.3) == pytest.approx(0.7)


def test_unequal_allocation_costs_more_total():
    eq = design.sample_size_means(0.2, 1.0)
    uneq = design.sample_size_means(0.2, 1.0, ratio=0.25)
    assert uneq.total > eq.total


def test_mde_inverts_sample_size_and_power():
    n = 2000
    d = design.mde_means(n, sd=2.0)
    assert design.power_means(n, d, 2.0) == pytest.approx(0.8, abs=1e-6)
    assert design.sample_size_means(d, 2.0).n_control == pytest.approx(n, abs=2)
    # CUPED with r2 = .36 shrinks the MDE by sqrt(1 - r2) = .8
    assert design.mde_means(n, 2.0, r2=0.36) == pytest.approx(0.8 * d, rel=1e-6)
    p = design.mde_proportions(5000, 0.10)
    assert design.power_proportions(5000, 0.10, 0.10 + p) == pytest.approx(0.8, abs=1e-6)


def test_simulated_power_matches_calculator_means():
    sd, delta = 3.0, 0.5
    n = design.sample_size_means(delta, sd).n_control
    rng = np.random.default_rng(101)
    sims = 1500
    rej = 0
    for _ in range(sims):
        t = rng.normal(delta, sd, n)
        c = rng.normal(0.0, sd, n)
        rej += welch_ttest(t, c).p_value < 0.05
    power = rej / sims  # sd ~ .0103
    assert abs(power - 0.80) < 0.04


def test_simulated_power_matches_calculator_proportions():
    p0, p1 = 0.10, 0.12
    n = design.sample_size_proportions(p0, p1).n_control
    rng = np.random.default_rng(202)
    sims = 4000
    xt = rng.binomial(n, p1, sims)
    xc = rng.binomial(n, p0, sims)
    pool = (xt + xc) / (2 * n)
    z = (xt / n - xc / n) / np.sqrt(pool * (1 - pool) * 2 / n)
    power = (np.abs(z) > stats.norm.isf(0.025)).mean()  # sd ~ .0063
    assert abs(power - 0.80) < 0.03


def test_ratio_metric_sizing_matches_simulated_variance():
    rng = np.random.default_rng(3)
    den = 1 + rng.poisson(3, 50_000).astype(float)
    num = den * rng.lognormal(0, 0.5, 50_000)
    var = design.ratio_metric_variance(num, den)
    # variance of the sample ratio over many small samples, scaled by n, should match
    n = 400
    ests = []
    for _ in range(1500):
        idx = rng.integers(0, len(num), n)
        ests.append(num[idx].sum() / den[idx].sum())
    assert np.var(ests) * n == pytest.approx(var, rel=0.12)
    ss = design.sample_size_ratio_metric(0.05, num, den)
    assert ss.n_control > 0


def test_input_validation():
    with pytest.raises(ValueError):
        design.sample_size_means(0.0, 1.0)
    with pytest.raises(ValueError):
        design.sample_size_means(0.1, 1.0, r2=1.0)
    with pytest.raises(ValueError):
        design.sample_size_proportions(0.1, 0.1)
