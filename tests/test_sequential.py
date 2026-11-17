import math

import numpy as np
import pytest

from expplatform.sequential import (
    cs_radius,
    group_sequential_bounds,
    msprt_log_lr,
    msprt_path,
    peeking_simulation,
)


def test_naive_peeking_inflates_but_sequential_methods_control_type_i_error():
    r = peeking_simulation(n_sims=3000, n_per_look=100, n_looks=50, effect=0.0, tau=0.2, seed=1)
    assert r["naive_peeking"] > 0.20  # 50 looks at nominal 5%
    assert abs(r["fixed"] - 0.05) < 0.02
    assert r["msprt"] < 0.05  # always-valid: conservative, never above alpha (+ MC noise)
    assert 0.03 < r["obf"] < 0.065  # Lan-DeMets is exact; MC sd ~ .004


def test_msprt_controls_error_with_estimated_variance():
    r = peeking_simulation(n_sims=400, n_per_look=100, n_looks=30, effect=0.0, tau=0.2,
                           estimate_variance=True, seed=2)
    assert r["naive_peeking"] > 0.15
    assert r["msprt"] < 0.065


def test_sequential_methods_retain_power():
    r = peeking_simulation(n_sims=1500, n_per_look=100, n_looks=50, effect=0.1, tau=0.1, seed=3)
    assert r["msprt"] > 0.9 and r["obf"] > 0.95
    assert r["msprt_mean_stop_fraction"] < 0.7  # stops early on average


def test_obf_boundaries_properties():
    b = group_sequential_bounds(np.arange(1, 6) / 5)
    assert np.all(np.diff(b) < 0)  # strictest early, relaxes toward the end
    assert b[0] == pytest.approx(4.38, abs=0.01)
    assert 2.0 < b[-1] < 2.1  # close to, but above, the fixed-sample 1.96
    assert group_sequential_bounds([1.0])[0] == pytest.approx(1.95996, abs=1e-4)
    with pytest.raises(ValueError):
        group_sequential_bounds([0.5, 0.4, 1.0])


def test_obf_boundary_error_spending_by_brownian_monte_carlo():
    t = np.arange(1, 6) / 5
    b = group_sequential_bounds(t)
    rng = np.random.default_rng(7)
    inc = rng.normal(scale=np.sqrt(0.2), size=(200_000, 5))
    z = np.cumsum(inc, 1) / np.sqrt(t)
    rate = (np.abs(z) >= b).any(axis=1).mean()  # MC sd ~ .0005
    assert rate == pytest.approx(0.05, abs=0.003)


def test_confidence_sequence_uniform_coverage():
    rng = np.random.default_rng(5)
    n_sims, looks, theta = 4000, 60, 0.3
    ns = 50 * np.arange(1, looks + 1)
    blocks = rng.normal(theta, math.sqrt(2 / 50), size=(n_sims, looks))
    d = np.cumsum(blocks, 1) / np.arange(1, looks + 1)
    v = 2.0 / ns
    r = cs_radius(v, tau2=0.25**2)
    miss = (np.abs(d - theta) > r).any(axis=1).mean()
    assert miss < 0.05
    # and the pointwise fixed-n interval would miss far more often under peeking
    assert (np.abs(d - theta) > 1.96 * np.sqrt(v)).any(axis=1).mean() > 0.2


def test_msprt_path_monotone_pvalues_and_nested_cs():
    rng = np.random.default_rng(8)
    ns = 100 * np.arange(1, 41)
    d = np.cumsum(rng.normal(0.15, math.sqrt(2 / 100), 40)) / np.arange(1, 41)
    path = msprt_path(d, 2.0 / ns, tau2=0.04)
    assert np.all(np.diff(path.p_always_valid) <= 1e-12)
    assert np.all(np.diff(path.cs_low) >= -1e-12) and np.all(np.diff(path.cs_high) <= 1e-12)
    assert path.first_rejection() is not None
    assert path.cs_low[-1] > 0  # CS excludes 0 once rejected
    # log-LR at d = 0 is negative (evidence for the null)
    assert msprt_log_lr(np.array([0.0]), np.array([0.01]), 0.04)[0] < 0
