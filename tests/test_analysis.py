import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests

from expplatform import analysis as an
from expplatform.simulator import SimConfig, simulate_experiment


def test_welch_matches_scipy():
    rng = np.random.default_rng(0)
    t, c = rng.normal(1, 2, 300), rng.normal(0.5, 1, 250)
    r = an.welch_ttest(t, c)
    ref = stats.ttest_ind(t, c, equal_var=False)
    assert r.p_value == pytest.approx(ref.pvalue, rel=1e-9)
    ci = ref.confidence_interval(0.95)
    assert r.ci_low == pytest.approx(ci.low, rel=1e-9) and r.ci_high == pytest.approx(ci.high, rel=1e-9)


def test_type_i_error_welch_under_null():
    rng = np.random.default_rng(11)
    sims = 1500
    rej = sum(
        an.welch_ttest(rng.normal(0, 1, 200), rng.normal(0, 3, 150)).p_value < 0.05
        for _ in range(sims)
    )
    # sd of the estimate ~ .0056; allow 4 sd
    assert abs(rej / sims - 0.05) < 0.0225


def test_type_i_error_two_proportion_under_null():
    rng = np.random.default_rng(12)
    sims, n = 3000, 2000
    xt, xc = rng.binomial(n, 0.2, sims), rng.binomial(n, 0.2, sims)
    rej = sum(an.two_proportion_ztest(a, n, b, n).p_value < 0.05 for a, b in zip(xt, xc))
    assert abs(rej / sims - 0.05) < 0.02


def _sim(seed, **kw):
    return simulate_experiment(SimConfig(n=2000, use_hash=False, seed=seed, **kw))


def test_cuped_reduces_variance_and_is_unbiased():
    true_effect, rho = 0.4, 0.5
    raw, adj = [], []
    for s in range(400):
        df = _sim(s, effect=true_effect, rho=rho, sd=2.0)
        t, c = df[df.variant == "treatment"], df[df.variant == "control"]
        raw.append(an.welch_ttest(t.y.to_numpy(), c.y.to_numpy()).estimate)
        adj.append(an.cuped(t.y.to_numpy(), t.x_pre.to_numpy(), c.y.to_numpy(),
                            c.x_pre.to_numpy()).estimate)
    raw, adj = np.array(raw), np.array(adj)
    # unbiased: mean within 4 Monte-Carlo standard errors of the truth
    assert abs(adj.mean() - true_effect) < 4 * adj.std() / np.sqrt(len(adj))
    ratio = adj.var() / raw.var()
    assert ratio == pytest.approx(1 - rho**2, abs=0.08)  # corr(x, y) = rho -> R^2 = rho^2


def test_cuped_standard_error_calibrated_and_reported_reduction():
    df = _sim(5, effect=0.0, rho=0.6, sd=2.0)
    t, c = df[df.variant == "treatment"], df[df.variant == "control"]
    r0 = an.welch_ttest(t.y.to_numpy(), c.y.to_numpy())
    r1 = an.cuped(t.y.to_numpy(), t.x_pre.to_numpy(), c.y.to_numpy(), c.x_pre.to_numpy())
    assert r1.se < r0.se
    assert 1 - (r1.se / r0.se) ** 2 == pytest.approx(0.36, abs=0.08)


def test_cuped_does_not_inflate_type_i_error():
    rej = 0
    sims = 800
    for s in range(sims):
        df = _sim(1000 + s, effect=0.0, rho=0.5)
        t, c = df[df.variant == "treatment"], df[df.variant == "control"]
        rej += an.cuped(t.y.to_numpy(), t.x_pre.to_numpy(), c.y.to_numpy(),
                        c.x_pre.to_numpy()).p_value < 0.05
    assert abs(rej / sims - 0.05) < 0.03


def test_regression_adjusted_matches_cuped_and_beats_unadjusted():
    df = _sim(9, effect=0.3, rho=0.5, sd=2.0)
    t = (df.variant == "treatment").to_numpy().astype(float)
    r = an.regression_adjusted(df.y.to_numpy(), t, df[["x_pre"]].to_numpy())
    raw = an.welch_ttest(df.y[t == 1].to_numpy(), df.y[t == 0].to_numpy())
    cp = an.cuped(df.y[t == 1].to_numpy(), df.x_pre[t == 1].to_numpy(),
                  df.y[t == 0].to_numpy(), df.x_pre[t == 0].to_numpy())
    assert r.se < raw.se
    assert r.estimate == pytest.approx(cp.estimate, abs=0.05)


def test_cupac_cross_fitted_covariate_beats_single_pre_metric():
    df = simulate_experiment(SimConfig(n=8000, use_hash=False, seed=21, effect=0.2, rho=0.5, sd=2.0))
    pred = an.cupac_covariate(df[["f1", "f2", "f3"]].to_numpy(), df.y.to_numpy(), seed=1)
    t = (df.variant == "treatment").to_numpy()
    se_pre = an.cuped(df.y[t].to_numpy(), df.x_pre[t].to_numpy(),
                      df.y[~t].to_numpy(), df.x_pre[~t].to_numpy()).se
    se_cupac = an.cuped(df.y[t].to_numpy(), pred[t], df.y[~t].to_numpy(), pred[~t]).se
    assert se_cupac < se_pre


def test_delta_method_ci_coverage():
    rng = np.random.default_rng(31)
    mu, s, rel = 0.0, 0.5, 0.10
    true_effect = rel * np.exp(mu + s**2 / 2)  # ratio = E[value]; treatment scales value by 1.1
    cover, sims, n = 0, 700, 1500
    for _ in range(sims):
        dc = 1 + rng.poisson(3, n).astype(float)
        dt = 1 + rng.poisson(3, n).astype(float)
        nc = np.array([rng.lognormal(mu, s, int(k)).sum() for k in dc])
        nt = (1 + rel) * np.array([rng.lognormal(mu, s, int(k)).sum() for k in dt])
        r = an.delta_method_ratio(nt, dt, nc, dc)
        cover += r.ci_low <= true_effect <= r.ci_high
    assert abs(cover / sims - 0.95) < 0.03


def test_delta_method_beats_naive_row_level_variance():
    # Heavy within-user clustering: naive per-session SE is too small, delta method is right.
    rng = np.random.default_rng(32)
    n = 2000
    den = 1 + rng.poisson(5, n).astype(float)
    user_level = rng.normal(0, 1, n)
    num = den * (1 + user_level)  # all sessions of a user share the user's level
    r = an.delta_method_ratio(num, den, num, den)
    rows = np.repeat(1 + user_level, den.astype(int))
    naive_se = np.sqrt(2 * rows.var(ddof=1) / len(rows))
    assert r.se > 1.3 * naive_se


def test_bootstrap_intervals_agree_with_welch_on_normal_data():
    rng = np.random.default_rng(41)
    t, c = rng.normal(0.3, 1, 400), rng.normal(0, 1, 400)
    w = an.welch_ttest(t, c)
    for method in ("percentile", "bca"):
        b = an.bootstrap_diff(t, c, method, n_resamples=2000, seed=0)
        assert b.ci_low == pytest.approx(w.ci_low, abs=0.04)
        assert b.ci_high == pytest.approx(w.ci_high, abs=0.04)
    with pytest.raises(ValueError):
        an.bootstrap_diff(t, c, "nope")


def test_bca_differs_from_percentile_for_skewed_data():
    rng = np.random.default_rng(42)
    t, c = rng.lognormal(0.1, 1.2, 150), rng.lognormal(0, 1.2, 150)
    p = an.bootstrap_diff(t, c, "percentile", n_resamples=3000, seed=0)
    b = an.bootstrap_diff(t, c, "bca", n_resamples=3000, seed=0)
    assert (b.ci_low, b.ci_high) != (p.ci_low, p.ci_high)
    assert b.ci_low < b.estimate < b.ci_high


def test_multiplicity_corrections_match_statsmodels():
    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
    assert an.holm(p)["adjusted"] == pytest.approx(multipletests(p, method="holm")[1])
    assert an.benjamini_hochberg(p)["adjusted"] == pytest.approx(
        multipletests(p, method="fdr_bh")[1])
    h = an.holm([0.01, 0.04, 0.03, 0.005])
    assert h["adjusted"] == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert h["reject"] == [True, False, False, True]


def test_fwer_control_under_global_null_holm_vs_uncorrected():
    rng = np.random.default_rng(51)
    sims, m = 2000, 10
    any_raw = any_holm = 0
    for _ in range(sims):
        p = rng.uniform(size=m)
        any_raw += (p < 0.05).any()
        any_holm += any(an.holm(p)["reject"])
    assert any_raw / sims > 0.3
    assert any_holm / sims < 0.075


def test_noninferiority_decisions():
    rng = np.random.default_rng(61)
    c = rng.normal(100, 10, 5000)
    t_same = rng.normal(100, 10, 5000)
    t_worse = rng.normal(96, 10, 5000)
    ok = an.noninferiority_test(an.welch_ttest(t_same, c), margin=1.5)
    bad = an.noninferiority_test(an.welch_ttest(t_worse, c), margin=1.5)
    assert ok["non_inferior"] and not bad["non_inferior"]
    # lower-is-better flips the sign: higher latency is the harm
    lat = an.noninferiority_test(an.welch_ttest(t_worse, c), margin=1.5, higher_is_better=False)
    assert lat["non_inferior"]
    with pytest.raises(ValueError):
        an.noninferiority_test(an.welch_ttest(t_same, c), margin=0)
