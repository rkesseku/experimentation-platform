"""Fixed-horizon inference: Welch, two-proportion, delta method, CUPED/CUPAC, bootstrap,
multiplicity correction and non-inferiority tests.

All effect functions return an :class:`EffectResult` for ``treatment - control``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats


@dataclass
class EffectResult:
    estimate: float
    se: float
    ci_low: float
    ci_high: float
    p_value: float
    n_control: int
    n_treatment: int
    mean_control: float
    mean_treatment: float
    method: str
    alpha: float = 0.05

    @property
    def relative_lift(self) -> float:
        return self.estimate / self.mean_control if self.mean_control else float("nan")

    def to_dict(self) -> dict:
        d = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in asdict(self).items()}
        d["relative_lift"] = float(self.relative_lift)
        return d


def _normal_result(
    est: float, se: float, mc: float, mt: float, nc: int, nt: int, method: str, alpha: float,
    df: float | None = None,
) -> EffectResult:
    if se <= 0 or not np.isfinite(se):
        p, lo, hi = (1.0 if est == 0 else 0.0), est, est
    elif df is None:
        z = est / se
        crit = stats.norm.isf(alpha / 2)
        p, lo, hi = 2 * stats.norm.sf(abs(z)), est - crit * se, est + crit * se
    else:
        t = est / se
        crit = stats.t.isf(alpha / 2, df)
        p, lo, hi = 2 * stats.t.sf(abs(t), df), est - crit * se, est + crit * se
    return EffectResult(float(est), float(se), float(lo), float(hi), float(p), nc, nt,
                        float(mc), float(mt), method, alpha)


def welch_ttest(treat: np.ndarray, control: np.ndarray, alpha: float = 0.05) -> EffectResult:
    """Welch's unequal-variance t-test with Satterthwaite degrees of freedom."""
    t, c = np.asarray(treat, float), np.asarray(control, float)
    nt, nc = len(t), len(c)
    vt, vc = t.var(ddof=1) / nt, c.var(ddof=1) / nc
    se = math.sqrt(vt + vc)
    df = (vt + vc) ** 2 / (vt**2 / (nt - 1) + vc**2 / (nc - 1)) if se > 0 else nt + nc - 2
    return _normal_result(t.mean() - c.mean(), se, c.mean(), t.mean(), nc, nt, "welch_t", alpha, df)


def two_proportion_ztest(
    x_treat: int, n_treat: int, x_control: int, n_control: int, alpha: float = 0.05
) -> EffectResult:
    """Two-proportion z-test: pooled variance for the p-value, unpooled Wald interval."""
    pt, pc = x_treat / n_treat, x_control / n_control
    pool = (x_treat + x_control) / (n_treat + n_control)
    se0 = math.sqrt(pool * (1 - pool) * (1 / n_treat + 1 / n_control))
    se1 = math.sqrt(pt * (1 - pt) / n_treat + pc * (1 - pc) / n_control)
    res = _normal_result(pt - pc, se1, pc, pt, n_control, n_treat, "two_proportion_z", alpha)
    if se0 > 0:
        res.p_value = float(2 * stats.norm.sf(abs((pt - pc) / se0)))
    return res


def delta_method_ratio(
    num_t: np.ndarray, den_t: np.ndarray, num_c: np.ndarray, den_c: np.ndarray,
    alpha: float = 0.05,
) -> EffectResult:
    """Difference in ratio-of-means metrics ``sum(N)/sum(D)`` with a delta-method variance.

    For one arm, ``Var(R_hat) ~= [Var(N) - 2 R Cov(N, D) + R^2 Var(D)] / (n * mean(D)^2)``;
    arms are independent so the variances add. Needed whenever the analysis unit (user) differs
    from the denominator unit (sessions, impressions), where treating per-event rows as iid
    understates the variance.
    """

    def arm(n_, d_):
        n_, d_ = np.asarray(n_, float), np.asarray(d_, float)
        n = len(n_)
        r = n_.sum() / d_.sum()
        resid = n_ - r * d_
        var = resid.var(ddof=1) / (n * d_.mean() ** 2)
        return r, var, n

    rt, vt, nt = arm(num_t, den_t)
    rc, vc, nc = arm(num_c, den_c)
    return _normal_result(rt - rc, math.sqrt(vt + vc), rc, rt, nc, nt, "delta_method_ratio", alpha)


def cuped_theta(y: np.ndarray, x: np.ndarray) -> float:
    """Variance-minimising CUPED coefficient ``Cov(Y, X) / Var(X)`` (pooled over arms)."""
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc**2).sum())


def cuped(
    y_treat: np.ndarray, x_treat: np.ndarray, y_control: np.ndarray, x_control: np.ndarray,
    alpha: float = 0.05,
) -> EffectResult:
    """CUPED: ``Y_adj = Y - theta (X - mean(X))`` with one pooled theta, then Welch.

    ``X`` must be pre-treatment (or otherwise independent of assignment). Because assignment is
    random, ``E[X]`` is equal across arms and the adjustment keeps the estimator unbiased while
    removing the share ``rho^2`` of the variance that ``X`` explains. Estimating theta from the
    same data has a negligible effect on the interval at experiment sample sizes.
    """
    yt, xt = np.asarray(y_treat, float), np.asarray(x_treat, float)
    yc, xc = np.asarray(y_control, float), np.asarray(x_control, float)
    x = np.concatenate([xt, xc])
    # Pool the within-arm covariance so a true treatment effect cannot contaminate theta.
    xt0, xc0 = xt - xt.mean(), xc - xc.mean()
    num = (xt0 * (yt - yt.mean())).sum() + (xc0 * (yc - yc.mean())).sum()
    theta = num / ((xt0**2).sum() + (xc0**2).sum())
    xm = x.mean()
    res = welch_ttest(yt - theta * (xt - xm), yc - theta * (xc - xm), alpha)
    # Report arm means on the original scale, adjusted estimate for the effect.
    res.mean_control, res.mean_treatment = float(yc.mean()), float(yt.mean())
    res.method = "cuped"
    return res


def regression_adjusted(
    y: np.ndarray, treat: np.ndarray, covariates: np.ndarray, alpha: float = 0.05
) -> EffectResult:
    """Lin (2013) fully-interacted regression adjustment with HC2 standard errors.

    ``y ~ 1 + T + Xc + T*Xc`` with covariates centred at their overall mean. The coefficient on
    ``T`` is a consistent estimate of the ATE even if the linear model is misspecified, and is
    never asymptotically less efficient than the unadjusted difference in means.
    """
    y = np.asarray(y, float)
    t = np.asarray(treat, float)
    x = np.asarray(covariates, float)
    if x.ndim == 1:
        x = x[:, None]
    xc = x - x.mean(axis=0)
    design = np.column_stack([np.ones_like(y), t, xc, xc * t[:, None]])
    xtx_inv = np.linalg.pinv(design.T @ design)
    beta = xtx_inv @ design.T @ y
    resid = y - design @ beta
    h = np.einsum("ij,jk,ik->i", design, xtx_inv, design)
    w = resid**2 / np.clip(1 - h, 1e-8, None)
    cov = xtx_inv @ (design.T * w) @ design @ xtx_inv
    nt, nc = int(t.sum()), int((1 - t).sum())
    return _normal_result(
        float(beta[1]), float(math.sqrt(cov[1, 1])), float(y[t == 0].mean()),
        float(y[t == 1].mean()), nc, nt, "lin_regression_hc2", alpha,
        df=len(y) - design.shape[1],
    )


def cupac_covariate(
    features: np.ndarray, y: np.ndarray, n_folds: int = 5, seed: int = 0
) -> np.ndarray:
    """CUPAC-style control variate: a cross-fitted ML prediction of ``y`` from pre-period features.

    Out-of-fold predictions guarantee each unit's covariate is independent of its own outcome
    noise, so the prediction is a legitimate pre-treatment covariate: use it as ``X`` in
    :func:`cuped` or :func:`regression_adjusted`. Non-linear models capture structure a
    single pre-period metric misses, raising the effective ``rho^2``.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import KFold

    features = np.asarray(features, float)
    y = np.asarray(y, float)
    pred = np.zeros(len(y))
    for tr, te in KFold(n_folds, shuffle=True, random_state=seed).split(features):
        m = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1, random_state=seed)
        m.fit(features[tr], y[tr])
        pred[te] = m.predict(features[te])
    return pred


def bootstrap_diff(
    treat: np.ndarray, control: np.ndarray, method: str = "percentile",
    n_resamples: int = 4000, alpha: float = 0.05, seed: int = 0,
) -> EffectResult:
    """Bootstrap CI for the difference in means (``percentile`` or ``bca``).

    The p-value is the two-sided tail mass of the bootstrap distribution beyond zero; the BCa interval corrects for bias and skew and is the
    better choice for heavy-tailed metrics at moderate n.
    """
    t, c = np.asarray(treat, float), np.asarray(control, float)
    rng = np.random.default_rng(seed)
    est = t.mean() - c.mean()
    it = rng.integers(0, len(t), size=(n_resamples, len(t)))
    ic = rng.integers(0, len(c), size=(n_resamples, len(c)))
    boots = t[it].mean(axis=1) - c[ic].mean(axis=1)
    if method == "percentile":
        lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    elif method == "bca":
        z0 = stats.norm.ppf(np.clip((boots < est).mean(), 1e-6, 1 - 1e-6))
        # Jackknife acceleration (two independent samples: pool the leave-one-out effects).
        jt = (t.sum() - t) / (len(t) - 1) - c.mean()
        jc = t.mean() - (c.sum() - c) / (len(c) - 1)
        jack = np.concatenate([jt, jc])
        d = jack.mean() - jack
        a = (d**3).sum() / (6 * ((d**2).sum() ** 1.5 + 1e-300))
        zs = stats.norm.ppf([alpha / 2, 1 - alpha / 2])
        adj = stats.norm.cdf(z0 + (z0 + zs) / (1 - a * (z0 + zs)))
        lo, hi = np.quantile(boots, adj)
    else:
        raise ValueError("method must be 'percentile' or 'bca'")
    tail = min((boots <= 0).mean(), (boots >= 0).mean())
    p = float(min(1.0, max(2 * tail, 1.0 / (n_resamples + 1))))
    return EffectResult(float(est), float(boots.std(ddof=1)), float(lo), float(hi), p, len(c),
                        len(t), float(c.mean()), float(t.mean()), f"bootstrap_{method}", alpha)


def holm(pvalues: list[float] | np.ndarray, alpha: float = 0.05) -> dict:
    """Holm step-down: controls the family-wise error rate under arbitrary dependence."""
    p = np.asarray(pvalues, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return {"adjusted": adj.tolist(), "reject": (adj <= alpha).tolist(), "method": "holm"}


def benjamini_hochberg(pvalues: list[float] | np.ndarray, alpha: float = 0.05) -> dict:
    """Benjamini-Hochberg step-up: controls FDR for independent / PRDS p-values."""
    p = np.asarray(pvalues, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * m / (rank + 1))
        adj[i] = prev
    return {"adjusted": adj.tolist(), "reject": (adj <= alpha).tolist(), "method": "bh"}


def correct_pvalues(pvalues: list[float], method: str = "holm", alpha: float = 0.05) -> dict:
    if method == "holm":
        return holm(pvalues, alpha)
    if method in ("bh", "fdr_bh"):
        return benjamini_hochberg(pvalues, alpha)
    if method == "none":
        p = np.asarray(pvalues, float)
        return {"adjusted": p.tolist(), "reject": (p <= alpha).tolist(), "method": "none"}
    raise ValueError(f"unknown correction method: {method}")


def noninferiority_test(
    effect: EffectResult, margin: float, higher_is_better: bool = True, alpha: float = 0.05,
) -> dict:
    """One-sided non-inferiority test for a guardrail using an estimate and its standard error.

    Higher-is-better: ``H0: delta <= -margin`` vs ``H1: delta > -margin`` (reject => the
    treatment is not worse than the margin). For lower-is-better metrics the sign is flipped.
    The decision uses the one-sided ``1 - alpha`` bound, equal to the two-sided ``1 - 2*alpha``
    interval. Guardrails form an intersection-union test ("all pass"), so no multiplicity
    correction is needed for the joint claim that every guardrail is non-inferior.
    """
    if margin <= 0:
        raise ValueError("margin must be positive")
    sign = 1.0 if higher_is_better else -1.0
    est = sign * effect.estimate
    z = (est + margin) / effect.se if effect.se > 0 else math.inf
    p = float(stats.norm.sf(z))
    bound = est - stats.norm.isf(alpha) * effect.se
    return {
        "margin": margin,
        "estimate": float(est),
        "one_sided_bound": float(bound),
        "p_value": p,
        "non_inferior": bool(p < alpha),
        "alpha": alpha,
    }
