"""Heterogeneous treatment effects: corrected segment analysis and a cross-fitted CATE model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .analysis import correct_pvalues, welch_ttest


def segment_effects(
    df: pd.DataFrame,
    segment_col: str,
    metric_col: str = "y",
    variant_col: str = "variant",
    control: str = "control",
    correction: str = "holm",
    alpha: float = 0.05,
) -> dict:
    """Per-segment Welch effects, multiplicity-corrected, plus an omnibus heterogeneity test.

    The omnibus test is Cochran's Q on the segment effects (inverse-variance weights):
    ``Q = sum w_s (tau_s - tau_bar)^2 ~ chi2_{S-1}`` under homogeneous effects. Look at
    individual segments only if Q (or a pre-registered hypothesis) justifies it; the
    per-segment p-values are always corrected over the number of segments examined.
    """
    rows = []
    for seg, g in df.groupby(segment_col):
        t = g.loc[g[variant_col] != control, metric_col].to_numpy()
        c = g.loc[g[variant_col] == control, metric_col].to_numpy()
        if len(t) < 2 or len(c) < 2:
            continue
        r = welch_ttest(t, c, alpha)
        rows.append({"segment": str(seg), **r.to_dict()})
    if not rows:
        return {"segments": [], "q_stat": None, "q_p_value": None, "correction": correction}
    adj = correct_pvalues([r["p_value"] for r in rows], correction, alpha)
    for r, pa, rej in zip(rows, adj["adjusted"], adj["reject"]):
        r["p_adjusted"], r["significant"] = pa, bool(rej)
    est = np.array([r["estimate"] for r in rows])
    w = 1.0 / np.array([r["se"] for r in rows]) ** 2
    bar = (w * est).sum() / w.sum()
    q = float((w * (est - bar) ** 2).sum())
    dof = len(rows) - 1
    p = float(stats.chi2.sf(q, dof)) if dof > 0 else float("nan")
    return {"segments": rows, "q_stat": q, "q_p_value": p, "correction": correction}


def t_learner_cate(
    df: pd.DataFrame,
    features: list[str],
    metric_col: str = "y",
    variant_col: str = "variant",
    control: str = "control",
    backend: str = "sklearn",
    n_folds: int = 4,
    n_groups: int = 4,
    seed: int = 0,
) -> dict:
    """Cross-fitted T-learner CATE with an honest, randomisation-based validation.

    A separate outcome model is fit on treated and control units in the training folds; the
    CATE for held-out units is ``mu1(x) - mu0(x)``. Because every unit's CATE is predicted by
    models that never saw it, units can be grouped into CATE quantiles and the *realised*
    effect in each group estimated with an ordinary Welch test (valid under randomisation).
    A useful model shows monotonically increasing realised effects across groups.

    ``backend='econml'`` uses :class:`econml.metalearners.TLearner`; ``'sklearn'`` uses two
    ``HistGradientBoostingRegressor`` fits and has no extra dependency.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import KFold

    x = df[features].to_numpy(float)
    y = df[metric_col].to_numpy(float)
    t = (df[variant_col] != control).to_numpy()
    cate = np.zeros(len(df))
    for tr, te in KFold(n_folds, shuffle=True, random_state=seed).split(x):
        if backend == "econml":
            from econml.metalearners import TLearner

            tl = TLearner(models=HistGradientBoostingRegressor(max_iter=100, random_state=seed))
            tl.fit(y[tr], t[tr].astype(int), X=x[tr])
            cate[te] = np.ravel(tl.effect(x[te]))
        elif backend == "sklearn":
            m1 = HistGradientBoostingRegressor(max_iter=100, random_state=seed).fit(
                x[tr][t[tr]], y[tr][t[tr]])
            m0 = HistGradientBoostingRegressor(max_iter=100, random_state=seed).fit(
                x[tr][~t[tr]], y[tr][~t[tr]])
            cate[te] = m1.predict(x[te]) - m0.predict(x[te])
        else:
            raise ValueError("backend must be 'sklearn' or 'econml'")

    order = np.argsort(cate)
    groups = np.array_split(order, n_groups)
    out = []
    for i, idx in enumerate(groups):
        tt, cc = y[idx][t[idx]], y[idx][~t[idx]]
        r = welch_ttest(tt, cc)
        out.append({"group": i + 1, "mean_predicted_cate": float(cate[idx].mean()),
                    "realised_effect": r.estimate, "ci_low": r.ci_low, "ci_high": r.ci_high,
                    "n": len(idx)})
    corr = float(np.corrcoef([g["mean_predicted_cate"] for g in out],
                             [g["realised_effect"] for g in out])[0, 1]) if n_groups > 2 else None
    return {"cate": cate, "groups": out, "group_corr": corr, "backend": backend}
