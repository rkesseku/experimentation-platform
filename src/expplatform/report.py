"""Experiment readout: composes SRM, adjusted effects, corrections, guardrails and monitoring."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import analysis as an
from .assignment import assign_many, srm_from_frame
from .cate import segment_effects
from .config import ExperimentConfig, MetricSpec
from .sequential import group_sequential_bounds, msprt_path


def _clean(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    cols = [c for c in cols if c]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")
    return df.dropna(subset=cols)


def metric_effect(
    df: pd.DataFrame, spec: MetricSpec, cfg: ExperimentConfig, adjusted: bool = True
) -> an.EffectResult:
    """Effect of treatment on one metric using the method implied by its spec."""
    vc = cfg.variant_col
    if spec.kind == "ratio":
        d = _clean(df, [spec.numerator, spec.denominator, vc])
        t, c = d[d[vc] == cfg.treatment], d[d[vc] == cfg.control]
        return an.delta_method_ratio(
            t[spec.numerator].to_numpy(), t[spec.denominator].to_numpy(),
            c[spec.numerator].to_numpy(), c[spec.denominator].to_numpy(), cfg.alpha)
    use_cov = adjusted and spec.covariate is not None
    d = _clean(df, [spec.col, vc] + ([spec.covariate] if use_cov else []))
    t, c = d[d[vc] == cfg.treatment], d[d[vc] == cfg.control]
    if use_cov:
        return an.cuped(t[spec.col].to_numpy(), t[spec.covariate].to_numpy(),
                        c[spec.col].to_numpy(), c[spec.covariate].to_numpy(), cfg.alpha)
    if spec.kind == "binary":
        return an.two_proportion_ztest(int(t[spec.col].sum()), len(t), int(c[spec.col].sum()),
                                       len(c), cfg.alpha)
    return an.welch_ttest(t[spec.col].to_numpy(), c[spec.col].to_numpy(), cfg.alpha)


def _sequential(df: pd.DataFrame, spec: MetricSpec, cfg: ExperimentConfig) -> dict:
    """mSPRT confidence sequence and O'Brien-Fleming z-path for the primary metric."""
    sq = cfg.sequential
    d = df.sort_values(cfg.time_col, kind="stable") if cfg.time_col in df.columns else df
    d = d.reset_index(drop=True)
    n = len(d)
    full = metric_effect(d, spec, cfg)
    sigma2 = full.se**2 * full.n_treatment * full.n_control / (full.n_treatment + full.n_control)
    tau = sq.mixture_sd or 0.1 * math.sqrt(sigma2)

    def look(m: int) -> an.EffectResult | None:
        sub = d.iloc[:m]
        if (sub[cfg.variant_col] == cfg.treatment).sum() < 30 or (
            sub[cfg.variant_col] == cfg.control
        ).sum() < 30:
            return None
        try:
            return metric_effect(sub, spec, cfg)
        except (KeyError, ValueError):
            return None

    sizes = np.unique(np.round(np.linspace(n / sq.looks, n, sq.looks)).astype(int))
    res = [(m, look(m)) for m in sizes]
    res = [(m, r) for m, r in res if r is not None and r.se > 0]
    if len(res) < 2:
        return {"enabled": False, "reason": "not enough data for sequential monitoring"}
    ms = np.array([m for m, _ in res])
    est = np.array([r.estimate for _, r in res])
    var = np.array([r.se**2 for _, r in res])
    path = msprt_path(est, var, tau**2, cfg.alpha)

    g = sq.planned_gs_looks
    fr = np.arange(1, g + 1) / g
    bounds = group_sequential_bounds(fr, cfg.alpha, "obf")
    gs = []
    for f, b in zip(fr, bounds):
        r = look(round(f * n))
        if r is not None and r.se > 0:
            gs.append({"info_fraction": float(f), "z": float(r.estimate / r.se),
                       "bound": float(b), "crossed": bool(abs(r.estimate / r.se) >= b)})
    return {
        "enabled": True,
        "metric": spec.name,
        "mixture_sd": float(tau),
        "n": ms.tolist(),
        "estimate": est.tolist(),
        "cs_low": path.cs_low.tolist(),
        "cs_high": path.cs_high.tolist(),
        "always_valid_p": path.p_always_valid.tolist(),
        "rejected_at_n": int(ms[path.first_rejection()]) if path.first_rejection() is not None
        else None,
        "significant": bool(path.crossed[-1]),
        "obf": gs,
    }


def build_readout(df: pd.DataFrame, cfg: ExperimentConfig, include_segments: bool = True) -> dict:
    """Full analysis JSON for one experiment (two variants)."""
    warnings: list[str] = []
    srm = srm_from_frame(df, cfg.variants, cfg.variant_col, cfg.srm_threshold)
    if srm["mismatch"]:
        warnings.append("Sample-ratio mismatch: results are not trustworthy until resolved.")

    assigned = assign_many(df[cfg.unit_col], cfg.salt, cfg.variants)
    n_bad = int((assigned != df[cfg.variant_col].to_numpy()).sum())
    assignment_check = {"checked": len(df), "mismatched": n_bad,
                        "ok": n_bad == 0}
    if n_bad:
        warnings.append(f"{n_bad} units carry a variant that differs from the hash assignment.")

    main_specs = [m for m in cfg.metrics if m.role != "guardrail"]
    results, pvals = [], []
    for spec in main_specs:
        adj = metric_effect(df, spec, cfg, adjusted=True)
        raw = metric_effect(df, spec, cfg, adjusted=False)
        vr = 1.0 - (adj.se / raw.se) ** 2 if raw.se > 0 else 0.0
        results.append({"name": spec.name, "role": spec.role, "kind": spec.kind,
                        "effect": adj.to_dict(), "unadjusted": raw.to_dict(),
                        "variance_reduction": float(vr)})
        pvals.append(adj.p_value)
    corr = an.correct_pvalues(pvals, cfg.correction, cfg.alpha)
    for r, pa, rej in zip(results, corr["adjusted"], corr["reject"]):
        r["p_adjusted"], r["significant"] = pa, bool(rej)

    guardrails = []
    for spec in (m for m in cfg.metrics if m.role == "guardrail"):
        e = metric_effect(df, spec, cfg, adjusted=True)
        ni = an.noninferiority_test(e, spec.ni_margin, spec.higher_is_better, cfg.alpha)
        guardrails.append({"name": spec.name, "effect": e.to_dict(), **ni})

    primary = next(m for m in cfg.metrics if m.role == "primary")
    seq = _sequential(df, primary, cfg) if cfg.sequential.enabled else {"enabled": False}
    seg = (segment_effects(df, cfg.segment_col, primary.col, cfg.variant_col, cfg.control,
                           cfg.correction, cfg.alpha)
           if include_segments and cfg.segment_col and primary.kind != "ratio" else None)

    guard_ok = all(g["non_inferior"] for g in guardrails)
    prim = next(r for r in results if r["role"] == "primary")
    if srm["mismatch"]:
        status = "invalid_srm"
    elif prim["significant"] and prim["effect"]["estimate"] > 0 and guard_ok:
        status = "ship_candidate"
    elif prim["significant"] and not guard_ok:
        status = "hold_guardrail"
    else:
        status = "inconclusive"
    return {
        "experiment": cfg.name,
        "status": status,
        "n_units": len(df),
        "srm": srm,
        "assignment_check": assignment_check,
        "metrics": results,
        "guardrails": guardrails,
        "correction": cfg.correction,
        "sequential": seq,
        "segments": seg,
        "warnings": warnings,
    }
