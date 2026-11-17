"""Anytime-valid and group-sequential monitoring.

* **mSPRT / confidence sequences** (Robbins; Johari et al.; Howard et al.). With a
  ``N(0, tau^2)`` mixing prior on the effect, the likelihood ratio of the running estimate
  ``D_n ~ N(theta, V_n)`` against ``theta = 0`` is a non-negative martingale under H0, so by
  Ville's inequality ``P(sup_n Lambda_n >= 1/alpha) <= alpha``: the test may be peeked at after
  every observation.
* **Group-sequential O'Brien-Fleming-type boundaries** via the Lan-DeMets alpha-spending
  function, computed exactly by numerical integration of the sub-density of the Brownian
  motion ``S_t`` (``Var S_t = t``) between looks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

# ----------------------------------------------------------------------------- mSPRT


def msprt_log_lr(d: np.ndarray, v: np.ndarray, tau2: float, theta0: float = 0.0) -> np.ndarray:
    """``log Lambda`` of the normal-mixture SPRT for an estimate ``d`` with variance ``v``.

    ``Lambda = sqrt(V / (V + tau^2)) * exp( tau^2 (d - theta0)^2 / (2 V (V + tau^2)) )``.
    """
    d, v = np.asarray(d, float), np.asarray(v, float)
    return 0.5 * np.log(v / (v + tau2)) + tau2 * (d - theta0) ** 2 / (2 * v * (v + tau2))


def cs_radius(v: np.ndarray, tau2: float, alpha: float = 0.05) -> np.ndarray:
    """Half-width of the mixture confidence sequence at variance ``v``.

    Solves ``Lambda(theta) < 1/alpha`` for ``theta``:
    ``r^2 = 2 V (V + tau^2) / tau^2 * ( log(1/alpha) + 0.5 log((V + tau^2) / V) )``.
    """
    v = np.asarray(v, float)
    return np.sqrt(2 * v * (v + tau2) / tau2 * (math.log(1 / alpha) + 0.5 * np.log((v + tau2) / v)))


@dataclass
class SequentialPath:
    log_lr: np.ndarray
    p_always_valid: np.ndarray
    cs_low: np.ndarray
    cs_high: np.ndarray
    crossed: np.ndarray  # boolean, whether H0 is rejected at or before each look

    def first_rejection(self) -> int | None:
        idx = np.flatnonzero(self.crossed)
        return int(idx[0]) if idx.size else None


def msprt_path(
    d: np.ndarray, v: np.ndarray, tau2: float, alpha: float = 0.05
) -> SequentialPath:
    """mSPRT statistics along a sequence of looks (``d[k]``, ``v[k]`` at look ``k``).

    The always-valid p-value is ``min(1, 1 / max_{j<=k} Lambda_j)`` (running minimum), and the
    confidence sequence is the running intersection of the per-look radii, which keeps the
    ``1 - alpha`` coverage uniform over time.
    """
    d, v = np.asarray(d, float), np.asarray(v, float)
    llr = msprt_log_lr(d, v, tau2)
    runmax = np.maximum.accumulate(llr)
    p = np.minimum(1.0, np.exp(-runmax))
    r = cs_radius(v, tau2, alpha)
    lo = np.maximum.accumulate(d - r)
    hi = np.minimum.accumulate(d + r)
    return SequentialPath(llr, p, lo, hi, runmax >= math.log(1 / alpha))


# ------------------------------------------------------------------ group sequential


def _spend(t: np.ndarray, alpha: float, kind: str) -> np.ndarray:
    if kind == "obf":
        return 2 - 2 * stats.norm.cdf(stats.norm.isf(alpha / 2) / np.sqrt(t))
    if kind == "pocock":
        return alpha * np.log(1 + (math.e - 1) * t)
    raise ValueError("spending must be 'obf' or 'pocock'")


def group_sequential_bounds(
    info_fractions: list[float] | np.ndarray, alpha: float = 0.05, spending: str = "obf",
    nodes: int = 400,
) -> np.ndarray:
    """Two-sided Lan-DeMets boundaries on the z scale at the given information fractions.

    Returns ``c_k`` such that the test rejects at look ``k`` if ``|Z_k| >= c_k``; the overall
    type-I error is ``alpha`` up to numerical-integration error. The sub-density of the
    surviving Brownian path is carried between looks on Gauss-Legendre nodes spanning
    ``[-u_k, u_k]`` (Armitage-McPherson-Rowe recursion).
    """
    t = np.asarray(info_fractions, float)
    if np.any(np.diff(t) <= 0) or t[0] <= 0 or abs(t[-1] - 1.0) > 1e-9:
        raise ValueError("info_fractions must be increasing and end at 1")
    spend = _spend(t, alpha, spending)
    increments = np.diff(np.concatenate([[0.0], spend]))
    gx, gw = np.polynomial.legendre.leggauss(nodes)
    bounds = np.empty(len(t))
    x = wts = dens = None  # nodes, quadrature weights, surviving density at nodes
    prev_t = 0.0
    for k, tk in enumerate(t):
        sd = math.sqrt(tk - prev_t)
        if k == 0:

            def stop_prob(u: float, sd=sd) -> float:
                return 2 * stats.norm.sf(u / sd)
        else:

            def stop_prob(u: float, sd=sd, x=x, m=wts * dens) -> float:
                tail = stats.norm.cdf((-u - x) / sd) + stats.norm.sf((u - x) / sd)
                return float((m * tail).sum())

        target = increments[k]
        u = optimize.brentq(lambda q, tgt=target: stop_prob(q) - tgt, 1e-6, 12.0 * math.sqrt(tk),
                            xtol=1e-12)
        bounds[k] = u / math.sqrt(tk)
        if k < len(t) - 1:
            xn, wn = u * gx, u * gw
            if k == 0:
                dn = stats.norm.pdf(xn, scale=sd)
            else:
                kern = stats.norm.pdf((xn[:, None] - x[None, :]) / sd) / sd
                dn = kern @ (wts * dens)
            x, wts, dens = xn, wn, dn
        prev_t = tk
    return bounds


# ------------------------------------------------------------------------ simulation


def peeking_simulation(
    n_sims: int = 2000,
    n_per_look: int = 100,
    n_looks: int = 50,
    effect: float = 0.0,
    sigma: float = 1.0,
    alpha: float = 0.05,
    tau: float = 0.2,
    gs_looks: int = 5,
    estimate_variance: bool = False,
    seed: int = 0,
) -> dict:
    """Monte-Carlo comparison of fixed-horizon, naive peeking, mSPRT and O'Brien-Fleming.

    Per arm, ``n_per_look`` new observations arrive between looks. With
    ``estimate_variance=False`` the arm-mean difference at each look is simulated exactly as
    ``N(effect, 2 sigma^2 / n)`` through its independent block increments (sigma known);
    with ``True`` raw normal data are generated and the Welch variance is re-estimated at every
    look (slower, memory ``n_sims * n_looks * n_per_look``).

    Returns rejection rates ("ever rejected") for:
      ``fixed`` (one look at the end), ``naive_peeking`` (|z| > z_{alpha/2} at any look),
      ``msprt`` (Lambda >= 1/alpha at any look), ``obf`` (Lan-DeMets at ``gs_looks`` equally
      spaced looks), plus the mean stopping information fraction among msprt rejections.
    """
    rng = np.random.default_rng(seed)
    ns = n_per_look * np.arange(1, n_looks + 1)
    if estimate_variance:
        a = rng.normal(effect, sigma, size=(n_sims, ns[-1]))
        b = rng.normal(0.0, sigma, size=(n_sims, ns[-1]))
        ca, cb = np.cumsum(a, 1)[:, ns - 1], np.cumsum(b, 1)[:, ns - 1]
        qa, qb = np.cumsum(a**2, 1)[:, ns - 1], np.cumsum(b**2, 1)[:, ns - 1]
        ma, mb = ca / ns, cb / ns
        va = (qa - ns * ma**2) / (ns - 1)
        vb = (qb - ns * mb**2) / (ns - 1)
        d = ma - mb
        v = (va + vb) / ns
    else:
        blocks = rng.normal(effect, sigma * math.sqrt(2.0 / n_per_look), size=(n_sims, n_looks))
        d = np.cumsum(blocks, axis=1) / np.arange(1, n_looks + 1)
        v = np.broadcast_to(2 * sigma**2 / ns, d.shape)
    z = d / np.sqrt(v)
    zcrit = stats.norm.isf(alpha / 2)
    tau2 = tau**2
    llr = msprt_log_lr(d, v, tau2)
    ms_rej = np.maximum.accumulate(llr, axis=1)[:, -1] >= math.log(1 / alpha)
    first = np.argmax(llr >= math.log(1 / alpha), axis=1)
    stop_frac = float(((first[ms_rej] + 1) / n_looks).mean()) if ms_rej.any() else float("nan")
    idx = np.unique(np.round(np.linspace(0, n_looks, gs_looks + 1)[1:]).astype(int)) - 1
    bounds = group_sequential_bounds(ns[idx] / ns[-1], alpha, "obf")
    return {
        "fixed": float((np.abs(z[:, -1]) > zcrit).mean()),
        "naive_peeking": float((np.abs(z) > zcrit).any(axis=1).mean()),
        "msprt": float(ms_rej.mean()),
        "obf": float((np.abs(z[:, idx]) >= bounds).any(axis=1).mean()),
        "msprt_mean_stop_fraction": stop_frac,
        "n_sims": n_sims,
        "n_looks": n_looks,
        "effect": effect,
        "obf_bounds": bounds.tolist(),
    }
