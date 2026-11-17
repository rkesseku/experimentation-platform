"""Synthetic experiment generator with known ground truth.

Each unit has pre-experiment features ``f1..f3``, a latent baseline ``u`` (a non-linear
function of the features), a pre-period metric ``x_pre`` correlated with the in-experiment
outcome, an arrival time ``ts`` (days since start) and a segment label. The in-experiment
outcome under control is ``Y(0)``; the treatment adds an effect that can be

* constant (``effect``),
* heterogeneous in ``f1`` (``hte`` * f1) and/or by segment (``segment_effects``),
* subject to a novelty effect that decays with arrival time,

and the generator can emulate an SRM bug (treated units silently dropped, optionally the
heaviest users first).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .assignment import assign_many


@dataclass
class SimConfig:
    n: int = 20_000
    metric: str = "normal"  # normal | lognormal | heavy_tail | binary | ratio
    effect: float = 0.0  # absolute for normal/heavy_tail/binary; relative for lognormal/ratio
    base: float = 10.0  # mean (normal/heavy_tail), median-scale (lognormal), rate (binary)
    sd: float = 4.0
    rho: float = 0.5  # corr(x_pre, Y(0)) at the unit level
    weights: dict[str, float] = field(default_factory=lambda: {"control": 0.5, "treatment": 0.5})
    days: float = 14.0
    hte: float = 0.0  # effect += hte * f1
    segment_effects: dict[str, float] = field(default_factory=dict)  # extra effect per segment
    novelty_boost: float = 0.0  # effect * (1 + boost * exp(-ts / decay))
    novelty_decay_days: float = 3.0
    srm_drop: float = 0.0  # fraction of treated units lost
    srm_biased: bool = True  # dropped units are the highest-Y users (else random)
    use_hash: bool = True
    salt: str = "sim-salt"
    seed: int = 0


def _latent(rng: np.random.Generator, n: int, rho: float) -> tuple[np.ndarray, ...]:
    f = rng.normal(size=(n, 3))
    raw = np.sin(2 * f[:, 0]) + f[:, 1] * f[:, 2] + 0.5 * f[:, 0]
    u = (raw - raw.mean()) / raw.std()
    load = np.sqrt(rho)
    x_pre = load * u + np.sqrt(1 - rho) * rng.normal(size=n)  # latent scale, rescaled below
    z0 = load * u + np.sqrt(1 - rho) * rng.normal(size=n)  # Y(0) latent, corr(x_pre, z0)=rho
    return f, x_pre, z0, u


def simulate_experiment(cfg: SimConfig | None = None, **kwargs) -> pd.DataFrame:
    """Simulate one experiment; returns one row per *observed* unit.

    Columns: ``unit_id, variant, ts, segment, f1, f2, f3, x_pre, y`` and, for
    ``metric='ratio'``, ``num`` (revenue) and ``den`` (sessions) with ``y = num``.
    Ground truth is attached in ``df.attrs['truth']``.
    """
    cfg = cfg or SimConfig(**kwargs)
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n
    f, x_lat, z0, u = _latent(rng, n, cfg.rho)
    ts = rng.uniform(0, cfg.days, n)
    seg = rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])
    ids = np.arange(n)
    if cfg.use_hash:
        variant = assign_many(ids, cfg.salt, cfg.weights)
    else:
        names = list(cfg.weights)
        variant = rng.choice(names, size=n, p=np.array(list(cfg.weights.values())) / sum(cfg.weights.values()))
    treated = variant != "control"

    tau = np.full(n, cfg.effect, float)
    tau += cfg.hte * f[:, 0]
    for s, extra in cfg.segment_effects.items():
        tau += np.where(seg == s, extra, 0.0)
    if cfg.novelty_boost:
        tau *= 1 + cfg.novelty_boost * np.exp(-ts / cfg.novelty_decay_days)
    tau_applied = np.where(treated, tau, 0.0)

    extra_cols: dict[str, np.ndarray] = {}
    m = cfg.metric
    if m == "normal":
        y = cfg.base + cfg.sd * z0 + tau_applied
        x_pre = cfg.base + cfg.sd * x_lat
        true_ate = float(tau.mean())
    elif m == "heavy_tail":
        # Student-t(3) idiosyncratic noise around the same correlated signal.
        e = stats.t.rvs(3, size=n, random_state=rng) / np.sqrt(3.0)
        y = cfg.base + cfg.sd * (np.sqrt(cfg.rho) * u + np.sqrt(1 - cfg.rho) * e) + tau_applied
        x_pre = cfg.base + cfg.sd * x_lat
        true_ate = float(tau.mean())
    elif m == "lognormal":
        y0 = cfg.base * np.exp(0.8 * z0)
        y = y0 * (1 + tau_applied)
        x_pre = cfg.base * np.exp(0.8 * x_lat)
        true_ate = float((cfg.base * np.exp(0.8**2 / 2) * tau).mean())
    elif m == "binary":
        p_t = np.clip(cfg.base + tau_applied, 1e-6, 1 - 1e-6)
        thresh = stats.norm.isf(p_t)  # y = 1 iff latent exceeds threshold => P(y=1) = p_t
        y = (z0 > thresh).astype(float)
        x_pre = (x_lat > stats.norm.isf(cfg.base)).astype(float)
        true_ate = float(tau.mean())
    elif m == "ratio":
        lam = 3.0 * np.exp(0.3 * np.sqrt(cfg.rho) * z0)
        sessions = 1 + rng.poisson(lam)
        value = np.exp(rng.normal(np.log(cfg.base) - 0.5, 0.7, size=n))  # mean per session
        num = sessions * value * (1 + tau_applied)
        extra_cols = {"num": num, "den": sessions.astype(float)}
        y = num
        x_pre = 1 + 3.0 * np.exp(0.3 * x_lat)
        true_ate = float((sessions * value).mean() * tau.mean())  # ATE on per-user revenue
    else:
        raise ValueError(f"unknown metric: {m}")

    df = pd.DataFrame(
        {"unit_id": ids, "variant": variant, "ts": ts, "segment": seg, "f1": f[:, 0],
         "f2": f[:, 1], "f3": f[:, 2], "x_pre": x_pre, "y": y, **extra_cols}
    )
    if cfg.srm_drop > 0:
        tidx = np.flatnonzero(treated)
        k = round(cfg.srm_drop * len(tidx))
        drop = tidx[np.argsort(-y[tidx])[:k]] if cfg.srm_biased else rng.choice(tidx, k, replace=False)
        df = df.drop(index=drop).reset_index(drop=True)
    df = df.sort_values("ts").reset_index(drop=True)
    df.attrs["truth"] = {"ate": true_ate, "unit_effects_mean": float(tau.mean())}
    return df
