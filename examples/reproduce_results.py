"""Regenerates every number quoted in the README "Results" section (all data simulated).

Run:  python examples/reproduce_results.py        (about 1-2 minutes)
Writes examples/results.json and prints a summary.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from expplatform import analysis as an
from expplatform import design
from expplatform.assignment import srm_from_frame
from expplatform.demo import demo_config, demo_frame
from expplatform.report import build_readout
from expplatform.sequential import group_sequential_bounds, peeking_simulation
from expplatform.simulator import SimConfig, simulate_experiment

W = {"control": 0.5, "treatment": 0.5}


def peeking() -> dict:
    out = {"seed": 2026}
    null = peeking_simulation(n_sims=20_000, n_per_look=100, n_looks=50, effect=0.0,
                              tau=0.2, seed=2026)
    alt = peeking_simulation(n_sims=20_000, n_per_look=100, n_looks=50, effect=0.1,
                             tau=0.1, seed=2027)
    est = peeking_simulation(n_sims=1000, n_per_look=100, n_looks=30, effect=0.0, tau=0.2,
                             estimate_variance=True, seed=2028)
    out["null_known_variance"] = null
    out["alt_known_variance_effect_0.1"] = alt
    out["null_estimated_variance"] = est
    out["obf_bounds_5_looks"] = group_sequential_bounds(np.arange(1, 6) / 5).tolist()
    return out


def cuped() -> dict:
    rho, eff, reps = 0.5, 0.4, 500
    raw, adj = [], []
    for s in range(reps):
        df = simulate_experiment(SimConfig(n=2000, use_hash=False, seed=s, effect=eff, rho=rho,
                                           sd=2.0))
        t, c = df[df.variant == "treatment"], df[df.variant == "control"]
        raw.append(an.welch_ttest(t.y.to_numpy(), c.y.to_numpy()).estimate)
        adj.append(an.cuped(t.y.to_numpy(), t.x_pre.to_numpy(), c.y.to_numpy(),
                            c.x_pre.to_numpy()).estimate)
    raw, adj = np.array(raw), np.array(adj)
    # CUPAC vs single pre-period covariate (n=8000, 8 datasets)
    se_pre, se_cupac, se_raw = [], [], []
    for s in range(8):
        df = simulate_experiment(SimConfig(n=8000, use_hash=False, seed=100 + s, effect=0.2,
                                           rho=0.5, sd=2.0))
        pred = an.cupac_covariate(df[["f1", "f2", "f3"]].to_numpy(), df.y.to_numpy(), seed=s)
        t = (df.variant == "treatment").to_numpy()
        y, x = df.y.to_numpy(), df.x_pre.to_numpy()
        se_raw.append(an.welch_ttest(y[t], y[~t]).se)
        se_pre.append(an.cuped(y[t], x[t], y[~t], x[~t]).se)
        se_cupac.append(an.cuped(y[t], pred[t], y[~t], pred[~t]).se)
    return {
        "reps": reps, "n_per_dataset": 2000, "true_effect": eff, "corr_x_y": rho,
        "mean_estimate_unadjusted": float(raw.mean()), "mean_estimate_cuped": float(adj.mean()),
        "mc_se_of_mean_cuped": float(adj.std() / np.sqrt(reps)),
        "variance_ratio_cuped_over_unadjusted": float(adj.var() / raw.var()),
        "theory_1_minus_rho2": 1 - rho**2,
        "cupac": {
            "datasets": 8, "n": 8000,
            "variance_reduction_pre_period_cuped": float(1 - np.mean(np.square(se_pre) / np.square(se_raw))),
            "variance_reduction_cupac_gbm": float(1 - np.mean(np.square(se_cupac) / np.square(se_raw))),
        },
    }


def calibration() -> dict:
    rng = np.random.default_rng(31)
    sims, n = 5000, 200
    rej = sum(an.welch_ttest(rng.normal(0, 1, n), rng.normal(0, 2, n)).p_value < 0.05
              for _ in range(sims))
    delta, sd = 0.5, 3.0
    nn = design.sample_size_means(delta, sd).n_control
    rng = np.random.default_rng(32)
    sims_p = 3000
    pw = sum(an.welch_ttest(rng.normal(delta, sd, nn), rng.normal(0, sd, nn)).p_value < 0.05
             for _ in range(sims_p))
    # delta-method coverage
    rng = np.random.default_rng(33)
    mu, s, rel, cover, reps, m = 0.0, 0.5, 0.10, 0, 2000, 1500
    truth = rel * np.exp(mu + s**2 / 2)
    for _ in range(reps):
        dc = 1 + rng.poisson(3, m).astype(float)
        dt = 1 + rng.poisson(3, m).astype(float)
        nc = np.array([rng.lognormal(mu, s, int(k)).sum() for k in dc])
        nt = (1 + rel) * np.array([rng.lognormal(mu, s, int(k)).sum() for k in dt])
        r = an.delta_method_ratio(nt, dt, nc, dc)
        cover += r.ci_low <= truth <= r.ci_high
    return {
        "type_i_welch": {"seed": 31, "sims": sims, "n_per_arm": n, "rejection_rate": rej / sims},
        "power": {"seed": 32, "sims": sims_p, "delta": delta, "sd": sd, "n_per_arm_from_calculator": nn,
                  "target_power": 0.8, "simulated_power": pw / sims_p},
        "delta_method_coverage": {"seed": 33, "reps": reps, "users_per_arm": m, "nominal": 0.95,
                                  "coverage": cover / reps},
        "sample_size_examples": {
            "means_d0.2_sd1": design.sample_size_means(0.2, 1.0).n_control,
            "means_d0.2_sd1_cuped_r2_0.25": design.sample_size_means(0.2, 1.0, r2=0.25).n_control,
            "props_10_to_12pct": design.sample_size_proportions(0.10, 0.12).n_control,
        },
    }


def srm() -> dict:
    rows = {}
    for drop in (0.0, 0.01, 0.02, 0.05):
        hits = 0
        for s in range(200):
            df = simulate_experiment(SimConfig(n=40_000, use_hash=False, seed=s, srm_drop=drop,
                                               srm_biased=False))
            hits += srm_from_frame(df, W)["mismatch"]
        rows[f"drop_{drop}"] = hits / 200
    return {"alpha_threshold": 0.001, "n": 40_000, "reps": 200, "flag_rate": rows}


def readout() -> dict:
    cfg = demo_config()
    clean = build_readout(demo_frame(seed=7, n=40_000), cfg)
    bugged = build_readout(demo_frame(seed=7, n=40_000, srm_drop=0.03), cfg)
    p = clean["metrics"][0]
    return {
        "seed": 7, "n": 40_000, "true_primary_effect_approx": 0.275,
        "clean": {"status": clean["status"], "srm_p": clean["srm"]["p_value"],
                  "primary_estimate": p["effect"]["estimate"],
                  "primary_ci": [p["effect"]["ci_low"], p["effect"]["ci_high"]],
                  "variance_reduction": p["variance_reduction"],
                  "mSPRT_rejected_at_n": clean["sequential"]["rejected_at_n"],
                  "segments_significant": [s["segment"] for s in clean["segments"]["segments"]
                                           if s["significant"]],
                  "cochran_q_p": clean["segments"]["q_p_value"]},
        "srm_bug_3pct_treated_lost": {"status": bugged["status"], "srm_p": bugged["srm"]["p_value"]},
    }


def main() -> None:
    t0 = time.time()
    res = {"peeking": peeking(), "cuped": cuped(), "calibration": calibration(), "srm": srm(),
           "readout": readout()}
    path = Path(__file__).with_name("results.json")
    path.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"\n(done in {time.time() - t0:.0f}s; wrote {path})")


if __name__ == "__main__":
    main()
