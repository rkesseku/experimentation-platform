"""Demo scenarios shared by the dashboard, the CLI demo and the README results script."""

from __future__ import annotations

import pandas as pd

from .config import ExperimentConfig, MetricSpec
from .report import build_readout
from .simulator import SimConfig, simulate_experiment


def demo_config(name: str = "checkout-redesign") -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        salt="sim-salt",
        segment_col="segment",
        metrics=[
            MetricSpec(name="revenue_per_user", column="y", role="primary", covariate="x_pre"),
            MetricSpec(name="active_flag", kind="binary", column="active", role="secondary"),
            MetricSpec(name="latency_ms", column="latency", role="guardrail",
                       higher_is_better=False, ni_margin=3.0),
        ],
    )


def demo_frame(seed: int = 7, n: int = 40_000, srm_drop: float = 0.0) -> pd.DataFrame:
    """Simulated data: +0.25 true lift on revenue/user, heterogeneous by segment."""
    df = simulate_experiment(SimConfig(
        n=n, metric="normal", effect=0.25, base=10.0, sd=4.0, rho=0.5, seed=seed,
        segment_effects={"A": 0.15, "C": -0.25}, srm_drop=srm_drop))
    import numpy as np

    rng = np.random.default_rng(seed + 1)
    df["active"] = (rng.random(len(df)) < 0.30 + 0.004 * (df["variant"] == "treatment")).astype(int)
    df["latency"] = rng.normal(200, 25, len(df)) + 0.4 * (df["variant"] == "treatment")
    return df


def main() -> None:
    cfg = demo_config()
    out = build_readout(demo_frame(), cfg)
    print(f"status: {out['status']}  units: {out['n_units']}  SRM p={out['srm']['p_value']:.3f}")
    for m in out["metrics"]:
        e = m["effect"]
        print(f"{m['name']:>18}: {e['estimate']:+.4f} [{e['ci_low']:+.4f}, {e['ci_high']:+.4f}] "
              f"p_adj={m['p_adjusted']:.4f} var_red={m['variance_reduction']:.1%}")
    for g in out["guardrails"]:
        print(f"guardrail {g['name']}: non_inferior={g['non_inferior']} (p={g['p_value']:.4f})")


if __name__ == "__main__":
    main()
