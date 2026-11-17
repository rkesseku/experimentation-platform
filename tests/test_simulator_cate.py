import numpy as np
import pytest
from scipy import stats

from expplatform.analysis import welch_ttest
from expplatform.cate import segment_effects, t_learner_cate
from expplatform.simulator import SimConfig, simulate_experiment


def _eff(df, col="y"):
    t, c = df[df.variant == "treatment"][col], df[df.variant == "control"][col]
    return welch_ttest(t.to_numpy(), c.to_numpy())


def test_simulator_recovers_known_effect_and_is_reproducible():
    df = simulate_experiment(SimConfig(n=60_000, effect=0.5, seed=3))
    r = _eff(df)
    assert abs(r.estimate - df.attrs["truth"]["ate"]) < 4 * r.se
    again = simulate_experiment(SimConfig(n=60_000, effect=0.5, seed=3))
    assert df.equals(again)


@pytest.mark.parametrize("metric", ["lognormal", "binary", "heavy_tail", "ratio"])
def test_metric_families_recover_truth(metric):
    cfg = SimConfig(n=60_000, metric=metric, seed=4, use_hash=False,
                    effect={"lognormal": 0.1, "binary": 0.02, "heavy_tail": 0.4, "ratio": 0.1}[metric],
                    base=0.2 if metric == "binary" else 10.0)
    df = simulate_experiment(cfg)
    r = _eff(df)
    truth = df.attrs["truth"]["ate"]
    assert abs(r.estimate - truth) < 4 * r.se


def test_heavy_tail_has_heavier_tails_than_normal():
    a = simulate_experiment(SimConfig(n=50_000, metric="normal", seed=1, rho=0.0))
    b = simulate_experiment(SimConfig(n=50_000, metric="heavy_tail", seed=1, rho=0.0))
    assert stats.kurtosis(b.y) > stats.kurtosis(a.y) + 1.0


def test_novelty_effect_decays_over_time():
    df = simulate_experiment(SimConfig(n=80_000, effect=0.3, novelty_boost=3.0,
                                       novelty_decay_days=2.0, days=14, seed=5))
    early, late = df[df.ts < 2], df[df.ts > 10]
    assert _eff(early).estimate > _eff(late).estimate + 0.5


def test_srm_bug_changes_arm_sizes_and_biases_naive_estimate():
    df = simulate_experiment(SimConfig(n=60_000, effect=0.0, srm_drop=0.05, seed=6))
    counts = df.variant.value_counts()
    assert counts["treatment"] < 0.97 * counts["control"]
    # dropping the heaviest treated users makes a null treatment look harmful
    assert _eff(df).estimate < -0.1


def test_segment_correction_and_heterogeneity_test():
    het = simulate_experiment(SimConfig(n=40_000, effect=0.0, seed=7, use_hash=False,
                                        segment_effects={"A": 0.6, "C": -0.6}))
    out = segment_effects(het, "segment", correction="holm")
    sig = {s["segment"]: s["significant"] for s in out["segments"]}
    assert sig["A"] and sig["C"] and not sig["B"]
    assert out["q_p_value"] < 1e-6
    null = simulate_experiment(SimConfig(n=40_000, effect=0.2, seed=8, use_hash=False))
    out0 = segment_effects(null, "segment")
    assert out0["q_p_value"] > 0.01


def test_t_learner_recovers_heterogeneity_via_out_of_fold_groups():
    df = simulate_experiment(SimConfig(n=24_000, effect=0.0, hte=1.5, sd=3.0, seed=9,
                                       use_hash=False))
    out = t_learner_cate(df, ["f1", "f2", "f3"], seed=1)
    g = out["groups"]
    assert out["group_corr"] > 0.9
    assert g[-1]["realised_effect"] > g[0]["realised_effect"] + 1.0
    assert g[-1]["ci_low"] > 0 > g[0]["ci_high"]  # top group helped, bottom group harmed


def test_t_learner_econml_backend():
    pytest.importorskip("econml")
    df = simulate_experiment(SimConfig(n=8000, effect=0.0, hte=1.5, sd=3.0, seed=10,
                                       use_hash=False))
    out = t_learner_cate(df, ["f1", "f2", "f3"], backend="econml", n_folds=3, seed=1)
    assert out["group_corr"] > 0.7
    assert np.isfinite(out["cate"]).all()
