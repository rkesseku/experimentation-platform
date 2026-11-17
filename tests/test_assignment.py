import numpy as np
import pytest
from scipy import stats

from expplatform.assignment import (
    Layer,
    assign,
    assign_many,
    hash_unit,
    srm_from_frame,
    srm_test,
)
from expplatform.simulator import SimConfig, simulate_experiment


def test_assignment_is_deterministic_and_salt_dependent():
    w = {"control": 0.5, "treatment": 0.5}
    assert assign("user-42", "s1", w) == assign("user-42", "s1", w)
    assert 0 <= hash_unit("x", "s") < 1
    # same hash value is stable across processes (SHA-256, not Python's randomised hash())
    # pinned value: guards against accidental changes to the hashing scheme (re-bucketing users)
    assert hash_unit("user-42", "s1") == pytest.approx(0.5622871673446819, abs=1e-12)
    a1 = assign_many(range(5000), "s1", w)
    a2 = assign_many(range(5000), "s2", w)
    assert (a1 != a2).mean() > 0.3  # re-salting reshuffles units


def test_split_matches_weights_and_passes_srm():
    w = {"control": 0.9, "treatment": 0.1}
    a = assign_many(range(60_000), "salt", w)
    frac = (a == "treatment").mean()
    assert abs(frac - 0.1) < 0.005
    counts = {k: int((a == k).sum()) for k in w}
    assert not srm_test(counts, w)["mismatch"]


def test_independent_salts_give_independent_assignments():
    w = {"a": 0.5, "b": 0.5}
    x = assign_many(range(40_000), "layer-1", w) == "a"
    y = assign_many(range(40_000), "layer-2", w) == "a"
    table = np.array([[(x & y).sum(), (x & ~y).sum()], [(~x & y).sum(), (~x & ~y).sum()]])
    assert stats.chi2_contingency(table)[1] > 0.001


def test_layer_mutual_exclusion_and_traffic_shares():
    layer = Layer("checkout", "layer-salt")
    layer.allocate({"exp1": 0.2, "exp2": 0.3})
    ids = range(50_000)
    owner = [layer.experiment_for(i) for i in ids]
    shares = {e: owner.count(e) / len(owner) for e in ("exp1", "exp2")}
    assert abs(shares["exp1"] - 0.2) < 0.01 and abs(shares["exp2"] - 0.3) < 0.01
    assert owner.count(None) / len(owner) == pytest.approx(0.5, abs=0.01)
    w = {"control": 0.5, "treatment": 0.5}
    v1 = [layer.variant_for(i, "exp1", w) for i in ids]
    assert all(v is None for v, o in zip(v1, owner) if o != "exp1")
    inside = [v for v in v1 if v is not None]
    assert abs(inside.count("treatment") / len(inside) - 0.5) < 0.02
    with pytest.raises(ValueError):
        layer.allocate({"a": 0.7, "b": 0.5})


def test_srm_detects_planted_imbalance():
    w = {"control": 0.5, "treatment": 0.5}
    assert srm_test({"control": 50_000, "treatment": 48_500}, w)["mismatch"]
    assert not srm_test({"control": 50_000, "treatment": 50_150}, w)["mismatch"]


def test_srm_on_simulated_bug_vs_clean():
    w = {"control": 0.5, "treatment": 0.5}
    clean = simulate_experiment(SimConfig(n=40_000, seed=1))
    bug = simulate_experiment(SimConfig(n=40_000, seed=1, srm_drop=0.05))
    assert not srm_from_frame(clean, w)["mismatch"]
    res = srm_from_frame(bug, w)
    assert res["mismatch"] and res["p_value"] < 1e-6
