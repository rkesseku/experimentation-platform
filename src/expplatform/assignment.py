"""Deterministic hash-based assignment, layers, and the sample-ratio-mismatch check."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

_BUCKETS = 10_000


def hash_unit(unit_id: object, salt: str) -> float:
    """Map ``(salt, unit_id)`` to a uniform float in ``[0, 1)`` via SHA-256."""
    digest = hashlib.sha256(f"{salt}:{unit_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _cum_edges(weights: Mapping[str, float]) -> tuple[list[str], np.ndarray]:
    names = list(weights)
    w = np.array([weights[k] for k in names], float)
    if np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be non-negative with a positive sum")
    return names, np.cumsum(w / w.sum())


def assign(unit_id: object, salt: str, weights: Mapping[str, float]) -> str:
    """Assign a unit to a variant; stable across calls, processes and machines."""
    names, edges = _cum_edges(weights)
    u = hash_unit(unit_id, salt)
    return names[min(int(np.searchsorted(edges, u, side="right")), len(names) - 1)]


def assign_many(unit_ids: Iterable[object], salt: str, weights: Mapping[str, float]) -> np.ndarray:
    """Assignment for many units (returns an object array of variant names)."""
    names, edges = _cum_edges(weights)
    u = np.fromiter((hash_unit(x, salt) for x in unit_ids), float)
    idx = np.minimum(np.searchsorted(edges, u, side="right"), len(names) - 1)
    return np.array(names, dtype=object)[idx]


@dataclass
class Layer:
    """A mutually-exclusive traffic layer.

    Units are hashed into 10,000 buckets with the *layer* salt; each experiment owns a
    contiguous bucket range sized by its traffic fraction, so experiments in one layer never
    overlap. Within an experiment the variant is drawn with a *different* hash (layer salt +
    experiment name), so the variant is independent of which bucket range was hit. Experiments
    in different layers use different salts and are therefore independent of each other.
    """

    name: str
    salt: str
    allocations: dict[str, tuple[int, int]] | None = None

    def allocate(self, traffic: Mapping[str, float]) -> None:
        if sum(traffic.values()) > 1 + 1e-9:
            raise ValueError("traffic fractions in a layer must sum to <= 1")
        start, out = 0, {}
        for exp, frac in traffic.items():
            size = round(frac * _BUCKETS)
            out[exp] = (start, start + size)
            start += size
        self.allocations = out

    def bucket(self, unit_id: object) -> int:
        return int(hash_unit(unit_id, f"{self.salt}:bucket") * _BUCKETS)

    def experiment_for(self, unit_id: object) -> str | None:
        b = self.bucket(unit_id)
        for exp, (lo, hi) in (self.allocations or {}).items():
            if lo <= b < hi:
                return exp
        return None

    def variant_for(
        self, unit_id: object, experiment: str, weights: Mapping[str, float]
    ) -> str | None:
        if self.experiment_for(unit_id) != experiment:
            return None
        return assign(unit_id, f"{self.salt}:{experiment}", weights)


def srm_test(
    counts: Mapping[str, int], weights: Mapping[str, float], threshold: float = 0.001
) -> dict:
    """Chi-square goodness-of-fit test of observed arm counts against the intended split.

    A very small p-value (conventionally < 0.001) indicates a sample-ratio mismatch: the
    experiment infrastructure, not the treatment, is distorting who is observed.
    """
    names = list(weights)
    obs = np.array([counts.get(k, 0) for k in names], float)
    w = np.array([weights[k] for k in names], float)
    exp = obs.sum() * w / w.sum()
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    p = float(stats.chi2.sf(chi2, len(names) - 1))
    return {
        "chi2": chi2,
        "p_value": p,
        "observed": {k: int(v) for k, v in zip(names, obs)},
        "expected": {k: float(v) for k, v in zip(names, exp)},
        "threshold": threshold,
        "mismatch": bool(p < threshold),
    }


def srm_from_frame(
    df: pd.DataFrame,
    weights: Mapping[str, float],
    variant_col: str = "variant",
    threshold: float = 0.001,
) -> dict:
    return srm_test(df[variant_col].value_counts().to_dict(), weights, threshold)
