from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from foldbeam_r.graph import Predicate


@dataclass(frozen=True)
class Summary:
    n: int
    mean: float
    se: float | None


@dataclass(frozen=True)
class Signature:
    base: Summary
    probes: dict[tuple[int, str], Summary]


def make_signature(X: np.ndarray, z: np.ndarray, rows: np.ndarray, probes: list[Predicate]) -> Signature:
    rows = np.asarray(rows, dtype=np.int64)
    base = _summary(z[rows])
    probe_rows: dict[tuple[int, str], Summary] = {}
    for i, predicate in enumerate(probes):
        values = X[rows, int(predicate.feature)] if rows.size else np.asarray([], dtype=np.float64)
        left = rows[values <= float(predicate.threshold)] if rows.size else rows
        right = rows[values > float(predicate.threshold)] if rows.size else rows
        probe_rows[(i, "left")] = _summary(z[left])
        probe_rows[(i, "right")] = _summary(z[right])
    return Signature(base=base, probes=probe_rows)


def c_score(sig_h: Signature, sig_v: Signature, offset: float = 2.0) -> float:
    d0 = _delta(sig_h.base.mean, sig_v.base.mean)
    numerator = 0.0
    denominator = 0.0
    for key in sorted(set(sig_h.probes).intersection(sig_v.probes)):
        a = sig_h.probes[key]
        b = sig_v.probes[key]
        support = float(min(a.n, b.n))
        if support <= 0.0:
            continue
        weight = support / (support + float(offset))
        numerator += weight * _delta(a.mean, b.mean)
        denominator += weight
    d_future = numerator / denominator if denominator > 0.0 else d0
    distance = 0.2 * d0 + 0.8 * d_future
    return float(100.0 * (1.0 - distance))


def incompatible(sig_h: Signature, sig_v: Signature, tau: float) -> tuple[bool, float]:
    worst = float("-inf")
    for key in sorted(set(sig_h.probes).intersection(sig_v.probes)):
        a = sig_h.probes[key]
        b = sig_v.probes[key]
        if a.n < 2 or b.n < 2 or a.se is None or b.se is None:
            continue
        lower_gap = abs(a.mean - b.mean) - 2.0 * (a.se + b.se)
        worst = max(worst, float(lower_gap))
    if worst == float("-inf"):
        worst = 0.0
    return bool(worst > float(tau)), float(worst)


def _summary(values: np.ndarray) -> Summary:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return Summary(n=0, mean=0.0, se=None)
    mean = float(np.mean(values))
    if values.size < 2:
        return Summary(n=int(values.size), mean=mean, se=None)
    se = float(np.std(values, ddof=1) / np.sqrt(values.size))
    return Summary(n=int(values.size), mean=mean, se=se)


def _delta(a: float, b: float) -> float:
    return float(1.0 - np.exp(-abs(float(a) - float(b))))
