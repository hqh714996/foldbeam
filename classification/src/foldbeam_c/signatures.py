from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from foldbeam_c.graph import Predicate


@dataclass(frozen=True)
class Summary:
    n: int
    rate: float
    se: float | None


@dataclass(frozen=True)
class Signature:
    base: Summary
    probes: dict[tuple[int, str], Summary]


def make_signature(X: np.ndarray, y: np.ndarray, rows: np.ndarray, probes: list[Predicate], alpha: float = 1.0) -> Signature:
    rows = np.asarray(rows, dtype=np.int64)
    base = _summary(y[rows], alpha)
    probe_rows: dict[tuple[int, str], Summary] = {}
    for i, predicate in enumerate(probes):
        values = X[rows, int(predicate.feature)] if rows.size else np.asarray([], dtype=np.float64)
        left = rows[values <= float(predicate.threshold)] if rows.size else rows
        right = rows[values > float(predicate.threshold)] if rows.size else rows
        probe_rows[(i, "left")] = _summary(y[left], alpha)
        probe_rows[(i, "right")] = _summary(y[right], alpha)
    return Signature(base=base, probes=probe_rows)


def c_score(sig_h: Signature, sig_v: Signature, offset: float = 2.0) -> float:
    d0 = _delta(sig_h.base.rate, sig_v.base.rate)
    numerator = 0.0
    denominator = 0.0
    for key in sorted(set(sig_h.probes).intersection(sig_v.probes)):
        a = sig_h.probes[key]
        b = sig_v.probes[key]
        support = float(min(a.n, b.n))
        if support <= 0.0:
            continue
        weight = support / (support + float(offset))
        numerator += weight * _delta(a.rate, b.rate)
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
        lower_gap = abs(a.rate - b.rate) - 2.0 * (a.se + b.se)
        worst = max(worst, float(lower_gap))
    if worst == float("-inf"):
        worst = 0.0
    return bool(worst > float(tau)), float(worst)


def _summary(y: np.ndarray, alpha: float = 1.0) -> Summary:
    y = np.asarray(y)
    n = int(y.size)
    if n == 0:
        return Summary(n=0, rate=0.5, se=None)
    # Laplace-smoothed positive rate: (pos + alpha*global_prior) / (n + alpha)
    # For signature, use simple Laplace with prior 0.5 (classification version of DESIGN_CN_CLASSIFY §6.2)
    pos = float(np.sum(y))
    rate = float((pos + float(alpha) * 0.5) / (n + float(alpha)))
    if n < 2:
        return Summary(n=n, rate=rate, se=None)
    # binomial SE on smoothed rate
    se = float(np.sqrt(rate * (1.0 - rate) / n))
    return Summary(n=n, rate=rate, se=se)


def _delta(a: float, b: float) -> float:
    return float(1.0 - np.exp(-abs(float(a) - float(b))))
