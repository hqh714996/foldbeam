from __future__ import annotations

import numpy as np

from foldbeam_c.graph import Predicate


def build_pool(X: np.ndarray, quantiles: list[float]) -> list[Predicate]:
    predicates: list[Predicate] = []
    seen: set[tuple[int, float]] = set()
    for feature in range(X.shape[1]):
        values = np.asarray(X[:, feature], dtype=np.float64)
        for q in quantiles:
            threshold = float(np.quantile(values, float(q)))
            key = (feature, round(threshold, 12))
            if key in seen:
                continue
            seen.add(key)
            predicates.append(Predicate(feature, threshold))
    return sorted(predicates, key=lambda p: (p.feature, p.threshold))


def _gini(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    p = float(np.mean(y))
    return 2.0 * p * (1.0 - p)


def _entropy(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    p = float(np.mean(y))
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)


def _impurity(y: np.ndarray, metric: str) -> float:
    if metric == "entropy":
        return _entropy(y)
    return _gini(y)


def split_gain(
    X: np.ndarray, y: np.ndarray, rows: np.ndarray, predicate: Predicate, metric: str = "gini"
) -> tuple[float, int, int]:
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size == 0:
        return 0.0, 0, 0
    values = X[rows, int(predicate.feature)]
    left = rows[values <= float(predicate.threshold)]
    right = rows[values > float(predicate.threshold)]
    n = len(rows)
    parent_imp = _impurity(y[rows], metric)
    left_imp = _impurity(y[left], metric) if left.size else 0.0
    right_imp = _impurity(y[right], metric) if right.size else 0.0
    gain = parent_imp - (len(left) / n * left_imp + len(right) / n * right_imp)
    return float(gain), int(left.size), int(right.size)


def top_new_predicates(
    X: np.ndarray, y: np.ndarray, rows: np.ndarray, pool: list[Predicate],
    m_new: int, min_parent: int, min_child: int, metric: str = "gini",
) -> list[tuple[Predicate, float]]:
    if len(rows) < int(min_parent):
        return []
    scored: list[tuple[Predicate, float]] = []
    for predicate in pool:
        gain, left_n, right_n = split_gain(X, y, rows, predicate, metric)
        if left_n < int(min_child) or right_n < int(min_child):
            continue
        if gain <= 0.0:
            continue
        scored.append((predicate, gain))
    scored.sort(key=lambda item: (-item[1], item[0].feature, item[0].threshold))
    return scored[: int(m_new)]


def build_probes(X: np.ndarray, y: np.ndarray, pool: list[Predicate], n_probes: int, metric: str = "gini") -> list[Predicate]:
    scored: list[tuple[Predicate, float]] = []
    rows = np.arange(X.shape[0], dtype=np.int64)
    for predicate in pool:
        gain, left_n, right_n = split_gain(X, y, rows, predicate, metric)
        if left_n == 0 or right_n == 0:
            continue
        scored.append((predicate, gain))
    scored.sort(key=lambda item: (-item[1], item[0].feature, item[0].threshold))
    return [item[0] for item in scored[: int(n_probes)]]
