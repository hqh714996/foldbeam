from __future__ import annotations

import numpy as np

from foldbeam_r.graph import Predicate


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


def split_gain(X: np.ndarray, z: np.ndarray, rows: np.ndarray, predicate: Predicate) -> tuple[float, int, int]:
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size == 0:
        return 0.0, 0, 0
    values = X[rows, int(predicate.feature)]
    left = rows[values <= float(predicate.threshold)]
    right = rows[values > float(predicate.threshold)]
    parent_z = z[rows]
    left_z = z[left]
    right_z = z[right]
    gain = _sse(parent_z) - _sse(left_z) - _sse(right_z)
    return float(gain), int(left.size), int(right.size)


def top_new_predicates(
    X: np.ndarray,
    z: np.ndarray,
    rows: np.ndarray,
    pool: list[Predicate],
    m_new: int,
    min_parent: int,
    min_child: int,
) -> list[tuple[Predicate, float]]:
    if len(rows) < int(min_parent):
        return []
    scored: list[tuple[Predicate, float]] = []
    for predicate in pool:
        gain, left_n, right_n = split_gain(X, z, rows, predicate)
        if left_n < int(min_child) or right_n < int(min_child):
            continue
        if gain <= 0.0:
            continue
        scored.append((predicate, gain / max(len(rows), 1)))
    scored.sort(key=lambda item: (-item[1], item[0].feature, item[0].threshold))
    return scored[: int(m_new)]


def build_probes(X: np.ndarray, z: np.ndarray, pool: list[Predicate], n_probes: int) -> list[Predicate]:
    scored: list[tuple[Predicate, float]] = []
    rows = np.arange(X.shape[0], dtype=np.int64)
    for predicate in pool:
        gain, left_n, right_n = split_gain(X, z, rows, predicate)
        if left_n == 0 or right_n == 0:
            continue
        scored.append((predicate, gain / max(len(rows), 1)))
    scored.sort(key=lambda item: (-item[1], item[0].feature, item[0].threshold))
    return [item[0] for item in scored[: int(n_probes)]]


def _sse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    mean = float(np.mean(values))
    return float(np.sum((values - mean) ** 2))
