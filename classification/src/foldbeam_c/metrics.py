from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score

from foldbeam_c.graph import Graph


def classification_metrics(
    graph: Graph, X: np.ndarray, y: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    y = np.asarray(y, dtype=np.int64)
    proba = graph.predict_proba(X)
    proba = np.clip(proba, 1e-7, 1 - 1e-7)
    pred = (proba >= float(threshold)).astype(np.int64)
    bal_acc = float(balanced_accuracy_score(y, pred))
    acc = float(accuracy_score(y, pred))
    ll = float(log_loss(y, proba))
    try:
        auc = float(roc_auc_score(y, proba)) if len(np.unique(y)) == 2 else float("nan")
    except ValueError:
        auc = float("nan")
    return {"balanced_accuracy": bal_acc, "accuracy": acc, "logloss": ll, "auc": auc, "threshold": float(threshold)}


def logloss_for_proba(y: np.ndarray, proba: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int64)
    proba = np.clip(np.asarray(proba, dtype=np.float64), 1e-7, 1 - 1e-7)
    return float(log_loss(y, proba))


def balanced_accuracy_for_threshold(y: np.ndarray, proba: np.ndarray, threshold: float) -> float:
    y = np.asarray(y, dtype=np.int64)
    pred = (np.asarray(proba, dtype=np.float64) >= float(threshold)).astype(np.int64)
    return float(balanced_accuracy_score(y, pred))
