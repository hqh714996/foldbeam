from __future__ import annotations

import numpy as np

from foldbeam_r.graph import Graph


def regression_metrics(graph: Graph, X: np.ndarray, y: np.ndarray, y_mean: float, y_std: float) -> dict[str, float]:
    pred_z = graph.predict_z(X)
    pred_y = y_mean + y_std * pred_z
    errors = pred_y - y
    mse = float(np.mean(errors ** 2))
    mae = float(np.mean(np.abs(errors)))
    denom = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = float("nan") if denom <= 0.0 else float(1.0 - np.sum(errors ** 2) / denom)
    return {
        "raw_mse": mse,
        "mae": mae,
        "r2": r2,
        "scaled_mse": float(mse / (float(y_std) ** 2)),
    }
