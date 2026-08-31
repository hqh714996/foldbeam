from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_factor_map(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(
                f"Missing factor map: {path}. Run compile_maps_regression.py first."
            )
        return {"factors": [], "features": {}}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {"factors": [], "features": {}}


def affinity_key(
    paths_h: list[list[tuple[int, str, float]]],
    paths_v: list[list[tuple[int, str, float]]],
    factor_map: dict[str, Any],
    n_features: int,
    feature_names: list[str] | None = None,
) -> tuple[float, float, float]:
    best = (-1.0, -1.0, float("-inf"))
    for path_h in paths_h:
        phi_h, psi_h = _path_vectors(path_h, factor_map, n_features, feature_names)
        for path_v in paths_v:
            phi_v, psi_v = _path_vectors(path_v, factor_map, n_features, feature_names)
            key = (_cos(phi_h, phi_v), _cos(psi_h, psi_v), -float(np.linalg.norm(phi_h - phi_v)))
            if key > best:
                best = key
    return best


def _path_vectors(
    path: list[tuple[int, str, float]],
    factor_map: dict[str, Any],
    n_features: int,
    feature_names: list[str] | None,
) -> tuple[np.ndarray, np.ndarray]:
    # If no frozen factor map is present yet, fall back to deterministic feature-space
    # path vectors. This keeps the regression pipeline runnable before LLM compilation.
    features_cfg = factor_map.get("features") if isinstance(factor_map, dict) else None
    if not features_cfg:
        phi = np.zeros(n_features, dtype=np.float64)
        psi = np.zeros(n_features, dtype=np.float64)
        for feature, op, _threshold in path:
            direction = 1.0 if op == ">" else -1.0
            if 0 <= int(feature) < n_features:
                phi[int(feature)] += direction
                psi[int(feature)] = 1.0
        return phi, psi
    # Minimal schema support: feature id string -> factor index, strength, direction.
    factors = factor_map.get("factors") or []
    dim = max(len(factors), 1)
    factor_index = {
        str(item.get("factor_id", f"F{i + 1}")): i
        for i, item in enumerate(factors)
        if isinstance(item, dict)
    }
    phi = np.zeros(dim, dtype=np.float64)
    psi = np.zeros(dim, dtype=np.float64)
    for feature, op, _threshold in path:
        feature_name = None
        if feature_names is not None and 0 <= int(feature) < len(feature_names):
            feature_name = feature_names[int(feature)]
        entry = (
            features_cfg.get(str(feature_name))
            if feature_name is not None
            else None
        ) or features_cfg.get(str(feature)) or features_cfg.get(f"f{feature}")
        if not isinstance(entry, dict):
            continue
        if "factor" in entry:
            factor = int(entry.get("factor", 0))
        else:
            factor = int(factor_index.get(str(entry.get("factor_id", "F1")), 0))
        if factor < 0 or factor >= dim:
            continue
        strength = float(entry.get("strength", 1.0))
        direction_name = str(entry.get("direction", "nonmonotone"))
        semantic_direction = 0.0
        if direction_name == "higher_target":
            semantic_direction = 1.0
        elif direction_name == "lower_target":
            semantic_direction = -1.0
        path_direction = 1.0 if op == ">" else -1.0
        phi[factor] += strength * semantic_direction * path_direction
        psi[factor] = max(psi[factor], abs(strength))
    return phi, psi


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)
