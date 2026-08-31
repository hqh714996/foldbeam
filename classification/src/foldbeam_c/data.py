from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from foldbeam_c.config import Config, ConfigError, json_dump, json_load, stable_hash


@dataclass(frozen=True)
class RoleArrays:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    feature_semantics: list[str]
    class_names: list[str]
    positive_label: str


def _frame_digest(frame: pd.DataFrame) -> str:
    canonical = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    return hashlib.sha256(canonical).hexdigest()[:32]


def _resolve_target(frame: pd.DataFrame, requested: str, dataset_id: str) -> str:
    if requested in frame.columns:
        return requested
    matches = [c for c in frame.columns if str(c).lower() == requested.lower()]
    if len(matches) == 1:
        return str(matches[0])
    raise ConfigError(f"{dataset_id}: target column {requested!r} absent. Available: {[str(c) for c in frame.columns]}")


def _split_indices(y: np.ndarray, seed: int, config: Config, stratify: bool = True) -> dict[str, np.ndarray]:
    data_cfg = config.section("data")
    test_fraction = float(data_cfg["test_fraction"])
    val_fraction_of_remaining = float(data_cfg["validation_fraction_of_remaining"])
    all_indices = np.arange(len(y))
    strat = y if stratify and len(np.unique(y)) <= 20 else None
    try:
        remaining_idx, test_idx = train_test_split(all_indices, test_size=test_fraction, random_state=0, stratify=strat)
    except ValueError:
        remaining_idx, test_idx = train_test_split(all_indices, test_size=test_fraction, random_state=0)
    strat_r = y[remaining_idx] if strat is not None else None
    try:
        train_idx, val_idx = train_test_split(remaining_idx, test_size=val_fraction_of_remaining, random_state=int(seed), stratify=strat_r)
    except ValueError:
        train_idx, val_idx = train_test_split(remaining_idx, test_size=val_fraction_of_remaining, random_state=int(seed))
    return {"train": np.sort(train_idx), "val": np.sort(val_idx), "test": np.sort(test_idx)}


def _encode_features_response_sorted(
    features: pd.DataFrame, y_train: np.ndarray, train_idx: np.ndarray
) -> tuple[pd.DataFrame, dict[str, dict[str, int]], dict[str, str]]:
    encoded = pd.DataFrame(index=features.index)
    codebooks: dict[str, dict[str, int]] = {}
    dtypes: dict[str, str] = {}
    for column in features.columns:
        series = features[column].replace({"?": np.nan, "*": np.nan, "NA": np.nan, "N/A": np.nan})
        is_categorical = str(series.dtype) in {"category", "object", "bool", "string"}
        if is_categorical:
            train_series = series.iloc[train_idx]
            mask_valid = train_series.notna()
            train_values_valid = train_series[mask_valid].astype(str)
            if len(train_values_valid) > 0:
                y_tr_valid = y_train[train_idx][mask_valid.to_numpy()]
                cat_means: dict[str, float] = {}
                for cat in train_values_valid.unique():
                    cat_y = y_tr_valid[train_values_valid == cat]
                    cat_means[cat] = float(np.mean(cat_y)) if len(cat_y) else 0.0
                levels = sorted(cat_means, key=lambda c: cat_means[c])
            else:
                levels = []
            mapping = {level: i for i, level in enumerate(levels)}
            codebooks[str(column)] = mapping
            dtypes[str(column)] = "categorical"
            values = series.astype(str).map(mapping)
            values = values.astype("float64")
            values[series.isna()] = -1.0
            values[values.isna()] = -2.0
            encoded[str(column)] = values.to_numpy(dtype=np.float64)
        else:
            dtypes[str(column)] = "numeric"
            encoded[str(column)] = pd.to_numeric(series, errors="coerce").astype(np.float64)
    return encoded, codebooks, dtypes


def _fit_transform_features(encoded: pd.DataFrame, train_idx: np.ndarray, dtypes: dict[str, str]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    matrix = encoded.to_numpy(dtype=np.float64, copy=True)
    keep: list[int] = []
    feature_names: list[str] = []
    params: dict[str, Any] = {"columns": {}, "dropped_zero_variance": []}
    for position, column in enumerate(encoded.columns):
        values = matrix[:, position]
        train_values = values[train_idx]
        finite = train_values[np.isfinite(train_values)]
        if dtypes[str(column)] == "numeric":
            mean = float(np.mean(finite)) if finite.size else 0.0
            values[~np.isfinite(values)] = mean
            train_filled = values[train_idx]
            std = float(np.sqrt(np.mean((train_filled - float(np.mean(train_filled))) ** 2)))
            if std <= 0.0:
                params["dropped_zero_variance"].append(str(column))
                continue
            center = float(np.mean(train_filled))
            values = (values - center) / std
            params["columns"][str(column)] = {"dtype": "numeric", "mean": center, "std": std, "impute": mean}
        else:
            params["columns"][str(column)] = {"dtype": "categorical"}
        matrix[:, position] = values
        keep.append(position)
        feature_names.append(str(column))
    if not keep:
        raise ConfigError("All features were dropped as zero variance.")
    return matrix[:, keep].astype(np.float64), feature_names, params


def _feature_semantics(dataset_id: str, names: list[str], config: Config) -> list[str]:
    semantics_path = config.path("semantics_file")
    if not semantics_path.is_file():
        return [name for name in names]
    import yaml

    with semantics_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    entry = (payload.get("datasets") or {}).get(dataset_id) or {}
    descriptions = entry.get("features") or {}
    return [str(descriptions.get(name, name)) for name in names]


def prepare_datasets(config: Config) -> list[dict[str, Any]]:
    config.ensure_output_directories()
    processed = config.path("processed_dir")
    index_path = processed / "index.json"
    manifest_path = processed / "manifest.json"
    configured_ids = set(config.dataset_ids)
    existing_index = json_load(index_path) if index_path.is_file() else []
    existing_manifest = json_load(manifest_path) if manifest_path.is_file() else {}
    if not isinstance(existing_index, list) or not isinstance(existing_manifest, dict):
        raise ConfigError("Existing processed index or manifest has an invalid format.")
    index: list[dict[str, Any]] = [r for r in existing_index if r.get("dataset_id") not in configured_ids]
    summaries: list[dict[str, Any]] = [s for s in existing_manifest.get("datasets", []) if s.get("dataset_id") not in configured_ids]
    for dataset_id in config.dataset_ids:
        spec = config.dataset(dataset_id)
        if spec.get("source") != "openml":
            raise ConfigError(f"Unsupported source for {dataset_id}: {spec.get('source')}")
        frame, openml_version = _load_frame(dataset_id, spec, config)
        if len(frame) != int(spec["expected_rows"]):
            raise ConfigError(f"{dataset_id}: expected {spec['expected_rows']} rows, found {len(frame)}")
        target_column = _resolve_target(frame, str(spec["target"]), dataset_id)
        raw_target_series = frame[target_column].astype(str)
        positive_label = str(spec.get("positive_label", raw_target_series.unique()[0]))
        # binary check
        unique_labels = sorted(raw_target_series.unique())
        if len(unique_labels) != 2:
            raise ConfigError(f"{dataset_id}: expected binary target, found {unique_labels}")
        raw_y = (raw_target_series == positive_label).astype(np.int64).to_numpy()
        features = frame.drop(columns=[target_column])
        if features.shape[1] != int(spec["expected_features"]):
            raise ConfigError(f"{dataset_id}: expected {spec['expected_features']} features, found {features.shape[1]}")
        summaries.append({"dataset_id": dataset_id, "openml_id": int(spec["openml_id"]), "openml_version": openml_version, "rows": int(len(frame)), "features_raw": int(features.shape[1]), "target_column": target_column, "positive_label": positive_label, "frame_digest": _frame_digest(frame)})
        stratify_flag = bool(config.section("data").get("stratified", True))
        for split_number, seed in enumerate(config.section("data")["split_seeds"]):
            split_id = f"split_{split_number:02d}"
            roles = _split_indices(raw_y, int(seed), config, stratify=stratify_flag)
            encoded, codebooks, dtypes = _encode_features_response_sorted(features, raw_y, roles["train"])
            X, feature_names, transform_params = _fit_transform_features(encoded, roles["train"], dtypes)
            y_train = raw_y[roles["train"]]
            arrays = {
                "X_train": X[roles["train"]],
                "y_train": raw_y[roles["train"]].astype(np.int64),
                "rows_train": roles["train"],
                "X_val": X[roles["val"]],
                "y_val": raw_y[roles["val"]].astype(np.int64),
                "rows_val": roles["val"],
                "X_test": X[roles["test"]],
                "y_test": raw_y[roles["test"]].astype(np.int64),
                "rows_test": roles["test"],
            }
            npz_name = f"{dataset_id}__{split_id}.npz"
            np.savez_compressed(processed / npz_name, **arrays)
            index.append({
                "dataset_id": dataset_id, "split_id": split_id, "seed": int(seed),
                "task": str(spec.get("task", dataset_id)), "file": npz_name,
                "feature_names": feature_names, "feature_semantics": _feature_semantics(dataset_id, feature_names, config),
                "target_column": target_column, "positive_label": positive_label,
                "class_names": [str(x) for x in sorted(np.unique(raw_y))],
                "n_train": int(len(roles["train"])), "n_val": int(len(roles["val"])), "n_test": int(len(roles["test"])),
                "codebooks": codebooks, "dtypes": dtypes, "transform_params": transform_params,
                "split_digest": stable_hash({key: roles[key].tolist() for key in roles}),
            })
    index.sort(key=lambda r: (str(r["dataset_id"]), str(r["split_id"])))
    summaries.sort(key=lambda s: str(s["dataset_id"]))
    json_dump(index_path, index)
    json_dump(manifest_path, {"config_hash": config.digest(), "datasets": summaries, "records": len(index), "prepare_status": "computed"})
    return index


def _load_frame(dataset_id: str, spec: dict[str, Any], config: Config) -> tuple[pd.DataFrame, Any]:
    raw_path = config.path("raw_dir") / f"{dataset_id}.csv"
    if raw_path.is_file():
        return pd.read_csv(raw_path), "manual_csv"
    root_csv_path = config.project_root / "data" / f"{dataset_id}.csv"
    if root_csv_path.is_file():
        return pd.read_csv(root_csv_path), "manual_root_csv"
    try:
        bunch = fetch_openml(data_id=int(spec["openml_id"]), as_frame=True, parser="auto", data_home=str(config.path("openml_cache")), n_retries=5, delay=5.0)
        return bunch.frame.copy(), bunch.details.get("version")
    except Exception as exc:
        raise ConfigError(f"{dataset_id}: OpenML download failed and no manual CSV at {raw_path}. Original error: {type(exc).__name__}: {exc}") from exc


def load_index(config: Config) -> list[dict[str, Any]]:
    path = config.path("processed_dir") / "index.json"
    if not path.is_file():
        raise ConfigError(f"Processed index is missing: {path}. Run prepare first.")
    return json_load(path)


def load_arrays(config: Config, record: dict[str, Any]) -> RoleArrays:
    path = config.path("processed_dir") / str(record["file"])
    with np.load(path) as payload:
        return RoleArrays(
            X_train=payload["X_train"], y_train=payload["y_train"],
            X_val=payload["X_val"], y_val=payload["y_val"],
            X_test=payload["X_test"], y_test=payload["y_test"],
            feature_names=list(record["feature_names"]), feature_semantics=list(record["feature_semantics"]),
            class_names=list(record.get("class_names", ["0", "1"])), positive_label=str(record.get("positive_label", "1")),
        )
