from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from foldbeam_r.config import Config, ConfigError, json_dump, json_load, stable_hash


@dataclass(frozen=True)
class RoleArrays:
    X_train: np.ndarray
    z_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    z_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    z_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    feature_semantics: list[str]
    y_mean: float
    y_std: float


def _frame_digest(frame: pd.DataFrame) -> str:
    canonical = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    return hashlib.sha256(canonical).hexdigest()[:32]


def _resolve_target(
    frame: pd.DataFrame,
    requested: str,
    dataset_id: str,
    target_names: list[str] | tuple[str, ...] | None = None,
) -> str:
    if requested in frame.columns:
        return requested
    matches = [column for column in frame.columns if str(column).lower() == requested.lower()]
    if len(matches) == 1:
        return str(matches[0])
    for target_name in target_names or []:
        if target_name in frame.columns:
            return str(target_name)
        matches = [column for column in frame.columns if str(column).lower() == str(target_name).lower()]
        if len(matches) == 1:
            return str(matches[0])
    raise ConfigError(
        f"{dataset_id}: target column {requested!r} is absent. Available columns: "
        f"{[str(column) for column in frame.columns]}"
    )


def _target_strata(target: np.ndarray, n_strata: int) -> np.ndarray:
    order = np.lexsort((np.arange(len(target)), target))
    strata = np.empty(len(target), dtype=np.int64)
    for rank, row in enumerate(order):
        strata[row] = min(int(np.floor(n_strata * rank / len(target))), n_strata - 1)
    counts = np.bincount(strata, minlength=n_strata)
    if int(counts.min()) < 10:
        raise ConfigError(f"Target strata are too small for stable splitting: {counts.tolist()}")
    return strata


def _split_indices(target: np.ndarray, seed: int, config: Config) -> dict[str, np.ndarray]:
    data_cfg = config.section("data")
    test_fraction = float(data_cfg["test_fraction"])
    val_fraction_of_remaining = float(data_cfg["validation_fraction_of_remaining"])
    all_indices = np.arange(len(target))

    # Match LLEGO's implementation exactly: hold out one fixed test split,
    # then use each run seed only to divide the remaining rows into train/val.
    remaining_idx, test_idx = train_test_split(
        all_indices,
        test_size=test_fraction,
        random_state=0,
    )
    train_idx, val_idx = train_test_split(
        remaining_idx,
        test_size=val_fraction_of_remaining,
        random_state=int(seed),
    )
    return {
        "train": np.sort(train_idx),
        "val": np.sort(val_idx),
        "test": np.sort(test_idx),
    }


def _encode_features(
    features: pd.DataFrame,
    train_idx: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]], dict[str, str]]:
    encoded = pd.DataFrame(index=features.index)
    codebooks: dict[str, dict[str, int]] = {}
    dtypes: dict[str, str] = {}
    for column in features.columns:
        series = features[column].replace({"?": np.nan, "*": np.nan, "NA": np.nan, "N/A": np.nan})
        if str(series.dtype) in {"category", "object", "bool", "string"}:
            train_values = series.iloc[train_idx].dropna().astype(str)
            levels = sorted(train_values.unique())
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


def _fit_transform_features(
    encoded: pd.DataFrame,
    train_idx: np.ndarray,
    dtypes: dict[str, str],
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
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
            params["columns"][str(column)] = {
                "dtype": "numeric",
                "mean": center,
                "std": std,
                "impute": mean,
            }
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
    index: list[dict[str, Any]] = [
        record for record in existing_index if record.get("dataset_id") not in configured_ids
    ]
    summaries: list[dict[str, Any]] = [
        summary
        for summary in existing_manifest.get("datasets", [])
        if summary.get("dataset_id") not in configured_ids
    ]
    for dataset_id in config.dataset_ids:
        spec = config.dataset(dataset_id)
        if spec.get("source") != "openml":
            raise ConfigError(f"Unsupported source for {dataset_id}: {spec.get('source')}")
        frame, openml_version = _load_frame(dataset_id, spec, config)
        if len(frame) != int(spec["expected_rows"]):
            raise ConfigError(
                f"{dataset_id}: expected {spec['expected_rows']} rows, found {len(frame)}"
            )
        target_column = _resolve_target(frame, str(spec["target"]), dataset_id, None)
        raw_target = pd.to_numeric(frame[target_column], errors="coerce").to_numpy(np.float64)
        if not np.isfinite(raw_target).all():
            raise ConfigError(f"{dataset_id}: target has non-finite values.")
        if float(np.max(raw_target) - np.min(raw_target)) <= 0.0:
            raise ConfigError(f"{dataset_id}: target is constant.")
        features = frame.drop(columns=[target_column])
        if features.shape[1] != int(spec["expected_features"]):
            raise ConfigError(
                f"{dataset_id}: expected {spec['expected_features']} features, found {features.shape[1]}"
            )

        summaries.append(
            {
                "dataset_id": dataset_id,
                "openml_id": int(spec["openml_id"]),
                "openml_version": openml_version,
                "rows": int(len(frame)),
                "features_raw": int(features.shape[1]),
                "target_column": target_column,
                "frame_digest": _frame_digest(frame),
            }
        )

        for split_number, seed in enumerate(config.section("data")["split_seeds"]):
            split_id = f"split_{split_number:02d}"
            roles = _split_indices(raw_target, int(seed), config)
            encoded, codebooks, dtypes = _encode_features(features, roles["train"])
            X, feature_names, transform_params = _fit_transform_features(
                encoded,
                roles["train"],
                dtypes,
            )
            y_train = raw_target[roles["train"]]
            y_mean = float(np.mean(y_train))
            y_std = float(np.std(y_train, ddof=1))
            if y_std <= 0.0:
                raise ConfigError(f"{dataset_id}/{split_id}: train target is constant.")
            z = (raw_target - y_mean) / y_std

            arrays = {
                "X_train": X[roles["train"]],
                "z_train": z[roles["train"]].astype(np.float64),
                "y_train": raw_target[roles["train"]].astype(np.float64),
                "rows_train": roles["train"],
                "X_val": X[roles["val"]],
                "z_val": z[roles["val"]].astype(np.float64),
                "y_val": raw_target[roles["val"]].astype(np.float64),
                "rows_val": roles["val"],
                "X_test": X[roles["test"]],
                "z_test": z[roles["test"]].astype(np.float64),
                "y_test": raw_target[roles["test"]].astype(np.float64),
                "rows_test": roles["test"],
            }
            npz_name = f"{dataset_id}__{split_id}.npz"
            np.savez_compressed(processed / npz_name, **arrays)

            index.append(
                {
                    "dataset_id": dataset_id,
                    "split_id": split_id,
                    "seed": int(seed),
                    "task": str(spec.get("task", dataset_id)),
                    "file": npz_name,
                    "feature_names": feature_names,
                    "feature_semantics": _feature_semantics(dataset_id, feature_names, config),
                    "target_column": target_column,
                    "n_train": int(len(roles["train"])),
                    "n_val": int(len(roles["val"])),
                    "n_test": int(len(roles["test"])),
                    "y_mean": y_mean,
                    "y_std": y_std,
                    "codebooks": codebooks,
                    "dtypes": dtypes,
                    "transform_params": transform_params,
                    "split_digest": stable_hash({key: roles[key].tolist() for key in roles}),
                }
            )

    index.sort(key=lambda record: (str(record["dataset_id"]), str(record["split_id"])))
    summaries.sort(key=lambda summary: str(summary["dataset_id"]))
    json_dump(index_path, index)
    json_dump(
        manifest_path,
        {
            "config_hash": config.digest(),
            "datasets": summaries,
            "records": len(index),
            "prepare_status": "computed",
        },
    )
    return index


def _load_frame(dataset_id: str, spec: dict[str, Any], config: Config) -> tuple[pd.DataFrame, Any]:
    raw_path = config.path("raw_dir") / f"{dataset_id}.csv"
    if raw_path.is_file():
        return pd.read_csv(raw_path), "manual_csv"
    root_csv_path = config.project_root / "data" / f"{dataset_id}.csv"
    if root_csv_path.is_file():
        return pd.read_csv(root_csv_path), "manual_root_csv"
    uci_path = config.project_root / "data" / f"{dataset_id}.data"
    if dataset_id == "abalone" and uci_path.is_file():
        columns = [
            "Sex",
            "Length",
            "Diameter",
            "Height",
            "Whole weight",
            "Shucked weight",
            "Viscera weight",
            "Shell weight",
            "Rings",
        ]
        return pd.read_csv(uci_path, header=None, names=columns), "uci_raw_data"
    try:
        bunch = fetch_openml(
            data_id=int(spec["openml_id"]),
            as_frame=True,
            parser="auto",
            data_home=str(config.path("openml_cache")),
            n_retries=5,
            delay=5.0,
        )
        return bunch.frame.copy(), bunch.details.get("version")
    except Exception as exc:
        raise ConfigError(
            f"{dataset_id}: OpenML download failed and no manual CSV was found at {raw_path}. "
            "Place a CSV with the configured target column there, or retry when OpenML is reachable. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc


def load_index(config: Config) -> list[dict[str, Any]]:
    path = config.path("processed_dir") / "index.json"
    if not path.is_file():
        raise ConfigError(f"Processed index is missing: {path}. Run prepare first.")
    return json_load(path)


def load_arrays(config: Config, record: dict[str, Any]) -> RoleArrays:
    path = config.path("processed_dir") / str(record["file"])
    with np.load(path) as payload:
        return RoleArrays(
            X_train=payload["X_train"],
            z_train=payload["z_train"],
            y_train=payload["y_train"],
            X_val=payload["X_val"],
            z_val=payload["z_val"],
            y_val=payload["y_val"],
            X_test=payload["X_test"],
            z_test=payload["z_test"],
            y_test=payload["y_test"],
            feature_names=list(record["feature_names"]),
            feature_semantics=list(record["feature_semantics"]),
            y_mean=float(record["y_mean"]),
            y_std=float(record["y_std"]),
        )
