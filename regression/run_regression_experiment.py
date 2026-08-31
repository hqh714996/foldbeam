from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from foldbeam_r.beam import refit_final, select_budget
from foldbeam_r.config import Config, json_dump, write_jsonl
from foldbeam_r.data import load_arrays, load_index, prepare_datasets
from foldbeam_r.metrics import regression_metrics
from foldbeam_r.semantic import load_factor_map


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FoldBeam-R regression experiment runner")
    parser.add_argument("stage", choices=["prepare", "foldbeam", "all", "summary"])
    parser.add_argument("--config", default=str(HERE / "config_regression.yaml"))
    parser.add_argument("--limit", type=int, default=None, help="Limit processed records for smoke tests.")
    parser.add_argument("--dataset", action="append", help="Restrict to one or more dataset ids.")
    args = parser.parse_args(argv)

    config = Config.load(args.config)
    if args.dataset:
        config = _filter_config(config, set(args.dataset))
    config.ensure_output_directories()
    if args.stage in {"prepare", "all"}:
        index = prepare_datasets(config)
        print(f"prepared {len(index)} records")
    if args.stage in {"foldbeam", "all"}:
        run_foldbeam(config, args.limit)
    if args.stage in {"summary", "all"}:
        write_summary(config)
    return 0


def _records(config: Config, limit: int | None) -> list[dict]:
    allowed = set(config.dataset_ids)
    records = [record for record in load_index(config) if record["dataset_id"] in allowed]
    return records[:limit] if limit is not None else records


def _filter_config(config: Config, keep: set[str]) -> Config:
    missing = sorted(keep.difference(config.dataset_ids))
    if missing:
        raise ValueError(f"Unknown dataset ids: {missing}")
    payload = dict(config.payload)
    payload["datasets"] = {
        dataset_id: spec
        for dataset_id, spec in config.section("datasets").items()
        if dataset_id in keep
    }
    return Config(source=config.source, payload=payload)


def run_foldbeam(
    config: Config,
    limit: int | None = None,
    *,
    method: str = "FoldBeam-R",
) -> pd.DataFrame:
    rows = []
    for record in _records(config, limit):
        arrays = load_arrays(config, record)
        fmap_path = config.path("factor_maps_dir") / f"{record['dataset_id']}.json"
        factor_map = load_factor_map(fmap_path)
        factor_map_sha256 = hashlib.sha256(fmap_path.read_bytes()).hexdigest()
        result = select_budget(
            arrays.X_train,
            arrays.z_train,
            arrays.X_val,
            arrays.z_val,
            config,
            factor_map,
            arrays.feature_names,
        )
        X_final = np.vstack([arrays.X_train, arrays.X_val])
        z_final = np.concatenate([arrays.z_train, arrays.z_val])
        final_graph = refit_final(result.graph, X_final, z_final, config, result.terminal_lambda)
        metrics = regression_metrics(final_graph, arrays.X_test, arrays.y_test, arrays.y_mean, arrays.y_std)
        row = _base_row(record, method)
        row.update(
            {
                "val_mse_z": result.val_mse_z,
                "test_mse": metrics["scaled_mse"],
                "test_raw_mse": metrics["raw_mse"],
                "test_mae": metrics["mae"],
                "test_r2": metrics["r2"],
                "test_scaled_mse": metrics["scaled_mse"],
                "budget": result.budget,
                "terminal_lambda": result.terminal_lambda,
                "reachable": final_graph.reachable_count(),
                "multi_parent": final_graph.multi_parent_count(),
                "graph_evals": result.graph_evals,
                "refine_evals": result.refine_evals,
                "refine_moves": result.refine_moves,
                "wall_seconds": result.wall_seconds,
                "graph_hash": final_graph.canonical_hash(),
                "factor_map_sha256": factor_map_sha256,
                "llm_factor_map": True,
                "llm_factor_features": False,
            }
        )
        rows.append(row)
        log_path = config.path("logs_dir") / method / f"{record['dataset_id']}__{record['split_id']}.jsonl"
        write_jsonl(
            log_path,
            [
                *result.lineage,
                {
                    "event": "final",
                    "dataset_id": record["dataset_id"],
                    "split_id": record["split_id"],
                    "budget": result.budget,
                    "terminal_lambda": result.terminal_lambda,
                    "champion_J": result.val_mse_z,
                    "factor_map_sha256": factor_map_sha256,
                    "llm_factor_map": True,
                    "llm_factor_features": False,
                    "test_mse": metrics["scaled_mse"],
                    "test_raw_mse": metrics["raw_mse"],
                    "test_mae": metrics["mae"],
                    "test_r2": metrics["r2"],
                    "test_scaled_mse": metrics["scaled_mse"],
                    "reachable": final_graph.reachable_count(),
                    "multi_parent": final_graph.multi_parent_count(),
                    "graph_evals": result.graph_evals,
                    "refine_evals": result.refine_evals,
                    "refine_moves": result.refine_moves,
                    "wall_seconds": result.wall_seconds,
                },
            ],
        )
        graph_path = config.path("results_dir") / "graphs" / f"{record['dataset_id']}__{record['split_id']}.json"
        json_dump(graph_path, final_graph.to_dict())
        print(
            f"foldbeam {record['dataset_id']} {record['split_id']} "
            f"scaled_test_mse={metrics['scaled_mse']:.6g}"
        )
    return _write_results(config, "foldbeam_results.parquet", rows)


def write_summary(config: Config) -> None:
    frames = []
    for name in ("foldbeam_results.parquet",):
        path = config.path("results_dir") / name
        if path.is_file():
            frames.append(pd.read_parquet(path))
    if not frames:
        print("no result files found")
        return
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[frame["dataset_id"].isin(config.dataset_ids)].copy()
    if frame.empty:
        print("no result files found for the requested datasets")
        return
    split_columns = [
        "method",
        "dataset_id",
        "split_id",
        "seed",
        "test_mse",
        "test_raw_mse",
        "test_mae",
        "test_r2",
        "budget",
        "terminal_lambda",
        "reachable",
        "multi_parent",
        "graph_evals",
        "refine_evals",
        "refine_moves",
        "wall_seconds",
        "factor_map_sha256",
        "llm_factor_map",
        "llm_factor_features",
    ]
    split_columns = [column for column in split_columns if column in frame.columns]
    split_report = frame[split_columns].sort_values(["method", "dataset_id", "split_id"])
    dataset_summary = frame.groupby(["method", "dataset_id"])["test_mse"].agg(["mean", "std"]).reset_index()
    dataset_summary["mean_plus_minus_std"] = dataset_summary.apply(
        lambda row: f"{row['mean']:.6f} +/- {row['std']:.6f}", axis=1
    )
    pivot = dataset_summary.pivot(index="method", columns="dataset_id", values="mean_plus_minus_std")
    ranks = frame.copy()
    ranks["rank"] = ranks.groupby(["dataset_id", "split_id"])["test_mse"].rank(method="average")
    avg_rank = ranks.groupby("method")["rank"].mean().rename("avg_rank")
    report = pivot.join(avg_rank).reset_index()
    out_dir = config.path("reports_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(split_report, out_dir / "split_results.csv")
    _write_csv(dataset_summary, out_dir / "dataset_summary.csv")
    _write_csv(report, out_dir / "main_table.csv")
    print("\nPer-split train-std scaled test MSE")
    print(split_report[["method", "dataset_id", "split_id", "test_mse"]].to_string(index=False))
    print("\nDataset summary")
    print(report.to_string(index=False))


def _base_row(record: dict, method: str) -> dict:
    return {
        "method": method,
        "dataset_id": record["dataset_id"],
        "split_id": record["split_id"],
        "seed": int(record["seed"]),
    }


def _write_results(config: Config, name: str, rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    path = config.path("results_dir") / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and not frame.empty:
        previous = pd.read_parquet(path)
        keys = ["method", "dataset_id", "split_id"]
        incoming = set(map(tuple, frame[keys].itertuples(index=False, name=None)))
        previous = previous.loc[
            ~previous[keys].apply(tuple, axis=1).isin(incoming)
        ]
        frame = pd.concat([previous, frame], ignore_index=True, sort=False)
    frame = frame.sort_values(["method", "dataset_id", "split_id"]).reset_index(drop=True)
    frame.to_parquet(path, index=False)
    return frame


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    try:
        frame.to_csv(path, index=False)
        return path
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        alternate = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
        frame.to_csv(alternate, index=False)
        print(f"report target was locked; wrote {alternate}")
        return alternate


if __name__ == "__main__":
    raise SystemExit(main())
