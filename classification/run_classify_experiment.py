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

from foldbeam_c.beam import refit_final, select_budget
from foldbeam_c.config import Config, json_dump, write_jsonl
from foldbeam_c.data import load_arrays, load_index, prepare_datasets
from foldbeam_c.metrics import classification_metrics
from foldbeam_c.semantic import load_factor_map


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FoldBeam-C classification experiment runner")
    parser.add_argument("stage", choices=["prepare", "foldbeam", "all", "summary"])
    parser.add_argument("--config", default=str(HERE / "config_classify.yaml"))
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
    payload["datasets"] = {k: v for k, v in config.section("datasets").items() if k in keep}
    return Config(source=config.source, payload=payload)


def run_foldbeam(config: Config, limit: int | None = None) -> pd.DataFrame:
    rows = []
    for record in _records(config, limit):
        arrays = load_arrays(config, record)
        fmap_path = config.path("factor_maps_dir") / f"{record['dataset_id']}.json"
        factor_map = load_factor_map(fmap_path)
        factor_map_sha256 = hashlib.sha256(fmap_path.read_bytes()).hexdigest()
        result = select_budget(arrays.X_train, arrays.y_train, arrays.X_val, arrays.y_val, config, factor_map, arrays.feature_names)
        X_final = np.vstack([arrays.X_train, arrays.X_val])
        y_final = np.concatenate([arrays.y_train, arrays.y_val])
        final_graph = refit_final(result.graph, X_final, y_final, config, result.laplace_alpha)
        metrics = classification_metrics(final_graph, arrays.X_test, arrays.y_test, threshold=result.threshold)
        row = _base_row(record, "FoldBeam-C")
        row.update({
            "val_score": result.val_score,
            "test_balanced_accuracy": metrics["balanced_accuracy"],
            "test_accuracy": metrics["accuracy"],
            "test_logloss": metrics["logloss"],
            "test_auc": metrics["auc"],
            "test_balanced_acc": metrics["balanced_accuracy"],
            "threshold": result.threshold,
            "laplace_alpha": result.laplace_alpha,
            "budget": result.budget,
            "reachable": final_graph.reachable_count(),
            "multi_parent": final_graph.multi_parent_count(),
            "graph_evals": result.graph_evals,
            "refine_evals": result.refine_evals,
            "refine_moves": result.refine_moves,
            "wall_seconds": result.wall_seconds,
            "graph_hash": final_graph.canonical_hash(),
            "factor_map_sha256": factor_map_sha256,
            "llm_factor_map": True,
        })
        rows.append(row)
        log_path = config.path("logs_dir") / "FoldBeam-C" / f"{record['dataset_id']}__{record['split_id']}.jsonl"
        write_jsonl(log_path, [
            *result.lineage,
            {
                "event": "final", "dataset_id": record["dataset_id"], "split_id": record["split_id"],
                "budget": result.budget, "threshold": result.threshold, "laplace_alpha": result.laplace_alpha,
                "champion_J": result.val_score, "factor_map_sha256": factor_map_sha256, "llm_factor_map": True,
                "test_balanced_accuracy": metrics["balanced_accuracy"], "test_accuracy": metrics["accuracy"],
                "test_logloss": metrics["logloss"], "test_auc": metrics["auc"],
                "reachable": final_graph.reachable_count(), "multi_parent": final_graph.multi_parent_count(),
                "graph_evals": result.graph_evals, "refine_evals": result.refine_evals, "refine_moves": result.refine_moves,
                "wall_seconds": result.wall_seconds,
            },
        ])
        graph_path = config.path("results_dir") / "graphs" / f"{record['dataset_id']}__{record['split_id']}.json"
        json_dump(graph_path, final_graph.to_dict())
        print(f"foldbeam {record['dataset_id']} {record['split_id']} bal_acc={metrics['balanced_accuracy']:.4f} logloss={metrics['logloss']:.4f} thr={result.threshold:.2f}")
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
    metric = "test_balanced_accuracy"
    if metric not in frame.columns and "test_balanced_acc" in frame.columns:
        metric = "test_balanced_acc"
    split_columns = ["method", "dataset_id", "split_id", "seed", metric, "test_logloss", "test_accuracy", "test_auc", "budget", "threshold", "laplace_alpha", "reachable", "multi_parent", "graph_evals", "refine_evals", "refine_moves", "wall_seconds", "factor_map_sha256"]
    split_columns = [c for c in split_columns if c in frame.columns]
    split_report = frame[split_columns].sort_values(["method", "dataset_id", "split_id"])
    dataset_summary = frame.groupby(["method", "dataset_id"])[metric].agg(["mean", "std"]).reset_index()
    dataset_summary["mean_plus_minus_std"] = dataset_summary.apply(lambda row: f"{row['mean']:.4f} +/- {row['std']:.4f}", axis=1)
    pivot = dataset_summary.pivot(index="method", columns="dataset_id", values="mean_plus_minus_std")
    ranks = frame.copy()
    ranks["rank"] = ranks.groupby(["dataset_id", "split_id"])[metric].rank(method="average", ascending=False)
    avg_rank = ranks.groupby("method")["rank"].mean().rename("avg_rank")
    report = pivot.join(avg_rank).reset_index()
    out_dir = config.path("reports_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(split_report, out_dir / "split_results.csv")
    _write_csv(dataset_summary, out_dir / "dataset_summary.csv")
    _write_csv(report, out_dir / "main_table.csv")
    print(f"\nPer-split {metric}")
    print(split_report[["method", "dataset_id", "split_id", metric]].to_string(index=False))
    print("\nDataset summary")
    print(report.to_string(index=False))


def _base_row(record: dict, method: str) -> dict:
    return {"method": method, "dataset_id": record["dataset_id"], "split_id": record["split_id"], "seed": int(record["seed"])}


def _write_results(config: Config, name: str, rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    path = config.path("results_dir") / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and not frame.empty:
        previous = pd.read_parquet(path)
        keys = ["method", "dataset_id", "split_id"]
        incoming = set(map(tuple, frame[keys].itertuples(index=False, name=None)))
        previous = previous.loc[~previous[keys].apply(tuple, axis=1).isin(incoming)]
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
