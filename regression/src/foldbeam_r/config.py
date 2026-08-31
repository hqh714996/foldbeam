from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when the FoldBeam-R configuration is invalid."""


@dataclass(frozen=True)
class Config:
    source: Path
    payload: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise ConfigError(f"Configuration file does not exist: {source}")
        with source.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        if not isinstance(payload, dict):
            raise ConfigError("The YAML root must be a mapping.")
        config = cls(source=source, payload=payload)
        config.validate()
        return config

    @property
    def project_root(self) -> Path:
        value = self.payload["paths"].get("project_root", ".")
        return (self.source.parent / value).resolve()

    def path(self, key: str) -> Path:
        paths = self.section("paths")
        if key not in paths:
            raise ConfigError(f"Unknown configured path: {key}")
        return (self.project_root / paths[key]).resolve()

    def section(self, key: str) -> dict[str, Any]:
        value = self.payload.get(key)
        if not isinstance(value, dict):
            raise ConfigError(f"Configuration section must be a mapping: {key}")
        return value

    @property
    def dataset_ids(self) -> list[str]:
        return list(self.section("datasets"))

    def dataset(self, dataset_id: str) -> dict[str, Any]:
        datasets = self.section("datasets")
        if dataset_id not in datasets:
            raise ConfigError(f"Unknown dataset: {dataset_id}")
        return datasets[dataset_id]

    def digest(self) -> str:
        canonical = json.dumps(self.payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def ensure_output_directories(self) -> None:
        for key in (
            "raw_dir",
            "openml_cache",
            "processed_dir",
            "factor_maps_dir",
            "results_dir",
            "logs_dir",
            "reports_dir",
        ):
            self.path(key).mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        required = {"paths", "data", "graph", "analysis", "datasets"}
        missing = sorted(required.difference(self.payload))
        if missing:
            raise ConfigError(f"Missing configuration sections: {missing}")

        data = self.section("data")
        fractions = [
            float(data["train_fraction"]),
            float(data["validation_fraction"]),
            float(data["test_fraction"]),
        ]
        if any(value <= 0.0 or value >= 1.0 for value in fractions):
            raise ConfigError("Split fractions must lie inside (0, 1).")
        if abs(sum(fractions) - 1.0) > 1e-9:
            raise ConfigError("Split fractions must sum to one.")
        if abs(float(data["validation_fraction_of_remaining"]) - 0.4) > 1e-9:
            raise ConfigError("validation_fraction_of_remaining must be 0.4 for LLEGO alignment.")
        if len(set(data["split_seeds"])) != len(data["split_seeds"]):
            raise ConfigError("data.split_seeds must be unique.")

        graph = self.section("graph")
        kd = int(graph["shortlist_data"])
        ks = int(graph["shortlist_semantic"])
        k = int(graph["shortlist_k"])
        if kd + ks > k:
            raise ConfigError("shortlist_data + shortlist_semantic cannot exceed shortlist_k.")
        if int(graph["beam_width"]) < 1:
            raise ConfigError("beam_width must be positive.")
        if sorted(graph["budget_grid"]) != list(graph["budget_grid"]):
            raise ConfigError("budget_grid must be ascending.")
        grid = graph.get("terminal_lambda_grid")
        if grid is not None:
            if not isinstance(grid, list) or not grid:
                raise ConfigError("terminal_lambda_grid must be a non-empty list.")
            if any(float(value) <= 0.0 for value in grid):
                raise ConfigError("terminal_lambda_grid values must be positive.")
        selection_metric = graph.get("selection_metric", "val")
        if selection_metric not in {"val", "train"}:
            raise ConfigError(f"selection_metric must be 'val' or 'train', got {selection_metric!r}.")


def stable_hash(value: Any, length: int = 16) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
