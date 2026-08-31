from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx


HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from foldbeam_r.config import Config, ConfigError, json_dump
from foldbeam_r.data import load_index


ENV_FILE = HERE.parent / "gap1_minimal_experiment_package" / "experiment_1_candidate_ranking" / ".env"
DIRECTIONS = {"higher_target", "lower_target", "nonmonotone"}
JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile FoldBeam-R regression factor maps")
    parser.add_argument("--config", default=str(HERE / "config_regression.yaml"))
    parser.add_argument("--dataset", action="append", help="Dataset id to compile. Defaults to all prepared datasets.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing factor maps.")
    args = parser.parse_args(argv)

    _load_dotenv()
    config = Config.load(args.config)
    config.ensure_output_directories()
    records = load_index(config)
    by_dataset: dict[str, dict[str, Any]] = {}
    for record in records:
        by_dataset.setdefault(record["dataset_id"], record)
    requested = args.dataset or list(by_dataset)
    missing = sorted(set(requested).difference(by_dataset))
    if missing:
        raise ConfigError(f"Datasets are not prepared yet: {missing}. Run prepare first.")

    client = LLMClient.from_env()
    for dataset_id in requested:
        output = config.path("factor_maps_dir") / f"{dataset_id}.json"
        if output.is_file() and not args.force:
            _write_manifest(config, dataset_id, output, client.model)
            print(f"skip existing {output}")
            continue
        record = by_dataset[dataset_id]
        payload = compile_one(client, dataset_id, record)
        json_dump(output, payload)
        _write_manifest(config, dataset_id, output, client.model, dict(client.last_usage))
        usage = client.last_usage
        if usage:
            print(f"wrote {output}  tokens prompt={usage.get('prompt_tokens')} completion={usage.get('completion_tokens')} total={usage.get('total_tokens')}")
        else:
            print(f"wrote {output}")
    client.close()
    return 0


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.client = httpx.Client(timeout=180.0, limits=httpx.Limits(max_connections=8))
        self.last_usage: dict[str, Any] = {}

    @classmethod
    def from_env(cls) -> "LLMClient":
        base_url = os.getenv("EXP1_LLM_BASE_URL", "").strip()
        api_key = os.getenv("EXP1_LLM_API_KEY", "").strip()
        model = os.getenv("EXP1_LLM_MODEL", "").strip()
        missing = [
            name
            for name, value in (
                ("EXP1_LLM_BASE_URL", base_url),
                ("EXP1_LLM_API_KEY", api_key),
                ("EXP1_LLM_MODEL", model),
            )
            if not value
        ]
        if missing:
            raise ConfigError(f"Missing LLM environment variables: {missing}")
        return cls(base_url, api_key, model)

    def close(self) -> None:
        self.client.close()

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 2048,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        error = None
        for attempt in range(6):
            try:
                response = self.client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
                if response.status_code in {429, 500, 502, 503, 504}:
                    error = f"HTTP {response.status_code}: {response.text[:200]}"
                    time.sleep(min(2.0 * (2 ** attempt), 30.0))
                    continue
                response.raise_for_status()
                body = response.json()
                try:
                    self.last_usage = dict(body.get("usage") or {})
                except Exception:
                    self.last_usage = {}
                text = body["choices"][0]["message"]["content"]
                match = JSON_OBJECT.search(text)
                if not match:
                    raise ConfigError("LLM response did not contain a JSON object.")
                return json.loads(match.group(0))
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                time.sleep(min(2.0 * (2 ** attempt), 30.0))
        raise ConfigError(f"LLM request failed after retries: {error}")


def compile_one(client: LLMClient, dataset_id: str, record: dict[str, Any]) -> dict[str, Any]:
    feature_block = "\n".join(
        f"- {name}: {desc}"
        for name, desc in zip(record["feature_names"], record["feature_semantics"], strict=True)
    )
    user_prompt = f"""
You are helping to organize the semantics of a tabular regression task. You will
see ONLY the task description, target meaning, and feature meanings. You will NOT
see any data rows, target values, summaries, label rates, or statistics.

## Task
{record["task"]}

Target column: {record["target_column"]}
Target meaning: a continuous numeric regression target for this task.

## Features
{feature_block}

## Your job
Group the features into a small number of latent factors, between 2 and 6. A
factor is an underlying real-world quantity or condition that several features
jointly describe. Then, for every feature, state:

1. which single factor it belongs to, as factor_id;
2. how strongly it reflects that factor, as a number in [0, 1];
3. its direction with respect to the continuous target:
   - "higher_target" if higher values of the feature tend to indicate a higher target,
   - "lower_target" if higher values tend to indicate a lower target,
   - "nonmonotone" if the relationship is non-monotonic or you cannot tell.

Return ONLY this JSON object, no extra text:
{{
  "factors": [
    {{"factor_id": "F1", "name": "<short name>", "description": "<one sentence>"}}
  ],
  "features": {{
    "<feature_name>": {{"factor_id": "F1", "strength": 0.8, "direction": "higher_target"}}
  }}
}}

Rules: every feature listed above must appear exactly once in "features", with
its name copied exactly; every factor_id used must be defined in "factors"; use
between 2 and 6 factors.
""".strip()
    raw = client.complete_json(
        "Return valid JSON only. Do not use data values or statistics.",
        user_prompt,
    )
    return validate_factor_map(dataset_id, record["feature_names"], raw)


def validate_factor_map(dataset_id: str, feature_names: list[str], payload: dict[str, Any]) -> dict[str, Any]:
    factors = payload.get("factors")
    features = payload.get("features")
    if not isinstance(factors, list) or not 2 <= len(factors) <= 6:
        raise ConfigError(f"{dataset_id}: factors must be a list of length 2..6")
    if not isinstance(features, dict):
        raise ConfigError(f"{dataset_id}: features must be a mapping")
    factor_ids = []
    clean_factors = []
    for item in factors:
        if not isinstance(item, dict):
            raise ConfigError(f"{dataset_id}: each factor must be a mapping")
        factor_id = str(item.get("factor_id", "")).strip()
        if not factor_id:
            raise ConfigError(f"{dataset_id}: factor_id is missing")
        factor_ids.append(factor_id)
        clean_factors.append(
            {
                "factor_id": factor_id,
                "name": str(item.get("name", factor_id))[:80],
                "description": str(item.get("description", ""))[:300],
            }
        )
    if len(set(factor_ids)) != len(factor_ids):
        raise ConfigError(f"{dataset_id}: duplicate factor_id")

    expected = set(feature_names)
    observed = set(features)
    if expected != observed:
        raise ConfigError(
            f"{dataset_id}: factor map feature mismatch. Missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}"
        )
    clean_features = {}
    for name in feature_names:
        entry = features[name]
        if not isinstance(entry, dict):
            raise ConfigError(f"{dataset_id}/{name}: feature entry must be a mapping")
        factor_id = str(entry.get("factor_id", "")).strip()
        if factor_id not in factor_ids:
            raise ConfigError(f"{dataset_id}/{name}: unknown factor_id {factor_id!r}")
        strength = float(entry.get("strength"))
        if not 0.0 <= strength <= 1.0:
            raise ConfigError(f"{dataset_id}/{name}: strength out of [0,1]")
        direction = str(entry.get("direction", ""))
        if direction not in DIRECTIONS:
            raise ConfigError(f"{dataset_id}/{name}: invalid direction {direction!r}")
        clean_features[name] = {
            "factor_id": factor_id,
            "strength": strength,
            "direction": direction,
        }
    return {"dataset_id": dataset_id, "factors": clean_factors, "features": clean_features}


def _load_dotenv() -> None:
    if not ENV_FILE.is_file():
        return
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _write_manifest(config: Config, dataset_id: str, output: Path, model: str, usage: dict[str, Any] | None = None) -> None:
    manifest: dict[str, Any] = {
        "config_hash": config.digest(),
        "dataset_id": dataset_id,
        "model": model,
        "factor_map_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "compiler": "compile_maps_regression.py",
        "prompt_schema": "foldbeam_r_factor_map_v1",
    }
    if usage:
        manifest["usage"] = dict(usage)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if key in usage:
                manifest[key] = usage[key]
    json_dump(output.with_suffix(".manifest.json"), manifest)


if __name__ == "__main__":
    raise SystemExit(main())
