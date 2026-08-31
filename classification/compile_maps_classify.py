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

from foldbeam_c.config import Config, ConfigError, json_dump
from foldbeam_c.data import load_index

ENV_FILE = HERE.parent / "gap1_minimal_experiment_package" / "experiment_1_candidate_ranking" / ".env"
DIRECTIONS = {"positive_class", "negative_class", "nonmonotone"}
JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile FoldBeam-C classification factor maps")
    parser.add_argument("--config", default=str(HERE / "config_classify.yaml"))
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
        category_levels = _load_category_levels(config, dataset_id)
        payload = compile_one(client, dataset_id, record, category_levels)
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
        missing = [name for name, value in (("EXP1_LLM_BASE_URL", base_url), ("EXP1_LLM_API_KEY", api_key), ("EXP1_LLM_MODEL", model)) if not value]
        if missing:
            raise ConfigError(f"Missing LLM environment variables: {missing}")
        return cls(base_url, api_key, model)

    def close(self) -> None:
        self.client.close()

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        payload = {"model": self.model, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], "temperature": 0.0, "max_tokens": 4096, "response_format": {"type": "json_object"}}
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


SYSTEM_PROMPT = """You are the semantic annotation module of FoldBeam, a structure-learning system that
builds a small shared decision graph for tabular binary classification. Your factor map is
the only world-knowledge input the search receives, and its quality directly changes
predictive accuracy, so reason carefully before answering. You see no data rows and no
label statistics; the encoding rules stated in the request are fixed properties of the
pipeline, not observed statistics. Return valid JSON only, with no markdown fences and no
extra text."""


def compile_one(client: LLMClient, dataset_id: str, record: dict[str, Any], category_levels: dict[str, dict[str, str]]) -> dict[str, Any]:
    feature_block = _feature_block(record, category_levels)
    positive_label = str(record.get("positive_label", "positive"))
    user_prompt = f"""
## Task
{record["task"]}

Positive class: {positive_label} (the class we want to predict)
Target meaning: whether the sample belongs to the positive class.

## How your output is used
The search grows a decision graph over the features below. When it considers letting two
regions of the graph share structure (a "reuse" edge), it embeds each region's root-to-node
path as a vector over YOUR factors: every predicate on the path adds
(+strength x direction) to its factor when the path takes the HIGH side of the encoded
feature, and (-strength x direction) when it takes the LOW side. Two regions are judged
semantically compatible when their vectors point the same way in factor space (cosine
similarity). Therefore:
- a factor must be ONE latent construct, such that "high on the construct" means the same
  thing whichever feature measured it;
- direction must be correct with respect to the ENCODED column value described below;
- strength must express how reliably the feature measures its factor.

## Encoding contract (critical)
- Numeric features are standardized; their direction is unchanged by the scaling.
- Categorical features are integer-coded by a fixed pipeline rule: the levels are ordered
  by their positive-class rate on the training split, ascending, and given codes 0, 1, 2, ...
  So for EVERY categorical feature, a higher code is BY CONSTRUCTION the level associated
  with a higher positive rate on the training split. You are not shown those rates; the
  rule itself is a property of the pipeline.
- A split "value <= t" takes the low side of the encoded column; "value > t" takes the
  high side.

## Features
{feature_block}

## Annotation protocol
1. Factors: choose the smallest number of latent factors (between 2 and 6) such that each
   factor is a single underlying condition that the features probe from different angles.
   Keep constructs causally distinct - for example current assets, past repayment
   behaviour, and job stability are three different constructs and must not be merged into
   one "general goodness" factor. Keep the factors balanced: no factor may contain more
   than about a third of all features; if one grouping grows larger, split it into its
   real constituents.
2. Direction is with respect to the ENCODED value and the positive class:
   - Categorical feature: higher code = higher positive rate by construction, so the
     correct direction is "positive_class". This is a property of the coding rule, NOT a
     domain judgement - your real-world belief about which level is riskier is already
     reflected in the coding and must not override it.
   - Numeric feature: judge the DOMINANT monotone trend from domain knowledge and choose
     "positive_class" or "negative_class". Reserve "nonmonotone" for a genuinely dominant
     U-shape over most of the range; a nonmonotone mark removes the feature from polarity
     matching entirely, so never use it merely because you are unsure - pick the dominant
     trend instead.
3. Strength in [0, 1] is how reliably the feature measures its factor's level:
   0.9-1.0 near-definitional; 0.7-0.8 strong proxy; 0.4-0.6 moderate proxy; 0.1-0.3 weak
   or noisy proxy. Use the whole scale and differentiate between features; do not cluster
   every value near the top. For a categorical feature, lower the strength when its level
   ordering is likely noisy or only weakly tied to the factor.

## Output
Return ONLY this JSON object, no extra text:
{{
  "reasoning": "<brief justification of the factor design, 3-6 sentences>",
  "factors": [
    {{"factor_id": "F1", "name": "<short name>", "description": "<one sentence>"}}
  ],
  "features": {{
    "<feature_name>": {{"factor_id": "F1", "strength": 0.8, "direction": "positive_class"}}
  }}
}}

Rules: every feature listed above must appear exactly once in "features", with its name
copied exactly; every factor_id used must be defined in "factors"; use between 2 and 6
factors; no factor may contain more than about a third of all features; the direction of
every feature shown as [categorical, coded by the rule above] MUST be "positive_class" -
never "negative_class" or "nonmonotone" for such features.
""".strip()
    raw = client.complete_json(SYSTEM_PROMPT, user_prompt)
    return validate_factor_map(dataset_id, record["feature_names"], raw)


def _feature_block(record: dict[str, Any], category_levels: dict[str, dict[str, str]]) -> str:
    dtypes = record.get("dtypes") or {}
    codebooks = record.get("codebooks") or {}
    lines: list[str] = []
    for name, desc in zip(record["feature_names"], record["feature_semantics"], strict=True):
        kind = str(dtypes.get(name, "numeric"))
        if kind != "categorical":
            lines.append(f"- {name} [numeric]: {desc}")
            continue
        levels = sorted(str(level) for level in (codebooks.get(name) or {}))
        meanings = category_levels.get(name) or {}
        if not levels:
            lines.append(f"- {name} [categorical, coded by the rule above]: {desc}")
            continue
        level_lines = []
        for level in levels:
            meaning = meanings.get(level)
            level_lines.append(f"    * {level} - {meaning}" if meaning else f"    * {level}")
        lines.append(f"- {name} [categorical, coded by the rule above]: {desc}\n" + "\n".join(level_lines))
    return "\n".join(lines)


def _load_category_levels(config: Config, dataset_id: str) -> dict[str, dict[str, str]]:
    import yaml

    semantics_path = config.path("semantics_file")
    if not semantics_path.is_file():
        return {}
    with semantics_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    entry = (payload.get("datasets") or {}).get(dataset_id) or {}
    levels = entry.get("category_levels") or {}
    return {str(feature): {str(k): str(v) for k, v in (mapping or {}).items()} for feature, mapping in levels.items()}


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
        clean_factors.append({"factor_id": factor_id, "name": str(item.get("name", factor_id))[:80], "description": str(item.get("description", ""))[:300]})
    if len(set(factor_ids)) != len(factor_ids):
        raise ConfigError(f"{dataset_id}: duplicate factor_id")
    expected = set(feature_names)
    observed = set(features)
    if expected != observed:
        raise ConfigError(f"{dataset_id}: factor map feature mismatch. Missing={sorted(expected-observed)}, extra={sorted(observed-expected)}")
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
        clean_features[name] = {"factor_id": factor_id, "strength": strength, "direction": direction}
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
    manifest: dict[str, Any] = {"config_hash": config.digest(), "dataset_id": dataset_id, "model": model, "factor_map_sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "compiler": "compile_maps_classify.py", "prompt_schema": "foldbeam_c_factor_map_v2"}
    if usage:
        manifest["usage"] = dict(usage)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if key in usage:
                manifest[key] = usage[key]
    json_dump(output.with_suffix(".manifest.json"), manifest)


if __name__ == "__main__":
    raise SystemExit(main())
