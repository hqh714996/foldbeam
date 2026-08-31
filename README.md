# FoldBeam

LLM-guided beam search for decision-DAG optimization with dynamic prompting.

One LLM factor-map compilation per dataset (zero calls inside the search loop):
the search itself is deterministic NEW / REUSE / STOP beam search over
decision DAGs, with data-channel and semantic-channel REUSE nomination,
incompatibility screening, and full adjudication on the selection split.

## Layout

```
foldbeam_release/
  regression/          # FoldBeam-R (foldbeam_r)
    src/foldbeam_r/    # model implementation
    config.yaml        # active hp_v3 configuration (LLEGO-aligned protocol)
    run_regression_experiment.py     # prepare / foldbeam / summary runner
    compile_maps_regression.py       # per-dataset LLM factor-map compiler
  classification/      # FoldBeam-C (foldbeam_c)
    src/foldbeam_c/
    config.yaml
    run_classify_experiment.py
    compile_maps_classify.py
```

## Protocol (LLEGO-aligned)

- Fixed test split (`test_size=0.4`, `random_state=0`); each run seed re-divides
  the remaining 60% into train/validation (36% / 24% / 40%).
- Regression: train-std scaled test MSE (sample std, ddof=1), seeds 0..9.
- Classification: test balanced accuracy, seeds 0..4, stratified splits.
- Both runners support `prepare / foldbeam / all / summary` stages and
  `--dataset`, `--limit`, `--config` flags.

## Requirements

Python 3.11+ with numpy, pandas, scikit-learn, scipy, pyarrow, httpx, pyyaml.

## Usage

```
# regression
python regression/run_regression_experiment.py prepare   --dataset cars
python regression/compile_maps_regression.py             --dataset cars
python regression/run_regression_experiment.py foldbeam  --dataset cars
python regression/run_regression_experiment.py summary   --dataset cars

# classification
python classification/run_classify_experiment.py prepare   --dataset breast
python classification/compile_maps_classify.py             --dataset breast
python classification/run_classify_experiment.py foldbeam  --dataset breast
python classification/run_classify_experiment.py summary   --dataset breast
```

LLM credentials are read from `EXP1_LLM_BASE_URL`, `EXP1_LLM_API_KEY`,
`EXP1_LLM_MODEL` environment variables; the compilers fall back to a local
`.env` file when present. Per-dataset token usage is recorded in
`factor_maps/<dataset>.manifest.json`.

## Data

Datasets are fetched from OpenML by fixed data id (regression: abalone 44956,
wine 287, cholesterol 204, wage 534, cars 44994; classification: breast 15,
compas 42192, credit_g 31, diabetes 37, heart_statlog 53, liver 1480,
vehicle 994) and cached locally; a manually placed CSV under `data/` is used
first when present.
