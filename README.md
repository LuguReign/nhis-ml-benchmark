# nhisml

[![Tests](https://github.com/LuguReign/nhis-ml-benchmark/actions/workflows/tests.yml/badge.svg)](https://github.com/LuguReign/nhis-ml-benchmark/actions/workflows/tests.yml)
[![PyPI version](https://badge.fury.io/py/nhisml.svg)](https://badge.fury.io/py/nhisml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)

**nhisml** is a survey-aware machine learning toolkit for the National Health Interview Survey (NHIS) Adults public-use microdata. It provides a reproducible, end-to-end pipeline — from raw data download through model training, cross-year evaluation, and subgroup fairness analysis — designed for researchers in public health, epidemiology, and health services research.

## Statement of Need

The NHIS Adults dataset is a rich, nationally representative survey with complex survey weights, multi-year structure, and domain-specific missing-data conventions. Researchers who want to apply machine learning to NHIS data must repeatedly solve the same data-engineering problems: downloading and caching raw files, harmonizing variable names across survey years, handling NHIS-specific missing codes (7/8/9/97/98/99), applying survey weights correctly in model fitting and evaluation, and computing fairness metrics across demographic subgroups. **nhisml** encodes these survey-specific conventions into reusable, well-tested software components, lowering the barrier to rigorous, reproducible ML research on NHIS data.

## Features

- **Survey-aware preprocessing** — NHIS missing-code remapping, binary-coded (1/2 → 1/0) variables, rare-category bucketing, and stable missingness-flag columns
- **Task-aware label generation** — built-in definitions for self-rated health (SRH) and current cigarette smoking, with clean eligibility masking
- **Survey-weighted metrics** — AUC, PR-AUC, F1, Brier score, log-loss, and ECE, all computed with NHIS analytic weights (WTFA_A)
- **Calibration** — out-of-fold (OOF) probability calibration via isotonic regression with threshold optimization
- **Cross-year evaluation** — train on 2023, evaluate on 2024 in a single command
- **Subgroup fairness analysis** — built-in sex, age-band, and education-level recoding with per-subgroup metric deltas vs. overall
- **Publication-ready outputs** — structured run directories with manifests, OOF predictions, metrics JSON, and CSV subgroup tables
- **Extensible** — add new tasks, feature sets, or models by registering them in the appropriate module

## Installation

```bash
pip install nhisml
```

To include visualization support:

```bash
pip install "nhisml[viz]"
```

For development and testing:

```bash
git clone https://github.com/LuguReign/nhis-ml-benchmark.git
cd nhis-ml-benchmark
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.9, numpy ≥ 1.26, pandas ≥ 2.1, scikit-learn ≥ 1.4, joblib ≥ 1.3, requests ≥ 2.31, tqdm ≥ 4.66, pyarrow ≥ 14.

## Quick Start

### 1. Fetch raw data

```bash
# Download and cache raw NHIS Adults public-use files from the CDC FTP server
nhisml fetch --year 2023 --year 2024
```

### 2. Build the core analysis dataset

```bash
# Extract and harmonize predictor and label columns into a clean parquet file
nhisml build-core --year 2023
nhisml build-core --year 2024
```

### 3. Train a baseline model

```bash
# Elastic-net logistic regression (default) with survey-weighted OOF threshold tuning
nhisml train --in data/core_2023.parquet --task srh_binary

# Random forest with probability calibration
nhisml train --in data/core_2023.parquet --task srh_binary --model rf --calibrate
```

### 4. Evaluate on held-out year data

```bash
nhisml evaluate --task srh_binary --latest --year 2024
```

### 5. Subgroup fairness analysis

```bash
nhisml subgroup --task srh_binary --latest --year 2024 --by sex age education
```

## Supported Prediction Tasks

| Task name         | Target variable    | Positive class             |
|-------------------|--------------------|----------------------------|
| `srh_binary`      | PHSTAT_A           | Fair or Poor self-rated health (values 4–5) |
| `smoking_current` | SMKCIGST_A / SMKNOW_A | Current every-day or some-day smoker |

List all available tasks and describe them from the command line:

```bash
nhisml list-tasks
nhisml describe-task srh_binary
```

## Available Feature Sets

The `core` feature set includes 69 NHIS Adults predictors spanning:

- **Health conditions** (hypertension, diabetes, cardiovascular disease, respiratory, etc.)
- **Mental health** (depression, anxiety, psychological distress indices)
- **Healthcare access** (insurance coverage, usual place of care, medication delays)
- **Socioeconomic status** (income-to-poverty ratio, education, employment)
- **Demographics** (region, urbanicity, marital status)

```bash
nhisml list-featuresets
nhisml describe-featureset core
```

## Python API

In addition to the CLI, all functionality is accessible as a Python library:

```python
import nhisml

# Task and feature set definitions
task = nhisml.make_task("srh_binary")
featureset = nhisml.get_featureset("core")

# Build the preprocessing pipeline (unfitted)
preprocessor = nhisml.build_preprocessor(
    binary_cols=featureset.binary_12,
    ordinal_cols=featureset.ordinal,
    categorical_cols=featureset.categorical,
)

# Normalize survey weights
import pandas as pd
df = pd.read_parquet("data/core_2023.parquet")
weights = nhisml.normalize_weights(df["WTFA_A"])

# Generate labels and eligibility mask
y, eligible = task.make_labels(df)
```

## Run Directory Structure

Each `nhisml train` call produces a timestamped run directory under `runs/`:

```
runs/
└── 20240115-143022_task=srh_binary_model=lasso_fs=core/
    ├── manifest.json          # full provenance record
    ├── model.joblib           # fitted sklearn Pipeline
    ├── thresholds.json        # OOF-tuned decision threshold
    ├── oof_predictions.parquet
    └── oof_metrics.json
```

After evaluation and subgroup analysis:

```
    ├── metrics_task=srh_binary.json
    ├── predictions_task=srh_binary.parquet
    └── subgroups_task=srh_binary.csv
```

## Generating Paper Figures and Tables

```bash
python scripts/make_paper_outputs.py \
  --tasks srh_binary smoking_current \
  --run-for-task srh_binary=runs/<srh_run_dir> \
  --run-for-task smoking_current=runs/<smoking_run_dir>
```

## Running the Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

The test suite covers task logic, preprocessing correctness, metric computation, run-resolution utilities, subgroup recoding, fetch caching behavior, and the CLI entry points. Tests do not require downloading NHIS data; synthetic data is generated within each test where needed.

## Contributing

Contributions are welcome. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on filing issues, submitting pull requests, adding new prediction tasks, and extending the feature set registry.

## Citation


```bibtex
@software{nhisml,
  title   = {{nhisml: A survey-aware machine learning toolkit for NHIS Adults data}},
  author  = {Lugu Reign, Nicholas and Lamoreaux, Catherine and Simson, Jan and Kern, Christoph and Kreuter, Frauke},
  year    = {2026},
  version = {0.5.1},
  url     = {https://github.com/LuguReign/nhis-ml-benchmark}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
