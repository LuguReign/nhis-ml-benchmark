
  # Quick Start Guide
# 1) Install
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -U pip
pip install nhisml

# 2) Fetch raw NHIS Adults files (CSV -> Parquet core)
nhis-fetch 2023 2024

# 3) Process to modeling-ready “core” Parquet (types, missing codes, stable categories)
nhis-process 2023 2024 --rawdir data/processed --outdir data/processed

# 4) Train baseline on 2023 (Logistic L1 + RandomForest, with OOF threshold tuning; optional calibration)
nhis-train --train-core data/processed/Adults23_core.parquet --outdir artifacts --calibrate

# 5) Evaluate on 2024 (external test). Threshold auto-loaded from saved JSON.
MODEL=$(ls artifacts/l1_pipeline_*.joblib | tail -n1)
nhis-evaluate --model "$MODEL" --test-core data/processed/Adults24_core.parquet --outdir artifacts

# 6) (Optional) Subgroup reporting
nhis-srh-subgroup --preds artifacts/test_predictions.csv \
  --test-core data/processed/Adults24_core.parquet \
  --thr 0.30 --by REGION EDUCP_A --out artifacts/subgroup_REGION_EDUCP.csv

  


  # Description of Commands
nhis-fetch 2023 2024 	                     Downloads NHIS Adults CSVs from CDC, saves data/processed/AdultsYY_core.parquet.

nhis-process 2023 2024 	                   Applies survey-aware preprocessing to create modeling-ready core Parquet.

nhis-train --train-core ...	               Trains L1 (sparse logistic) and RF pipelines; tunes thresholds via OOF; optional calibrated versions.

nhis-evaluate --model ...	                 Evaluates on external test (2024); auto-loads threshold JSON when --threshold omitted.

nhis-srh-subgroup --preds ... --by ...	   Computes weighted metrics by subgroup (e.g., REGION, EDUCP_A, etc.).
nhis-explore --core ...	                   Prints a quick, survey-aware EDA summary for a core Parquet (rows, missing, top levels).
nhis-srh-baseline	                         End-to-end baseline runner (fetch → process → train → evaluate) with sensible defaults.

Tip: run any command with -h to see full options.
