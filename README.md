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
