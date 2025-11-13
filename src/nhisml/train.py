import argparse, os, joblib, numpy as np, pandas as pd, datetime as dt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from .utils import weighted_threshold_via_oof, fit_calibrated_from_oof
from .preprocess import build_preprocessor, normalize_weights
from .featuresets import get_featureset


def cli():
    p = argparse.ArgumentParser("nhis-train")
    p.add_argument("--train-core", default="data/processed/Adults23_core.parquet")
    p.add_argument("--outdir", default="artifacts")
    p.add_argument("--featureset", default="srh_core")
    p.add_argument("--calibrate", action="store_true")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_parquet(args.train_core)

    feats = get_featureset(args.featureset)
    keep = [c for c in feats.all_columns if c in df.columns]
    core = df[keep].copy()

    y = (pd.to_numeric(core["PHSTAT_A"], errors="coerce") >= 4).astype(int)
    w = normalize_weights(core["WTFA_A"]) if "WTFA_A" in core.columns else np.ones(len(core))

    # Build preprocessor (PrepareFrame → ColumnTransformer)
    preproc, schema, df_fit = build_preprocessor(
        train_core=core,
        binary_cols=[c for c in feats.binary_12 if c in core.columns],
        ordinal_cols=[c for c in feats.ordinal if c in core.columns],
        categorical_cols=[c for c in feats.categorical if c in core.columns],
        rare_min_count=50,
    )

    # Define models
    rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=10, random_state=42, n_jobs=-1)
    l1 = LogisticRegression(penalty="l1", solver="saga", C=0.5, max_iter=2000, n_jobs=-1, random_state=42)

    # Final pipelines: preprocessor + estimator
    rf_pipe = Pipeline([("prep", preproc), ("rf", rf)])
    l1_pipe = Pipeline([("prep", preproc), ("lasso", l1)])

    # Fit with weights
    rf_pipe.fit(core, y, rf__sample_weight=w)
    l1_pipe.fit(core, y, lasso__sample_weight=w)

    # Threshold tuning via weighted OOF on the training year
    thr_rf, _ = weighted_threshold_via_oof(rf_pipe, core, y, w, step_name="rf")
    thr_l1, _ = weighted_threshold_via_oof(l1_pipe, core, y, w, step_name="lasso")

    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    rf_path = os.path.join(args.outdir, f"rf_pipeline_{ts}.joblib")
    l1_path = os.path.join(args.outdir, f"l1_pipeline_{ts}.joblib")
    joblib.dump(rf_pipe, rf_path)
    joblib.dump(l1_pipe, l1_path)

    # calibrated variants from OOF
    out_thresholds = {
        "rf": float(thr_rf),
        "l1": float(thr_l1),
    }
    if args.calibrate:
        rf_cal, thr_rf_cal, _ = fit_calibrated_from_oof(rf_pipe, core, y, w, step_name="rf")
        l1_cal, thr_l1_cal, _ = fit_calibrated_from_oof(l1_pipe, core, y, w, step_name="lasso")
        joblib.dump(rf_cal, os.path.join(args.outdir, f"rf_calibrated_{ts}.joblib"))
        joblib.dump(l1_cal, os.path.join(args.outdir, f"l1_calibrated_{ts}.joblib"))
        out_thresholds.update({
            "rf_cal": float(thr_rf_cal),
            "l1_cal": float(thr_l1_cal),
        })

    with open(os.path.join(args.outdir, f"thresholds_{ts}.json"), "w") as f:
        import json; json.dump(out_thresholds, f, indent=2)

    print("Training complete.")
    print(f" - {rf_path}")
    print(f" - {l1_path}")
