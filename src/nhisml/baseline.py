# src/nhisml/baseline.py
"""
End-to-end convenience runner:
- fetch 2023+2024
- process (build preprocessor per year)
- train L1/RF on 2023 (with OOF threshold + optional calibration)
- evaluate on 2024 and write artifacts
"""
import argparse, os, glob, json, joblib, numpy as np, pandas as pd, datetime as dt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from .fetch import fetch_year
from .process import process_year
from .featuresets import get_featureset
from .preprocess import normalize_weights, build_preprocessor
from .utils import weighted_threshold_via_oof, fit_calibrated_from_oof

def _bin_srh(s: pd.Series) -> np.ndarray:
    x = pd.to_numeric(s, errors="coerce")
    return (x >= 4).astype(int).to_numpy()

def cli():
    p = argparse.ArgumentParser("nhis-srh-baseline")
    p.add_argument("--outdir", default="artifacts")
    p.add_argument("--datadir", default="data/processed")
    p.add_argument("--calibrate", action="store_true")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.datadir, exist_ok=True)

    # 1) fetch
    fetch_year(2023, outdir=args.datadir)
    fetch_year(2024, outdir=args.datadir)

    # 2) process per-year (locks preproc per year; training will refit on 2023 again)
    process_year(2023, os.path.join(args.datadir, "Adults23_core.parquet"), args.datadir)
    process_year(2024, os.path.join(args.datadir, "Adults24_core.parquet"), args.datadir)

    # 3) train on 2023
    train_core = pd.read_parquet(os.path.join(args.datadir, "Adults23_core.parquet"))
    feats = get_featureset("srh_core")
    keep = [c for c in feats.all_columns if c in train_core.columns]
    core = train_core[keep].copy()
    y = _bin_srh(core["PHSTAT_A"])
    w = normalize_weights(core["WTFA_A"]) if "WTFA_A" in core.columns else np.ones(len(core))

    ct, _, df_fit = build_preprocessor(
        core,
        binary_cols=[c for c in feats.binary_12 if c in core.columns],
        ordinal_cols=[c for c in feats.ordinal if c in core.columns],
        categorical_cols=[c for c in feats.categorical if c in core.columns],
        add_missing_flags=True, rare_min_count=50
    )

    rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=10, random_state=42, n_jobs=-1)
    l1 = LogisticRegression(penalty="l1", solver="saga", C=0.5, max_iter=2000, n_jobs=-1, random_state=42)

    rf_pipe = Pipeline([("prep", ct), ("rf", rf)])
    l1_pipe = Pipeline([("prep", ct), ("lasso", l1)])

    rf_pipe.fit(df_fit, y, rf__sample_weight=w)
    l1_pipe.fit(df_fit, y, lasso__sample_weight=w)

    thr_rf, perf_rf = weighted_threshold_via_oof(rf_pipe, df_fit, y, w, step_name="rf")
    thr_l1, perf_l1 = weighted_threshold_via_oof(l1_pipe, df_fit, y, w, step_name="lasso")

    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    m_rf = os.path.join(args.outdir, f"rf_pipeline_{ts}.joblib")
    m_l1 = os.path.join(args.outdir, f"l1_pipeline_{ts}.joblib")
    joblib.dump(rf_pipe, m_rf)
    joblib.dump(l1_pipe, m_l1)

    thresholds = {"rf": float(thr_rf), "l1": float(thr_l1)}

    # optional: calibrated variants
    if args.calibrate:
        rf_cal, thr_rf_cal, _ = fit_calibrated_from_oof(rf_pipe, df_fit, y, w, step_name="rf")
        l1_cal, thr_l1_cal, _ = fit_calibrated_from_oof(l1_pipe, df_fit, y, w, step_name="lasso")
        joblib.dump(rf_cal, os.path.join(args.outdir, f"rf_calibrated_{ts}.joblib"))
        joblib.dump(l1_cal, os.path.join(args.outdir, f"l1_calibrated_{ts}.joblib"))
        thresholds.update({"rf_cal": float(thr_rf_cal), "l1_cal": float(thr_l1_cal)})

    with open(os.path.join(args.outdir, f"thresholds_{ts}.json"), "w") as f:
        json.dump(thresholds, f, indent=2)

    # 4) evaluate on 2024 with the L1 model + tuned threshold (example)
    test_core = pd.read_parquet(os.path.join(args.datadir, "Adults24_core.parquet"))
    y_te = _bin_srh(test_core["PHSTAT_A"])
    w_te = normalize_weights(test_core["WTFA_A"]) if "WTFA_A" in test_core.columns else np.ones(len(test_core))
    prob = l1_pipe.predict_proba(test_core)[:, 1]
    thr = thresholds["l1"]
    pred = (prob >= thr).astype(int)

    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score, average_precision_score
    metrics = dict(
        weighted_auc=float(roc_auc_score(y_te, prob, sample_weight=w_te)),
        weighted_accuracy=float(accuracy_score(y_te, pred, sample_weight=w_te)),
        balanced_accuracy=float(balanced_accuracy_score(y_te, pred)),
        weighted_f1=float(f1_score(y_te, pred, sample_weight=w_te)),
        avg_precision=float(average_precision_score(y_te, prob, sample_weight=w_te)),
        threshold=float(thr)
    )
    pd.DataFrame([metrics]).to_csv(os.path.join(args.outdir, "test_metrics.csv"), index=False)
    pd.DataFrame({"prob": prob, "pred": pred, "srh_bin": y_te, "wt": w_te}).to_csv(
        os.path.join(args.outdir, "test_predictions.csv"), index=False
    )
    print("Baseline complete. Artifacts saved.")
