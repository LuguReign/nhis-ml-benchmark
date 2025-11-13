import argparse, os, glob, json, numpy as np, pandas as pd, joblib
from sklearn.metrics import roc_auc_score

def _wmean(x, w):
    w = np.asarray(w, dtype=float)
    x = np.asarray(x, dtype=float)
    return (x * w).sum() / (w.sum() if w.sum() > 0 else 1.0)

def _prf_auc(g, prob_col, y_col, w_col):
    y = g[y_col].to_numpy()
    w = g[w_col].to_numpy()
    yhat = g["pred"].to_numpy()

    tp = w[(y == 1) & (yhat == 1)].sum()
    fp = w[(y == 0) & (yhat == 1)].sum()
    fn = w[(y == 1) & (yhat == 0)].sum()

    prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    rec  = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1   = 2 * prec * rec / (prec + rec) if prec > 0 and rec > 0 else np.nan
    acc  = _wmean((y == yhat).astype(float), w)

    # AUC needs both classes present
    auc = np.nan
    if y.sum() > 0 and (1 - y).sum() > 0:
        auc = roc_auc_score(y, g[prob_col], sample_weight=w)

    return pd.Series(dict(
        weighted_accuracy=acc,
        weighted_precision=prec,
        weighted_recall=rec,
        weighted_f1=f1,
        weighted_auc=auc
    ))

def subgroup_metrics(df, prob_col, y_col, w_col, by_cols, thr):
    df = df.copy()
    df["pred"] = (df[prob_col] >= float(thr)).astype(int)

    # keep only available grouping columns; warn for missing
    missing = [c for c in by_cols if c not in df.columns]
    if missing:
        print(f"[subgroup] WARNING: missing subgroup columns: {missing}. They will be skipped.")
    by_cols = [c for c in by_cols if c in df.columns]
    if not by_cols:
        raise ValueError("No valid subgroup columns found in the provided data.")

    out = (
        df.groupby(by_cols, dropna=False)
          .apply(lambda g: _prf_auc(g, prob_col, y_col, w_col))
          .reset_index()
    )
    return out

def _latest_thresholds(thr_dir="artifacts"):
    files = sorted(glob.glob(os.path.join(thr_dir, "thresholds_*.json")))
    return files[-1] if files else None

def _infer_thr_key_from_model_path(model_path: str) -> str:
    name = os.path.basename(model_path).lower()
    # Heuristic based on our filenames
    if "calibrated" in name and name.startswith("l1"):
        return "l1_cal"
    if "calibrated" in name and name.startswith("rf"):
        return "rf_cal"
    if name.startswith("l1"):
        return "l1"
    if name.startswith("rf"):
        return "rf"
    return "l1"  # default

def _resolve_threshold(thr, thresholds_path, thr_key, model_path):
    # If user passed --thr explicitly, use it
    if thr is not None:
        return float(thr), "(explicit --thr)"

    # If not, try thresholds file
    if thresholds_path is None:
        thresholds_path = _latest_thresholds()
    if thresholds_path and os.path.exists(thresholds_path):
        with open(thresholds_path) as f:
            d = json.load(f)
        # If no thr_key provided, infer from model filename; else default to 'l1'
        if thr_key is None:
            thr_key = _infer_thr_key_from_model_path(model_path) if model_path else "l1"
        if thr_key in d:
            return float(d[thr_key]), f"{thr_key} in {thresholds_path}"
        # Fall-through: try common keys
        for k in ("l1", "rf", "l1_cal", "rf_cal"):
            if k in d:
                return float(d[k]), f"{k} in {thresholds_path}"
        # Nothing usable in JSON
        print(f"[subgroup] WARNING: thresholds file found but no usable keys. Using 0.5.")
        return 0.5, "default 0.5 (no matching key in thresholds)"
    else:
        print("[subgroup] INFO: no thresholds file found. Using 0.5.")
        return 0.5, "default 0.5 (no thresholds file)"

def cli():
    p = argparse.ArgumentParser("nhis-srh-subgroup")
    p.add_argument("--preds", help="Path to artifacts/test_predictions.csv (prob/pred/srh_bin/wt). Fast path.")
    p.add_argument("--model", help="Path to artifacts/*_pipeline_*.joblib (compute probs on the fly).")
    p.add_argument("--test-core", default="data/processed/Adults24_core.parquet",
                   help="When using --model, core parquet with PHSTAT_A and WTFA_A.")
    p.add_argument("--thr", type=float, default=None,
                   help="Decision threshold. If omitted, will try thresholds JSON, else 0.5.")
    p.add_argument("--thresholds", help="Path to thresholds_*.json (optional).")
    p.add_argument("--thr-key", help="Key in thresholds JSON (e.g., l1, rf, l1_cal, rf_cal). Optional.")
    p.add_argument("--by", nargs="+", required=True, help="Subgroup columns, e.g. REGION URBRRL23 EDUCP_A")
    p.add_argument("--out", default="artifacts/subgroup.csv")
    args = p.parse_args()

    if not args.preds and not args.model:
        raise SystemExit("Provide either --preds or --model.")

    # Resolve threshold
    thr, how = _resolve_threshold(args.thr, args.thresholds, args.thr_key, args.model)
    print(f"[subgroup] Using threshold={thr:.4f} ({how})")

    if args.preds:
        df = pd.read_csv(args.preds)
        # Ensure subgroup columns present; if not, merge from core
        need = [c for c in args.by if c not in df.columns]
        if need:
            core = pd.read_parquet(args.test_core)
            add = core[need].reset_index(drop=True)
            df = pd.concat([df.reset_index(drop=True), add], axis=1)
        out = subgroup_metrics(df, "prob", "srh_bin", "wt", args.by, thr)
    else:
        # Compute probs from model + core
        from .preprocess import normalize_weights
        pipe = joblib.load(args.model)
        core = pd.read_parquet(args.test_core)
        if "PHSTAT_A" not in core.columns:
            raise ValueError("--test-core must contain PHSTAT_A for labels")
        y = (pd.to_numeric(core["PHSTAT_A"], errors="coerce") >= 4).astype(int)
        w = normalize_weights(core["WTFA_A"]) if "WTFA_A" in core.columns else np.ones(len(core))
        proba = pipe.predict_proba(core)[:, 1]
        df = core[args.by].copy()
        df["srh_bin"], df["wt"], df["prob"] = y, w, proba
        out = subgroup_metrics(df, "prob", "srh_bin", "wt", args.by, thr)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[subgroup] Saved {args.out}")

if __name__ == "__main__":
    cli()
