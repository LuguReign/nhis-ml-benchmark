# src/nhisml/evaluate.py
import argparse
import json
import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

from .preprocess import normalize_weights


def _infer_threshold_from_json(model_path: str, fallback: float = 0.5) -> float:
    """
    If there's a sibling thresholds_*.json in the same directory, load the one with the
    newest mtime and try to pick the matching key from its basename ('rf'/'l1').
    """
    d = os.path.dirname(model_path) or "."
    jsons = [os.path.join(d, f) for f in os.listdir(d) if f.startswith("thresholds_") and f.endswith(".json")]
    if not jsons:
        return fallback
    j = max(jsons, key=lambda p: os.path.getmtime(p))
    try:
        with open(j, "r") as f:
            obj = json.load(f)
        key = "rf" if "rf_pipeline" in os.path.basename(model_path) else "l1"
        return float(obj.get(key, fallback))
    except Exception:
        return fallback


def _add_missing_flags_if_needed(core: pd.DataFrame, expected_flag_cols: list) -> pd.DataFrame:
    """
    Backward-compat guard for old pipelines that expect __ismissing columns
    directly in the input. New pipelines generate them internally (PrepareFrame),
    so this function will be a no-op in that case.
    """
    if not expected_flag_cols:
        return core
    out = core.copy()
    for c in expected_flag_cols:
        if c not in out.columns:
            # create 0 filled flag (assumes flags are 0/1)
            out[c] = 0.0
    return out


def cli():
    p = argparse.ArgumentParser("nhis-evaluate")
    p.add_argument("--model", required=True, help="path to *_pipeline_*.joblib")
    p.add_argument("--test-core", default="data/processed/Adults24_core.parquet")
    p.add_argument("--threshold", type=float, default=None, help="Probability cutoff; if omitted, auto-load from thresholds_*.json")
    p.add_argument("--outdir", default="artifacts")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    pipe = joblib.load(args.model)
    core = pd.read_parquet(args.test_core)

    # y, weights
    y = (pd.to_numeric(core["PHSTAT_A"], errors="coerce") >= 4).astype(int)
    w = normalize_weights(core["WTFA_A"]) if "WTFA_A" in core.columns else np.ones(len(core))

    # threshold
    thr = args.threshold if args.threshold is not None else _infer_threshold_from_json(args.model, fallback=0.5)
    print(f"[evaluate] threshold={thr:.4f}", flush=True)

    # Try predict_proba directly. If a legacy artifact requires __ismissing columns, add them and retry.
    try:
        proba = pipe.predict_proba(core)[:, 1]
    except ValueError as e:
        # Heuristic: create flags for any ord/cat columns we find in the model's expected names.
        # If the pipeline has 'prepframe', it won't reach here; this is only for older CT-only models.
        expected_flags = []
        try:
            ct = None
            if hasattr(pipe, "named_steps") and "ct" in pipe.named_steps:
                ct = pipe.named_steps["ct"]
            elif hasattr(pipe, "named_steps") and "prep" in pipe.named_steps:
                ct = pipe.named_steps["prep"]  # just in case older name
            if ct is not None and hasattr(ct, "transformers_"):
                for name, trans, cols in ct.transformers_:
                    if isinstance(cols, list):
                        expected_flags.extend([c for c in cols if c.endswith("__ismissing")])
            if expected_flags:
                core2 = _add_missing_flags_if_needed(core, expected_flags)
                proba = pipe.predict_proba(core2)[:, 1]
                core = core2  # keep for consistency below
            else:
                raise
        except Exception:
            raise e

    pred = (proba >= thr).astype(int)

    metrics = dict(
        weighted_auc=float(roc_auc_score(y, proba, sample_weight=w)),
        weighted_accuracy=float(accuracy_score(y, pred, sample_weight=w)),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        weighted_f1=float(f1_score(y, pred, sample_weight=w)),
        avg_precision=float(average_precision_score(y, proba, sample_weight=w)),
    )

    pd.DataFrame([metrics]).to_csv(os.path.join(args.outdir, "test_metrics.csv"), index=False)
    pd.DataFrame({"prob": proba, "pred": pred, "srh_bin": y, "wt": w}).to_csv(
        os.path.join(args.outdir, "test_predictions.csv"), index=False
    )
    print("Saved artifacts/test_metrics.csv and artifacts/test_predictions.csv")


if __name__ == "__main__":
    cli()
