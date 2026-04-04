from __future__ import annotations

import warnings
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    module="sklearn.utils.extmath",
)

import argparse
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .featuresets import get_featureset
from .preprocess import build_preprocessor, normalize_weights
from .tasks import make_task
from .utils import fit_calibrated_from_oof, weighted_threshold_via_oof


@dataclass
class RunManifest:
    run_id: str
    created_at: str
    task: str
    model: str
    featureset: str
    input_core: str
    n_rows: int
    n_eligible: int
    weight_col: str
    used_weights: bool
    artifacts: Dict[str, Optional[str]]
    versions: Dict[str, str]


def _now_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _version_info() -> Dict[str, str]:
    out = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import sklearn  # type: ignore

        out["scikit-learn"] = sklearn.__version__
    except Exception:
        pass
    try:
        import pandas  # type: ignore

        out["pandas"] = pandas.__version__
    except Exception:
        pass
    try:
        import numpy  # type: ignore

        out["numpy"] = numpy.__version__
    except Exception:
        pass
    return out


def _make_estimator(model_name: str) -> Tuple[Any, str]:
    """
    Returns (estimator, step_name).

    Note: We intentionally use elastic-net logistic regression as the default
    "lasso-like" baseline because pure L1 (saga) can emit numeric warnings on
    sparse/high-cardinality designs. Elastic-net is typically more stable
    while still encouraging sparsity.
    """
    m = model_name.lower().strip()

    if m in {"lasso", "l1", "logreg_l1"}:
        # L1-like baseline (more stable than pure L1)
        est = LogisticRegression(
            penalty="elasticnet",
            l1_ratio=0.5,
            solver="saga",
            C=0.5,
            max_iter=5000,
            n_jobs=-1,
            random_state=42,
        )
        return est, "lasso"

    if m in {"rf", "random_forest"}:
        est = RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        )
        return est, "rf"

    raise ValueError(f"Unknown model: {model_name}. Use 'lasso' or 'rf'.")


def cli(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser("nhisml train")
    p.add_argument("--in", dest="core_path", required=True, help="Path to core parquet (e.g., data/core_2023.parquet)")
    p.add_argument("--task", default="srh_binary", help="Task name (srh_binary, smoking_current)")
    p.add_argument("--featureset", default="core", help="Featureset name")
    p.add_argument("--model", default="lasso", help="Model: lasso | rf")
    p.add_argument("--run-dir", default="runs", help="Base directory for run outputs")
    p.add_argument("--calibrate", action="store_true", help="Fit calibrated variant using OOF predictions")
    p.add_argument("--weight-col", default="WTFA_A", help="Weight column name in core parquet")
    p.add_argument("--rare-min-count", type=int, default=50, help="Rare category minimum count for bucketing")
    args = p.parse_args(argv)

    task = make_task(args.task)
    feats = get_featureset(args.featureset)

    df = pd.read_parquet(args.core_path)
    n_rows = int(len(df))

    # Columns: predictors + task label columns + weight column
    required = set(feats.all_columns)
    required |= set(task.required_columns())
    required.add(args.weight_col)

    keep = [c for c in df.columns if c in required]
    core = df[keep].copy()

    # Labels + eligibility mask are task-defined
    y, eligible = task.make_labels(core)
    if eligible is None:
        eligible = np.ones(len(core), dtype=bool)
    eligible = np.asarray(eligible, dtype=bool)

    core = core.loc[eligible].reset_index(drop=True)
    y = np.asarray(y)[eligible].astype(int)

    if len(core) == 0:
        raise ValueError(f"No eligible rows for task '{task.name}'. Check label coding and required columns.")

    # Weights (default: WTFA_A; fallback: uniform)
    used_weights = args.weight_col in core.columns
    if used_weights:
        w = normalize_weights(core[args.weight_col])
    else:
        w = np.ones(len(core), dtype=float)

    # UNFITTED preprocessor
    preproc = build_preprocessor(
        binary_cols=[c for c in feats.binary_12 if c in core.columns],
        ordinal_cols=[c for c in feats.ordinal if c in core.columns],
        categorical_cols=[c for c in feats.categorical if c in core.columns],
        rare_min_count=int(args.rare_min_count),
    )

    estimator, step_name = _make_estimator(args.model)
    pipe = Pipeline([("prep", preproc), (step_name, estimator)])

    # Fit (with weights routed to the estimator step)
    fit_kwargs: Dict[str, Any] = {f"{step_name}__sample_weight": w}
    pipe.fit(core, y, **fit_kwargs)

    # Threshold tuning via weighted OOF
    thr, perf, oof = weighted_threshold_via_oof(pipe, core, y, w, step_name=step_name)

    run_id = _now_id()
    run_name = f"{run_id}_task={task.name}_model={step_name}_fs={args.featureset}"
    outdir = os.path.join(args.run_dir, run_name)
    _ensure_dir(outdir)

    # Save model
    model_path = os.path.join(outdir, "model.joblib")
    joblib.dump(pipe, model_path)

    # Save thresholds
    thresholds = {step_name: float(thr)}
    cal_path: Optional[str] = None

    # Save OOF artifacts
    oof_path = os.path.join(outdir, "oof_predictions.parquet")
    pd.DataFrame({"y": y, "p": oof, "w": w}).to_parquet(oof_path, index=False)

    oof_metrics_path = os.path.join(outdir, "oof_metrics.json")
    with open(oof_metrics_path, "w") as f:
        json.dump(perf, f, indent=2)

    if args.calibrate:
        cal_model, cal_thr, cal_perf, _oof_cal = fit_calibrated_from_oof(
            pipe, core, y, w, step_name=step_name
        )
        cal_path = os.path.join(outdir, "model_calibrated.joblib")
        joblib.dump(cal_model, cal_path)
        thresholds[f"{step_name}_cal"] = float(cal_thr)

        cal_metrics_path = os.path.join(outdir, "oof_cal_metrics.json")
        with open(cal_metrics_path, "w") as f:
            json.dump(cal_perf, f, indent=2)

    thresholds_path = os.path.join(outdir, "thresholds.json")
    with open(thresholds_path, "w") as f:
        json.dump(thresholds, f, indent=2)

    manifest = RunManifest(
        run_id=run_id,
        created_at=datetime.now().isoformat(timespec="seconds"),
        task=task.name,
        model=step_name,
        featureset=args.featureset,
        input_core=os.path.abspath(args.core_path),
        n_rows=int(n_rows),
        n_eligible=int(len(core)),
        weight_col=args.weight_col,
        used_weights=bool(used_weights),
        artifacts={
            "model": "model.joblib",
            "model_calibrated": ("model_calibrated.joblib" if cal_path else None),
            "thresholds": "thresholds.json",
            "oof_predictions": "oof_predictions.parquet",
            "oof_metrics": "oof_metrics.json",
            "oof_cal_metrics": ("oof_cal_metrics.json" if args.calibrate else None),
        },
        versions=_version_info(),
    )

    manifest_path = os.path.join(outdir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(asdict(manifest), f, indent=2)

    # Simple automatic display
    print(f"Run: {outdir}")
    print(f"Task: {task.name} | Model: {step_name} | Featureset: {args.featureset}")
    print(f"N eligible: {len(core):,} | Weights: {'yes' if used_weights else 'no'}")
    print(f"Threshold ({step_name}): {thr:.4f}")
    if args.calibrate and f"{step_name}_cal" in thresholds:
        print(f"Threshold ({step_name}_cal): {thresholds[f'{step_name}_cal']:.4f}")
    print("Artifacts: manifest.json, model.joblib, thresholds.json, oof_predictions.parquet")
