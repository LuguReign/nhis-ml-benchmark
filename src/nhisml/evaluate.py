from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)

from .preprocess import normalize_weights
from .tasks import make_task


def _load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _write_json(path: str, obj: dict) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _resolve_run_path(run: str) -> str:
    """
    Accept either a run directory or a manifest.json path.
    Return absolute path to manifest.json.
    """
    run = os.path.abspath(run)
    if os.path.isdir(run):
        m = os.path.join(run, "manifest.json")
        if not os.path.exists(m):
            raise FileNotFoundError(f"No manifest.json found in {run}")
        return m
    if run.endswith("manifest.json") and os.path.exists(run):
        return run
    raise FileNotFoundError(f"Run must be a run directory or a manifest.json path: {run}")


def _find_latest_manifest_for_task(runs_dir: str, task: str) -> str:
    """
    Find the most recent run (by manifest['created_at']) whose manifest['task'] == task.
    Returns absolute path to manifest.json.
    """
    runs_dir = os.path.abspath(runs_dir)
    if not os.path.isdir(runs_dir):
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    candidates = []
    for name in os.listdir(runs_dir):
        run_path = os.path.join(runs_dir, name)
        manifest_path = os.path.join(run_path, "manifest.json")
        if not os.path.isfile(manifest_path):
            continue
        try:
            m = _load_json(manifest_path)
            if m.get("task") == task:
                created_at = str(m.get("created_at", ""))
                candidates.append((created_at, manifest_path))
        except Exception:
            continue

    if not candidates:
        raise FileNotFoundError(f"No runs found for task='{task}' under {runs_dir}")

    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def _resolve_core_path(core_path: Optional[str], year: Optional[int], data_dir: str) -> str:
    if core_path:
        return os.path.abspath(core_path)
    if year is None:
        raise SystemExit("Provide --in CORE_PATH or --year YYYY")
    return os.path.abspath(os.path.join(data_dir, f"core_{int(year)}.parquet"))


def _weighted_binary_metrics(y: np.ndarray, p: np.ndarray, w: np.ndarray, thr: float) -> Dict[str, float]:
    pred = (p >= thr).astype(int)
    return {
        "weighted_auc": float(roc_auc_score(y, p, sample_weight=w)) if (y.sum() > 0 and (1 - y).sum() > 0) else float("nan"),
        "weighted_pr_auc": float(average_precision_score(y, p, sample_weight=w)) if (y.sum() > 0 and (1 - y).sum() > 0) else float("nan"),
        "weighted_log_loss": float(log_loss(y, p, sample_weight=w, labels=[0, 1])),
        "weighted_brier": float(brier_score_loss(y, p, sample_weight=w)),
        "weighted_f1": float(f1_score(y, pred, sample_weight=w)),
        "threshold": float(thr),
    }


def cli(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser("nhisml evaluate")

    # Two modes:
    #  A) explicit: --run <run_dir_or_manifest>
    #  B) latest:   --task <task> --latest [--runs-dir ...]
    p.add_argument("--run", default=None, help="Run directory or path to manifest.json")
    p.add_argument("--task", default=None, help="Task name (used with --latest)")
    p.add_argument("--latest", action="store_true", help="Use latest run for the given --task")
    p.add_argument("--runs-dir", default="runs", help="Base runs directory (default: runs/)")

    # core input: either --in or --year
    p.add_argument("--in", dest="core_path", default=None, help="Core parquet to evaluate on (optional if using --year)")
    p.add_argument("--year", type=int, default=None, help="Shortcut for --in <data-dir>/core_YYYY.parquet")
    p.add_argument("--data-dir", default="data", help="Base data directory for --year shortcut (default: data/)")

    p.add_argument("--out", default=None, help="Optional output dir (default: run dir)")
    p.add_argument("--threshold-key", default=None, help="Which threshold key to use (default: model name from manifest)")
    p.add_argument("--weight-col", default="WTFA_A", help="Weight column name in core parquet")
    args = p.parse_args(argv)

    # Resolve manifest
    if args.latest:
        if not args.task:
            raise SystemExit("--latest requires --task")
        if args.run:
            raise SystemExit("Use either --run OR (--task with --latest), not both")
        manifest_path = _find_latest_manifest_for_task(args.runs_dir, args.task)
    else:
        if not args.run:
            raise SystemExit("Provide --run or use --task with --latest")
        manifest_path = _resolve_run_path(args.run)

    run_dir = os.path.dirname(os.path.abspath(manifest_path))
    out_dir = os.path.abspath(args.out) if args.out else run_dir
    os.makedirs(out_dir, exist_ok=True)

    manifest = _load_json(manifest_path)
    task = make_task(manifest["task"])

    model_rel = manifest["artifacts"]["model"]
    model_path = os.path.join(run_dir, model_rel)
    model = joblib.load(model_path)

    thresholds_path = os.path.join(run_dir, manifest["artifacts"]["thresholds"])
    thresholds = _load_json(thresholds_path)

    # Decide threshold
    model_name = manifest["model"]
    thr_key = args.threshold_key or model_name
    thr = float(thresholds.get(thr_key, 0.5))

    # Resolve core path (supports --year)
    core_path = _resolve_core_path(args.core_path, args.year, args.data_dir)
    df = pd.read_parquet(core_path)

    # Labels + eligibility
    y, eligible = task.make_labels(df)
    if eligible is None:
        eligible = np.ones(len(df), dtype=bool)
    eligible = np.asarray(eligible, dtype=bool)

    df_eval = df.loc[eligible].reset_index(drop=True)
    y = np.asarray(y)[eligible].astype(int)

    # Weights
    if args.weight_col in df_eval.columns:
        w = normalize_weights(df_eval[args.weight_col])
        used_weights = True
    else:
        w = np.ones(len(df_eval), dtype=float)
        used_weights = False

    # Predict
    p1 = model.predict_proba(df_eval)[:, 1]

    # Metrics
    metrics = {
        "task": task.name,
        "model": model_name,
        "threshold_key": thr_key,
        "used_weights": bool(used_weights),
        "n_rows": int(len(df)),
        "n_eligible": int(len(df_eval)),
        "pos_rate_unweighted": float(y.mean()) if len(y) else float("nan"),
        "pos_rate_weighted": float(np.average(y, weights=w)) if len(y) else float("nan"),
        "core_path": core_path,
        "run_dir": run_dir,
    }
    metrics.update(_weighted_binary_metrics(y, p1, w, thr))

    # Save predictions
    pred_path = os.path.join(out_dir, f"predictions_task={task.name}.parquet")
    pd.DataFrame(
        {"y": y, "p": p1, "pred": (p1 >= thr).astype(int), "w": w}
    ).to_parquet(pred_path, index=False)

    # Save metrics json
    metrics_path = os.path.join(out_dir, f"metrics_task={task.name}.json")
    _write_json(metrics_path, metrics)

    # Display
    print(f"Evaluated run: {run_dir}")
    print(f"Task: {task.name} | Model: {model_name} | N eligible: {len(df_eval):,} | Weights: {'yes' if used_weights else 'no'}")
    print(f"AUC(w): {metrics['weighted_auc']:.4f} | PR-AUC(w): {metrics['weighted_pr_auc']:.4f} | Brier(w): {metrics['weighted_brier']:.4f}")
    print(f"F1(w)@{thr:.2f}: {metrics['weighted_f1']:.4f} | LogLoss(w): {metrics['weighted_log_loss']:.4f}")
    print(f"Wrote: {os.path.relpath(metrics_path, start=run_dir)} and {os.path.relpath(pred_path, start=run_dir)}")
