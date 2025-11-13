# src/nhisml/utils.py
from __future__ import annotations

import os
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold


def _route_fit_weights_kwargs(step_name: Optional[str], w: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Build the kwargs that route sample_weight to an estimator inside a Pipeline.
    - If step_name is provided (e.g., 'rf' or 'lasso'), returns {f'{step_name}__sample_weight': w}
    - Otherwise returns {'sample_weight': w}
    """
    if step_name:
        return {f"{step_name}__sample_weight": w}
    return {"sample_weight": w}


def oof_proba(
    pipe,
    X: pd.DataFrame,
    y: np.ndarray,
    w: np.ndarray,
    step_name: Optional[str] = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """
    Compute out-of-fold predicted probabilities for class 1 with weighted fits.
    Returns a 1-D array (len(X),).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros_like(y, dtype=float)

    for tr_idx, va_idx in skf.split(X, y):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, wtr = y[tr_idx], w[tr_idx]

        fold_model = clone(pipe)
        fit_kwargs = _route_fit_weights_kwargs(step_name, wtr)
        fold_model.fit(Xtr, ytr, **fit_kwargs)

        oof[va_idx] = fold_model.predict_proba(Xva)[:, 1]

    return oof


def _pick_best_threshold(probs: np.ndarray, y: np.ndarray, w: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """
    Sweep thresholds and choose the best by weighted F1. Also compute AUC/AP
    on the OOF probabilities for reference.
    """
    thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    best_thr, best_f1 = 0.5, -1.0

    for thr in thresholds:
        pred = (probs >= thr).astype(int)
        f1w = f1_score(y, pred, sample_weight=w)
        if f1w > best_f1:
            best_thr, best_f1 = float(thr), float(f1w)

    perf = {
        "oof_weighted_auc": float(roc_auc_score(y, probs, sample_weight=w)),
        "oof_avg_precision": float(average_precision_score(y, probs, sample_weight=w)),
        "oof_best_weighted_f1": float(best_f1),
        "oof_best_threshold": float(best_thr),
    }
    return best_thr, perf


def weighted_threshold_via_oof(
    pipe,
    X: pd.DataFrame,
    y: np.ndarray,
    w: np.ndarray,
    step_name: Optional[str] = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[float, Dict[str, float]]:
    """
    Get OOF probabilities (with weighted fits), then choose a probability
    threshold that maximizes weighted F1.
    Returns (best_threshold, perf_dict).
    """
    probs = oof_proba(pipe, X, y, w, step_name=step_name, n_splits=n_splits, random_state=random_state)
    return _pick_best_threshold(probs, y, w)


def fit_calibrated_from_oof(
    pipe,
    X: pd.DataFrame,
    y: np.ndarray,
    w: np.ndarray,
    step_name: Optional[str] = None,
    method: str = "isotonic",
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[CalibratedClassifierCV, float, Dict[str, float]]:
    """
    Fit a calibrated version using scikit-learn >= 1.4 API:
      - First, obtain OOF calibrated probabilities for threshold selection
        while preserving *weighted* base fits per fold.
        We do this by:
          1) Fit base (weighted) on the training fold.
          2) Wrap it with CalibratedClassifierCV(estimator=base, cv="prefit", method=method).
          3) Fit the calibrator on the same training fold (sklearn won't
             route sample_weight internally), and predict on validation fold.
      - Next, fit a final calibrated model on ALL data:
          a) Fit base (weighted) on all data.
          b) Wrap with CalibratedClassifierCV(cv="prefit") and fit on all data.

    Returns: (calibrated_model, best_threshold_on_oof_calibrated, perf_dict)
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_cal = np.zeros_like(y, dtype=float)

    # OOF calibrated probabilities
    for tr_idx, va_idx in skf.split(X, y):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, wtr = y[tr_idx], w[tr_idx]

        base = clone(pipe)
        fit_kwargs = _route_fit_weights_kwargs(step_name, wtr)
        base.fit(Xtr, ytr, **fit_kwargs)

        # Calibrate a *prefit* base on the same training fold
        cal = CalibratedClassifierCV(estimator=base, cv="prefit", method=method)
        cal.fit(Xtr, ytr)  # sklearn does not accept sample_weight here

        oof_cal[va_idx] = cal.predict_proba(Xva)[:, 1]

    # Pick threshold on calibrated OOF probs
    best_thr, perf = _pick_best_threshold(oof_cal, y, w)
    # Rename keys to clarify these are calibrated-oof metrics
    perf = {
        "oof_cal_weighted_auc": perf["oof_weighted_auc"],
        "oof_cal_avg_precision": perf["oof_avg_precision"],
        "oof_cal_best_weighted_f1": perf["oof_best_weighted_f1"],
        "oof_cal_best_threshold": best_thr,
    }

    # Final calibrated model on ALL data
    base_full = clone(pipe)
    fit_kwargs_full = _route_fit_weights_kwargs(step_name, w)
    base_full.fit(X, y, **fit_kwargs_full)

    cal_full = CalibratedClassifierCV(estimator=base_full, cv="prefit", method=method)
    cal_full.fit(X, y)  # sample_weight not used internally

    return cal_full, best_thr, perf


def save_metrics_csv(path: str, metrics: Dict[str, float]) -> None:
    """
    Save a single-row CSV of metrics.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame([metrics]).to_csv(path, index=False)
