from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


def infer_estimator_step_name(model) -> Optional[str]:
    """
    If `model` is a sklearn Pipeline, return the final step name.
    Otherwise return None (caller can pass sample_weight directly).
    """
    if isinstance(model, Pipeline) and len(model.steps) > 0:
        return model.steps[-1][0]
    return None


def _route_fit_weights_kwargs(model, w: np.ndarray, step_name: Optional[str] = None) -> Dict[str, np.ndarray]:
    """
    Route sample_weight to the estimator.
    If step_name is None and model is Pipeline, infer final step name.
    """
    if step_name is None:
        step_name = infer_estimator_step_name(model)
    if step_name:
        return {f"{step_name}__sample_weight": w}
    return {"sample_weight": w}


def oof_proba(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    w: np.ndarray,
    step_name: Optional[str] = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """
    Out-of-fold predicted probabilities for class 1.
    Fits each fold with sample weights (routed to the estimator).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros(len(y), dtype=float)

    for tr_idx, va_idx in skf.split(X, y):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, wtr = y[tr_idx], w[tr_idx]

        fold_model = clone(model)
        fold_model.fit(Xtr, ytr, **_route_fit_weights_kwargs(fold_model, wtr, step_name=step_name))
        oof[va_idx] = fold_model.predict_proba(Xva)[:, 1]

    return oof


def pick_threshold_max_f1(
    probs: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    grid: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Return (best_threshold, best_weighted_f1).
    """
    if grid is None:
        grid = np.round(np.arange(0.05, 0.96, 0.05), 2)

    best_thr, best_f1 = 0.5, -1.0
    for thr in grid:
        pred = (probs >= thr).astype(int)
        f1w = f1_score(y, pred, sample_weight=w)
        if f1w > best_f1:
            best_thr, best_f1 = float(thr), float(f1w)
    return best_thr, float(best_f1)


def threshold_perf(probs: np.ndarray, y: np.ndarray, w: np.ndarray, thr: float) -> Dict[str, float]:
    """
    A small, consistent perf dict for OOF probabilities.
    """
    pred = (probs >= thr).astype(int)
    return {
        "oof_weighted_auc": float(roc_auc_score(y, probs, sample_weight=w)),
        "oof_avg_precision": float(average_precision_score(y, probs, sample_weight=w)),
        "oof_weighted_f1": float(f1_score(y, pred, sample_weight=w)),
        "oof_threshold": float(thr),
    }


def weighted_threshold_via_oof(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    w: np.ndarray,
    step_name: Optional[str] = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[float, Dict[str, float], np.ndarray]:
    """
    Compute weighted-fit OOF probabilities, then choose threshold (max weighted F1).
    Returns (best_threshold, perf_dict, oof_probs).
    """
    probs = oof_proba(model, X, y, w, step_name=step_name, n_splits=n_splits, random_state=random_state)
    thr, best_f1 = pick_threshold_max_f1(probs, y, w)
    perf = threshold_perf(probs, y, w, thr)
    perf["oof_best_weighted_f1"] = float(best_f1)
    return thr, perf, probs


def _calibrator_fit(cal: CalibratedClassifierCV, X, y, w: np.ndarray) -> None:
    """
    Fit calibrator with weights if supported by sklearn version, else unweighted.
    """
    try:
        cal.fit(X, y, sample_weight=w)
    except TypeError:
        cal.fit(X, y)


def fit_calibrated_from_oof(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    w: np.ndarray,
    step_name: Optional[str] = None,
    method: str = "isotonic",
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[CalibratedClassifierCV, float, Dict[str, float], np.ndarray]:
    """
    Returns:
      (calibrated_model_fit_on_all, best_threshold_on_oof_cal, perf_dict, oof_cal_probs)
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_cal = np.zeros(len(y), dtype=float)

    for tr_idx, va_idx in skf.split(X, y):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, wtr = y[tr_idx], w[tr_idx]

        base = clone(model)
        base.fit(Xtr, ytr, **_route_fit_weights_kwargs(base, wtr, step_name=step_name))

        cal = CalibratedClassifierCV(estimator=base, cv="prefit", method=method)
        _calibrator_fit(cal, Xtr, ytr, wtr)
        oof_cal[va_idx] = cal.predict_proba(Xva)[:, 1]

    thr, best_f1 = pick_threshold_max_f1(oof_cal, y, w)
    perf = threshold_perf(oof_cal, y, w, thr)
    perf = {
        "oof_cal_weighted_auc": perf["oof_weighted_auc"],
        "oof_cal_avg_precision": perf["oof_avg_precision"],
        "oof_cal_best_weighted_f1": float(best_f1),
        "oof_cal_best_threshold": float(thr),
    }

    base_full = clone(model)
    base_full.fit(X, y, **_route_fit_weights_kwargs(base_full, w, step_name=step_name))

    cal_full = CalibratedClassifierCV(estimator=base_full, cv="prefit", method=method)
    _calibrator_fit(cal_full, X, y, w)

    return cal_full, float(thr), perf, oof_cal


def save_metrics_csv(path: str, metrics: Dict[str, float]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame([metrics]).to_csv(path, index=False)
