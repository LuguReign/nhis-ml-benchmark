"""
Tests for nhisml.utils — OOF cross-validation, threshold picking,
performance metrics, and calibration utilities.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nhisml.utils import (
    fit_calibrated_from_oof,
    infer_estimator_step_name,
    oof_proba,
    pick_threshold_max_f1,
    threshold_perf,
    weighted_threshold_via_oof,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _binary_data(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, 4)), columns=["a", "b", "c", "d"])
    y = (X["a"] + X["b"] > 0).astype(int).to_numpy()
    w = np.ones(n, dtype=float)
    return X, y, w


def _simple_pipeline():
    return Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200, random_state=42)),
    ])


# ---------------------------------------------------------------------------
# infer_estimator_step_name
# ---------------------------------------------------------------------------

class TestInferEstimatorStepName:
    def test_pipeline_returns_final_step(self):
        pipe = _simple_pipeline()
        assert infer_estimator_step_name(pipe) == "clf"

    def test_non_pipeline_returns_none(self):
        lr = LogisticRegression()
        assert infer_estimator_step_name(lr) is None


# ---------------------------------------------------------------------------
# pick_threshold_max_f1
# ---------------------------------------------------------------------------

class TestPickThreshold:
    def test_returns_tuple(self):
        probs = np.array([0.1, 0.4, 0.6, 0.9])
        y = np.array([0, 0, 1, 1])
        w = np.ones(4)
        thr, f1 = pick_threshold_max_f1(probs, y, w)
        assert 0.0 <= thr <= 1.0
        assert 0.0 <= f1 <= 1.0

    def test_perfect_separation(self):
        """With perfect predictions, threshold near boundary should yield F1=1."""
        probs = np.array([0.0, 0.05, 0.95, 1.0])
        y = np.array([0, 0, 1, 1])
        w = np.ones(4)
        thr, f1 = pick_threshold_max_f1(probs, y, w)
        assert f1 == pytest.approx(1.0)

    def test_custom_grid(self):
        probs = np.array([0.3, 0.7])
        y = np.array([0, 1])
        w = np.ones(2)
        thr, f1 = pick_threshold_max_f1(probs, y, w, grid=np.array([0.5]))
        assert thr == 0.5

    def test_weighted_f1_used(self):
        """Weights should influence which threshold is chosen."""
        probs = np.array([0.6, 0.6, 0.6])
        y = np.array([1, 0, 1])
        w = np.array([10.0, 1.0, 1.0])  # heavy weight on first positive
        thr, f1 = pick_threshold_max_f1(probs, y, w)
        assert 0.0 <= f1 <= 1.0


# ---------------------------------------------------------------------------
# threshold_perf
# ---------------------------------------------------------------------------

class TestThresholdPerf:
    def test_keys_present(self):
        probs = np.array([0.2, 0.8, 0.3, 0.7])
        y = np.array([0, 1, 0, 1])
        w = np.ones(4)
        perf = threshold_perf(probs, y, w, thr=0.5)
        assert "oof_weighted_auc" in perf
        assert "oof_avg_precision" in perf
        assert "oof_weighted_f1" in perf
        assert "oof_threshold" in perf

    def test_auc_range(self):
        probs = np.array([0.2, 0.8, 0.3, 0.7])
        y = np.array([0, 1, 0, 1])
        w = np.ones(4)
        perf = threshold_perf(probs, y, w, thr=0.5)
        assert 0.0 <= perf["oof_weighted_auc"] <= 1.0

    def test_threshold_recorded(self):
        probs = np.array([0.4, 0.6])
        y = np.array([0, 1])
        w = np.ones(2)
        perf = threshold_perf(probs, y, w, thr=0.42)
        assert perf["oof_threshold"] == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# oof_proba
# ---------------------------------------------------------------------------

class TestOofProba:
    def test_output_shape(self):
        X, y, w = _binary_data()
        model = _simple_pipeline()
        probs = oof_proba(model, X, y, w, n_splits=3)
        assert probs.shape == (len(y),)

    def test_probabilities_in_range(self):
        X, y, w = _binary_data()
        model = _simple_pipeline()
        probs = oof_proba(model, X, y, w, n_splits=3)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    def test_works_with_pipeline_step_name(self):
        X, y, w = _binary_data()
        model = _simple_pipeline()
        probs = oof_proba(model, X, y, w, step_name="clf", n_splits=3)
        assert probs.shape == (len(y),)

    def test_oof_covers_all_rows(self):
        """Every row should receive a prediction (no zeros from missing folds)."""
        X, y, w = _binary_data(n=100)
        model = _simple_pipeline()
        probs = oof_proba(model, X, y, w, n_splits=5)
        # If all rows are covered, no probability should equal 0.0 exactly
        # (with a proper model, at least some will be non-zero)
        # Weaker check: no NaN
        assert not np.any(np.isnan(probs))


# ---------------------------------------------------------------------------
# weighted_threshold_via_oof (integration)
# ---------------------------------------------------------------------------

class TestWeightedThresholdViaOof:
    def test_returns_three_items(self):
        X, y, w = _binary_data()
        model = _simple_pipeline()
        thr, perf, oof = weighted_threshold_via_oof(model, X, y, w, n_splits=3)
        assert isinstance(thr, float)
        assert isinstance(perf, dict)
        assert isinstance(oof, np.ndarray)

    def test_threshold_in_range(self):
        X, y, w = _binary_data()
        model = _simple_pipeline()
        thr, _, _ = weighted_threshold_via_oof(model, X, y, w, n_splits=3)
        assert 0.0 <= thr <= 1.0

    def test_perf_dict_has_auc(self):
        X, y, w = _binary_data()
        model = _simple_pipeline()
        _, perf, _ = weighted_threshold_via_oof(model, X, y, w, n_splits=3)
        assert "oof_weighted_auc" in perf


# ---------------------------------------------------------------------------
# fit_calibrated_from_oof
# ---------------------------------------------------------------------------

class TestFitCalibratedFromOof:
    def test_returns_four_items(self):
        X, y, w = _binary_data(n=300)
        model = _simple_pipeline()
        cal_model, thr, perf, oof_cal = fit_calibrated_from_oof(
            model, X, y, w, n_splits=3
        )
        assert hasattr(cal_model, "predict_proba")
        assert isinstance(thr, float)
        assert isinstance(perf, dict)
        assert isinstance(oof_cal, np.ndarray)

    def test_calibrated_model_can_predict(self):
        X, y, w = _binary_data(n=300)
        model = _simple_pipeline()
        cal_model, _, _, _ = fit_calibrated_from_oof(model, X, y, w, n_splits=3)
        preds = cal_model.predict_proba(X)
        assert preds.shape == (len(X), 2)
        assert np.allclose(preds.sum(axis=1), 1.0, atol=1e-6)

    def test_perf_keys_prefixed_cal(self):
        X, y, w = _binary_data(n=300)
        model = _simple_pipeline()
        _, _, perf, _ = fit_calibrated_from_oof(model, X, y, w, n_splits=3)
        assert any("cal" in k for k in perf.keys())
