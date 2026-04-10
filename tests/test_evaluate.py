"""
Tests for nhisml.evaluate — weighted metrics, run resolution helpers,
and end-to-end evaluation on synthetic data.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nhisml.evaluate import (
    _find_latest_manifest_for_task,
    _resolve_core_path,
    _resolve_run_path,
    _weighted_binary_metrics,
)


# ---------------------------------------------------------------------------
# _weighted_binary_metrics
# ---------------------------------------------------------------------------

class TestWeightedBinaryMetrics:
    def _call(self, y, p, w=None, thr=0.5):
        if w is None:
            w = np.ones(len(y))
        return _weighted_binary_metrics(np.array(y), np.array(p, dtype=float),
                                        np.array(w, dtype=float), thr)

    def test_keys_present(self):
        m = self._call([0, 1, 0, 1], [0.2, 0.8, 0.3, 0.7])
        for key in ("weighted_auc", "weighted_pr_auc", "weighted_log_loss",
                    "weighted_brier", "weighted_f1", "threshold"):
            assert key in m, f"Missing key: {key}"

    def test_auc_perfect(self):
        m = self._call([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        assert m["weighted_auc"] == pytest.approx(1.0)

    def test_auc_random(self):
        m = self._call([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])
        assert 0.0 <= m["weighted_auc"] <= 1.0

    def test_nan_when_single_class(self):
        """AUC and PR-AUC are NaN when only one class is present."""
        m = self._call([1, 1, 1], [0.7, 0.8, 0.9])
        assert np.isnan(m["weighted_auc"])
        assert np.isnan(m["weighted_pr_auc"])

    def test_f1_at_threshold(self):
        m = self._call([0, 1, 0, 1], [0.2, 0.8, 0.3, 0.7], thr=0.5)
        assert m["weighted_f1"] == pytest.approx(1.0)

    def test_brier_range(self):
        m = self._call([0, 1, 0, 1], [0.2, 0.8, 0.3, 0.7])
        assert 0.0 <= m["weighted_brier"] <= 1.0

    def test_weights_affect_metrics(self):
        y = [0, 1]
        p = [0.3, 0.7]
        w_equal = [1.0, 1.0]
        w_skewed = [0.01, 100.0]
        m1 = self._call(y, p, w_equal)
        m2 = self._call(y, p, w_skewed)
        # With very different weights, metrics should differ
        assert m1["weighted_log_loss"] != m2["weighted_log_loss"]

    def test_threshold_recorded(self):
        m = self._call([0, 1], [0.3, 0.7], thr=0.42)
        assert m["threshold"] == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# _resolve_run_path
# ---------------------------------------------------------------------------

class TestResolveRunPath:
    def test_directory_with_manifest(self, tmp_path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"task": "srh_binary"}))
        result = _resolve_run_path(str(tmp_path))
        assert result == str(manifest.resolve())

    def test_directory_without_manifest_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="manifest.json"):
            _resolve_run_path(str(tmp_path))

    def test_direct_manifest_path(self, tmp_path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text("{}")
        result = _resolve_run_path(str(manifest))
        assert result == str(manifest.resolve())

    def test_nonexistent_path_raises(self):
        with pytest.raises(FileNotFoundError):
            _resolve_run_path("/nonexistent/path/manifest.json")


# ---------------------------------------------------------------------------
# _find_latest_manifest_for_task
# ---------------------------------------------------------------------------

class TestFindLatestManifest:
    def _make_run(self, base: Path, run_name: str, task: str, created_at: str) -> None:
        d = base / run_name
        d.mkdir()
        (d / "manifest.json").write_text(json.dumps({
            "task": task,
            "created_at": created_at,
        }))

    def test_finds_correct_task(self, tmp_path):
        self._make_run(tmp_path, "run_a", "srh_binary", "2024-01-01T10:00:00")
        self._make_run(tmp_path, "run_b", "smoking_current", "2024-01-02T10:00:00")
        result = _find_latest_manifest_for_task(str(tmp_path), "smoking_current")
        assert "run_b" in result

    def test_finds_latest_among_same_task(self, tmp_path):
        self._make_run(tmp_path, "run_old", "srh_binary", "2024-01-01T08:00:00")
        self._make_run(tmp_path, "run_new", "srh_binary", "2024-06-15T12:00:00")
        result = _find_latest_manifest_for_task(str(tmp_path), "srh_binary")
        assert "run_new" in result

    def test_missing_task_raises(self, tmp_path):
        self._make_run(tmp_path, "run_a", "srh_binary", "2024-01-01T00:00:00")
        with pytest.raises(FileNotFoundError, match="No runs found"):
            _find_latest_manifest_for_task(str(tmp_path), "nonexistent_task")

    def test_nonexistent_runs_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            _find_latest_manifest_for_task("/nonexistent/runs_dir", "srh_binary")

    def test_skips_dirs_without_manifest(self, tmp_path):
        empty_dir = tmp_path / "empty_run"
        empty_dir.mkdir()
        self._make_run(tmp_path, "real_run", "srh_binary", "2024-01-01T00:00:00")
        result = _find_latest_manifest_for_task(str(tmp_path), "srh_binary")
        assert "real_run" in result


# ---------------------------------------------------------------------------
# _resolve_core_path
# ---------------------------------------------------------------------------

class TestResolveCorePathFn:
    def test_explicit_path(self):
        result = _resolve_core_path("/some/path/core.parquet", None, "data")
        assert result == os.path.abspath("/some/path/core.parquet")

    def test_year_builds_path(self):
        result = _resolve_core_path(None, 2023, "data")
        assert result.endswith("core_2023.parquet")

    def test_neither_raises(self):
        with pytest.raises(SystemExit):
            _resolve_core_path(None, None, "data")
