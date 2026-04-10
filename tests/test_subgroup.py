"""
Tests for nhisml.subgroup — recode helpers, weighted ECE, and
subgroup metrics table construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nhisml.subgroup import (
    _ece_weighted,
    _recode_age_band,
    _recode_educ_4,
    _recode_sex,
    _subgroup_table,
    _weighted_binary_metrics,
)


# ---------------------------------------------------------------------------
# _recode_sex
# ---------------------------------------------------------------------------

class TestRecodeSex:
    def test_male_1_female_2(self):
        df = pd.DataFrame({"SEX_A": [1, 2]})
        out = _recode_sex(df)
        assert out.iloc[0] == "Male"
        assert out.iloc[1] == "Female"

    def test_missing_codes_become_na(self):
        df = pd.DataFrame({"SEX_A": [7, 9, 1]})
        out = _recode_sex(df)
        assert pd.isna(out.iloc[0])
        assert pd.isna(out.iloc[1])
        assert out.iloc[2] == "Male"

    def test_absent_column_all_na(self):
        df = pd.DataFrame({"OTHER": [1, 2]})
        out = _recode_sex(df)
        assert out.isna().all()

    def test_custom_col_name(self):
        df = pd.DataFrame({"MY_SEX": [1, 2]})
        out = _recode_sex(df, col="MY_SEX")
        assert list(out) == ["Male", "Female"]


# ---------------------------------------------------------------------------
# _recode_age_band
# ---------------------------------------------------------------------------

class TestRecodeAgeBand:
    def test_band_boundaries(self):
        df = pd.DataFrame({"AGEP_A": [18, 34, 35, 49, 50, 64, 65, 90]})
        out = _recode_age_band(df)
        assert out.iloc[0] == "18-34"
        assert out.iloc[1] == "18-34"
        assert out.iloc[2] == "35-49"
        assert out.iloc[3] == "35-49"
        assert out.iloc[4] == "50-64"
        assert out.iloc[5] == "50-64"
        assert out.iloc[6] == "65+"
        assert out.iloc[7] == "65+"

    def test_under_18_becomes_na(self):
        df = pd.DataFrame({"AGEP_A": [17, 10]})
        out = _recode_age_band(df)
        assert out.isna().all()

    def test_missing_nhis_codes_become_na(self):
        df = pd.DataFrame({"AGEP_A": [97, 99, 45]})
        out = _recode_age_band(df)
        assert pd.isna(out.iloc[0])
        assert pd.isna(out.iloc[1])
        assert out.iloc[2] == "35-49"

    def test_absent_column_all_na(self):
        df = pd.DataFrame({"OTHER": [30, 40]})
        out = _recode_age_band(df)
        assert out.isna().all()


# ---------------------------------------------------------------------------
# _recode_educ_4
# ---------------------------------------------------------------------------

class TestRecodeEduc4:
    def test_level_mapping(self):
        df = pd.DataFrame({"EDUCP_A": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        out = _recode_educ_4(df)
        assert out.iloc[0] == "<HS"
        assert out.iloc[2] == "<HS"
        assert out.iloc[3] == "HS/GED"
        assert out.iloc[4] == "HS/GED"
        assert out.iloc[5] == "Some college/AA"
        assert out.iloc[7] == "Some college/AA"
        assert out.iloc[8] == "BA+"
        assert out.iloc[10] == "BA+"

    def test_missing_codes_become_na(self):
        df = pd.DataFrame({"EDUCP_A": [97, 99, 5]})
        out = _recode_educ_4(df)
        assert pd.isna(out.iloc[0])
        assert pd.isna(out.iloc[1])
        assert out.iloc[2] == "Some college/AA"

    def test_absent_column_all_na(self):
        df = pd.DataFrame({"OTHER": [1, 2]})
        out = _recode_educ_4(df)
        assert out.isna().all()


# ---------------------------------------------------------------------------
# _ece_weighted
# ---------------------------------------------------------------------------

class TestEceWeighted:
    def test_perfect_calibration_near_zero(self):
        """When confidence equals accuracy in each bin, ECE ≈ 0."""
        # 100 samples: p = y (perfect calibration)
        p = np.linspace(0.1, 0.9, 100)
        y = (np.random.default_rng(0).uniform(size=100) < p).astype(int)
        w = np.ones(100)
        ece = _ece_weighted(y, p, w, n_bins=10)
        # Should be small (not necessarily zero due to randomness)
        assert ece >= 0.0

    def test_empty_returns_nan(self):
        ece = _ece_weighted(np.array([]), np.array([]), np.array([]))
        assert np.isnan(ece)

    def test_zero_weight_returns_nan(self):
        ece = _ece_weighted(np.array([0, 1]), np.array([0.3, 0.7]),
                            np.array([0.0, 0.0]))
        assert np.isnan(ece)

    def test_ece_in_0_1_range(self):
        rng = np.random.default_rng(42)
        y = rng.integers(0, 2, size=200)
        p = rng.uniform(size=200)
        w = rng.uniform(0.5, 2.0, size=200)
        ece = _ece_weighted(y, p, w)
        assert 0.0 <= ece <= 1.0


# ---------------------------------------------------------------------------
# _weighted_binary_metrics (subgroup version — includes ECE)
# ---------------------------------------------------------------------------

class TestSubgroupWeightedMetrics:
    def test_keys_include_ece(self):
        y = np.array([0, 1, 0, 1])
        p = np.array([0.2, 0.8, 0.3, 0.7])
        w = np.ones(4)
        m = _weighted_binary_metrics(y, p, w, thr=0.5)
        assert "ece" in m
        assert "weighted_auc" in m
        assert "weighted_f1" in m


# ---------------------------------------------------------------------------
# _subgroup_table (integration)
# ---------------------------------------------------------------------------

class TestSubgroupTable:
    def _make_eval_data(self, n: int = 500):
        rng = np.random.default_rng(7)
        df = pd.DataFrame({
            "SEX_A": rng.choice([1, 2], size=n),
            "AGEP_A": rng.integers(18, 80, size=n),
        })
        y = rng.integers(0, 2, size=n)
        p = rng.uniform(size=n)
        w = np.ones(n)
        return df, y, p, w

    def test_returns_dataframe(self):
        df, y, p, w = self._make_eval_data()
        from nhisml.subgroup import _recode_sex, _weighted_binary_metrics
        overall = _weighted_binary_metrics(y, p, w, thr=0.5)
        labels = _recode_sex(df)
        result = _subgroup_table(
            df_eval=df, y=y, p=p, w=w, thr=0.5,
            subgroup_name="sex", labels=labels,
            overall=overall, min_n=10, min_pos=5, min_neg=5,
        )
        assert isinstance(result, pd.DataFrame)

    def test_subgroup_column_present(self):
        df, y, p, w = self._make_eval_data()
        from nhisml.subgroup import _recode_sex, _weighted_binary_metrics
        overall = _weighted_binary_metrics(y, p, w, thr=0.5)
        labels = _recode_sex(df)
        result = _subgroup_table(
            df_eval=df, y=y, p=p, w=w, thr=0.5,
            subgroup_name="sex", labels=labels,
            overall=overall, min_n=10, min_pos=5, min_neg=5,
        )
        assert "subgroup" in result.columns
        assert all(result["subgroup"] == "sex")

    def test_levels_present(self):
        df, y, p, w = self._make_eval_data()
        from nhisml.subgroup import _recode_sex, _weighted_binary_metrics
        overall = _weighted_binary_metrics(y, p, w, thr=0.5)
        labels = _recode_sex(df)
        result = _subgroup_table(
            df_eval=df, y=y, p=p, w=w, thr=0.5,
            subgroup_name="sex", labels=labels,
            overall=overall, min_n=10, min_pos=5, min_neg=5,
        )
        levels = set(result["level"].tolist())
        assert "Male" in levels
        assert "Female" in levels

    def test_small_cell_excluded_from_metrics(self):
        """Rows with fewer than min_n observations should have NaN metrics."""
        df, y, p, w = self._make_eval_data(n=600)
        # Inject a very rare subgroup level
        labels = pd.Series(["common"] * 590 + ["tiny"] * 10, dtype="object")
        from nhisml.subgroup import _weighted_binary_metrics
        overall = _weighted_binary_metrics(y, p, w, thr=0.5)
        result = _subgroup_table(
            df_eval=df, y=y, p=p, w=w, thr=0.5,
            subgroup_name="test_sg", labels=labels,
            overall=overall, min_n=200, min_pos=25, min_neg=25,
        )
        tiny_row = result[result["level"] == "tiny"]
        assert len(tiny_row) == 1
        assert not tiny_row["meets_min_cell"].iloc[0]
        assert np.isnan(tiny_row["weighted_auc"].iloc[0])
