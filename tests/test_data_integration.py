"""
Data integration tests for nhisml.

These tests verify that the processed NHIS core parquets match known
reference statistics derived from the public-use NHIS Adults files.
They are automatically **skipped** when the data files are not present,
so the test suite passes cleanly in CI environments without data.

To run these tests locally after fetching and building the data:

    nhisml fetch --year 2023 --year 2024
    nhisml build-core --year 2023
    nhisml build-core --year 2024
    pytest tests/test_data_integration.py -v

Reference statistics are documented in docs/reference_statistics.md.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Paths (relative to project root; tests run from there by convention)
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")
CORE_2023 = DATA_DIR / "core_2023.parquet"
CORE_2024 = DATA_DIR / "core_2024.parquet"

needs_2023 = pytest.mark.skipif(not CORE_2023.exists(), reason="core_2023.parquet not found — run nhisml fetch/build-core first")
needs_2024 = pytest.mark.skipif(not CORE_2024.exists(), reason="core_2024.parquet not found — run nhisml fetch/build-core first")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _nhis_eligible(s: pd.Series, valid: set) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").isin(valid)


def _weighted_mean(y: pd.Series, w: pd.Series) -> float:
    mask = y.notna() & w.notna() & (w > 0)
    return float(np.average(y[mask].astype(float), weights=w[mask]))


# ---------------------------------------------------------------------------
# 2023 core parquet — structural checks
# ---------------------------------------------------------------------------

@needs_2023
class TestCore2023Structure:
    def setup_method(self):
        self.df = _load(CORE_2023)

    def test_row_count(self):
        """NHIS 2023 Adults public-use file has exactly 29,522 respondents."""
        assert len(self.df) == 29_522, (
            f"Expected 29,522 rows; got {len(self.df)}. "
            "This usually means a different year's data was built under this filename."
        )

    def test_minimum_column_count(self):
        """Core 2023 parquet must have at least 64 columns."""
        assert self.df.shape[1] >= 64, f"Expected ≥64 columns; got {self.df.shape[1]}"

    def test_required_columns_present(self):
        required = {"PHSTAT_A", "SMKCIGST_A", "WTFA_A", "SEX_A", "AGEP_A", "EDUCP_A"}
        missing = required - set(self.df.columns)
        assert not missing, f"Missing required columns: {missing}"

    def test_survey_weights_positive(self):
        """All WTFA_A weights must be strictly positive (no zeroes or negatives)."""
        w = pd.to_numeric(self.df["WTFA_A"], errors="coerce").dropna()
        assert (w > 0).all(), f"{(w <= 0).sum()} non-positive weights found"

    def test_survey_weights_finite(self):
        w = pd.to_numeric(self.df["WTFA_A"], errors="coerce")
        assert w.isna().sum() == 0, f"{w.isna().sum()} missing weight values"
        assert np.isfinite(w).all(), "Non-finite survey weights found"

    def test_sex_codes_valid(self):
        """SEX_A must be 1 (Male) or 2 (Female) — NHIS coding."""
        s = pd.to_numeric(self.df["SEX_A"], errors="coerce")
        valid_or_missing = s.isin([1, 2]) | s.isna()
        assert valid_or_missing.all(), f"Unexpected SEX_A codes: {s[~valid_or_missing].unique()}"

    def test_phstat_codes_valid(self):
        """PHSTAT_A must be in 1–5 or NHIS missing codes."""
        s = pd.to_numeric(self.df["PHSTAT_A"], errors="coerce")
        valid_or_missing = s.isin([1, 2, 3, 4, 5, 7, 8, 9]) | s.isna()
        assert valid_or_missing.all(), (
            f"Unexpected PHSTAT_A codes: {s[~valid_or_missing].unique()}"
        )

    def test_age_range_plausible(self):
        """AGEP_A should be ≥18 and ≤85 (top-coded) after removing NHIS missing
        codes 97 (Refused), 98 (Not ascertained), 99 (Don't know)."""
        MISSING = {97, 98, 99}
        s = pd.to_numeric(self.df["AGEP_A"], errors="coerce")
        s = s[~s.isin(MISSING)].dropna()
        assert (s >= 18).all(), f"Ages below 18 found: {s[s < 18].tolist()[:10]}"
        assert (s <= 85).all(), f"Ages above 85 found after removing missing codes: {s[s > 85].tolist()[:10]}"


# ---------------------------------------------------------------------------
# 2024 core parquet — structural checks
# ---------------------------------------------------------------------------

@needs_2024
class TestCore2024Structure:
    def setup_method(self):
        self.df = _load(CORE_2024)

    def test_row_count(self):
        """NHIS 2024 Adults public-use file has exactly 32,629 respondents."""
        assert len(self.df) == 32_629, (
            f"Expected 32,629 rows; got {len(self.df)}."
        )

    def test_minimum_column_count(self):
        assert self.df.shape[1] >= 75, f"Expected ≥75 columns; got {self.df.shape[1]}"

    def test_required_columns_present(self):
        required = {"PHSTAT_A", "SMKCIGST_A", "WTFA_A", "SEX_A", "AGEP_A", "EDUCP_A"}
        missing = required - set(self.df.columns)
        assert not missing, f"Missing required columns: {missing}"

    def test_survey_weights_positive(self):
        w = pd.to_numeric(self.df["WTFA_A"], errors="coerce").dropna()
        assert (w > 0).all(), f"{(w <= 0).sum()} non-positive weights found"

    def test_survey_weights_finite(self):
        w = pd.to_numeric(self.df["WTFA_A"], errors="coerce")
        assert w.isna().sum() == 0
        assert np.isfinite(w).all()

    def test_2024_has_additional_columns(self):
        """2024 expanded the questionnaire; HOPELESS_A and LONELY_A should be present."""
        for col in ["HOPELESS_A", "LONELY_A", "NERVOUS_A", "SAD_A"]:
            assert col in self.df.columns, f"2024 expansion column missing: {col}"


# ---------------------------------------------------------------------------
# 2023 — label prevalence checks
# ---------------------------------------------------------------------------

@needs_2023
class TestCore2023Prevalence:
    def setup_method(self):
        self.df = _load(CORE_2023)
        self.w = pd.to_numeric(self.df["WTFA_A"], errors="coerce")

    def test_srh_eligible_count(self):
        """
        After removing NHIS missing codes (7/8/9), 29,502 of 29,522 rows
        should have a valid PHSTAT_A response.
        """
        s = pd.to_numeric(self.df["PHSTAT_A"], errors="coerce")
        eligible = s.isin([1, 2, 3, 4, 5]).sum()
        assert eligible == 29_502, (
            f"Expected 29,502 eligible for SRH; got {eligible}. "
            "Check that NHIS missing codes are not being stripped before save."
        )

    def test_smoking_eligible_count(self):
        """
        28,481 rows should have a valid SMKCIGST_A or SMKNOW_A response.
        (1,041 rows are missing both primary and fallback smoking columns.)
        """
        primary = pd.to_numeric(self.df.get("SMKCIGST_A"), errors="coerce").isin([1, 2, 3, 4])
        fallback = pd.to_numeric(self.df.get("SMKNOW_A"), errors="coerce").isin([1, 2, 3])
        # Use primary if present, fallback otherwise (mirrors tasks._smoking_current)
        eligible = primary | (~primary & fallback)
        assert eligible.sum() == 28_481, (
            f"Expected 28,481 eligible for smoking_current; got {eligible.sum()}."
        )

    def test_srh_unweighted_prevalence(self):
        """
        Unweighted Fair/Poor SRH rate in 2023 should be approximately 15.7%
        (within ±1 percentage point).
        """
        s = pd.to_numeric(self.df["PHSTAT_A"], errors="coerce")
        eligible = s.isin([1, 2, 3, 4, 5])
        rate = float((s[eligible] >= 4).mean())
        assert 0.147 <= rate <= 0.167, (
            f"Unweighted SRH prevalence 2023: {rate:.3f} is outside expected range [0.147, 0.167]. "
            "This may indicate a data harmonization error."
        )


# ---------------------------------------------------------------------------
# 2024 — label prevalence checks
# ---------------------------------------------------------------------------

@needs_2024
class TestCore2024Prevalence:
    def setup_method(self):
        self.df = _load(CORE_2024)
        self.w = pd.to_numeric(self.df["WTFA_A"], errors="coerce")

    def test_srh_eligible_count(self):
        """32,613 of 32,629 rows should have a valid PHSTAT_A value."""
        s = pd.to_numeric(self.df["PHSTAT_A"], errors="coerce")
        eligible = s.isin([1, 2, 3, 4, 5]).sum()
        assert eligible == 32_613, (
            f"Expected 32,613 eligible for SRH; got {eligible}."
        )

    def test_smoking_eligible_count(self):
        """31,876 of 32,629 rows should be eligible for smoking_current."""
        primary = pd.to_numeric(self.df.get("SMKCIGST_A"), errors="coerce").isin([1, 2, 3, 4])
        eligible = primary.sum()
        assert eligible == 31_876, (
            f"Expected 31,876 eligible for smoking_current (via SMKCIGST_A); got {eligible}."
        )

    def test_srh_unweighted_prevalence(self):
        """
        Unweighted Fair/Poor SRH rate in 2024 should be approximately 16.4%
        (within ±1 pp). Published reference: pos_rate_unweighted = 0.1642.
        """
        s = pd.to_numeric(self.df["PHSTAT_A"], errors="coerce")
        eligible = s.isin([1, 2, 3, 4, 5])
        rate = float((s[eligible] >= 4).mean())
        assert 0.154 <= rate <= 0.174, (
            f"Unweighted SRH prevalence 2024: {rate:.3f} is outside [0.154, 0.174]. "
            f"Published reference: 0.1642."
        )

    def test_srh_weighted_prevalence(self):
        """
        Survey-weighted Fair/Poor SRH rate in 2024 should be approximately 14.8%
        (within ±1 pp). Published reference: pos_rate_weighted = 0.1484.
        """
        s = pd.to_numeric(self.df["PHSTAT_A"], errors="coerce")
        eligible_mask = s.isin([1, 2, 3, 4, 5])
        y = (s[eligible_mask] >= 4).astype(float)
        w = self.w[eligible_mask]
        rate = _weighted_mean(y, w)
        assert 0.138 <= rate <= 0.158, (
            f"Survey-weighted SRH prevalence 2024: {rate:.3f} is outside [0.138, 0.158]. "
            f"Published reference: 0.1484."
        )

    def test_smoking_unweighted_prevalence(self):
        """
        Unweighted current smoking rate in 2024 should be approximately 10.6%
        (within ±1 pp). Published reference: pos_rate_unweighted = 0.1055.
        """
        s = pd.to_numeric(self.df["SMKCIGST_A"], errors="coerce")
        eligible = s.isin([1, 2, 3, 4])
        rate = float(s[eligible].isin([1, 2]).mean())
        assert 0.095 <= rate <= 0.115, (
            f"Unweighted smoking prevalence 2024: {rate:.3f} is outside [0.095, 0.115]. "
            f"Published reference: 0.1055."
        )

    def test_smoking_weighted_prevalence(self):
        """
        Survey-weighted current smoking rate in 2024 should be approximately 9.9%
        (within ±1 pp). Published reference: pos_rate_weighted = 0.0994.
        """
        s = pd.to_numeric(self.df["SMKCIGST_A"], errors="coerce")
        eligible_mask = s.isin([1, 2, 3, 4])
        y = s[eligible_mask].isin([1, 2]).astype(float)
        w = self.w[eligible_mask]
        rate = _weighted_mean(y, w)
        assert 0.089 <= rate <= 0.109, (
            f"Survey-weighted smoking prevalence 2024: {rate:.3f} is outside [0.089, 0.109]. "
            f"Published reference: 0.0994."
        )

    def test_sex_distribution_plausible(self):
        """
        NHIS Adults 2024 should be approximately 52% female (within ±3 pp).
        """
        s = pd.to_numeric(self.df["SEX_A"], errors="coerce")
        valid = s.isin([1, 2])
        pct_female = float((s[valid] == 2).mean())
        assert 0.49 <= pct_female <= 0.55, (
            f"Female proportion 2024: {pct_female:.3f} is outside expected [0.49, 0.55]."
        )

    def test_region_codes_all_present(self):
        """All four NCHS regions (1=Northeast, 2=Midwest, 3=South, 4=West) should appear."""
        s = pd.to_numeric(self.df["REGION"], errors="coerce")
        present = set(s.dropna().astype(int).unique())
        assert {1, 2, 3, 4}.issubset(present), (
            f"Not all NCHS regions present. Found: {sorted(present)}"
        )


# ---------------------------------------------------------------------------
# Cross-year consistency
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not CORE_2023.exists() or not CORE_2024.exists(),
    reason="Both core_2023.parquet and core_2024.parquet required",
)
class TestCrossYearConsistency:
    def setup_method(self):
        self.df23 = _load(CORE_2023)
        self.df24 = _load(CORE_2024)

    def test_2024_larger_than_2023(self):
        """NHIS 2024 has more respondents than 2023 (32,629 vs 29,522)."""
        assert len(self.df24) > len(self.df23), (
            f"Expected 2024 ({len(self.df24)}) > 2023 ({len(self.df23)})."
        )

    def test_shared_columns_present_in_both(self):
        """Core demographic and label columns should exist in both years."""
        shared = {"PHSTAT_A", "SMKCIGST_A", "WTFA_A", "SEX_A", "AGEP_A",
                  "EDUCP_A", "REGION", "HYPEV_A", "HICOV_A"}
        missing_23 = shared - set(self.df23.columns)
        missing_24 = shared - set(self.df24.columns)
        assert not missing_23, f"Missing from 2023: {missing_23}"
        assert not missing_24, f"Missing from 2024: {missing_24}"

    def test_srh_prevalence_stable_across_years(self):
        """
        Weighted Fair/Poor SRH prevalence should be within 3 pp between 2023 and 2024.
        A larger shift would suggest a data harmonization error.
        """
        results = {}
        for year, df in [(2023, self.df23), (2024, self.df24)]:
            s = pd.to_numeric(df["PHSTAT_A"], errors="coerce")
            w = pd.to_numeric(df["WTFA_A"], errors="coerce")
            eligible = s.isin([1, 2, 3, 4, 5])
            y = (s[eligible] >= 4).astype(float)
            results[year] = _weighted_mean(y, w[eligible])

        diff = abs(results[2023] - results[2024])
        assert diff <= 0.03, (
            f"SRH prevalence shifted by {diff:.3f} between years "
            f"(2023={results[2023]:.3f}, 2024={results[2024]:.3f}). "
            "Shifts >3 pp may indicate a coding discrepancy."
        )
