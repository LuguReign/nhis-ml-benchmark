"""
nhisml.validate_data — dataset integrity and reference-statistic checks.

Provides :func:`validate_core_year`, which verifies that a processed core
parquet matches the known NHIS Adults public-use file statistics.  Users
can call this from the CLI (``nhisml validate-data``) or from Python after
running ``nhisml fetch`` and ``nhisml build-core``.

Reference statistics were derived from the official CDC/NCHS NHIS
Adults public-use files and are documented in ``docs/reference_statistics.md``.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Reference statistics (ground truth)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class YearRef:
    """Known reference statistics for one NHIS Adults survey year."""
    year: int
    n_rows: int
    n_cols_min: int
    srh_eligible: int
    smoking_eligible: int          # via primary SMKCIGST_A column
    srh_pos_rate_unweighted: float  # fraction with PHSTAT_A in {4,5}
    srh_pos_rate_weighted: float    # survey-weighted equivalent
    smoking_pos_rate_unweighted: float  # fraction with SMKCIGST_A in {1,2}
    smoking_pos_rate_weighted: float
    required_cols: Tuple[str, ...] = (
        "PHSTAT_A", "SMKCIGST_A", "WTFA_A", "SEX_A", "AGEP_A", "EDUCP_A", "REGION",
    )
    tolerance_pp: float = 0.01   # ±1 percentage point tolerance on rates


REFERENCE: Dict[int, YearRef] = {
    2023: YearRef(
        year=2023,
        n_rows=29_522,
        n_cols_min=64,
        srh_eligible=29_502,
        smoking_eligible=28_481,
        srh_pos_rate_unweighted=0.157,   # ±1 pp tolerance applied
        srh_pos_rate_weighted=0.151,     # observed from core_2023.parquet
        smoking_pos_rate_unweighted=0.115,
        smoking_pos_rate_weighted=0.105,
    ),
    2024: YearRef(
        year=2024,
        n_rows=32_629,
        n_cols_min=75,
        srh_eligible=32_613,
        smoking_eligible=31_876,
        srh_pos_rate_unweighted=0.1642,
        srh_pos_rate_weighted=0.1484,
        smoking_pos_rate_unweighted=0.1055,
        smoking_pos_rate_weighted=0.0994,
    ),
}


# ---------------------------------------------------------------------------
# Check result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    expected: Optional[str] = None
    actual: Optional[str] = None


@dataclass
class ValidationReport:
    year: int
    path: str
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def n_passed(self) -> int:
        return sum(c.passed for c in self.checks)

    @property
    def n_failed(self) -> int:
        return sum(not c.passed for c in self.checks)

    @property
    def all_passed(self) -> bool:
        return self.n_failed == 0

    def print_report(self, verbose: bool = False) -> None:
        status = "PASS" if self.all_passed else "FAIL"
        print(f"\n{'='*60}")
        print(f"  nhisml validate-data — NHIS Adults {self.year}")
        print(f"  File : {self.path}")
        print(f"  Result: {status}  ({self.n_passed} passed, {self.n_failed} failed)")
        print(f"{'='*60}")
        for c in self.checks:
            icon = "✓" if c.passed else "✗"
            print(f"  {icon} {c.name}")
            if not c.passed or verbose:
                if c.expected is not None:
                    print(f"      expected : {c.expected}")
                    print(f"      actual   : {c.actual}")
                if not c.passed:
                    print(f"      → {c.message}")
        print()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

NHIS_MISSING_CODES = {7, 8, 9, 97, 98, 99}


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _weighted_mean(y: pd.Series, w: pd.Series) -> float:
    mask = y.notna() & w.notna() & (w > 0)
    if mask.sum() == 0:
        return float("nan")
    return float(np.average(y[mask].astype(float), weights=w[mask]))


def _check(name: str, condition: bool, message: str,
           expected: Optional[str] = None, actual: Optional[str] = None) -> CheckResult:
    return CheckResult(name=name, passed=bool(condition), message=message,
                       expected=expected, actual=actual)


# ---------------------------------------------------------------------------
# Core validation logic
# ---------------------------------------------------------------------------

def validate_core_year(
    year: int,
    path: str,
    ref: Optional[YearRef] = None,
) -> ValidationReport:
    """
    Validate a processed core parquet against known reference statistics.

    Parameters
    ----------
    year : int
        Survey year (2023 or 2024).
    path : str
        Path to the core parquet file (e.g., ``data/core_2023.parquet``).
    ref : YearRef, optional
        Reference statistics to validate against.  Defaults to the built-in
        reference for the given *year*.  Pass a custom :class:`YearRef` to
        validate against different targets.

    Returns
    -------
    ValidationReport
        Structured report with per-check results and a summary.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    KeyError
        If no built-in reference exists for *year* and *ref* is None.
    """
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Core parquet not found: {path}\n"
            f"Run: nhisml fetch --year {year} && nhisml build-core --year {year}"
        )

    if ref is None:
        if year not in REFERENCE:
            raise KeyError(
                f"No built-in reference statistics for year {year}. "
                f"Supported years: {sorted(REFERENCE)}. "
                "Pass a custom YearRef via the ref= argument."
            )
        ref = REFERENCE[year]

    df = pd.read_parquet(path)
    report = ValidationReport(year=year, path=path)
    add = report.checks.append
    tol = ref.tolerance_pp

    # ---- Structural checks -----------------------------------------------
    add(_check(
        "Row count",
        len(df) == ref.n_rows,
        "Row count mismatch — wrong year's data may have been built here.",
        expected=str(ref.n_rows),
        actual=str(len(df)),
    ))

    add(_check(
        "Minimum column count",
        df.shape[1] >= ref.n_cols_min,
        "Fewer columns than expected; featureset or task columns may be missing.",
        expected=f"≥{ref.n_cols_min}",
        actual=str(df.shape[1]),
    ))

    missing_cols = [c for c in ref.required_cols if c not in df.columns]
    add(_check(
        "Required columns present",
        len(missing_cols) == 0,
        f"Missing columns: {missing_cols}",
        expected="all present",
        actual=f"missing: {missing_cols}" if missing_cols else "all present",
    ))

    # ---- Survey weight checks -------------------------------------------
    w = _to_num(df["WTFA_A"]) if "WTFA_A" in df.columns else pd.Series(dtype=float)
    if len(w):
        add(_check(
            "Survey weights: no missing values",
            int(w.isna().sum()) == 0,
            f"{int(w.isna().sum())} missing WTFA_A values found.",
            expected="0 missing",
            actual=str(int(w.isna().sum())),
        ))
        add(_check(
            "Survey weights: all positive",
            bool((w > 0).all()),
            f"{int((w <= 0).sum())} non-positive weights found.",
            expected="all > 0",
            actual=f"{int((w <= 0).sum())} non-positive",
        ))

    # ---- SRH eligibility and prevalence ---------------------------------
    if "PHSTAT_A" in df.columns:
        s = _to_num(df["PHSTAT_A"])
        eligible_mask = s.isin([1, 2, 3, 4, 5])
        n_eligible = int(eligible_mask.sum())

        add(_check(
            "SRH eligible count",
            n_eligible == ref.srh_eligible,
            f"Expected {ref.srh_eligible} eligible SRH rows; got {n_eligible}. "
            "Check NHIS missing-code handling.",
            expected=str(ref.srh_eligible),
            actual=str(n_eligible),
        ))

        if n_eligible > 0:
            rate_unw = float((s[eligible_mask] >= 4).mean())
            lo, hi = ref.srh_pos_rate_unweighted - tol, ref.srh_pos_rate_unweighted + tol
            add(_check(
                f"SRH unweighted prevalence (ref ≈ {ref.srh_pos_rate_unweighted:.1%}, ±{tol:.0%})",
                lo <= rate_unw <= hi,
                f"Fair/Poor SRH rate outside expected range [{lo:.3f}, {hi:.3f}].",
                expected=f"{lo:.3f}–{hi:.3f}",
                actual=f"{rate_unw:.4f}",
            ))

            if len(w) == len(df):
                rate_w = _weighted_mean((s[eligible_mask] >= 4).astype(float), w[eligible_mask])
                lo_w = ref.srh_pos_rate_weighted - tol
                hi_w = ref.srh_pos_rate_weighted + tol
                add(_check(
                    f"SRH weighted prevalence (ref ≈ {ref.srh_pos_rate_weighted:.1%}, ±{tol:.0%})",
                    lo_w <= rate_w <= hi_w,
                    f"Survey-weighted Fair/Poor SRH rate outside [{lo_w:.3f}, {hi_w:.3f}].",
                    expected=f"{lo_w:.3f}–{hi_w:.3f}",
                    actual=f"{rate_w:.4f}",
                ))

    # ---- Smoking eligibility and prevalence -----------------------------
    if "SMKCIGST_A" in df.columns:
        s = _to_num(df["SMKCIGST_A"])
        eligible_mask = s.isin([1, 2, 3, 4])
        n_eligible = int(eligible_mask.sum())

        add(_check(
            "Smoking eligible count (SMKCIGST_A)",
            n_eligible == ref.smoking_eligible,
            f"Expected {ref.smoking_eligible}; got {n_eligible}.",
            expected=str(ref.smoking_eligible),
            actual=str(n_eligible),
        ))

        if n_eligible > 0:
            rate_unw = float(s[eligible_mask].isin([1, 2]).mean())
            lo = ref.smoking_pos_rate_unweighted - tol
            hi = ref.smoking_pos_rate_unweighted + tol
            add(_check(
                f"Smoking unweighted prevalence (ref ≈ {ref.smoking_pos_rate_unweighted:.1%}, ±{tol:.0%})",
                lo <= rate_unw <= hi,
                f"Current smoking rate outside [{lo:.3f}, {hi:.3f}].",
                expected=f"{lo:.3f}–{hi:.3f}",
                actual=f"{rate_unw:.4f}",
            ))

            if len(w) == len(df):
                y_sm = s[eligible_mask].isin([1, 2]).astype(float)
                rate_w = _weighted_mean(y_sm, w[eligible_mask])
                lo_w = ref.smoking_pos_rate_weighted - tol
                hi_w = ref.smoking_pos_rate_weighted + tol
                add(_check(
                    f"Smoking weighted prevalence (ref ≈ {ref.smoking_pos_rate_weighted:.1%}, ±{tol:.0%})",
                    lo_w <= rate_w <= hi_w,
                    f"Survey-weighted smoking rate outside [{lo_w:.3f}, {hi_w:.3f}].",
                    expected=f"{lo_w:.3f}–{hi_w:.3f}",
                    actual=f"{rate_w:.4f}",
                ))

    # ---- Demographic plausibility ---------------------------------------
    if "SEX_A" in df.columns:
        s = _to_num(df["SEX_A"])
        valid = s.isin([1, 2])
        pct_female = float((s[valid] == 2).mean())
        add(_check(
            "Sex distribution plausible (49–55% female)",
            0.49 <= pct_female <= 0.55,
            f"Female proportion {pct_female:.3f} outside [0.49, 0.55].",
            expected="0.49–0.55",
            actual=f"{pct_female:.4f}",
        ))

    if "REGION" in df.columns:
        regions = set(_to_num(df["REGION"]).dropna().astype(int).unique())
        add(_check(
            "All four NCHS regions present",
            {1, 2, 3, 4}.issubset(regions),
            f"Missing region codes: {sorted({1,2,3,4} - regions)}.",
            expected="{1, 2, 3, 4}",
            actual=str(sorted(regions)),
        ))

    if "AGEP_A" in df.columns:
        # Strip NHIS missing codes (97=Refused, 98=Not ascertained, 99=Don't know)
        # before range check — these are not actual ages.
        ages = _to_num(df["AGEP_A"])
        ages = ages[~ages.isin(NHIS_MISSING_CODES)].dropna()
        add(_check(
            "Age range plausible (18–85 top-coded, excl. missing codes 97–99)",
            bool((ages >= 18).all() and (ages <= 85).all()),
            "Ages outside [18, 85] found after removing missing codes.",
            expected="18–85",
            actual=f"min={ages.min():.0f} max={ages.max():.0f}",
        ))

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def cli(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(
        "nhisml validate-data",
        description=(
            "Validate processed core parquets against known NHIS Adults reference statistics. "
            "Exits with code 1 if any checks fail."
        ),
    )
    p.add_argument(
        "--year", type=int, action="append", required=True,
        help="Year(s) to validate, e.g. --year 2023 --year 2024",
    )
    p.add_argument(
        "--data-dir", default="data",
        help="Directory containing core_YYYY.parquet files (default: data/)",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show expected/actual values for passing checks too",
    )
    args = p.parse_args(argv)

    any_failed = False
    for year in args.year:
        path = os.path.join(args.data_dir, f"core_{year}.parquet")
        try:
            report = validate_core_year(year, path)
        except FileNotFoundError as e:
            print(f"\n[validate-data] ERROR: {e}\n")
            any_failed = True
            continue
        except KeyError as e:
            print(f"\n[validate-data] ERROR: {e}\n")
            any_failed = True
            continue

        report.print_report(verbose=args.verbose)
        if not report.all_passed:
            any_failed = True

    import sys
    sys.exit(1 if any_failed else 0)
