"""
Tests for nhisml.build_core — column collection, basic normalization,
and end-to-end build_core_year using a synthetic zip file.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nhisml.build_core import (
    _basic_normalize,
    _collect_required_columns,
    build_core_year,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_zip(tmp_path: Path, year: int) -> Path:
    """
    Create a minimal adult CSV zip that build_core_year can parse.
    Includes NHIS columns needed by the 'core' featureset + label columns.
    """
    # Minimal columns: label cols + weight + a few featureset cols
    n = 100
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "PHSTAT_A":   rng.choice([1, 2, 3, 4, 5], size=n),
        "SMKCIGST_A": rng.choice([1, 2, 3, 4], size=n),
        "SMKNOW_A":   rng.choice([1, 2, 3], size=n),
        "WTFA_A":     rng.uniform(1000, 5000, size=n),
        "SEX_A":      rng.choice([1, 2], size=n),
        "AGEP_A":     rng.integers(18, 80, size=n),
        "EDUCP_A":    rng.integers(0, 11, size=n),
        "REGION":     rng.choice([1, 2, 3, 4], size=n),
        "MARITAL_A":  rng.choice([1, 2, 3, 4, 5, 6], size=n),
        "HYPEV_A":    rng.choice([1, 2], size=n),
        "HICOV_A":    rng.choice([1, 2], size=n),
        "RATCAT_A":   rng.integers(1, 15, size=n),
    })

    yy = str(year)[-2:]
    zip_name = f"adult{yy}csv.zip"
    csv_name = f"adult{yy}.csv"

    raw_dir = tmp_path / "raw" / str(year)
    raw_dir.mkdir(parents=True)
    zip_path = raw_dir / zip_name

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(csv_name, df.to_csv(index=False))
    zip_path.write_bytes(buf.getvalue())

    return zip_path


# ---------------------------------------------------------------------------
# _collect_required_columns
# ---------------------------------------------------------------------------

class TestCollectRequiredColumns:
    def test_includes_featureset_columns(self):
        cols = _collect_required_columns("core", ["srh_binary"], "WTFA_A")
        from nhisml.featuresets import get_featureset
        fs = get_featureset("core")
        for c in fs.all_columns:
            assert c in cols, f"Missing featureset column: {c}"

    def test_includes_weight_col(self):
        cols = _collect_required_columns("core", ["srh_binary"], "WTFA_A")
        assert "WTFA_A" in cols

    def test_includes_task_label_cols(self):
        cols = _collect_required_columns("core", ["srh_binary"], "WTFA_A")
        assert "PHSTAT_A" in cols

    def test_includes_subgroup_cols(self):
        cols = _collect_required_columns("core", ["srh_binary"], "WTFA_A")
        assert "SEX_A" in cols
        assert "AGEP_A" in cols
        assert "EDUCP_A" in cols

    def test_extra_cols_included(self):
        cols = _collect_required_columns("core", ["srh_binary"], "WTFA_A",
                                         extra_cols=["MY_EXTRA"])
        assert "MY_EXTRA" in cols

    def test_sorted_output(self):
        cols = _collect_required_columns("core", ["srh_binary"], "WTFA_A")
        assert cols == sorted(cols)

    def test_no_duplicates(self):
        cols = _collect_required_columns("core", ["srh_binary", "smoking_current"], "WTFA_A")
        assert len(cols) == len(set(cols))


# ---------------------------------------------------------------------------
# _basic_normalize
# ---------------------------------------------------------------------------

class TestBasicNormalize:
    def test_selects_only_requested_cols(self):
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        out = _basic_normalize(df, ["A", "C"])
        assert list(out.columns) == ["A", "C"]

    def test_absent_cols_silently_skipped(self):
        df = pd.DataFrame({"A": [1, 2]})
        out = _basic_normalize(df, ["A", "NONEXISTENT"])
        assert "NONEXISTENT" not in out.columns

    def test_weight_col_coerced_to_numeric(self):
        df = pd.DataFrame({"WTFA_A": ["1234.5", "2345.6"]})
        out = _basic_normalize(df, ["WTFA_A"])
        assert pd.api.types.is_numeric_dtype(out["WTFA_A"])

    def test_phstat_col_coerced(self):
        df = pd.DataFrame({"PHSTAT_A": ["3", "5"]})
        out = _basic_normalize(df, ["PHSTAT_A"])
        assert pd.api.types.is_numeric_dtype(out["PHSTAT_A"])

    def test_original_df_not_mutated(self):
        df = pd.DataFrame({"WTFA_A": ["1000"]})
        _basic_normalize(df, ["WTFA_A"])
        assert df["WTFA_A"].dtype == object


# ---------------------------------------------------------------------------
# build_core_year — end-to-end with synthetic zip
# ---------------------------------------------------------------------------

class TestBuildCoreYear:
    def test_creates_parquet(self, tmp_path):
        _make_synthetic_zip(tmp_path, 2023)
        out_path = build_core_year(
            year=2023,
            data_dir=str(tmp_path),
            out_dir=str(tmp_path),
            featureset="core",
            tasks=["srh_binary", "smoking_current"],
        )
        assert out_path.exists()
        assert out_path.suffix == ".parquet"

    def test_creates_manifest_json(self, tmp_path):
        _make_synthetic_zip(tmp_path, 2023)
        build_core_year(
            year=2023, data_dir=str(tmp_path), out_dir=str(tmp_path),
        )
        manifest_path = tmp_path / "core_2023.manifest.json"
        assert manifest_path.exists()
        with open(manifest_path) as f:
            m = json.load(f)
        assert m["year"] == 2023

    def test_parquet_has_expected_rows(self, tmp_path):
        _make_synthetic_zip(tmp_path, 2023)
        out_path = build_core_year(
            year=2023, data_dir=str(tmp_path), out_dir=str(tmp_path),
        )
        df = pd.read_parquet(out_path)
        # The synthetic CSV has 100 rows; all should be retained
        assert len(df) == 100

    def test_parquet_has_label_columns(self, tmp_path):
        _make_synthetic_zip(tmp_path, 2023)
        out_path = build_core_year(
            year=2023, data_dir=str(tmp_path), out_dir=str(tmp_path),
            tasks=["srh_binary"],
        )
        df = pd.read_parquet(out_path)
        assert "PHSTAT_A" in df.columns

    def test_missing_zip_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Missing raw zip"):
            build_core_year(
                year=2025,  # no zip created
                data_dir=str(tmp_path),
                out_dir=str(tmp_path),
            )

    def test_custom_weight_col(self, tmp_path):
        _make_synthetic_zip(tmp_path, 2023)
        out_path = build_core_year(
            year=2023, data_dir=str(tmp_path), out_dir=str(tmp_path),
            weight_col="WTFA_A",
        )
        df = pd.read_parquet(out_path)
        assert "WTFA_A" in df.columns
