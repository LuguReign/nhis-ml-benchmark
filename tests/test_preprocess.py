"""
Tests for nhisml.preprocess — NHIS missing-code handling, PrepareFrame,
build_preprocessor, schema extraction, and feature name utilities.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from nhisml.preprocess import (
    NHIS_MISSING,
    PrepareFrame,
    PreprocessSchema,
    _map_missing,
    _recode_binary_12,
    build_preprocessor,
    build_schema_from_fitted,
    get_feature_names,
    normalize_weights,
)


# ---------------------------------------------------------------------------
# normalize_weights
# ---------------------------------------------------------------------------

class TestNormalizeWeights:
    def test_mean_one(self):
        w = pd.Series([1.0, 2.0, 3.0])
        nw = normalize_weights(w)
        assert abs(nw.mean() - 1.0) < 1e-9

    def test_zero_mean_fallback(self):
        """All-zero weights should not divide by zero."""
        w = pd.Series([0.0, 0.0, 0.0])
        nw = normalize_weights(w)
        assert np.all(nw == 0.0)

    def test_negative_clipped(self):
        w = pd.Series([-1.0, 2.0, 3.0])
        nw = normalize_weights(w)
        assert nw[0] == 0.0

    def test_nan_treated_as_zero(self):
        w = pd.Series([np.nan, 2.0, 3.0])
        nw = normalize_weights(w)
        assert nw[0] == 0.0

    def test_output_dtype_float64(self):
        w = pd.Series([1, 2, 3], dtype=int)
        nw = normalize_weights(w)
        assert nw.dtype == np.float64

    def test_string_values_coerced(self):
        w = pd.Series(["1.0", "2.0", "3.0"])
        nw = normalize_weights(w)
        assert abs(nw.mean() - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# _map_missing
# ---------------------------------------------------------------------------

class TestMapMissing:
    def test_nhis_codes_become_nan(self):
        df = pd.DataFrame({"X": [1, 7, 8, 9, 97, 98, 99, 2]})
        out = _map_missing(df, ["X"])
        assert pd.isna(out["X"].iloc[1])
        assert pd.isna(out["X"].iloc[2])
        assert out["X"].iloc[0] == 1
        assert out["X"].iloc[7] == 2

    def test_absent_columns_ignored(self):
        df = pd.DataFrame({"X": [1, 2, 3]})
        out = _map_missing(df, ["X", "NONEXISTENT"])
        assert "NONEXISTENT" not in out.columns
        assert list(out["X"]) == [1, 2, 3]

    def test_original_df_not_mutated(self):
        df = pd.DataFrame({"X": [7, 8, 9]})
        _map_missing(df, ["X"])
        assert df["X"].iloc[0] == 7  # original unchanged


# ---------------------------------------------------------------------------
# _recode_binary_12
# ---------------------------------------------------------------------------

class TestRecodeBinary12:
    def test_1_to_1_and_2_to_0(self):
        df = pd.DataFrame({"Y": [1, 2, 1, 2]})
        out = _recode_binary_12(df, ["Y"])
        assert list(out["Y"]) == [1.0, 0.0, 1.0, 0.0]

    def test_other_values_become_nan(self):
        df = pd.DataFrame({"Y": [1, 2, 3, np.nan]})
        out = _recode_binary_12(df, ["Y"])
        assert pd.isna(out["Y"].iloc[2])
        assert pd.isna(out["Y"].iloc[3])

    def test_missing_codes_become_nan_first(self):
        df = pd.DataFrame({"Y": [7, 1, 2]})
        out = _recode_binary_12(df, ["Y"])
        # 7 is an NHIS missing code; recode maps it from coerce->NaN
        assert pd.isna(out["Y"].iloc[0])


# ---------------------------------------------------------------------------
# PrepareFrame
# ---------------------------------------------------------------------------

def _make_synthetic_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "BIN_A": rng.choice([1, 2, 7, np.nan], size=n),
            "ORD_A": rng.choice([1, 2, 3, 4, 5, 9, np.nan], size=n).astype(float),
            "CAT_A": rng.choice(["A", "B", "C", "__RARE_THING__", None], size=n),
        }
    )


class TestPrepareFrame:
    def _fitted(self, df=None):
        if df is None:
            df = _make_synthetic_df()
        pf = PrepareFrame(
            binary_cols=["BIN_A"],
            ordinal_cols=["ORD_A"],
            categorical_cols=["CAT_A"],
            rare_min_count=5,
        )
        pf.fit(df)
        return pf, df

    def test_fit_sets_cols(self):
        pf, _ = self._fitted()
        assert "BIN_A" in pf.binary_cols_
        assert "ORD_A" in pf.ordinal_cols_
        assert "CAT_A" in pf.categorical_cols_

    def test_missing_flags_created(self):
        pf, _ = self._fitted()
        # one flag per ordinal + categorical
        assert "ORD_A__ismissing" in pf.added_missing_flags_
        assert "CAT_A__ismissing" in pf.added_missing_flags_

    def test_transform_returns_dataframe(self):
        pf, df = self._fitted()
        out = pf.transform(df)
        assert isinstance(out, pd.DataFrame)

    def test_transform_adds_missing_flags(self):
        pf, df = self._fitted()
        out = pf.transform(df)
        assert "ORD_A__ismissing" in out.columns
        assert "CAT_A__ismissing" in out.columns

    def test_missing_flags_are_binary(self):
        pf, df = self._fitted()
        out = pf.transform(df)
        vals = out["ORD_A__ismissing"].dropna().unique()
        assert set(vals).issubset({0.0, 1.0})

    def test_rare_bucketing(self):
        df = pd.DataFrame({"CAT_A": ["A"] * 100 + ["B"] * 100 + ["RARE"] * 2})
        pf = PrepareFrame(binary_cols=[], ordinal_cols=[], categorical_cols=["CAT_A"],
                          rare_min_count=10)
        pf.fit(df)
        levels = pf.categorical_levels_["CAT_A"]
        assert "__RARE__" in levels
        assert "A" in levels
        assert "B" in levels
        assert "RARE" not in levels

    def test_unseen_category_becomes_rare(self):
        # Include a rare training category so __RARE__ is added to the level set.
        # Without it, unseen values are left as-is (no __RARE__ bucket exists).
        df_train = pd.DataFrame({"CAT_A": ["A"] * 100 + ["B"] * 100 + ["RARE_TRAIN"] * 2})
        pf = PrepareFrame(binary_cols=[], ordinal_cols=[], categorical_cols=["CAT_A"],
                          rare_min_count=5)
        pf.fit(df_train)
        df_test = pd.DataFrame({"CAT_A": ["A", "UNSEEN_VALUE"]})
        out = pf.transform(df_test)
        assert out["CAT_A"].iloc[1] == "__RARE__"

    def test_absent_columns_skipped(self):
        df = pd.DataFrame({"OTHER": [1, 2, 3]})
        pf = PrepareFrame(binary_cols=["BIN_A"], ordinal_cols=["ORD_A"],
                          categorical_cols=["CAT_A"])
        pf.fit(df)
        assert pf.binary_cols_ == []
        assert pf.ordinal_cols_ == []
        assert pf.categorical_cols_ == []

    def test_output_missing_flags_method(self):
        pf, _ = self._fitted()
        flags = pf.output_missing_flags()
        assert isinstance(flags, list)
        assert all("__ismissing" in f for f in flags)

    def test_no_missing_flags_option(self):
        df = _make_synthetic_df()
        pf = PrepareFrame(binary_cols=["BIN_A"], ordinal_cols=["ORD_A"],
                          categorical_cols=["CAT_A"], add_missing_flags=False)
        pf.fit(df)
        assert pf.added_missing_flags_ == []

    def test_nhis_missing_codes_cleared(self):
        """Values 7, 8, 9 in ordinal should become NaN after transform."""
        df = pd.DataFrame({"ORD_A": [1, 7, 9, 3]})
        pf = PrepareFrame(binary_cols=[], ordinal_cols=["ORD_A"],
                          categorical_cols=[], add_missing_flags=False)
        pf.fit(df)
        out = pf.transform(df)
        assert pd.isna(out["ORD_A"].iloc[1])
        assert pd.isna(out["ORD_A"].iloc[2])
        assert not pd.isna(out["ORD_A"].iloc[0])


# ---------------------------------------------------------------------------
# build_preprocessor + get_feature_names + build_schema_from_fitted
# ---------------------------------------------------------------------------

class TestBuildPreprocessor:
    def _fitted_pipeline(self):
        df = _make_synthetic_df(n=300)
        y = np.random.default_rng(1).integers(0, 2, size=300)
        pipe = build_preprocessor(
            binary_cols=["BIN_A"],
            ordinal_cols=["ORD_A"],
            categorical_cols=["CAT_A"],
            rare_min_count=5,
        )
        pipe.fit(df, y)
        return pipe, df

    def test_returns_pipeline(self):
        pipe = build_preprocessor(["BIN_A"], ["ORD_A"], ["CAT_A"])
        assert isinstance(pipe, Pipeline)

    def test_has_frame_and_ct_steps(self):
        pipe = build_preprocessor(["BIN_A"], ["ORD_A"], ["CAT_A"])
        assert "frame" in pipe.named_steps
        assert "ct" in pipe.named_steps

    def test_transform_produces_array(self):
        pipe, df = self._fitted_pipeline()
        out = pipe.transform(df)
        # sklearn ColumnTransformer returns ndarray or sparse matrix
        assert hasattr(out, "shape")
        assert out.shape[0] == len(df)

    def test_get_feature_names_returns_list(self):
        pipe, df = self._fitted_pipeline()
        names = get_feature_names(pipe)
        assert isinstance(names, list)
        assert len(names) > 0

    def test_get_feature_names_no_duplicates(self):
        pipe, df = self._fitted_pipeline()
        names = get_feature_names(pipe)
        assert len(names) == len(set(names))

    def test_build_schema_returns_schema(self):
        pipe, df = self._fitted_pipeline()
        schema = build_schema_from_fitted(pipe)
        assert isinstance(schema, PreprocessSchema)

    def test_schema_records_columns(self):
        pipe, df = self._fitted_pipeline()
        schema = build_schema_from_fitted(pipe)
        assert "BIN_A" in schema.binary_cols
        assert "ORD_A" in schema.ordinal_cols
        assert "CAT_A" in schema.categorical_cols

    def test_schema_to_json_roundtrip(self, tmp_path):
        pipe, df = self._fitted_pipeline()
        schema = build_schema_from_fitted(pipe)
        p = str(tmp_path / "schema.json")
        schema.to_json(p)
        import json
        with open(p) as f:
            loaded = json.load(f)
        assert loaded["binary_cols"] == schema.binary_cols
        assert loaded["ordinal_cols"] == schema.ordinal_cols

    def test_build_schema_bad_input_raises(self):
        with pytest.raises(ValueError):
            build_schema_from_fitted(object())  # type: ignore

    def test_get_feature_names_bad_input_raises(self):
        with pytest.raises(ValueError):
            get_feature_names(object())  # type: ignore
