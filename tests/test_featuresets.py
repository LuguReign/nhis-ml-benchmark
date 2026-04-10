"""
Tests for nhisml.featuresets — feature set registry and FeatureSet dataclass.
"""
from __future__ import annotations

import pytest

from nhisml.featuresets import FeatureSet, get_featureset, list_featuresets


class TestRegistry:
    def test_list_featuresets_nonempty(self):
        assert len(list_featuresets()) >= 1

    def test_list_featuresets_contains_core(self):
        assert "core" in list_featuresets()

    def test_list_featuresets_sorted(self):
        fsl = list_featuresets()
        assert fsl == sorted(fsl)

    def test_get_featureset_core(self):
        fs = get_featureset("core")
        assert isinstance(fs, FeatureSet)
        assert fs.name == "core"

    def test_get_featureset_default_is_core(self):
        assert get_featureset().name == "core"

    def test_get_featureset_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown featureset"):
            get_featureset("nonexistent_featureset_xyz")


class TestCoreFeatureSet:
    def setup_method(self):
        self.fs = get_featureset("core")

    def test_has_description(self):
        assert self.fs.description and isinstance(self.fs.description, str)

    def test_binary_cols_nonempty(self):
        assert len(self.fs.binary_12) > 0

    def test_ordinal_cols_nonempty(self):
        assert len(self.fs.ordinal) > 0

    def test_categorical_cols_nonempty(self):
        assert len(self.fs.categorical) > 0

    def test_all_columns_no_duplicates(self):
        cols = self.fs.all_columns
        assert len(cols) == len(set(cols))

    def test_all_columns_union(self):
        """all_columns should contain every column from each subgroup."""
        col_set = set(self.fs.all_columns)
        for c in self.fs.binary_12:
            assert c in col_set
        for c in self.fs.ordinal:
            assert c in col_set
        for c in self.fs.categorical:
            assert c in col_set

    def test_expected_binary_cols_present(self):
        assert "HYPEV_A" in self.fs.binary_12
        assert "HICOV_A" in self.fs.binary_12

    def test_expected_ordinal_cols_present(self):
        assert "EDUCP_A" in self.fs.ordinal
        assert "RATCAT_A" in self.fs.ordinal

    def test_expected_categorical_cols_present(self):
        assert "REGION" in self.fs.categorical
        assert "MARITAL_A" in self.fs.categorical

    def test_featureset_is_frozen(self):
        """FeatureSet is declared frozen=True — mutation should raise."""
        with pytest.raises((AttributeError, TypeError)):
            self.fs.name = "modified"  # type: ignore[misc]

    def test_all_columns_returns_list(self):
        assert isinstance(self.fs.all_columns, list)
