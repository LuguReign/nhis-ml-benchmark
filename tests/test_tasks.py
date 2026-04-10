"""
Tests for nhisml.tasks — task registry, label generation, eligibility masking.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nhisml.tasks import Task, list_tasks, make_task


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_list_tasks_returns_known(self):
        tasks = list_tasks()
        assert "srh_binary" in tasks
        assert "smoking_current" in tasks

    def test_list_tasks_sorted(self):
        tasks = list_tasks()
        assert tasks == sorted(tasks)

    def test_make_task_srh(self):
        t = make_task("srh_binary")
        assert isinstance(t, Task)
        assert t.name == "srh_binary"
        assert t.problem_type == "binary"
        assert "PHSTAT_A" in t.required_cols

    def test_make_task_smoking(self):
        t = make_task("smoking_current")
        assert t.name == "smoking_current"
        assert t.problem_type == "binary"

    def test_make_task_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            make_task("nonexistent_task_xyz")

    def test_required_columns_method(self):
        t = make_task("srh_binary")
        assert t.required_columns() == t.required_cols


# ---------------------------------------------------------------------------
# srh_binary label generation
# ---------------------------------------------------------------------------

class TestSrhBinary:
    def _df(self, values):
        return pd.DataFrame({"PHSTAT_A": values})

    def test_fair_poor_labeled_1(self):
        """PHSTAT_A 4 (Fair) and 5 (Poor) -> label 1."""
        df = self._df([4, 5])
        t = make_task("srh_binary")
        y, eligible = t.make_labels(df)
        assert list(y) == [1, 1]
        assert eligible.sum() == 2

    def test_good_excellent_labeled_0(self):
        """PHSTAT_A 1,2,3 -> label 0."""
        df = self._df([1, 2, 3])
        t = make_task("srh_binary")
        y, eligible = t.make_labels(df)
        assert list(y) == [0, 0, 0]
        assert eligible.sum() == 3

    def test_mixed_labels(self):
        df = self._df([1, 3, 4, 5, 2])
        t = make_task("srh_binary")
        y, eligible = t.make_labels(df)
        expected = [0, 0, 1, 1, 0]
        assert list(y) == expected

    def test_missing_codes_ineligible(self):
        """PHSTAT_A 7,8,9 -> ineligible (eligible mask = False for those rows)."""
        df = self._df([9, 7, 8])
        t = make_task("srh_binary")
        y, eligible = t.make_labels(df)
        assert eligible.sum() == 0

    def test_all_eligible_values(self):
        df = self._df([1, 2, 3, 4, 5])
        t = make_task("srh_binary")
        y, eligible = t.make_labels(df)
        assert eligible.sum() == 5

    def test_missing_column_returns_zeros(self):
        """If PHSTAT_A is absent, .get() returns None -> no eligible rows."""
        df = pd.DataFrame({"OTHER_A": [1, 2, 3]})
        t = make_task("srh_binary")
        y, eligible = t.make_labels(df)
        # No eligible rows because PHSTAT_A values are NaN
        assert eligible.sum() == 0

    def test_string_values_coerced(self):
        """Numeric strings should still work via pd.to_numeric."""
        df = self._df(["4", "2", "5"])
        t = make_task("srh_binary")
        y, eligible = t.make_labels(df)
        assert list(y) == [1, 0, 1]

    def test_output_shapes(self):
        df = self._df([1, 2, 3, 4, 5, 7])
        t = make_task("srh_binary")
        y, eligible = t.make_labels(df)
        assert len(y) == len(df)
        assert len(eligible) == len(df)


# ---------------------------------------------------------------------------
# smoking_current label generation
# ---------------------------------------------------------------------------

class TestSmokingCurrent:
    def test_primary_column_smokers(self):
        """SMKCIGST_A 1,2 -> current smoker (1); 3,4 -> non-smoker (0)."""
        df = pd.DataFrame({"SMKCIGST_A": [1, 2, 3, 4]})
        t = make_task("smoking_current")
        y, eligible = t.make_labels(df)
        assert list(y) == [1, 1, 0, 0]
        assert eligible.sum() == 4

    def test_primary_column_eligibility(self):
        """SMKCIGST_A outside {1,2,3,4} -> ineligible."""
        df = pd.DataFrame({"SMKCIGST_A": [1, 7, 9]})
        t = make_task("smoking_current")
        y, eligible = t.make_labels(df)
        # rows 1 and 2 (SMKCIGST_A 7, 9) are not in {1,2,3,4}
        assert eligible[0] == True
        assert eligible[1] == False
        assert eligible[2] == False

    def test_fallback_column_used_when_primary_absent(self):
        """Without SMKCIGST_A, falls back to SMKNOW_A: 1,2->1; 3->0."""
        df = pd.DataFrame({"SMKNOW_A": [1, 2, 3]})
        t = make_task("smoking_current")
        y, eligible = t.make_labels(df)
        assert list(y) == [1, 1, 0]
        assert eligible.sum() == 3

    def test_primary_takes_precedence(self):
        """Both columns present -> SMKCIGST_A is used."""
        df = pd.DataFrame({"SMKCIGST_A": [1, 4], "SMKNOW_A": [3, 1]})
        t = make_task("smoking_current")
        y, eligible = t.make_labels(df)
        # Based on SMKCIGST_A: 1->1, 4->0
        assert list(y) == [1, 0]

    def test_output_dtype_integer(self):
        df = pd.DataFrame({"SMKCIGST_A": [1, 2, 3, 4]})
        t = make_task("smoking_current")
        y, eligible = t.make_labels(df)
        assert y.dtype in (np.int32, np.int64, int)
