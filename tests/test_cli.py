"""
Tests for nhisml.cli — CLI entry-point, subcommand dispatch,
and informational commands (list-tasks, describe-task, etc.).
These tests do not require downloading NHIS data.
"""
from __future__ import annotations

import pytest

from nhisml.cli import main


# ---------------------------------------------------------------------------
# list-tasks
# ---------------------------------------------------------------------------

class TestListTasks:
    def test_list_tasks_prints_srh(self, capsys):
        main(["list-tasks"])
        out = capsys.readouterr().out
        assert "srh_binary" in out

    def test_list_tasks_prints_smoking(self, capsys):
        main(["list-tasks"])
        out = capsys.readouterr().out
        assert "smoking_current" in out

    def test_list_tasks_one_per_line(self, capsys):
        main(["list-tasks"])
        out = capsys.readouterr().out.strip()
        lines = out.split("\n")
        assert len(lines) >= 2


# ---------------------------------------------------------------------------
# describe-task
# ---------------------------------------------------------------------------

class TestDescribeTask:
    def test_describe_srh(self, capsys):
        main(["describe-task", "srh_binary"])
        out = capsys.readouterr().out
        assert "srh_binary" in out
        assert "binary" in out

    def test_describe_smoking(self, capsys):
        main(["describe-task", "smoking_current"])
        out = capsys.readouterr().out
        assert "smoking_current" in out

    def test_describe_shows_required_cols(self, capsys):
        main(["describe-task", "srh_binary"])
        out = capsys.readouterr().out
        assert "PHSTAT_A" in out

    def test_describe_unknown_task_raises(self):
        with pytest.raises(SystemExit):
            main(["describe-task", "nonexistent_task"])


# ---------------------------------------------------------------------------
# list-featuresets
# ---------------------------------------------------------------------------

class TestListFeaturesets:
    def test_list_featuresets_prints_core(self, capsys):
        main(["list-featuresets"])
        out = capsys.readouterr().out
        assert "core" in out


# ---------------------------------------------------------------------------
# describe-featureset
# ---------------------------------------------------------------------------

class TestDescribeFeatureset:
    def test_describe_core(self, capsys):
        main(["describe-featureset", "core"])
        out = capsys.readouterr().out
        assert "core" in out
        assert "binary_12" in out or "n_binary" in out or "HYPEV_A" in out

    def test_describe_core_shows_column_counts(self, capsys):
        main(["describe-featureset", "core"])
        out = capsys.readouterr().out
        # Should report counts for each column group
        assert "n_binary" in out or "n_ordinal" in out

    def test_describe_unknown_featureset_raises(self):
        with pytest.raises(SystemExit):
            main(["describe-featureset", "nonexistent_fs"])


# ---------------------------------------------------------------------------
# validate-data — wiring check (no real data needed)
# ---------------------------------------------------------------------------

class TestValidateDataCli:
    def test_missing_data_file_exits_nonzero(self, tmp_path):
        """validate-data should exit 1 when the parquet does not exist."""
        with pytest.raises(SystemExit) as exc_info:
            main(["validate-data", "--year", "2023", "--data-dir", str(tmp_path)])
        assert exc_info.value.code != 0

    def test_unknown_year_exits_nonzero(self, tmp_path):
        """validate-data should exit 1 for a year with no reference statistics."""
        import pandas as pd
        # Create a dummy parquet so the file-not-found path is skipped
        pd.DataFrame({"WTFA_A": [1.0]}).to_parquet(tmp_path / "core_1990.parquet")
        with pytest.raises(SystemExit) as exc_info:
            main(["validate-data", "--year", "1990", "--data-dir", str(tmp_path)])
        assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# Top-level dispatch — missing subcommand
# ---------------------------------------------------------------------------

class TestTopLevel:
    def test_no_subcommand_exits(self):
        with pytest.raises(SystemExit):
            main([])

    def test_unknown_subcommand_exits(self):
        with pytest.raises(SystemExit):
            main(["not-a-command"])
