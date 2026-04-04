from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Task:
    """
    Definition of a prediction task.

    - required_cols: columns that MUST be present in the core parquet
      for this task to be valid.
    - make_labels: function returning (y, eligible_mask).
      eligible_mask can be None to indicate all rows eligible.
    """
    name: str
    problem_type: str  # currently only "binary"
    description: str
    required_cols: List[str]
    make_labels: Callable[[pd.DataFrame], Tuple[np.ndarray, Optional[np.ndarray]]]

    # Compatibility helper (used by train/build-core)
    def required_columns(self) -> List[str]:
        return self.required_cols


# Helpers
def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

# SRH binary task
def _srh_binary(df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Self-rated health (binary):
      PHSTAT_A >= 4  -> Fair/Poor (1)
      PHSTAT_A <= 3  -> Excellent/Very good/Good (0)

    Eligible: PHSTAT_A in {1,2,3,4,5}
    """
    s = _to_num(df.get("PHSTAT_A"))
    eligible = s.isin([1, 2, 3, 4, 5]).to_numpy()
    y = (s >= 4).astype(int).fillna(0).to_numpy()
    return y, eligible


# Smoking current task

def _smoking_current(df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Current cigarette smoker (binary).

    Primary definition:
      SMKCIGST_A:
        1 Current every day smoker -> 1
        2 Current some day smoker  -> 1
        3 Former smoker            -> 0
        4 Never smoker             -> 0
      Eligible: values in {1,2,3,4}

    Fallback definition:
      SMKNOW_A:
        1 Every day -> 1
        2 Some days -> 1
        3 Not at all -> 0
      Eligible: values in {1,2,3}
    """
    if "SMKCIGST_A" in df.columns:
        s = _to_num(df["SMKCIGST_A"])
        eligible = s.isin([1, 2, 3, 4]).to_numpy()
        y = s.map({1: 1, 2: 1, 3: 0, 4: 0}).fillna(0).astype(int).to_numpy()
        return y, eligible

    # Fallback if derived status is not present
    s = _to_num(df.get("SMKNOW_A"))
    eligible = s.isin([1, 2, 3]).to_numpy()
    y = s.map({1: 1, 2: 1, 3: 0}).fillna(0).astype(int).to_numpy()
    return y, eligible

# Task registry
_TASKS = {
    "srh_binary": Task(
        name="srh_binary",
        problem_type="binary",
        description="Self-rated health: Fair/Poor vs Excellent/Very good/Good (PHSTAT_A >= 4).",
        required_cols=["PHSTAT_A"],
        make_labels=_srh_binary,
    ),
    "smoking_current": Task(
        name="smoking_current",
        problem_type="binary",
        description="Current cigarette smoker (SMKCIGST_A primary, SMKNOW_A fallback).",
        required_cols=["SMKCIGST_A", "SMKNOW_A"],
        make_labels=_smoking_current,
    ),
}

# Public API
def make_task(name: str) -> Task:
    try:
        return _TASKS[name]
    except KeyError:
        raise ValueError(
            f"Unknown task '{name}'. Available tasks: {', '.join(sorted(_TASKS))}"
        )


def list_tasks() -> List[str]:
    return sorted(_TASKS.keys())
