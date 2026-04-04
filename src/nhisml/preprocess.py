# src/nhisml/preprocess.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Survey weights helper
def normalize_weights(w: pd.Series) -> np.ndarray:
    """
    Normalize weights to mean 1 for numerical stability.
    Returns float64 ndarray.
    """
    w = pd.to_numeric(w, errors="coerce").fillna(0.0).clip(lower=0.0).astype(float)
    m = float(w.mean()) if float(w.mean()) > 0 else 1.0
    return (w / m).to_numpy(dtype=float)


# NHIS missing codes + helpers
NHIS_MISSING = {
    7: np.nan, 8: np.nan, 9: np.nan,
    97: np.nan, 98: np.nan, 99: np.nan,
}


def _map_missing(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    inter = [c for c in cols if c in out.columns]
    if inter:
        out[inter] = out[inter].replace(NHIS_MISSING)
    return out


def _recode_binary_12(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Map 1 -> 1.0, 2 -> 0.0 for binary 'Yes/No' items; keep NaN."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            s = pd.to_numeric(out[c], errors="coerce").replace(NHIS_MISSING)
            out[c] = s.map({1: 1.0, 2: 0.0})
    return out


def _as_cat_str(s: pd.Series) -> pd.Series:
    """
    Convert to consistent object/str dtype while preserving NaN.
    """
    s = s.replace(NHIS_MISSING).astype("object")
    mask = pd.notna(s)
    if mask.any():
        s.loc[mask] = s.loc[mask].astype(str)
    return s

# Schema object
@dataclass
class PreprocessSchema:
    binary_cols: List[str]
    ordinal_cols: List[str]
    categorical_cols: List[str]
    rare_min_count: int
    add_missing_flags: bool
    missing_flag_min_frac: float
    added_missing_flags: List[str]
    categorical_levels: Dict[str, List[str]]
    scaler_stats: Dict[str, Dict[str, float]]

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)


# PrepareFrame
class PrepareFrame(BaseEstimator, TransformerMixin):
    """
    Learns:
      - which requested columns exist
      - categorical levels with optional rare bucketing via "__RARE__"
    Applies:
      - NHIS missing mapping
      - binary 1/2 -> 1/0
      - ordinal coercion to numeric (stability)
      - categorical casting + rare/unseen mapping
      - missingness flags (stable schema: one per ordinal/categorical col)
    """
    def __init__(
        self,
        binary_cols: List[str],
        ordinal_cols: List[str],
        categorical_cols: List[str],
        rare_min_count: int = 50,
        add_missing_flags: bool = True,
        missing_flag_min_frac: float = 0.20,  # retained for backward compatibility
    ):
        self.binary_cols = binary_cols
        self.ordinal_cols = ordinal_cols
        self.categorical_cols = categorical_cols
        self.rare_min_count = rare_min_count
        self.add_missing_flags = add_missing_flags
        self.missing_flag_min_frac = missing_flag_min_frac

        # learned
        self.binary_cols_: List[str] = []
        self.ordinal_cols_: List[str] = []
        self.categorical_cols_: List[str] = []
        self.added_missing_flags_: List[str] = []
        self.categorical_levels_: Dict[str, List[str]] = {}

    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()

        bin_cols = [c for c in (self.binary_cols or []) if c in df.columns]
        ord_cols = [c for c in (self.ordinal_cols or []) if c in df.columns]
        cat_cols = [c for c in (self.categorical_cols or []) if c in df.columns]

        all_cols = list(set(bin_cols + ord_cols + cat_cols))
        df = _map_missing(df, all_cols)


        for c in ord_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = _recode_binary_12(df, bin_cols)

        self.binary_cols_ = bin_cols
        self.ordinal_cols_ = ord_cols
        self.categorical_cols_ = cat_cols

        # stable schema: one flag per ordinal/categorical column
        self.added_missing_flags_ = []
        if self.add_missing_flags:
            for c in (ord_cols + cat_cols):
                self.added_missing_flags_.append(f"{c}__ismissing")

        # learn categorical levels after rare decision
        self.categorical_levels_ = {}
        for c in cat_cols:
            s = _as_cat_str(df[c])
            vc = s.value_counts(dropna=True)
            keep = set(vc[vc >= int(self.rare_min_count)].index.astype(str))
            levels = sorted(list(keep))
            if (len(vc) - len(keep)) > 0:
                levels = sorted(list(keep | {"__RARE__"}))
            self.categorical_levels_[c] = levels

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        bin_cols = [c for c in self.binary_cols_ if c in df.columns]
        ord_cols = [c for c in self.ordinal_cols_ if c in df.columns]
        cat_cols = [c for c in self.categorical_cols_ if c in df.columns]

        all_cols = list(set(bin_cols + ord_cols + cat_cols))
        df = _map_missing(df, all_cols)

        # coerce ordinals to numeric on transform as well
        for c in ord_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = _recode_binary_12(df, bin_cols)

        # missing flags
        for flag in self.added_missing_flags_:
            base = flag.replace("__ismissing", "")
            if base in df.columns:
                df[flag] = df[base].isna().astype("float32")
            else:
                df[flag] = 0.0

        # categoricals with rare/unseen mapping
        for c in cat_cols:
            s = _as_cat_str(df[c])
            levels = self.categorical_levels_.get(c)
            if levels:
                level_set = set(levels)
                has_rare = "__RARE__" in level_set

                def map_level(val):
                    if pd.isna(val):
                        return np.nan
                    v = str(val)
                    if (v not in level_set) and has_rare:
                        return "__RARE__"
                    return v

                s = s.map(map_level)

            df[c] = s

        return df

    def output_missing_flags(self) -> List[str]:
        return list(self.added_missing_flags_)

# Full preprocessor (UNFITTED)
def build_preprocessor(
    binary_cols: List[str],
    ordinal_cols: List[str],
    categorical_cols: List[str],
    rare_min_count: int = 50,
    add_missing_flags: bool = True,
    missing_flag_min_frac: float = 0.20,
) -> Pipeline:
    """
    Returns an UNFITTED Pipeline:
      ("frame" -> PrepareFrame, "ct" -> ColumnTransformer)

    Fit happens exactly once when the full model Pipeline is fit.
    """
    frame = PrepareFrame(
        binary_cols=binary_cols,
        ordinal_cols=ordinal_cols,
        categorical_cols=categorical_cols,
        rare_min_count=rare_min_count,
        add_missing_flags=add_missing_flags,
        missing_flag_min_frac=missing_flag_min_frac,
    )

    # Missing flags are always created for ordinal+categorical columns (stable schema)
    missing_flag_cols = [f"{c}__ismissing" for c in (ordinal_cols + categorical_cols)]

    ord_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    bin_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent"))])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    ct = ColumnTransformer(
        transformers=[
            ("bin", bin_pipe, list(binary_cols)),
            ("ord", ord_pipe, list(ordinal_cols)),
            ("cat", cat_pipe, list(categorical_cols)),
            ("miss", "passthrough", missing_flag_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=False,
    )

    return Pipeline([("frame", frame), ("ct", ct)])


# Schema extraction after fitting
def build_schema_from_fitted(preproc: Pipeline) -> PreprocessSchema:
    if not isinstance(preproc, Pipeline) or "frame" not in preproc.named_steps or "ct" not in preproc.named_steps:
        raise ValueError("Expected a fitted Pipeline with 'frame' and 'ct' steps.")

    frame: PrepareFrame = preproc.named_steps["frame"]
    ct: ColumnTransformer = preproc.named_steps["ct"]

    scaler_stats: Dict[str, Dict[str, float]] = {}
    if "ord" in ct.named_transformers_:
        ord_t = ct.named_transformers_["ord"]
        try:
            sc = ord_t.named_steps["sc"]
            mean = getattr(sc, "mean_", None)
            scale = getattr(sc, "scale_", None)
            cols = ct.transformers_[1][2]  # ("ord", ..., cols)
            if mean is not None and scale is not None:
                for c, m, s in zip(list(cols), mean.tolist(), scale.tolist()):
                    scaler_stats[str(c)] = {"mean": float(m), "std": float(s)}
        except Exception:
            pass

    return PreprocessSchema(
        binary_cols=list(frame.binary_cols_),
        ordinal_cols=list(frame.ordinal_cols_),
        categorical_cols=list(frame.categorical_cols_),
        rare_min_count=int(frame.rare_min_count),
        add_missing_flags=bool(frame.add_missing_flags),
        missing_flag_min_frac=float(frame.missing_flag_min_frac),
        added_missing_flags=list(frame.added_missing_flags_),
        categorical_levels=dict(frame.categorical_levels_),
        scaler_stats=scaler_stats,
    )


# Feature name extraction
def get_feature_names(preproc: Pipeline) -> List[str]:
    if not isinstance(preproc, Pipeline) or "ct" not in preproc.named_steps:
        raise ValueError("Expected a fitted Pipeline with a 'ct' step.")

    ct: ColumnTransformer = preproc.named_steps["ct"]

    try:
        return ct.get_feature_names_out().tolist()
    except Exception:
        pass

    names: List[str] = []
    for name, trans, cols in ct.transformers_:
        if name == "cat":
            ohe = trans.named_steps["ohe"]
            names.extend(ohe.get_feature_names_out(cols).tolist())
        elif name in ("bin", "ord", "miss"):
            names.extend(list(cols))
    return names
