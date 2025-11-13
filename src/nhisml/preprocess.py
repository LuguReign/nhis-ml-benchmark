# src/nhisml/preprocess.py
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -------------------------------
# Survey weights helper
# -------------------------------
def normalize_weights(w: pd.Series) -> pd.Series:
    w = pd.to_numeric(w, errors="coerce").fillna(0).clip(lower=0)
    m = w.mean() or 1.0
    return w / m


# -------------------------------
# NHIS missing codes + helpers
# -------------------------------
NHIS_MISSING = {
    7: np.nan, 8: np.nan, 9: np.nan,
    97: np.nan, 98: np.nan, 99: np.nan
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
            out[c] = pd.to_numeric(out[c], errors="coerce")
            out[c] = out[c].replace(NHIS_MISSING)
            out[c] = out[c].map({1: 1.0, 2: 0.0})
    return out

def _as_cat_str(s: pd.Series) -> pd.Series:
    """
    Convert series to uniform object/str dtype (preserving NaN) so OHE
    sees a single dtype and we avoid mixed-type issues + FutureWarnings.
    """
    s = s.replace(NHIS_MISSING)
    s = s.astype("object")
    mask = pd.notna(s)
    if mask.any():
        s.loc[mask] = s.loc[mask].astype(str)
    return s


# -------------------------------
# Schema object (for reproducibility/debug)
# -------------------------------
@dataclass
class PreprocessSchema:
    binary_cols: List[str]
    ordinal_cols: List[str]
    categorical_cols: List[str]
    rare_min_count: int
    add_missing_flags: bool
    missing_flag_min_frac: float
    added_missing_flags: List[str]
    categorical_levels: Dict[str, List[str]]  # learned category levels (including "__RARE__" if present)
    scaler_stats: Dict[str, Dict[str, float]]  # per-ordinal mean/std (post-imputation)

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)


# -------------------------------
# PrepareFrame: train-time learning & transform-time application
# -------------------------------
class PrepareFrame(BaseEstimator, TransformerMixin):
    """
    A lightweight, clone-safe frame preprocessor that:
      - applies NHIS missing mapping
      - recodes binary 1/2 -> 1/0
      - learns categorical levels w/ rare-bucketing on TRAIN
      - adds missingness flags (selected on TRAIN) for specified cols
      - returns a DataFrame with the original feature columns (plus selected flags)
        ready for ColumnTransformer (impute/scale/OHE).
    """
    def __init__(
        self,
        binary_cols: List[str],
        ordinal_cols: List[str],
        categorical_cols: List[str],
        rare_min_count: int = 50,
        add_missing_flags: bool = True,
        missing_flag_min_frac: float = 0.20,
    ):
        # IMPORTANT: store params EXACTLY as passed (clone-safe)
        self.binary_cols = binary_cols
        self.ordinal_cols = ordinal_cols
        self.categorical_cols = categorical_cols
        self.rare_min_count = rare_min_count
        self.add_missing_flags = add_missing_flags
        self.missing_flag_min_frac = missing_flag_min_frac

        # learned in fit
        self.binary_cols_: List[str] = []
        self.ordinal_cols_: List[str] = []
        self.categorical_cols_: List[str] = []
        self.added_missing_flags_: List[str] = []
        self.categorical_levels_: Dict[str, List[str]] = {}

    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()

        # determine which columns actually exist
        bin_cols = [c for c in (self.binary_cols or []) if c in df.columns]
        ord_cols = [c for c in (self.ordinal_cols or []) if c in df.columns]
        cat_cols = [c for c in (self.categorical_cols or []) if c in df.columns]

        # map missing & recode binaries on a working copy
        all_cols = list(set((self.binary_cols or []) + (self.ordinal_cols or []) + (self.categorical_cols or [])))
        df = _map_missing(df, all_cols)
        df = _recode_binary_12(df, bin_cols)

        self.binary_cols_ = bin_cols
        self.ordinal_cols_ = ord_cols
        self.categorical_cols_ = cat_cols

        # learn which missingness flags to add
        self.added_missing_flags_ = []
        if self.add_missing_flags:
            for c in (ord_cols + cat_cols):
                if c in df.columns:
                    frac = df[c].isna().mean()
                    if frac >= float(self.missing_flag_min_frac):
                        self.added_missing_flags_.append(f"{c}__ismissing")

        # learn categorical levels (post rare-bucketing decision)
        self.categorical_levels_ = {}
        for c in cat_cols:
            s = _as_cat_str(df[c])
            vc = s.value_counts(dropna=True)
            keep = set(vc[vc >= int(self.rare_min_count)].index.astype(str))
            levels = sorted(list(keep))
            if (len(vc) - len(keep)) > 0:
                # we will route rare/unseen to "__RARE__"
                levels = sorted(list(keep | {"__RARE__"}))
            self.categorical_levels_[c] = levels

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # operate on available cols only
        bin_cols = [c for c in self.binary_cols_ if c in df.columns]
        ord_cols = [c for c in self.ordinal_cols_ if c in df.columns]
        cat_cols = [c for c in self.categorical_cols_ if c in df.columns]

        # map missing and recode binaries on transform as well
        all_cols = list(set(bin_cols + ord_cols + cat_cols))
        df = _map_missing(df, all_cols)
        df = _recode_binary_12(df, bin_cols)

        # create missingness flag columns decided at train
        for flag in self.added_missing_flags_:
            base = flag.replace("__ismissing", "")
            if base in df.columns:
                df[flag] = df[base].isna().astype("float32")
            else:
                # if base missing entirely, still create the flag (all zeros)
                df[flag] = 0.0

        # cast categoricals + apply rare-bucketing using learned levels
        for c in cat_cols:
            s = _as_cat_str(df[c])
            levels = self.categorical_levels_.get(c, None)
            if levels is not None and len(levels) > 0:
                has_rare = "__RARE__" in set(levels)
                # map unseen to "__RARE__" if we learned a rare bucket, else keep as-is
                def map_level(val):
                    if pd.isna(val):
                        return np.nan
                    val = str(val)
                    if (val not in levels) and has_rare:
                        return "__RARE__"
                    return val
                s = s.map(map_level)
            df[c] = s

        return df


# -------------------------------
# Build the full preprocessor (Pipeline(PrepareFrame -> ColumnTransformer))
# -------------------------------
def build_preprocessor(
    train_core: pd.DataFrame,
    binary_cols: List[str],
    ordinal_cols: List[str],
    categorical_cols: List[str],
    rare_min_count: int = 50,
    add_missing_flags: bool = True,
    missing_flag_min_frac: float = 0.20,
) -> Tuple[Pipeline, PreprocessSchema, pd.DataFrame]:
    """
    Returns:
      - fitted Pipeline: ("frame" -> PrepareFrame, "ct" -> ColumnTransformer)
      - PreprocessSchema with learned artifacts
      - df_fit: the frame returned by "frame".fit_transform(train_core) (for debugging)
    """
    frame = PrepareFrame(
        binary_cols=binary_cols,
        ordinal_cols=ordinal_cols,
        categorical_cols=categorical_cols,
        rare_min_count=rare_min_count,
        add_missing_flags=add_missing_flags,
        missing_flag_min_frac=missing_flag_min_frac,
    )

    # The ColumnTransformer uses plain column-name lists
    ord_imputer = SimpleImputer(strategy="median")
    ord_scaler = StandardScaler(with_mean=True, with_std=True)

    bin_imputer = SimpleImputer(strategy="most_frequent")

    cat_imputer = SimpleImputer(strategy="most_frequent")
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)

    # Build a temporary frame to figure out which columns will exist after PrepareFrame
    df_fit = frame.fit_transform(train_core)

    # figure out which columns to route to each branch (post PrepareFrame)
    bin_cols_exist = [c for c in binary_cols if c in df_fit.columns]
    ord_cols_exist = [c for c in ordinal_cols if c in df_fit.columns]
    cat_cols_exist = [c for c in categorical_cols if c in df_fit.columns]

    # the missing flags we added in PrepareFrame.fit
    miss_flags_exist = [f for f in frame.added_missing_flags_ if f in df_fit.columns]

    transformers = []
    if bin_cols_exist:
        transformers.append(
            ("bin", Pipeline([("imp", bin_imputer)]), bin_cols_exist)
        )
    if ord_cols_exist:
        transformers.append(
            ("ord", Pipeline([("imp", ord_imputer), ("sc", ord_scaler)]), ord_cols_exist)
        )
    if cat_cols_exist:
        transformers.append(
            ("cat", Pipeline([("imp", cat_imputer), ("ohe", ohe)]), cat_cols_exist)
        )
    if miss_flags_exist:
        # pass missing flags through as-is
        transformers.append(("miss", "passthrough", miss_flags_exist))

    ct = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
        n_jobs=None,
        verbose_feature_names_out=False,
    )

    pipe = Pipeline([("frame", frame), ("ct", ct)])
    # fit ct on the frame-transformed data
    pipe.named_steps["ct"].fit(df_fit)

    # build schema (for debug/portability)
    # Extract scaler stats for ordinals
    scaler_stats: Dict[str, Dict[str, float]] = {}
    if ord_cols_exist:
        sc = pipe.named_steps["ct"].named_transformers_["ord"].named_steps["sc"]
        mean = getattr(sc, "mean_", None)
        scale = getattr(sc, "scale_", None)
        if mean is not None and scale is not None:
            for c, m, s in zip(ord_cols_exist, mean.tolist(), scale.tolist()):
                scaler_stats[c] = {"mean": float(m), "std": float(s)}

    schema = PreprocessSchema(
        binary_cols=binary_cols,
        ordinal_cols=ordinal_cols,
        categorical_cols=categorical_cols,
        rare_min_count=rare_min_count,
        add_missing_flags=add_missing_flags,
        missing_flag_min_frac=missing_flag_min_frac,
        added_missing_flags=frame.added_missing_flags_,
        categorical_levels=frame.categorical_levels_,
        scaler_stats=scaler_stats,
    )

    return pipe, schema, df_fit


# -------------------------------
# Feature name extraction
# -------------------------------
def get_feature_names(pipe: Pipeline) -> List[str]:
    """
    Returns the feature names after the full pipeline ("frame" -> "ct").
    Works after the pipeline is fitted.
    """
    if not isinstance(pipe, Pipeline) or "ct" not in pipe.named_steps:
        raise ValueError("Expected a fitted Pipeline with a 'ct' step.")
    ct: ColumnTransformer = pipe.named_steps["ct"]
    names: List[str] = []
    for name, trans, cols in ct.transformers_:
        if name == "cat":
            ohe = trans.named_steps["ohe"]
            # sklearn will expand per input column; output names reflect input names
            names.extend(ohe.get_feature_names_out(cols).tolist())
        elif name in ("bin", "ord"):
            # after impute/scale the columns persist with their original names
            names.extend(list(cols))
        elif name == "miss":
            names.extend(list(cols))
        # ignore remainder='drop'
    return names
