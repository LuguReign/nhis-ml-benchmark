"""
nhisml — Survey-aware machine learning toolkit for NHIS Adults data.

Public API
----------
Tasks (prediction targets)
    list_tasks()           → list of available task names
    make_task(name)        → Task object with label-generation logic

Feature sets (predictor column groups)
    list_featuresets()     → list of available feature set names
    get_featureset(name)   → FeatureSet dataclass

Preprocessing
    build_preprocessor(...)        → unfitted sklearn Pipeline
    normalize_weights(series)      → weight array normalized to mean 1
    get_feature_names(pipeline)    → list of output feature names
    build_schema_from_fitted(pipe) → PreprocessSchema (serializable)

Data utilities
    fetch_year(year, ...)           → download raw NHIS zip
    build_core_year(year, ...)      → build clean core parquet

Model utilities (lower-level, used by the CLI)
    weighted_threshold_via_oof(...)  → (threshold, perf_dict, oof_probs)
    fit_calibrated_from_oof(...)     → (calibrated_model, threshold, perf, oof)
"""

from .tasks import Task, list_tasks, make_task
from .featuresets import FeatureSet, get_featureset, list_featuresets
from .preprocess import (
    PrepareFrame,
    PreprocessSchema,
    build_preprocessor,
    build_schema_from_fitted,
    get_feature_names,
    normalize_weights,
)
from .fetch import fetch_year
from .build_core import build_core_year
from .utils import (
    fit_calibrated_from_oof,
    oof_proba,
    pick_threshold_max_f1,
    weighted_threshold_via_oof,
)

__version__ = "0.5.1"

__all__ = [
    # tasks
    "Task",
    "list_tasks",
    "make_task",
    # featuresets
    "FeatureSet",
    "get_featureset",
    "list_featuresets",
    # preprocessing
    "PrepareFrame",
    "PreprocessSchema",
    "build_preprocessor",
    "build_schema_from_fitted",
    "get_feature_names",
    "normalize_weights",
    # data utilities
    "fetch_year",
    "build_core_year",
    # model utilities
    "fit_calibrated_from_oof",
    "oof_proba",
    "pick_threshold_max_f1",
    "weighted_threshold_via_oof",
    # version
    "__version__",
]
