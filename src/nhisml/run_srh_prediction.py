# file: src/run_srh_prediction.py
# NHIS SRH prediction: Adults23 -> Adults24
# Pipeline: preprocess -> (L1 or RF) -> calibrate -> OOF threshold -> external eval -> lite interpretation

import warnings
warnings.filterwarnings("ignore")

import os
import copy
import datetime
from inspect import signature
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from packaging import version
import sklearn

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

RNG = 42
np.random.seed(RNG)

# =========================
# CONFIG (edit paths as needed)
# =========================
DATA_TRAIN = "/Users/lugu_reign/Desktop/RA PROJECT/adult24csv/Adults23_corefeatures.parquet"
DATA_TEST  = "/Users/lugu_reign/Desktop/RA PROJECT/adult24csv/Adults24_corefeatures.parquet"
WEIGHT_COL = "WTFA_A"
TARGET_RAW = "PHSTAT_A"  # 1..5

# Models
RF_CLASS_WEIGHT = "balanced_subsample"    # or None
LOGIT_CLASS_WEIGHT = "balanced"           # or None
CALIBRATION_METHOD = "isotonic"           # "isotonic" or "sigmoid"
CALIBRATION_CV = 5
N_SPLITS_OOF = 5

# =========================
# FEATURES
# =========================
FEATURES = [
    "RATCAT_A","POVRATTC_A",
    "EDUCP_A","MAXEDUCP_A",
    "EMPWKHRS3_A","EMPWRKFT1_A","EMPHEALINS_A","EMPSICKLV_A","EMPLASTWK_A",
    "EMPNOWRK_A","EMPWHENWRK_A","EMDSUPER_A",
    "MARITAL_A","MARSTAT_A","LONELY_A","SUPPORT_A","URBRRL23","REGION",
    "FDSCAT3_A","FDSCAT4_A",
    "DISAB3_A","ANYDIFF_A","DIFF_A","COGMEMDFF_A","COMDIFF_A","VISIONDF_A","HEARINGDF_A",
    "K6SPD_A","WORTHLESS_A","HOPELESS_A","SAD_A","NERVOUS_A","RESTLESS_A","EFFORT_A",
    "DEPFREQ_A","ANXFREQ_A","DEPLEVEL_A","DEPMED_A","ANXMED_A","MHRX_A","MHTHRPY_A",
    "MHTHDLY_A","MHTHND_A",
    "HYPEV_A","DIBEV_A","CHDEV_A","MIEV_A","STREV_A","ANGEV_A",
    "ASEV_A","ASTILL_A","ARTHEV_A","COPDEV_A","CANEV_A",
    "CHLEV_A","CHL12M_A","HYP12M_A","HYPMED_A","KIDWEAKEV_A","LIVEREV_A","HEPEV_A",
    "CROHNSEV_A","ULCCOLEV_A","PSOREV_A","CFSNOW_A",
    "HICOV_A","USUALPL_A","MEDNG12M_A","MEDDL12M_A","RXDG12M_A","LASTDR_A","WELLVIS_A"
]

BINARY_12 = [
    "EMPWRKFT1_A","EMPHEALINS_A","EMPSICKLV_A","EMPLASTWK_A","DISAB3_A","ANYDIFF_A",
    "DIFF_A","COGMEMDFF_A","COMDIFF_A","VISIONDF_A","HEARINGDF_A",
    "K6SPD_A","DEPMED_A","ANXMED_A","MHRX_A","MHTHRPY_A","MHTHDLY_A","MHTHND_A",
    "HYPEV_A","DIBEV_A","CHDEV_A","MIEV_A","STREV_A","ANGEV_A","ASEV_A","ASTILL_A",
    "ARTHEV_A","COPDEV_A","CANEV_A","CHLEV_A","CHL12M_A","HYP12M_A","HYPMED_A",
    "KIDWEAKEV_A","LIVEREV_A","HEPEV_A","CROHNSEV_A","ULCCOLEV_A","PSOREV_A","CFSNOW_A",
    "HICOV_A","USUALPL_A","MEDNG12M_A","MEDDL12M_A","RXDG12M_A","EMDSUPER_A"
]

ORDINAL_NUMERIC = [
    "RATCAT_A","POVRATTC_A","EDUCP_A","MAXEDUCP_A",
    "LONELY_A","SUPPORT_A","FDSCAT3_A","FDSCAT4_A",
    "WORTHLESS_A","HOPELESS_A","SAD_A","NERVOUS_A","RESTLESS_A","EFFORT_A",
    "DEPFREQ_A","ANXFREQ_A","DEPLEVEL_A",
    "EMPWKHRS3_A","LASTDR_A","WELLVIS_A"
]

CATEGORICAL = ["MARITAL_A","MARSTAT_A","URBRRL23","REGION","EMPNOWRK_A","EMPWHENWRK_A"]

# =========================
# Pickle-safe top-level transformers
# =========================
def select_reindex_func(X, cols):
    import pandas as pd, numpy as np
    X = pd.DataFrame(X)
    return X.reindex(columns=cols, fill_value=np.nan)

def map_12_binary_func(X, cols):
    import pandas as pd, numpy as np
    out = pd.DataFrame(X, copy=True)
    for c in cols:
        if c in out.columns:
            out[c] = out[c].replace({1: 1.0, 2: 0.0})
            out[c] = out[c].where(out[c].isin([0.0, 1.0]), np.nan)
    return out

def clean_ordinals_func(X, cols):
    import pandas as pd, numpy as np
    out = pd.DataFrame(X, copy=True)
    for c in cols:
        if c in out.columns:
            out[c] = out[c].replace({7: np.nan, 8: np.nan, 9: np.nan})
    return out

# =========================
# General helpers
# =========================
def normalize_weights(w: pd.Series) -> np.ndarray:
    w = w.fillna(0).clip(lower=0)
    return (w / (w.mean() if w.mean() > 0 else 1.0)).to_numpy()

def make_binary_srh(df: pd.DataFrame, col="PHSTAT_A") -> pd.Series:
    x = df[col].replace({7: np.nan, 8: np.nan, 9: np.nan})
    return (x >= 4).astype("float64")  # 1 = Fair/Poor

def subset_existing(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]

def make_ohe() -> OneHotEncoder:
    # sklearn>=1.4 uses sparse_output; older uses sparse
    if version.parse(sklearn.__version__) >= version.parse("1.4"):
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    else:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

def feature_names_from_preprocessor(prep: ColumnTransformer, n_features: int) -> List[str]:
    """Ensure we always return exactly n_features names; fall back to generic."""
    names: List[str] = []
    try:
        names = list(prep.get_feature_names_out())
    except Exception:
        names = []
    if not names:
        tmp = []
        for name, trans, cols in getattr(prep, "transformers_", []):
            if name == "cat":
                try:
                    ohe = trans.named_steps["ohe"]
                    tmp.extend(ohe.get_feature_names_out().tolist())
                except Exception:
                    pass
            elif name in ("bin", "ord"):
                # Prefer the 'select' step's kw_args (exact list used)
                try:
                    sel = trans.named_steps.get("select", None)
                    if sel is not None and hasattr(sel, "kw_args"):
                        sel_cols = sel.kw_args.get("cols", None)
                        if sel_cols is not None:
                            tmp.extend(list(sel_cols))
                        else:
                            tmp.extend(list(cols) if isinstance(cols, list) else [])
                    else:
                        tmp.extend(list(cols) if isinstance(cols, list) else [])
                except Exception:
                    tmp.extend(list(cols) if isinstance(cols, list) else [])
        names = tmp
    if len(names) != n_features:
        names = [f"f{i}" for i in range(n_features)]
    return names

def class_prevalence(y: np.ndarray, w: np.ndarray) -> Dict[str, float]:
    p_unw = y.mean()
    p_w = (y * w).sum() / (w.sum() if w.sum() > 0 else 1.0)
    return {"positive_unweighted": float(p_unw), "positive_weighted": float(p_w)}

def fit_with_weights(pipeline: Pipeline, X, y, w):
    last_name, _ = pipeline.steps[-1]
    param = f"{last_name}__sample_weight"
    try:
        pipeline.fit(X, y, **{param: w})
    except TypeError:
        print(f"Note: '{last_name}' does not accept sample_weight; fitting unweighted.")
        pipeline.fit(X, y)

def fit_calibrated(cal: CalibratedClassifierCV, X, y, w):
    try:
        cal.fit(X, y, sample_weight=w)  # sklearn may ignore this; fine.
    except TypeError:
        print("Note: CalibratedClassifierCV.fit has no sample_weight; fitting unweighted.")
        cal.fit(X, y)

def oof_predict_proba(estimator, X: pd.DataFrame, y: np.ndarray, w: np.ndarray,
                      n_splits: int = 5) -> np.ndarray:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RNG)
    oof = np.zeros(len(y), dtype=float)
    for tr, va in skf.split(X, y):
        model = copy.deepcopy(estimator)
        if isinstance(model, Pipeline):
            try:
                fit_with_weights(model, X.iloc[tr], y[tr], w[tr])
            except Exception:
                model.fit(X.iloc[tr], y[tr])
        else:
            try:
                model.fit(X.iloc[tr], y[tr], sample_weight=w[tr])
            except TypeError:
                model.fit(X.iloc[tr], y[tr])
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
    return oof

def find_best_threshold(y_true: np.ndarray, p_hat: np.ndarray, w: np.ndarray,
                        metric: str = "f1") -> Tuple[float, float]:
    ths = np.linspace(0.05, 0.95, 19)
    best_score, best_t = -np.inf, 0.5
    for t in ths:
        yb = (p_hat >= t).astype(int)
        if metric == "f1":
            s = f1_score(y_true, yb, sample_weight=w)
        elif metric == "bal_acc":
            s = balanced_accuracy_score(y_true, yb)
        else:
            s = average_precision_score(y_true, p_hat, sample_weight=w)
        if s > best_score:
            best_score, best_t = s, t
    return best_score, best_t

def subgroup_report(y_true, p_hat, y_pred, w, group: pd.Series, name: str):
    if group is None or group.empty:
        return
    rows = []
    gser = group.astype(str)
    for g in sorted(gser.dropna().unique()):
        idx = gser.index[gser == g]
        yi, pi, wi, yi_hat = y_true[idx], p_hat[idx], w[idx], y_pred[idx]
        try:
            auc = roc_auc_score(yi, pi, sample_weight=wi)
        except Exception:
            auc = np.nan
        rows.append({
            name: g,
            "n": int(len(idx)),
            "prevalence_w": float((yi*wi).sum()/wi.sum() if wi.sum() > 0 else np.nan),
            "AUC_w": float(auc) if auc==auc else np.nan,
            "F1_w": float(f1_score(yi, yi_hat, sample_weight=wi)) if len(np.unique(yi_hat))>1 else np.nan,
            "BalAcc": float(balanced_accuracy_score(yi, yi_hat)) if len(np.unique(yi_hat))>1 else np.nan
        })
    df = pd.DataFrame(rows).sort_values("n", ascending=False)
    os.makedirs("artifacts", exist_ok=True)
    df.to_csv(f"artifacts/subgroup_{name}.csv", index=False)
    print(f"Saved artifacts/subgroup_{name}.csv")

def save_pipeline_safely(pipeline_or_model, path: str):
    """Try to joblib.dump(); if it fails, persist prep and estimator separately."""
    try:
        joblib.dump(pipeline_or_model, path)
        return
    except Exception as e:
        print(f"[warn] Could not pickle full object ({e}). Saving components.")
        if isinstance(pipeline_or_model, Pipeline):
            prep = pipeline_or_model.named_steps.get("prep", None)
            est_name, est = pipeline_or_model.steps[-1]
            if prep is not None:
                joblib.dump(prep, path.replace(".joblib", "_prep.joblib"))
            joblib.dump(est,  path.replace(".joblib", f"_{est_name}.joblib"))
        else:
            base = getattr(pipeline_or_model, "base_estimator", None)
            if base is not None:
                save_pipeline_safely(base, path.replace(".joblib", "_base.joblib"))

def ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[cols]

# =========================
# Build preprocess (pickle-safe)
# =========================
def build_preprocess(df_union: pd.DataFrame) -> Tuple[ColumnTransformer, List[str]]:
    bin_cols_all = subset_existing(df_union, BINARY_12)
    ord_cols_all = subset_existing(df_union, ORDINAL_NUMERIC)
    cat_cols_all = subset_existing(df_union, CATEGORICAL)

    binary_pipe = Pipeline([
        ("select", FunctionTransformer(select_reindex_func, kw_args={"cols": bin_cols_all}, validate=False)),
        ("map12",  FunctionTransformer(map_12_binary_func, kw_args={"cols": bin_cols_all}, validate=False)),
        ("imp",    SimpleImputer(strategy="most_frequent")),
    ])

    ordinal_pipe = Pipeline([
        ("select", FunctionTransformer(select_reindex_func, kw_args={"cols": ord_cols_all}, validate=False)),
        ("clean",  FunctionTransformer(clean_ordinals_func, kw_args={"cols": ord_cols_all}, validate=False)),
        ("imp",    SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("select", FunctionTransformer(select_reindex_func, kw_args={"cols": cat_cols_all}, validate=False)),
        ("imp",    SimpleImputer(strategy="most_frequent")),
        ("ohe",    make_ohe()),
    ])

    transformers = []
    if len(bin_cols_all): transformers.append(("bin", binary_pipe, bin_cols_all))
    if len(ord_cols_all): transformers.append(("ord", ordinal_pipe, ord_cols_all))
    if len(cat_cols_all): transformers.append(("cat", categorical_pipe, cat_cols_all))

    preprocess = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3
    )
    return preprocess, (bin_cols_all + ord_cols_all + cat_cols_all)

# =========================
# Data
# =========================
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(DATA_TRAIN)
    test  = pd.read_parquet(DATA_TEST)
    cols_needed = list(set(FEATURES + [WEIGHT_COL, TARGET_RAW]))
    train = train[[c for c in cols_needed if c in train.columns]].copy()
    test  = test[[c for c in cols_needed if c in test.columns]].copy()
    return train, test

# =========================
# Models
# =========================
def build_models(preprocess: ColumnTransformer) -> Dict[str, object]:
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=10,
        random_state=RNG, n_jobs=-1, class_weight=RF_CLASS_WEIGHT
    )
    l1 = LogisticRegression(
        penalty="l1", solver="saga", C=0.5, max_iter=2000,
        n_jobs=-1, random_state=RNG, class_weight=LOGIT_CLASS_WEIGHT
    )

    models = {
        "RF": Pipeline([("prep", preprocess), ("rf", rf)]),
        "L1": Pipeline([("prep", preprocess), ("lasso", l1)]),
    }
    models["RF-Cal"] = CalibratedClassifierCV(models["RF"], method=CALIBRATION_METHOD, cv=CALIBRATION_CV)
    models["L1-Cal"] = CalibratedClassifierCV(models["L1"], method=CALIBRATION_METHOD, cv=CALIBRATION_CV)
    return models

# =========================
# Evaluation
# =========================
def evaluate_on_test(model, Xtr, ytr, wtr, Xte, yte, wte, label: str, threshold: float = 0.5):
    if isinstance(model, Pipeline):
        fit_with_weights(model, Xtr, ytr, wtr)
    else:
        fit_calibrated(model, Xtr, ytr, wtr)

    p = model.predict_proba(Xte)[:, 1]
    ypred = (p >= threshold).astype(int)
    metrics = {
        "weighted_auc": roc_auc_score(yte, p, sample_weight=wte),
        "weighted_accuracy": accuracy_score(yte, ypred, sample_weight=wte),
        "balanced_accuracy": balanced_accuracy_score(yte, ypred),
        "weighted_f1": f1_score(yte, ypred, sample_weight=wte),
        "avg_precision": average_precision_score(yte, p, sample_weight=wte),
        "threshold": threshold
    }
    print(f"\n=== {label} (thr={threshold:.2f}) ===")
    for k, v in metrics.items():
        if k != "threshold":
            print(f"{k}: {v:.4f}")
    print("Confusion matrix (unweighted counts):\n", confusion_matrix(yte, ypred))
    return metrics, p, ypred

# =========================
# Interpretation (lite)
# =========================
def interpret_models(fitted_rf: Pipeline, fitted_l1: Pipeline):
    # RF importances
    rf = fitted_rf.named_steps["rf"]
    prep_rf = fitted_rf.named_steps["prep"]
    n_rf = rf.feature_importances_.shape[0]
    rf_names = feature_names_from_preprocessor(prep_rf, n_features=n_rf)
    rf_imp = pd.Series(rf.feature_importances_, index=rf_names).sort_values(ascending=False)
    print("\nTop 20 RF importances:")
    print(rf_imp.head(20))
    os.makedirs("artifacts", exist_ok=True)
    rf_imp.to_csv("artifacts/rf_importances.csv")

    # L1 coefficients
    l1 = fitted_l1.named_steps["lasso"]
    prep_l1 = fitted_l1.named_steps["prep"]
    if hasattr(l1, "coef_"):
        coef = l1.coef_.ravel()
        n_l1 = coef.shape[0]
        l1_names = feature_names_from_preprocessor(prep_l1, n_features=n_l1)
        l1_coefs = pd.Series(coef, index=l1_names).sort_values(key=np.abs, ascending=False)
        print("\nTop 20 |L1| coefficients:")
        print(l1_coefs.head(20))
        l1_coefs.to_csv("artifacts/l1_coefficients.csv")

# =========================
# MAIN
# =========================
def main():
    os.makedirs("artifacts", exist_ok=True)

    # Load
    train, test = load_data()

    # Targets & weights
    y_train = make_binary_srh(train).to_numpy()
    y_test  = make_binary_srh(test).to_numpy()
    w_train = normalize_weights(train[WEIGHT_COL]) if WEIGHT_COL in train.columns else np.ones(len(train))
    w_test  = normalize_weights(test[WEIGHT_COL])  if WEIGHT_COL  in test.columns  else np.ones(len(test))

    # Features schema (exclude target)
    all_model_cols = sorted(set(FEATURES) | set(BINARY_12) | set(ORDINAL_NUMERIC) | set(CATEGORICAL))
    if TARGET_RAW in all_model_cols:
        all_model_cols.remove(TARGET_RAW)

    X_train_raw = train[[c for c in FEATURES if c in train.columns]].copy()
    X_test_raw  = test[[c for c in FEATURES if c in test.columns]].copy()
    X_train = ensure_columns(X_train_raw, all_model_cols)
    X_test  = ensure_columns(X_test_raw,  all_model_cols)

    # Preprocess (fit on union schema)
    preprocess, used_cols = build_preprocess(pd.concat([X_train, X_test], axis=0, ignore_index=True))
    with open("artifacts/feature_groups.txt", "w") as f:
        f.write("USED_COLS:\n")
        for c in used_cols: f.write(f"{c}\n")

    # Prevalence snapshot
    pd.DataFrame(
        [class_prevalence(y_train, w_train), class_prevalence(y_test, w_test)],
        index=["train", "test"]
    ).to_csv("artifacts/class_prevalence.csv")

    # Build models (uncalibrated + calibrated)
    models = build_models(preprocess)

    # OOF threshold tuning on Adults23 (weighted F1)
    print("\n[Threshold tuning] OOF predictions on Adults23...")
    thresholds = {}
    for name in ["RF", "L1", "RF-Cal", "L1-Cal"]:
        print(f" - {name} OOF ...")
        oof = oof_predict_proba(models[name], X_train, y_train, w_train, n_splits=N_SPLITS_OOF)
        score, thr = find_best_threshold(y_train, oof, w_train, metric="f1")
        thresholds[name] = thr
        print(f"   best weighted F1={score:.4f} at thr={thr:.2f}")

    # Evaluate on Adults24
    print("\n[Step] Evaluation on Adults24 (external test)")
    metrics_rows, preds_dict = [], {}
    for name in ["RF", "L1", "RF-Cal", "L1-Cal"]:
        thr = thresholds[name]
        m, p, yhat = evaluate_on_test(
            models[name], X_train, y_train, w_train, X_test, y_test, w_test,
            label=name, threshold=thr
        )
        metrics_rows.append({"model": name, **m})
        preds_dict[f"{name}_prob"] = p
        preds_dict[f"{name}_pred"] = yhat

    # Fit final uncalibrated models once for interpretation
    from copy import deepcopy
    rf_final = deepcopy(models["RF"])
    l1_final = deepcopy(models["L1"])
    fit_with_weights(rf_final, X_train, y_train, w_train)
    fit_with_weights(l1_final, X_train, y_train, w_train)
    interpret_models(rf_final, l1_final)

    # Subgroup diagnostics (optional, if present)
    if "REGION" in test.columns:
        subgroup_report(y_test, preds_dict["RF_prob"], preds_dict["RF_pred"], w_test, test["REGION"], "REGION")
    if "URBRRL23" in test.columns:
        subgroup_report(y_test, preds_dict["RF_prob"], preds_dict["RF_pred"], w_test, test["URBRRL23"], "URBRRL23")

    # Save artifacts
    out_df = pd.DataFrame({"srh_bin": y_test, "wt": w_test})
    for k, v in preds_dict.items():
        out_df[k] = v
    out_df.to_csv("artifacts/test_predictions.csv", index=False)
    pd.DataFrame(metrics_rows).to_csv("artifacts/test_metrics.csv", index=False)

    # Save pipelines (pickle-safe)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_pipeline_safely(models["RF"],     f"artifacts/rf_pipeline_{ts}.joblib")
    save_pipeline_safely(models["L1"],     f"artifacts/l1_pipeline_{ts}.joblib")
    save_pipeline_safely(models["RF-Cal"], f"artifacts/rf_calibrated_{ts}.joblib")
    save_pipeline_safely(models["L1-Cal"], f"artifacts/l1_calibrated_{ts}.joblib")

    print("\nArtifacts saved to ./artifacts:")
    print(" - feature_groups.txt")
    print(" - class_prevalence.csv")
    print(" - rf_importances.csv, l1_coefficients.csv")
    print(" - test_predictions.csv")
    print(" - test_metrics.csv")
    print(" - rf_pipeline_*.joblib, l1_pipeline_*.joblib")
    print(" - rf_calibrated_*.joblib, l1_calibrated_*.joblib")
    print(" - subgroup_REGION.csv / subgroup_URBRRL23.csv (when present)")

if __name__ == "__main__":
    main()
