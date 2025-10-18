# file: src/grouped_importance.py
# Purpose: Compute grouped permutation importance at ORIGINAL NHIS variable level
#          using a saved pipeline (RF, L1, calibrated, etc.). No OHE name wrangling.

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# --- tiny stubs so unpickling works if your pipeline referenced these helpers ---
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
# -------------------------------------------------------------------------------

# If you want pretty labels on the plot/table, add them here:
PRETTY = {
    "PHSTAT_A": "Self-rated health (1=Excellent…5=Poor)",
    "WTFA_A": "NHIS weight",
    "POVRATTC_A": "Poverty ratio",
    "RATCAT_A": "Family income-to-poverty category",
    "EDUCP_A": "Education (sample adult)",
    "MAXEDUCP_A": "Highest education in family",
    "EMPWKHRS3_A": "Hours worked per week",
    "EMPWRKFT1_A": "Usually works full-time",
    "EMPHEALINS_A": "Job offers health insurance",
    "EMPSICKLV_A": "Paid sick leave",
    "EMPLASTWK_A": "Worked last week",
    "EMPNOWRK_A": "Reason not working now",
    "EMPWHENWRK_A": "Work schedule",
    "EMDSUPER_A": "Has supervisor",
    "MARITAL_A": "Marital status",
    "MARSTAT_A": "Marital status (recode)",
    "LONELY_A": "Felt lonely (freq.)",
    "SUPPORT_A": "Emotional/social support",
    "URBRRL23": "Urban–Rural classification",
    "REGION": "Census region",
    "FDSCAT3_A": "Food security (3-cat)",
    "FDSCAT4_A": "Food security (4-cat)",
    "DISAB3_A": "Disability composite",
    "ANYDIFF_A": "Any difficulty",
    "DIFF_A": "Walking/climbing difficulty",
    "COGMEMDFF_A": "Cognitive difficulty",
    "COMDIFF_A": "Communication difficulty",
    "VISIONDF_A": "Vision difficulty",
    "HEARINGDF_A": "Hearing difficulty",
    "K6SPD_A": "Serious psychological distress",
    "WORTHLESS_A": "Felt worthless",
    "HOPELESS_A": "Felt hopeless",
    "SAD_A": "Felt sad",
    "NERVOUS_A": "Felt nervous",
    "RESTLESS_A": "Felt restless",
    "EFFORT_A": "Everything an effort",
    "DEPFREQ_A": "Depression frequency",
    "ANXFREQ_A": "Anxiety frequency",
    "DEPLEVEL_A": "Depression severity",
    "DEPMED_A": "On depression medication",
    "ANXMED_A": "On anxiety medication",
    "MHRX_A": "Any mental-health Rx",
    "MHTHRPY_A": "Counseling/therapy",
    "MHTHDLY_A": "Delayed counseling (cost)",
    "MHTHND_A": "Needed MH care, didn’t get",
    "HYPEV_A": "Hypertension (ever)",
    "DIBEV_A": "Diabetes (ever)",
    "CHDEV_A": "CHD (ever)",
    "MIEV_A": "Heart attack (ever)",
    "STREV_A": "Stroke (ever)",
    "ANGEV_A": "Angina (ever)",
    "ASEV_A": "Asthma (ever)",
    "ASTILL_A": "Asthma (current)",
    "ARTHEV_A": "Arthritis (ever)",
    "COPDEV_A": "COPD (ever)",
    "CANEV_A": "Cancer (ever)",
    "CHLEV_A": "High cholesterol (ever)",
    "CHL12M_A": "High cholesterol (12m)",
    "HYP12M_A": "High BP (12m)",
    "HYPMED_A": "On BP medication",
    "KIDWEAKEV_A": "Weak/failing kidneys (ever)",
    "LIVEREV_A": "Liver disease (ever)",
    "HEPEV_A": "Hepatitis (ever)",
    "CROHNSEV_A": "Crohn’s disease (ever)",
    "ULCCOLEV_A": "Ulcerative colitis (ever)",
    "PSOREV_A": "Psoriasis (ever)",
    "CFSNOW_A": "Chronic fatigue syndrome (now)",
    "HICOV_A": "Any health insurance",
    "USUALPL_A": "Usual place for care",
    "MEDNG12M_A": "Needed care, not get (12m, cost)",
    "MEDDL12M_A": "Delayed care due to cost (12m)",
    "RXDG12M_A": "Couldn’t afford Rx (12m)",
    "LASTDR_A": "Time since last doctor",
    "WELLVIS_A": "Time since wellness visit",
}

# These should match your training FEATURES (minus target/weight)
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

TARGET_RAW = "PHSTAT_A"
WEIGHT_COL = "WTFA_A"

def binarize_srh(series: pd.Series) -> np.ndarray:
    x = series.replace({7: np.nan, 8: np.nan, 9: np.nan})
    return (x >= 4).astype(int).to_numpy()  # 1 = Fair/Poor

def normalized_weights(w: pd.Series, n: int) -> np.ndarray:
    if w is None or w.empty:
        return np.ones(n, dtype=float)
    w = w.fillna(0).clip(lower=0).to_numpy(dtype=float)
    m = w.mean() if w.mean() > 0 else 1.0
    return w / m

def weighted_auc(y_true, y_prob, w):
    return roc_auc_score(y_true, y_prob, sample_weight=w)

def main():
    ap = argparse.ArgumentParser(description="Grouped permutation importance at original-variable level.")
    ap.add_argument("--pipe", required=True, help="Path to saved pipeline .joblib (RF/L1/Cal).")
    ap.add_argument("--data", required=True, help="Parquet with columns incl. TARGET and WEIGHT (Adults24 recommended).")
    ap.add_argument("--out", default="artifacts/grouped_importance.csv", help="Output CSV path.")
    ap.add_argument("--n_repeats", type=int, default=5, help="Repeats per variable.")
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Load data
    df = pd.read_parquet(args.data)
    cols_needed = [c for c in FEATURES if c in df.columns]
    if TARGET_RAW not in df.columns:
        raise ValueError(f"{TARGET_RAW} not in {args.data}")
    if WEIGHT_COL not in df.columns:
        print("[warn] Weight column not found; using equal weights.")
    X = df[cols_needed].copy()
    y = binarize_srh(df[TARGET_RAW])
    w = normalized_weights(df.get(WEIGHT_COL, pd.Series(index=df.index)), len(df))

    # Load pipeline
    pipe = joblib.load(args.pipe)

    # Baseline AUC
    p_base = pipe.predict_proba(X)[:, 1]
    base_auc = weighted_auc(y, p_base, w)

    rng = np.random.RandomState(args.random_state)
    results = []

    for col in cols_needed:
        deltas = []
        for _ in range(args.n_repeats):
            Xp = X.copy()
            # Shuffle this column only (preserve missing pattern)
            mask = Xp[col].notna().to_numpy()
            vals = Xp.loc[mask, col].to_numpy()
            rng.shuffle(vals)
            Xp.loc[mask, col] = vals
            # Predict & score
            p = pipe.predict_proba(Xp)[:, 1]
            auc = weighted_auc(y, p, w)
            deltas.append(base_auc - auc)
        results.append({
            "nhis_var": col,
            "delta_auc": float(np.mean(deltas)),
            "delta_auc_std": float(np.std(deltas)),
            "pretty_label": PRETTY.get(col, col)
        })

    out = pd.DataFrame(results).sort_values("delta_auc", ascending=False)
    out.to_csv(args.out, index=False)

    print(f"\nBaseline weighted AUC: {base_auc:.4f}")
    print(f"Saved grouped permutation importance to {args.out}")
    print(out.head(20).to_string(index=False))

if __name__ == "__main__":
    main()
