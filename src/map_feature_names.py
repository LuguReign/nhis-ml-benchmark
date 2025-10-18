# file: src/map_feature_names.py
# Robust f# -> readable names using fitted OneHotEncoder.get_feature_names_out

import argparse, os, warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- stubs so your pickled pipelines unpickle fine (names must match) ---
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
# ------------------------------------------------------------------------

def _extract_prep(obj):
    if isinstance(obj, ColumnTransformer):
        return obj
    if isinstance(obj, Pipeline) and "prep" in obj.named_steps:
        return obj.named_steps["prep"]
    if hasattr(obj, "base_estimator"):
        base = getattr(obj, "base_estimator")
        if isinstance(base, Pipeline) and "prep" in base.named_steps:
            return base.named_steps["prep"]
    if hasattr(obj, "calibrated_classifiers_"):
        ccs = getattr(obj, "calibrated_classifiers_")
        if ccs and hasattr(ccs[0], "base_estimator"):
            base = ccs[0].base_estimator
            if isinstance(base, Pipeline) and "prep" in base.named_steps:
                return base.named_steps["prep"]
    raise ValueError("Could not locate a fitted ColumnTransformer 'prep'.")

def _expected_cols_from_prep(prep: ColumnTransformer):
    cols = []
    for name, trans, assigned in getattr(prep, "transformers_", []):
        if name == "remainder":
            continue
        if hasattr(trans, "named_steps"):
            sel = trans.named_steps.get("select", None)
            if sel is not None and hasattr(sel, "kw_args") and sel.kw_args.get("cols") is not None:
                cols.extend(list(sel.kw_args["cols"]))
            else:
                cols.extend(list(assigned) if isinstance(assigned, (list, tuple)) else [assigned])
        else:
            cols.extend(list(assigned) if isinstance(assigned, (list, tuple)) else [assigned])
    # de-dup but preserve order
    seen, ordered = set(), []
    for c in cols:
        if c not in seen:
            seen.add(c); ordered.append(c)
    return ordered

def _names_from_prep(prep: ColumnTransformer) -> list:
    """Exact names by walking fitted branches; for cat use OHE.get_feature_names_out(sel_cols)."""
    names = []
    for name, trans, assigned in getattr(prep, "transformers_", []):
        if name == "remainder":
            continue
        if not hasattr(trans, "named_steps"):
            # unexpected: pass through
            names.extend(list(assigned) if isinstance(assigned, (list, tuple)) else [assigned])
            continue

        sel_cols = None
        sel = trans.named_steps.get("select", None)
        if sel is not None and hasattr(sel, "kw_args"):
            sel_cols = sel.kw_args.get("cols", None)
        if sel_cols is None:
            sel_cols = list(assigned) if isinstance(assigned, (list, tuple)) else [assigned]

        if name in ("bin", "ord"):
            names.extend(list(sel_cols))
        elif name == "cat":
            ohe = trans.named_steps.get("ohe", None)
            if ohe is not None:
                try:
                    # This returns the *exact* expanded names in correct order & count.
                    ohe_names = list(ohe.get_feature_names_out(sel_cols))
                except Exception:
                    # very old sklearn fallback: try without args
                    ohe_names = list(ohe.get_feature_names_out())
                names.extend(ohe_names)
            else:
                names.extend(list(sel_cols))
        else:
            names.extend(list(sel_cols))
    return names

def main():
    ap = argparse.ArgumentParser(description="Create f# -> readable names mapping.")
    ap.add_argument("--prep", type=str, default=None, help="Path to *_prep.joblib")
    ap.add_argument("--pipe", type=str, default=None, help="Path to full *pipeline*.joblib")
    ap.add_argument("--data", type=str, required=True, help="Parquet with original columns (Adults23/Adults24/union).")
    ap.add_argument("--out", type=str, default="artifacts/feature_name_mapping.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    obj = joblib.load(args.prep or args.pipe)
    prep = _extract_prep(obj)

    # Reindex to the exact columns the preprocessor expects
    df = pd.read_parquet(args.data)
    expected = _expected_cols_from_prep(prep)
    Xs = df.reindex(columns=expected, fill_value=np.nan).head(10)

    # Transform to get final width
    Xt = prep.transform(Xs)
    n_features = Xt.shape[1]

    # Build names using fitted OHE.get_feature_names_out
    names = _names_from_prep(prep)

    if len(names) != n_features:
        print(f"[warn] Name length mismatch: built={len(names)} vs transformed={n_features}. Falling back to generic f#.")
        names = [f"f{i}" for i in range(n_features)]

    mapping = pd.DataFrame({"f": [f"f{i}" for i in range(n_features)], "name": names})
    mapping.to_csv(args.out, index=False)
    print(f"Saved mapping: {args.out}")
    print(mapping.head(12).to_string(index=False))

if __name__ == "__main__":
    main()
