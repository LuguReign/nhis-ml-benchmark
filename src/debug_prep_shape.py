import joblib, pandas as pd, numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def _extract_prep(obj):
    if isinstance(obj, ColumnTransformer): return obj
    if isinstance(obj, Pipeline) and "prep" in obj.named_steps: return obj.named_steps["prep"]
    if hasattr(obj, "base_estimator"):
        base = getattr(obj, "base_estimator")
        if isinstance(base, Pipeline) and "prep" in base.named_steps: return base.named_steps["prep"]
    if hasattr(obj, "calibrated_classifiers_"):
        ccs = getattr(obj, "calibrated_classifiers_")
        if ccs and hasattr(ccs[0], "base_estimator"):
            base = ccs[0].base_estimator
            if isinstance(base, Pipeline) and "prep" in base.named_steps: return base.named_steps["prep"]
    raise RuntimeError("prep not found")

# stubs so unpickling works if your pipeline referenced them
def select_reindex_func(X, cols):
    import pandas as pd, numpy as np
    X = pd.DataFrame(X); return X.reindex(columns=cols, fill_value=np.nan)
def map_12_binary_func(X, cols):
    import pandas as pd, numpy as np
    out = pd.DataFrame(X, copy=True)
    for c in cols:
        if c in out.columns:
            out[c] = out[c].replace({1:1.0, 2:0.0})
            out[c] = out[c].where(out[c].isin([0.0,1.0]), np.nan)
    return out
def clean_ordinals_func(X, cols):
    import pandas as pd, numpy as np
    out = pd.DataFrame(X, copy=True)
    for c in cols:
        if c in out.columns:
            out[c] = out[c].replace({7:np.nan,8:np.nan,9:np.nan})
    return out

pipe = joblib.load("artifacts/l1_pipeline_20251013-012002.joblib")
prep = _extract_prep(pipe)

# columns the prep expects (in branch order)
expected = []
for name, trans, assigned in prep.transformers_:
    if name=="remainder": continue
    sel = trans.named_steps.get("select", None) if hasattr(trans,"named_steps") else None
    if sel is not None and hasattr(sel,"kw_args") and sel.kw_args.get("cols") is not None:
        expected.extend(list(sel.kw_args["cols"]))
    else:
        expected.extend(list(assigned) if isinstance(assigned,(list,tuple)) else [assigned])

# transform a tiny frame with all expected cols present
dfu = pd.read_parquet("data/Adults23_24_union.parquet")
Xs = dfu.reindex(columns=expected, fill_value=np.nan).head(5)
Xt = prep.transform(Xs)
total = 0
print("---- Branch counts ----")
for name, trans, assigned in prep.transformers_:
    if name=="remainder": continue
    sel = trans.named_steps.get("select", None)
    sel_cols = sel.kw_args["cols"] if (sel is not None and hasattr(sel,"kw_args") and sel.kw_args.get("cols") is not None) \
               else (list(assigned) if isinstance(assigned,(list,tuple)) else [assigned])
    if name in ("bin","ord"):
        k = len(sel_cols)
        print(f"{name}: {k}")
        total += k
    elif name=="cat":
        ohe = trans.named_steps.get("ohe")
        k = sum(len(cats) for cats in ohe.categories_)
        print(f"{name}: {k}  (per-col: {[len(c) for c in ohe.categories_]})")
        total += k
print(f"Total from branches = {total}")
print(f"Xt shape = {Xt.shape}")
