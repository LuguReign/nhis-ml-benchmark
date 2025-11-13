import argparse, os
import numpy as np
import pandas as pd

from .featuresets import get_featureset
from .preprocess import build_preprocessor, normalize_weights


def _feature_names_from_ct(ct) -> list:
    # works on fitted ColumnTransformer
    try:
        return ct.get_feature_names_out().tolist()
    except Exception:
        # sklearn fallback: compose names manually
        names = []
        for name, trans, cols in ct.transformers_:
            if name == "cat":
                ohe = trans.named_steps["ohe"]
                names.extend(ohe.get_feature_names_out(cols).tolist())
            elif name in ("bin", "ord", "miss"):
                names.extend(list(cols))
        return names


def process_year(year: int, raw_core_path: str, outdir: str, featureset: str = "srh_core",
                 rare_min_count: int = 50):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_parquet(raw_core_path)

    # select columns
    feats = get_featureset(featureset)
    keep = [c for c in feats.all_columns if c in df.columns]
    core = df[keep].copy()

    # normalize core fields
    if "WTFA_A" in core.columns:
        core["WTFA_A"] = normalize_weights(core["WTFA_A"])
    if "PHSTAT_A" in core.columns:
        core["PHSTAT_A"] = pd.to_numeric(core["PHSTAT_A"], errors="coerce")

    # building preprocessor on THIS year (for export/debug; training will refit on train-year)
    preproc, schema, df_fit = build_preprocessor(
        train_core=core,
        binary_cols=[c for c in feats.binary_12 if c in core.columns],
        ordinal_cols=[c for c in feats.ordinal if c in core.columns],
        categorical_cols=[c for c in feats.categorical if c in core.columns],
        rare_min_count=rare_min_count,
    )

    # transform to feature matrix
    Xt = preproc.transform(core)
    Xd = Xt.toarray() if hasattr(Xt, "toarray") else Xt

    # outputs
    core_out   = os.path.join(outdir, f"Adults{str(year)[-2:]}_core.parquet")
    feats_out  = os.path.join(outdir, f"Adults{str(year)[-2:]}_features.parquet")
    schema_out = os.path.join(outdir, f"Adults{str(year)[-2:]}_schema.json")

    core.to_parquet(core_out, index=False)

    feat_names = _feature_names_from_ct(preproc.named_steps["ct"])
    pd.DataFrame(Xd, columns=feat_names).to_parquet(feats_out, index=False)

    schema.to_json(schema_out)
    print(f"Processed {year}:")
    print(f" - {core_out}")
    print(f" - {feats_out}")
    print(f" - {schema_out}")


def cli():
    p = argparse.ArgumentParser("nhis-process")
    p.add_argument("years", nargs="+", type=int, help="e.g., 2023 2024")
    p.add_argument("--rawdir", default="data/processed", help="dir with AdultsYY_core.parquet (from fetch step)")
    p.add_argument("--outdir", default="data/processed", help="output directory")
    p.add_argument("--featureset", default="srh_core")
    p.add_argument("--rare-min-count", type=int, default=50)
    args = p.parse_args()

    for yr in args.years:
        core_path = os.path.join(args.rawdir, f"Adults{str(yr)[-2:]}_core.parquet")
        process_year(
            year=yr,
            raw_core_path=core_path,
            outdir=args.outdir,
            featureset=args.featureset,
            rare_min_count=args.rare_min_count,
        )
