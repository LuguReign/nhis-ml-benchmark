# src/nhisml/explore.py
import argparse, os, pandas as pd

def cli():
    p = argparse.ArgumentParser("nhis-explore")
    p.add_argument("--core", required=True, help="path to AdultsYY_core.parquet")
    args = p.parse_args()

    df = pd.read_parquet(args.core)
    print("\n[explore] Basic info")
    print(f"Rows: {len(df):,}  Cols: {df.shape[1]}")
    print("Columns:", ", ".join(df.columns[:20]), " ...")

    print("\n[explore] PHSTAT_A value counts:")
    print(df["PHSTAT_A"].value_counts(dropna=False))

    if "WTFA_A" in df.columns:
        w = df["WTFA_A"]
        print("\n[explore] Weight summary (WTFA_A):")
        print(w.describe())

    # Save a quick 100-row sample to artifacts
    os.makedirs("artifacts", exist_ok=True)
    df.sample(min(100, len(df)), random_state=42).to_csv("artifacts/explore_sample.csv", index=False)
    print("\nSaved artifacts/explore_sample.csv")
