# file: src/viz_grouped_importance.py
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="artifacts/grouped_importance_l1_adults24.csv")
    ap.add_argument("--out_png", default="artifacts/grouped_importance_top15.png")
    ap.add_argument("--out_csv", default="artifacts/grouped_importance_top15.csv")
    ap.add_argument("--n_repeats", type=int, default=5, help="n_repeats used in the importance script")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)

    df = pd.read_csv(args.in_csv)
    # 95% CI for mean across repeats: 1.96 * (std / sqrt(n))
    df["delta_auc_ci95"] = 1.96 * (df["delta_auc_std"] / max(args.n_repeats, 1))

    top = df.sort_values("delta_auc", ascending=False).head(15).copy()
    top = top[::-1]  # reverse for horizontal plot (small->large up top)

    # Save tidy CSV for slides
    top_csv = top[["nhis_var", "pretty_label", "delta_auc", "delta_auc_std", "delta_auc_ci95"]][::-1]
    top_csv.to_csv(args.out_csv, index=False)

    # Plot
    plt.figure(figsize=(8, 6))
    y = np.arange(len(top))
    plt.barh(y, top["delta_auc"].values)
    # error bars as whiskers (optional)
    plt.errorbar(top["delta_auc"].values, y, xerr=top["delta_auc_ci95"].values, fmt="none", capsize=3)
    labels = top["pretty_label"].fillna(top["nhis_var"]).values
    plt.yticks(y, labels)
    plt.xlabel("Δ AUC when shuffled (Adults24)")
    plt.title("Top-15 grouped permutation importance (L1)")
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)
    print(f"Saved: {args.out_png}\nSaved: {args.out_csv}")

if __name__ == "__main__":
    main()
