# file: src/name_importances_quick.py
import pandas as pd

map_l1 = pd.read_csv("artifacts/feature_name_mapping_l1_012002.csv")  # f, name
map_rf = pd.read_csv("artifacts/feature_name_mapping_rf_012002.csv")  # f, name

rf_imp = pd.read_csv("artifacts/rf_importances.csv", index_col=0)     # index=f#, value=importance
l1_coefs = pd.read_csv("artifacts/l1_coefficients.csv", index_col=0)  # index=f#, value=coef

rf_named = map_rf.merge(rf_imp, left_on="f", right_index=True).rename(columns={rf_imp.columns[0]: "importance"})
l1_named = map_l1.merge(l1_coefs, left_on="f", right_index=True).rename(columns={l1_coefs.columns[0]: "coef"})

rf_named.sort_values("importance", ascending=False).to_csv("artifacts/rf_importances_named.csv", index=False)
l1_named.reindex(l1_named["coef"].abs().sort_values(ascending=False).index).to_csv("artifacts/l1_coefficients_named.csv", index=False)

print("Saved: artifacts/rf_importances_named.csv, artifacts/l1_coefficients_named.csv")
