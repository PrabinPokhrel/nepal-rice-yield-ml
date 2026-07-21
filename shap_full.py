"""
shap_full.py
------------
SHAP interpretation of the FULL (trend-retained) Random Forest model.

Why the full model and NOT the detrended model:
  The detrended model's apparent climate structure is the linear-detrend
  artifact the paper diagnoses (Section 3.3). Running SHAP on it would only
  re-express that artifact in a more sophisticated form and lend it false
  authority. SHAP on the full model instead answers the honest question --
  what drives the predictions we actually trust -- and is expected to show
  calendar year (the intensification trend) dominating, corroborating the
  impurity-importance result without its known bias under collinearity.

Produces:
  figs/07_shap_full_summary.png   (beeswarm summary)
  console: mean(|SHAP|) ranking

Usage:
  pip install shap    # if not already installed
  python shap_full.py
Expects:
  data/nepal_rice_climate.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import shap

SEED = 42
OUT_DIR = "figs"          # change to wherever your other figure PNGs live
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv("data/nepal_rice_climate.csv").dropna().reset_index(drop=True)
y = df["rice_yield_kgha"].values
X = df.drop(columns=["rice_yield_kgha"])          # full model keeps `year`

rf = RandomForestRegressor(n_estimators=200, random_state=SEED).fit(X, y)

explainer = shap.TreeExplainer(rf)
sv = explainer.shap_values(X)

# mean absolute SHAP ranking
mean_abs = np.abs(sv).mean(axis=0)
order = np.argsort(mean_abs)[::-1]
total = mean_abs.sum()
print("Mean(|SHAP|) ranking (full model):")
for i in order:
    print(f"  {X.columns[i]:<16} {mean_abs[i]:8.1f}  ({100*mean_abs[i]/total:4.1f}%)")

# beeswarm summary
plt.figure()
shap.summary_plot(sv, X, show=False, max_display=len(X.columns))
plt.title("SHAP summary: full (trend-retained) Random Forest", fontsize=11)
plt.tight_layout()
_out = os.path.join(OUT_DIR, "07_shap_full_summary.png")
plt.savefig(_out, dpi=200, bbox_inches="tight")
print(f"\nSaved {_out}")
print("Expected: `year` dominates, consistent with the 88% impurity-importance"
      " result and the intensification-trend thesis.")
