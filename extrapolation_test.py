"""
extrapolation_test.py
---------------------
Uniform-warming extrapolation test for the full (trend-retained) models.

Purpose: quantify the exact value the Random Forest (and other models)
flatten to when temperature inputs are pushed beyond the observed
1963-2023 range. This is the number reported in Methods 2.2 / Results 3.2
of the manuscript (the "~3,800 kg/ha" placeholder). Run this to replace
the placeholder with your actual figure.

Reproduces the manuscript's full-model configuration:
  Linear (default), RandomForest(200), XGBoost(200, lr=0.05, depth=4),
  MLP((100,50), relu, max_iter=2000); random_state=42; predictors
  standardized inside a pipeline.

Usage:
  python extrapolation_test.py
Expects:
  data/nepal_rice_climate.csv  (same file used everywhere else)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

SEED = 42
TEMP_COLS = ["tas_mean", "tasmax", "tasmin"]   # the temperature inputs to warm
WARMING = [0, 1, 2, 3]                          # deg C added uniformly

# --- load (drop the missing-2006 row exactly as elsewhere) --------------------
df = pd.read_csv("data/nepal_rice_climate.csv").dropna().reset_index(drop=True)
y = df["rice_yield_kgha"].values
X = df.drop(columns=["rice_yield_kgha"])        # full model keeps `year`
feat = list(X.columns)

baseline_mean_yield = y.mean()
print(f"Observed mean yield           : {baseline_mean_yield:,.1f} kg/ha")
print(f"Observed yield range          : {y.min():,.0f} - {y.max():,.0f} kg/ha")
print(f"Observed tas_mean range       : {df['tas_mean'].min():.2f} - {df['tas_mean'].max():.2f} C")
print(f"Warming temperature columns   : {TEMP_COLS}\n")

models = {
    "Linear":       make_pipeline(StandardScaler(), LinearRegression()),
    "RandomForest": make_pipeline(StandardScaler(),
                                  RandomForestRegressor(n_estimators=200, random_state=SEED)),
    "XGBoost":      make_pipeline(StandardScaler(),
                                  XGBRegressor(n_estimators=200, learning_rate=0.05,
                                               max_depth=4, verbosity=0, random_state=SEED)),
    "MLP":          make_pipeline(StandardScaler(),
                                  MLPRegressor(hidden_layer_sizes=(100, 50), activation="relu",
                                               max_iter=2000, random_state=SEED)),
}

# --- fit on all data, then predict under uniform warming ----------------------
# We report the MEAN predicted yield across all years at each warming level.
# For the trees, watch how successive +1 C increments stop moving the prediction.
print(f"{'model':<13} " + " ".join(f"+{w}C".rjust(9) for w in WARMING) + "   span(kg/ha)")
print("-" * 62)

rows = {}
for name, model in models.items():
    model.fit(X, y)
    preds = []
    for w in WARMING:
        Xw = X.copy()
        Xw[TEMP_COLS] = Xw[TEMP_COLS] + w      # uniform warming on all temp inputs
        preds.append(model.predict(Xw).mean())
    rows[name] = preds
    span = max(preds) - min(preds)
    print(f"{name:<13} " + " ".join(f"{p:9.1f}" for p in preds) + f"   {span:8.1f}")

print("\nReading:")
rf = rows["RandomForest"]
print(f"  RandomForest moves only {max(rf)-min(rf):.1f} kg/ha across 0..+3 C and")
print(f"  settles near {np.mean(rf[1:]):.0f} kg/ha (vs observed mean {baseline_mean_yield:.0f}).")
print("  Successive +1 C increments returning near-identical values confirms the")
print("  trees cannot represent conditions beyond the observed range.")
print("\n  -> Use the RandomForest flattening value above to replace the")
print("     '~3,800 kg/ha' placeholder in Methods 2.2 / Results 3.2.")
