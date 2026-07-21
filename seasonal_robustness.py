"""
seasonal_robustness.py
Run the same detrending-robustness gauntlet used on the annual data, now on the
monsoon growing-season predictors, to test whether a real interannual climate
signal exists at seasonal resolution that annual aggregation washed out.

For each predictor set (whole season, and each phenological window), report
pooled blocked-CV R2 under three detrends: linear, spline, first-difference.
Trend is fit within each training fold. Climate predictors only (no year).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy.interpolate import UnivariateSpline
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv(Path("data/nepal_rice_seasonal.csv")).sort_values("year").reset_index(drop=True)
YIELD, YEAR = "rice_yield_kgha", "year"
yr = df[YEAR].values.astype(float)
y  = df[YIELD].values.astype(float)
blocked = list(KFold(10, shuffle=False).split(np.arange(len(df))))
MODELS = ["Linear", "RandomForest", "XGBoost"]

# predictor sets: whole season + each phenological window
SETS = {
    "SEASON (Jun-Oct)":   ["tas_season","tasmax_season","tasmin_season","pr_season"],
    "ESTABLISH (Jun-Jul)":["tas_establish","tasmax_establish","tasmin_establish","pr_establish"],
    "FLOWER (Aug-Sep)":   ["tas_flower","tasmax_flower","tasmin_flower","pr_flower"],
    "GRAINFILL (Oct)":    ["tas_grainfill","tasmax_grainfill","tasmin_grainfill","pr_grainfill"],
}

def mk(name):
    est = {"RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
           "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4,
                                   random_state=42, verbosity=0),
           "Linear": LinearRegression()}[name]
    return make_pipeline(StandardScaler(), est)

def blocked_r2(X, detrend):
    res = {}
    for name in MODELS:
        ot, op = [], []
        for tr, te in blocked:
            if detrend == "firstdiff":
                dY = np.diff(y); dX = np.diff(X, axis=0)
                tr2 = tr[tr < len(dY)]; te2 = te[te < len(dY)]
                m = mk(name); m.fit(dX[tr2], dY[tr2])
                ot.append(dY[te2]); op.append(m.predict(dX[te2]))
            else:
                if detrend == "linear":
                    yd = y - np.polyval(np.polyfit(yr[tr], y[tr], 1), yr)
                else:  # spline
                    order = np.argsort(yr[tr]); xs, ys = yr[tr][order], y[tr][order]
                    spl = UnivariateSpline(xs, ys, k=3, s=0.30*len(xs)*np.var(ys))
                    yd = y - spl(yr)
                m = mk(name); m.fit(X[tr], yd[tr])
                ot.append(yd[te]); op.append(m.predict(X[te]))
        res[name] = r2_score(np.concatenate(ot), np.concatenate(op))
    return res

for label, cols in SETS.items():
    X = df[cols].values
    print(f"\n=== {label} ===")
    print(f"{'detrend':<12}{'Linear':>10}{'RandomForest':>14}{'XGBoost':>10}")
    for d in ["linear", "spline", "firstdiff"]:
        r = blocked_r2(X, d)
        print(f"{d:<12}{r['Linear']:>+10.3f}{r['RandomForest']:>+14.3f}{r['XGBoost']:>+10.3f}")

# reference: the annual result was linear RF +0.319, spline RF -0.008, firstdiff RF -0.133
print("\n(Annual national result for comparison: linear RF +0.319 -> spline RF -0.008 -> firstdiff RF -0.133)")