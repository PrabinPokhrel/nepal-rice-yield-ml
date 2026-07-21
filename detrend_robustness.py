"""
detrend_robustness.py
Robustness of the detrended climate signal to how the trend is removed.

We isolate the interannual climate signal three ways and check that the
conclusion (a modest signal that survives blocked CV) does not depend on the
detrending method:

  linear  - OLS straight line (the main-text method)
  spline  - cubic smoothing spline (flexible, captures Green-Revolution
            acceleration and any plateau)
  firstdiff - first differences (delta-yield vs delta-predictors); removes any
            trend implicitly without fitting one at all

For each method we report pooled out-of-fold R2 under Shuffled-10 and Blocked-10.
(Forward TimeSeries is omitted here because extrapolating each trend type is a
separate question; blocked is the fair attribution test.)
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

DATA_PATH = Path("data/nepal_rice_climate.csv")
YIELD, YEAR = "rice_yield_kgha", "year"

df = pd.read_csv(DATA_PATH).sort_values(YEAR).reset_index(drop=True)
clim = [c for c in df.select_dtypes(include=[np.number]).columns
        if c not in (YIELD, YEAR)]
yr, y = df[YEAR].values.astype(float), df[YIELD].values.astype(float)

def model(name):
    est = {"RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
           "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4,
                                   random_state=42, verbosity=0),
           "Linear": LinearRegression()}[name]
    return make_pipeline(StandardScaler(), est)

def detrend_target(method, train_idx, all_idx):
    """Return detrended y for all rows, using a trend fit on train rows only."""
    if method == "linear":
        s, i = np.polyfit(yr[train_idx], y[train_idx], 1)
        return y - (s * yr + i)
    if method == "spline":
        # fit smoothing spline on training years, evaluate everywhere
        order = np.argsort(yr[train_idx])
        xs, ys = yr[train_idx][order], y[train_idx][order]
        spl = UnivariateSpline(xs, ys, k=3, s=len(xs) * np.var(ys) * 0.15)
        return y - spl(yr)
    raise ValueError

def run(method):
    print(f"\n=== detrending: {method} ===")
    print(f"{'Model':<14}{'Shuffled R2':>13}{'Blocked R2':>13}")
    schemes = {
        "shuffled": list(KFold(10, shuffle=True, random_state=42).split(np.arange(len(df)))),
        "blocked":  list(KFold(10, shuffle=False).split(np.arange(len(df)))),
    }
    for name in ["Linear", "RandomForest", "XGBoost"]:
        out = {}
        for sn, splits in schemes.items():
            ot, op = [], []
            for tr, te in splits:
                yd = detrend_target(method, tr, np.arange(len(df)))
                m = model(name); m.fit(df[clim].values[tr], yd[tr])
                ot.append(yd[te]); op.append(m.predict(df[clim].values[te]))
            out[sn] = r2_score(np.concatenate(ot), np.concatenate(op))
        print(f"{name:<14}{out['shuffled']:>+13.3f}{out['blocked']:>+13.3f}")

# ---- first-difference: no trend fitting at all --------------------------------
def run_firstdiff():
    print("\n=== detrending: firstdiff (delta-yield ~ delta-predictors) ===")
    print(f"{'Model':<14}{'Shuffled R2':>13}{'Blocked R2':>13}")
    dY = np.diff(y)
    dX = np.diff(df[clim].values, axis=0)
    schemes = {
        "shuffled": list(KFold(10, shuffle=True, random_state=42).split(np.arange(len(dY)))),
        "blocked":  list(KFold(10, shuffle=False).split(np.arange(len(dY)))),
    }
    for name in ["Linear", "RandomForest", "XGBoost"]:
        out = {}
        for sn, splits in schemes.items():
            ot, op = [], []
            for tr, te in splits:
                m = model(name); m.fit(dX[tr], dY[tr])
                ot.append(dY[te]); op.append(m.predict(dX[te]))
            out[sn] = r2_score(np.concatenate(ot), np.concatenate(op))
        print(f"{name:<14}{out['shuffled']:>+13.3f}{out['blocked']:>+13.3f}")

for meth in ["linear", "spline"]:
    run(meth)
run_firstdiff()