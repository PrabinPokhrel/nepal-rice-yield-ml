"""
detrend_sensitivity.py
Is the 'climate signal collapses under proper detrending' result robust to the
detrending flexibility, or an artifact of one arbitrary choice?

Two sweeps, both reporting blocked-CV pooled R2 for the climate models
(detrend fitted within each fold, train years only):

  (A) Polynomial trend of degree 1..4
      degree 1 = the linear detrend that showed a signal.
      If the signal is leftover nonlinear trend, it should vanish at degree >=2.

  (B) Cubic smoothing spline over a range of smoothing strengths s = k*n*var(y).
      Confirms the earlier s (k=0.15) was not special.
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

df = pd.read_csv(Path("data/nepal_rice_climate.csv")).sort_values("year").reset_index(drop=True)
YIELD, YEAR = "rice_yield_kgha", "year"
clim = [c for c in df.select_dtypes(include=[np.number]).columns if c not in (YIELD, YEAR)]
yr, y = df[YEAR].values.astype(float), df[YIELD].values.astype(float)
X = df[clim].values
blocked = list(KFold(10, shuffle=False).split(np.arange(len(df))))

def mk(name):
    est = {"RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
           "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4,
                                   random_state=42, verbosity=0),
           "Linear": LinearRegression()}[name]
    return make_pipeline(StandardScaler(), est)

def blocked_r2(detrend_all):
    """detrend_all(train_idx) -> detrended y for ALL rows (trend fit on train)."""
    res = {}
    for name in ["Linear", "RandomForest", "XGBoost"]:
        ot, op = [], []
        for tr, te in blocked:
            yd = detrend_all(tr)
            m = mk(name); m.fit(X[tr], yd[tr])
            ot.append(yd[te]); op.append(m.predict(X[te]))
        res[name] = r2_score(np.concatenate(ot), np.concatenate(op))
    return res

print("(A) POLYNOMIAL TREND DEGREE  ->  blocked climate R2")
print(f"{'degree':<8}{'Linear':>10}{'RandomForest':>14}{'XGBoost':>10}")
for d in [1, 2, 3, 4]:
    def dt(tr, d=d):
        c = np.polyfit(yr[tr], y[tr], d)
        return y - np.polyval(c, yr)
    r = blocked_r2(dt)
    print(f"{d:<8}{r['Linear']:>+10.3f}{r['RandomForest']:>+14.3f}{r['XGBoost']:>+10.3f}")

print("\n(B) SPLINE SMOOTHING k (s = k*n*var(y))  ->  blocked climate R2")
print(f"{'k':<8}{'Linear':>10}{'RandomForest':>14}{'XGBoost':>10}")
for k in [0.05, 0.10, 0.15, 0.30, 0.60, 1.20]:
    def dt(tr, k=k):
        order = np.argsort(yr[tr])
        xs, ys = yr[tr][order], y[tr][order]
        s = k * len(xs) * np.var(ys)
        spl = UnivariateSpline(xs, ys, k=3, s=s)
        return y - spl(yr)
    r = blocked_r2(dt)
    print(f"{k:<8}{r['Linear']:>+10.3f}{r['RandomForest']:>+14.3f}{r['XGBoost']:>+10.3f}")