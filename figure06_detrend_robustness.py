"""
figure06_detrend_robustness.py
The central methodological figure: the detrended 'climate signal' exists only
under a linear trend and vanishes once the trend is removed properly.

Panel A: blocked-CV climate R2 vs polynomial trend degree (1-4).
Panel B: blocked-CV climate R2 under three detrending methods
         (linear / cubic spline / first-difference).

All R2 are pooled out-of-fold under Blocked-10 CV, trend fitted within each
training fold, climate predictors only (no year).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
yr, y, X = df[YEAR].values.astype(float), df[YIELD].values.astype(float), df[clim].values
blocked = list(KFold(10, shuffle=False).split(np.arange(len(df))))
MODELS = ["Linear", "RandomForest", "XGBoost"]
COLORS = {"Linear": "#1a3a6b", "RandomForest": "#b2182b", "XGBoost": "#e07b00"}

def mk(name):
    est = {"RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
           "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4,
                                   random_state=42, verbosity=0),
           "Linear": LinearRegression()}[name]
    return make_pipeline(StandardScaler(), est)

def blocked_r2(detrend_train, use_diff=False):
    res = {}
    for name in MODELS:
        ot, op = [], []
        for tr, te in blocked:
            if use_diff:
                dY = np.diff(y); dX = np.diff(X, axis=0)
                # remap fold indices onto the differenced series (drop last)
                tr2 = tr[tr < len(dY)]; te2 = te[te < len(dY)]
                m = mk(name); m.fit(dX[tr2], dY[tr2])
                ot.append(dY[te2]); op.append(m.predict(dX[te2]))
            else:
                yd = detrend_train(tr)
                m = mk(name); m.fit(X[tr], yd[tr])
                ot.append(yd[te]); op.append(m.predict(X[te]))
        res[name] = r2_score(np.concatenate(ot), np.concatenate(op))
    return res

# Panel A data: polynomial degrees
degrees = [1, 2, 3, 4]
polyA = {n: [] for n in MODELS}
for d in degrees:
    r = blocked_r2(lambda tr, d=d: y - np.polyval(np.polyfit(yr[tr], y[tr], d), yr))
    for n in MODELS:
        polyA[n].append(r[n])

# Panel B data: three methods
def spline_detrend(tr):
    order = np.argsort(yr[tr]); xs, ys = yr[tr][order], y[tr][order]
    spl = UnivariateSpline(xs, ys, k=3, s=0.30 * len(xs) * np.var(ys))
    return y - spl(yr)

methods = ["Linear\n(degree 1)", "Cubic\nspline", "First\ndifference"]
linB = blocked_r2(lambda tr: y - np.polyval(np.polyfit(yr[tr], y[tr], 1), yr))
splB = blocked_r2(spline_detrend)
diffB = blocked_r2(None, use_diff=True)
dataB = {n: [linB[n], splB[n], diffB[n]] for n in MODELS}

# ---- Plot --------------------------------------------------------------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2))

for n in MODELS:
    axA.plot(degrees, polyA[n], "o-", color=COLORS[n], lw=2, ms=7, label=n)
axA.axhline(0, color="0.5", lw=0.9, ls="--")
axA.axvspan(0.7, 1.3, color="#cccccc", alpha=0.35)
axA.set_xticks(degrees)
axA.set_xlabel("Polynomial degree of removed trend")
axA.set_ylabel("Blocked-CV climate R\u00B2 (pooled)")
axA.set_title("A. Signal appears only with a straight-line trend")
axA.legend(frameon=False, fontsize=9)
axA.grid(alpha=0.25)
axA.annotate("linear\ntrend", xy=(1, polyA["RandomForest"][0]),
             xytext=(1.5, 0.42), fontsize=8, color="0.35",
             arrowprops=dict(arrowstyle="->", color="0.5"))

xpos = np.arange(len(methods)); width = 0.26
for i, n in enumerate(MODELS):
    axB.bar(xpos + (i - 1) * width, dataB[n], width,
            color=COLORS[n], alpha=0.9, label=n, edgecolor="white")
axB.axhline(0, color="black", lw=0.9)
axB.set_xticks(xpos); axB.set_xticklabels(methods)
axB.set_ylabel("Blocked-CV climate R\u00B2 (pooled)")
axB.set_title("B. Signal vanishes once the trend is removed properly")
axB.legend(frameon=False, fontsize=9)
axB.grid(axis="y", alpha=0.25)

fig.suptitle("The detrended climate signal is an artifact of linear detrending "
             "(national annual scale, 1963-2023)", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("figures/06_detrend_robustness.png", dpi=200, bbox_inches="tight")
print("Saved figures/06_detrend_robustness.png")