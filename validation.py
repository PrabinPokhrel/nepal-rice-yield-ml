"""
validation.py  (v2)
Time-aware cross-validation, reporting pooled out-of-fold metrics.

Why pooled R2 + RMSE + MAE, not per-fold R2:
Under blocked / time-series CV each contiguous test block has small internal
variance, so per-fold R2 explodes to large negatives even when predictions are
reasonable. Pooling all held-out predictions restores a meaningful variance
baseline; RMSE and MAE are absolute and immune to that pathology.

Three schemes:
  Shuffled 10-fold  - optimistic baseline (lets the model interpolate years)
  Blocked 10-fold   - contiguous test blocks, train = all other years
  TimeSeriesSplit-5 - expanding window, predict the future (strictest)

Detrending is fitted WITHIN each fold (train years only) for the detrended runs.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

DATA_PATH = Path("data/nepal_rice_climate.csv")
YIELD, YEAR = "rice_yield_kgha", "year"

df = pd.read_csv(DATA_PATH).sort_values(YEAR).reset_index(drop=True)
predictors_full = [c for c in df.select_dtypes(include=[np.number]).columns if c != YIELD]
predictors_clim = [c for c in predictors_full if c != YEAR]
print(f"Loaded {len(df)} obs, {df[YEAR].min()}-{df[YEAR].max()}\n")

MODEL_NAMES = ["Linear", "RandomForest", "XGBoost", "MLP"]

def fresh_model(name):
    if name == "Linear":
        est = LinearRegression()
    elif name == "RandomForest":
        est = RandomForestRegressor(n_estimators=200, random_state=42)
    elif name == "XGBoost":
        est = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4,
                           random_state=42, verbosity=0)
    elif name == "MLP":
        est = MLPRegressor(hidden_layer_sizes=(100, 50), activation="relu",
                           max_iter=2000, random_state=42)
    return make_pipeline(StandardScaler(), est)

n = len(df)
SCHEMES = {
    "Shuffled-10":  list(KFold(n_splits=10, shuffle=True, random_state=42).split(np.arange(n))),
    "Blocked-10":   list(KFold(n_splits=10, shuffle=False).split(np.arange(n))),
    "TimeSeries-5": list(TimeSeriesSplit(n_splits=5).split(np.arange(n))),
}

def evaluate(splits, analysis, name):
    feats = predictors_full if analysis == "full" else predictors_clim
    yr, y_all, X_all = df[YEAR].values, df[YIELD].values, df[feats].values
    ot, op = [], []
    for tr, te in splits:
        if analysis == "detrended":
            slope, intercept = np.polyfit(yr[tr], y_all[tr], 1)
            ytr = y_all[tr] - (slope * yr[tr] + intercept)
            yte = y_all[te] - (slope * yr[te] + intercept)
        else:
            ytr, yte = y_all[tr], y_all[te]
        m = fresh_model(name); m.fit(X_all[tr], ytr)
        ot.append(yte); op.append(m.predict(X_all[te]))
    ot, op = np.concatenate(ot), np.concatenate(op)
    return (r2_score(ot, op),
            np.sqrt(mean_squared_error(ot, op)),
            mean_absolute_error(ot, op))

for analysis in ["full", "detrended"]:
    print("=" * 74)
    print(f"ANALYSIS: {analysis.upper()}"
          + ("   (trend fitted within each fold)" if analysis == "detrended" else ""))
    print("=" * 74)
    print(f"{'Model':<14}{'Scheme':<15}{'pooled R2':>11}{'RMSE':>10}{'MAE':>10}")
    print("-" * 74)
    for name in MODEL_NAMES:
        for s, sp in SCHEMES.items():
            r2, rmse, mae = evaluate(sp, analysis, name)
            print(f"{name:<14}{s:<15}{r2:>+11.3f}{rmse:>10.1f}{mae:>10.1f}")
        print()