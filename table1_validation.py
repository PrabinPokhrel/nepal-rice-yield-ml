"""
table1_validation.py
--------------------
Standalone regenerator for Table 1: Linear, RandomForest, XGBoost, and the
regularized MLP, each under the full model (calendar year retained) and the
detrended model (linear trend removed in-fold, year dropped), across three CV
schemes (shuffled, blocked, time-series). Pooled out-of-fold R2 / RMSE / MAE.

The regularized MLP = small net (8 units) + L2 weight decay, with BOTH inputs
and target standardized, predictions averaged over 15 random inits.

Consistency check: Linear / RandomForest / XGBoost should print the same
numbers as your current Table 1 (their setup is unchanged and tree/linear
predictions are scale-invariant). If they match, the new MLP row is consistent
and can go straight into the manuscript.

Usage:  python table1_validation.py
Expects: data/nepal_rice_climate.csv
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
warnings.filterwarnings("ignore")

SEED = 42

# ------------------------------------------------------------------ regularized MLP
class SeedEnsembleMLP(BaseEstimator, RegressorMixin):
    """Average of several small MLPs differing only in initialization seed."""
    def __init__(self, hidden_layer_sizes=(8,), alpha=0.1, max_iter=3000, n_seeds=15):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.max_iter = max_iter
        self.n_seeds = n_seeds

    def fit(self, X, y):
        self.models_ = []
        for s in range(self.n_seeds):
            m = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes, activation="relu",
                             alpha=self.alpha, max_iter=self.max_iter, random_state=s)
            m.fit(X, y)
            self.models_.append(m)
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models_], axis=0)


def make_reg_mlp(hidden=(8,), alpha=0.1, n_seeds=15):
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(),
                                SeedEnsembleMLP(hidden, alpha=alpha, n_seeds=n_seeds)),
        transformer=StandardScaler())


# ------------------------------------------------------------------ data
df = pd.read_csv("data/nepal_rice_climate.csv").dropna().reset_index(drop=True)
y = df["rice_yield_kgha"].values
year = df["year"].values
X_full = df.drop(columns=["rice_yield_kgha"]).values          # includes year
X_detr = df.drop(columns=["rice_yield_kgha", "year"]).values  # excludes year
print(f"n = {len(y)} observations; {X_full.shape[1]} full predictors, "
      f"{X_detr.shape[1]} detrended predictors\n")


def build_models():
    return {
        "Linear":       make_pipeline(StandardScaler(), LinearRegression()),
        "RandomForest": make_pipeline(StandardScaler(),
                                      RandomForestRegressor(n_estimators=200, random_state=SEED)),
        "XGBoost":      make_pipeline(StandardScaler(),
                                      XGBRegressor(n_estimators=200, learning_rate=0.05,
                                                   max_depth=4, random_state=SEED, verbosity=0)),
        "MLP (reg.)":   make_reg_mlp(),
    }

SCHEMES = {
    "shuffled":   KFold(n_splits=10, shuffle=True, random_state=SEED),
    "blocked":    KFold(n_splits=10, shuffle=False),
    "timeseries": TimeSeriesSplit(n_splits=5),
}


# ------------------------------------------------------------------ CV evaluators
def eval_full(model, cv):
    yhat = np.full(len(y), np.nan)
    for tr, te in cv.split(X_full):
        est = clone(model)
        est.fit(X_full[tr], y[tr])
        yhat[te] = est.predict(X_full[te])
    m = ~np.isnan(yhat)
    return (r2_score(y[m], yhat[m]),
            np.sqrt(mean_squared_error(y[m], yhat[m])),
            mean_absolute_error(y[m], yhat[m]))


def eval_detrended(model, cv):
    """Linear trend fit on TRAIN years only, removed from train and test (in-fold)."""
    ytrue = np.full(len(y), np.nan)
    yhat = np.full(len(y), np.nan)
    for tr, te in cv.split(X_detr):
        slope, intercept = np.polyfit(year[tr], y[tr], 1)
        ytr_d = y[tr] - (slope * year[tr] + intercept)
        yte_d = y[te] - (slope * year[te] + intercept)
        est = clone(model)
        est.fit(X_detr[tr], ytr_d)
        yhat[te] = est.predict(X_detr[te])
        ytrue[te] = yte_d
    m = ~np.isnan(yhat)
    return (r2_score(ytrue[m], yhat[m]),
            np.sqrt(mean_squared_error(ytrue[m], yhat[m])),
            mean_absolute_error(ytrue[m], yhat[m]))


# ------------------------------------------------------------------ run + print
def print_block(title, evaluator):
    print("=" * 78)
    print(title)
    print(f"{'model':<14}" + "".join(f"{s:>21}" for s in SCHEMES))
    print(f"{'':14}" + "".join(f"{'R2   RMSE   MAE':>21}" for _ in SCHEMES))
    print("-" * 78)
    for name, model in build_models().items():
        cells = []
        for cv in SCHEMES.values():
            r2, rmse, mae = evaluator(model, cv)
            cells.append(f"{r2:+.3f} {rmse:4.0f} {mae:4.0f}")
        print(f"{name:<14}" + "".join(f"{c:>21}" for c in cells))
    print()

print_block("FULL MODEL  (calendar year retained)", eval_full)
print_block("DETRENDED MODEL  (linear trend removed in-fold; year dropped)", eval_detrended)

print("Notes:")
print("  - Linear/RandomForest/XGBoost should match your current Table 1 (consistency check).")
print("  - 'MLP (reg.)' = (8,) hidden, alpha=0.1, target-scaled, 15-seed average.")
print("  - Paste this whole output back and it drops straight into Table 1.")
