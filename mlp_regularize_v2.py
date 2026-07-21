"""
mlp_regularize_v2.py
--------------------
Corrected version. The previous script over-regularized because alpha was
applied to a network fitting an UNSCALED target (~2500 kg/ha): the L2 penalty
on the large output weights needed to reach that scale dominated the loss and
collapsed predictions (R2 ~ -10). Fix: scale the target with
TransformedTargetRegressor so the network fits a ~N(0,1) output, which makes
alpha meaningful. We then sweep a few alphas rather than trusting one value.

Compares the paper's current MLP (100,50; alpha=1e-4; unscaled target, as in
the manuscript) against small, target-scaled, seed-averaged MLPs across the
three CV schemes used for Table 1.

Usage:  python mlp_regularize_v2.py
Expects: data/nepal_rice_climate.csv
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
warnings.filterwarnings("ignore")

df = pd.read_csv("data/nepal_rice_climate.csv").dropna().reset_index(drop=True)
y = df["rice_yield_kgha"].values
X = df.drop(columns=["rice_yield_kgha"]).values
print(f"n = {len(y)}   y mean {y.mean():.0f}  sd {y.std():.0f} kg/ha\n")


class SeedEnsembleMLP(BaseEstimator, RegressorMixin):
    """Average predictions of several MLPs differing only in init seed."""
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


def make_reg(hidden, alpha, n_seeds=15):
    """Small MLP ensemble with BOTH inputs and target standardized."""
    inner = make_pipeline(StandardScaler(),
                          SeedEnsembleMLP(hidden, alpha=alpha, n_seeds=n_seeds))
    return TransformedTargetRegressor(regressor=inner, transformer=StandardScaler())


SCHEMES = {
    "shuffled":   KFold(n_splits=10, shuffle=True, random_state=42),
    "blocked":    KFold(n_splits=10, shuffle=False),
    "timeseries": TimeSeriesSplit(n_splits=5),
}

def pooled_oof(estimator, X, y, cv):
    yhat = np.full(len(y), np.nan)
    for tr, te in cv.split(X):
        est = clone(estimator)
        est.fit(X[tr], y[tr])
        yhat[te] = est.predict(X[te])
    m = ~np.isnan(yhat)
    return r2_score(y[m], yhat[m]), np.sqrt(mean_squared_error(y[m], yhat[m]))

configs = {
    "current (100,50) a=1e-4": make_pipeline(
        StandardScaler(), MLPRegressor(hidden_layer_sizes=(100, 50), alpha=1e-4,
                                       max_iter=2000, random_state=42)),
    "reg (8,) a=0.01 [y-scaled]":  make_reg((8,), 0.01),
    "reg (8,) a=0.1  [y-scaled]":  make_reg((8,), 0.1),
    "reg (8,) a=1.0  [y-scaled]":  make_reg((8,), 1.0),
    "reg (5,) a=0.1  [y-scaled]":  make_reg((5,), 0.1),
}

print(f"{'config':<28}{'shuffled':>16}{'blocked':>16}{'timeseries':>16}")
print("-" * 76)
for name, est in configs.items():
    row = []
    for cv in SCHEMES.values():
        r2, rm = pooled_oof(est, X, y, cv)
        row.append(f"{r2:+.3f} ({rm:.0f})")
    print(f"{name:<28}" + "".join(f"{c:>16}" for c in row))

print("\nReading:")
print("  - All y-scaled regularized configs should now give sensible (not -10) R2.")
print("  - Compare their shuffled/blocked R2 to the current MLP: is the small net")
print("    as good in-sample? And is its forward (timeseries) R2 less negative?")
print("  - Pick the simplest config that matches the current MLP in-sample and is")
print("    no worse forward; drop it into ml_models.py/validation.py for Table 1.")
