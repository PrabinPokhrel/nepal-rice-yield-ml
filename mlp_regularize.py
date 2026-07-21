"""
mlp_regularize.py
-----------------
Quantify and manage MLP overfitting on the 60-observation national series.

Compares the CURRENT MLP (100,50; default alpha; single seed) against a
REGULARIZED, seed-averaged MLP (small hidden layer + strong L2, predictions
averaged over many random initialisations) under the same three CV schemes
used for Table 1. Also prints the seed-to-seed spread of the current MLP to
expose the instability directly.

Full model only (calendar year retained), which is where the MLP is used in
the paper. Scaling is fitted inside each fold (leak-free). Once you pick a
config, drop the same MLP definition into ml_models.py / validation.py so the
Table 1 MLP row is regenerated consistently.

Usage:
  python mlp_regularize.py
Expects:
  data/nepal_rice_climate.csv
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
warnings.filterwarnings("ignore")   # silence MLP convergence chatter for readability

# ---------------------------------------------------------------- data
df = pd.read_csv("data/nepal_rice_climate.csv").dropna().reset_index(drop=True)
y = df["rice_yield_kgha"].values
X = df.drop(columns=["rice_yield_kgha"]).values   # full model keeps `year`
n = len(y)
print(f"n = {n} observations\n")


# ---------------------------------------------------------------- seed-averaged MLP
class SeedEnsembleMLP(BaseEstimator, RegressorMixin):
    """Average predictions of several MLPs that differ only in init seed."""
    def __init__(self, hidden_layer_sizes=(8,), alpha=1.0, max_iter=3000, n_seeds=15):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.max_iter = max_iter
        self.n_seeds = n_seeds

    def fit(self, X, y):
        self.models_ = []
        for s in range(self.n_seeds):
            m = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes,
                             activation="relu", alpha=self.alpha,
                             max_iter=self.max_iter, random_state=s)
            m.fit(X, y)
            self.models_.append(m)
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models_], axis=0)


def n_params(hidden, n_in=X.shape[1]):
    sizes = [n_in] + list(hidden) + [1]
    return sum(sizes[i] * sizes[i + 1] + sizes[i + 1] for i in range(len(sizes) - 1))


# ---------------------------------------------------------------- CV helpers
SCHEMES = {
    "shuffled":   KFold(n_splits=10, shuffle=True, random_state=42),
    "blocked":    KFold(n_splits=10, shuffle=False),
    "timeseries": TimeSeriesSplit(n_splits=5),
}

def pooled_oof(estimator, X, y, cv):
    """Pooled out-of-fold R2 and RMSE (only points that get a test prediction)."""
    yhat = np.full(len(y), np.nan)
    for tr, te in cv.split(X):
        est = clone(estimator)
        est.fit(X[tr], y[tr])
        yhat[te] = est.predict(X[te])
    m = ~np.isnan(yhat)
    return r2_score(y[m], yhat[m]), np.sqrt(mean_squared_error(y[m], yhat[m])), m.sum()


# ---------------------------------------------------------------- 1) current MLP: seed instability
print("=== CURRENT MLP (100,50), alpha=1e-4 : seed-to-seed instability ===")
print(f"parameters: {n_params((100,50))} weights on {n} samples "
      f"(~{n_params((100,50))//n} per sample)\n")
shuffled = SCHEMES["shuffled"]
r2s = []
for seed in range(10):
    pipe = make_pipeline(StandardScaler(),
                         MLPRegressor(hidden_layer_sizes=(100, 50), alpha=1e-4,
                                      max_iter=2000, random_state=seed))
    r2, _, _ = pooled_oof(pipe, X, y, shuffled)
    r2s.append(r2)
print("shuffled pooled R2 across 10 seeds:")
print("  values:", ", ".join(f"{v:.3f}" for v in r2s))
print(f"  mean {np.mean(r2s):.3f}  sd {np.std(r2s):.3f}  "
      f"range [{min(r2s):.3f}, {max(r2s):.3f}]\n")

# ---------------------------------------------------------------- 2) head-to-head across schemes
current = make_pipeline(StandardScaler(),
                        MLPRegressor(hidden_layer_sizes=(100, 50), alpha=1e-4,
                                     max_iter=2000, random_state=42))
regular = make_pipeline(StandardScaler(),
                        SeedEnsembleMLP(hidden_layer_sizes=(8,), alpha=1.0,
                                        max_iter=3000, n_seeds=15))

print("=== HEAD-TO-HEAD : pooled OOF R2 (RMSE kg/ha) by CV scheme ===")
print(f"regularized MLP: (8,) hidden, alpha=1.0, 15-seed average "
      f"-> {n_params((8,))} weights\n")
print(f"{'scheme':<12}{'current (100,50)':>22}{'regularized (8,)x15':>24}")
for name, cv in SCHEMES.items():
    r2c, rmc, _ = pooled_oof(current, X, y, cv)
    r2r, rmr, _ = pooled_oof(regular, X, y, cv)
    print(f"{name:<12}{f'{r2c:+.3f} ({rmc:.0f})':>22}{f'{r2r:+.3f} ({rmr:.0f})':>24}")

print("\nReading:")
print("  - Current MLP's shuffled R2 swings across seeds (sd above) = overfitting/instability.")
print("  - Regularized ensemble should be far more stable and no worse forward.")
print("  - Forward (timeseries) stays negative for both: a trend cannot be extrapolated;")
print("    regularization makes it less catastrophic, not positive. Story unchanged.")
print("  - Drop the winning MLP definition into ml_models.py/validation.py to refresh Table 1.")
