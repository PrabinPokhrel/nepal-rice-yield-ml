"""
reg_mlp_for_validation.py
-------------------------
Drop-in replacement for the MLP in validation.py (your Table 1 generator).
Swapping only the MLP estimator keeps Linear / RandomForest / XGBoost
byte-identical, so the rest of Table 1 does not move; only the MLP row
refreshes. This estimator scales BOTH inputs and target internally, so it
works whether or not your CV loop already wraps models in a StandardScaler.

INTEGRATION (three steps in validation.py):

  1) add these imports near the top:
        import numpy as np
        from sklearn.base import BaseEstimator, RegressorMixin
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.neural_network import MLPRegressor

  2) paste the class + factory below (once, anywhere above your model list)

  3) find where validation.py currently defines the MLP, e.g.
        MLPRegressor(hidden_layer_sizes=(100, 50), alpha=1e-4,
                     max_iter=2000, random_state=42)
     and replace that single object with:
        make_reg_mlp()

Then re-run validation.py and paste the full output. It should print the same
Linear/RF/XGB numbers as before (a good consistency check) and new MLP numbers
for full + detrended across all three CV schemes. Runtime: a few minutes
(15 seeds x folds x schemes), all small networks.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor


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
            m = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes,
                             activation="relu", alpha=self.alpha,
                             max_iter=self.max_iter, random_state=s)
            m.fit(X, y)
            self.models_.append(m)
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models_], axis=0)


def make_reg_mlp(hidden=(8,), alpha=0.1, n_seeds=15):
    """Regularized MLP: small net + L2, target-scaled, seed-averaged.
    alpha=0.1 is a pre-specified, standard weight-decay value; the sweep in
    mlp_regularize_v2.py showed results are stable across alpha 0.01-1.0."""
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(),
                                SeedEnsembleMLP(hidden, alpha=alpha, n_seeds=n_seeds)),
        transformer=StandardScaler())
