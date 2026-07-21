"""
figure05_partial_dependence.py
Figure 5 for the Nepal rice-yield ML paper.

Partial dependence of rice yield on the two most important predictors,
tasmin (minimum temperature) and rainfall_mm, computed WITHIN the observed
range only. Both the full model (with Year / trend) and the detrended model
(climate signal only) are shown on each panel.

Why partial dependence instead of a future projection:
Random Forests predict by averaging training rows within each leaf, so they
cannot extrapolate beyond the observed range. Adding uniform warming pushed
temperatures past every value seen in 1963-2023, collapsing all warming levels
onto identical predictions. Partial dependence stays inside the observed range
and shows the response the model actually learned, which is an honest
interpretation rather than an artefact.

Run from the project root:
    python figure05_partial_dependence.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

# --- Config -----------------------------------------------------------------
DATA_PATH = Path("data/nepal_rice_climate.csv")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)
OUT_PATH = FIG_DIR / "05_partial_dependence.png"

YIELD_COL = "rice_yield_kgha"
YEAR_COL = "year"
PD_FEATURES = ["tasmin", "rainfall_mm"]   # the two panels
GRID_POINTS = 60
RF_KWARGS = dict(n_estimators=500, random_state=42)
# Match RF_KWARGS to ml_models.py, or load your saved models, for exact
# consistency with the Results table.

# --- Load -------------------------------------------------------------------
df = pd.read_csv(DATA_PATH).sort_values(YEAR_COL).reset_index(drop=True)
print("Loaded", DATA_PATH, "->", df.shape)

predictors = [c for c in df.select_dtypes(include=[np.number]).columns
              if c != YIELD_COL]
predictors_no_year = [c for c in predictors if c != YEAR_COL]

# --- Fit both models --------------------------------------------------------
rf_full = RandomForestRegressor(**RF_KWARGS)
rf_full.fit(df[predictors], df[YIELD_COL])

# Detrend yield on Year (linear); train on residuals without Year.
slope, intercept = np.polyfit(df[YEAR_COL], df[YIELD_COL], 1)
resid = df[YIELD_COL] - (slope * df[YEAR_COL] + intercept)
rf_det = RandomForestRegressor(**RF_KWARGS)
rf_det.fit(df[predictors_no_year], resid)

# --- Partial dependence (manual, centered) ----------------------------------
def partial_dependence(model, X, feature, grid):
    """Mean prediction as `feature` is swept across `grid`, averaging over the
    observed joint distribution of all other features."""
    base = X.copy()
    out = np.empty(len(grid))
    for i, v in enumerate(grid):
        base[feature] = v
        out[i] = model.predict(base).mean()
    return out

results = {}
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, feat in zip(axes, PD_FEATURES):
    grid = np.linspace(df[feat].min(), df[feat].max(), GRID_POINTS)
    pd_full = partial_dependence(rf_full, df[predictors], feat, grid)
    pd_det  = partial_dependence(rf_det,  df[predictors_no_year], feat, grid)
    results[feat] = (grid, pd_full, pd_det)

    # Center each curve on its own mean so both sit on a comparable
    # "change in yield" axis despite different absolute levels.
    pd_full_c = pd_full - pd_full.mean()
    pd_det_c  = pd_det  - pd_det.mean()

    ax.plot(grid, pd_full_c, "-", color="#b2182b", lw=2,
            label="Full model (with Year / trend)")
    ax.plot(grid, pd_det_c, "-", color="#2166ac", lw=2,
            label="Detrended model (climate signal only)")

    # Rug of observed values so readers see where the data actually is.
    ymin = ax.get_ylim()[0]
    ax.plot(df[feat], np.full(len(df), ymin), "|", color="0.4", ms=8, alpha=0.6)

    ax.axhline(0, color="0.7", lw=0.8, ls="--")
    ax.set_xlabel(feat)
    ax.set_title(f"Partial dependence on {feat}")
    ax.grid(alpha=0.25)

axes[0].set_ylabel("Centered partial dependence\n(change in yield, kg/ha)")
axes[0].legend(frameon=False, fontsize=9, loc="best")
fig.suptitle("Figure 5. Learned yield response within the observed range (1963-2023)",
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
print("Saved", OUT_PATH)

# --- Console summary --------------------------------------------------------
for feat, (grid, pd_full, pd_det) in results.items():
    print(f"\n{feat}: observed range {df[feat].min():.2f} to {df[feat].max():.2f}")
    print(f"  full model PD span:      {pd_full.max()-pd_full.min():7.1f} kg/ha")
    print(f"  detrended model PD span: {pd_det.max()-pd_det.min():7.1f} kg/ha")