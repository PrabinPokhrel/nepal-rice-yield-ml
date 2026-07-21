"""
classical_stats.py
Classical statistics to support claims already made in the manuscript:
  1. Yield trend       - Mann-Kendall (nonparametric trend test) + Sen's slope
  2. Rainfall trend    - same, to back the 'no significant rainfall trend' claim
  3. Stationarity      - Augmented Dickey-Fuller on raw yield (expect nonstationary,
                         justifying detrending) and on detrended yield (expect
                         stationary)
  4. Multicollinearity - Variance Inflation Factor for every predictor, to put a
                         number on the collinearity described qualitatively

Needs: pip install pymannkendall statsmodels   (statsmodels you likely have)
"""

import numpy as np
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path("data/nepal_rice_climate.csv")).sort_values("year").reset_index(drop=True)

# ---- 1 & 2: Mann-Kendall + Sen's slope --------------------------------------
import pymannkendall as mk

for col, label in [("rice_yield_kgha", "Rice yield"), ("rainfall_mm", "Rainfall")]:
    r = mk.original_test(df[col].values)
    print(f"{label:11s}: trend={r.trend:10s}  p={r.p:.4g}  "
          f"Sen slope={r.slope:.2f} per year  (tau={r.Tau:.3f})")

# ---- 3: Augmented Dickey-Fuller stationarity --------------------------------
from statsmodels.tsa.stattools import adfuller

# linear-detrended yield (matches main-text detrend)
yr, y = df["year"].values.astype(float), df["rice_yield_kgha"].values.astype(float)
s, i = np.polyfit(yr, y, 1)
resid = y - (s * yr + i)

print("\nADF stationarity (H0 = has a unit root / nonstationary):")
for series, name in [(y, "raw yield"), (resid, "linearly detrended yield")]:
    stat, p, *_ = adfuller(series, autolag="AIC")
    verdict = "stationary" if p < 0.05 else "NONstationary"
    print(f"  {name:26s}: ADF={stat:7.3f}  p={p:.4g}  -> {verdict}")

# ---- 4: Variance Inflation Factor -------------------------------------------
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

feats = [c for c in df.select_dtypes(include=[np.number]).columns
         if c not in ("rice_yield_kgha", "year")]
Xv = add_constant(df[feats])
print("\nVariance Inflation Factor (VIF>10 = severe collinearity):")
vif = [(f, variance_inflation_factor(Xv.values, k + 1)) for k, f in enumerate(feats)]
for f, v in sorted(vif, key=lambda t: -t[1]):
    flag = "  <-- severe" if v > 10 else ""
    print(f"  {f:18s}: {v:8.1f}{flag}")