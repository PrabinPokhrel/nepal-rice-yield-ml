import numpy as np, pandas as pd
from pathlib import Path
from statsmodels.tsa.stattools import adfuller
from scipy.interpolate import UnivariateSpline

df = pd.read_csv(Path("data/nepal_rice_climate.csv")).sort_values("year").reset_index(drop=True)
yr, y = df["year"].values.astype(float), df["rice_yield_kgha"].values.astype(float)

# spline-detrended (smooth trend removed)
spl = UnivariateSpline(yr, y, k=3, s=0.30 * len(yr) * np.var(y))
spline_resid = y - spl(yr)
# first-differenced
firstdiff = np.diff(y)

for series, name in [(y, "raw yield"),
                     (y - np.polyval(np.polyfit(yr, y, 1), yr), "linear-detrended"),
                     (spline_resid, "spline-detrended"),
                     (firstdiff, "first-differenced")]:
    stat, p, *_ = adfuller(series, autolag="AIC")
    print(f"{name:20s}: ADF={stat:7.3f}  p={p:.4g}  -> "
          f"{'stationary' if p < 0.05 else 'NONstationary'}")