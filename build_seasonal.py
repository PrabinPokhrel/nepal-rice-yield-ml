"""
build_seasonal.py
Aggregate monthly ERA5 (World Bank CCKP, era5-x0.25, same product as the annual
data) into main-monsoon growing-season and phenological-window predictors for
Nepal rice, then merge with rice yield.

Rationale: ~92% of Nepal's rice is main-season (Barkhe) rice, grown Jun/Jul to
Oct/Nov, so the national FAOSTAT yield is effectively a monsoon signal. We build:
  - whole main season (Jun-Oct) means/sum, and
  - three phenological windows:
      establishment  Jun-Jul
      flowering      Aug-Sep   (most heat/water sensitive)
      grain-fill     Oct       (Oct-Nov, but Nov contributes little and overlaps
                                harvest; kept to Oct here, extendable)

Temperature windows -> mean of the monthly means (tas), and mean of monthly
tasmax / tasmin. Precipitation windows -> SUM of monthly totals (mm).

Output: data/nepal_rice_seasonal.csv, one row per year, ready for the same
robustness gauntlet used on the annual data.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path("data")
TEMP_XLSX = DATA / "era5-x0.25_timeseries_tasmax,tas,tasmin_timeseries_monthly_1950-2025_mean_historical_era5_x0.25_mean.xlsx"
PR_XLSX   = DATA / "era5-x0.25_timeseries_pr_timeseries_monthly_1950-2025_mean_historical_era5_x0.25_mean.xlsx"
YIELD_CSV = DATA / "nepal_rice_climate.csv"   # reuse existing yield column

# ----------------------------------------------------------------------------
def monthly_long(xlsx, sheet, varname):
    """Read one wide 1-row CCKP sheet -> long df with year, month, value."""
    df = pd.read_excel(xlsx, sheet_name=sheet)
    month_cols = [c for c in df.columns if isinstance(c, str) and "-" in c and c[:4].isdigit()]
    s = df.iloc[0][month_cols]
    out = pd.DataFrame({
        "year":  [int(c[:4]) for c in month_cols],
        "month": [int(c[5:7]) for c in month_cols],
        varname: s.values.astype(float),
    })
    return out

# temperature: three sheets
tas    = monthly_long(TEMP_XLSX, "tas",    "tas")
tasmax = monthly_long(TEMP_XLSX, "tasmax", "tasmax")
tasmin = monthly_long(TEMP_XLSX, "tasmin", "tasmin")
pr     = monthly_long(PR_XLSX,   "all",    "pr")

m = tas.merge(tasmax, on=["year","month"]).merge(tasmin, on=["year","month"]).merge(pr, on=["year","month"])

# ----------------------------------------------------------------------------
# window definitions: (name, months, is_precip_summed)
WINDOWS = {
    "season":       [6,7,8,9,10],   # whole main monsoon growing season
    "establish":    [6,7],          # transplanting + vegetative
    "flower":       [8,9],          # flowering / reproductive (most sensitive)
    "grainfill":    [10],           # grain filling (Oct)
}

def aggregate(group):
    row = {}
    for wname, months in WINDOWS.items():
        sub = group[group["month"].isin(months)]
        # temperature -> mean of monthly values
        row[f"tas_{wname}"]    = sub["tas"].mean()
        row[f"tasmax_{wname}"] = sub["tasmax"].mean()
        row[f"tasmin_{wname}"] = sub["tasmin"].mean()
        # precipitation -> SUM of monthly totals (mm)
        row[f"pr_{wname}"]     = sub["pr"].sum()
    return pd.Series(row)

seasonal = m.groupby("year").apply(aggregate, include_groups=False).reset_index()

# ----------------------------------------------------------------------------
# merge with existing yield (drops to the yield's year coverage, e.g. 1963-2023)
yld = pd.read_csv(YIELD_CSV)[["year", "rice_yield_kgha"]]
out = yld.merge(seasonal, on="year", how="inner").sort_values("year").reset_index(drop=True)

out.to_csv(DATA / "nepal_rice_seasonal.csv", index=False)
print(f"Wrote {DATA/'nepal_rice_seasonal.csv'}: {out.shape[0]} years, {out.shape[1]} cols")
print(f"Year range: {out.year.min()}-{out.year.max()}  (missing: "
      f"{sorted(set(range(out.year.min(), out.year.max()+1)) - set(out.year))})")
print("\nColumns:", list(out.columns))
print("\nSeason-mean sanity check (first + last 3 years):")
cols = ["year","tas_season","tasmax_flower","tasmin_flower","pr_season","pr_flower"]
print(pd.concat([out[cols].head(3), out[cols].tail(3)]).to_string(index=False))
print("\nSeasonal vs known annual (should be warmer + wetter than annual means):")
print(f"  tas_season mean:  {out['tas_season'].mean():.2f} C   (annual tas_mean ~ lower, includes winter)")
print(f"  pr_season mean:   {out['pr_season'].mean():.0f} mm   (annual rainfall ~1900mm; season is most of it)")
