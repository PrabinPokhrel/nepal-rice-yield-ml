# data_build.py
# Loads and merges all datasets for Nepal rice yield ML study
# Sources:
#   - FAOSTAT: Nepal rice yield 1961-2024
#   - ERA5 World Bank: Temperature, rainfall, humidity,
#                      hot days, frost days, heavy rain days
#   - FAOSTAT: Fertilizer use (nitrogen, phosphate, potash)

import pandas as pd
import os

os.makedirs("data", exist_ok=True)

# ── File names ────────────────────────────────────────────────────────────────
RICE_FILE    = "FAOSTAT_data_en_7-10-2026.csv"
FERT_FILE    = "FAOSTAT_data_en_7-15-2026.csv"
TEMP_FILE    = "era5-x0.25_timeseries_tasmax,tas,tasmin_timeseries_annual_1950-2024_mean_historical_era5_x0.25_mean.xlsx"
RAIN_FILE    = "era5-x0.25_timeseries_pr_timeseries_annual_1950-2024_mean_historical_era5_x0.25_mean.xlsx"
HURS_FILE    = "era5-x0.25_timeseries_hurs_timeseries_annual_1950-1950,1950-2024_mean_historical_era5_x0.25_mean.xlsx"
CLIM2_FILE   = "era5-x0.25_timeseries_r50mm,fd,hd35_timeseries_annual_1950-2024_mean_historical_era5_x0.25_mean.xlsx"

def extract_nepal(xl_file, sheet, suffix="-07"):
    """Extract Nepal annual time series from World Bank ERA5 Excel file."""
    xl = pd.ExcelFile(xl_file)
    df = pd.read_excel(xl, sheet_name=sheet)
    nepal = df[df["name"].str.contains("Nepal", case=False, na=False)].copy()
    year_cols = [c for c in nepal.columns if str(c).endswith(suffix)]
    years  = [int(str(c).split("-")[0]) for c in year_cols]
    values = nepal[year_cols].values[0]
    return pd.Series(values, index=years, name=sheet)

# ── Step 1: Rice yield ────────────────────────────────────────────────────────
print("Loading rice yield...")
rice = pd.read_csv(RICE_FILE)[["Year","Value"]].copy()
rice.columns = ["year","rice_yield_kgha"]
rice = rice.sort_values("year").reset_index(drop=True)
print(f"  {len(rice)} years: {rice.year.min()}-{rice.year.max()}")
print(f"  Yield: {rice.rice_yield_kgha.min():.0f} to "
      f"{rice.rice_yield_kgha.max():.0f} kg/ha")

# ── Step 2: Temperature ───────────────────────────────────────────────────────
print("Loading temperature...")
temp = pd.DataFrame({
    "year":     extract_nepal(TEMP_FILE, "tas").index,
    "tas_mean": extract_nepal(TEMP_FILE, "tas").values,
    "tasmax":   extract_nepal(TEMP_FILE, "tasmax").values,
    "tasmin":   extract_nepal(TEMP_FILE, "tasmin").values,
})
print(f"  {len(temp)} years: {temp.year.min()}-{temp.year.max()}")

# ── Step 3: Rainfall ──────────────────────────────────────────────────────────
print("Loading rainfall...")
rain_s = extract_nepal(RAIN_FILE, "all")
rainfall = pd.DataFrame({"year": rain_s.index, "rainfall_mm": rain_s.values})
print(f"  {len(rainfall)} years: {rainfall.year.min()}-{rainfall.year.max()}")
print(f"  Range: {rainfall.rainfall_mm.min():.0f} to "
      f"{rainfall.rainfall_mm.max():.0f} mm")

# ── Step 4: Humidity ──────────────────────────────────────────────────────────
print("Loading humidity...")
hurs_s = extract_nepal(HURS_FILE, "1950-2024")
humidity = pd.DataFrame({"year": hurs_s.index, "humidity_pct": hurs_s.values})
print(f"  {len(humidity)} years: {humidity.year.min()}-{humidity.year.max()}")
print(f"  Range: {humidity.humidity_pct.min():.1f} to "
      f"{humidity.humidity_pct.max():.1f} %")

# ── Step 5: Hot days, frost days, heavy rain days ─────────────────────────────
print("Loading climate stress variables...")
hd35_s  = extract_nepal(CLIM2_FILE, "hd35")
fd_s    = extract_nepal(CLIM2_FILE, "fd")
r50mm_s = extract_nepal(CLIM2_FILE, "r50mm")
climate2 = pd.DataFrame({
    "year":       hd35_s.index,
    "hot_days":   hd35_s.values,
    "frost_days": fd_s.values,
    "heavy_rain_days": r50mm_s.values,
})
print(f"  {len(climate2)} years: {climate2.year.min()}-{climate2.year.max()}")
print(f"  Hot days (>35C):     {climate2.hot_days.min():.1f} to "
      f"{climate2.hot_days.max():.1f}")
print(f"  Frost days:          {climate2.frost_days.min():.1f} to "
      f"{climate2.frost_days.max():.1f}")
print(f"  Heavy rain days:     {climate2.heavy_rain_days.min():.1f} to "
      f"{climate2.heavy_rain_days.max():.1f}")

# ── Step 6: Fertilizer ────────────────────────────────────────────────────────
print("Loading fertilizer data...")
fert_raw = pd.read_csv(FERT_FILE)
nitrogen  = fert_raw[fert_raw["Item"].str.contains("nitrogen",  case=False)]
phosphate = fert_raw[fert_raw["Item"].str.contains("phosphate", case=False)]
potash    = fert_raw[fert_raw["Item"].str.contains("potash",    case=False)]

fertilizer = pd.DataFrame({
    "year":              nitrogen["Year"].values,
    "nitrogen_tonnes":   nitrogen["Value"].values,
}).merge(
    pd.DataFrame({"year": phosphate["Year"].values,
                  "phosphate_tonnes": phosphate["Value"].values}),
    on="year", how="outer"
).merge(
    pd.DataFrame({"year": potash["Year"].values,
                  "potash_tonnes": potash["Value"].values}),
    on="year", how="outer"
).sort_values("year").reset_index(drop=True)

print(f"  {len(fertilizer)} years: "
      f"{fertilizer.year.min()}-{fertilizer.year.max()}")
print(f"  Nitrogen: {fertilizer.nitrogen_tonnes.min():.0f} to "
      f"{fertilizer.nitrogen_tonnes.max():.0f} tonnes")

# ── Step 7: Merge all datasets ────────────────────────────────────────────────
print()
print("Merging all datasets...")
merged = rice.copy()
for df in [temp, rainfall, humidity, climate2, fertilizer]:
    merged = pd.merge(merged, df, on="year", how="inner")

merged = merged.sort_values("year").reset_index(drop=True)
merged = merged.dropna()

print(f"  Final shape: {merged.shape}")
print(f"  Years: {merged.year.min()} to {merged.year.max()}")
print(f"  Columns: {list(merged.columns)}")
print()
print("First 3 rows:")
print(merged.head(3).to_string(index=False))
print()
print("Last 3 rows:")
print(merged.tail(3).to_string(index=False))

# ── Step 8: Save ──────────────────────────────────────────────────────────────
merged.to_csv("data/nepal_rice_climate.csv", index=False)
print()
print(f"Saved: data/nepal_rice_climate.csv")
print()
print("=== DATASET SUMMARY ===")
print(merged.describe().round(2).to_string())