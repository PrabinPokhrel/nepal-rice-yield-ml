import pandas as pd
df = pd.read_csv("data/nepal_rice_climate.csv")
print("rows:", len(df), "| range:", df.year.min(), "to", df.year.max())
missing = sorted(set(range(df.year.min(), df.year.max()+1)) - set(df.year))
print("missing interior years:", missing)

# raw yield extent, to see if 2024 exists upstream
rice = pd.read_csv("FAOSTAT_data_en_7-10-2026.csv")
print("raw yield years:", rice.Year.min(), "to", rice.Year.max())

# correlations recomputed on the 60-obs set, for the main text
for c in ["tasmin","tas_mean","tasmax","rainfall_mm","humidity_pct","frost_days"]:
    print(f"{c:14s} r={df['rice_yield_kgha'].corr(df[c]):+.3f}")