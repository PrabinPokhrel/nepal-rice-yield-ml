# figure01_explore.py
# Exploratory visualisation of Nepal rice yield and climate data

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

os.makedirs("figures", exist_ok=True)

df = pd.read_csv("data/nepal_rice_climate.csv")
yr0, yr1 = int(df["year"].min()), int(df["year"].max())
print(f"Data loaded: {len(df)} observations ({yr0}-{yr1})")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f"Nepal Rice Yield and Climate Variables {yr0}-{yr1}\n"
             "FAOSTAT Rice Yield | ERA5 Temperature | World Bank",
             fontsize=13, fontweight="bold")

# ── Panel A: Rice yield trend ─────────────────────────────────────────────────
ax = axes[0, 0]
ax.plot(df["year"], df["rice_yield_kgha"],
        color="#DC143C", linewidth=2, marker="o", markersize=3)
slope, intercept, r, p, se = stats.linregress(df["year"], df["rice_yield_kgha"])
ax.plot(df["year"], slope * df["year"] + intercept,
        color="black", linewidth=1.5, linestyle="--",
        label=f"Trend: +{slope:.1f} kg/ha/year")
ax.set_xlabel("Year")
ax.set_ylabel("Rice Yield (kg/ha)")
ax.set_title(f"A. Nepal Rice Yield Trend\n(r²={r**2:.3f}, p<0.001)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

# ── Panel B: Temperature trend ────────────────────────────────────────────────
ax2 = axes[0, 1]
ax2.plot(df["year"], df["tas_mean"],
         color="#1a3a6b", linewidth=2, label="Mean temp")
ax2.plot(df["year"], df["tasmax"],
         color="#e07b00", linewidth=1.5, linestyle="--", label="Max temp")
ax2.plot(df["year"], df["tasmin"],
         color="#2d6a2d", linewidth=1.5, linestyle="--", label="Min temp")
slope_t, _, r_t, _, _ = stats.linregress(df["year"], df["tas_mean"])
ax2.set_xlabel("Year")
ax2.set_ylabel("Temperature (°C)")
ax2.set_title(f"B. Nepal Temperature Trend\n(mean: +{slope_t:.4f}°C/year)")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.2)

# ── Panel C: Yield vs Mean Temperature scatter ────────────────────────────────
ax3 = axes[1, 0]
scatter = ax3.scatter(df["tas_mean"], df["rice_yield_kgha"],
                      c=df["year"], cmap="RdYlGn",
                      s=60, alpha=0.8, edgecolors="white", linewidth=0.5)
plt.colorbar(scatter, ax=ax3, label="Year")
r_yt, p_yt = stats.pearsonr(df["tas_mean"], df["rice_yield_kgha"])
ax3.set_xlabel("Mean Temperature (°C)")
ax3.set_ylabel("Rice Yield (kg/ha)")
ax3.set_title(f"C. Yield vs Temperature\n(r={r_yt:.3f}, p={p_yt:.3f})")
ax3.grid(True, alpha=0.2)

# ── Panel D: Yield vs Max Temperature scatter ─────────────────────────────────
ax4 = axes[1, 1]
scatter2 = ax4.scatter(df["tasmax"], df["rice_yield_kgha"],
                       c=df["year"], cmap="RdYlGn",
                       s=60, alpha=0.8, edgecolors="white", linewidth=0.5)
plt.colorbar(scatter2, ax=ax4, label="Year")
r_ymax, p_ymax = stats.pearsonr(df["tasmax"], df["rice_yield_kgha"])
ax4.set_xlabel("Maximum Temperature (°C)")
ax4.set_ylabel("Rice Yield (kg/ha)")
ax4.set_title(f"D. Yield vs Max Temperature\n(r={r_ymax:.3f}, p={p_ymax:.3f})")
ax4.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("figures/01_exploratory.png", dpi=200)
plt.close()
print("Saved: figures/01_exploratory.png")

# ── Print correlation summary ─────────────────────────────────────────────────
print("\n=== CORRELATION SUMMARY ===")
for col in ["tas_mean", "tasmax", "tasmin"]:
    r_val, p_val = stats.pearsonr(df[col], df["rice_yield_kgha"])
    print(f"  Rice yield vs {col:12s}: r={r_val:.3f}, p={p_val:.4f}")

print(f"\nRice yield trend: +{slope:.1f} kg/ha per year")
print(f"Temperature trend: +{slope_t:.4f} °C per year")