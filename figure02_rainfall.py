# figure02_rainfall.py
# Rainfall analysis and correlation with rice yield

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os

os.makedirs("figures", exist_ok=True)

df = pd.read_csv("data/nepal_rice_climate.csv")
yr0, yr1 = int(df["year"].min()), int(df["year"].max())

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle(f"Nepal Annual Rainfall and Rice Yield {yr0}-{yr1}\n"
             "ERA5 Reanalysis Data | World Bank Climate Knowledge Portal",
             fontsize=13, fontweight="bold")

# ── Panel A: Rainfall trend ───────────────────────────────────────────────────
ax = axes[0]
ax.bar(df["year"], df["rainfall_mm"],
       color="#1a3a6b", alpha=0.7, width=0.8)
slope_r, intercept_r, r_r, p_r, _ = stats.linregress(df["year"],
                                                       df["rainfall_mm"])
ax.plot(df["year"], slope_r * df["year"] + intercept_r,
        color="red", linewidth=2, linestyle="--",
        label=f"Trend: {slope_r:.1f} mm/year")

# Annotate notable years
for yr, label in [(2021, "2021\n(highest)"), (1992, "1992\n(lowest)")]:
    val = df.loc[df["year"] == yr, "rainfall_mm"].values
    if len(val):
        ax.annotate(label,
                    xy=(yr, val[0]),
                    xytext=(yr + 2, val[0] + 80),
                    fontsize=8, color="red",
                    arrowprops=dict(arrowstyle="->",
                                    color="red", lw=0.8))

ax.set_xlabel("Year")
ax.set_ylabel("Annual Rainfall (mm)")
ax.set_title(f"A. Nepal Annual Rainfall Trend\n"
             f"(slope={slope_r:.1f} mm/year, p={p_r:.3f})")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

# ── Panel B: Rainfall vs Rice yield scatter ───────────────────────────────────
ax2 = axes[1]
scatter = ax2.scatter(df["rainfall_mm"], df["rice_yield_kgha"],
                      c=df["year"], cmap="RdYlGn",
                      s=60, alpha=0.8,
                      edgecolors="white", linewidth=0.5)
plt.colorbar(scatter, ax=ax2, label="Year")
r_ry, p_ry = stats.pearsonr(df["rainfall_mm"], df["rice_yield_kgha"])

# Annotate 2021
yr21 = df[df["year"] == 2021]
ax2.annotate("2021\n(high rain,\nlow yield)",
             xy=(yr21["rainfall_mm"].values[0],
                 yr21["rice_yield_kgha"].values[0]),
             xytext=(yr21["rainfall_mm"].values[0] - 350,
                     yr21["rice_yield_kgha"].values[0] + 150),
             fontsize=8, color="red",
             arrowprops=dict(arrowstyle="->", color="red", lw=0.8))

ax2.set_xlabel("Annual Rainfall (mm)")
ax2.set_ylabel("Rice Yield (kg/ha)")
ax2.set_title(f"B. Rainfall vs Rice Yield\n(r={r_ry:.3f}, p={p_ry:.3f})")
ax2.grid(True, alpha=0.2)

# ── Panel C: Correlation comparison all variables ─────────────────────────────
ax3 = axes[2]
variables = ["tas_mean", "tasmax", "tasmin", "rainfall_mm"]
labels    = ["Mean\nTemp", "Max\nTemp", "Min\nTemp", "Rainfall"]
colors    = ["#e07b00", "#DC143C", "#2d6a2d", "#1a3a6b"]

correlations = []
for var in variables:
    r_val, p_val = stats.pearsonr(df[var], df["rice_yield_kgha"])
    correlations.append(r_val)

bars = ax3.bar(labels, correlations, color=colors,
               alpha=0.85, edgecolor="white", width=0.5)
for bar, val in zip(bars, correlations):
    ax3.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.01,
             f"r={val:.3f}",
             ha="center", fontsize=10, fontweight="bold")

ax3.axhline(0, color="black", linewidth=0.8)
ax3.set_ylabel("Pearson Correlation with Rice Yield")
ax3.set_title("C. Correlation of All Variables\nwith Rice Yield")
ax3.set_ylim(-0.2, 0.9)
ax3.grid(True, alpha=0.2, axis="y")
ax3.text(0.03, 0.97,
         "All temperature correlations\nare positive and significant\n"
         "Rainfall: weak positive\n(monsoon supports rain-fed farming)\nBut weaker than temperature",
         transform=ax3.transAxes, va="top", fontsize=8.5,
         bbox=dict(boxstyle="round,pad=0.3",
                   facecolor="white",
                   edgecolor="#1a3a6b", alpha=0.9))

plt.tight_layout()
plt.savefig("figures/02_rainfall.png", dpi=200)
plt.close()
print("Saved: figures/02_rainfall.png")
print()
print("=== CORRELATION SUMMARY ===")
for var, label, r_val in zip(variables, labels, correlations):
    print(f"  {label:12s}: r={r_val:.3f}")