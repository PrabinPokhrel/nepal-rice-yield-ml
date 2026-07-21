# figure03_ml_results.py
# ML model results visualisation
# Two panels: full model and detrended model comparison

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from xgboost import XGBRegressor
from scipy import stats as sp_stats
import warnings
warnings.filterwarnings("ignore")
import os

os.makedirs("figures", exist_ok=True)

df = pd.read_csv("data/nepal_rice_climate.csv")

# ── Detrend ───────────────────────────────────────────────────────────────────
slope_t, intercept_t, _, _, _ = sp_stats.linregress(
    df["year"], df["rice_yield_kgha"])
df["yield_trend"]     = slope_t * df["year"] + intercept_t
df["yield_detrended"] = df["rice_yield_kgha"] - df["yield_trend"]

features_full = ["year", "tas_mean", "tasmax", "tasmin",
                 "rainfall_mm", "humidity_pct", "hot_days",
                 "frost_days", "heavy_rain_days",
                 "nitrogen_tonnes", "phosphate_tonnes", "potash_tonnes"]

features_climate = ["tas_mean", "tasmax", "tasmin",
                    "rainfall_mm", "humidity_pct", "hot_days",
                    "frost_days", "heavy_rain_days",
                    "nitrogen_tonnes", "phosphate_tonnes", "potash_tonnes"]

kf = KFold(n_splits=10, shuffle=True, random_state=42)

models = {
    "Linear\nRegression":  LinearRegression(),
    "Random\nForest":      RandomForestRegressor(
                               n_estimators=200, random_state=42),
    "XGBoost":             XGBRegressor(
                               n_estimators=200, learning_rate=0.05,
                               max_depth=4, random_state=42,
                               verbosity=0),
    "MLP Neural\nNetwork": MLPRegressor(
                               hidden_layer_sizes=(100, 50),
                               activation="relu",
                               max_iter=2000, random_state=42),
}

def get_r2(X_raw, y):
    from sklearn.pipeline import make_pipeline
    r2s = {}
    for name, model in models.items():
        pipe = make_pipeline(StandardScaler(), model)
        scores = cross_val_score(pipe, X_raw, y, cv=kf, scoring="r2")
        r2s[name] = (scores.mean(), scores.std())
    return r2s

# Get results
print("Running Analysis 1...")
r2_full = get_r2(df[features_full].values,
                  df["rice_yield_kgha"].values)

print("Running Analysis 2...")
r2_det  = get_r2(df[features_climate].values,
                  df["yield_detrended"].values)

# Feature importance
print("Computing feature importance...")
scaler = StandardScaler()

rf1 = RandomForestRegressor(n_estimators=200, random_state=42)
rf1.fit(scaler.fit_transform(df[features_full].values),
        df["rice_yield_kgha"].values)
imp1 = dict(zip(features_full, rf1.feature_importances_))

rf2 = RandomForestRegressor(n_estimators=200, random_state=42)
rf2.fit(scaler.fit_transform(df[features_climate].values),
        df["yield_detrended"].values)
imp2 = dict(zip(features_climate, rf2.feature_importances_))

# ── Actual vs predicted for best model (Random Forest, full) ─────────────────
scaler2 = StandardScaler()
X_full  = scaler2.fit_transform(df[features_full].values)
rf_best = RandomForestRegressor(n_estimators=200, random_state=42)
rf_best.fit(X_full, df["rice_yield_kgha"].values)
y_pred  = rf_best.predict(X_full)

# ── Plot ──────────────────────────────────────────────────────────────────────
CRIMSON = "#DC143C"
BLUE    = "#1a3a6b"
GREEN   = "#2d6a2d"
ORANGE  = "#e07b00"
GREY    = "#888888"

fig = plt.figure(figsize=(22, 16))
fig.suptitle(
    "Machine Learning Models for Nepal Rice Yield Prediction 1963-2023\n"
    "Random Forest Best Model | ERA5 Climate + FAOSTAT Fertilizer Data",
    fontsize=13, fontweight="bold")

gs = gridspec.GridSpec(2, 3, figure=fig,
                        hspace=0.45, wspace=0.38)

# ── Panel A: R² comparison - full model ───────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
names  = list(r2_full.keys())
r2vals = [r2_full[n][0] for n in names]
r2std  = [r2_full[n][1] for n in names]
colors = [CRIMSON if v == max(r2vals) else BLUE for v in r2vals]
bars   = ax1.bar(names, r2vals, color=colors,
                  alpha=0.85, edgecolor="white", width=0.5,
                  yerr=r2std, capsize=4)
for bar, val in zip(bars, r2vals):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.01,
             f"{val:.3f}", ha="center", fontsize=9,
             fontweight="bold")
ax1.set_ylabel("R² (10-fold cross-validation)")
ax1.set_title("A. Model Performance\n(Full model with year)")
ax1.set_ylim(0, 1.1)
ax1.axhline(0.8, color=GREY, linestyle="--",
             alpha=0.5, linewidth=1, label="R²=0.8 threshold")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.2, axis="y")
ax1.tick_params(axis="x", labelsize=8)

# ── Panel B: R² comparison - detrended ───────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
r2vals2 = [r2_det[n][0] for n in names]
r2std2  = [r2_det[n][1] for n in names]
colors2 = [GREEN if v == max(r2vals2) else ORANGE for v in r2vals2]
bars2   = ax2.bar(names, r2vals2, color=colors2,
                   alpha=0.85, edgecolor="white", width=0.5,
                   yerr=r2std2, capsize=4)
for bar, val in zip(bars2, r2vals2):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.01,
             f"{val:.3f}", ha="center", fontsize=9,
             fontweight="bold")
ax2.set_ylabel("R² (10-fold cross-validation)")
ax2.set_title("B. Model Performance\n(Detrended yield - climate signal)")
ax2.set_ylim(-0.3, 0.6)
ax2.axhline(0, color="black", linewidth=0.8)
ax2.grid(True, alpha=0.2, axis="y")
ax2.tick_params(axis="x", labelsize=8)
ax2.text(0.03, 0.97,
         "Lower R² is expected:\ntime trend removed\nOnly climate signal remains",
         transform=ax2.transAxes, va="top", fontsize=8,
         bbox=dict(boxstyle="round,pad=0.3",
                   facecolor="white", edgecolor=GREEN, alpha=0.9))

# ── Panel C: Actual vs Predicted ──────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(df["rice_yield_kgha"], y_pred,
            c=df["year"], cmap="RdYlGn",
            s=60, alpha=0.8,
            edgecolors="white", linewidth=0.5)
min_v = min(df["rice_yield_kgha"].min(), y_pred.min())
max_v = max(df["rice_yield_kgha"].max(), y_pred.max())
ax3.plot([min_v, max_v], [min_v, max_v],
         color="black", linewidth=1.5,
         linestyle="--", label="Perfect prediction")
r2_full_fit = r2_full["Random\nForest"][0]
ax3.set_xlabel("Actual Rice Yield (kg/ha)")
ax3.set_ylabel("Predicted Rice Yield (kg/ha)")
ax3.set_title(f"C. Actual vs Predicted\n"
              f"Random Forest (R²={r2_full_fit:.3f})")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.2)

# ── Name mapping for readable labels ─────────────────────────────────────────
name_map = {
    "year":             "Year (time trend)",
    "tas_mean":         "Mean Temperature",
    "tasmax":           "Max Temperature",
    "tasmin":           "Min Temperature",
    "rainfall_mm":      "Rainfall",
    "humidity_pct":     "Humidity",
    "hot_days":         "Hot Days (>35°C)",
    "frost_days":       "Frost Days",
    "heavy_rain_days":  "Heavy Rain Days",
    "nitrogen_tonnes":  "Nitrogen Fertilizer",
    "phosphate_tonnes": "Phosphate Fertilizer",
    "potash_tonnes":    "Potash Fertilizer",
}

# ── Panel D: Feature importance - full model ──────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
imp1_sorted = sorted(imp1.items(), key=lambda x: x[1])
feat_labels = [name_map.get(f, f) for f, _ in imp1_sorted]
imp_vals    = [v for _, v in imp1_sorted]
colors4     = []
for f, _ in imp1_sorted:
    if "year" in f:            colors4.append(GREY)
    elif "tonnes" in f:        colors4.append(ORANGE)
    else:                      colors4.append(BLUE)
ax4.barh(feat_labels, imp_vals,
          color=colors4, alpha=0.85,
          edgecolor="white", height=0.6)
ax4.set_xlabel("Feature Importance")
ax4.set_title("D. Feature Importance\n(Full model)")
ax4.grid(True, alpha=0.2, axis="x")
from matplotlib.patches import Patch
legend4 = [Patch(color=GREY,   label="Time trend"),
           Patch(color=ORANGE, label="Fertilizer"),
           Patch(color=BLUE,   label="Climate")]
ax4.legend(handles=legend4, fontsize=8)

# ── Panel E: Feature importance - detrended ───────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
imp2_sorted = sorted(imp2.items(), key=lambda x: x[1])
feat_labels2 = [name_map.get(f, f) for f, _ in imp2_sorted]
imp_vals2    = [v for _, v in imp2_sorted]
colors5      = []
for f, _ in imp2_sorted:
    if "tonnes" in f: colors5.append(ORANGE)
    else:             colors5.append(BLUE)
ax5.barh(feat_labels2, imp_vals2,
          color=colors5, alpha=0.85,
          edgecolor="white", height=0.6)
ax5.set_xlabel("Feature Importance")
ax5.set_title("E. Feature Importance\n(Detrended - climate signal)")
ax5.grid(True, alpha=0.2, axis="x")
legend5 = [Patch(color=ORANGE, label="Fertilizer"),
           Patch(color=BLUE,   label="Climate")]
ax5.legend(handles=legend5, fontsize=8)

# ── Panel F: Yield trend and detrended ───────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
ax6.plot(df["year"], df["rice_yield_kgha"],
         color=CRIMSON, linewidth=2,
         label="Actual yield", zorder=3)
ax6.plot(df["year"], df["yield_trend"],
         color=GREY, linewidth=2,
         linestyle="--", label=f"Trend (+{slope_t:.0f} kg/ha/yr)")
ax6.fill_between(df["year"],
                  df["rice_yield_kgha"],
                  df["yield_trend"],
                  where=df["rice_yield_kgha"] >= df["yield_trend"],
                  color=GREEN, alpha=0.3,
                  label="Above trend")
ax6.fill_between(df["year"],
                  df["rice_yield_kgha"],
                  df["yield_trend"],
                  where=df["rice_yield_kgha"] < df["yield_trend"],
                  color=CRIMSON, alpha=0.3,
                  label="Below trend")
ax6.set_xlabel("Year")
ax6.set_ylabel("Rice Yield (kg/ha)")
ax6.set_title("F. Yield Trend and Deviations\n"
              "(green=above trend, red=below trend)")
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.2)
ax6.text(0.03, 0.97,
         "Deviations from trend\nare what climate variables\nexplain (R²=0.25)",
         transform=ax6.transAxes, va="top", fontsize=8,
         bbox=dict(boxstyle="round,pad=0.3",
                   facecolor="white",
                   edgecolor=CRIMSON, alpha=0.9))

plt.savefig("figures/03_ml_results.png", dpi=200)
plt.close()
print("Saved: figures/03_ml_results.png")