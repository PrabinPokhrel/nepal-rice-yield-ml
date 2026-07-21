"""
figure04_correlation.py
Correlation matrix (Figure 4) for the Nepal rice-yield ML paper.

Loads the final dataset produced by data_build.py and plots a Pearson
correlation heatmap across rice yield and all predictors.

Run from the project root:
    python figure04_correlation.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------------------------
# Config - point this at whatever data_build.py writes out.
# ---------------------------------------------------------------------------
DATA_PATH = Path("data/nepal_rice_climate.csv")  
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)
OUT_PATH = FIG_DIR / "04_correlation_matrix.png"

# Column that holds the target, so we can order the matrix with it first.
# Change this string to match your actual yield column name.
YIELD_COL = "rice_yield_kgha"

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print("Loaded", DATA_PATH, "->", df.shape)
print("Columns:", list(df.columns))

# Keep numeric predictors only; drop Year from the matrix so the plot shows
# physical relationships rather than the time index (Year correlates with
# everything because of the shared trend and would dominate the colour scale).
num = df.select_dtypes(include=[np.number]).copy()
for drop_col in ["Year", "year", "YEAR"]:
    if drop_col in num.columns:
        num = num.drop(columns=drop_col)

# Put yield first if present, so the top row / left column reads as
# "predictor vs yield".
if YIELD_COL in num.columns:
    ordered = [YIELD_COL] + [c for c in num.columns if c != YIELD_COL]
    num = num[ordered]
else:
    print(f"WARNING: '{YIELD_COL}' not found. Check YIELD_COL. "
          f"Available: {list(num.columns)}")

corr = num.corr(method="pearson")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

fig, ax = plt.subplots(figsize=(9, 7.5))

# Mask the upper triangle for a cleaner read.
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

sns.heatmap(
    corr,
    mask=mask,
    cmap="RdBu_r",
    vmin=-1, vmax=1,
    center=0,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 7},
    square=True,
    linewidths=0.5,
    linecolor="white",
    cbar_kws={"shrink": 0.7, "label": "Pearson r"},
    ax=ax,
)

ax.set_title("Correlation matrix: rice yield and predictors (1963-2023)",
             fontsize=11, pad=12)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
print("Saved", OUT_PATH)

# ---------------------------------------------------------------------------
# Console summary - the numbers you'll want for the Results text.
# ---------------------------------------------------------------------------
if YIELD_COL in corr.columns:
    yield_corr = corr[YIELD_COL].drop(YIELD_COL).sort_values(
        key=lambda s: s.abs(), ascending=False)
    print("\nCorrelation with rice yield, strongest first:")
    print(yield_corr.round(3).to_string())