# Nepal Rice Yield: Intensification vs. Climate (1963-2023)

Machine-learning analysis of Nepal's national rice-yield record, testing whether an
apparent climate signal survives once the long-term agricultural-intensification trend
is properly removed.

## Summary

Using a 60-year national dataset (1963–2023) linking FAOSTAT rice yield and fertilizer
use with ERA5 climate variables, we ask how much of Nepal's rice-yield history is
attributable to input intensification versus climate. The headline result is
methodological: an apparent interannual climate signal appears under conventional
shuffled cross-validation with **linear** detrending, but it does **not** survive
time-aware validation or more flexible trend removal (spline detrending,
first-differencing), and it collapses at every growing-season and phenological window
tested. At the national annual scale, the apparent signal is best explained by residual
trend confounding rather than a genuine climate response. The intensification trend,
captured by calendar year and fertilizer use, dominates the record.

This repository contains the data and all code needed to reproduce the analysis and
figures in the manuscript.

## Repository layout

data/
nepal_rice_climate.csv # 60 obs, 1963–2023: yield + 11 predictors + year


nepal_rice_seasonal.csv # monsoon growing-season / phenological-window aggregates

--- data preparation ---

build_seasonal.py # aggregates monthly ERA5 into seasonal/phenological windows

--- main analysis ---

table1_validation.py # Table 1: all 4 models x full/detrended x 3 CV schemes
classical_stats.py 

# Mann-Kendall + Sen slope, ADF stationarity arc, VIF
detrend_robustness.py 

# blocked-CV climate R2 vs trend-removal method
detrend_sensitivity.py 

# polynomial-degree sweep + spline-smoothing sweep
seasonal_robustness.py 

# robustness gauntlet at phenological resolution
shap_full.py 

# SHAP interpretation of the full (trend-retained) RF

--- figures ---

make_conceptual.py # Figure 1: conceptual trend-confounding schematic


figure01_explore.py # Figure 2: yield and temperature trends, scatter plots


figure02_rainfall.py # Figure 3: rainfall trend and rainfall-yield relationship


figure03_ml_results.py # Figure 4: model performance, importances, deviations


figure04_correlation.py # Figure 5: correlation matrix of yield and predictors


figure05_projection.py # Figure 6: partial dependence within observed range


figure06_detrend_robustness.py# Figure 7: detrend-robustness centrepiece

figures/
00_conceptual.png # Figure 1


01_exploratory.png # Figure 2


02_rainfall.png # Figure 3


03_ml_results.png # Figure 4


04_correlation_matrix.png # Figure 5


05_partial_dependence.png # Figure 6


06_detrend_robustness.png # Figure 7


07_shap_full_summary.png # Supplementary Figure S1

requirements.txt


## Data sources

- **Rice yield and fertilizer (N/P/K):** FAOSTAT, Food and Agriculture Organization of
  the United Nations (https://www.fao.org/faostat), accessed 15-16 July 2026.
- **Climate variables:** ERA5 reanalysis accessed through the World Bank Climate Change
  Knowledge Portal (https://climateknowledgeportal.worldbank.org), accessed 15-16 July
  2026.

The two CSVs in `data/` are the derived, analysis-ready tables. Raw downloads are not
tracked; see the sources above to regenerate them if needed.

## Setup

Python 3.x with the packages pinned in `requirements.txt`.

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

## Reproducing the results

Run all scripts from the repository root. Figures are written to the `figures/` folder.

```bash
# Step 1 - build the seasonal dataset
python build_seasonal.py

# Step 2 - main results (Table 1)
python table1_validation.py

# Step 3 - classical statistics
python classical_stats.py

# Step 4 - central robustness analyses
python detrend_robustness.py
python detrend_sensitivity.py

# Step 5 - seasonal robustness
python seasonal_robustness.py

# Step 6 - SHAP interpretation
python shap_full.py

# Step 7 - figures
python make_conceptual.py


python figure01_explore.py


python figure02_rainfall.py


python figure03_ml_results.py


python figure04_correlation.py


python figure05_projection.py


python figure06_detrend_robustness.py
```

## Key methodological notes

- **Temporally honest validation.** Results are reported under three cross-validation
  schemes: conventional shuffled 10-fold, blocked 10-fold, and expanding-window
  time-series splitting. Conclusions rest on the latter two.
- **Detrending is done in-fold.** The linear trend is fit on training years only and
  removed from both training and held-out folds.
- **Capacity-matched neural network.** Single hidden layer of 8 units, L2 weight decay
  (alpha = 0.1), target-scaled, predictions averaged over 15 random initializations.
- **Robustness of the null.** Confirmed by three independent methods: time-aware
  cross-validation, flexible trend removal (spline and first-difference), and
  growing-season / phenological-window analysis.

## Citation

Pokhrel P, Khatiwada D (in preparation) Machine-learning attribution of Nepal's
rice-yield history: disentangling agricultural intensification from climate
variability, 1963-2023.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for
details.
