# Nepal Rice Yield: Intensification vs. Climate (1963–2023)

Machine-learning analysis of Nepal's national rice-yield record, testing whether an
apparent climate signal survives once the long-term agricultural-intensification trend
is properly removed.

## Summary

Using a 60-year national dataset (1963–2023) of FAOSTAT rice yield and fertilizer use
together with ERA5 climate variables, we ask how much of Nepal's rice-yield history is
attributable to input intensification versus climate. The headline result is
methodological: an apparent interannual climate signal appears under conventional
shuffled cross-validation with **linear** detrending, but it does **not** survive
time-aware validation or more flexible trend removal (spline detrending,
first-differencing), and it collapses at every growing-season and phenological window
tested. At the national annual scale, the apparent signal is best explained by residual
trend confounding rather than a genuine climate response. The intensification trend
(captured by calendar year and fertilizer use) dominates the record.

This repository contains the data and all code needed to reproduce the analysis and
figures.

## Repository layout

```
data/
  nepal_rice_climate.csv     # 60 obs, 1963–2023: yield + 11 predictors + year
  nepal_rice_seasonal.csv    # growing-season / phenological-window climate aggregates
build_seasonal.py            # builds nepal_rice_seasonal.csv from monthly ERA5 inputs
classical_stats.py           # Mann–Kendall + Sen slope, ADF stationarity arc, VIF
table1_validation.py         # Table 1: 4 models x full/detrended x 3 CV schemes
detrend_robustness.py        # signal vs. trend-removal method (linear/spline/first-diff)
detrend_sensitivity.py       # polynomial-degree and spline-smoothing sweeps
seasonal_robustness.py       # robustness gauntlet at seasonal/phenological resolution
shap_full.py                 # SHAP interpretation of the full (trend-retained) model
figure01_explore.py          # exploratory: yield & temperature trends, scatters
figure02_rainfall.py         # rainfall trend and rainfall–yield relationship
figure03_ml_results.py       # model performance, importances, trend deviations
figure04_correlation.py      # correlation matrix of yield and predictors
figure05_projection.py       # partial dependence within the observed range
figure06_detrend_robustness.py  # detrend-robustness centrepiece
make_conceptual.py           # conceptual schematic of the trend-confounding mechanism
requirements.txt
```

(Adjust the list to match your actual filenames if any differ.)

## Data sources

- **Rice yield and fertilizer (N/P/K):** FAOSTAT, Food and Agriculture Organization of
  the United Nations (https://www.fao.org/faostat).
- **Climate variables:** ERA5 reanalysis, accessed through the World Bank Climate Change
  Knowledge Portal (https://climateknowledgeportal.worldbank.org).

The two CSVs in `data/` are the derived, analysis-ready tables. Raw downloads are not
tracked; see the sources above to regenerate them.

## Setup

Python 3.x with the packages pinned in `requirements.txt`.

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

## Reproducing the results

Run from the repository root (scripts read `data/nepal_rice_climate.csv`). Figures are
written to a `figures/` folder.

```bash
# main results table (model performance, full and detrended, three CV schemes)
python table1_validation.py

# classical statistics: trend tests, stationarity, collinearity
python classical_stats.py

# the central robustness analyses
python detrend_robustness.py
python detrend_sensitivity.py

# seasonal / phenological-window robustness (needs nepal_rice_seasonal.csv)
python build_seasonal.py
python seasonal_robustness.py

# interpretation and figures
python shap_full.py
python figure01_explore.py
python figure02_rainfall.py
python figure03_ml_results.py
python figure04_correlation.py
python figure05_projection.py
python figure06_detrend_robustness.py
python make_conceptual.py
```

## Notes on method

- **Validation is temporally honest.** Results are reported under shuffled, blocked, and
  forward-chaining (time-series) cross-validation; conclusions rest on the latter two.
- **Detrending is done in-fold** where relevant (trend fit on the training years only).
- **The neural network is capacity-matched to the sample.** With only 60 observations it
  uses a single small hidden layer with L2 weight decay, standardized inputs and target,
  and predictions averaged over several initializations.

## Citation

Pokhrel, P., & Khatiwada, D. (in preparation). *Machine-learning attribution of Nepal's
rice-yield history: disentangling agricultural intensification from climate variability,
1963–2023.*

## License

See `LICENSE`.
