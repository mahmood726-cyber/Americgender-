# Americgender-

County-level analysis of the **gender gap in cardiovascular disease (CVD) mortality**
across US counties, using embedded CDC mortality data and County Health Rankings
socioeconomic/behavioral metrics, with a Random Forest model for interpretation.

## What it does

`gender_gap_model.py`:

1. Loads two **embedded** datasets (no external files required):
   - 2019 CDC age-adjusted Heart Disease Mortality, stratified by sex, per county.
   - 2023 County Health Rankings metrics (% smokers, % obesity, % physically
     inactive, % uninsured, primary-care-physician rate, % some college,
     % children in poverty, median household income).
2. Computes the **gender gap** per county as `Male - Female` age-adjusted
   mortality rate.
3. Merges the gender gap with the health-rankings features on a cleaned 5-digit
   FIPS code (inner join).
4. Trains a `RandomForestRegressor` to predict the gender gap from the
   county-level features, reporting held-out **R²**.
5. Interprets the model with **SHAP** when available, otherwise with random-forest
   feature importances, and prints a short publication-style abstract.
6. Saves a gender-gap distribution histogram (and SHAP plots if SHAP is installed).

On the embedded data the pipeline merges **15 counties** (those present in both
datasets) and reports an R² around 0.39 (the dataset is small and illustrative —
treat results as a worked example, not a publishable finding).

### Data handling

All data is **embedded directly in the script** as CSV strings, so the analysis
runs end-to-end **offline** with no data download. The merge is an inner join on
FIPS code, so a county missing from either source (e.g. Denver County, which has
mortality data but no matching health-ranking row) is dropped.

## How to run

```bash
pip install -r requirements.txt
python gender_gap_model.py
```

This prints progress and the generated abstract, and writes
`1_gender_gap_distribution.png` (plus SHAP figures if the optional `shap`
package is installed). Generated figures are gitignored.

Importing the module is **side-effect-free** — no I/O, plotting, or training runs
on import, so the core functions (`build_merged_dataset`, `compute_gender_gap`,
`prepare_features`, `train_model`, `describe_drivers`, `run_analysis`, …) can be
reused or tested directly.

## How to test

```bash
python -m pytest -q
```

15 tests cover the importable core: gender-gap = Male − Female, FIPS padding,
single-sex-county handling, no target leakage, all-features-numeric after
encoding, train/test split disjointness (no leakage), prediction shape, bounded
finite R², a planted-signal feature getting top importance, reproducibility under
a fixed seed, and an end-to-end run on the embedded data.

## Dependencies

scikit-learn, pandas, numpy, seaborn, matplotlib (pinned by major version in
`requirements.txt`); pytest for the test suite. `shap` is **optional** — the
script falls back to random-forest feature importances when it is absent.

## Notes / caveats

- The embedded sample is small (15 merged counties); R² and importances are
  illustrative, not robust population estimates.
- `LocationAbbr` / `LocationDesc` are included as encoded categorical features,
  matching the original analysis design; they act as county/state identifiers
  rather than mechanistic drivers.
- matplotlib uses the non-interactive `Agg` backend so the script runs headless.
