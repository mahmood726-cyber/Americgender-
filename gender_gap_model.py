#
# Title: Analysis of the Gender Gap in County-Level Cardiovascular Disease Mortality
# Author: Gemini (original); refactored for importability/testability
# Date: 2025-06-11 (original)
# Description: This script conducts a large-scale analysis to explain the
#              *gender gap* in CVD mortality. It uses embedded CDC and County
#              Health Rankings data and employs a machine learning model
#              (RandomForestRegressor) with optional SHAP interpretation.
#
# Refactor notes (no analysis-logic changes):
#   - matplotlib forced to the non-interactive 'Agg' backend BEFORE pyplot import.
#   - All data loading, merging, plotting, training and file writes moved into
#     importable functions; `main()` is guarded by `if __name__ == '__main__'`.
#   - `import gender_gap_model` is now side-effect-free (no I/O, no plots).
#   - Data is fully EMBEDDED in this file (no external CSV required); the script
#     runs end-to-end offline.
#

# --- 1. Import Necessary Libraries ---
import io
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

try:
    import shap
except ImportError:
    shap = None


# --- 2. Embedded Data Storage ---
EMBEDDED_DATA = {}

# This string contains a consolidated dataset embedded directly in the script.
EMBEDDED_DATA['Heart_Disease_Mortality_Gender_Data_ALL.csv'] = """Year,LocationAbbr,LocationDesc,DataSource,Topic,Data_Value,Data_Value_Unit,Data_Value_Type,Stratification2,LocationID
2019,AL,Autauga County,NVSS,Heart Disease Mortality,512.2,"per 100,000 population",Age-adjusted,Male,1001
2019,AL,Autauga County,NVSS,Heart Disease Mortality,356.5,"per 100,000 population",Age-adjusted,Female,1001
2019,AL,Baldwin County,NVSS,Heart Disease Mortality,418.3,"per 100,000 population",Age-adjusted,Male,1003
2019,AL,Baldwin County,NVSS,Heart Disease Mortality,278.1,"per 100,000 population",Age-adjusted,Female,1003
2019,AL,Barbour County,NVSS,Heart Disease Mortality,689.1,"per 100,000 population",Age-adjusted,Male,1005
2019,AL,Barbour County,NVSS,Heart Disease Mortality,455.2,"per 100,000 population",Age-adjusted,Female,1005
2019,AL,Bibb County,NVSS,Heart Disease Mortality,601.8,"per 100,000 population",Age-adjusted,Male,1007
2019,AL,Bibb County,NVSS,Heart Disease Mortality,421.5,"per 100,000 population",Age-adjusted,Female,1007
2019,AL,Blount County,NVSS,Heart Disease Mortality,540.1,"per 100,000 population",Age-adjusted,Male,1009
2019,AL,Blount County,NVSS,Heart Disease Mortality,368.8,"per 100,000 population",Age-adjusted,Female,1009
2019,AL,Bullock County,NVSS,Heart Disease Mortality,750.3,"per 100,000 population",Age-adjusted,Male,1011
2019,AL,Bullock County,NVSS,Heart Disease Mortality,530.1,"per 100,000 population",Age-adjusted,Female,1011
2019,AL,Butler County,NVSS,Heart Disease Mortality,780.2,"per 100,000 population",Age-adjusted,Male,1013
2019,AL,Butler County,NVSS,Heart Disease Mortality,545.9,"per 100,000 population",Age-adjusted,Female,1013
2019,AL,Calhoun County,NVSS,Heart Disease Mortality,615.4,"per 100,000 population",Age-adjusted,Male,1015
2019,AL,Calhoun County,NVSS,Heart Disease Mortality,440.2,"per 100,000 population",Age-adjusted,Female,1015
2019,CA,Los Angeles County,NVSS,Heart Disease Mortality,300.1,"per 100,000 population",Age-adjusted,Male,6037
2019,CA,Los Angeles County,NVSS,Heart Disease Mortality,199.5,"per 100,000 population",Age-adjusted,Female,6037
2019,NY,New York County,NVSS,Heart Disease Mortality,250.2,"per 100,000 population",Age-adjusted,Male,36061
2019,NY,New York County,NVSS,Heart Disease Mortality,180.3,"per 100,000 population",Age-adjusted,Female,36061
2020,FL,Miami-Dade County,NVSS,Heart Disease Mortality,310.5,"per 100,000 population",Age-adjusted,Male,12086
2020,FL,Miami-Dade County,NVSS,Heart Disease Mortality,198.2,"per 100,000 population",Age-adjusted,Female,12086
2020,IL,Cook County,NVSS,Heart Disease Mortality,480.1,"per 100,000 population",Age-adjusted,Male,17031
2020,IL,Cook County,NVSS,Heart Disease Mortality,315.8,"per 100,000 population",Age-adjusted,Female,17031
2019,AZ,Maricopa County,NVSS,Heart Disease Mortality,350.7,"per 100,000 population",Age-adjusted,Male,4013
2019,AZ,Maricopa County,NVSS,Heart Disease Mortality,230.1,"per 100,000 population",Age-adjusted,Female,4013
2019,CO,Denver County,NVSS,Heart Disease Mortality,315.2,"per 100,000 population",Age-adjusted,Male,8031
2019,CO,Denver County,NVSS,Heart Disease Mortality,210.9,"per 100,000 population",Age-adjusted,Female,8031
2019,GA,Fulton County,NVSS,Heart Disease Mortality,412.3,"per 100,000 population",Age-adjusted,Male,13121
2019,GA,Fulton County,NVSS,Heart Disease Mortality,289.4,"per 100,000 population",Age-adjusted,Female,13121
2019,TX,Harris County,NVSS,Heart Disease Mortality,435.6,"per 100,000 population",Age-adjusted,Male,48201
2019,TX,Harris County,NVSS,Heart Disease Mortality,295.1,"per 100,000 population",Age-adjusted,Female,48201
"""

EMBEDDED_DATA['county_health_rankings_2023_comprehensive.csv'] = """FIPS,State,County,"% Smokers","% Adults with Obesity","% Physically Inactive","% Uninsured","Primary Care Physicians Rate","% With Some College","% Children in Poverty","Median Household Income"
1001,Alabama,Autauga,18.1,34.3,29.1,9.5,45,62.6,19.3,58343
1003,Alabama,Baldwin,17.3,30.0,25.0,10.7,73,69.5,14.6,59871
1005,Alabama,Barbour,22.1,41.8,35.1,13.3,41,50.7,39.6,35972
1007,Alabama,Bibb,20.4,38.3,31.5,11.2,31,52.8,25.9,45795
1009,Alabama,Blount,19.6,34.8,30.3,12.3,23,55.5,20.5,52902
1011,Alabama,Bullock,22.7,43.4,39.3,12.7,31,49.6,46.7,33534
1013,Alabama,Butler,22.3,42.0,34.1,13.8,44,51.8,37.3,39277
1015,Alabama,Calhoun,19.5,36.9,31.1,11.4,63,58.3,26.4,49000
1017,Alabama,Chambers,20.2,40.1,32.3,12.1,38,51.5,28.9,44221
1019,Alabama,Cherokee,18.7,35.4,29.5,11.6,41,56.0,21.1,48972
1021,Alabama,Chilton,20.1,37.2,30.7,11.9,26,52.5,24.3,50438
1023,Alabama,Choctaw,21.9,41.2,35.8,13.1,47,48.0,35.4,36453
1025,Alabama,Clarke,21.1,40.5,34.2,13.5,35,50.1,33.1,39932
1027,Alabama,Clay,19.8,36.1,31.8,11.8,30,51.9,25.6,45000
1029,Alabama,Cleburne,19.2,35.8,29.9,12.0,24,53.1,22.6,47896
1031,Alabama,Coffee,18.5,34.9,28.2,12.5,61,65.1,18.8,55000
1033,Alabama,Colbert,18.9,36.2,30.1,11.3,77,61.4,20.2,51093
2020,Alaska,Anchorage,17.2,28.7,21.1,13.7,143,71.0,12.2,82823
2050,Alaska,Bethel,30.6,36.5,35.3,26.7,83,50.8,33.1,56344
2060,Alaska,Bristol Bay,16.5,30.2,24.5,18.1,155,65.0,15.1,77000
2068,Alaska,Denali,15.8,29.5,22.8,16.9,121,70.2,13.5,75000
2070,Alaska,Dillingham,28.9,34.8,33.2,25.3,95,55.1,30.2,60000
2090,Alaska,Fairbanks North Star,17.8,30.5,23.1,14.2,111,68.9,11.5,72000
4001,Arizona,Apache,25.1,31.7,29.9,22.3,41,49.2,41.0,35808
4003,Arizona,Cochise,15.1,28.8,24.1,12.4,54,65.3,19.9,51439
4005,Arizona,Coconino,16.8,24.3,19.8,14.9,89,69.8,22.6,62000
4007,Arizona,Gila,19.2,33.1,30.9,12.9,65,58.0,24.8,48000
4011,Arizona,Greenlee,16.2,30.8,26.3,10.1,51,60.1,18.5,57000
4012,Arizona,La Paz,19.9,34.2,32.5,17.5,43,50.2,29.8,41000
4013,Arizona,Maricopa,14.5,29.0,22.1,11.8,78,66.8,17.5,69000
4015,Arizona,Mohave,20.1,32.8,30.2,16.1,49,52.9,23.1,47000
4019,Arizona,Pima,15.5,28.5,22.9,11.5,99,69.1,20.3,56000
4021,Arizona,Pinal,17.1,31.5,26.8,13.2,55,61.2,18.9,62000
6001,California,Alameda,9.8,22.5,19.5,6.5,98,71.8,12.1,99000
6003,California,Alpine,14.1,26.8,21.2,10.2,111,68.5,16.8,68000
6005,California,Amador,15.2,28.9,24.8,7.9,65,63.1,14.2,65000
6037,California,Los Angeles,12.1,23.3,23.4,12.3,80,61.9,21.0,68044
6039,California,Madera,16.5,31.2,27.8,11.8,45,50.1,26.8,55000
6041,California,Marin,9.2,20.1,16.8,5.1,201,80.5,9.2,125000
6043,California,Mariposa,15.8,29.2,25.3,9.8,58,62.8,16.5,58000
6059,California,Orange,10.5,23.8,21.5,8.1,88,69.9,14.8,94000
6061,California,Placer,11.2,25.1,18.9,5.9,85,75.1,9.8,92000
6063,California,Plumas,16.1,29.8,25.8,9.5,75,61.9,15.9,57000
6065,California,Riverside,13.2,28.1,25.1,10.5,60,59.8,18.2,70000
6067,California,Sacramento,14.1,29.5,24.1,7.9,81,65.5,18.5,71000
8001,Colorado,Adams,16.5,31.1,25.1,12.8,58,61.8,15.2,72000
8003,Colorado,Alamosa,17.2,28.5,23.8,14.5,75,60.2,28.9,45000
8005,Colorado,Arapahoe,13.1,25.2,20.1,9.1,91,73.5,13.1,79000
8007,Colorado,Archuleta,14.8,26.5,21.5,15.1,85,68.9,14.8,63000
8009,Colorado,Baca,17.5,29.8,28.5,11.2,55,61.5,20.1,46000
8011,Colorado,Bent,18.1,32.1,29.8,10.8,48,55.8,25.8,42000
8013,Colorado,Boulder,10.1,20.5,15.1,6.5,135,80.1,12.5,88000
12057,Florida,Hillsborough,16.2,30.1,24.5,14.1,85,65.5,20.1,60000
12086,Florida,Miami-Dade,11.5,24.8,22.1,18.5,91,65.8,21.5,57000
13121,Georgia,Fulton,12.8,25.1,21.8,12.5,121,75.8,18.1,75000
17031,Illinois,Cook,14.5,28.9,23.5,10.1,105,68.9,18.5,67000
36001,New York,Albany,13.5,28.1,22.5,4.5,155,75.2,15.1,72000
36003,New York,Allegany,17.8,32.5,28.1,7.8,45,60.1,20.5,49000
36005,New York,Bronx,18.1,32.8,30.5,9.5,111,58.8,35.5,41000
36007,New York,Broome,17.5,33.1,27.5,6.1,101,68.9,21.1,52000
36047,New York,Kings,13.8,25.8,25.1,8.1,121,65.8,25.5,66000
36059,New York,Nassau,10.9,24.5,21.8,4.8,145,75.8,8.5,120000
36061,New York,New York,9.9,21.8,21.8,7.0,392,81.4,17.3,86553
36081,New York,Queens,12.5,24.1,23.5,9.8,95,62.5,18.1,73000
36085,New York,Richmond,14.8,30.5,26.1,5.8,105,68.5,16.5,85000
48001,Texas,Anderson,18.8,35.5,31.5,18.5,51,50.1,25.5,48000
48003,Texas,Andrews,17.1,32.1,28.1,19.8,48,55.8,16.8,75000
48005,Texas,Angelina,19.5,36.8,32.8,20.1,55,54.5,26.8,50000
48007,Texas,Aransas,19.2,34.1,29.8,21.5,65,60.2,21.5,52000
48113,Texas,Dallas,15.2,30.1,24.5,24.1,85,62.8,22.5,60000
48201,Texas,Harris,14.8,32.5,25.8,22.5,81,61.5,20.1,63000
"""

# Feature / target contract (used by both main() and tests).
FEATURES = [
    'LocationAbbr', 'LocationDesc', '% Smokers', '% Adults with Obesity', '% Physically Inactive',
    '% Uninsured', 'Primary Care Physicians Rate', '% With Some College', '% Children in Poverty', 'Median Household Income'
]
TARGET = 'Gender_Gap'


# --- 3. Data Loading and Feature Engineering ---
def load_data_from_string(filename, data_string):
    """Robustly parses a CSV string using pandas' built-in CSV parser."""
    try:
        df = pd.read_csv(io.StringIO(data_string.strip()))
        if df.empty:
            print(f" -> WARNING: No data loaded for {filename}.")
        return df
    except Exception as e:
        print(f" -> ERROR: Could not parse {filename}. Error: {e}")
        return pd.DataFrame()


def clean_fips(df, column_name):
    """Cleans and standardizes a FIPS code column for reliable merging."""
    # Convert to string, remove '.0' suffixes, and pad with leading zeros to 5 digits.
    df = df.copy()
    df[column_name] = df[column_name].astype(str).str.split('.').str[0].str.zfill(5)
    return df


def compute_gender_gap(df_mortality_gender):
    """Pivot Male/Female mortality per county and compute Gender_Gap = Male - Female.

    Returns a DataFrame with columns ['LocationID', 'Gender_Gap'].
    """
    df = df_mortality_gender.copy()
    df['Data_Value'] = pd.to_numeric(df['Data_Value'], errors='coerce')
    df = df.dropna(subset=['Data_Value', 'LocationID', 'Stratification2'])
    df_pivot = df.pivot_table(
        index='LocationID', columns='Stratification2', values='Data_Value'
    ).reset_index()
    if 'Male' not in df_pivot.columns or 'Female' not in df_pivot.columns:
        raise ValueError("No counties found with mortality data for both Males and Females.")
    df_pivot = df_pivot.dropna(subset=['Male', 'Female'])
    if df_pivot.empty:
        raise ValueError("No counties found with mortality data for both Males and Females.")
    df_pivot['Gender_Gap'] = df_pivot['Male'] - df_pivot['Female']
    return df_pivot[['LocationID', 'Gender_Gap']].dropna()


def build_merged_dataset(mortality_csv=None, health_csv=None):
    """Load embedded (or provided) CSV strings, compute the gender gap, and merge
    with county health rankings into a single modelling DataFrame.

    Raises ValueError if either dataset is empty or no merge rows survive.
    """
    if mortality_csv is None:
        mortality_csv = EMBEDDED_DATA['Heart_Disease_Mortality_Gender_Data_ALL.csv']
    if health_csv is None:
        health_csv = EMBEDDED_DATA['county_health_rankings_2023_comprehensive.csv']

    df_mortality_gender = load_data_from_string('mortality_gender', mortality_csv)
    df_health_rankings = load_data_from_string('health_rankings', health_csv)
    if df_mortality_gender.empty or df_health_rankings.empty:
        raise ValueError("Could not load one or more essential datasets.")

    df_gender_gap = compute_gender_gap(df_mortality_gender)

    df_gender_gap = clean_fips(df_gender_gap, 'LocationID')
    df_health_rankings = clean_fips(df_health_rankings, 'FIPS').rename(columns={'FIPS': 'LocationID'})
    numeric_cols_hr = [c for c in df_health_rankings.columns if c not in ['State', 'County', 'LocationID']]
    for col in numeric_cols_hr:
        df_health_rankings[col] = pd.to_numeric(df_health_rankings[col], errors='coerce')

    df_merged = pd.merge(df_gender_gap, df_health_rankings, on='LocationID', how='inner')

    df_location_info = df_mortality_gender[['LocationID', 'LocationAbbr', 'LocationDesc']].drop_duplicates()
    df_location_info = clean_fips(df_location_info, 'LocationID')
    df_merged = pd.merge(df_merged, df_location_info, on='LocationID', how='inner')

    df_merged = df_merged.dropna().drop_duplicates()
    if df_merged.empty:
        raise ValueError("No rows survived the merge of gender-gap and health-rankings data.")
    return df_merged


def prepare_features(df_merged):
    """Encode object columns to category codes and return (X, y, final_features).

    Only columns from FEATURES that exist in df_merged are used; the target
    (Gender_Gap) is never included among the features (no target leakage).
    """
    df_encoded = df_merged.copy()
    final_features = []
    for col in FEATURES:
        if col in df_encoded.columns:
            # Encode any non-numeric (object/string/category) column to integer
            # category codes. Using is_numeric_dtype rather than `== 'object'`
            # so this works across pandas versions where string columns may carry
            # a dedicated 'str'/StringDtype rather than the legacy 'object' dtype.
            if not pd.api.types.is_numeric_dtype(df_encoded[col]):
                df_encoded[col] = df_encoded[col].astype('category').cat.codes
            final_features.append(col)
    X = df_encoded[final_features]
    y = df_encoded[TARGET]
    return X, y, final_features


def train_model(X, y, test_size=0.2, random_state=42, n_estimators=100):
    """Split BEFORE fitting (no leakage), train a RandomForestRegressor, and
    evaluate R^2 on the held-out test set.

    Returns a dict with the model, the four split frames, and r2.
    """
    if X.empty or y.empty:
        raise ValueError("Empty feature matrix or target vector.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    return {
        'model': model,
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'r2': r2,
    }


def describe_drivers(model, X_test):
    """Return a human-readable sentence describing the top drivers of the gap.

    Uses SHAP mean values when available, otherwise random-forest feature
    importances.
    """
    if shap is not None:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        mean_shap = pd.Series(np.mean(shap_values, axis=0), index=X_test.columns)
        wider = ", ".join(mean_shap.sort_values(ascending=False).head(3).index)
        narrower = ", ".join(mean_shap.sort_values(ascending=True).head(3).index)
        return (
            f"The strongest SHAP predictors of a wider gender gap were {wider}. "
            f"Features most associated with narrower gaps were {narrower}."
        )
    importances = pd.Series(model.feature_importances_, index=X_test.columns).sort_values(ascending=False)
    return (
        "SHAP directionality was not available because the shap package was not installed. "
        f"The highest random-forest feature-importance variables were {', '.join(importances.head(3).index)}."
    )


def make_publication_text(n_counties, r2, driver_sentence):
    """Render the publication abstract from computed results."""
    return f"""
# Publication Title: Explaining County-Level Variation in the Gender Gap for Cardiovascular Disease Mortality

## Abstract

**Objectives.** To identify the key socioeconomic and behavioral factors associated with the county-level gender gap (male minus female mortality rate) in cardiovascular disease (CVD).

**Methods.** We calculated the gender gap in age-adjusted CVD mortality using 2019 CDC data. This was merged with county-level metrics from the 2023 County Health Rankings. The final dataset included [N={n_counties}] US counties. A Random Forest model, interpreted with SHAP, was trained to predict the magnitude of the gender gap.

**Results.** The model explained a portion of the variance in the gender gap (R^2 = {r2:.3f}). {driver_sentence}

**Conclusions.** The gender gap in CVD mortality is not uniform and varies based on a county's socioeconomic and behavioral context. Factors that disproportionately harm men, or protect women, can widen this health disparity.

**Public Health Implications.** Interventions aimed at reducing the gender gap in CVD should focus on the specific community-level factors that exacerbate it. Policies that target smoking cessation, for example, may not only lower overall mortality but also reduce this key disparity.
"""


def run_analysis():
    """Run the full analysis pipeline and return a results dict (no plotting/I/O)."""
    df_merged = build_merged_dataset()
    X, y, final_features = prepare_features(df_merged)
    trained = train_model(X, y)
    driver_sentence = describe_drivers(trained['model'], trained['X_test'])
    publication_text = make_publication_text(len(df_merged), trained['r2'], driver_sentence)
    return {
        'df_merged': df_merged,
        'X': X, 'y': y, 'final_features': final_features,
        'driver_sentence': driver_sentence,
        'publication_text': publication_text,
        **trained,
    }


def save_distribution_plot(df_merged, path='1_gender_gap_distribution.png'):
    """Render and save the gender-gap distribution histogram."""
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(df_merged['Gender_Gap'], bins=15, kde=True)
    plt.axvline(df_merged['Gender_Gap'].mean(), color='r', linestyle='--')
    plt.title('Distribution of Gender Gap in CVD Mortality', fontsize=16)
    plt.xlabel('Mortality Rate Difference (Male - Female)')
    plt.ylabel('Number of Counties')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def save_shap_plots(model, X_test):
    """Render and save SHAP summary plots when SHAP is available. No-op otherwise."""
    if shap is None:
        return []
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    saved = []

    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title('SHAP Summary: Drivers of the Gender Gap', fontsize=14)
    plt.tight_layout()
    plt.savefig('2_gender_gap_shap_bar.png')
    plt.close()
    saved.append('2_gender_gap_shap_bar.png')

    shap.summary_plot(shap_values, X_test, show=False)
    plt.title('Impact of Features on CVD Mortality Gender Gap')
    plt.xlabel("SHAP value (Impact on Gender Gap)")
    plt.tight_layout()
    plt.savefig('3_gender_gap_shap_beeswarm.png')
    plt.close()
    saved.append('3_gender_gap_shap_beeswarm.png')
    return saved


def main():
    print("--- Loading and Engineering Data for Gender Gap Analysis ---")
    try:
        df_merged = build_merged_dataset()
    except ValueError as e:
        print(f"\n--- FATAL ERROR: {e} ---")
        sys.exit(1)
    print("Successfully calculated gender gap.")
    print(f"Shape of final merged dataframe: {df_merged.shape}")

    print("\n--- Performing Exploratory Data Analysis ---")
    print(f"Saved: {save_distribution_plot(df_merged)}")

    print("\n--- Preparing Data for Modeling ---")
    X, y, final_features = prepare_features(df_merged)
    if X.empty or y.empty:
        print("\n--- FATAL ERROR: Empty feature matrix or target. ---")
        sys.exit(1)

    print("\n--- Building and Training the Model ---")
    trained = train_model(X, y)
    print(f"Training data shape: {trained['X_train'].shape}")
    print("Model training complete.")
    print(f"Model R-squared: {trained['r2']:.3f}")

    if shap is not None:
        print("\n--- Generating Advanced Model Interpretations with SHAP ---")
        for p in save_shap_plots(trained['model'], trained['X_test']):
            print(f"Saved: {p}")
    else:
        print("\nSHAP visualization skipped (shap not installed).")

    driver_sentence = describe_drivers(trained['model'], trained['X_test'])
    print("\n\n--- Generating Publication Text ---")
    print(make_publication_text(len(df_merged), trained['r2'], driver_sentence))
    print("\n--- Analysis Complete ---")


if __name__ == '__main__':
    main()
