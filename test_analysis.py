"""Tests for gender_gap_model.py.

These validate the importable core functions on both the embedded dataset and a
small synthetic dataset with known, planted properties. No plotting or file I/O
is exercised here; matplotlib uses the Agg backend on import.
"""
import io

import numpy as np
import pandas as pd
import pytest

import gender_gap_model as g


# --- Synthetic fixtures with planted signal -------------------------------

def _synthetic_mortality_csv(n=40, seed=0):
    """Mortality CSV where Male - Female gap is driven by a planted feature.

    Each county gets a Male and Female row. The gap (Male - Female) is set to be
    a strong linear function of the 'planted' health metric (% Smokers), so a
    correctly-wired model must assign that feature nonzero importance.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        fips = 1000 + i  # 4-digit-ish FIPS, cleaned/padded downstream
        smokers = 10.0 + i  # monotonic planted signal, mirrored in health CSV
        female = 200.0 + rng.normal(0, 5)
        gap = 2.0 * smokers + rng.normal(0, 1)  # gap driven by smokers
        male = female + gap
        rows.append((2019, "ST", f"County {i}", "NVSS", "Heart Disease Mortality",
                     round(male, 2), "per 100,000 population", "Age-adjusted", "Male", fips))
        rows.append((2019, "ST", f"County {i}", "NVSS", "Heart Disease Mortality",
                     round(female, 2), "per 100,000 population", "Age-adjusted", "Female", fips))
    header = ("Year,LocationAbbr,LocationDesc,DataSource,Topic,Data_Value,"
              "Data_Value_Unit,Data_Value_Type,Stratification2,LocationID")
    body = "\n".join(
        f'{y},{ab},{desc},{src},{topic},{val},"{unit}",{vt},{strat},{loc}'
        for (y, ab, desc, src, topic, val, unit, vt, strat, loc) in rows
    )
    return header + "\n" + body + "\n", n


def _synthetic_health_csv(n=40):
    rows = []
    for i in range(n):
        fips = 1000 + i
        smokers = 10.0 + i  # matches the planted gap driver
        rows.append((fips, "State", f"County {i}", smokers, 30.0, 25.0, 10.0,
                     50, 60.0, 20.0, 55000))
    header = ('FIPS,State,County,"% Smokers","% Adults with Obesity",'
              '"% Physically Inactive","% Uninsured","Primary Care Physicians Rate",'
              '"% With Some College","% Children in Poverty","Median Household Income"')
    body = "\n".join(",".join(str(c) for c in r) for r in rows)
    return header + "\n" + body + "\n"


@pytest.fixture
def synthetic_merged():
    mort_csv, n = _synthetic_mortality_csv()
    health_csv = _synthetic_health_csv()
    df = g.build_merged_dataset(mortality_csv=mort_csv, health_csv=health_csv)
    return df, n


# --- Embedded-data sanity --------------------------------------------------

def test_embedded_build_merged_nonempty():
    df = g.build_merged_dataset()
    assert not df.empty
    assert 'Gender_Gap' in df.columns
    assert 'LocationID' in df.columns


def test_gender_gap_is_male_minus_female():
    csv = (
        "Year,LocationAbbr,LocationDesc,DataSource,Topic,Data_Value,"
        "Data_Value_Unit,Data_Value_Type,Stratification2,LocationID\n"
        '2019,ST,A County,NVSS,HD,500,"u",Age-adjusted,Male,1001\n'
        '2019,ST,A County,NVSS,HD,300,"u",Age-adjusted,Female,1001\n'
    )
    df_mort = g.load_data_from_string("m", csv)
    gap = g.compute_gender_gap(df_mort)
    assert gap.loc[gap['LocationID'] == 1001, 'Gender_Gap'].iloc[0] == pytest.approx(200.0)


def test_compute_gender_gap_drops_single_sex_counties():
    csv = (
        "Year,LocationAbbr,LocationDesc,DataSource,Topic,Data_Value,"
        "Data_Value_Unit,Data_Value_Type,Stratification2,LocationID\n"
        '2019,ST,A County,NVSS,HD,500,"u",Age-adjusted,Male,1001\n'  # only Male
    )
    df_mort = g.load_data_from_string("m", csv)
    with pytest.raises(ValueError):
        g.compute_gender_gap(df_mort)


def test_clean_fips_pads_to_five_digits():
    df = pd.DataFrame({"FIPS": [1001, 6037.0, "36061"]})
    out = g.clean_fips(df, "FIPS")
    assert list(out["FIPS"]) == ["01001", "06037", "36061"]


# --- Feature preparation: no target leakage --------------------------------

def test_prepare_features_excludes_target(synthetic_merged):
    df, _ = synthetic_merged
    X, y, final_features = g.prepare_features(df)
    assert g.TARGET not in final_features
    assert g.TARGET not in X.columns
    assert len(X) == len(y)


def test_prepare_features_all_numeric_after_encoding(synthetic_merged):
    df, _ = synthetic_merged
    X, _, _ = g.prepare_features(df)
    # Every feature column must be numeric so the model can consume it
    # (regression test for the pandas-3 'str' dtype encoding bug).
    for col in X.columns:
        assert pd.api.types.is_numeric_dtype(X[col]), f"{col} not numeric"


# --- Training: split before fit, metric on held-out data -------------------

def test_train_model_split_no_leakage(synthetic_merged):
    df, _ = synthetic_merged
    X, y, _ = g.prepare_features(df)
    res = g.train_model(X, y, test_size=0.25, random_state=42)
    n = len(X)
    # Train and test partitions are disjoint and cover all rows.
    assert len(res['X_train']) + len(res['X_test']) == n
    train_idx = set(res['X_train'].index)
    test_idx = set(res['X_test'].index)
    assert train_idx.isdisjoint(test_idx)


def test_train_model_predict_shape(synthetic_merged):
    df, _ = synthetic_merged
    X, y, _ = g.prepare_features(df)
    res = g.train_model(X, y)
    preds = res['model'].predict(res['X_test'])
    assert preds.shape[0] == res['X_test'].shape[0]


def test_train_model_r2_finite_and_bounded(synthetic_merged):
    df, _ = synthetic_merged
    X, y, _ = g.prepare_features(df)
    res = g.train_model(X, y)
    # R^2 can in principle be negative, but is bounded above by 1; with a strong
    # planted signal we expect a clearly positive value.
    assert np.isfinite(res['r2'])
    assert res['r2'] <= 1.0
    assert res['r2'] > 0.5


def test_planted_feature_has_nonzero_importance(synthetic_merged):
    df, _ = synthetic_merged
    X, y, _ = g.prepare_features(df)
    res = g.train_model(X, y)
    importances = pd.Series(res['model'].feature_importances_, index=X.columns)
    # '% Smokers' is the planted driver of the gap and must dominate.
    assert importances['% Smokers'] > 0
    assert importances.idxmax() == '% Smokers'


def test_train_model_reproducible(synthetic_merged):
    df, _ = synthetic_merged
    X, y, _ = g.prepare_features(df)
    r2_a = g.train_model(X, y, random_state=42)['r2']
    r2_b = g.train_model(X, y, random_state=42)['r2']
    assert r2_a == pytest.approx(r2_b)


def test_train_model_empty_raises():
    empty = pd.DataFrame()
    with pytest.raises(ValueError):
        g.train_model(empty, pd.Series(dtype=float))


# --- End-to-end pipeline ---------------------------------------------------

def test_run_analysis_embedded_endtoend():
    res = g.run_analysis()
    assert np.isfinite(res['r2'])
    assert res['r2'] <= 1.0
    assert 'Publication Title' in res['publication_text']
    assert isinstance(res['driver_sentence'], str) and res['driver_sentence']


def test_describe_drivers_returns_sentence(synthetic_merged):
    df, _ = synthetic_merged
    X, y, _ = g.prepare_features(df)
    res = g.train_model(X, y)
    sentence = g.describe_drivers(res['model'], res['X_test'])
    assert isinstance(sentence, str) and len(sentence) > 0


def test_make_publication_text_includes_n_and_r2():
    txt = g.make_publication_text(15, 0.385, "drivers here")
    assert "N=15" in txt
    assert "0.385" in txt
    assert "drivers here" in txt
