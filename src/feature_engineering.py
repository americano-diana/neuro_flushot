"""
Feature engineering for Flu Shot Learning (Sprint 2)
----------------------------------------------------
Adds composite, behavioral, and socioeconomic features
on top of the encoded datasets produced by preprocessing.py.
Includes validation utilities for CI or notebook testing.
"""

import pandas as pd
import numpy as np


# =========================================================
# FEATURE BUILDERS
# =========================================================
def add_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate and normalize behavioral and opinion-based signals."""
    df = df.copy()

    behavior_cols = [
        "behavioral_antiviral_meds", "behavioral_avoidance", "behavioral_face_mask",
        "behavioral_wash_hands", "behavioral_large_gatherings",
        "behavioral_outside_home", "behavioral_touch_face"
    ]
    existing = [c for c in behavior_cols if c in df.columns]
    df["PBI"] = df[existing].sum(axis=1) / len(existing)

    df["PVAS_h1n1"] = (
        df["opinion_h1n1_vacc_effective"] +
        df["opinion_h1n1_risk"] -
        df["opinion_h1n1_sick_from_vacc"]
    ) / 3

    df["PVAS_seas"] = (
        df["opinion_seas_vacc_effective"] +
        df["opinion_seas_risk"] -
        df["opinion_seas_sick_from_vacc"]
    ) / 3

    df["HKI"] = (df["h1n1_concern"] + df["h1n1_knowledge"]) / 2

    df["doctor_any"] = ((df["doctor_recc_h1n1"] == 1) | (df["doctor_recc_seasonal"] == 1)).astype(int)

    return df


def add_socioeconomic_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple 0/1 flags derived from categorical socioeconomic fields."""
    df = df.copy()

    def safe_flag(col, values):
        if col not in df.columns:
            return np.zeros(len(df))
        return df[col].isin(values).astype(int)

    df["below_poverty"] = safe_flag("income_poverty", ["Below Poverty"])
    df["low_edu"] = safe_flag("education", ["< 12 Years", "12 Years"])
    df["housing_insecure"] = safe_flag("rent_or_own", ["Rent", "Missing"])
    df["no_job"] = safe_flag("employment_status", ["Unemployed", "Missing"])
    df["no_insurance"] = safe_flag("health_insurance", ["Missing"])
    df["is_married"] = safe_flag("marital_status", ["Married"])
    df["is_senior"] = safe_flag("age_group", ["65+ Years"])

    return df


def add_health_risk_index(df: pd.DataFrame) -> pd.DataFrame:
    """Compute composite Health Risk Index (HRI) using safe numeric and flag variables."""
    df = df.copy()
    for col in ["chronic_med_condition", "child_under_6_months", "health_worker"]:
        if col not in df.columns:
            df[col] = 0
    for flag in ["is_senior", "below_poverty"]:
        if flag not in df.columns:
            df[flag] = 0

    df["HRI"] = (
        df["chronic_med_condition"].fillna(0)
        + df["child_under_6_months"].fillna(0)
        + df["health_worker"].fillna(0)
        + df["is_senior"].astype(int)
        + df["below_poverty"].astype(int)
    ).clip(upper=4)

    return df


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Add selected domain-driven interactions."""
    df = df.copy()

    def safe_product(a, b):
        return df[a].fillna(0) * df[b].fillna(0) if a in df and b in df else 0

    df["PVASxPBI_h1n1"] = safe_product("PVAS_h1n1", "PBI")
    df["PVASxDoctor_seas"] = safe_product("PVAS_seas", "doctor_any")
    df["PBIxNoInsur"] = safe_product("PBI", "no_insurance")
    df["SeniorxChronic"] = safe_product("is_senior", "chronic_med_condition")
    df["PovertyxRent"] = safe_product("below_poverty", "rent_or_own_Rent") if "rent_or_own_Rent" in df else 0

    if all(c in df for c in ["employment_industry_te_h1n1_vaccine", "employment_industry_te_seasonal_vaccine"]):
        df["industry_occ_te_mean"] = df[
            ["employment_industry_te_h1n1_vaccine", "employment_industry_te_seasonal_vaccine"]
        ].mean(axis=1)

    return df


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run all feature engineering steps sequentially."""
    df = (
        df.pipe(add_behavioral_features)
          .pipe(add_socioeconomic_flags)
          .pipe(add_health_risk_index)
          .pipe(add_interactions)
    )
    return df


# =========================================================
# VALIDATION UTILITIES
# =========================================================
def validate_engineered_features(df: pd.DataFrame) -> None:
    """
    Validate integrity of engineered dataset.
    Raises AssertionError if any test fails.
    """
    expected_features = [
        "PBI", "PVAS_h1n1", "PVAS_seas", "HKI",
        "doctor_any", "below_poverty", "low_edu",
        "housing_insecure", "no_job", "no_insurance", "is_married",
        "is_senior", "HRI", "PVASxPBI_h1n1", "PVASxDoctor_seas",
        "PBIxNoInsur", "SeniorxChronic"
    ]

    # --- presence check ---
    missing = [f for f in expected_features if f not in df.columns]
    assert not missing, f"Missing engineered features: {missing}"

    # --- type & NA checks ---
    num_cols = df[expected_features].select_dtypes(include=[np.number])
    assert len(num_cols.columns) == len(expected_features), "Some engineered features are not numeric"
    assert df[expected_features].isna().sum().sum() == 0, "NaN values detected in engineered features"

    # --- reasonable value range checks ---
    assert df["PBI"].between(0, 1).all(), "PBI out of expected range [0,1]"
    assert df["HRI"].between(0, 4).all(), "HRI out of expected range [0,4]"
    print("Feature validation passed: all checks OK.")