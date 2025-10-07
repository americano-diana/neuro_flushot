import pandas as pd

def fit_imputers(df_train: pd.DataFrame) -> dict:
    """
    Learn median values for numeric features from the training set.
    Returns a dictionary of {column: median_value}.
    """
    imputers = {}
    
    # Opinion/behavioral features
    opinion_behavior_cols = [
        "h1n1_concern", "h1n1_knowledge",
        "behavioral_antiviral_meds", "behavioral_avoidance", "behavioral_face_mask",
        "behavioral_wash_hands", "behavioral_large_gatherings",
        "behavioral_outside_home", "behavioral_touch_face",
        "chronic_med_condition", "child_under_6_months", "health_worker",
        "opinion_h1n1_vacc_effective", "opinion_h1n1_risk", "opinion_h1n1_sick_from_vacc",
        "opinion_seas_vacc_effective", "opinion_seas_risk", "opinion_seas_sick_from_vacc"
    ]
    for col in opinion_behavior_cols:
        if col in df_train.columns:
            imputers[col] = df_train[col].median()
    
    # Household counts
    for col in ["household_adults", "household_children"]:
        if col in df_train.columns:
            imputers[col] = df_train[col].median()
    
    return imputers


def impute_dataset(df: pd.DataFrame, imputers: dict) -> pd.DataFrame:
    """
    Apply imputation using fixed rules.
    For numeric features, use provided train-based imputers (median values).
    """
    df_imputed = df.copy()

    # 1. Employment-related
    for col in ["employment_industry", "employment_occupation"]:
        if col in df_imputed.columns:
            df_imputed[col] = df_imputed[col].fillna("Missing")

    # 2. Health insurance
    if "health_insurance" in df_imputed.columns:
        df_imputed["health_insurance"] = df_imputed["health_insurance"].fillna("Missing")

    # 3. Socio-economic categorical
    cat_cols = ["income_poverty", "education", "marital_status", "employment_status", "rent_or_own"]
    for col in cat_cols:
        if col in df_imputed.columns:
            df_imputed[col] = df_imputed[col].fillna("Missing")

    # 4. Doctor recommendations
    for col in ["doctor_recc_h1n1", "doctor_recc_seasonal"]:
        if col in df_imputed.columns:
            df_imputed[col] = df_imputed[col].fillna(0)

    # 5. Opinion/behavioral + household (use train medians)
    for col, median_val in imputers.items():
        if col in df_imputed.columns:
            df_imputed[col] = df_imputed[col].fillna(median_val)

    return df_imputed

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply categorical encodings:
    - Ordinal encodings for age_group, education, income_poverty
    - Label encodings for small/binary categoricals
    - One-hot encoding for medium-cardinality categoricals
    - Frequency encoding for high-cardinality categoricals
    """
    from src.mappings import (
        AGE_MAPPING, EDUCATION_MAPPING, INCOME_MAPPING,
        SEX_MAPPING, MARITAL_STATUS_MAPPING, RENT_OWN_MAPPING,
        HEALTH_INSURANCE_MAPPING
    )
    
    df_encoded = df.copy()

    # --- Ordinal mappings ---
    if "age_group" in df_encoded.columns:
        df_encoded["age_group"] = df_encoded["age_group"].map(AGE_MAPPING)

    if "education" in df_encoded.columns:
        df_encoded["education"] = df_encoded["education"].map(EDUCATION_MAPPING)

    if "income_poverty" in df_encoded.columns:
        df_encoded["income_poverty"] = df_encoded["income_poverty"].map(INCOME_MAPPING)

    # --- Small nominal/binary mappings ---
    if "sex" in df_encoded.columns:
        df_encoded["sex"] = df_encoded["sex"].map(SEX_MAPPING)

    if "marital_status" in df_encoded.columns:
        df_encoded["marital_status"] = df_encoded["marital_status"].map(MARITAL_STATUS_MAPPING)

    if "rent_or_own" in df_encoded.columns:
        df_encoded["rent_or_own"] = df_encoded["rent_or_own"].map(RENT_OWN_MAPPING)

    if "health_insurance" in df_encoded.columns:
        df_encoded["health_insurance"] = df_encoded["health_insurance"].astype(str).map(HEALTH_INSURANCE_MAPPING)

    # --- One-hot encoding ---
    onehot_cols = ["race", "employment_status", "census_msa", "hhs_geo_region"]
    df_encoded = pd.get_dummies(df_encoded, 
                                columns=[col for col in onehot_cols if col in df_encoded.columns],
                                drop_first=True)

    # --- Frequency encoding for high-cardinality ---
    for col in ["employment_industry", "employment_occupation"]:
        if col in df_encoded.columns:
            freqs = df_encoded[col].value_counts(normalize=True)
            df_encoded[col] = df_encoded[col].map(freqs)

    return df_encoded