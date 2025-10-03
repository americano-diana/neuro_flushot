import pandas as pd

def impute_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply predefined imputation rules to handle missing values.
    Rules:
    - Employment-related (industry, occupation): fill with 'Missing'
    - Health insurance: fill with 'Missing'
    - Socio-economic categorical: fill with 'Missing'
    - Doctor recommendations (binary): fill with 0
    - Opinion/behavioral (ordinal): median imputation
    - Household counts: median imputation
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

    # 4. Doctor recommendations (binary, assume missing = 0)
    for col in ["doctor_recc_h1n1", "doctor_recc_seasonal"]:
        if col in df_imputed.columns:
            df_imputed[col] = df_imputed[col].fillna(0)

    # 5. Opinion/behavioral features (median imputation)
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
        if col in df_imputed.columns:
            median_val = df_imputed[col].median()
            df_imputed[col] = df_imputed[col].fillna(median_val)

    # 6. Household counts (median imputation)
    for col in ["household_adults", "household_children"]:
        if col in df_imputed.columns:
            median_val = df_imputed[col].median()
            df_imputed[col] = df_imputed[col].fillna(median_val)

    return df_imputed
