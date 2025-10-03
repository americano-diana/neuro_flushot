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