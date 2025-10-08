import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from src.mappings import AGE_MAPPING, EDUCATION_MAPPING, INCOME_MAPPING


# ----------------------------
# 1. IMPUTATION
# ----------------------------
def fit_imputers(df_train: pd.DataFrame) -> dict:
    """Learn median values for numeric features from the training set."""
    imputers = {}

    opinion_behavior_cols = [
        "h1n1_concern", "h1n1_knowledge",
        "behavioral_antiviral_meds", "behavioral_avoidance", "behavioral_face_mask",
        "behavioral_wash_hands", "behavioral_large_gatherings",
        "behavioral_outside_home", "behavioral_touch_face",
        "chronic_med_condition", "child_under_6_months", "health_worker",
        "opinion_h1n1_vacc_effective", "opinion_h1n1_risk", "opinion_h1n1_sick_from_vacc",
        "opinion_seas_vacc_effective", "opinion_seas_risk", "opinion_seas_sick_from_vacc"
    ]
    for col in opinion_behavior_cols + ["household_adults", "household_children"]:
        if col in df_train.columns:
            imputers[col] = df_train[col].median()

    return imputers


def impute_dataset(df: pd.DataFrame, imputers: dict) -> pd.DataFrame:
    """Apply imputation using fixed rules and train-based medians."""
    df_imputed = df.copy()

    # Employment-related
    for col in ["employment_industry", "employment_occupation"]:
        if col in df_imputed.columns:
            df_imputed[col] = df_imputed[col].fillna("Missing")

    # Health insurance
    if "health_insurance" in df_imputed.columns:
        df_imputed["health_insurance"] = df_imputed["health_insurance"].fillna("Missing")

    # Socio-economic categorical
    cat_cols = ["income_poverty", "education", "marital_status", "employment_status", "rent_or_own"]
    for col in cat_cols:
        if col in df_imputed.columns:
            df_imputed[col] = df_imputed[col].fillna("Missing")

    # Doctor recommendations (binary → missing = 0)
    for col in ["doctor_recc_h1n1", "doctor_recc_seasonal"]:
        if col in df_imputed.columns:
            df_imputed[col] = df_imputed[col].fillna(0)

    # Opinion/behavioral + household medians
    for col, median_val in imputers.items():
        if col in df_imputed.columns:
            df_imputed[col] = df_imputed[col].fillna(median_val)

    return df_imputed


# ----------------------------
# 2. TARGET ENCODING (SAFE)
# ----------------------------
def target_encode(train_df, test_df, col, target, n_splits=5):
    """KFold mean target encoding to avoid leakage."""
    global_mean = train_df[target].mean()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_encoded = np.zeros(train_df.shape[0])

    for train_idx, val_idx in kf.split(train_df):
        X_train, X_val = train_df.iloc[train_idx], train_df.iloc[val_idx]
        means = X_train.groupby(col)[target].mean()
        train_encoded[val_idx] = X_val[col].map(means)

    train_encoded = np.where(np.isnan(train_encoded), global_mean, train_encoded)
    means_full = train_df.groupby(col)[target].mean()
    test_encoded = test_df[col].map(means_full).fillna(global_mean)

    return train_encoded, test_encoded


# ----------------------------
# 3. CATEGORICAL ENCODING
# ----------------------------
def encode_categoricals(train_df: pd.DataFrame, test_df: pd.DataFrame, train_labels: pd.DataFrame):
    """
    Apply all categorical encodings:
    - Ordinal: age_group, education, income_poverty
    - One-hot: sex, marital_status, rent_or_own, health_insurance, race, employment_status, census_msa, hhs_geo_region
    - Target encoding (per vaccine): employment_industry, employment_occupation
    """
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()

    # --- Ordinal mappings ---
    for col, mapping in {
        "age_group": AGE_MAPPING,
        "education": EDUCATION_MAPPING,
        "income_poverty": INCOME_MAPPING
    }.items():
        if col in train_encoded.columns:
            train_encoded[col] = train_encoded[col].map(mapping)
            test_encoded[col] = test_encoded[col].map(mapping)

    # --- One-hot encoding ---
    onehot_cols = [
        "sex", "marital_status", "rent_or_own", "health_insurance",
        "race", "employment_status", "census_msa", "hhs_geo_region"
    ]
    train_encoded = pd.get_dummies(train_encoded, columns=[c for c in onehot_cols if c in train_encoded.columns], drop_first=True)
    test_encoded  = pd.get_dummies(test_encoded, columns=[c for c in onehot_cols if c in test_encoded.columns], drop_first=True)

    # --- Align train/test columns ---
    train_encoded, test_encoded = train_encoded.align(test_encoded, join="left", axis=1, fill_value=0)

    # --- Target encoding for high-cardinality ---
    for col in ["employment_industry", "employment_occupation"]:
        if col in train_df.columns:
            for target in ["h1n1_vaccine", "seasonal_vaccine"]:
                train_te, test_te = target_encode(train_df, test_df, col=col, target=target)
                train_encoded[f"{col}_te_{target}"] = train_te
                test_encoded[f"{col}_te_{target}"] = test_te

            # Drop raw categorical column after encoding
            train_encoded.drop(columns=[col], inplace=True)
            test_encoded.drop(columns=[col], inplace=True)

    return train_encoded, test_encoded