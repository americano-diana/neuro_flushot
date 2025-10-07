import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from src.mappings import AGE_MAPPING, EDUCATION_MAPPING, INCOME_MAPPING


# ============================================
# 1. IMPUTATION
# ============================================

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
        "opinion_seas_vacc_effective", "opinion_seas_risk", "opinion_seas_sick_from_vacc",
        "household_adults", "household_children"
    ]

    for col in opinion_behavior_cols:
        if col in df_train.columns:
            imputers[col] = df_train[col].median()

    return imputers


# ============================================
# 2. TARGET ENCODING (SAFE)
# ============================================

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


# ============================================
# 3. PREPROCESSOR CLASS
# ============================================

class FluShotPreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible preprocessing pipeline for the Flu Shot Learning dataset.
    Performs:
    - Imputation (categorical + numeric)
    - Ordinal encoding for ordered features
    - One-hot encoding for nominal features
    - Target encoding for high-cardinality features (per vaccine target)
    - Train/test column alignment
    """

    def __init__(self):
        self.imputers_ = None
        self.onehot_cols_ = [
            "sex", "marital_status", "rent_or_own", "health_insurance",
            "race", "employment_status", "census_msa", "hhs_geo_region"
        ]
        self.ordinal_mappings_ = {
            "age_group": AGE_MAPPING,
            "education": EDUCATION_MAPPING,
            "income_poverty": INCOME_MAPPING
        }
        self.high_card_cols_ = ["employment_industry", "employment_occupation"]

    # --------------------------
    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        """Fit imputers on training data."""
        self.imputers_ = fit_imputers(X)
        return self

    # --------------------------
    def _apply_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply predefined imputation strategy."""
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

        # Doctor recommendations
        for col in ["doctor_recc_h1n1", "doctor_recc_seasonal"]:
            if col in df_imputed.columns:
                df_imputed[col] = df_imputed[col].fillna(0)

        # Behavioral + numeric medians
        for col, med in self.imputers_.items():
            if col in df_imputed.columns:
                df_imputed[col] = df_imputed[col].fillna(med)

        return df_imputed

    # --------------------------
    def _base_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply ordinal and one-hot encoding."""
        df_encoded = df.copy()

        # Ordinal mappings
        for col, mapping in self.ordinal_mappings_.items():
            if col in df_encoded.columns:
                df_encoded[col] = df_encoded[col].map(mapping)

        # One-hot encoding
        df_encoded = pd.get_dummies(
            df_encoded,
            columns=[c for c in self.onehot_cols_ if c in df_encoded.columns],
            drop_first=True
        )

        return df_encoded

    # --------------------------
    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        """
        Fit on training data (features + labels), apply imputations,
        encodings, and target encoding per vaccine target.
        """
        # Merge target columns for encoding
        if "respondent_id" in X_train.columns:
            merged_train = X_train.merge(y_train, on="respondent_id", how="left")
        else:
            merged_train = X_train.copy()
            merged_train[["h1n1_vaccine", "seasonal_vaccine"]] = y_train[["h1n1_vaccine", "seasonal_vaccine"]]

        # Step 1: Fit imputers
        self.fit(merged_train)

        # Step 2: Apply imputation + encodings
        train_base = self._apply_imputation(merged_train)
        train_base = self._base_encode(train_base)

        # Step 3: Target encoding for high-cardinality columns
        for col in self.high_card_cols_:
            if col in merged_train.columns:
                for target in ["h1n1_vaccine", "seasonal_vaccine"]:
                    train_te, _ = target_encode(merged_train, merged_train, col=col, target=target)
                    train_base[f"{col}_te_{target}"] = train_te
                if col in train_base.columns:
                    train_base.drop(columns=[col], inplace=True)

        # Drop targets from final features
        train_base = train_base.drop(columns=["h1n1_vaccine", "seasonal_vaccine"], errors="ignore")

        return train_base

    # --------------------------
    def transform_test(self, X_test: pd.DataFrame, X_train: pd.DataFrame, y_train: pd.DataFrame):
        """Transform test data using train-based encoders and target means."""
        # Merge labels for encoding computation
        merged_train = X_train.merge(y_train, on="respondent_id", how="left")

        # Step 1: Imputation + base encoding
        train_base = self._apply_imputation(merged_train)
        train_base = self._base_encode(train_base)

        test_base = self._apply_imputation(X_test)
        test_base = self._base_encode(test_base)

        # Step 2: Apply target encodings (use train mapping)
        for col in self.high_card_cols_:
            if col in merged_train.columns:
                for target in ["h1n1_vaccine", "seasonal_vaccine"]:
                    train_te, test_te = target_encode(merged_train, X_test, col=col, target=target)
                    train_base[f"{col}_te_{target}"] = train_te
                    test_base[f"{col}_te_{target}"] = test_te

                if col in train_base.columns:
                    train_base.drop(columns=[col], inplace=True)
                if col in test_base.columns:
                    test_base.drop(columns=[col], inplace=True)

        # Step 3: Align columns AFTER all encodings
        train_base, test_base = train_base.align(test_base, join="outer", axis=1, fill_value=0)

        # Drop targets from test if present accidentally
        test_base = test_base.drop(columns=["h1n1_vaccine", "seasonal_vaccine"], errors="ignore")

        return test_base
