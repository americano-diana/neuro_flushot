# src/mappings.py

"""
Reusable encoding mappings for categorical features.
These mappings ensure consistent ordinal encodings between train and test datasets.
"""

# --- Ordinal Encodings ---

AGE_MAPPING = {
    "18 - 34 Years": 1,
    "35 - 44 Years": 2,
    "45 - 54 Years": 3,
    "55 - 64 Years": 4,
    "65+ Years": 5
}

EDUCATION_MAPPING = {
    "< 12 Years": 1,
    "12 Years": 2,
    "Some College": 3,
    "College Graduate": 4,
    "Missing": 0
}

INCOME_MAPPING = {
    "Below Poverty": 1,
    "<= $75,000, Above Poverty": 2,
    "> $75,000": 3,
    "Missing": 0
}

# --- Notes ---
# - Small nominal features (sex, marital_status, rent_or_own, health_insurance)
#   are one-hot encoded directly in preprocessing (not mapped here).
# - High-cardinality features (employment_industry, employment_occupation)
#   are target encoded for both vaccine targets.