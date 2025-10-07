"""
Reusable encoding mappings for categorical features.
These mappings ensure consistency between train and test datasets.
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
    "Missing": 0   # explicit "missing" category
}

INCOME_MAPPING = {
    "Below Poverty": 1,
    "<= $75,000, Above Poverty": 2,
    "> $75,000": 3,
    "Missing": 0  # explicit "missing" category
}

# --- Small Nominal / Binary Encodings ---

SEX_MAPPING = {"Female": 0, "Male": 1}

MARITAL_STATUS_MAPPING = {"Not Married": 0, "Married": 1, "Missing": 2}

RENT_OWN_MAPPING = {"Rent": 0, "Own": 1, "Missing": 2}

HEALTH_INSURANCE_MAPPING = {"0.0": 0, "1.0": 1, "Missing": 2}

"""
--- Notes ---

High-cardinality variables (employment_industry, employment_occupation) are not mapped here.
They should be frequency-encoded dynamically from the training data, then applied to the test set.
For one-hot encoding features (race, employment_status, census_msa, hhs_geo_region), no mappings are needed.

"""