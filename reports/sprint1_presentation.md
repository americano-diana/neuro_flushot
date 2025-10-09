# Slide 1 — Data Overview (Flu Shot Learning)

**Dataset**
- Train features: **26,707 × 36** | Test features: **26,708 × 36**  
- Labels: **26,707 × 2** → `h1n1_vaccine`, `seasonal_vaccine`
- Feature mix: behavioral & opinions (ordinal/binary), demographics & household, geography, employment

**Targets (class balance)**
- **H1N1**: 21.25% yes / 78.75% no  
- **Seasonal**: 46.56% yes / 53.44% no

**Missingness (key patterns)**
- High: `employment_industry` (~50%), `employment_occupation` (~50%), `health_insurance` (~46%)
- Mid: `income_poverty` (~16%), `education`, `marital_status`, `employment_status`, `rent_or_own` (5–10%)
- Paired gaps: `doctor_recc_h1n1` & `doctor_recc_seasonal` (~8%)  
- Takeaway: missingness is **informative** (non-random), especially across socio-economic items


# Slide 2 — Main Insights / Results

**Data quality & distributions**
- Ordinal scales and binaries sit in expected discrete buckets; **no meaningful outliers**
- Household counts capped at **0–3** (3 = “3+”); no extremes

**Encodings & artifacts**
- Created **aligned** `train_encoded` / `test_encoded` (same columns)
- High-card features (`employment_industry`, `employment_occupation`) **target-encoded per target** (K-Fold, no leakage)
- Saved correlation heatmap with hierarchical clustering →  
  `reports/figures/correlation_heatmap_clustered.png`

**Correlation & redundancy**
- No pairs ≥ **0.9**; at **0.8** only **expected** correlations:
  - One-hot siblings (e.g., `rent_or_own_*`, `health_insurance_*`)
  - Missingness indicators co-occurring (`marital_status_Missing` ↔ `employment_status_Missing`)
  - Target encodings across the two targets (`*_te_h1n1` ↔ `*_te_seasonal`)
- **Decision:** keep all features; none are truly redundant in meaning


# Slide 3 — Key Steps & Decisions (WHAT and WHY)

**Imputation (WHAT)**
- Categorical gaps → `"Missing"` label  
- `doctor_recc_*` (binary) → **0**  
- Ordinal/numeric (incl. household) → **median**
- **WHY:** preserve signal from non-response; keep scales sensible; avoid bias

**Leakage control (WHAT)**
- Medians **fit on train** then applied to test  
- Target encoding via **K-Fold** means (per target)  
- **WHY:** prevent look-ahead; generalize honestly

**Encoding (WHAT)**
- **Ordinal mapping:** `age_group`, `education`, `income_poverty`  
- **One-hot:** small/medium nominal (`sex`, `marital_status`, `rent_or_own`, `health_insurance`, `race`, `employment_status`, `census_msa`, `hhs_geo_region`)  
- **Target encoding:** high-card (`employment_industry`, `employment_occupation`) → `*_te_h1n1_vaccine`, `*_te_seasonal_vaccine`
- **WHY:** model-agnostic, low dimensionality, retain target signal

**Redundancy policy (WHAT)**
- Review at |r|≥0.8; keep expected pairs (one-hot siblings, missingness, cross-target TE)  
- **WHY:** correlated ≠ redundant; pairs encode different semantics

> _Note:_ Two additional slides with **visualization highlights** will be added separately (team hand-off).


---

### 🎙️ **Presentation Script (for 3 given slides)**

---

**Slide 1 – Data Overview**

> Let me start with a quick overview of the dataset we’ve been working on.
> This is the *Flu Shot Learning* dataset from DrivenData, which combines survey responses, demographic information, and behavioral indicators to predict two outcomes: whether someone received the H1N1 vaccine, and whether they received the seasonal flu vaccine.
>
> The training data has about twenty-six thousand records and thirty-six features, and the test set is almost identical in size.
> The features mix behavioral and opinion questions, household information, and employment or geographic identifiers.
>
> Both targets are binary, but the classes are imbalanced: roughly twenty-one percent of people received the H1N1 vaccine, while about forty-six percent received the seasonal one.
>
> When exploring missing values, we saw that some features — like *employment industry*, *occupation*, and *health insurance* — have almost half of their data missing. Others, like *income poverty* and *education*, have moderate levels of missingness.
> Interestingly, certain groups of variables tend to be missing together — for example, the two *doctor recommendation* features.
> So our takeaway was that missingness itself carries useful information about socio-economic behavior, not just data quality issues.

---

**Slide 2 – Main Insights and Results**

> Once the data was cleaned and encoded, we verified that the numeric features behaved as expected.
> Most of the variables are on discrete ordinal scales — things like one-to-five ratings or binary yes-no responses — and there were no meaningful outliers.
> Household counts are capped at three, which is how the survey encoded “three or more”, so again, everything looked consistent.
>
> We created aligned training and test datasets, so they now share exactly the same feature structure.
> For the two high-cardinality employment fields, we used target encoding — but separately for each vaccine target — which helps the model learn predictive patterns without blowing up the number of columns.
>
> Then, we generated a full correlation heatmap with hierarchical clustering, saved as *correlation_heatmap_clustered.png* in the reports folder.
> This gave us a visual overview of how features group together.
>
> When we examined correlations numerically, we found no pairs above zero-point-nine, and only five pairs above zero-point-eight.
> These all made perfect sense:
> one-hot dummy columns for the same variable, missingness indicators that co-occur, and target encodings that are similar because vaccination behaviors tend to align.
> So overall, the data is diverse, not redundant, and well prepared for modeling.

---

**Slide 3 – Key Steps and Decisions**

> Let’s finish with the main decisions and why we made them.
>
> For **imputation**, categorical gaps were filled with the label “Missing”, doctor-recommendation binaries were set to zero, and numeric or ordinal values used the median from the training set.
> This approach preserves real survey patterns — we don’t just erase missingness — and it avoids leakage by applying train-based statistics to the test set.
>
> For **encoding**, we followed a structured strategy:
> ordinal mappings for ordered features like *education* or *income*,
> one-hot encoding for small or medium categorical variables,
> and target encoding for the high-cardinality employment features.
> This balance keeps the data model-agnostic and efficient, while retaining predictive signal.
>
> Finally, in the **redundancy check**, we reviewed all pairs with high correlation.
> Since every pair reflected a meaningful relationship — such as dummy variables or parallel encodings — we decided not to remove anything at this stage.
> The feature space remains complete, interpretable, and ready for scaling and model experimentation.
>
> Two more slides with visualization results will follow later from the other team member, but this concludes the data preparation and feature analysis work achieved in this sprint.
