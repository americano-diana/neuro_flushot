
---

# Ticket 1.1.4 – Initial Observations and Questions

## 🔹 Key Observations

### Dataset structure

* **Training set**: 26,707 rows, 36 features
* **Labels**: 26,707 rows, 2 targets (`h1n1_vaccine`, `seasonal_vaccine`)
* **Test set**: 26,708 rows, 36 features
* Features include **behavioral**, **opinion-based**, **demographic**, and **household** information.

### Missing values

* Several features contain missing values.

  * **Employment-related**: `employment_industry` (~50%), `employment_occupation` (~50%)
  * **Health insurance**: `health_insurance` (~46%)
  * **Socio-economic**: `income_poverty` (~16%), `education` and `marital_status` (~5%)
  * **Doctor recommendations**: `doctor_recc_h1n1`, `doctor_recc_seasonal` (~8%)
* Many opinion/behavioral features have small but non-trivial gaps (<3%).

### Numerical features

* Most behavioral/opinion features are ordinal or binary (0–1, sometimes up to 5).
* Example: `h1n1_concern` ranges 0–3, `opinion_h1n1_vacc_effective` ranges 1–5.
* Household features (`household_adults`, `household_children`) are small integer counts.

### Categorical features

* **Age** is fairly balanced but slightly skewed towards older groups (65+).
* **Sex**: More females (59%) than males (41%).
* **Race**: Majority White (79%), with smaller representation of other groups.
* **Income**: Most respondents above poverty threshold.
* **Employment**: Majority employed (54%), ~41% not in labor force, small unemployed group.

### Targets

* **H1N1 vaccine**: Strong class imbalance (21% vaccinated vs 79% not vaccinated).
* **Seasonal vaccine**: More balanced (46% vaccinated vs 54% not vaccinated).

---

## 🔹 Potential Challenges

1. **Class imbalance for H1N1 vaccine** → may require resampling, class weights, or careful metric choice.
2. **High missingness** in employment and insurance data → decisions needed: drop, impute, or encode "unknown".
3. **Mixed feature types** (ordinal, categorical, binary, continuous) → preprocessing pipeline must handle appropriately.
4. **High-cardinality categories** (`employment_industry`, `employment_occupation`, `hhs_geo_region`) → need encoding strategies (target encoding, frequency encoding, etc.).
5. **Correlation among opinion variables** → risk of multicollinearity, might need dimensionality reduction or feature grouping.

---

## 🔹 Questions for Further Investigation

* How do **doctor recommendations** correlate with vaccination uptake?
* Are **opinion-based features** (risk, effectiveness) stronger predictors than demographics?
* Do **employment/occupation/industry** variables add predictive power despite missingness?
* Are there **geographic differences** (region, MSA type) in vaccination rates?
* Do **household structure variables** (adults, children) influence likelihood of vaccination?

---