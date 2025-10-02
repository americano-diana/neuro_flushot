# Flu Shot Project

This project is part of the ReDI School Data Circle. It is organized into **three sprints**, each focusing on a key stage of the data science workflow, using the Flu Shot dataset.

## Sprint 1: Data Exploration & Processing
### Goals
- Load and inspect the dataset.
- Understand data types, distributions, and missing values.
- Apply preprocessing (cleaning, transformations, feature creation).
- Document findings and prepare datasets for modeling.

### Deliverables
- Processed dataset stored in `data/processed/`.
- Sprint 1 notebook (`sprint1_eda.ipynb`) with data exploration and processing steps.
- Metadata or data dictionary in `data/references/`.

## Sprint 2: Modeling & Validation
### Goals
- Define baseline models and training strategies.
- Implement feature engineering and transformations.
- Evaluate models using suitable metrics and validation approaches.
- Compare different modeling approaches.

### Deliverables
- Trained models stored in `models/`.
- Sprint 2 notebook (`sprint2_modeling.ipynb`) with training, validation, and performance comparison.
- Reusable functions for modeling in `src/modeling.py` and evaluation in `src/evaluation.py`.

## Sprint 3: Insights & Reporting
### Goals
- Interpret model outputs and identify key drivers of predictions.
- Summarize results with clear visualizations.
- Prepare reporting materials for stakeholders.
- Prototype dashboard or reporting scripts if applicable.

### Deliverables
- Visualizations saved in `reports/figures/`.
- Sprint 3 notebook (`sprint3_insights.ipynb`) with insights, interpretation, and reporting.
- Final project summary in `reports/final_report.md`.

## Repository Structure
```
flu_shot/
│
├── data/
│   ├── raw/                   # initial dataset(s)
│   ├── processed/             # cleaned/feature-engineered datasets
│   └── references/            # data dictionary, metadata, manuals
│
├── notebooks/
│   ├── sprint1_eda.ipynb      # exploration and preprocessing
│   ├── sprint2_modeling.ipynb # modeling and validation
│   └── sprint3_insights.ipynb # insights and reporting
│
├── models/                    # trained models
│
├── src/                       # reusable Python modules
│   ├── config.py              # central config (paths, constants, variables)
│   ├── utils.py               # helper functions
│   ├── data_loader.py         # functions to load and validate data
│   ├── preprocessing.py       # cleaning and feature transformations
│   ├── modeling.py            # model training/tuning
│   └── evaluation.py          # evaluation metrics/visualizations
│
├── reports/
│   ├── figures/               # exported plots and figures
│   └── final_report.md        # project summary
│
├── requirements.txt           # Python dependencies
├── .gitignore                 # files and folders to ignore in Git
└── README.md                  # project description and instructions
```

## Getting Started
1. Clone the repository.
2. Place raw dataset(s) into `data/raw/`.
3. Add any metadata or data dictionaries into `data/references/`.
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Open the notebooks in VSCode:
   ```bash
   code notebooks/
   ```

## Summary of Deliverables
- **Sprint 1:** Cleaned dataset, data exploration notebook.  
- **Sprint 2:** Models, modeling notebook, evaluation scripts.  
- **Sprint 3:** Insights notebook, visualizations, final report.  
