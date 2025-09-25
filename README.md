# Flu Shot Project - Sprint 1

This project is part of the ReDI School Data Circle, focusing on **Sprint 1** tasks for the Flu Shot dataset.

## Goals of Sprint 1
- Explore the dataset (understand variables, distributions, and missing data).
- Provide **demographic breakdowns** (age, gender, etc.) related to flu vaccination.
- Provide **geographic/mapping insights** if location data is available.
- Summarize insights visually with charts and maps.

## Repository Structure
```
flu_shot_sprint1/
│
├── data/
│   ├── raw/                   # put initial dataset(s) here
│   └── processed/             # cleaned data for exploration / visuals
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_demographics.ipynb
│   └── 03_geographic_and_maps.ipynb
│
├── src/
│   ├── data_cleaning.py       # functions for cleaning and preprocessing
│   ├── demographics.py        # functions for demographic summaries
│   └── maps.py                # functions for geographic visualizations
│
├── reports/
│   └── figures/               # exported plots and maps
│
├── README.md                  # project description and instructions
├── requirements.txt           # list of Python dependencies
└── .gitignore                 # files and folders to ignore in Git
```

## Getting Started
1. Clone the repository.
2. Place the raw dataset(s) into `data/raw/`.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Open Jupyter notebooks to run analyses:
   ```bash
   jupyter notebook notebooks/
   ```

## Deliverables
- Cleaned dataset in `data/processed/`
- Jupyter notebooks with exploration, demographic, and geographic insights
- Visualizations saved in `reports/figures/`
