"""
LightGBM Modeling for FluShotML — Sprint 2
-------------------------------------------
Trains separate LightGBM models for H1N1 and Seasonal vaccine uptake
with K-Fold cross-validation.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Tuple
import re

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


TARGET_COLS = ["h1n1_vaccine", "seasonal_vaccine"]


# =========================================================
# Utilities
# =========================================================
def load_data(train_features_path, test_features_path, train_labels_path):
    """
    Load features and labels, remove unwanted columns,
    and clean feature names to be compatible with LightGBM.
    """
    def clean_feature_names(df: pd.DataFrame) -> pd.DataFrame:
        cleaned = []
        for c in df.columns:
            # replace any non-alphanumeric character with underscore
            new_c = re.sub(r'[^A-Za-z0-9_]', '_', c)
            # collapse multiple underscores and strip leading/trailing ones
            new_c = re.sub(r'_+', '_', new_c).strip('_')
            cleaned.append(new_c)
        df.columns = cleaned
        return df

    # Load datasets
    X = pd.read_csv(train_features_path)
    y = pd.read_csv(train_labels_path)[TARGET_COLS].astype(int)

    # Remove irrelevant columns
    X = X.drop(columns=[c for c in ["Unnamed_0", "Unnamed: 0", "respondent_id"] if c in X.columns])

    # Clean feature names for LightGBM compatibility
    X = clean_feature_names(X)

    return X, y



def compute_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


# =========================================================
# Main CV Function
# =========================================================
def lgbm_cv(
    X: pd.DataFrame,
    y: pd.Series,
    label_name: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, float]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    metrics_all = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"Running fold {fold}/{n_splits} for {label_name}...")

        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        train_set = lgb.Dataset(X_tr, label=y_tr)
        val_set = lgb.Dataset(X_va, label=y_va)

        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "learning_rate": 0.03,
            "num_leaves": 31,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 3,
            "verbose": -1,
            "seed": random_state,
            "force_col_wise": True,
        }

        model = lgb.train(
            params,
            train_set,
            valid_sets=[train_set, val_set],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )


        y_prob = model.predict(X_va, num_iteration=model.best_iteration)
        y_pred = (y_prob >= 0.5).astype(int)

        m = compute_metrics(y_va, y_pred, y_prob)
        metrics_all.append(m)

    avg_metrics = {k: float(np.mean([m[k] for m in metrics_all])) for k in metrics_all[0].keys()}
    print(f"\n{label_name} CV results: " + ", ".join([f"{k}={v:.3f}" for k, v in avg_metrics.items()]))

    return avg_metrics


# =========================================================
# Runner
# =========================================================
def run_lgbm_models(
    train_features_path="data/processed/training_fe_full.csv",
    test_features_path="data/processed/test_fe_full.csv",
    train_labels_path="data/raw/training_set_labels.csv",
    n_splits=5,
    out_dir="artifacts_lgbm",
) -> Dict:
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    X, y = load_data(train_features_path, test_features_path, train_labels_path)

    results = {}
    for target in TARGET_COLS:
        results[target] = lgbm_cv(X, y[target], target, n_splits=n_splits)

    # macro average
    macro_f1 = np.mean([results[t]["f1"] for t in TARGET_COLS])
    results["macro_f1"] = float(macro_f1)

    with open(Path(out_dir) / "lgbm_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== Summary (Macro F1) ===")
    for t in TARGET_COLS:
        print(f"{t:20s}: {results[t]['f1']:.3f}")
    print(f"Overall macro F1: {macro_f1:.3f}")

    return results


if __name__ == "__main__":
    run_lgbm_models()