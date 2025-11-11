"""
LightGBM modeling for FluShotML project
---------------------------------------
This script trains and evaluates LightGBM models on multi-label vaccination data.

Features:
- Cleans feature names automatically (safe for LightGBM)
- Supports K-fold cross-validation
- Evaluates and logs both training and validation F1 to check overfitting
- Saves results and feature importances

Author: FluShotML
"""

from __future__ import annotations
import re
import json
from pathlib import Path
from typing import Dict, Tuple

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
import matplotlib.pyplot as plt


# ----------------------------
# Globals
# ----------------------------
TARGET_COLS = ["h1n1_vaccine", "seasonal_vaccine"]


# ----------------------------
# Utility functions
# ----------------------------
def clean_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names to be LightGBM-compatible."""
    cleaned = []
    for c in df.columns:
        new_c = re.sub(r"[^A-Za-z0-9_]", "_", c)
        new_c = re.sub(r"_+", "_", new_c).strip("_")
        cleaned.append(new_c)
    df.columns = cleaned
    return df


def load_data(train_features_path: str, test_features_path: str, train_labels_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test data and clean feature names."""
    X = pd.read_csv(train_features_path)
    y = pd.read_csv(train_labels_path)[TARGET_COLS].astype(int)
    X = X.drop(columns=[c for c in ["Unnamed: 0", "Unnamed_0", "respondent_id"] if c in X.columns])
    X = clean_feature_names(X)
    return X, y


def compute_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
    """Compute standard binary classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


# ----------------------------
# LightGBM CV evaluation
# ----------------------------
def lgbm_cv(
    X: pd.DataFrame,
    y: pd.Series,
    label_name: str,
    params: Dict | None = None,
    n_splits: int = 5,
    seed: int = 42,
) -> Dict[str, float]:
    """Run LightGBM cross-validation for one label and return mean metrics."""
    if params is None:
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": 6,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 3,
            "lambda_l1": 0.5,
            "lambda_l2": 0.5,
            "verbose": -1,
            "force_col_wise": True,
            "seed": seed,
        }

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        print(f"Running fold {fold}/{n_splits} for {label_name}...")
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_va, label=y_va)

        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dval],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        y_pred = (model.predict(X_va, num_iteration=model.best_iteration) >= 0.5).astype(int)
        y_prob = model.predict(X_va, num_iteration=model.best_iteration)
        metrics = compute_metrics(y_va, y_pred, y_prob)
        all_metrics.append(metrics)

    # Aggregate metrics
    avg_metrics = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
    print(f"\n{label_name} CV results: " + ", ".join([f"{k}={v:.3f}" for k, v in avg_metrics.items()]))
    return avg_metrics


# ----------------------------
# Extended CV: Train vs Validation comparison
# ----------------------------
def lgbm_cv_with_train_eval(
    X: pd.DataFrame,
    y: pd.Series,
    label_name: str = "target",
    params: Dict | None = None,
    n_splits: int = 5,
    seed: int = 42,
    plot: bool = True,
) -> Dict[str, float]:
    """CV evaluation comparing training vs validation F1 per fold."""
    if params is None:
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": 6,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 3,
            "lambda_l1": 0.5,
            "lambda_l2": 0.5,
            "verbose": -1,
            "force_col_wise": True,
            "seed": seed,
        }

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_f1s, val_f1s = [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        print(f"Running fold {fold}/{n_splits} for {label_name}...")
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_va, label=y_va)

        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dtrain, dval],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        y_pred_tr = (model.predict(X_tr, num_iteration=model.best_iteration) >= 0.5).astype(int)
        y_pred_va = (model.predict(X_va, num_iteration=model.best_iteration) >= 0.5).astype(int)

        f1_tr = f1_score(y_tr, y_pred_tr, zero_division=0)
        f1_va = f1_score(y_va, y_pred_va, zero_division=0)
        train_f1s.append(f1_tr)
        val_f1s.append(f1_va)

        print(f"Fold {fold}: Train F1={f1_tr:.3f} | Val F1={f1_va:.3f}")

    mean_train, mean_val = np.mean(train_f1s), np.mean(val_f1s)
    gap = mean_train - mean_val
    print(f"\n=== {label_name} CV Summary ===")
    print(f"Train F1: {mean_train:.3f} | Val F1: {mean_val:.3f} | Gap: {gap:.3f}")

    if plot:
        plt.figure(figsize=(7, 4))
        plt.plot(range(1, n_splits + 1), train_f1s, marker="o", label="Train F1", color="blue")
        plt.plot(range(1, n_splits + 1), val_f1s, marker="s", label="Validation F1", color="red")
        plt.title(f"Train vs Validation F1 ({label_name})")
        plt.xlabel("Fold")
        plt.ylabel("F1 Score")
        plt.xticks(range(1, n_splits + 1))
        plt.grid(alpha=0.4, linestyle="--")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        "train_f1_mean": mean_train,
        "val_f1_mean": mean_val,
        "gap": gap,
        "train_f1_per_fold": train_f1s,
        "val_f1_per_fold": val_f1s,
    }


# ----------------------------
# Orchestrator for both labels
# ----------------------------
def run_lgbm_models(
    train_features_path: str = "data/processed/training_fe_full.csv",
    test_features_path: str = "data/processed/test_fe_full.csv",
    train_labels_path: str = "data/raw/training_set_labels.csv",
    n_splits: int = 5,
    out_dir: str = "artifacts_lgbm",
) -> Dict[str, Dict]:
    """Train LGBM models for both vaccine labels with cross-validation."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    X, y = load_data(train_features_path, test_features_path, train_labels_path)

    results = {}
    for label in TARGET_COLS:
        print(f"\n=== Training model for {label} ===")
        res = lgbm_cv(X, y[label], label_name=label, n_splits=n_splits)
        results[label] = res

    macro_f1 = float(np.mean([results[t]["f1"] for t in TARGET_COLS]))
    results["macro_f1"] = macro_f1

    with open(Path(out_dir) / "lgbm_cv_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== Summary (Macro F1) ===")
    for t in TARGET_COLS:
        print(f"{t:20s}: {results[t]['f1']:.3f}")
    print(f"Overall macro F1: {macro_f1:.3f}")
    return results
