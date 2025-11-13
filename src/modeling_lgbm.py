"""
LightGBM modeling for FluShotML project
---------------------------------------
Trains, evaluates, and fine-tunes LightGBM models for multi-label vaccination prediction.

Includes:
- Cross-validated LightGBM baseline
- RandomizedSearchCV fine-tuning (F1 optimization)
- Confusion matrices (raw + normalized)
- scale_pos_weight applied only to H1N1 model
"""

from __future__ import annotations
import re
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from scipy.stats import uniform, randint
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# ----------------------------
# Globals
# ----------------------------
TARGET_COLS = ["h1n1_vaccine", "seasonal_vaccine"]


# ----------------------------
# Utilities
# ----------------------------
def clean_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = []
    for c in df.columns:
        new_c = re.sub(r"[^A-Za-z0-9_]", "_", c)
        new_c = re.sub(r"_+", "_", new_c).strip("_")
        cleaned.append(new_c)
    df.columns = cleaned
    return df


def load_data(train_features_path: str, test_features_path: str, train_labels_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X = pd.read_csv(train_features_path)
    y = pd.read_csv(train_labels_path)[TARGET_COLS].astype(int)
    X = X.drop(columns=[c for c in ["Unnamed: 0", "Unnamed_0", "respondent_id"] if c in X.columns])
    X = clean_feature_names(X)
    return X, y


def compute_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


# ----------------------------
# Confusion Matrices
# ----------------------------
def create_confusion_matrices(y_true: np.ndarray, y_pred: np.ndarray, label_name: str, out_dir: str):
    """Generate and save raw and normalized confusion matrix heatmaps."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Raw confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix: {label_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    raw_path = out_dir / f"confusion_matrix_{label_name}.png"
    plt.savefig(raw_path)
    plt.close()

    # Normalized confusion matrix
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens", cbar=True)
    plt.title(f"Normalized Confusion Matrix: {label_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    norm_path = out_dir / f"confusion_matrix_{label_name}_normalized.png"
    plt.savefig(norm_path)
    plt.close()

    print(f"Saved confusion matrices for {label_name}:")
    print(f"  Raw        → {raw_path}")
    print(f"  Normalized → {norm_path}")


# ----------------------------
# Baseline LightGBM CV
# ----------------------------
def lgbm_cv(X: pd.DataFrame, y: pd.Series, label_name: str, params: Dict | None = None, n_splits: int = 5, seed: int = 42) -> Dict[str, float]:
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
            "verbosity": -1,
            "force_col_wise": True,
            "seed": seed,
        }

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_metrics, y_true_all, y_pred_all = [], [], []

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

        y_prob = model.predict(X_va, num_iteration=model.best_iteration)
        y_pred = (y_prob >= 0.5).astype(int)
        y_true_all.extend(y_va)
        y_pred_all.extend(y_pred)
        all_metrics.append(compute_metrics(y_va, y_pred, y_prob))

    avg_metrics = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
    avg_metrics["y_true_all"] = y_true_all
    avg_metrics["y_pred_all"] = y_pred_all
    print(f"{label_name} CV results: " + ", ".join([f"{k}={v:.3f}" for k, v in avg_metrics.items() if k not in ['y_true_all', 'y_pred_all']]))
    return avg_metrics


# ----------------------------
# Fine-tuning with RandomizedSearchCV
# ----------------------------
def tune_lgbm_model(X: pd.DataFrame, y: pd.Series, target: str, n_iter: int = 30, cv: int = 5, random_state: int = 42):
    """Fine-tune LightGBM using RandomizedSearchCV to maximize F1-score."""
    print(f"\n=== Fine-tuning {target} model ===")
    spw = (y == 0).sum() / (y == 1).sum() if target == "h1n1_vaccine" else 1.0
    print(f"Applying scale_pos_weight={spw:.2f} for {target}")

    base_model = LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=500,
        learning_rate=0.03,
        random_state=random_state,
        scale_pos_weight=spw,
        verbose=-1,
    )

    param_distributions = {
        "num_leaves": randint(20, 80),
        "max_depth": randint(3, 12),
        "feature_fraction": uniform(0.6, 0.4),
        "bagging_fraction": uniform(0.6, 0.4),
        "bagging_freq": randint(1, 8),
        "lambda_l1": uniform(0, 1),
        "lambda_l2": uniform(0, 1),
        "min_child_samples": randint(10, 100),
    }

    searcher = RandomizedSearchCV(
        base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=0,
        random_state=random_state,
    )

    searcher.fit(X, y)
    print(f"Best params for {target}: {searcher.best_params_}")
    print(f"Best F1-score (CV): {searcher.best_score_:.4f}")

    best_model = searcher.best_estimator_
    y_pred = best_model.predict(X)
    y_prob = best_model.predict_proba(X)[:, 1]

    metrics = compute_metrics(y, y_pred, y_prob)
    print("\nFine-tuned metrics:")
    for k, v in metrics.items():
        print(f"{k:10s}: {v:.3f}")

    create_confusion_matrices(y, y_pred, f"{target}_tuned", "artifacts_lgbm")

    return {
        "best_model": best_model,
        "best_params": searcher.best_params_,
        "cv_f1": searcher.best_score_,
        "metrics": metrics,
    }


# ----------------------------
# Run for both labels
# ----------------------------
def run_lgbm_models(
    train_features_path: str = "data/processed/training_fe_full.csv",
    test_features_path: str = "data/processed/test_fe_full.csv",
    train_labels_path: str = "data/raw/training_set_labels.csv",
    n_splits: int = 5,
    out_dir: str = "artifacts_lgbm",
) -> Dict[str, Dict]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    X, y = load_data(train_features_path, test_features_path, train_labels_path)

    results = {}
    for label in TARGET_COLS:
        print(f"\n=== Training model for {label} ===")
        res = lgbm_cv(X, y[label], label_name=label, n_splits=n_splits)
        create_confusion_matrices(np.array(res["y_true_all"]), np.array(res["y_pred_all"]), label_name=label, out_dir=out_dir)
        del res["y_true_all"], res["y_pred_all"]
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
