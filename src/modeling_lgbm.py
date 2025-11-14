"""
LightGBM modeling for FluShotML
Fixed version addressing:
- Train/validation split for proper evaluation
- Consistent model saving/loading
- No data leakage
- Proper artifact management
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier
from scipy.stats import uniform, randint
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    accuracy_score, confusion_matrix
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from joblib import dump, load

import matplotlib.pyplot as plt
import seaborn as sns

# Targets
TARGET_COLS = ["h1n1_vaccine", "seasonal_vaccine"]

# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------

def clean_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [
        re.sub(r"_+", "_",
               re.sub(r"[^A-Za-z0-9_]", "_", c)
               ).strip("_")
        for c in df.columns
    ]
    return df


def load_training_data(train_features_path, train_labels_path):
    X = pd.read_csv(train_features_path)
    y = pd.read_csv(train_labels_path)[TARGET_COLS].astype(int)

    X = X.drop(columns=["respondent_id", "Unnamed: 0", "Unnamed_0"], errors="ignore")
    X = clean_feature_names(X)

    return X, y


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def save_confusion_matrices(y_true, y_pred, label, outdir):
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    # Raw
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{label} – Confusion Matrix")
    plt.tight_layout()
    plt.savefig(outdir / f"{label}_cm.png")
    plt.close()

    # Normalized
    cmn = confusion_matrix(y_true, y_pred, normalize="true")
    plt.figure(figsize=(4, 3))
    sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Greens")
    plt.title(f"{label} – Confusion Matrix (Normalized)")
    plt.tight_layout()
    plt.savefig(outdir / f"{label}_cm_normalized.png")
    plt.close()


# -------------------------------------------------------------------
# 1. BASELINE TRAINING (with train/val split)
# -------------------------------------------------------------------

def train_baseline_models(train_features, train_labels, outdir="artifacts_lgbm", random_state=42):
    Path(outdir).mkdir(exist_ok=True, parents=True)
    X, y = load_training_data(train_features, train_labels)
    
    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y["h1n1_vaccine"]
    )

    baseline_results = {}

    for target in TARGET_COLS:
        model = LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            learning_rate=0.03,
            n_estimators=300,
            verbose=-1,
            random_state=random_state
        )
        model.fit(X_train, y_train[target])

        # Evaluate on validation set
        y_prob_val = model.predict_proba(X_val)[:, 1]
        y_pred_val = (y_prob_val >= 0.5).astype(int)

        metrics = compute_metrics(y_val[target], y_pred_val, y_prob_val)

        save_confusion_matrices(y_val[target], y_pred_val, f"baseline_{target}", outdir)

        # Save model
        dump(model, Path(outdir) / f"baseline_{target}_model.pkl")

        with open(Path(outdir) / f"baseline_{target}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        baseline_results[target] = metrics

    with open(Path(outdir) / "baseline_results.json", "w") as f:
        json.dump(baseline_results, f, indent=2)

    return baseline_results


# -------------------------------------------------------------------
# 2. FINE-TUNING (with proper CV - no data leakage)
# -------------------------------------------------------------------

def tune_model(X, y, target, outdir="artifacts_lgbm", random_state=42):
    """
    Fine-tune model using cross-validation.
    Metrics are computed from CV, not on training data.
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)

    spw = (y == 0).sum() / (y == 1).sum() if target == "h1n1_vaccine" else 1.0

    base_model = LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        learning_rate=0.03,
        n_estimators=500,
        scale_pos_weight=spw,
        verbose=-1,
        random_state=random_state
    )

    param_dist = {
        "num_leaves": randint(20, 80),
        "max_depth": randint(3, 12),
        "feature_fraction": uniform(0.6, 0.4),
        "bagging_fraction": uniform(0.6, 0.4),
        "bagging_freq": randint(1, 8),
        "lambda_l1": uniform(0, 1),
        "lambda_l2": uniform(0, 1),
        "min_child_samples": randint(10, 100),
    }

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=30,
        scoring="f1",
        cv=5,
        random_state=random_state,
        verbose=0,
        n_jobs=-1
    )

    search.fit(X, y)

    best_params = search.best_params_
    best_model = search.best_estimator_
    
    # Get CV score (no data leakage)
    cv_score = search.best_score_

    # Save params
    with open(Path(outdir) / f"{target}_best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    # Save model
    dump(best_model, Path(outdir) / f"{target}_tuned_model.pkl")

    # Create a validation split for detailed metrics
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    # Retrain on train split and evaluate on validation
    best_model.fit(X_train, y_train)
    y_prob_val = best_model.predict_proba(X_val)[:, 1]
    y_pred_val = (y_prob_val >= 0.5).astype(int)
    metrics = compute_metrics(y_val, y_pred_val, y_prob_val)
    metrics["cv_f1_score"] = float(cv_score)

    save_confusion_matrices(y_val, y_pred_val, f"tuned_{target}", outdir)

    with open(Path(outdir) / f"tuned_{target}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return best_params, metrics


# -------------------------------------------------------------------
# 3. RETRAIN FINAL MODEL WITH TUNED PARAMS (on full dataset)
# -------------------------------------------------------------------

def train_final_model(X, y, target, best_params, outdir="artifacts_lgbm", random_state=42):
    """
    Train final model on full dataset using best parameters.
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)

    final_model = LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        learning_rate=0.03,
        n_estimators=500,
        verbose=-1,
        random_state=random_state,
        **best_params
    )

    final_model.fit(X, y)

    dump(final_model, Path(outdir) / f"{target}_final_model.pkl")

    print(f"✓ Final model for {target} saved")
    return final_model


# -------------------------------------------------------------------
# 4. PREDICT TEST SET
# -------------------------------------------------------------------

def load_final_models(model_dir="artifacts_lgbm"):
    """
    Load final models saved with joblib.
    """
    models = {}
    for target in TARGET_COLS:
        model_path = Path(model_dir) / f"{target}_final_model.pkl"
        models[target] = load(model_path)
    return models


def predict_test(models: dict, test_features_path: str) -> pd.DataFrame:
    """
    Make predictions on test set.
    
    Args:
        models: dict of {target: LGBMClassifier}
        test_features_path: path to test CSV
    
    Returns:
        DataFrame with respondent_id and probability predictions
    """
    df = pd.read_csv(test_features_path)

    # Extract IDs
    ids = df["respondent_id"].copy()

    # Drop the same columns dropped during training
    drop_cols = ["respondent_id", "Unnamed: 0", "Unnamed_0"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Clean feature names exactly as in training
    X = clean_feature_names(df)

    # Ensure train–test column consistency
    # Get feature names from one of the models
    feature_order = models["h1n1_vaccine"].feature_names_in_

    # Align columns
    missing_cols = set(feature_order) - set(X.columns)
    if missing_cols:
        print(f"Warning: Test data missing features: {missing_cols}")
        for col in missing_cols:
            X[col] = 0  # Add missing columns with default value
    
    X = X[feature_order]  # Reorder to match training

    preds = {"respondent_id": ids}

    for target in TARGET_COLS:
        model = models[target]
        prob = model.predict_proba(X)[:, 1]
        preds[target] = prob

    return pd.DataFrame(preds)