# src/model_selection_auc.py

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, RocCurveDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier


TARGET_COLS = ["h1n1_vaccine", "seasonal_vaccine"]


def load_data(train_features_path: str, train_labels_path: str):
    """Load feature and label data, cleaning up column names."""
    X = pd.read_csv(train_features_path)
    y = pd.read_csv(train_labels_path)[TARGET_COLS].astype(int)

    # Drop technical columns
    X = X.drop(columns=[c for c in ["Unnamed: 0", "respondent_id"] if c in X.columns])
    # Clean feature names
    X.columns = X.columns.str.replace("[^A-Za-z0-9_]+", "_", regex=True)
    return X, y


def cross_validate_model(model, X, y, model_name, n_splits=5):
    """Perform StratifiedKFold CV and return AUCs, confusion matrices."""
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {t: [] for t in TARGET_COLS}
    conf_mats = {t: np.zeros((2, 2), dtype=int) for t in TARGET_COLS}

    for label in TARGET_COLS:
        for train_idx, val_idx in kf.split(X, y[label]):
            X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_va = y[label].iloc[train_idx], y[label].iloc[val_idx]

            model.fit(X_tr, y_tr)
            y_prob = model.predict_proba(X_va)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            auc = roc_auc_score(y_va, y_prob)
            results[label].append(auc)

            cm = confusion_matrix(y_va, y_pred)
            conf_mats[label] += cm

    return {label: np.mean(scores) for label, scores in results.items()}, conf_mats


def run_model_benchmark(train_features_path, train_labels_path, n_splits=5, out_dir=None):
    """Compare Logistic Regression, RF, and LGBM using mean ROC-AUC."""
    X, y = load_data(train_features_path, train_labels_path)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            objective="binary",
            learning_rate=0.05,
            n_estimators=500,
            num_leaves=31,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            random_state=42,
        ),
    }

    all_results, all_conf_mats = [], {}

    for name, model in models.items():
        print(f"\nRunning {name}...")
        aucs, conf_mats = cross_validate_model(model, X, y, model_name=name, n_splits=n_splits)
        macro_auc = np.mean(list(aucs.values()))
        all_results.append({"model": name, **aucs, "macro_auc": macro_auc})
        all_conf_mats[name] = conf_mats

    results_df = pd.DataFrame(all_results).sort_values("macro_auc", ascending=False)

    # --- Visualization ---
    plt.figure(figsize=(8, 5))
    sns.barplot(data=results_df, x="model", y="macro_auc", palette="viridis")
    plt.title("Model Comparison by Mean ROC-AUC")
    plt.ylabel("Mean ROC-AUC (macro)")
    plt.xlabel("")
    plt.tight_layout()

    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(out_dir) / "auc_model_comparison.png", dpi=300)
    plt.show()

    # --- Confusion matrices (aggregate) ---
    fig, axes = plt.subplots(len(models), len(TARGET_COLS), figsize=(10, 9))
    for i, (model_name, confs) in enumerate(all_conf_mats.items()):
        for j, label in enumerate(TARGET_COLS):
            cm = confs[label]
            ax = axes[i, j] if len(models) > 1 else axes[j]
            sns.heatmap(
                cm,
                annot=True, fmt="d", cmap="Blues", cbar=False,
                ax=ax, annot_kws={"size": 10}
            )
            ax.set_title(f"{model_name} – {label}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
    plt.tight_layout()

    if out_dir:
        plt.savefig(Path(out_dir) / "confusion_matrices.png", dpi=300)
    plt.show()

    print("\n=== Model Comparison by Mean ROC-AUC ===")
    print(results_df.round(3))
    return results_df