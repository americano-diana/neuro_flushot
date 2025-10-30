"""
Modeling (Sprint 2) — Multi-label with Classifier Chains
--------------------------------------------------------
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier


TARGET_COLS = ["h1n1_vaccine", "seasonal_vaccine"]


# =========================================================
# DATA LOADING
# =========================================================
def load_data(train_features_path, test_features_path, train_labels_path):
    train_X = pd.read_csv(train_features_path)
    test_X = pd.read_csv(test_features_path)
    y = pd.read_csv(train_labels_path)[TARGET_COLS].astype(int)
    return train_X, test_X, y


def joint_label_stratification(y):
    """Create a joint label for stratified CV."""
    return np.char.add(y[TARGET_COLS[0]].astype(str).values, y[TARGET_COLS[1]].astype(str).values)


# =========================================================
# METRICS
# =========================================================
@dataclass
class FoldMetrics:
    per_label: Dict[str, Dict[str, float]]
    macro: Dict[str, float]
    micro: Dict[str, float]


def compute_metrics(y_true, y_pred, y_proba):
    per_label = {}
    for i, t in enumerate(TARGET_COLS):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        yp_proba = y_proba[:, i] if y_proba is not None else None

        per_label[t] = {
            "accuracy": accuracy_score(yt, yp),
            "precision": precision_score(yt, yp, zero_division=0),
            "recall": recall_score(yt, yp, zero_division=0),
            "f1": f1_score(yt, yp, zero_division=0),
            "roc_auc": roc_auc_score(yt, yp_proba) if yp_proba is not None else np.nan,
        }

    macro = {"f1": f1_score(y_true, y_pred, average="macro")}
    micro = {"f1": f1_score(y_true.ravel(), y_pred.ravel(), average="micro")}
    return FoldMetrics(per_label=per_label, macro=macro, micro=micro)


def aggregate_metrics(folds):
    out = {"per_label": {t: {} for t in TARGET_COLS}, "macro": {}, "micro": {}}
    for t in TARGET_COLS:
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            vals = [fm.per_label[t][metric] for fm in folds]
            out["per_label"][t][metric] = float(np.mean(vals))
    out["macro"]["f1"] = float(np.mean([fm.macro["f1"] for fm in folds]))
    out["micro"]["f1"] = float(np.mean([fm.micro["f1"] for fm in folds]))
    return out


# =========================================================
# MODELS
# =========================================================
def build_base_estimator():
    return LogisticRegression(
        solver="lbfgs",
        class_weight="balanced",
        max_iter=2000,
        n_jobs=-1,
        random_state=42,
    )


def build_independent_model():
    return MultiOutputClassifier(build_base_estimator())


def build_chain_model_pair(order):
    """Return a pair of ClassifierChains so we predict both targets."""
    chain1 = ClassifierChain(estimator=build_base_estimator(), order=order, random_state=42)
    chain2 = ClassifierChain(estimator=build_base_estimator(), order=order[::-1], random_state=42)
    return [chain1, chain2]


# =========================================================
# CROSS-VALIDATION
# =========================================================
def cross_val_evaluate(model_builder, X, y, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_joint = joint_label_stratification(y)

    folds = []
    logs = []

    X_np = X.values
    y_np = y[TARGET_COLS].values

    for fold_idx, (tr, va) in enumerate(skf.split(X_np, y_joint), start=1):
        print(f"Running fold {fold_idx}/{n_splits}")
        X_tr, X_va = X_np[tr], X_np[va]
        y_tr, y_va = y_np[tr], y_np[va]

        model_obj = model_builder()
        # If model_builder returns a list (two chains)
        if isinstance(model_obj, list):
            preds_list = []
            probas_list = []
            for m in model_obj:
                m.fit(X_tr, y_tr)
                p = m.predict(X_va)
                if p.ndim == 1:
                    p = p.reshape(-1, 1)
                preds_list.append(p)
                if hasattr(m, "predict_proba"):
                    probs = m.predict_proba(X_va)
                    if isinstance(probs, list):
                        probs = np.column_stack([x[:, 1] if x.ndim == 2 else np.zeros_like(x) for x in probs])
                    probas_list.append(probs)
            y_pred = np.column_stack(preds_list)[:, :2]
            y_proba = np.column_stack(probas_list)[:, :2] if probas_list else None
        else:
            model = model_obj
            model.fit(X_tr, y_tr)

            y_pred = None
            y_proba = None

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_va)
                if isinstance(probs, list):
                    y_proba = np.column_stack([p[:, 1] if p.ndim == 2 else np.zeros_like(p) for p in probs])
                elif isinstance(probs, np.ndarray) and probs.ndim == 3:
                    y_proba = np.stack([p[:, 1] for p in probs], axis=1)
                if y_proba is not None:
                    y_pred = (y_proba >= 0.5).astype(int)

            if y_pred is None:
                y_pred = model.predict(X_va)
                if y_pred.ndim == 1:
                    y_pred = y_pred.reshape(-1, 1)
                if y_pred.shape[1] != len(TARGET_COLS):
                    y_pred = np.column_stack([y_pred] * len(TARGET_COLS))

        fm = compute_metrics(y_va, y_pred, y_proba)
        folds.append(fm)
        logs.append({"fold": fold_idx, "macro_f1": fm.macro["f1"]})

    avg = aggregate_metrics(folds)
    return {"avg": avg, "folds": logs}


# =========================================================
# RUNNER
# =========================================================
def run_all(
    train_features_path="data/processed/training_fe_full.csv",
    test_features_path="data/processed/test_fe_full.csv",
    train_labels_path="data/raw/training_set_labels.csv",
    n_splits=5,
    out_dir="artifacts",
    fit_best_on_full=True,
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train = load_data(train_features_path, test_features_path, train_labels_path)

    print("Running Independent models...")
    res_indep = cross_val_evaluate(build_independent_model, X_train, y_train, n_splits=n_splits)

    print("Running Classifier Chain pair (h1n1 -> seasonal)...")
    res_chain_01 = cross_val_evaluate(lambda: build_chain_model_pair([0, 1]), X_train, y_train, n_splits=n_splits)

    print("Running Classifier Chain pair (seasonal -> h1n1)...")
    res_chain_10 = cross_val_evaluate(lambda: build_chain_model_pair([1, 0]), X_train, y_train, n_splits=n_splits)

    candidates = {
        "independent": res_indep["avg"]["macro"]["f1"],
        "chain_01 (pair)": res_chain_01["avg"]["macro"]["f1"],
        "chain_10 (pair)": res_chain_10["avg"]["macro"]["f1"],
    }
    best_name = max(candidates, key=candidates.get)

    results = {
        "independent": res_indep,
        "chain_01": res_chain_01,
        "chain_10": res_chain_10,
        "scores_macro_f1": candidates,
        "chosen_by_macro_f1": best_name,
    }

    with open(Path(out_dir) / "cv_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== Cross-validation summary (macro F1) ===")
    for k, v in candidates.items():
        print(f"{k:25s}: {v:.4f}")
    print(f"Best model: {best_name}")

    return results


if __name__ == "__main__":
    summary = run_all()
    print("Results saved to artifacts/cv_results.json")