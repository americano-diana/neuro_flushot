from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import numpy as np
import time

def fit_model(model, X_train, y_train):
    """
    Fits a model on the training data and returns the trained model.
    """
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, average='binary'):
    """
    Evaluates a fitted model using several classification metrics.
    Returns a dictionary of metrics.
    """
    y_pred = model.predict(X_test)
    
    # In case the modelsupports predict_proba for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred  # fallback if no probability output

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_test, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_test, y_pred, average=average, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }
    return metrics

def run_tests(models, datasets, verbose=False):
    """
    Trains and evaluates multiple models on multiple datasets.
    Shows progress bars; optionally prints detailed logs.

    Args:
        models (dict): model_name -> sklearn model instance
        datasets (dict): dataset_name -> (X_train, X_test, y_train, y_test)
        verbose (bool): if True, print detailed training logs

    Returns:
        pd.DataFrame: evaluation results
    """
    results = []
    total_runs = len(models) * len(datasets)
    if verbose:
        print(f"\nStarting training for {total_runs} total combinations...\n")

    for ds_name, (X_train, X_test, y_train, y_test) in tqdm(
        datasets.items(), desc="Datasets", position=0
    ):
        if verbose:
            print(f"\n Dataset: {ds_name}")

        for model_name, model in tqdm(
            models.items(), desc=f"Training on {ds_name}", leave=False, position=1
        ):
            start_time = time.time()

            if verbose:
                print(f" Training {model_name} on {ds_name}...")

            trained_model = fit_model(model, X_train, y_train)
            metrics = evaluate_model(trained_model, X_test, y_test)
            elapsed = time.time() - start_time

            if verbose:
                print(f"✅ Finished {model_name} on {ds_name} in {elapsed:.2f}s")

            metrics.update({
                "dataset": ds_name,
                "model": model_name,
                "train_time_sec": round(elapsed, 2),
            })
            results.append(metrics)

    if verbose:
        print("\n Test of all models completed \n")

    return pd.DataFrame(results)