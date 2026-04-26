"""
Evaluation metrics for binary / multi-class classification.
"""

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

import torch


def compute_metrics(
    all_labels: list,
    all_preds:  list,
    all_probs:  list,
    num_classes: int = 2,
    class_names: list = None,
) -> Dict[str, float]:
    """
    Compute a suite of classification metrics.

    Parameters
    ----------
    all_labels  : ground-truth integer labels
    all_preds   : predicted integer labels (argmax)
    all_probs   : predicted probabilities for each class (list of arrays)
    num_classes : total number of classes
    class_names : optional display names

    Returns
    -------
    dict with keys: accuracy, f1, precision, recall, auc
    """
    labels = np.array(all_labels)
    preds  = np.array(all_preds)
    probs  = np.array(all_probs)   # (N, num_classes)

    acc  = accuracy_score(labels, preds)
    f1   = f1_score(labels, preds, average="weighted", zero_division=0)
    prec = precision_score(labels, preds, average="weighted", zero_division=0)
    rec  = recall_score(labels, preds, average="weighted", zero_division=0)

    # AUC – binary: use probability of positive class;
    #        multi-class: use OvR macro strategy
    try:
        if num_classes == 2:
            auc = roc_auc_score(labels, probs[:, 1])
        else:
            auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(labels, preds)

    return {
        "accuracy":  float(acc),
        "f1":        float(f1),
        "precision": float(prec),
        "recall":    float(rec),
        "auc":       float(auc),
        "confusion_matrix": cm,
    }


def print_metrics(metrics: Dict, split: str = "Test") -> None:
    """Pretty-print a metrics dict."""
    print(f"\n{'─'*45}")
    print(f"  {split} Metrics")
    print(f"{'─'*45}")
    for k, v in metrics.items():
        if k == "confusion_matrix":
            print(f"  Confusion Matrix:\n{v}")
        else:
            print(f"  {k:<12}: {v:.4f}")
    print(f"{'─'*45}\n")
