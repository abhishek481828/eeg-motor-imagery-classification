"""Evaluation metrics computation for EEG classification."""

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str] | None = None
) -> dict[str, Any]:
    """Compute comprehensive classification performance metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred).tolist()

    # Per-class metrics
    per_class_p, per_class_r, per_class_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    per_class = {}
    for i, f1 in enumerate(per_class_f1):
        name = class_names[i] if class_names and i < len(class_names) else f"class_{i}"
        per_class[name] = {
            "precision": float(per_class_p[i]),
            "recall": float(per_class_r[i]),
            "f1_score": float(f1),
        }

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "cohens_kappa": kappa,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "confusion_matrix": cm,
    }
