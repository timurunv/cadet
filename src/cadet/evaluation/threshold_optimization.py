"""
Threshold optimization for hate speech detection models.

This module provides tools for finding optimal decision thresholds and evaluating
model performance across multiple thresholds for binary classification tasks.
"""

import logging

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


def optimize_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1_macro",
    thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
    """
    Find optimal threshold for binary classification.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        metric: Metric to optimize ('f1_macro', 'f1', 'precision', 'recall', 'accuracy')
        thresholds: Array of thresholds to test (default: 0.05 to 0.95 in 19 steps)

    Returns:
        Tuple of (best_threshold, best_score)
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    # Extract positive class probabilities if 2D
    if y_proba.ndim == 2:
        y_proba_pos = y_proba[:, 1]
    else:
        y_proba_pos = y_proba

    best_threshold = 0.5
    best_score = 0.0

    for threshold in thresholds:
        y_pred = (y_proba_pos >= threshold).astype(int)

        if metric == "f1_macro":
            score = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        elif metric == "f1":
            score = float(f1_score(y_true, y_pred, average="binary", zero_division=0))
        elif metric == "precision":
            score = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        elif metric == "recall":
            score = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        elif metric == "accuracy":
            score = float(accuracy_score(y_true, y_pred))
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score
