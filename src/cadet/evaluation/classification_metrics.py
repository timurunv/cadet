"""
Classification-based metrics for hate speech detection.

This module provides comprehensive classification metrics including accuracy, precision,
recall, F1-score, AUC-ROC, AUPR, confusion matrix, and classification reports.
Also includes plotting functions for visualizing results.
"""

import logging
import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """
    Calculate basic classification metrics for binary classification.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary containing basic classification metrics
    """
    # Basic classification metrics (binary classification)
    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel().tolist()

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }

    return metrics


def calculate_probability_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:
    """
    Calculate probability-based metrics for binary classification.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities

    Returns:
        Dictionary containing AUC-ROC and AUPR metrics
    """
    try:
        # Handle probability format for binary classification
        if y_proba.ndim == 2 and y_proba.shape[1] == 2:
            # Use positive class probabilities
            y_proba_pos = y_proba[:, 1]
        elif y_proba.ndim == 1:
            # Already positive class probabilities
            y_proba_pos = y_proba
        else:
            y_proba_pos = y_proba

        # AUC-ROC and AUPR for binary classification
        metrics = {
            "auc_roc": float(roc_auc_score(y_true, y_proba_pos)),
            "aupr": float(average_precision_score(y_true, y_proba_pos)),
        }

    except Exception as e:
        logger.warning("Could not calculate probability-based metrics: %s", e)
        metrics = {
            "auc_roc": 0.0,
            "aupr": 0.0,
        }

    return metrics


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "ROC Curve",
    figsize: tuple[int, int] = (8, 6),
    save_path: str | None = None,
) -> Any:
    """
    Plot ROC curve for binary classification.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Path to save the plot

    Returns:
        Matplotlib figure object
    """
    # Handle probability format
    if y_proba.ndim == 2:
        y_proba = y_proba[:, 1]  # Use positive class probabilities

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc_score:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random classifier")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "Precision-Recall Curve",
    figsize: tuple[int, int] = (8, 6),
    save_path: str | None = None,
) -> Any:
    """
    Plot Precision-Recall curve for binary classification.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Path to save the plot

    Returns:
        Matplotlib figure object
    """
    # Handle probability format
    if y_proba.ndim == 2:
        y_proba = y_proba[:, 1]  # Use positive class probabilities

    # Calculate Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    aupr_score = average_precision_score(y_true, y_proba)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        recall, precision, color="darkorange", lw=2, label=f"PR curve (AUPR = {aupr_score:.3f})"
    )

    # Add baseline (random classifier)
    positive_ratio = float(np.mean(y_true))
    ax.axhline(
        y=positive_ratio,
        color="navy",
        linestyle="--",
        lw=2,
        label=f"Random classifier (AP = {positive_ratio:.3f})",
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
