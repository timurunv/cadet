"""Simple evaluator for classification metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .classification_metrics import (
    calculate_classification_metrics,
    calculate_probability_metrics,
    plot_precision_recall_curve,
    plot_roc_curve,
)
from .evaluator import Evaluator


class SimpleEvaluator(Evaluator):
    """Simple evaluator that computes classification metrics.

    This evaluator loads predictions and computes basic classification
    metrics (accuracy, precision, recall, F1, AUC-ROC, AUPR).
    """

    def __init__(self, output_path: str | Path, metrics: list[str] | None = None):
        """Initialize simple evaluator.

        Args:
            output_path: Directory to save evaluation results
            metrics: List of metrics to compute (defaults to standard set)
        """
        super().__init__(output_path)
        self.metrics_to_compute = metrics or [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "auc_roc",
            "aupr",
        ]
        self.results: dict[str, float] | None = None

        # Initialize prediction arrays
        self.y_true: np.ndarray | None = None
        self.y_pred: np.ndarray | None = None
        self.y_proba: np.ndarray | None = None

        # Create required directories
        (self.output_path / "predictions").mkdir(parents=True, exist_ok=True)
        (self.output_path / "metrics").mkdir(parents=True, exist_ok=True)
        (self.output_path / "reports").mkdir(parents=True, exist_ok=True)

    def evaluate(self, predictions_file: str = "test_predictions.csv") -> dict[str, float]:
        """Evaluate predictions from CSV file and compute metrics.

        Args:
            predictions_file: Name of CSV file with predictions (default: test_predictions.csv)

        Returns:
            Dictionary containing evaluation metrics
        """
        # Load predictions using the dedicated method
        self.load_predictions(predictions_file)

        # Ensure predictions are loaded
        if self.y_true is None or self.y_pred is None or self.y_proba is None:
            raise ValueError("Predictions not loaded. Call load_predictions() first.")

        # Calculate classification metrics
        metrics = calculate_classification_metrics(self.y_true, self.y_pred)

        # Calculate probability-based metrics if available
        prob_metrics = calculate_probability_metrics(self.y_true, self.y_proba)
        metrics.update(prob_metrics)

        self.results = metrics

        # Save all results using the unified save method
        self.save_results()

        return metrics

    def load_predictions(self, filename: str | Path) -> None:
        """Load predictions from CSV file.

        Args:
            filename: Path to CSV predictions file or just filename if in predictions/ directory
        """
        # Handle both full paths and just filenames
        if isinstance(filename, str) and not filename.startswith("/"):
            csv_path = self.output_path / "predictions" / filename
        else:
            csv_path = Path(filename)

        if not csv_path.exists():
            raise FileNotFoundError(f"Predictions file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # Extract required columns and store as instance variables
        self.y_true = np.array(df["true_label"])
        self.y_pred = np.array(df["pred_label"])
        self.y_proba = np.array(df["prob"])

    def save_results(self) -> None:
        """Save metrics and plots to standardized locations."""
        if not self.results:
            raise ValueError("No results to save. Run evaluate() first.")

        self._save_metrics()
        self._generate_plots()

    def _save_metrics(self) -> None:
        """Helper method to save metrics to JSON file in metrics/ directory.

        Requires:
            self.results to be populated (not None).

        Raises:
            ValueError: If self.results is None.
        """
        output_file = self.output_path / "metrics" / "metrics.json"

        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"Metrics saved to {output_file}")

    def _generate_plots(self) -> None:
        """Generate and save visualization plots to reports/ directory."""
        if self.y_true is None or self.y_proba is None:
            print("Warning: Cannot generate plots - predictions not available")
            return

        try:
            # Ensure reports directory exists
            reports_path = self.output_path / "reports"
            reports_path.mkdir(parents=True, exist_ok=True)

            # Generate ROC curve
            roc_path = reports_path / "roc_curve.png"
            plot_roc_curve(
                self.y_true,
                self.y_proba,
                title="ROC Curve - Classification Performance",
                save_path=str(roc_path),
            )
            print(f"ROC curve saved to {roc_path}")

            # Generate Precision-Recall curve
            pr_path = reports_path / "precision_recall_curve.png"
            plot_precision_recall_curve(
                self.y_true,
                self.y_proba,
                title="Precision-Recall Curve - Classification Performance",
                save_path=str(pr_path),
            )
            print(f"Precision-Recall curve saved to {pr_path}")

        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")

    def generate_report(self) -> str:
        """Generate human-readable evaluation report.

        Returns:
            Formatted evaluation report string
        """
        if self.results is None:
            return "No metrics available. Run evaluate() first."

        report = "=== Evaluation Report ===\n\n"
        report += "Classification Metrics:\n"
        report += f"  Accuracy:  {self.results['accuracy']:.4f}\n"
        report += f"  Precision: {self.results['precision']:.4f}\n"
        report += f"  Recall:    {self.results['recall']:.4f}\n"
        report += f"  F1 Score:  {self.results['f1']:.4f}\n"

        if "auc_roc" in self.results:
            report += "\nProbability-based Metrics:\n"
            report += f"  AUC-ROC:   {self.results['auc_roc']:.4f}\n"
            report += f"  AUPR:      {self.results['aupr']:.4f}\n"

        report += "\nConfusion Matrix:\n"
        report += f"  True Negatives:  {self.results['tn']}\n"
        report += f"  False Positives: {self.results['fp']}\n"
        report += f"  False Negatives: {self.results['fn']}\n"
        report += f"  True Positives:  {self.results['tp']}\n"

        return report
