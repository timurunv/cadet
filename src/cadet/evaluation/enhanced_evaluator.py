"""Enhanced evaluator: SimpleEvaluator + optional t-SNE plot.

This class builds on SimpleEvaluator (classification metrics + ROC/PR plots)
and adds a minimal ability to load precomputed embeddings and generate a
single t-SNE visualization saved under reports/.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from .simple_evaluator import SimpleEvaluator
from .tsne_visualization import plot_tsne_visualization, prepare_tsne_embeddings

logger = logging.getLogger(__name__)


class EnhancedEvaluator(SimpleEvaluator):
    """Enhanced evaluator with optional t-SNE visualization.

    Extends SimpleEvaluator to support:
    - Loading embeddings
    - Generating a t-SNE plot saved to reports/
    """

    def __init__(
        self,
        output_path: str | Path,
        metrics: list[str] | None = None,
        enable_embeddings: bool = True,
    ):
        """Initialize enhanced evaluator.

        Args:
            output_path: Directory to save outputs
            metrics: List of metrics to compute (inherited from SimpleEvaluator)
            enable_embeddings: Whether to enable embedding-related functionality
        """
        super().__init__(output_path=output_path, metrics=metrics)

        self.enable_embeddings = enable_embeddings
        self.embeddings_data = None
        self.embeddings_metadata = None

        # Paths matching project conventions
        self.embeddings_path = self.output_path / "metrics"
        self.visualizations_path = self.output_path / "reports"

        # SimpleEvaluator already ensures directories exist, but keep safe
        if self.enable_embeddings:
            self.visualizations_path.mkdir(parents=True, exist_ok=True)

    def load_embeddings(self, filename: str = "embeddings.npz") -> dict[str, Any]:
        """Load saved embeddings from file.

        Args:
            filename: Name of the embeddings file to load

        Returns:
            Dictionary containing embeddings and metadata

        Raises:
            FileNotFoundError: If embeddings file doesn't exist
            ValueError: If embeddings are disabled
        """
        if not self.enable_embeddings:
            raise ValueError("Embeddings are disabled for this evaluator")

        embeddings_file = self.output_path / "models" / filename

        if not embeddings_file.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")

        logger.info("Loading embeddings from %s", embeddings_file)

        # Load embeddings data
        embeddings_data = np.load(embeddings_file)

        self.embeddings_data = {
            "embeddings": embeddings_data["embeddings"],
            "labels": embeddings_data["labels"],
            "text_ids": embeddings_data.get("text_ids", None),
            "style_labels": embeddings_data.get("style_labels", None),
        }

        # Store metadata
        self.embeddings_metadata = {
            "shape": self.embeddings_data["embeddings"].shape,
            "num_samples": len(self.embeddings_data["embeddings"]),
            "embedding_dim": self.embeddings_data["embeddings"].shape[1],
            "num_classes": len(np.unique(self.embeddings_data["labels"])),
        }

        logger.info("Loaded embeddings: %s", self.embeddings_metadata)

        return self.embeddings_data

    def generate_tsne(
        self,
        perplexity: int = 30,
        n_iter: int = 1000,
        random_state: int = 42,
        save_plots: bool = True,
        plot_filename: str = "tsne_plot.png",
    ) -> tuple[np.ndarray, Figure | None]:
        """Generate a t-SNE visualization for loaded embeddings.

        Args:
            perplexity: t-SNE perplexity parameter
            n_iter: Number of t-SNE iterations
            random_state: Random seed for reproducibility
            save_plots: Whether to save the t-SNE plot image to disk
            plot_filename: Filename for the saved t-SNE plot

        Returns:
            Tuple of (tsne_embeddings, matplotlib_figure).
            Figure is None if save_plots=True (automatically closed to prevent memory leaks).

        Raises:
            ValueError: If embeddings not loaded or disabled
        """
        if not self.enable_embeddings:
            raise ValueError("Embeddings are disabled for this evaluator")

        if self.embeddings_data is None:
            raise ValueError("No embeddings loaded. Call load_embeddings() first.")

        logger.info("Generating t-SNE visualization...")

        # Prepare t-SNE embeddings
        tsne_embeddings, labels = prepare_tsne_embeddings(
            self.embeddings_data["embeddings"],
            self.embeddings_data["labels"],
            perplexity=perplexity,
            n_iter=n_iter,
            random_state=random_state,
        )

        # Create visualization
        fig = plot_tsne_visualization(
            tsne_embeddings,
            labels,
            label_names=["Non-Hate", "Hate"],
            title="t-SNE Visualization of Model Embeddings",
            save_path=str(self.visualizations_path / plot_filename) if save_plots else None,
        )

        logger.info("t-SNE visualization complete")
        if not save_plots and fig is not None:
            logger.info(
                "Figure returned - remember to close manually with plt.close(fig) "
                "to prevent memory leaks"
            )

        return tsne_embeddings, fig

    def evaluate_with_embeddings(
        self,
        predictions_file: str = "test_predictions.csv",
        embeddings_file: str = "embeddings.npz",
        generate_visualizations: bool = True,
    ) -> dict[str, Any]:
        """Run evaluation and optionally add a t-SNE plot.

        Args:
            predictions_file: Name of predictions CSV file
            embeddings_file: Name of embeddings NPZ file
            generate_visualizations: Whether to generate and save the t-SNE plot

        Returns:
            Dictionary with complete evaluation results
        """
        logger.info("Running evaluation with embeddings analysis...")

        # Run standard evaluation
        standard_results = self.evaluate(predictions_file)

        if not self.enable_embeddings or not generate_visualizations:
            if not self.enable_embeddings:
                logger.info("Embeddings disabled, returning classification results only")
            return standard_results

        try:
            # Load embeddings and generate a single t-SNE plot
            self.load_embeddings(embeddings_file)
            _, _ = self.generate_tsne(save_plots=True, plot_filename="tsne_plot.png")

            combined_results = {
                **standard_results,
                "tsne_plot_path": str(self.visualizations_path / "tsne_plot.png"),
            }

            logger.info("Evaluation with t-SNE plot complete")
            return combined_results

        except Exception as e:
            logger.warning("t-SNE plotting failed: %s", e)
            logger.warning("Returning classification results only")
            return standard_results

    def __repr__(self) -> str:
        return (
            f"EnhancedEvaluator(output_path={self.output_path}, "
            f"enable_embeddings={self.enable_embeddings}, "
            f"embeddings_loaded={self.embeddings_data is not None})"
        )
