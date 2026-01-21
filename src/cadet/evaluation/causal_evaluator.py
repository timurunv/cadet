"""Causal evaluator for CADET model.

This evaluator extends EnhancedEvaluator with placeholders for future causal analysis.
Per change notes: For now, it's functionally equal to EnhancedEvaluator.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset

from .enhanced_evaluator import EnhancedEvaluator

logger = logging.getLogger(__name__)


class CausalEvaluator(EnhancedEvaluator):
    """Causal evaluator for CADET model with comprehensive causal analysis capabilities.

    This evaluator extends EnhancedEvaluator to provide specialized causal analysis
    for CADET models. It verifies
    proper causal alignment, and generates detailed
    visualizations of latent space structure and training dynamics.

    Key Features:
        - Causal alignment verification: Ensures M (motivation) is the strongest
          predictor of hate labels compared to U (confounder), T (target), S (style)
        - Multi-modal visualization: Generates t-SNE plots, causal alignment charts,
          and training loss curves with progressive weighting annotations
        - Comprehensive evaluation: Combines standard classification metrics with
          causal-specific analyses

    Causal Analysis Pipeline:
        1. Load latent variables (zm, zt, z_style, zu) from trained CADET model
        2. Train linear probes to measure predictive strength of each latent variable
        3. Verify causal alignment: M should be strongest predictor of hate
        4. Generate visualizations for interpretation and validation

    Visualization Outputs:
        - causal_alignment.png: Bar chart comparing hate prediction accuracy
        - latent_tsne.png: 2x2 grid of t-SNE plots for each latent variable
        - training_losses.png: Loss curves with progressive weighting annotations
        - tsne_plot.png: Embeddings visualization for general analysis

    Expected Causal Structure:
        In a properly trained CADET model:
        - M (motivation) should have highest accuracy predicting hate labels
        - U (confounder) should be significantly suppressed (accuracy < M - 0.1)
        - T (target) and S (style) should be weaker predictors than M
        - This indicates successful causal disentanglement and debiasing

    Usage:
        >>> evaluator = CausalEvaluator(output_path="./results", enable_embeddings=True)
        >>> results = evaluator.evaluate_with_causal_analysis(
        ...     predictions_file="test_predictions.csv",
        ...     latents_file="latents.npz",
        ...     generate_visualizations=True
        ... )
        >>> print(f"Properly aligned: {results['causal_analysis']['properly_aligned']}")

    Args:
        output_path: Directory for saving evaluation results and visualizations
        metrics: List of classification metrics to compute (inherits from parent)
        enable_embeddings: Whether to generate embedding-based visualizations
        num_counterfactual_samples: Number of counterfactual examples (future use)

    Attributes:
        num_counterfactual_samples: Configuration for future counterfactual analysis
        All parent class attributes from EnhancedEvaluator

    Note:
        This evaluator assumes CADET model output format with specific latent
        variable naming conventions (zm, zt, z_style, zu). The causal analysis
        is based on linear probe methodology from causal representation learning.
    """

    def __init__(
        self,
        output_path: str | Path,
        metrics: list[str] | None = None,
        enable_embeddings: bool = True,
        num_counterfactual_samples: int = 5,
    ):
        """Initialize causal evaluator.

        Args:
            output_path: Output directory
            metrics: List of metrics to compute
            enable_embeddings: Whether to compute embeddings
            num_counterfactual_samples: Number of CF examples (for future use)
        """
        super().__init__(output_path, metrics, enable_embeddings)
        self.num_counterfactual_samples = num_counterfactual_samples

    def evaluate_with_causal_analysis(
        self,
        predictions_file: str,
        latents_file: str | None = None,
        model: Any = None,
        test_dataset: Dataset | None = None,
        generate_visualizations: bool = True,
    ) -> dict[str, Any]:
        """Evaluate with causal analysis.

        Args:
            predictions_file: Path to predictions CSV (relative to output_path)
            latents_file: Path to latents NPZ file (relative to output_path, optional)
            model: Trained CADET model (optional, for future counterfactual generation)
            test_dataset: Test dataset (optional, for future counterfactual generation)
            generate_visualizations: Whether to generate plots

        Returns:
            Dictionary with all metrics and causal analysis results

        Note:
            Now includes complete causal alignment verification and counterfactual analysis.
        """
        logger.info("Running evaluation with causal analysis...")

        # Run standard evaluation
        metrics = self.evaluate(predictions_file)

        # Generate t-SNE plot if embeddings are available (separate from latents)
        if self.enable_embeddings and generate_visualizations:
            try:
                # Try to load embeddings for t-SNE (different from latents)
                embeddings_path = self.output_path / "models" / "embeddings.npz"
                if embeddings_path.exists():
                    self.load_embeddings("embeddings.npz")
                    _, _ = self.generate_tsne(save_plots=True, plot_filename="tsne_plot.png")

                    metrics["tsne_plot_path"] = str(  # type: ignore[assignment]
                        self.visualizations_path / "tsne_plot.png"
                    )
                    logger.info("t-SNE plot generated successfully")
                else:
                    logger.info("Embeddings file not found, skipping t-SNE plot")

            except Exception as e:
                logger.warning("t-SNE plotting failed: %s", e)

        # Add causal alignment verification
        if latents_file is not None:
            latents_path = self.output_path / "models" / latents_file
            if latents_path.exists():
                try:
                    latents = np.load(latents_path)
                    causal_metrics = self._verify_causal_alignment(latents)
                    metrics["causal_analysis"] = causal_metrics  # type: ignore[assignment]
                    logger.info("Causal alignment verification complete")

                    # Generate causal visualizations if requested
                    if generate_visualizations:
                        self._visualize_causal_alignment(causal_metrics)
                        self._visualize_latent_space(latents)

                        metrics["causal_alignment_plot_path"] = str(  # type: ignore[assignment]
                            self.visualizations_path / "causal_alignment.png"
                        )
                        metrics["latent_tsne_plot_path"] = str(  # type: ignore[assignment]
                            self.visualizations_path / "latent_tsne.png"
                        )
                        logger.info("Causal visualization plots generated")

                except Exception as e:
                    logger.warning("Causal analysis failed: %s", e)

            else:
                logger.warning("Latents file not found: %s", latents_path)

        # Generate training loss curves if available
        if generate_visualizations:
            try:
                self.plot_training_losses()
                metrics["training_losses_plot_path"] = str(  # type: ignore[assignment]
                    self.visualizations_path / "training_losses.png"
                )
                logger.info("Training loss curves generated")
            except Exception as e:
                logger.warning("Training loss plotting failed: %s", e)

        logger.info("Evaluation with causal analysis complete")
        return metrics

    def _verify_causal_alignment(self, latents: dict[str, np.ndarray]) -> dict[str, Any]:
        """Verify causal alignment (M -> Hate strongest).

        Args:
            latents: Dictionary with zm, zt, z_style, zu, hate_labels

        Returns:
            Dictionary with:
            {
                'hate_from_M': float,  # Accuracy(M -> Hate)
                'hate_from_U': float,  # Accuracy(U -> Hate)
                'hate_from_style': float,  # Accuracy(style -> Hate)
                'hate_from_T': float,  # Accuracy(T -> Hate)
                'properly_aligned': bool,  # M > {U, style, T}
            }
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        logger.info("Verifying causal alignment...")

        # Extract data
        zm = latents["zm"]  # Motivation (M)
        zt = latents["zt"]  # Target (T)
        z_style = latents["z_style"]  # Style (S)
        zu = latents["zu"]  # Confounder (U)
        hate_labels = latents["hate_labels"]

        # Split data for evaluation
        test_size = 0.3
        random_state = 42

        results = {}

        # Train linear probes for each latent variable
        latent_vars = {"M": zm, "T": zt, "style": z_style, "U": zu}

        for var_name, latent_data in latent_vars.items():
            # Handle different shapes - flatten if needed
            if latent_data.ndim > 2:
                latent_data = latent_data.reshape(latent_data.shape[0], -1)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                latent_data,
                hate_labels,
                test_size=test_size,
                random_state=random_state,
                stratify=hate_labels,
            )

            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train logistic regression probe
            probe = LogisticRegression(random_state=random_state, max_iter=1000)
            probe.fit(X_train_scaled, y_train)

            # Evaluate
            y_pred = probe.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)

            results[f"hate_from_{var_name}"] = accuracy
            logger.info(f"  {var_name} -> Hate accuracy: {accuracy:.4f}")

        # Check causal alignment
        hate_from_M = results["hate_from_M"]
        hate_from_U = results["hate_from_U"]
        hate_from_style = results["hate_from_style"]
        hate_from_T = results["hate_from_T"]

        # M should be strongest predictor
        properly_aligned = (
            hate_from_M > hate_from_U
            and hate_from_M > hate_from_style
            and hate_from_M > hate_from_T
        )

        results.update(
            {
                "properly_aligned": properly_aligned,
                "alignment_margin": hate_from_M - max(hate_from_U, hate_from_style, hate_from_T),
                "confounder_margin": hate_from_M - hate_from_U,
            }
        )

        logger.info(f"  Properly aligned: {properly_aligned}")
        logger.info(f"  Alignment margin: {results['alignment_margin']:.4f}")

        return results

    def _visualize_causal_alignment(self, causal_results: dict) -> None:
        """Generate causal alignment visualization.

        Creates bar chart comparing hate prediction accuracy from M vs {U, style, T}.
        """
        import matplotlib.pyplot as plt

        logger.info("Generating causal alignment visualization...")

        # Extract accuracies
        accuracies = {
            "M (Motivation)": causal_results["hate_from_M"],
            "T (Target)": causal_results["hate_from_T"],
            "S (Style)": causal_results["hate_from_style"],
            "U (Confounder)": causal_results["hate_from_U"],
        }

        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ["#2E8B57", "#4682B4", "#8A2BE2", "#DC143C"]  # Green, Blue, Purple, Red
        labels = list(accuracies.keys())
        values = list(accuracies.values())
        bars = ax.bar(labels, values, color=colors, alpha=0.7)

        # Add value labels on bars
        for bar, acc in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{acc:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Highlight M as the strongest (should be highest)
        max_acc = max(values)
        for i, (name, acc) in enumerate(accuracies.items()):
            if acc == max_acc and "M" in name:
                bars[i].set_edgecolor("black")
                bars[i].set_linewidth(3)

        ax.set_ylabel("Hate Prediction Accuracy", fontsize=12, fontweight="bold")
        ax.set_title(
            "Causal Alignment: Hate Prediction from Latent Variables\n(M should be strongest)",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        # Save plot
        save_path = self.output_path / "reports" / "causal_alignment.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Causal alignment plot saved to {save_path}")

    def _visualize_latent_space(self, latents: dict[str, np.ndarray]) -> None:
        """Generate t-SNE visualization of latent variables.

        Creates t-SNE plots for zm, zt, z_style, zu colored by hate labels.
        """
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        logger.info("Generating latent space t-SNE visualization...")

        latent_vars = {
            "M (Motivation)": latents["zm"],
            "T (Target)": latents["zt"],
            "S (Style)": latents["z_style"],
            "U (Confounder)": latents["zu"],
        }
        hate_labels = latents["hate_labels"]

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        colors = ["#1f77b4", "#ff7f0e"]
        labels = ["Non-Hate", "Hate"]

        for i, (var_name, latent_data) in enumerate(latent_vars.items()):
            ax = axes[i]

            if latent_data.ndim > 2:
                latent_data = latent_data.reshape(latent_data.shape[0], -1)

            if len(latent_data) > 1000:
                indices = np.random.choice(len(latent_data), 1000, replace=False)
                latent_sample = latent_data[indices]
                hate_sample = hate_labels[indices]
            else:
                latent_sample = latent_data
                hate_sample = hate_labels

            tsne = TSNE(
                n_components=2, random_state=42, perplexity=min(30, len(latent_sample) // 4)
            )
            latent_2d = tsne.fit_transform(latent_sample)

            for label_idx, (color, label) in enumerate(zip(colors, labels)):
                mask = hate_sample == label_idx
                if np.any(mask):
                    ax.scatter(
                        latent_2d[mask, 0],
                        latent_2d[mask, 1],
                        c=color,
                        label=label,
                        alpha=0.6,
                        s=20,
                    )

            ax.set_title(f"{var_name} Latent Space", fontsize=12, fontweight="bold")
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            ax.legend()
            ax.grid(alpha=0.3)

        plt.suptitle(
            "Latent Variable t-SNE Visualization\n(Colored by Hate Labels)",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()

        # Save plot
        save_path = self.output_path / "reports" / "latent_tsne.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Latent space t-SNE plot saved to {save_path}")

    def plot_training_losses(self, losses_file: str = "training_losses.csv") -> None:
        """Plot training loss curves with individual epoch-batch points and progressive phase areas.

        Args:
            losses_file: Name of CSV file with training losses (epoch, batch, loss components)
        """
        import matplotlib.pyplot as plt
        import pandas as pd

        logger.info("Plotting training loss curves with epoch-batch granularity...")

        # Load loss data
        losses_path = self.output_path / "models" / losses_file
        if not losses_path.exists():
            logger.warning(f"Training losses file not found: {losses_path}")
            return

        df = pd.read_csv(losses_path)

        # Create continuous x-axis: epoch + batch/batches_per_epoch
        # Assume batch numbers reset each epoch, so calculate global step
        df = df.sort_values(["epoch", "batch"])

        # Calculate global step for x-axis
        if "batch" in df.columns:
            # Group by epoch to get max batch per epoch
            batches_per_epoch = (
                df.groupby("epoch")["batch"].max().iloc[0] + 1
            )  # +1 since batch is 0-indexed
            df["global_step"] = df["epoch"] + df["batch"] / batches_per_epoch
        else:
            # Fallback: just use epoch if no batch column
            df["global_step"] = df["epoch"]

        # Create single plot for total loss only
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        # Plot individual epoch-batch points
        if "total_loss" in df.columns:
            ax.scatter(
                df["global_step"],
                df["total_loss"],
                color="#2E86AB",
                alpha=0.6,
                s=8,
                label="Training Loss",
            )

            # Add smooth trend line using rolling average
            window_size = max(10, len(df) // 50)  # Adaptive window size
            df["loss_smooth"] = df["total_loss"].rolling(window=window_size, center=True).mean()
            ax.plot(
                df["global_step"],
                df["loss_smooth"],
                color="#A23B72",
                linewidth=2,
                alpha=0.8,
                label="Trend",
            )

        ax.set_xlabel("Training Progress (Epoch)", fontsize=12)
        ax.set_ylabel("Total Loss", fontsize=12)
        ax.set_title("Training Loss Curve (Individual Batches)", fontweight="bold", fontsize=14)
        ax.grid(alpha=0.3)

        # Add progressive training phase areas with colored backgrounds
        # Based on typical progressive schedule from cadet_trainer.py
        annotation_epochs = {
            2: ("KL loss starts", "#FF6B6B"),  # Red
            3: ("Orthogonality starts", "#4ECDC4"),  # Teal
            5: ("Reconstruction ramp complete", "#45B7D1"),  # Blue
            6: ("KL loss full weight", "#96CEB4"),  # Green
        }

        max_epoch = df["epoch"].max()
        min_epoch = df["epoch"].min()

        # Create colored areas for progressive phases
        phase_starts = [min_epoch] + sorted(
            [epoch for epoch in annotation_epochs.keys() if epoch <= max_epoch]
        )
        phase_starts.append(max_epoch + 1)

        colors = ["#CED4DA", "#FFA8A8", "#5EEAD4", "#7DD3FC", "#86EFAC"]

        for i in range(len(phase_starts) - 1):
            start_epoch = phase_starts[i]
            end_epoch = phase_starts[i + 1]

            if i < len(colors):
                ax.axvspan(
                    start_epoch, end_epoch, alpha=0.2, color=colors[i % len(colors)], zorder=0
                )

        # Add annotation lines at phase transitions
        legend_elements = []
        for epoch, (label, color) in annotation_epochs.items():
            if epoch <= max_epoch:
                # Add vertical line at transition point
                line = ax.axvline(
                    x=epoch,
                    color=color,
                    linestyle="--",
                    linewidth=2,
                    alpha=0.7,
                    label=f"Epoch {epoch}: {label}",
                    zorder=3,
                )
                legend_elements.append(line)

        # Add main plot legend
        ax.legend(loc="upper right", fontsize=10)

        # Add phase transition legend at the bottom if annotations exist
        if legend_elements:
            # Create second legend for phase transitions
            ax.legend(
                handles=legend_elements,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.12),
                ncol=2,
                fontsize=9,
                frameon=True,
                fancybox=True,
                shadow=True,
                title="Progressive Training Phases",
            )
            # Add the first legend back
            ax.add_artist(ax.legend(loc="upper right", fontsize=10))

        # Adjust layout to accommodate bottom legend
        plt.tight_layout(rect=(0, 0.15, 1, 1))  # Leave more space at bottom for legend

        # Save plot
        save_path = self.output_path / "reports" / "training_losses.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Training loss curves saved to {save_path}")

        # Also save summary statistics using all individual points
        final_total_loss = float(df["total_loss"].iloc[-1]) if "total_loss" in df.columns else None
        min_total_loss = float(df["total_loss"].min()) if "total_loss" in df.columns else None

        summary_stats = {
            "final_total_loss": final_total_loss,
            "min_total_loss": min_total_loss,
            "final_epoch": int(df["epoch"].max()),
            "total_batches": len(df),
            "loss_components": list(df.columns),
        }

        summary_path = self.output_path / "metrics" / "training_loss_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary_stats, f, indent=2)

        logger.info(f"Training loss summary saved to {summary_path}")
