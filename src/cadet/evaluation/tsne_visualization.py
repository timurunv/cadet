"""
t-SNE visualization for hate speech detection embeddings.

This module provides tools for visualizing high-dimensional embeddings using t-SNE,
including quality metrics and plotting functions for representation analysis.
"""

from __future__ import annotations

import logging

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)


def prepare_tsne_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    perplexity: int = 30,
    n_iter: int = 1000,
    random_state: int = 42,
    use_pca: bool = True,
    pca_components: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare t-SNE visualization of embeddings.

    Args:
        embeddings: High-dimensional embeddings (n_samples, n_features)
        labels: Labels for coloring points
        perplexity: t-SNE perplexity parameter
        n_iter: Number of t-SNE iterations
        random_state: Random seed for reproducibility
        use_pca: Whether to apply PCA preprocessing
        pca_components: Number of PCA components if use_pca=True

    Returns:
        Tuple of (tsne_embeddings, processed_labels)
    """
    logger.info("Preparing t-SNE embeddings with perplexity=%d, n_iter=%d", perplexity, n_iter)

    # Apply PCA preprocessing if requested and embeddings are high-dimensional
    if use_pca and embeddings.shape[1] > pca_components:
        logger.info(
            "Applying PCA preprocessing: %d -> %d dimensions", embeddings.shape[1], pca_components
        )
        pca = PCA(n_components=pca_components, random_state=random_state)
        embeddings_processed = pca.fit_transform(embeddings)
        logger.info("PCA explained variance ratio: %.3f", pca.explained_variance_ratio_.sum())
    else:
        embeddings_processed = embeddings

    # Apply t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,  # Changed from n_iter to max_iter for sklearn compatibility
        random_state=random_state,
        verbose=1,
    )

    tsne_embeddings = tsne.fit_transform(embeddings_processed)

    logger.info("t-SNE completed. Final KL divergence: %.3f", tsne.kl_divergence_)

    return tsne_embeddings, labels


def plot_tsne_visualization(
    tsne_embeddings: np.ndarray,
    labels: np.ndarray,
    label_names: list[str] | None = None,
    title: str = "t-SNE Visualization",
    figsize: tuple[int, int] = (10, 8),
    alpha: float = 0.6,
    s: int = 50,
    save_path: str | None = None,
) -> Figure | None:
    """
    Plot t-SNE visualization with colored points.

    Args:
        tsne_embeddings: 2D t-SNE embeddings
        labels: Labels for coloring points
        label_names: Names for the labels (for legend)
        title: Plot title
        figsize: Figure size (width, height)
        alpha: Point transparency
        s: Point size
        save_path: Path to save the plot

    Returns:
        Matplotlib figure object if save_path is None, otherwise None
        (figure is automatically closed when saved to prevent memory leaks)
    """
    fig, ax = plt.subplots(figsize=figsize)

    unique_labels = np.unique(labels)
    # Use classic blue and orange for binary classification
    if len(unique_labels) == 2:
        colors = ["#1f77b4", "#ff7f0e"]  # Classic matplotlib blue and orange
    else:
        colors = mpl.colormaps["tab10"](np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        label_name = label_names[i] if label_names else f"Class {label}"

        ax.scatter(
            tsne_embeddings[mask, 0],
            tsne_embeddings[mask, 1],
            c=colors[i] if isinstance(colors, list) else [colors[i]],
            label=label_name,
            alpha=alpha,
            s=s,
        )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        # Close figure if saved to disk to prevent memory accumulation
        plt.close(fig)
        return None

    return fig
