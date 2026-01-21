"""Evaluation modules for model performance assessment."""

from .causal_evaluator import CausalEvaluator
from .classification_metrics import (
    calculate_classification_metrics,
    calculate_probability_metrics,
    plot_precision_recall_curve,
    plot_roc_curve,
)
from .enhanced_evaluator import EnhancedEvaluator
from .evaluator import Evaluator
from .simple_evaluator import SimpleEvaluator
from .threshold_optimization import (
    optimize_threshold,
)
from .tsne_visualization import (
    plot_tsne_visualization,
    prepare_tsne_embeddings,
)

__all__ = [
    # Evaluators
    "Evaluator",
    "SimpleEvaluator",
    "EnhancedEvaluator",
    "CausalEvaluator",
    # Classification metrics
    "calculate_classification_metrics",
    "calculate_probability_metrics",
    "plot_roc_curve",
    "plot_precision_recall_curve",
    # Threshold optimization
    "optimize_threshold",
    # t-SNE visualization
    "prepare_tsne_embeddings",
    "plot_tsne_visualization",
]
