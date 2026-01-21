"""Main evaluation interface for CADET framework.

Pure abstract base with generic contracts and no assumptions about file types
or directory structure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Evaluator(ABC):
    """Abstract evaluator contract."""

    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)

    @abstractmethod
    def evaluate(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Run evaluation end-to-end: load predictions, compute metrics, save results."""
        raise NotImplementedError

    @abstractmethod
    def load_predictions(self, filename: str | Path) -> None:
        """Load model predictions from file.

        Args:
            filename: Predictions filename

        Returns:
            Loaded predictions dictionary
        """
        raise NotImplementedError

    @abstractmethod
    def save_results(self) -> None:
        """Save evaluation results to standardized locations.

        Saves all evaluation outputs to their appropriate directories:
        - Metrics to metrics/ directory
        - Reports/plots to reports/ directory
        """
        raise NotImplementedError

    @abstractmethod
    def generate_report(self) -> Any:
        """Generate human-readable evaluation report.

        Returns:
            Formatted evaluation report string
        """
        pass
