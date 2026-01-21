"""Base trainer interface for CADET framework.

Pure abstract orchestrator for model training/inference. Keeps the interface
minimal and format-agnostic.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from cadet.utils.device import get_device, set_random_seed

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Abstract base trainer for all model types in CADET framework.

    Provides a minimal interface while allowing maximum flexibility for
    different model architectures and training requirements.
    """

    def __init__(
        self,
        source_style: str,
        target_style: str,
        model_config: dict[str, Any],
        output_path: Path,
        device: str | None = None,
        random_seed: int | None = None,
    ):
        """Initialize base trainer.

        Args:
            source_style: Training data style ('explicit' or 'implicit')
            target_style: Testing data style ('explicit' or 'implicit')
            model_config: Model-specific configuration and hyperparameters
            output_path: Directory to save all experiment outputs
            device: Device for computation ('cpu', 'cuda', 'auto', or None for auto).
            random_seed: Random seed for reproducibility

        Note:
            This signature is designed for PLM-based models that require training.
            For inference-only models (e.g., LLM Guard), source_style may be set
            to a placeholder value since no training occurs.
        """
        self.source_style = source_style
        self.target_style = target_style
        self.model_config = model_config
        self.output_path = Path(output_path)

        # Device and seed management
        self._device = get_device(device)
        logger.info(f"Trainer using device: {self._device}")

        if random_seed is not None:
            self._random_seed = set_random_seed(random_seed)
        else:
            self._random_seed = None

        # Create output root directory (subdirs are up to concrete trainers)
        self.output_path.mkdir(parents=True, exist_ok=True)
        # Initialize model (to be implemented by subclasses)
        self.model = None

    @property
    def device(self) -> str:
        """Get the device this trainer uses."""
        return self._device

    @property
    def random_seed(self) -> int | None:
        """Get the random seed used for this trainer."""
        return self._random_seed

    @abstractmethod
    def load_model(self) -> Any:
        """Load and initialize the model (and optionally restore weights).

        Args:
            checkpoint_path: Optional path to a checkpoint to restore. If None,
                subclasses may choose a sensible default (e.g., best.pt).
            strict: Whether to enforce that the keys in state_dict match the
                model's keys exactly.

        Returns:
            Initialized (and possibly restored) model instance
        """
        pass

    @abstractmethod
    def load_data(self) -> Any:
        """Load training, validation, and test datasets.

        Returns:
            Tuple of (train_data, val_data, test_data) in model-appropriate format
        """
        pass

    @abstractmethod
    def train(self) -> Any:
        """Train the model end-to-end: load data, train, validate, save checkpoints."""
        raise NotImplementedError

    @abstractmethod
    def inference(self) -> Any:
        """Run inference end-to-end: load model, run predictions, save outputs."""
        raise NotImplementedError
