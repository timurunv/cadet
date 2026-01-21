"""Minimal base model skeleton for CADET.

Pure abstract base designed to support both trainable PLMs and evaluation-only
models (e.g., LLM-guard). It only enforces common identifiers and abstract
methods; subclasses decide training/inference specifics.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from cadet.utils.device import get_device, set_random_seed

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "llamaguard": "meta-llama/Llama-Guard-3-8B",
    "promptguard": "meta-llama/Prompt-Guard-86M",
    "shieldgemma": "google/shieldgemma-2b",
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
    "distilbert": "distilbert-base-uncased",
    "bart": "facebook/bart-base",
    "cadet": "cadet",
}


class BaseModel(ABC):
    """Abstract base for all models."""

    def __init__(
        self,
        model_name: str,
        model_id: str | None = None,
        device: str | None = None,
        random_seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize base model.

        Args:
            model_name: Short-form model name to select implementation/class.
            model_id: Concrete identifier (HF hub or local checkpoint path).
            device: Device for computation ('cpu', 'cuda', 'auto', or None for auto).
            random_seed: Random seed for reproducibility (None for random).
            **kwargs: Extra parameters per implementation.
        """
        self.model_name = model_name
        self.model_id = model_id or MODEL_REGISTRY.get(model_name, None)
        self.kwargs: dict[str, Any] = kwargs
        assert self.model_id is not None

        # Device and seed management
        self._device = get_device(device)
        logger.info(f"{self.model_name} using device: {self._device}")

        if random_seed is not None:
            self._random_seed = set_random_seed(random_seed)
        else:
            self._random_seed = None

    @property
    def device(self) -> str:
        """Get the device this model uses."""
        return self._device

    @property
    def random_seed(self) -> int | None:
        """Get the random seed used for this model."""
        return self._random_seed

    @abstractmethod
    def load(self, *args, **kwargs) -> None:
        """Load model and tokenizer."""

    @abstractmethod
    def train(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    @abstractmethod
    def inference(self, *args, **kwargs) -> Any:
        """Run model prediction/inference."""
        raise NotImplementedError

    @abstractmethod
    def free_memory(self) -> Any:
        """Free model resources to release memory.

        This method should be implemented by all model classes to properly
        release GPU memory, model weights, and other resources.

        Common objects to free:
        - model: Move to CPU and delete reference (e.g., self.model.cpu(); del self.model)
        - tokenizer: Delete reference (e.g., del self.tokenizer)
        - pipeline: Delete pipeline objects (e.g., del self.pipeline)
        - torch.cuda.empty_cache(): Clear CUDA cache if using PyTorch
        - gc.collect(): Force garbage collection for thorough cleanup

        Example implementation:
            if hasattr(self, 'model') and self.model is not None:
                self.model.cpu()
                del self.model
                self.model = None
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, *args, **kwargs) -> Any:
        raise NotImplementedError
