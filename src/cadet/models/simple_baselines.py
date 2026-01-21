"""Simple transformer baselines (BERT, RoBERTa, DistilBERT, BART)."""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class SimpleBaselineModel(BaseModel):
    """Wrapper for simple baseline models (BERT, RoBERTa, DistilBERT, BART).

    Provides unified interface for standard transformer models using
    Hugging Face transformers library.

    Note: This is a lightweight wrapper and does not inherit from BaseModel
    since it's mainly used for model ID resolution. Device management is
    handled by the Trainer that uses this model.
    """

    def __init__(
        self,
        model_name: str,
        model_id: str | None = None,
        num_labels: int = 2,
        device: str | None = None,
        random_seed: int | None = None,
        **kwargs,
    ):
        """Initialize simple transformer model.

        Args:
            model_name: Model name (short name like 'bert', 'roberta', 'distilbert', 'bart')
            model_id: Optional override for model ID. If None, uses MODEL_REGISTRY lookup.
            num_labels: Number of output labels (default: 2 for binary classification)
            device: Device for computation ('cpu', 'cuda', 'auto', or None for auto)
            random_seed: Random seed for reproducibility
            **kwargs: Additional model arguments passed to AutoModelForSequenceClassification
        """
        # BaseModel will handle model_id via MODEL_REGISTRY if not provided
        super().__init__(
            model_name=model_name,
            model_id=model_id,  # Let BaseModel handle the default via MODEL_REGISTRY
            device=device,
            random_seed=random_seed,
            **kwargs,
        )

        self.num_labels = num_labels
        self.model = None
        self.tokenizer = None

    def load(self) -> Any:
        """Load model and tokenizer from Hugging Face.

        For checkpoints, loads model from checkpoint path but tokenizer from original HF model.

        Returns:
            Loaded model instance
        """
        if not self.model:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id,
                num_labels=self.num_labels,
                output_hidden_states=True,
                **self.kwargs,  # type: ignore
            )
        if not self.tokenizer:
            # For tokenizer, always use the original HuggingFace model name
            # even if model_id points to a local checkpoint
            tokenizer_id = self.model_id

            # If model_id is a local path, use original model name for tokenizer
            if isinstance(self.model_id, str) and Path(self.model_id).exists():
                from cadet.models.base_model import MODEL_REGISTRY

                original_model_id = MODEL_REGISTRY.get(self.model_name, self.model_name)
                tokenizer_id = original_model_id
                logger.info("Loading tokenizer from original model: %s", tokenizer_id)

            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
            # Add pad token if missing (needed for some models)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    def inference(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, Any]:
        """Run inference on input data.

        Args:
            input_ids: Tokenized input IDs
            attention_mask: Attention mask for inputs

        Returns:
            Dictionary with predictions and hidden states
        """
        if self.model is None:
            raise AttributeError("Model is not loaded. Call load_model() first.")

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            # Get predictions and probabilities
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)

            # Extract embeddings from last hidden state
            hidden_states = outputs.hidden_states[-1]  # Last layer
            embeddings = hidden_states[:, 0, :]  # CLS token embeddings

            return {
                "logits": logits,
                "probabilities": probabilities,
                "predictions": predictions,
                "embeddings": embeddings,
                "hidden_states": hidden_states,
            }

    def save(self, save_path: str | Path) -> None:
        """Save model and tokenizer.

        Args:
            save_path: Directory to save model
        """
        if self.model is None:
            raise ValueError("Model must be loaded before saving.")

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(save_path)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_path)

    def get_embeddings_for_batch(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Extract embeddings for a batch of inputs.

        Args:
            input_ids: Tokenized input IDs
            attention_mask: Attention mask

        Returns:
            Embeddings as a torch.Tensor
        """
        outputs = self.inference(input_ids, attention_mask)
        return outputs["embeddings"].cpu()

    def train(self, *args, **kwargs) -> Any:
        """Training is handled by SimpleBaselineTrainer.

        This model uses HuggingFace Trainer for training, which is managed
        by the SimpleBaselineTrainer class. This method is a placeholder
        to satisfy the BaseModel interface.

        Returns:
            Dict with message indicating training is handled externally
        """
        return {
            "message": "Training is handled by SimpleBaselineTrainer using HuggingFace Trainer",
            "status": "delegated_to_trainer",
        }

    def free_memory(self) -> None:
        """Free model resources to release memory.

        Releases GPU memory, model weights, and other resources.
        """
        if hasattr(self, "model") and self.model is not None:
            # Move model to CPU first to free GPU memory
            if torch.cuda.is_available():
                self.model.cpu()
            del self.model
            self.model = None

        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Force garbage collection
        gc.collect()

        logger.info("Memory freed for %s", self.model_name)
