"""Simple baseline trainer for transformer models with direct model management."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from cadet.datasets.dataloader import SimpleBaselineLoader
from cadet.evaluation.threshold_optimization import optimize_threshold
from cadet.models.base_model import MODEL_REGISTRY
from cadet.training.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class SimpleBaselineTrainer(BaseTrainer):
    def __init__(
        self,
        data_loader: SimpleBaselineLoader,
        model_name: str,
        use_validation: bool = False,
        model_config: dict[str, Any] | None = None,
        output_path: str | Path = "results",
        device: str | None = None,
        random_seed: int | None = None,
    ):
        """Initialize trainer with direct model management.

        Args:
            data_loader: Data loader instance (decoupled component)
            model_name: Short model identifier (e.g., "distilbert")
            use_validation: Whether to use validation during training
            model_config: Training configuration dictionary
            output_path: Directory to save outputs
            device: Device for computation ('cpu', 'cuda', 'auto', or None for auto)
            random_seed: Random seed for reproducibility
        """
        source_style = data_loader.source_style
        target_style = data_loader.target_style

        super().__init__(
            source_style=source_style,
            target_style=target_style,
            model_config=model_config or {},
            output_path=Path(output_path),
            device=device,
            random_seed=random_seed,
        )

        # Store decoupled components
        self.data_loader = data_loader
        self.model_name = model_name
        self.use_validation = use_validation

        # Get full HuggingFace model ID from registry
        self.model_id = MODEL_REGISTRY.get(model_name, model_name)

        # Direct model management - no wrapper class
        self.model = None
        self.tokenizer = None

        # Will be set during training
        self.threshold = 0.5  # Default threshold

        # Create output directories
        (self.output_path / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.output_path / "predictions").mkdir(parents=True, exist_ok=True)
        (self.output_path / "models").mkdir(parents=True, exist_ok=True)

    def load_model(
        self, checkpoint_path: str | Path | None = None, for_inference: bool = False
    ) -> Any:
        """Load model and tokenizer with optional hidden states output.

        Args:
            checkpoint_path: Optional path to checkpoint for model weights
            for_inference: If True, enable output_hidden_states for embeddings extraction.
                          Only needed for t-SNE visualization during inference.
                          Disabling during training saves significant memory!

        Returns:
            Self (for interface compatibility)
        """
        # Clear any existing model
        self._cleanup_model()

        if checkpoint_path and Path(checkpoint_path).exists():
            # Load from local checkpoint. Do NOT override num_labels:
            # checkpoints preserve the original model configuration,
            # including the number of labels. Overriding num_labels here could
            # cause mismatches and errors.
            logger.info("Loading model from checkpoint: %s", checkpoint_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                checkpoint_path,
                output_hidden_states=for_inference,  # Only for inference!
            )
            # Tokenizer always from original HF model (checkpoints don't include tokenizer)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        else:
            # Load from HuggingFace Hub
            logger.info("Loading model from HuggingFace: %s (%s)", self.model_name, self.model_id)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id,
                num_labels=2,
                output_hidden_states=for_inference,  # Only for inference!
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Move model to device
        self.model.to(torch.device(self.device))

        if for_inference:
            logger.info("Model loaded with hidden_states output for embeddings extraction")
        else:
            logger.info("Model loaded without hidden_states (memory-efficient training mode)")

        return self

    def load_data(self) -> tuple[Dataset, Dataset | None, Dataset]:
        """Load training, validation, and test datasets.

        Delegates to the data_loader component.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        return self.data_loader.load_data()

    def train(self) -> dict[str, Any]:
        """Train model with direct management and optional validation.

        Returns:
            Dictionary with training results
        """
        logger.info("Starting training with model: %s on device: %s", self.model_name, self.device)

        # 1. Load model directly (without hidden states for memory efficiency)
        self.load_model(for_inference=False)
        assert self.model is not None, "Model not loaded"

        # 2. Load data
        train_dataset, val_dataset, test_dataset = self.load_data()

        num_train = len(train_dataset)
        num_val = len(val_dataset) if val_dataset else 0
        num_test = len(test_dataset)
        logger.info(
            "Loaded %d train, %d val, %d test samples",
            num_train,
            num_val,
            num_test,
        )

        # 3. Configure training arguments
        train_batch_size = self.model_config.get("batch_size", 16)
        # Use smaller eval batch size to avoid OOM during validation
        eval_batch_size = self.model_config.get("eval_batch_size", max(1, train_batch_size // 2))

        training_args = TrainingArguments(
            output_dir=str(self.output_path / "checkpoints"),
            num_train_epochs=self.model_config.get("num_epochs", 3),
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            learning_rate=self.model_config.get("learning_rate", 2e-5),
            weight_decay=self.model_config.get("weight_decay", 0.01),
            warmup_ratio=self.model_config.get("warmup_ratio", 0.1),
            # Validation - DISABLED during training to avoid memory issues
            # We'll do manual validation after training completes
            eval_strategy="no",  # Disable automatic validation during training
            save_strategy="epoch",
            load_best_model_at_end=False,  # Disabled since we're not evaluating during training
            # Other settings
            save_total_limit=2,
            seed=self.random_seed if self.random_seed is not None else 42,
            logging_dir=str(self.output_path / "logs"),
            logging_steps=50,
            dataloader_pin_memory=False,  # Avoid memory issues
            dataloader_num_workers=0,  # Avoid multiprocessing issues
            # Keep all dataset columns so our custom collator can map 'hate_label' -> 'labels'
            remove_unused_columns=False,
            # Use no_cuda flag to control device (True means use CPU)
            no_cuda=(self.device == "cpu"),
            # Memory optimizations
            fp16=torch.cuda.is_available(),  # Mixed precision if GPU available
            # Use gradient accumulation instead for effective larger batches
            gradient_accumulation_steps=self.model_config.get("gradient_accumulation_steps", 1),
            # Disable external reporting integrations (wandb, comet, etc.) to avoid optional deps
            # See: https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.report_to
            report_to="none",
        )

        # 4. Create trainer with custom collator
        def _collate_with_labels(batch):
            batch = default_data_collator(batch)
            # Map 'hate_label' -> 'labels' (HF Trainer expects 'labels' for loss computation)
            if "labels" not in batch and "hate_label" in batch:
                batch["labels"] = batch.pop("hate_label")
            return batch

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=None,  # Disable eval during training
            processing_class=self.tokenizer,
            data_collator=_collate_with_labels,
        )

        # 5. Train
        logger.info("Starting training...")
        trainer.train()

        # 6. Save best model
        best_model_path = self.output_path / "checkpoints" / "best"
        trainer.save_model(str(best_model_path))
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(str(best_model_path))
        logger.info("Model saved to: %s", best_model_path)

        # Clear trainer to free memory
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 7. Optimize threshold (only if validation is used)
        best_threshold = 0.5
        if self.use_validation and val_dataset:
            try:
                logger.info("Optimizing decision threshold...")
                best_threshold = self._optimize_threshold(val_dataset)
                logger.info("Optimal threshold: %.3f", best_threshold)
            except Exception as e:
                logger.warning("Threshold optimization failed: %s", e)
                best_threshold = 0.5

        self.threshold = best_threshold

        return {
            "status": "success",
            "threshold": self.threshold,
            "best_checkpoint": str(best_model_path),
            "num_train_samples": num_train,
            "num_val_samples": num_val,
            "num_test_samples": num_test,
        }

    def _optimize_threshold(self, val_dataset: Dataset) -> float:
        """Find optimal classification threshold on validation set.

        Uses threshold_optimization module for consistent threshold selection.

        Args:
            val_dataset: Validation dataset

        Returns:
            Best threshold value
        """
        logger.info("Running threshold optimization on validation set...")

        # Ensure model is loaded and in eval mode
        assert self.model is not None, "Model not loaded"
        self.model.eval()

        val_loader = DataLoader(val_dataset, batch_size=32, num_workers=0)  # type: ignore

        all_labels = []
        all_probs = []

        device = torch.device(self.device)

        with torch.no_grad():
            for batch in val_loader:
                # Get model outputs directly
                outputs = self.model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )

                # Get probabilities from logits
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)[:, 1]  # Positive class probability

                # Collect labels and probabilities
                all_labels.extend(batch["hate_label"].cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # Optimize threshold using threshold_optimization module
        y_true = np.array(all_labels)
        y_proba = np.array(all_probs)

        best_threshold, best_score = optimize_threshold(
            y_true=y_true,
            y_proba=y_proba,
            metric=self.model_config.get("threshold_metric", "f1_macro"),
            thresholds=np.linspace(
                self.model_config.get("threshold_range", [0.05, 0.95])[0],
                self.model_config.get("threshold_range", [0.05, 0.95])[1],
                self.model_config.get("n_thresholds", 19),
            ),
        )

        logger.info(
            "Threshold optimization complete: %.3f (score=%.4f)", best_threshold, best_score
        )
        return best_threshold

    def inference(self) -> dict[str, Any]:
        """Run inference on test set with memory-optimized embeddings streaming.

        Optimized for t-SNE visualization: saves embeddings directly to disk
        in batches to avoid memory accumulation. Uses memory mapping for
        efficient sequential writes.

        Returns:
            Dictionary with predictions and metadata (embeddings path)
        """
        logger.info("Running inference with memory-optimized embeddings streaming...")

        # Load best model directly from checkpoint (with hidden states for t-SNE embeddings)
        best_model_path = self.output_path / "checkpoints" / "best"
        self.load_model(checkpoint_path=best_model_path, for_inference=True)
        assert self.model is not None, "Model not loaded"
        self.model.eval()

        logger.info("Running inference on device: %s", self.device)

        # Load test data
        _, _, test_dataset = self.load_data()
        test_loader = DataLoader(test_dataset, batch_size=32, num_workers=0)  # type: ignore

        total_samples = len(test_dataset)
        logger.info("Processing %d test samples...", total_samples)

        # Setup paths
        embeddings_path = self.output_path / "models" / "embeddings.npz"
        temp_embeddings_path = self.output_path / "models" / "temp_embeddings.mmap"

        # Pre-allocate memory-mapped file for embeddings
        # Get embedding dimension from first batch
        first_batch = next(iter(test_loader))
        device = torch.device(self.device)

        with torch.no_grad():
            # Get sample outputs to determine embedding dimension
            sample_outputs = self.model(
                input_ids=first_batch["input_ids"][:1].to(device),
                attention_mask=first_batch["attention_mask"][:1].to(device),
                output_hidden_states=True,
            )
            # Extract embeddings - handle BART (encoder-decoder) vs BERT (encoder-only)
            if (
                hasattr(sample_outputs, "encoder_hidden_states")
                and sample_outputs.encoder_hidden_states is not None
            ):
                # BART/T5 - use encoder's last hidden state
                embedding_dim = sample_outputs.encoder_hidden_states[-1][:, 0, :].shape[-1]
                logger.info("Using encoder hidden states for BART/T5 model")
            elif (
                hasattr(sample_outputs, "hidden_states")
                and sample_outputs.hidden_states is not None
            ):
                # BERT/RoBERTa/DistilBERT - use last hidden state
                embedding_dim = sample_outputs.hidden_states[-1][:, 0, :].shape[-1]
                logger.info("Using hidden states for BERT/RoBERTa/DistilBERT model")
            else:
                raise RuntimeError(
                    f"Model output doesn't have hidden_states or encoder_hidden_states. "
                    f"Output type: {type(sample_outputs)}"
                )
            logger.info("Embedding dimension: %d", embedding_dim)

        # Create memory-mapped array for embeddings
        embeddings_mmap = np.memmap(
            temp_embeddings_path, dtype=np.float32, mode="w+", shape=(total_samples, embedding_dim)
        )

        all_labels = []
        all_predictions = []
        all_probs = []
        sample_idx = 0

        logger.info("Streaming embeddings to disk...")

        for batch_idx, batch in enumerate(test_loader):
            batch_size = len(batch["hate_label"])

            with torch.no_grad():
                # Get model outputs with hidden states
                outputs = self.model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    output_hidden_states=True,
                )

                # Extract embeddings - handle BART (encoder-decoder) vs BERT (encoder-only)
                if (
                    hasattr(outputs, "encoder_hidden_states")
                    and outputs.encoder_hidden_states is not None
                ):
                    # BART/T5 - use encoder's last hidden state
                    embeddings = outputs.encoder_hidden_states[-1][:, 0, :]
                elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                    # BERT/RoBERTa/DistilBERT - use last hidden state
                    embeddings = outputs.hidden_states[-1][:, 0, :]
                else:
                    raise RuntimeError(
                        f"Model output doesn't have hidden_states or encoder_hidden_states. "
                        f"Output type: {type(outputs)}"
                    )

                # Get probabilities from logits
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)[:, 1]  # Positive class probability
                preds = (probs >= self.threshold).long()

                # Stream embeddings directly to memory-mapped file
                embeddings_np = embeddings.cpu().numpy().astype(np.float32)
                embeddings_mmap[sample_idx : sample_idx + batch_size] = embeddings_np

                # Store predictions in memory (much smaller)
                all_labels.extend(batch["hate_label"].cpu().numpy())
                all_predictions.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                sample_idx += batch_size

            # Periodic GPU memory cleanup
            if batch_idx % 10 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("Processed %d/%d batches", batch_idx + 1, len(test_loader))

        # Ensure all data is written to disk
        embeddings_mmap.flush()
        logger.info("Embeddings streaming completed. Shape: %s", embeddings_mmap.shape)

        # Create predictions dictionary
        predictions_dict = {
            "text_id": list(range(len(all_labels))),
            "true_label": all_labels,
            "pred_label": all_predictions,
            "prob": all_probs,
        }

        # Save predictions
        self.save_predictions(predictions_dict)

        # Convert memory-mapped embeddings to compressed numpy archive
        logger.info("Converting to compressed format...")
        target_style = self.data_loader.target_style
        style_labels = (
            np.ones(len(all_labels)) if target_style == "explicit" else np.zeros(len(all_labels))
        )

        # Save final compressed embeddings file
        np.savez_compressed(
            embeddings_path,
            embeddings=embeddings_mmap[: len(all_labels)],  # Only save actual samples
            labels=np.array(all_labels),
            text_ids=np.arange(len(all_labels)),
            style_labels=style_labels,
            # Metadata for t-SNE visualization
            model_name=self.model_name,
            target_style=target_style,
            source_style=self.data_loader.source_style,
            dataset_name=getattr(self.data_loader, "dataset_name", "unknown"),
        )

        # Clean up temporary file
        del embeddings_mmap  # Close memory map
        temp_embeddings_path.unlink(missing_ok=True)

        logger.info("Embeddings saved to: %s", embeddings_path)
        logger.info(
            "Final shape: (%d, %d), Target style: %s", len(all_labels), embedding_dim, target_style
        )

        # Final GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "predictions": predictions_dict,
            "embeddings_path": str(embeddings_path),  # Return path instead of array
            "embedding_shape": (len(all_labels), embedding_dim),
            "num_samples": len(all_labels),
        }

    def save_predictions(self, predictions: dict[str, list]) -> None:
        """Save predictions to CSV file.

        Args:
            predictions: Dictionary containing predictions with keys:
                - text_id: List of text IDs
                - true_label: List of true labels
                - pred_label: List of predicted labels
                - prob: List of prediction probabilities
        """
        predictions_df = pd.DataFrame(predictions)
        predictions_path = self.output_path / "predictions" / "test_predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)
        logger.info("Predictions saved to: %s", predictions_path)

    def _cleanup_model(self) -> None:
        """Clean up model and tokenizer to free memory."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        # Force garbage collection
        import gc

        gc.collect()

        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def free_memory(self) -> None:
        """Free GPU and system memory by cleaning up model."""
        logger.info("Cleaning up trainer memory...")
        self._cleanup_model()

        # Clear GPU cache
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info("Memory cleanup completed")
