"""Trainer for LLM Guard models (inference-only)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from cadet.datasets.dataloader import LLMGuardLoader
from cadet.models.llm_guard import LlamaGuard, LLMGuardModel, PromptGuard, ShieldGemma

from .base_trainer import BaseTrainer


class LLMGuardTrainer(BaseTrainer):
    """Trainer for LLM Guard models.

    LLM Guard models are pre-trained and used for inference only (no training).
    This trainer focuses on running inference and saving predictions.
    """

    def __init__(
        self,
        data_loader: LLMGuardLoader,
        model_name: str,
        batch_size: int | None,
        output_path: str,
        random_seed: int | None = None,
    ):
        """Initialize LLM Guard trainer.

        Args:
            data_loader: Data loader instance (decoupled component)
            model_name: Name of the model (e.g., "promptguard", "llamaguard", "shieldgemma")
            batch_size: Batch size for inference
            output_path: Directory to save outputs
            random_seed: Random seed for reproducibility

        Note:
            LLM Guard models are inference-only and don't perform training.
            We pass target_style as source_style to satisfy BaseTrainer's API,
            which is designed for PLM-based models that require training data.
            The source_style value is not used during inference.
        """
        # Infer target_style from data loader
        target_style = data_loader.target_style

        # For inference-only models, use target_style as source_style placeholder
        super().__init__(
            source_style=target_style,  # Placeholder - not used for inference
            target_style=target_style,
            model_config={"model_name": model_name, "batch_size": batch_size},
            output_path=Path(output_path),
            random_seed=random_seed,
        )

        # Store decoupled components
        self.data_loader = data_loader
        self.model_name = model_name.lower()
        self.batch_size = batch_size if batch_size and batch_size > 0 else 1

        # Create standard output directories
        (self.output_path / "predictions").mkdir(exist_ok=True)

        # Model will be initialized in load_model
        self.model: LLMGuardModel | None = None

    def load_model(self) -> LLMGuardModel:
        """Load LLM Guard model.

        Returns:
            Loaded model instance
        """
        # Instantiate the appropriate model based on model_name
        if self.model_name == "promptguard":
            self.model = PromptGuard()
        elif self.model_name == "llamaguard":
            self.model = LlamaGuard()
        elif self.model_name == "shieldgemma":
            self.model = ShieldGemma()
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")

        return self.model

    def load_data(self):
        """Load test dataset.

        Returns:
            Test dataset for inference
        """
        return self.data_loader.load_data()

    def train(self) -> dict[str, str]:
        """Training is not applicable for LLM Guard models.

        Returns:
            Empty dictionary (no training performed)
        """
        return {
            "message": "LLM Guard models are pre-trained and do not require training.",
            "status": "skipped",
        }

    def inference(self) -> dict[str, list]:
        """Run inference on test dataset.

        Returns:
            Dictionary containing predictions and metadata
        """
        # Load model if not already loaded
        if self.model is None:
            self.load_model()
            assert self.model is not None  # For type checker

        # Load test dataset
        test_dataset = self.load_data()

        # Extract texts and labels
        texts = test_dataset["text"]
        text_ids = test_dataset["text_id"]
        true_labels = test_dataset["hate_label"]

        # Run inference in batches
        print(f"Running inference on {len(texts)} samples...")
        predictions = []
        probabilities = []

        for i in tqdm(range(0, len(texts), self.batch_size)):
            batch_texts = texts[i : i + self.batch_size]
            batch_results = self.model.inference(batch_texts, batch_size=self.batch_size)

            for result in batch_results:
                predictions.append(int(result["label"]))
                probabilities.append(float(result["prob"]))

        # Create predictions dictionary
        predictions_dict = {
            "text_id": text_ids,
            "text": texts,
            "true_label": [int(label) for label in true_labels],
            "pred_label": predictions,
            "prob": probabilities,
        }

        # Save predictions to CSV (only format needed)
        self.save_predictions(predictions_dict)

        return predictions_dict

    def save_predictions(self, predictions: dict[str, list]) -> None:
        """Save predictions to CSV file.

        Args:
            predictions: Dictionary containing predictions
        """
        df = pd.DataFrame(predictions)
        csv_path = self.output_path / "predictions" / "test_predictions.csv"
        df.to_csv(csv_path, index=False)
        print(f"Predictions saved to {csv_path}")
