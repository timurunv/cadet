"""Simple baseline pipeline for transformer models."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from omegaconf import DictConfig

from cadet.datasets.dataloader import SimpleBaselineLoader
from cadet.evaluation.enhanced_evaluator import EnhancedEvaluator
from cadet.models.base_model import MODEL_REGISTRY
from cadet.pipeline.pipeline import Pipeline
from cadet.training.simple_baseline_trainer import SimpleBaselineTrainer
from cadet.utils.logging_utils import (
    log_experiment_end,
    log_experiment_start,
    log_gpu_memory_usage,
    setup_pipeline_logger,
)
from cadet.utils.validation_utils import save_validation_results, validate_experiment_outputs

logger = logging.getLogger(__name__)


class SimpleBaselinePipeline(Pipeline):
    """Pipeline for running Simple Baseline model experiments.

    This pipeline orchestrates the 5-step process defined in DESIGN.md:
    1. DataLoader: Load data via SimpleBaselineLoader
    2. Model: Initialize transformer model (BERT, RoBERTa, DistilBERT, BART)
    3. Training: Train model with optional validation
    4. Inference: Generate predictions on test data
    5. Evaluation: Compute metrics, visualizations, and embeddings analysis

    All components are decoupled as per DESIGN.md architecture.
    """

    def __init__(self, config: DictConfig):
        """Initialize Simple Baseline pipeline.

        Args:
            config: Hydra configuration containing:
                - model: Model configuration (model_name, hyperparams)
                - data: Data configuration (dataset_name, source_style, target_style, data_path)
                - training: Training configuration (num_epochs, batch_size, learning_rate, etc.)
                - pipeline: Pipeline configuration (output_path, run_name, seed, device)
        """
        # Initialize base pipeline (creates unified directory structure and saves config)
        super().__init__(config)

        # Extract configuration sections
        self.model_config = config.model
        self.data_config = config.data
        self.training_config = config.training
        self.evaluation_config = config.evaluation
        self.pipeline_config = config.pipeline

        # Set up logging early
        self.logger = setup_pipeline_logger(
            output_path=self.output_path,
            run_name=self.run_name,
            log_level=self.pipeline_config.get("log_level", "INFO"),
            console_output=True,
        )

        # Step 1: Initialize data loader (decoupled component)
        # Resolve tokenizer_id from model_name or data config
        tokenizer_id = self.data_config.get("tokenizer_id") or MODEL_REGISTRY.get(
            self.model_config.model_name, self.model_config.model_name
        )
        if not tokenizer_id:
            raise ValueError("tokenizer_id required for dataloader")

        self.data_loader = SimpleBaselineLoader(
            dataset_name=self.data_config.dataset_name,
            source_style=self.data_config.source_style,
            tokenizer_id=str(tokenizer_id),
            target_style=self.data_config.target_style,
            max_length=self.data_config.get("max_length", 512),
            root=self.data_config.data_path,
        )

        # Step 2-4: Initialize trainer (handles model loading, training, and inference)
        self.trainer = SimpleBaselineTrainer(
            data_loader=self.data_loader,
            model_name=self.model_config.model_name,
            use_validation=self.training_config.get("use_validation", False),
            model_config=self.model_config.get("hyperparams", {}),
            output_path=str(self.output_path),
            device=self.device,
            random_seed=self.random_seed,
        )

        # Step 5: Initialize evaluator (decoupled component)
        self.evaluator = EnhancedEvaluator(
            output_path=self.output_path,
            metrics=self.evaluation_config.get(
                "metrics", ["accuracy", "precision", "recall", "f1", "auc_roc", "aupr"]
            ),
            enable_embeddings=True,
        )

    def run_experiment(self) -> dict[str, Any]:
        """Run the complete Simple Baseline experiment pipeline.

        Steps:
        1. Train model on source style data
        2. Run inference on target style test data
        3. Evaluate predictions with embeddings analysis
        4. Save all results and validate

        Returns:
            Dictionary containing experiment results and metadata

        Note:
            With Hydra multirun, each experiment runs in a separate process,
            so manual memory cleanup is unnecessary. Process termination
            automatically frees all memory.
        """
        start_time = datetime.now()
        success = False
        error = None

        # Save initial metadata with status='running'
        initial_metadata = self._create_experiment_metadata(
            start_time=start_time,
            status="running",
            success=False,
        )
        self._save_experiment_metadata(initial_metadata)

        try:
            # Log experiment start
            log_experiment_start(self.logger, self.config)

            # Step 1: Train model
            self.logger.info("Step 1/3: Training model...")
            log_gpu_memory_usage(self.logger, "Before training")
            training_results = self.trainer.train()
            log_gpu_memory_usage(self.logger, "After training")
            self.logger.info("  ✓ Training complete")
            self.logger.info("")

            # Step 2: Run inference
            self.logger.info("Step 2/3: Running inference...")
            log_gpu_memory_usage(self.logger, "Before inference")
            inference_results = self.trainer.inference()
            log_gpu_memory_usage(self.logger, "After inference")
            num_preds = inference_results["num_samples"]
            self.logger.info(f"  ✓ Inference complete: {num_preds} predictions")
            self.logger.info("")

            # Step 3: Evaluate with embeddings analysis
            self.logger.info("Step 3/3: Evaluating results...")
            metrics = self.evaluator.evaluate_with_embeddings(
                predictions_file="test_predictions.csv",
                embeddings_file="embeddings.npz",
                generate_visualizations=self.evaluation_config.get("save_embeddings", False),
            )
            self.logger.info("  ✓ Evaluation complete")
            self.logger.info("")

            # Print evaluation report
            report = self.evaluator.generate_report()
            self.logger.info("Evaluation Report:")
            self.logger.info(report)
            self.logger.info("")

            # Update experiment metadata with results
            end_time = datetime.now()
            experiment_metadata = self._create_experiment_metadata(
                start_time=start_time,
                end_time=end_time,
                status="completed",
                success=True,
                num_train_samples=training_results.get("num_train_samples"),
                num_val_samples=training_results.get("num_val_samples"),
                num_test_samples=training_results.get("num_test_samples"),
                num_predictions=inference_results.get("num_samples"),
                optimal_threshold=training_results.get("threshold"),
                metrics=metrics,
            )

            # Save updated metadata so validation can check for run.json
            self._save_experiment_metadata(experiment_metadata)

            # Validate outputs after metadata is saved
            self.logger.info("Validating experiment outputs...")
            validation_results = validate_experiment_outputs(self.output_path)

            if validation_results["success"]:
                self.logger.info("  ✓ All expected outputs validated successfully")
                experiment_metadata["validation"] = "passed"
            else:
                self.logger.warning("  ⚠ Some validation checks failed")
                for error_msg in validation_results["errors"]:
                    self.logger.warning(f"    - {error_msg}")
                experiment_metadata["validation"] = "failed"
                experiment_metadata["validation_errors"] = validation_results["errors"]

            # Update metadata with validation results
            self._save_experiment_metadata(experiment_metadata)

            # Save validation results
            save_validation_results(self.output_path, validation_results)

            success = True

            self.logger.info("=" * 60)
            self.logger.info("Experiment Complete!")
            self.logger.info(f"Results saved to: {self.output_path.absolute()}")
            self.logger.info("=" * 60)

            return experiment_metadata

        except Exception as e:
            error = e
            success = False
            self.logger.error(f"Experiment failed with error: {e}")

            # Still try to save metadata with error info
            end_time = datetime.now()
            experiment_metadata = self._create_experiment_metadata(
                start_time=start_time,
                end_time=end_time,
                status="failed",
                success=False,
                error=str(e),
            )

            self._save_experiment_metadata(experiment_metadata)
            raise

        finally:
            # Clean up memory before process ends
            try:
                if hasattr(self, "trainer") and self.trainer:
                    self.trainer.free_memory()

                # Additional GPU cleanup
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception as cleanup_error:
                self.logger.warning(f"Memory cleanup failed: {cleanup_error}")

            # Log experiment end
            log_experiment_end(self.logger, success, start_time, datetime.now(), error)
