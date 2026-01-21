"""CADET pipeline for running CADET model experiments."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from omegaconf import DictConfig

from cadet.datasets.dataloader import CADETLoader
from cadet.evaluation.causal_evaluator import CausalEvaluator
from cadet.pipeline.pipeline import Pipeline
from cadet.training.cadet_trainer import CADETTrainer
from cadet.utils.logging_utils import (
    log_experiment_end,
    log_experiment_start,
    log_gpu_memory_usage,
    setup_pipeline_logger,
)
from cadet.utils.validation_utils import save_validation_results, validate_experiment_outputs

logger = logging.getLogger(__name__)


class CADETPipeline(Pipeline):
    """Pipeline for CADET model experiments.

    Orchestrates the 5-step process:
    1. DataLoader: CADETLoader (tokenization, balanced sampling)
    2. Model: CADET initialization
    3. Training: CADETTrainer (progressive schedule, early stopping)
    4. Inference: Extract predictions and latents
    5. Evaluation: CausalEvaluator (metrics + embeddings + causal analysis)

    All components are decoupled per DESIGN.md architecture.
    """

    def __init__(self, config: DictConfig, *args, **kwargs):
        """Initialize CADET pipeline.

        Args:
            config: Hydra configuration containing:
                - model: Model configuration
                - data: Data configuration
                - training: Training configuration
                - evaluation: Evaluation configuration
                - pipeline: Pipeline configuration
        """

        # Initialize base pipeline (creates output structure)
        super().__init__(config)

        # Extract configuration sections
        self.model_config = config.model
        self.data_config = config.data
        self.training_config = config.training
        self.evaluation_config = config.evaluation
        self.pipeline_config = config.pipeline

        # Set up logging
        self.logger = setup_pipeline_logger(
            output_path=self.output_path,
            run_name=self.run_name,
            log_level=self.pipeline_config.get("log_level", "INFO"),
            console_output=True,
        )

        # Step 1: Initialize data loader
        self.data_loader = CADETLoader(
            dataset_name=self.data_config.dataset_name,
            source_style=self.data_config.source_style,
            target_style=self.data_config.get("target_style"),
            target_conf_threshold=self.data_config.target_conf_threshold,
            encoder_tokenizer_id=self.model_config.encoder_tokenizer,
            decoder_tokenizer_id=self.model_config.decoder_tokenizer,
            max_length=self.data_config.max_length,
            root=self.data_config.data_path,
            random_seed=self.random_seed,
        )

        # Step 2-4: Initialize trainer
        self.trainer = CADETTrainer(
            data_loader=self.data_loader,
            model_config=dict(self.model_config),
            training_config=dict(self.training_config),
            output_path=str(self.output_path),
            device=self.device,
            random_seed=self.random_seed,
        )

        # Step 5: Initialize evaluator (decoupled component)
        self.evaluator = CausalEvaluator(
            output_path=self.output_path,
            metrics=self.evaluation_config.get(
                "metrics",
                ["accuracy", "precision", "recall", "f1", "auc_roc", "aupr"],
            ),
            enable_embeddings=True,
            num_counterfactual_samples=self.evaluation_config.get("num_counterfactual_samples", 5),
        )

    def run_experiment(self) -> dict[str, Any]:
        """Run the complete CADET experiment pipeline.

        Steps:
        1. Train model on source style with progressive schedule
        2. Run inference on test set, extract latents
        3. Evaluate predictions with embeddings analysis
        4. Save all results and validate

        Returns:
            Dictionary containing experiment results and metadata
        """
        start_time = datetime.now()
        success = False
        error = None

        # Save initial metadata
        initial_metadata = self._create_experiment_metadata(
            start_time=start_time, status="running", success=False
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
            self.logger.info(f"    Best epoch: {training_results['best_epoch']}")
            self.logger.info(f"    Best val F1: {training_results['best_val_f1']:.4f}")
            self.logger.info("")

            # Step 2: Run inference
            self.logger.info("Step 2/3: Running inference...")
            log_gpu_memory_usage(self.logger, "Before inference")
            inference_results = self.trainer.inference()
            log_gpu_memory_usage(self.logger, "After inference")
            self.logger.info(
                f"  ✓ Inference complete: {inference_results['num_samples']} predictions"
            )
            self.logger.info("")

            # Step 3: Evaluate with causal analysis
            self.logger.info("Step 3/3: Evaluating with causal analysis...")
            metrics = self.evaluator.evaluate_with_causal_analysis(
                predictions_file="test_predictions.csv",
                latents_file="latents.npz",
                generate_visualizations=self.evaluation_config.get("save_embeddings", False),
            )
            self.logger.info("  ✓ Causal evaluation complete")

            # Log causal analysis results
            if "causal_analysis" in metrics:
                causal = metrics["causal_analysis"]
                self.logger.info(f"    M->Hate accuracy: {causal['hate_from_M']:.4f}")
                self.logger.info(f"    T->Hate accuracy: {causal['hate_from_T']:.4f}")
                self.logger.info(f"    S->Hate accuracy: {causal['hate_from_style']:.4f}")
                self.logger.info(f"    U->Hate accuracy: {causal['hate_from_U']:.4f}")
                self.logger.info(f"    Properly aligned: {causal['properly_aligned']}")
            self.logger.info("")

            # Print evaluation report
            report = self.evaluator.generate_report()
            self.logger.info("Evaluation Report:")
            self.logger.info(report)
            self.logger.info("")

            # Update experiment metadata
            end_time = datetime.now()
            experiment_metadata = self._create_experiment_metadata(
                start_time=start_time,
                end_time=end_time,
                status="completed",
                success=True,
                num_train_samples=training_results.get("num_train_samples"),
                num_val_samples=training_results.get("num_val_samples"),
                num_test_samples=inference_results.get("num_samples"),
                best_epoch=training_results["best_epoch"],
                best_val_f1=training_results["best_val_f1"],
                optimal_threshold=training_results["best_threshold"],
                metrics=metrics,
            )

            # Save metadata
            self._save_experiment_metadata(experiment_metadata)

            # Validate outputs
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

            # Update metadata
            self._save_experiment_metadata(experiment_metadata)
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

            # Save error metadata
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
            # Clean up memory
            try:
                if hasattr(self, "trainer") and self.trainer:
                    self.trainer.free_memory()

                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception as cleanup_error:
                self.logger.warning(f"Memory cleanup failed: {cleanup_error}")

            # Log experiment end
            log_experiment_end(self.logger, success, start_time, datetime.now(), error)
