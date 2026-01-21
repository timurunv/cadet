"""Pipeline for LLM Guard models."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from omegaconf import DictConfig

from cadet.datasets.dataloader import LLMGuardLoader
from cadet.evaluation.simple_evaluator import SimpleEvaluator
from cadet.training.llm_guard_trainer import LLMGuardTrainer
from cadet.utils.logging_utils import (
    log_experiment_end,
    log_experiment_start,
    log_gpu_memory_usage,
    setup_pipeline_logger,
)
from cadet.utils.validation_utils import save_validation_results, validate_experiment_outputs

from .pipeline import Pipeline


class LLMGuardPipeline(Pipeline):
    """Pipeline for running LLM Guard model experiments.

    This pipeline orchestrates the 5-step process defined in DESIGN.md:
    1. DataLoader: Load test data via LLMGuardLoader
    2. Model: Load pre-trained LLM Guard model
    3. Training: Skipped (inference-only)
    4. Inference: Generate predictions on test data
    5. Evaluation: Compute metrics and visualizations

    All components are decoupled as per DESIGN.md architecture.
    """

    def __init__(self, config: DictConfig):
        """Initialize LLM Guard pipeline.

        Args:
            config: Hydra configuration containing:
                - model: Model configuration (model_name, batch_size)
                - data: Data configuration (dataset_name, target_style, data_path)
                - pipeline: Pipeline configuration (output_path, run_name, seed)
        """
        # Initialize base pipeline (creates unified directory structure and saves config)
        super().__init__(config)

        # Extract configuration sections
        self.model_config = config.model
        self.data_config = config.data
        self.pipeline_config = config.pipeline

        # Set up logging early
        self.logger = setup_pipeline_logger(
            output_path=self.output_path,
            run_name=self.run_name,
            log_level="INFO",
            console_output=True,
        )

        # Step 1: Initialize data loader (decoupled component)
        self.data_loader = LLMGuardLoader(
            dataset_name=self.data_config.dataset_name,
            target_style=self.data_config.target_style,
            root=self.data_config.data_path,
        )

        # Step 2-4: Initialize trainer (handles model loading and inference)
        self.trainer = LLMGuardTrainer(
            data_loader=self.data_loader,
            model_name=self.model_config.model_name,
            batch_size=self.model_config.batch_size,
            output_path=str(self.output_path),
            random_seed=self.pipeline_config.get("seed", 42),
        )

        # Step 5: Initialize evaluator (decoupled component)
        self.evaluator = SimpleEvaluator(output_path=self.output_path)

    def run_experiment(self) -> dict[str, Any]:
        """Run the complete LLM Guard experiment pipeline.

        Steps:
        1. Load model (pre-trained LLM Guard model)
        2. Run inference on test data
        3. Evaluate predictions
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

            # Step 1: Load model
            self.logger.info("Step 1/3: Loading model...")
            log_gpu_memory_usage(self.logger, "Before model loading")
            model = self.trainer.load_model()
            log_gpu_memory_usage(self.logger, "After model loading")
            self.logger.info(f"  ✓ Model loaded: {model.model_id}")
            self.logger.info("")

            # Step 2: Run inference
            self.logger.info("Step 2/3: Running inference...")
            log_gpu_memory_usage(self.logger, "Before inference")
            predictions = self.trainer.inference()
            log_gpu_memory_usage(self.logger, "After inference")
            self.logger.info(f"  ✓ Inference complete: {len(predictions['text_id'])} predictions")
            self.logger.info("")

            # Step 3: Evaluate
            self.logger.info("Step 3/3: Evaluating results...")
            metrics = self.evaluator.evaluate(predictions_file="test_predictions.csv")
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
                num_predictions=len(predictions["text_id"]),
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
            # Log experiment end (no manual cleanup needed)
            log_experiment_end(self.logger, success, start_time, datetime.now(), error)
