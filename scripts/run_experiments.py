#!/usr/bin/env python3
"""Unified experiment runner for all CADET framework models.

This script provides a unified interface for running experiments with:
- Simple baselines (BERT, RoBERTa, DistilBERT, BART)
- LLM Guard models (PromptGuard, LlamaGuard, ShieldGemma)
- CADET model

Supports both single runs and multiruns via Hydra configuration.

Usage:
    # Single runs
    python scripts/run_experiment.py -cn cadet
    python scripts/run_experiment.py -cn simple_baselines
    python scripts/run_experiment.py -cn llm_guard

    # Multirun via config
    python scripts/run_experiment.py -cn cadet_multirun
    python scripts/run_experiment.py -cn simple_baselines_multirun
    python scripts/run_experiment.py -cn llm_guard_multirun

    # Override from command line
    python scripts/run_experiment.py -cn cadet -m data.dataset_name=AbuseEval,DynaHate

    # Ad-hoc testing with limited samples
    python scripts/run_experiment.py --config-name=cadet +adhoc.n_samples=100

Design:
    The script automatically detects the pipeline type based on model_name and routes
    to the appropriate pipeline class:
    - "bert", "roberta", "distilbert", "bart" → SimpleBaselinePipeline
    - "promptguard", "llamaguard", "shieldgemma" → LLMGuardPipeline
    - "cadet" → CADETPipeline
"""

import logging
import os
from datetime import datetime

import hydra
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from cadet.pipeline.cadet_pipeline import CADETPipeline
from cadet.pipeline.llm_guard_pipeline import LLMGuardPipeline
from cadet.pipeline.simple_baseline_pipeline import SimpleBaselinePipeline

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)

# Pipeline routing map
PIPELINE_MAP = {
    # Simple baselines
    "bert": "simple_baseline",
    "roberta": "simple_baseline",
    "distilbert": "simple_baseline",
    "bart": "simple_baseline",
    # LLM Guard models
    "promptguard": "llm_guard",
    "llamaguard": "llm_guard",
    "shieldgemma": "llm_guard",
    # CADET model
    "cadet": "cadet",
}

PIPELINE_CLASS_MAP = {
    "simple_baseline": SimpleBaselinePipeline,
    "llm_guard": LLMGuardPipeline,
    "cadet": CADETPipeline,
}


def _check_hf_token() -> None:
    """Ensure HF_TOKEN is set for models that require it."""
    token = os.getenv("HF_TOKEN")
    if not token:
        logger.warning("HF_TOKEN not found in environment. Some models may require authentication.")


def _get_pipeline_type(model_name: str) -> str:
    """Determine pipeline type from model name.

    Args:
        model_name: Name of the model from config

    Returns:
        Pipeline type: "simple_baseline", "llm_guard", or "cadet"

    Raises:
        ValueError: If model_name is not recognized
    """
    pipeline_type = PIPELINE_MAP.get(model_name.lower())
    if pipeline_type is None:
        valid_models = ", ".join(sorted(PIPELINE_MAP.keys()))
        msg = f"Unknown model_name: {model_name}. Valid models: {valid_models}"
        raise ValueError(msg)
    return pipeline_type


def _get_pipeline_class(pipeline_type: str):
    """Import and return the appropriate pipeline class.

    Args:
        pipeline_type: Type of pipeline ("simple_baseline", "llm_guard", "cadet")

    Returns:
        Pipeline class constructor
    """
    pipeline_class = PIPELINE_CLASS_MAP.get(pipeline_type)
    if pipeline_class is None:
        msg = f"Unknown pipeline type: {pipeline_type}"
        raise ValueError(msg)
    return pipeline_class


def _print_job_header(cfg: DictConfig) -> None:
    """Print experiment information banner."""
    separator = "=" * 80
    try:
        hydra_cfg = HydraConfig.get()
        job_num = getattr(hydra_cfg.job, "num", None) if hydra_cfg.job else None

        # Calculate total jobs for multirun
        total_num = 1
        if hasattr(hydra_cfg, "sweeper") and hasattr(hydra_cfg.sweeper, "params"):
            params = hydra_cfg.sweeper.params
            for group in params.values():
                options = group.split(",")
                total_num *= len(options)

        if job_num is not None:
            job_num += 1  # Convert to 1-based index
            logger.info("%s", separator)
            logger.info("Job #%s of %s", job_num, total_num)
            logger.info("%s", separator)
        else:
            logger.info("%s", separator)
            logger.info("Single Experiment")
            logger.info("%s", separator)
    except Exception:
        logger.info("%s", separator)
        logger.info("Experiment")
        logger.info("%s", separator)

    # Print experiment details
    logger.info("Model:        %s", cfg.model.model_name)
    logger.info("Dataset:      %s", cfg.data.dataset_name)
    logger.info("Source style: %s", cfg.data.source_style)
    logger.info("Target style: %s", cfg.data.get("target_style", "auto"))
    logger.info("Seed:         %s", cfg.get("seed", 42))
    logger.info("Timestamp:    %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Ad-hoc testing mode
    n_samples = cfg.get("adhoc", {}).get("n_samples", None)
    if n_samples is not None:
        logger.warning("Ad-hoc testing mode: limited to %s samples", n_samples)
    logger.info("%s", separator)


def _apply_adhoc_limitation(pipeline, n_samples: int) -> None:
    """Apply ad-hoc sample limitation to pipeline.

    Args:
        pipeline: Pipeline instance
        n_samples: Number of samples to limit to
    """
    logger.warning("Ad-hoc testing mode: limited to %d samples", n_samples)

    # Determine pipeline type
    pipeline_class_name = pipeline.__class__.__name__

    if "LLMGuard" in pipeline_class_name:
        # LLM Guard: single dataset
        original_load_data = pipeline.data_loader.load_data

        def limited_llm_guard_load_data():
            dataset = original_load_data()
            if len(dataset) > n_samples:
                dataset = dataset.select(range(n_samples))
                logger.info("Dataset limited to %d samples", len(dataset))
            return dataset

        pipeline.data_loader.load_data = limited_llm_guard_load_data

    else:
        # Simple baselines and CADET: train/val/test splits
        original_load_data = pipeline.data_loader.load_data

        def limited_split_load_data():
            train_dataset, val_dataset, test_dataset = original_load_data()

            # Limit train dataset
            if len(train_dataset) > n_samples:
                train_dataset = train_dataset.select(range(n_samples))
                logger.info("Train dataset limited to %d samples", len(train_dataset))

            # Limit validation dataset if it exists
            if val_dataset and len(val_dataset) > n_samples // 2:
                val_size = min(len(val_dataset), n_samples // 2)
                val_dataset = val_dataset.select(range(val_size))
                logger.info("Validation dataset limited to %d samples", len(val_dataset))

            # Limit test dataset
            if len(test_dataset) > n_samples // 2:
                test_size = min(len(test_dataset), n_samples // 2)
                test_dataset = test_dataset.select(range(test_size))
                logger.info("Test dataset limited to %d samples", len(test_dataset))

            return train_dataset, val_dataset, test_dataset

        pipeline.data_loader.load_data = limited_split_load_data


def _run_experiment(cfg: DictConfig) -> bool:
    """Run a single experiment and return success status.

    Args:
        cfg: Hydra configuration

    Returns:
        True if experiment succeeded, False otherwise
    """
    try:
        # Determine pipeline type and get class
        model_name = cfg.model.get("model_name") or cfg.model.get("name")
        pipeline_type = _get_pipeline_type(model_name)
        PipelineClass = _get_pipeline_class(pipeline_type)

        logger.info("Using %s for model %s", PipelineClass.__name__, model_name)

        # Initialize pipeline (sanitizes config and creates output structure)
        pipeline = PipelineClass(cfg)

        # Print experiment header with sanitized config
        _print_job_header(pipeline.config)
        logger.info("Output path: %s", pipeline.output_path.absolute())

        # Apply ad-hoc sample limitation if requested
        n_samples = cfg.get("adhoc", {}).get("n_samples", None)
        if n_samples is not None:
            _apply_adhoc_limitation(pipeline, n_samples)

        # Run experiment
        results = pipeline.run_experiment()

        # Print results summary
        metrics = results.get("metrics", {})
        logger.info("Experiment completed successfully")
        logger.info("Accuracy:  %.4f", metrics.get("accuracy", 0.0))
        logger.info("Precision: %.4f", metrics.get("precision", 0.0))
        logger.info("Recall:    %.4f", metrics.get("recall", 0.0))
        logger.info("F1 Score:  %.4f", metrics.get("f1", 0.0))
        if "auc_roc" in metrics:
            logger.info("AUC-ROC:   %.4f", metrics["auc_roc"])

        # CADET-specific metrics
        if "causal_analysis" in metrics:
            ca = metrics["causal_analysis"]
            alignment = "aligned" if ca.get("properly_aligned", False) else "not aligned"
            logger.info("Causal Alignment: %s", alignment)

        logger.info("Results:   %s", pipeline.output_path.absolute())
        logger.info("%s", "=" * 80)

        return True

    except Exception:
        logger.exception("Experiment failed")
        return False


@hydra.main(config_path="../configs", config_name="cadet", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main function for experiment runner.

    Args:
        cfg: Hydra configuration

    Raises:
        RuntimeError: If experiment fails
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Print banner
    logger.info("%s", "=" * 80)
    logger.info("Experiment Runner")
    logger.info("%s", "=" * 80)

    # Check for HF token (warning only)
    _check_hf_token()

    # Run experiment (Hydra handles multirun iteration automatically)
    success = _run_experiment(cfg)

    if not success:
        raise RuntimeError("Experiment failed")


if __name__ == "__main__":
    main()
