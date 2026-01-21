"""Logging utilities for CADET pipeline."""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from huggingface_hub import whoami
from omegaconf import DictConfig


def setup_pipeline_logger(
    output_path: Path,
    run_name: str,
    log_level: str = "INFO",
    console_output: bool = True,
) -> logging.Logger:
    """Set up logging for pipeline execution.

    Creates both file and console handlers with proper formatting.
    Logs important system info like GPU, HF account, model cache location.

    Args:
        output_path: Directory to save log file
        run_name: Unique run identifier for log filename
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        console_output: Whether to also log to console

    Returns:
        Configured logger instance
    """
    # Create logs directory
    logs_dir = output_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger("cadet_pipeline")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear any existing handlers to avoid duplication
    logger.handlers.clear()
    # Prevent messages from propagating to the root logger which may have
    # existing handlers (e.g., from libraries or test harnesses). Leaving
    # propagation enabled leads to duplicate output (our handlers + root
    # handlers).
    logger.propagate = False

    # Create formatters
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_formatter = logging.Formatter(
        "%(levelname)s - %(message)s",
    )

    # File handler
    log_file = logs_dir / f"pipeline_{run_name}.log"
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler (optional)
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # Log system information
    _log_system_info(logger)

    return logger


def _log_system_info(logger: logging.Logger) -> None:
    """Log important system and environment information.

    Args:
        logger: Logger instance to use
    """
    logger.info("=" * 60)
    logger.info("CADET Pipeline System Information")
    logger.info("=" * 60)

    # Basic system info
    logger.info("System Information:")
    logger.info(f"  Python version: {sys.version.split()[0]}")
    logger.info(f"  Working directory: {os.getcwd()}")

    # Hugging Face info
    try:
        user_info = whoami()
        logger.info(f"  HuggingFace user: {user_info.get('name', 'Unknown')}")
        logger.info(f"  HF cache directory: {os.getenv('HF_HOME', 'Default cache location')}")
    except Exception as e:
        logger.warning(f"  HuggingFace auth check failed: {e}")

    # GPU/CUDA info
    if torch.cuda.is_available():
        logger.info("  CUDA available: Yes")
        logger.info(f"  CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"    GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")

        # Current GPU memory usage
        if torch.cuda.device_count() > 0:
            current_device = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(current_device) / 1024**3
            cached = torch.cuda.memory_reserved(current_device) / 1024**3
            logger.info(
                f"  Initial GPU memory - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB"
            )
    else:
        logger.info("  CUDA available: No")

    logger.info("=" * 60)


def log_experiment_start(logger: logging.Logger, config: DictConfig) -> None:
    """Log experiment configuration and start information.

    Args:
        logger: Logger instance
        config: Experiment configuration dictionary
    """
    logger.info("Experiment Starting:")

    # Log configuration
    if "model" in config:
        logger.info(f"  Model: {config['model'].get('model_name', 'Unknown')}")
    if "data" in config:
        logger.info(f"  Dataset: {config['data'].get('dataset_name', 'Unknown')}")
        logger.info(f"  Source style: {config['data'].get('source_style', 'Unknown')}")
        logger.info(f"  Target style: {config['data'].get('target_style', 'Unknown')}")
    if "pipeline" in config:
        logger.info(f"  Run name: {config['pipeline'].get('run_name', 'Unknown')}")
        logger.info(f"  Seed: {config['pipeline'].get('seed', 'Unknown')}")
        logger.info(f"  Output path: {config['pipeline'].get('output_path', 'Unknown')}")

    logger.info("=" * 60)


def log_gpu_memory_usage(logger: logging.Logger, step: str) -> None:
    """Log current GPU memory usage.

    Args:
        logger: Logger instance
        step: Description of current step for context
    """
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        current_device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(current_device) / 1024**3
        cached = torch.cuda.memory_reserved(current_device) / 1024**3
        total = torch.cuda.get_device_properties(current_device).total_memory / 1024**3

        logger.info(
            f"GPU Memory [{step}]: {allocated:.2f}GB allocated, "
            f"{cached:.2f}GB cached, {total:.2f}GB total"
        )


def log_experiment_end(
    logger: logging.Logger,
    success: bool,
    start_time: datetime,
    end_time: datetime,
    error: Exception | None = None,
) -> None:
    """Log experiment completion information.

    Args:
        logger: Logger instance
        success: Whether experiment completed successfully
        start_time: Experiment start time
        end_time: Experiment end time
        error: Exception if experiment failed
    """
    duration = (end_time - start_time).total_seconds()

    logger.info("=" * 60)
    if success:
        logger.info("Experiment Completed Successfully")
        logger.info(f"Duration: {duration:.2f} seconds")
    else:
        logger.error("Experiment Failed")
        logger.error(f"Duration: {duration:.2f} seconds")
        if error:
            logger.error(f"Error: {error}")

    # Log final GPU memory state
    log_gpu_memory_usage(logger, "Final")
    logger.info("=" * 60)
