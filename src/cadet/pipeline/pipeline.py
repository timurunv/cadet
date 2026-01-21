"""Base pipeline for orchestrating CADET experiments."""

from __future__ import annotations

import json
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, final

from omegaconf import DictConfig, OmegaConf
from pyprojroot import here

from cadet.datasets import AVAILABLE_DATASETS, AVAILABLE_STYLES
from cadet.utils.config_utils import generate_run_name, opposite_style
from cadet.utils.device import get_device, set_random_seed


class Pipeline(ABC):
    """Base pipeline for orchestrating CADET experiments.

    Provides a composable "Lego block" approach to building experiments
    by combining data loaders, trainers, and evaluators.
    """

    def __init__(self, config: DictConfig):
        """Initialize pipeline with configuration.

        Args:
            config: Hydra configuration for the experiment
        """
        # Sanitize config (validates, normalizes, generates run_name, resolves output_path)
        self.config = self._sanitize_config(config)

        # Extract values that were set during sanitization
        self.run_name = self.config.pipeline.run_name
        self.output_path = Path(self.config.pipeline.output_path)

        # Device and seed management from config
        self.device = self.config.pipeline.device
        self.random_seed = self.config.pipeline.seed

        # Create standardized output structure immediately
        self.directories = self._create_output_structure()

        # Save sanitized config snapshot (immutable throughout execution)
        self._save_config()

    @final
    def _create_output_structure(self) -> dict[str, Path]:
        """Create standardized output directory structure.

        This method is called automatically during initialization and creates
        a unified directory structure for all pipeline types.

        Returns:
            Dictionary mapping directory names to paths
        """
        directories = {
            "config": self.output_path / "config",
            "checkpoints": self.output_path / "checkpoints",
            "predictions": self.output_path / "predictions",
            "metrics": self.output_path / "metrics",
            "reports": self.output_path / "reports",
            "logs": self.output_path / "logs",
        }

        for path in directories.values():
            path.mkdir(parents=True, exist_ok=True)

        return directories

    def _sanitize_config(self, config: DictConfig) -> DictConfig:
        """Sanitize and validate configuration.

        Normalizes and validates dataset name and styles. Populates missing
        fields with sensible defaults. Generates run_name and resolves output_path.

        Args:
            config: Raw configuration from Hydra

        Returns:
            Sanitized configuration with all fields populated

        Raises:
            ValueError: If dataset_name, source_style, or target_style are invalid

        Note:
            - Normalizes styles to lowercase (e.g., "Explicit" -> "explicit")
            - Validates dataset_name against AVAILABLE_DATASETS
            - Validates styles against AVAILABLE_STYLES
            - If target_style is None and source_style exists, sets target_style to opposite
            - Allows same-style experiments only when explicitly specified
            - For LLM Guard models, source_style can be None (inference-only)
            - Generates run_name from template
            - Resolves output_path and handles directory conflicts
        """
        # Normalize model config: map model.name to model.model_name for backwards compatibility
        if hasattr(config, "model"):
            if hasattr(config.model, "name") and not hasattr(config.model, "model_name"):
                config.model.model_name = config.model.name

        # Validate and normalize data config
        if hasattr(config, "data"):
            # Validate dataset_name
            dataset_name = config.data.get("dataset_name")
            if dataset_name and dataset_name not in AVAILABLE_DATASETS:
                raise ValueError(
                    f"Invalid dataset_name: {dataset_name}. Must be one of {AVAILABLE_DATASETS}"
                )

            # Get and normalize styles to lowercase
            source_style = config.data.get("source_style")
            target_style = config.data.get("target_style")

            if source_style is not None:
                source_style = source_style.lower()
                config.data.source_style = source_style

            if target_style is not None:
                target_style = target_style.lower()
                config.data.target_style = target_style

            # Case 1: Both None - invalid (at least one must be specified)
            if source_style is None and target_style is None:
                raise ValueError(
                    "At least one of source_style or target_style must be specified in config.data"
                )

            # Case 2: source_style exists, target_style is None - populate opposite
            if source_style is not None and target_style is None:
                target_style = opposite_style(source_style)
                config.data.target_style = target_style

            # Case 3: source_style is None, target_style exists (LLM Guard case)
            # Set source_style to opposite for consistency, but it won't be used for training
            if source_style is None and target_style is not None:
                source_style = opposite_style(target_style)
                config.data.source_style = source_style

            # Case 4: Both specified - allow as-is (enables same-style experiments)
            # Validate styles against AVAILABLE_STYLES
            if source_style not in AVAILABLE_STYLES:
                raise ValueError(
                    f"Invalid source_style: {source_style}. Must be one of {AVAILABLE_STYLES}"
                )

            if target_style not in AVAILABLE_STYLES:
                raise ValueError(
                    f"Invalid target_style: {target_style}. Must be one of {AVAILABLE_STYLES}"
                )

            # Warn if same-style experiment (allowed)
            if source_style == target_style:
                warnings.warn(
                    f"""
                    Same-style experiment detected:
                    both source_style and target_style are '{source_style}'.
                """,
                    UserWarning,
                    stacklevel=2,
                )

        # Handle device and seed at pipeline level
        if hasattr(config, "pipeline"):
            # Device: check pipeline.device, fallback to top-level device, then auto
            device = config.pipeline.get("device") if hasattr(config.pipeline, "device") else None
            if device is None and hasattr(config, "device"):
                device = config.device
            device = get_device(device)
            config.pipeline.device = device

            # Seed: check pipeline.seed, fallback to top-level seed, then None
            seed = config.pipeline.get("seed")
            if seed is None and hasattr(config, "seed"):
                seed = config.seed
            if seed is not None:
                seed = set_random_seed(seed)
            config.pipeline.seed = seed

        # Generate run_name from template
        if hasattr(config, "pipeline"):
            run_index = config.pipeline.get("run_index", 1)
            run_name = generate_run_name(config, include_run_index=True, run_index=run_index)
            config.pipeline.run_name = run_name

            # Handle output path resolution
            output_path = config.pipeline.get("output_path")

            if output_path:
                if Path(output_path).is_absolute():
                    output_base = Path(output_path)
                else:
                    # relative to project root
                    output_base = here(Path(output_path))
            else:
                # Default to results/ in project root
                output_base = here("results")

            # Resolve directory conflicts
            desired_output_path = output_base / run_name
            resolved_output_path = resolve_output_directory_conflict(desired_output_path)

            # Update run_name if path was changed due to conflict
            if resolved_output_path != desired_output_path:
                run_name = resolved_output_path.name
                config.pipeline.run_name = run_name

            # Store resolved output_path in config
            config.pipeline.output_path = str(resolved_output_path)

        return config

    @final
    def _save_config(self) -> None:
        """Save sanitized configuration snapshot.

        Saves the config to config/config.yaml in the output directory.
        The config remains unchanged throughout the pipeline execution.
        """
        config_path = self.directories["config"] / "config.yaml"
        OmegaConf.save(self.config, config_path)

    def _create_experiment_metadata(
        self,
        start_time: datetime,
        end_time: datetime | None = None,
        status: str = "running",
        success: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create base experiment metadata.

        This method creates the essential metadata structure that all pipelines
        should include. Subclasses can extend with additional fields via kwargs.

        Args:
            start_time: Experiment start time
            end_time: Experiment end time (None if still running)
            status: Experiment status ('running', 'completed', 'failed')
            success: Whether experiment completed successfully
            **kwargs: Additional metadata fields specific to the pipeline

        Returns:
            Dictionary containing experiment metadata

        Example:
            >>> metadata = self._create_experiment_metadata(
            ...     start_time=datetime.now(),
            ...     model="llamaguard",
            ...     dataset="AbuseEval",
            ... )
        """
        metadata = {
            "run_name": self.run_name,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat() if end_time else None,
            "duration_seconds": (end_time - start_time).total_seconds() if end_time else None,
            "status": status,
            "success": success,
        }

        # Add model and dataset info if available in config
        if hasattr(self.config, "model"):
            metadata["model"] = self.config.model.get("model_name", "unknown")

        if hasattr(self.config, "data"):
            metadata["dataset"] = self.config.data.get("dataset_name", "unknown")
            metadata["source_style"] = self.config.data.get("source_style")
            metadata["target_style"] = self.config.data.get("target_style")

        # Merge additional fields from kwargs
        metadata.update(kwargs)

        return metadata

    @final
    def _save_experiment_metadata(self, metadata: dict[str, Any]) -> None:
        """Save experiment metadata to run.json.

        This method can be called multiple times during pipeline execution
        to update the status and add new fields as they become available.

        Args:
            metadata: Experiment metadata dictionary
        """
        metadata_path = self.output_path / "run.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    @abstractmethod
    def run_experiment(self) -> dict[str, Any]:
        """Run the complete experiment pipeline.

        Returns:
            Experiment results and metadata
        """
        raise NotImplementedError


def resolve_output_directory_conflict(output_path: Path) -> Path:
    """Resolve output directory conflicts by adding suffixes.

    If the directory already exists, adds a suffix (_1, _2, etc.) until
    a unique directory is found.

    Args:
        output_path: Desired output path

    Returns:
        Path that doesn't exist yet

    Example:
        >>> resolve_output_directory_conflict(Path("results/experiment"))
        # If exists: returns Path("results/experiment_1")
        # If experiment_1 exists: returns Path("results/experiment_2")
    """
    if not output_path.exists():
        return output_path

    # Issue warning about existing directory
    warnings.warn(
        f"Output directory already exists: {output_path}. Creating unique directory with suffix.",
        UserWarning,
        stacklevel=2,
    )

    # Find unique suffix
    counter = 1
    while True:
        new_path = output_path.parent / f"{output_path.name}_{counter}"
        if not new_path.exists():
            return new_path
        counter += 1
