"""Hate speech datasets

This module provides a HateSpeechDataset class that returns flexible Hugging Face Datasets.
"""

from __future__ import annotations

import logging
from itertools import product
from pathlib import Path
from typing import cast

from datasets import Dataset, DatasetDict, load_dataset

from cadet.utils import opposite_style

logger = logging.getLogger(__name__)


# Project-wide constants for dataset validation
AVAILABLE_DATASETS: list[str] = ["AbuseEval", "DynaHate", "Implicit-Hate-Corpus", "IsHate"]
AVAILABLE_STYLES: list[str] = ["implicit", "explicit"]
AVAILABLE_SPLITS: list[str] = ["train", "test"]

HF_REPO_ID = "Shuwan/cadet-datasets"


class HateSpeechDataset:
    """Loader for hate speech datasets.

    The primary entry point is :meth:`load_dataset`, which mirrors the previous
    function-based API and returns either a :class:`datasets.Dataset` or a
    mapping of split names to datasets. Instances of this class store the loaded
    data on ``self.data`` for convenience.
    """

    def __init__(
        self,
        dataset: str,
        root: str | Path = HF_REPO_ID,
        style: str | list[str] | None = None,
        split: str | list[str] | None = None,
    ) -> None:
        """Instantiate a dataset loader and immediately load data.

        The loaded artifact is exposed on :attr:`data` so callers can treat an
        instance similarly to the legacy callable loader.
        """

        if dataset not in AVAILABLE_DATASETS:
            raise ValueError(f"Dataset must be one of {AVAILABLE_DATASETS}, got {dataset}")

        self.dataset_name = dataset
        self.root = Path(root)
        self.style: list[str] = self._normalize(style, AVAILABLE_STYLES, "style")
        self.split: list[str] = self._normalize(split, AVAILABLE_SPLITS, "split")

        # Determine if we're using HF Hub or local data
        self.use_hf = str(root) == HF_REPO_ID

        if self.use_hf:
            # Using HuggingFace Hub
            self.hf_repo = HF_REPO_ID
            logger.info("Using HF Hub dataset: %s", self.hf_repo)
        else:
            # Load from local
            self.dataset_path = self.root / self.dataset_name
            if not self._check_exists():
                raise FileNotFoundError(
                    "Dataset not found at "
                    f"{self.dataset_path}. Please download the dataset from the Hugging "
                    "Face Hub, or provide a local `root` directory containing the hate speech "
                    "datasets of the same schema."
                )

        self.data: Dataset | DatasetDict = self.load_data()

    def load_data(self) -> Dataset | DatasetDict:
        """Load hate speech datasets.

        Returns:
            - A single :class:`datasets.Dataset` when both ``style`` and ``split``
              resolve to single values.
            - Otherwise a :class:`datasets.DatasetDict` keyed by ``"{style}-{split}"`` entries.

        Raises:
            ValueError: If the dataset name, style, or split is invalid.
            FileNotFoundError: If data does not exist and raw data is missing.
        """

        if self.use_hf:
            # Load from HF Hub
            dataset_dict = self._load_from_hf()
        else:
            # Load from local
            dataset_dict = load_dataset(str(self.dataset_path))

        # Get requested split keys (use underscore to be HF-compatible)
        requested_keys = [f"{style}_{split}" for style, split in product(self.style, self.split)]

        # Collect existing (valid) requested keys
        valid_keys = [k for k in requested_keys if k in dataset_dict]

        # Identify invalid requested keys
        if len(valid_keys) == 0 or len(valid_keys) < len(requested_keys):
            # Throw an error for invalid keys, showing what's available
            raise ValueError(
                f"Invalid style-split combinations: {list(set(requested_keys) - set(valid_keys))}. "
            )

        # Return single dataset if only one split requested
        if len(valid_keys) == 1:
            return cast(Dataset, dataset_dict[valid_keys[0]])

        # Return DatasetDict for multiple splits
        subset = DatasetDict()
        for key in valid_keys:
            subset[key] = cast(Dataset, dataset_dict[key])
        return subset

    def _load_from_hf(self) -> DatasetDict:
        """Load dataset from HF Hub with error handling for split naming issues."""
        try:
            # Try loading with dataset name as config
            logger.info("Loading from HF Hub: %s, config: %s", self.hf_repo, self.dataset_name)
            dataset = load_dataset(self.hf_repo, self.dataset_name)
            return dataset  # type: ignore

        except Exception as e:
            # If HF loading fails, provide helpful error message
            logger.error("Failed to load from HF Hub: %s", e)
            if "split names" in str(e) or "dashes" in str(e):
                raise ValueError(
                    f"HF dataset has split naming issues (dashes not allowed). "
                    f"Consider using local data instead. Original error: {e}"
                ) from e
            else:
                raise ValueError(f"Failed to load HF dataset: {e}") from e

    def _check_exists(self) -> bool:
        """Check if dataset exists (local directory or HF repo)."""
        if self.use_hf:
            # For HF, assume it exists if we can't prove otherwise
            # The actual check happens during loading
            return True
        else:
            # Local file check
            return self.dataset_path.exists()

    @staticmethod
    def _normalize(
        value: str | list[str] | None,
        available_values: list[str],
        name: str,
    ) -> list[str]:
        if value is None:
            return list(available_values)
        if isinstance(value, str):
            if value not in available_values:
                raise ValueError(f"Invalid {name}: {value}. Must be one of {available_values}")
            return [value]
        if isinstance(value, (list, tuple)):
            normalized = list(value)
            for item in normalized:
                if item not in available_values:
                    raise ValueError(f"Invalid {name}: {item}. Must be one of {available_values}")
            return normalized
        raise TypeError(f"{name} must be None, str, or list of str")


def get_hate_speech_datasets(
    dataset: str,
    source_style: str,
    target_style: str | None = None,
    root: str | Path = HF_REPO_ID,
) -> tuple[Dataset, Dataset]:
    """Helper function to get train, test datasets for cross-style evaluation.

    This function sets up cross-style evaluation where:
    - Training uses source_style
    - Test defaults to opposite of source_style if None (implicit if source is explicit, vice versa)

    Args:
        dataset: Dataset name, one of HateSpeechDataset.AVAILABLE_DATASETS
        source_style: Style for training data ("implicit" or "explicit")
        target_style: Style for testing data ("implicit" or "explicit")
        root: Root directory for hate speech data

    Returns:
        Tuple of (train_dataset, test_dataset) as HuggingFace Dataset objects

    Raises:
        ValueError: If source_style is not valid
        FileNotFoundError: If required data files don't exist

    Example:
        >>> train, test = get_hate_speech_datasets(
        ...     dataset="AbuseEval", source_style="explicit")
        >>> print(f"Train: {len(train)}, Test: {len(test)}")
    """
    # sanitize inputs
    if source_style not in AVAILABLE_STYLES:
        raise ValueError(f"Invalid style: {source_style}. Must be one of {AVAILABLE_STYLES}")
    if target_style and target_style not in AVAILABLE_STYLES:
        raise ValueError(f"Invalid style: {target_style}. Must be one of {AVAILABLE_STYLES}")
    if not target_style:
        # default to be the opposite style
        target_style = opposite_style(source_style)

    loader = HateSpeechDataset(dataset=dataset, root=root)
    train = loader.data[f"{source_style}_train"]
    test = loader.data[f"{target_style}_test"]

    # Both should return single Dataset objects since we're requesting single style/split
    return cast(Dataset, train), cast(Dataset, test)
