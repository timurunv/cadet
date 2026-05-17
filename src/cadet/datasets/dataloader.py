"""Data loaders for different model types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader, Sampler
from transformers import AutoTokenizer

from cadet.utils import opposite_style

from .dataset import HF_REPO_ID, HateSpeechDataset


class HateSpeechDataLoader(ABC):
    """Abstract base class for hate speech data loaders.

    All data loaders should inherit from this class and implement load_data().
    This ensures consistent interface across different model types.
    """

    @abstractmethod
    def load_data(self) -> Dataset | tuple[Dataset, Dataset, Dataset]:
        """Load dataset(s).

        Returns:
            Either a single Dataset (for inference-only models) or
            tuple of (train_dataset, val_dataset, test_dataset) for trainable models
        """
        pass


class LLMGuardLoader(HateSpeechDataLoader):
    """Data loader for LLM Guard models (inference-only).

    LLM Guard models are pre-trained and work directly with raw text.
    This loader provides access to the test dataset only, as no training is needed.

    We don't need to define source_style since no training is done.
    """

    def __init__(
        self,
        dataset_name: str,
        target_style: str,
        root: str | Path = HF_REPO_ID,
    ):
        """Initialize LLM Guard loader.

        Args:
            dataset_name: Name of the dataset (e.g., "AbuseEval")
            target_style: Style for test data
            root: Root directory for processed data
        """
        self.dataset_name = dataset_name
        self.target_style = target_style

        self.root = Path(root)

        # Load dataset
        self.dataset = HateSpeechDataset(dataset=dataset_name, root=root)

    def load_data(self) -> Dataset:
        """Load test dataset for inference.

        Returns:
            Test dataset for the specified target_style

        Note:
            LLM Guard models are inference-only, so only test data is returned.
            No training or validation data is needed.
        """
        data = self.dataset.data[f"{self.target_style}_test"]
        return cast(Dataset, data)


class SimpleBaselineLoader(HateSpeechDataLoader):
    """Data loader for simple transformer baselines.

    Loads and tokenizes data for cross-style training/evaluation.
    Train on source_style, validate/test on target_style.
    """

    def __init__(
        self,
        dataset_name: str,
        source_style: str,
        tokenizer_id: str,
        target_style: str | None = None,
        max_length: int = 512,
        root: str | Path = HF_REPO_ID,
    ):
        """Initialize data loader.

        Args:
            dataset_name: Name of dataset (e.g., "AbuseEval")
            source_style: Style for training ("explicit" or "implicit")
            tokenizer_id: HuggingFace tokenizer identifier (short name or full ID)
            target_style: Style for val/test (opposite of source if None)
            max_length: Maximum sequence length
            root: Root directory for processed data
        """
        self.dataset = HateSpeechDataset(dataset=dataset_name, root=root)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        self.source_style = source_style
        self.target_style = target_style or opposite_style(source_style)
        self.max_length = max_length

        # Add pad token if missing (needed for some models)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_data(self) -> tuple[Dataset, Dataset, Dataset]:
        """Load and tokenize train/val/test splits.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
            - train: source_style-train
            - val: target_style-test (same as test for simplicity)
            - test: target_style-test
        """
        # Load raw splits from HateSpeechDataset
        train: Dataset = cast(Dataset, self.dataset.data[f"{self.source_style}_train"])
        test: Dataset = cast(Dataset, self.dataset.data[f"{self.target_style}_test"])

        # Tokenize all splits
        train = train.map(self._tokenize, batched=True, remove_columns=["text"])
        test = test.map(self._tokenize, batched=True, remove_columns=["text"])

        # Set format for PyTorch
        columns = ["input_ids", "attention_mask", "hate_label"]
        train.set_format(type="torch", columns=columns)
        test.set_format(type="torch", columns=columns)

        # NOTE: For validation, use the same as test split
        return train, test, test

    def _tokenize(self, batch: dict[str, list]) -> dict[str, list]:
        """Tokenize a batch of texts.

        Args:
            batch: Dictionary with "text" key containing list of strings

        Returns:
            Dictionary with tokenized outputs
        """
        return self.tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,  # Will be set by set_format later
        )


class BalancedSampler(Sampler):
    """Balanced sampler for hate/non-hate labels.

    Ensures each batch has equal numbers of positive and negative examples.
    """

    def __init__(self, labels: np.ndarray):
        """Initialize sampler.

        Args:
            labels: Array of binary labels (0/1)
        """
        self.pos_idx = np.where(labels == 1)[0]
        self.neg_idx = np.where(labels == 0)[0]
        self.min_count = min(len(self.pos_idx), len(self.neg_idx))

    def __iter__(self) -> Iterator[int]:
        """Generate balanced sequence of indices."""
        # Shuffle indices within each class
        pos_shuffled = np.random.permutation(self.pos_idx)[: self.min_count]
        neg_shuffled = np.random.permutation(self.neg_idx)[: self.min_count]

        # Interleave positive and negative samples
        indices = np.empty(2 * self.min_count, dtype=int)
        indices[0::2] = pos_shuffled
        indices[1::2] = neg_shuffled

        return iter(indices.tolist())

    def __len__(self) -> int:
        """Return dataset length (2 * min_count)."""
        return 2 * self.min_count


class CADETLoader(HateSpeechDataLoader):
    """Data loader for CADET model.

    Simplified design per change notes:
    - Load HuggingFace dataset directly
    - Filter by target confidence threshold
    - Tokenization: RoBERTa (encoder) + BART (decoder)
    - Return (train, test, test) - trainer handles balanced sampling via BalancedSampler
    """

    def __init__(
        self,
        dataset_name: str,
        source_style: str = "explicit",  # Training style
        target_style: str | None = None,  # Testing style
        target_conf_threshold: float = 0.9,
        encoder_tokenizer_id: str = "roberta-base",  # HF tokenizer (not checkpoint)
        decoder_tokenizer_id: str = "facebook/bart-base",  # HF tokenizer (not checkpoint)
        max_length: int = 256,
        root: str | Path = HF_REPO_ID,
        random_seed: int = 42,
    ):
        """Initialize CADET data loader.

        Args:
            dataset_name: Dataset to load (DynaHate, AbuseEval, etc.)
            source_style: Training style level (explicit/implicit)
            target_style: Testing style level (opposite of source if None)
            target_conf_threshold: Minimum target confidence to include
            encoder_tokenizer_id: HuggingFace tokenizer ID for encoder
            decoder_tokenizer_id: HuggingFace tokenizer ID for decoder
            max_length: Maximum sequence length
            root: Root directory for datasets
            random_seed: Random seed for reproducibility
        """
        self.dataset = HateSpeechDataset(dataset=dataset_name, root=root)
        self.source_style = source_style
        self.target_style = target_style or opposite_style(source_style)
        self.target_conf_threshold = target_conf_threshold
        self.max_length = max_length
        self.random_seed = random_seed

        # Initialize tokenizers
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_tokenizer_id)
        self.decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_tokenizer_id, num_labels=2)

        # Add pad token if missing
        if self.encoder_tokenizer.pad_token is None:
            self.encoder_tokenizer.pad_token = self.encoder_tokenizer.eos_token
        if self.decoder_tokenizer.pad_token is None:
            self.decoder_tokenizer.pad_token = self.decoder_tokenizer.eos_token

        # Set random seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        # Target label encoding (initialized on first load_data call)
        self.label2id: dict[str, int] | None = None
        self.id2label: dict[int, str] | None = None
        self.n_targets: int | None = None

    def load_data(self) -> tuple[Dataset, Dataset, Dataset]:
        """Load train/test datasets with style-based splits.

        Returns:
            (train_dataset, test_dataset, test_dataset)
            Note: Returns test twice for compatibility with Pipeline interface
            No separate validation split needed per change notes

        Implementation:
        1. Load dataset using HuggingFace datasets library
        2. Filter by target confidence threshold (≥ target_conf_threshold)
        3. Split by style field: source_style for train, opposite for test
        4. Convert target labels to target_id (multiclass encoding)
        5. Apply tokenizations (encoder + decoder in parallel)
        6. Return datasets - trainer will use BalancedSampler for balanced batching
        """
        # Load raw splits from HateSpeechDataset
        train: Dataset = cast(Dataset, self.dataset.data[f"{self.source_style}_train"])
        test: Dataset = cast(Dataset, self.dataset.data[f"{self.target_style}_test"])

        # Filter by target confidence
        train = self._filter_by_target_confidence(train, self.target_conf_threshold)
        test = self._filter_by_target_confidence(test, self.target_conf_threshold)

        # Build target label encoding from both train and test
        # (avoids missing classes in one split)
        if self.label2id is None:
            all_targets = set(train["target"]) | set(test["target"])
            sorted_targets = sorted(all_targets)
            self.label2id = {label: idx for idx, label in enumerate(sorted_targets)}
            self.id2label = {idx: label for label, idx in self.label2id.items()}
            self.n_targets = len(sorted_targets)

        # Convert target labels to IDs
        train = train.map(self._encode_target, batched=False)
        test = test.map(self._encode_target, batched=False)

        # Apply tokenization
        train = train.map(self._tokenize, batched=True, remove_columns=["text"])
        test = test.map(self._tokenize, batched=True, remove_columns=["text"])

        # Set format for PyTorch
        columns = [
            "enc_input_ids",
            "enc_attention_mask",
            "dec_input_ids",
            "dec_attention_mask",
            "hate_label",
            "style",
            "target_id",
            "target_conf",
        ]
        train.set_format(type="torch", columns=columns)
        test.set_format(type="torch", columns=columns)

        # Return (train, test, test) for compatibility
        return train, test, test

    def _filter_by_target_confidence(self, dataset: Dataset, threshold: float) -> Dataset:
        """Filter dataset by target confidence.

        Args:
            dataset: Input dataset
            threshold: Minimum confidence threshold

        Returns:
            Filtered dataset with target_conf >= threshold
        """
        return dataset.filter(lambda x: x["target_conf"] >= threshold)

    def _encode_target(self, example: dict) -> dict:
        """Encode target label to integer ID.

        Args:
            example: Single example with 'target' field

        Returns:
            Example with added 'target_id' field
        """
        if self.label2id is None:
            raise RuntimeError("label2id not initialized. Call load_data() first.")
        example["target_id"] = self.label2id[example["target"]]
        return example

    def _tokenize(self, examples: dict) -> dict:
        """Apply both RoBERTa and BART tokenization.

        Args:
            examples: Batch of examples with 'text' field

        Returns:
            Dictionary with both tokenizations:
            {
                'enc_input_ids': RoBERTa tokens,
                'enc_attention_mask': RoBERTa mask,
                'dec_input_ids': BART tokens,
                'dec_attention_mask': BART mask,
                ...other fields...
            }
        """
        # Tokenize with encoder (RoBERTa)
        enc_outputs = self.encoder_tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )

        # Tokenize with decoder (BART)
        dec_outputs = self.decoder_tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )

        # Return combined outputs
        return {
            "enc_input_ids": enc_outputs["input_ids"],
            "enc_attention_mask": enc_outputs["attention_mask"],
            "dec_input_ids": dec_outputs["input_ids"],
            "dec_attention_mask": dec_outputs["attention_mask"],
        }

    def get_balanced_sampler(self, dataset: Dataset) -> BalancedSampler:
        """Create balanced sampler for training.

        Args:
            dataset: Training dataset

        Returns:
            PyTorch Sampler that returns balanced batches (50% hate, 50% non-hate)

        Implementation:
        - Identify positive (hate=1) and negative (hate=0) indices
        - Interleave equal numbers from each class
        - Shuffle within each class
        """
        labels = np.array(dataset["hate_label"])
        return BalancedSampler(labels)

    def get_dataloaders(self, batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Get PyTorch DataLoaders for train/val/test.

        Args:
            batch_size: Batch size for training

        Returns:
            (train_loader, val_loader, test_loader)

        Notes:
        - Train loader uses balanced sampler
        - Val/test loaders use sequential sampling
        """
        train, val, test = self.load_data()

        # Validate dataset sizes before creating loaders
        if len(train) == 0:
            raise ValueError(
                f"Training dataset is empty! "
                f"Check dataset '{self.dataset}' with "
                f"source_style='{self.source_style}' and "
                f"target_conf_threshold={self.target_conf_threshold}"
            )

        if len(val) == 0:
            raise ValueError(
                f"Validation dataset is empty! "
                f"Check dataset '{self.dataset}' with "
                f"target_style='{self.target_style}' and "
                f"target_conf_threshold={self.target_conf_threshold}"
            )

        if len(test) == 0:
            raise ValueError(
                f"Test dataset is empty! "
                f"Check dataset '{self.dataset}' with "
                f"target_style='{self.target_style}' and "
                f"target_conf_threshold={self.target_conf_threshold}"
            )

        # Log dataset sizes for debugging
        print(f"Dataset sizes: train={len(train)}, val={len(val)}, test={len(test)}")

        # Create train loader with balanced sampler
        train_sampler = self.get_balanced_sampler(train)
        train_loader = DataLoader(
            train,  # type: ignore
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=0,
        )

        # Create val/test loaders with sequential sampling
        val_loader = DataLoader(
            val,  # type: ignore
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        test_loader = DataLoader(
            test,  # type: ignore
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        return train_loader, val_loader, test_loader


class CSVCADETLoader(CADETLoader):
    """CADET data loader for three local CSV files: train / val / test.

    Required CSV columns:
      - text: raw text
      - hate_label: 0/1
      - style: 0/1, where 0=implicit and 1=explicit
      - target: target group/category string
      - target_conf: confidence float; use 1.0 if unavailable

    Optional columns such as avg are ignored.
    """

    REQUIRED_COLUMNS = {"text", "hate_label", "style", "target", "target_conf"}

    def __init__(
        self,
        train_csv_path: str | Path,
        val_csv_path: str | Path,
        test_csv_path: str | Path,
        source_style: str = "explicit",
        target_style: str | None = None,
        target_conf_threshold: float = 0.0,
        encoder_tokenizer_id: str = "roberta-base",
        decoder_tokenizer_id: str = "facebook/bart-base",
        max_length: int = 256,
        random_seed: int = 42,
    ):
        self.dataset = "CSV files"
        self.dataset_name = "custom_csv"
        self.train_csv_path = Path(train_csv_path)
        self.val_csv_path = Path(val_csv_path)
        self.test_csv_path = Path(test_csv_path)
        self.source_style = source_style
        self.target_style = target_style or source_style
        self.target_conf_threshold = target_conf_threshold
        self.max_length = max_length
        self.random_seed = random_seed

        self.encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_tokenizer_id)
        self.decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_tokenizer_id, num_labels=2)

        if self.encoder_tokenizer.pad_token is None:
            self.encoder_tokenizer.pad_token = self.encoder_tokenizer.eos_token
        if self.decoder_tokenizer.pad_token is None:
            self.decoder_tokenizer.pad_token = self.decoder_tokenizer.eos_token

        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        self.label2id: dict[str, int] | None = None
        self.id2label: dict[int, str] | None = None
        self.n_targets: int | None = None

    def _read_csv(self, path: Path) -> Dataset:
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        df = pd.read_csv(path)
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        df = df.copy()
        df["text"] = df["text"].fillna("").astype(str)
        df["hate_label"] = df["hate_label"].astype(int)
        df["style"] = df["style"].astype(int)
        df["target"] = df["target"].fillna("none").astype(str)
        df["target_conf"] = df["target_conf"].fillna(1.0).astype(float)

        bad_labels = sorted(set(df["hate_label"]) - {0, 1})
        if bad_labels:
            raise ValueError(f"{path} has invalid hate_label values: {bad_labels}; expected 0/1")

        bad_styles = sorted(set(df["style"]) - {0, 1})
        if bad_styles:
            raise ValueError(f"{path} has invalid style values: {bad_styles}; expected 0/1")

        return Dataset.from_pandas(df, preserve_index=False)

    def load_data(self) -> tuple[Dataset, Dataset, Dataset]:
        train = self._read_csv(self.train_csv_path)
        val = self._read_csv(self.val_csv_path)
        test = self._read_csv(self.test_csv_path)

        train = self._filter_by_target_confidence(train, self.target_conf_threshold)
        val = self._filter_by_target_confidence(val, self.target_conf_threshold)
        test = self._filter_by_target_confidence(test, self.target_conf_threshold)

        if self.label2id is None:
            all_targets = set(train["target"]) | set(val["target"]) | set(test["target"])
            sorted_targets = sorted(all_targets)
            self.label2id = {label: idx for idx, label in enumerate(sorted_targets)}
            self.id2label = {idx: label for label, idx in self.label2id.items()}
            self.n_targets = len(sorted_targets)

        train = train.map(self._encode_target, batched=False)
        val = val.map(self._encode_target, batched=False)
        test = test.map(self._encode_target, batched=False)

        train = train.map(self._tokenize, batched=True, remove_columns=["text"])
        val = val.map(self._tokenize, batched=True, remove_columns=["text"])
        test = test.map(self._tokenize, batched=True, remove_columns=["text"])

        columns = [
            "enc_input_ids",
            "enc_attention_mask",
            "dec_input_ids",
            "dec_attention_mask",
            "hate_label",
            "style",
            "target_id",
            "target_conf",
        ]
        train.set_format(type="torch", columns=columns)
        val.set_format(type="torch", columns=columns)
        test.set_format(type="torch", columns=columns)

        return train, val, test
