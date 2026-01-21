"""Preprocess raw datasets into standardized format.

This module converts raw CSV files into standardized Hugging Face Datasets and
persists them under ``data/processed/<dataset>``. In addition to the HF
artifacts, it also writes a ``meta_info.json`` with lightweight metadata that
helps with downstream publishing and auditing.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from uuid import uuid4

from datasets import ClassLabel, Dataset, DatasetDict, Features, Value, load_dataset
from pyprojroot import here

AVAILABLE_DATASETS: list[str] = ["AbuseEval", "DynaHate", "Implicit-Hate-Corpus", "IsHate"]
AVAILABLE_STYLES: list[str] = ["implicit", "explicit"]
HATE_LABEL_NAMES = ["non_hate", "hate"]

STANDARDIZED_FEATURES = Features(
    {
        "text_id": Value("int64"),
        "text": Value("string"),  # post
        "hate_label": ClassLabel(names=HATE_LABEL_NAMES),
        "avg": Value("float32"),  # average Perspective API score across selected attributes
        # style is derived from avg using a threshold (>= 0.4 -> explicit)
        # NOTE: this is a pseudo-style, derived from PerspectiveAPI call on toxicity scores.
        "style": ClassLabel(names=AVAILABLE_STYLES),  # 0=implicit, 1=explicit
        # NOTE: true style that from the original dataset (implicit=0, explicit=1)
        "true_style": ClassLabel(names=AVAILABLE_STYLES),  # 0=implicit, 1=explicit
        "target": Value("string"),
        "target_conf": Value("float32"),
    }
)


def process_raw_datasets(
    root: str | Path = here("data/processed"),
    dataset: str | list[str] | None = None,
    style: str | list[str] | None = None,
    split: str | list[str] | None = None,
) -> None:
    """Process raw datasets and persist Hugging Face DatasetDict objects.

    Args:
        root: Destination root directory to save processed datasets under ``<root>/<dataset>/``.
        dataset: None, a dataset name, or a list of names. When None, use all available
            dataset: ``AVAILABLE_DATASETS``.
        style: None, a style name, or a list of names. When None, use all available
            style: ``AVAILABLE_STYLES``.
        split: None, a split name, or a list of names. When None, use all available
            split: ``["train", "test"]``.
    """

    dataset = _normalize(dataset, AVAILABLE_DATASETS, "dataset")
    style = _normalize(style, AVAILABLE_STYLES, "style")
    split = _normalize(split, ["train", "test"], "split")

    # check source exists
    raw_path = here(Path("data/raw"))
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data path {raw_path} does not exist.")
    raw_file_name = "{style}_hate_{split}_with_targets.csv"

    # create processed data directory
    processed_data_path = Path(root)
    processed_data_path.mkdir(exist_ok=True, parents=True)

    for ds_name in dataset:
        dataset_splits: dict[str, Dataset] = {}
        raw_sources: dict[str, str] = {}

        for st_name, sp_name in product(style, split):
            raw_file_path = raw_path / ds_name / raw_file_name.format(style=st_name, split=sp_name)
            if not raw_file_path.exists():
                raise FileNotFoundError(f"Raw file {raw_file_path} does not exist.")

            raw_dataset = load_dataset("csv", data_files=str(raw_file_path), split="train")
            # Set style from file name (explicit -> 1, implicit -> 0) and compute style from avg
            hf_dataset = process_single_dataset(raw_dataset, style_name=st_name)

            # Use underscore for split keys to be HF-compliant
            split_key = f"{st_name}_{sp_name}"
            dataset_splits[split_key] = hf_dataset
            raw_sources[split_key] = str(raw_file_path.relative_to(here("data")))
            print(f"Processed {raw_file_path} -> split '{split_key}' ({len(hf_dataset)} records)")

        dataset_dict = DatasetDict(dataset_splits.items())
        processed_dataset_path = processed_data_path / ds_name

        # Write atomically: save to a unique temporary path, then replace via rename swap
        tmp_path = processed_data_path / f".{ds_name}.tmp.{uuid4().hex}"
        dataset_dict.save_to_disk(tmp_path)

        # Replace existing directory using rename swap (avoids NFS unlink issues)
        if processed_dataset_path.exists():
            backup_path = processed_data_path / f".{ds_name}.old.{uuid4().hex}"
            try:
                processed_dataset_path.rename(backup_path)
            except Exception:
                # As a fallback, try to remove but ignore errors
                shutil.rmtree(processed_dataset_path, ignore_errors=True)
                backup_path = None
            # Move tmp into place
            tmp_path.rename(processed_dataset_path)
            # Best-effort cleanup of backup
            if backup_path is not None:
                shutil.rmtree(backup_path, ignore_errors=True)
        else:
            tmp_path.rename(processed_dataset_path)
        saved_splits = list(dataset_splits.keys())
        print(f"Saved dataset '{ds_name}' with splits {saved_splits} -> {processed_dataset_path}")

        # Write meta_info.json for lightweight provenance and schema
        try:
            commit = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(here(".")))
                .decode("utf-8")
                .strip()
            )
        except Exception:
            commit = None

        # Use any split to extract features (all are standardized identically)
        any_split = saved_splits[0]
        features_obj = dataset_dict[any_split].features
        if hasattr(features_obj, "to_dict"):
            features_dict = features_obj.to_dict()  # type: ignore[attr-defined]
        else:
            features_dict = {name: str(dtype) for name, dtype in features_obj.items()}

        counts = {k: int(len(v)) for k, v in dataset_dict.items()}
        meta = {
            "dataset": ds_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": commit,
            "available_styles": AVAILABLE_STYLES,
            "splits": saved_splits,
            "counts": counts,
            "features": features_dict,
            "label_names": HATE_LABEL_NAMES,
            "raw_sources": raw_sources,
            "standardized_features_note": (
                "All splits share the same standardized schema. "
                "'true_style' is derived from the file name and reflects the "
                "original style label as provided in the raw data. "
                "'style' is derived from the 'avg' toxicity score (>=0.4 "
                "threshold) for compatibility with downstream tasks. "
                "This naming convention is historical and maintained for "
                "backward compatibility, even though it may be counterintuitive. "
                "Please refer to the documentation for further details."
            ),
        }
        with (processed_dataset_path / "meta_info.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)


def process_single_dataset(dataset, style_name: str) -> Dataset:
    """Process a single dataset from raw to standardized Hugging Face format.

    Args:
        dataset: A Hugging Face Dataset loaded from a CSV file.
        style_name: Either "implicit" or "explicit" based on the source file name. This
            value determines the style column directly (implicit=0, explicit=1).
    """

    # Standardize label column names
    if "label" in dataset.column_names:
        dataset = dataset.rename_column("label", "hate_label")
    elif "hateful_layer" in dataset.column_names:
        dataset = dataset.rename_column("hateful_layer", "hate_label")

    # Fill nulls, compute style from avg, set style from file name, and add text_id
    def standardize_row(example, idx):
        """
        Standardize a single example row by filling missing values, generating a unique text_id,
        computing style, and setting style.

        Args:
            example (dict): The example row from the dataset.
            idx (int): The index of the example in the dataset.

        Transformations:
            - Generates a unique 'text_id' as an integer (just the index).
            - Fills missing 'target' with "none", 'target_conf' with 1.0, and 'avg' with 0.0.
            - Computes 'style' as 1 if 'avg' >= 0.4, else 0.
            - Sets 'true_style' to 1 if style_name is "explicit", else 0.
            - Returns the modified example dictionary.
        """
        # Generate unique text_id as integer for torch compatibility
        example["text_id"] = idx
        example["target"] = example["target"] or "none"
        if example["target_conf"] is None:
            example["target_conf"] = 1.0
        example["avg"] = example["avg"] or 0.0
        # style derived from avg threshold
        example["style"] = 1 if example["avg"] >= 0.4 else 0
        # style comes from the file naming (dataset split style), not from avg
        example["true_style"] = 1 if style_name == "explicit" else 0
        return example

    dataset = dataset.map(standardize_row, with_indices=True)
    dataset = dataset.cast(STANDARDIZED_FEATURES)

    return dataset


def _normalize(value: str | list[str] | None, all_values: list[str], name: str) -> list[str]:
    """Normalize and validate arguments."""
    if value is None:
        return list(all_values)
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        raise TypeError(f"{name} must be None, a str, or a list/tuple/set of str.")

    invalid = [v for v in values if v not in all_values]
    if invalid:
        raise ValueError(f"Invalid {name} value(s): {invalid}. Available {name}s: {all_values}")
    return values
