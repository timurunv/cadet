"""Validation utilities for pipeline runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def validate_experiment_outputs(output_path: Path) -> dict[str, Any]:
    """Validate that experiment completed successfully by checking expected outputs.

    Args:
        output_path: Path to experiment output directory

    Returns:
        Dictionary with validation results including success status and details
    """
    validation_results = {
        "success": True,
        "errors": [],
        "warnings": [],
        "files_checked": {},
    }

    # Expected files and directories
    expected_files = {
        "run.json": "Experiment metadata",
        "config/config.yaml": "Experiment configuration",
        "predictions/test_predictions.csv": "Model predictions",
        "metrics/metrics.json": "Evaluation metrics",
    }

    expected_dirs = {
        "logs": "Log files",
        "predictions": "Prediction outputs",
        "metrics": "Evaluation metrics",
        "config": "Configuration files",
    }

    # Check directories exist
    for dir_name, description in expected_dirs.items():
        dir_path = output_path / dir_name
        if dir_path.exists() and dir_path.is_dir():
            validation_results["files_checked"][dir_name] = {"exists": True, "type": "directory"}
        else:
            validation_results["errors"].append(f"Missing directory: {dir_name} ({description})")
            validation_results["success"] = False
            validation_results["files_checked"][dir_name] = {"exists": False, "type": "directory"}

    # Check files exist and are non-empty
    for file_path, description in expected_files.items():
        full_path = output_path / file_path
        if full_path.exists() and full_path.is_file():
            file_size = full_path.stat().st_size
            if file_size > 0:
                validation_results["files_checked"][file_path] = {
                    "exists": True,
                    "size": file_size,
                    "type": "file",
                }
            else:
                validation_results["errors"].append(f"Empty file: {file_path} ({description})")
                validation_results["success"] = False
                validation_results["files_checked"][file_path] = {
                    "exists": True,
                    "size": 0,
                    "type": "file",
                }
        else:
            validation_results["errors"].append(f"Missing file: {file_path} ({description})")
            validation_results["success"] = False
            validation_results["files_checked"][file_path] = {"exists": False, "type": "file"}

    # Additional validation checks
    _validate_predictions_file(output_path, validation_results)
    _validate_metrics_file(output_path, validation_results)

    return validation_results


def _validate_predictions_file(output_path: Path, validation_results: dict[str, Any]) -> None:
    """Validate predictions file content.

    Args:
        output_path: Path to experiment output directory
        validation_results: Validation results dictionary to update
    """
    predictions_file = output_path / "predictions" / "test_predictions.csv"

    if not predictions_file.exists():
        return  # Already handled by main validation

    try:
        import pandas as pd

        df = pd.read_csv(predictions_file)

        # Check required columns
        required_columns = ["text_id", "true_label", "pred_label", "prob"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            validation_results["errors"].append(
                f"Missing columns in predictions file: {missing_columns}"
            )
            validation_results["success"] = False

        # Check for empty dataframe
        if len(df) == 0:
            validation_results["errors"].append("Predictions file is empty")
            validation_results["success"] = False
        else:
            predictions_csv_key = "predictions/test_predictions.csv"
            validation_results["files_checked"][predictions_csv_key]["num_rows"] = len(df)

    except Exception as e:
        validation_results["errors"].append(f"Error reading predictions file: {e}")
        validation_results["success"] = False


def _validate_metrics_file(output_path: Path, validation_results: dict[str, Any]) -> None:
    """Validate metrics file content.

    Args:
        output_path: Path to experiment output directory
        validation_results: Validation results dictionary to update
    """
    metrics_file = output_path / "metrics" / "metrics.json"

    if not metrics_file.exists():
        return  # Already handled by main validation

    try:
        with open(metrics_file) as f:
            metrics = json.load(f)

        # Check for required metrics
        required_metrics = ["accuracy", "precision", "recall", "f1"]
        missing_metrics = [metric for metric in required_metrics if metric not in metrics]

        if missing_metrics:
            validation_results["warnings"].append(f"Missing metrics: {missing_metrics}")

        # Check for reasonable metric values
        for metric in required_metrics:
            if metric in metrics:
                value = metrics[metric]
                if not isinstance(value, (int, float)) or value < 0 or value > 1:
                    validation_results["warnings"].append(
                        f"Suspicious {metric} value: {value} (should be between 0 and 1)"
                    )

    except json.JSONDecodeError as e:
        validation_results["errors"].append(f"Error parsing metrics file: {e}")
        validation_results["success"] = False
    except Exception as e:
        validation_results["errors"].append(f"Error reading metrics file: {e}")
        validation_results["success"] = False


def save_validation_results(output_path: Path, validation_results: dict[str, Any]) -> None:
    """Save validation results to a JSON file.

    Args:
        output_path: Path to experiment output directory
        validation_results: Validation results dictionary
    """
    validation_file = output_path / "validation.json"

    try:
        with open(validation_file, "w") as f:
            json.dump(validation_results, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save validation results: {e}")
