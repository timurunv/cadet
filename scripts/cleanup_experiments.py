#!/usr/bin/env python3
"""
Standalone script to remove failed experiment directories.
Usage: python cleanup_experiments.py --results-path /path/to/results [--no-dry-run]
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_experiment_results(results_base_path):
    """Load all experiment results from run.json files."""
    results = []
    results_path = Path(results_base_path)

    for run_json_path in results_path.rglob("run.json"):
        try:
            with open(run_json_path, "r") as f:
                data = json.load(f)
                data["experiment_folder"] = str(run_json_path.parent)
                results.append(data)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Error loading {run_json_path}: {e}")
            continue

    return results


def check_experiment_status(experiments):
    """Analyze experiment status."""
    successful = []
    failed = []
    running = []

    for exp in experiments:
        status = exp.get("status", "unknown")
        success = exp.get("success", False)

        if success and status == "completed":
            successful.append(exp)
        elif status == "running":
            running.append(exp)
        else:
            failed.append(exp)

    return {
        "successful_experiments": successful,
        "failed_experiments": failed,
        "running_experiments": running,
    }


def main():
    parser = argparse.ArgumentParser(description="Remove failed experiment directories")
    parser.add_argument(
        "--results-path",
        default=None,
        help="Path to results directory (default: project root)",
    )
    parser.add_argument(
        "--no-dry-run",
        action="store_true",
        help="Actually delete files (default is dry-run)",
    )
    parser.add_argument(
        "--include-running",
        action="store_true",
        help="Also delete running experiments (use with caution)",
    )

    args = parser.parse_args()

    # Set default results path to project root
    if args.results_path is None:
        import pyprojroot

        args.results_path = str(pyprojroot.here())

    dry_run = not args.no_dry_run

    logger.info("=== EXPERIMENT CLEANUP SCRIPT ===")
    logger.info(f"Results path: {args.results_path}")
    logger.info(f"Mode: {'DRY RUN' if dry_run else 'ACTUAL DELETION'}")
    logger.info(f"Include running: {args.include_running}")

    # Load experiments
    experiments = load_experiment_results(args.results_path)
    status_report = check_experiment_status(experiments)

    failed_experiments = status_report["failed_experiments"]
    running_experiments = status_report["running_experiments"]

    logger.info(f"Found {len(failed_experiments)} failed experiments")
    logger.info(f"Found {len(running_experiments)} running experiments")

    # Collect directories to delete
    dirs_to_delete = []

    # Add failed experiments
    for exp in failed_experiments:
        exp_folder = exp.get("experiment_folder", "")
        if exp_folder and Path(exp_folder).exists():
            dirs_to_delete.append(
                {
                    "path": exp_folder,
                    "reason": "failed",
                    "run_name": exp.get("run_name", "unknown"),
                }
            )

    # Add running experiments if requested
    if args.include_running:
        for exp in running_experiments:
            exp_folder = exp.get("experiment_folder", "")
            if exp_folder and Path(exp_folder).exists():
                dirs_to_delete.append(
                    {
                        "path": exp_folder,
                        "reason": "running",
                        "run_name": exp.get("run_name", "unknown"),
                    }
                )

    # Calculate sizes and show summary
    total_size = 0
    for item in dirs_to_delete:
        try:
            folder_path = Path(item["path"])
            if folder_path.exists():
                size = sum(f.stat().st_size for f in folder_path.rglob("*") if f.is_file())
                item["size_bytes"] = size
                total_size += size
        except (OSError, PermissionError):
            item["size_bytes"] = 0

    logger.info("=== DELETION SUMMARY ===")
    logger.info(f"Directories to delete: {len(dirs_to_delete)}")
    logger.info(f"Total size to free: {total_size / (1024 * 1024):.2f} MB")

    if dirs_to_delete:
        logger.info("=== DIRECTORIES TO DELETE ===")
        for i, item in enumerate(dirs_to_delete, 1):
            size_mb = item["size_bytes"] / (1024 * 1024)
            logger.info(f"{i:3d}. {item['run_name']} ({item['reason']}) - {size_mb:.2f} MB")

    # Perform deletion
    if not dry_run:
        if input("\nProceed with deletion? (y/N): ").lower() != "y":
            logger.info("Deletion cancelled.")
            return

        logger.info("=== PERFORMING DELETION ===")
        deleted_count = 0
        for item in dirs_to_delete:
            try:
                folder_path = Path(item["path"])
                if folder_path.exists():
                    logger.info(f"Deleting: {item['run_name']}")
                    shutil.rmtree(folder_path)
                    deleted_count += 1
            except (OSError, PermissionError) as e:
                logger.error(f"Error deleting {item['path']}: {e}")

        logger.info(f"Deleted {deleted_count} directories")
    else:
        logger.info("=== DRY RUN COMPLETE ===")
        logger.info("To actually delete files, run with --no-dry-run")


if __name__ == "__main__":
    main()
