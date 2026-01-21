"""Device and GPU utilities for CADET framework."""

from __future__ import annotations

import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_random_seed(seed: int | None = None) -> int:
    """Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed to set. If None, generates a random seed.

    Returns:
        The seed that was actually set
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    # Set seeds for all libraries
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Random seed set to: {seed}")
    return seed


def get_device(device: str | None = None) -> str:
    """Get the best available device for computation.

    Args:
        device: Preferred device ('cpu', 'cuda', 'auto', or specific GPU like 'cuda:0').
                If None or 'auto', automatically selects best available device.

    Returns:
        Device string (e.g., 'cpu', 'cuda:0')
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"Auto-selected device: {device} (GPU: {torch.cuda.get_device_name()})")
        else:
            device = "cpu"
            logger.info("Auto-selected device: cpu (no CUDA available)")
    else:
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = "cpu"
        else:
            logger.info(f"Using specified device: {device}")

    return device


def get_gpu_memory_info() -> dict:
    """Get GPU memory information.

    Returns:
        Dictionary with memory info (empty if CUDA not available)
    """
    if not torch.cuda.is_available():
        return {}

    memory_info = {}
    for i in range(torch.cuda.device_count()):
        device = f"cuda:{i}"
        memory_info[device] = {
            "total": torch.cuda.get_device_properties(i).total_memory,
            "allocated": torch.cuda.memory_allocated(i),
            "cached": torch.cuda.memory_reserved(i),
            "free": (
                torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
            ),
        }

    return memory_info


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU memory cache cleared")


def log_gpu_memory(prefix: str = ""):
    """Log current GPU memory usage.

    Args:
        prefix: Optional prefix for log message
    """
    if not torch.cuda.is_available():
        return

    memory_info = get_gpu_memory_info()
    for device, info in memory_info.items():
        allocated_gb = info["allocated"] / (1024**3)
        total_gb = info["total"] / (1024**3)
        logger.info(f"{prefix} {device}: {allocated_gb:.2f}GB / {total_gb:.2f}GB allocated")


def check_gpu_availability() -> bool:
    """Check if GPU is available for computation.

    Returns:
        True if GPU is available, False otherwise
    """
    available = torch.cuda.is_available()
    if available:
        logger.info(f"GPU available: {torch.cuda.get_device_name()}")
    else:
        logger.info("GPU not available, using CPU")
    return available
