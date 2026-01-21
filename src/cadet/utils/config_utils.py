"""Configuration utilities for running multiple experiments."""

from datetime import datetime

from omegaconf import DictConfig


def process_run_name_template(template: str, cfg: DictConfig, run_index: int = 1) -> str | None:
    """Process run_name template with config variables and datetime.

    Args:
        template: Template string with {variable} placeholders
        cfg: Configuration object
        run_index: Current run index for multi-runs

    Returns:
        Processed run name string or None if template is null/empty

    Available template variables:
        {model_name} - Model name (e.g., promptguard)
        {dataset_name} - Dataset name (e.g., AbuseEval)
        {source_style} - Source style (e.g., explicit)
        {target_style} - Target style (e.g., implicit)
        {seed} - Random seed
        {run_index} - Run index for multi-runs (1, 2, 3...)
        {timestamp} - Current timestamp (YYYYMMDD_HHMMSS)
        {datetime} - Same as timestamp (YYYYMMDD_HHMMSS)
        {date} - Date only (YYYY-MM-DD)
        {time} - Time only (HHMMSS)
        {year} - Year (YYYY)
        {month} - Month (MM)
        {day} - Day (DD)
        {hour} - Hour (HH)
        {minute} - Minute (MM)
        {second} - Second (SS)

    Examples:
        >>> cfg = OmegaConf.create({
        ...     "model": {"model_name": "promptguard"},
        ...     "data": {"dataset_name": "AbuseEval", "source_style": "explicit"},
        ...     "pipeline": {"seed": 42}
        ... })
        >>> process_run_name_template("test_{model_name}_{dataset_name}_{date}", cfg)
        'test_promptguard_AbuseEval_2025-10-01'
        >>> process_run_name_template("{model_name}_run{run_index}_{time}", cfg, 2)
        'promptguard_run2_143022'
    """
    if not template or template == "null":
        return None

    now = datetime.now()

    # Built-in datetime variables
    datetime_vars = {
        "timestamp": now.strftime("%Y%m%d_%H%M%S"),
        "datetime": now.strftime("%Y%m%d_%H%M%S"),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H%M%S"),
        "year": now.strftime("%Y"),
        "month": now.strftime("%m"),
        "day": now.strftime("%d"),
        "hour": now.strftime("%H"),
        "minute": now.strftime("%M"),
        "second": now.strftime("%S"),
    }

    # Config variables
    config_vars = {
        "model_name": str(cfg.model.model_name),
        "dataset_name": str(cfg.data.dataset_name),
        "source_style": str(cfg.data.source_style),
        "target_style": opposite_style(cfg.data.source_style)
        if cfg.data.target_style is None
        else cfg.data.target_style,
        "seed": str(cfg.pipeline.seed),
        "run_index": str(run_index),
    }

    # Combine all variables
    template_vars = {**datetime_vars, **config_vars}

    # Process template
    try:
        result = template.format(**template_vars)
        return result
    except KeyError as e:
        available_vars = list(template_vars.keys())
        raise ValueError(f"Unknown template variable: {e}. Available variables: {available_vars}")


def generate_run_name(cfg: DictConfig, include_run_index: bool = True, run_index: int = 1) -> str:
    """Generate a descriptive run name for an experiment.

    Supports three run_name formats:
    1. Template with {variables}: "{model_name}_{date}" -> "promptguard_2025-10-01"
    2. Prefix with wildcard: "my_exp_*" -> "my_exp_AbuseEval-promptguard-explicit-seed42"
    3. Literal string: "fixed_name" -> "fixed_name"

    Args:
        cfg: Configuration for the experiment
        include_run_index: Whether to include run index in name (for repeated runs)
        run_index: Run index to use if not in config

    Returns:
        Descriptive run name

    Example:
        >>> cfg = OmegaConf.create({
        ...     "model": {"model_name": "promptguard"},
        ...     "data": {"dataset_name": "AbuseEval", "source_style": "explicit"},
        ...     "pipeline": {"seed": 42, "run_index": 1}
        ... })
        >>> generate_run_name(cfg)
        'AbuseEval-promptguard-explicit-seed42-run1'
        >>> cfg.pipeline.run_name = "my_exp_*"
        >>> generate_run_name(cfg)
        'my_exp_AbuseEval-promptguard-explicit-seed42-run1'
    """
    run_name_config = cfg.pipeline.get("run_name")
    actual_run_index = cfg.pipeline.get("run_index", run_index)

    # Case 1: Template with {variables}
    if run_name_config and "{" in str(run_name_config):
        run_name = process_run_name_template(run_name_config, cfg, actual_run_index)
        assert run_name is not None
        return run_name

    # Case 2: Prefix with wildcard (ends with *)
    if run_name_config and str(run_name_config).endswith("*"):
        prefix = str(run_name_config)[:-1]  # Remove the *
        auto_suffix = _generate_auto_suffix(cfg, include_run_index, actual_run_index)
        return f"{prefix}{auto_suffix}"

    # Case 3: Literal string (non-null, non-template)
    if run_name_config and run_name_config != "null":
        return str(run_name_config)

    # Case 4: Auto-generation (fallback)
    return _generate_auto_suffix(cfg, include_run_index, actual_run_index)


def _generate_auto_suffix(
    cfg: DictConfig, include_run_index: bool = True, run_index: int = 1
) -> str:
    """Generate the standard auto-generated suffix for run names.

    Args:
        cfg: Configuration for the experiment
        include_run_index: Whether to include run index in name
        run_index: Run index to use

    Returns:
        Auto-generated suffix like "AbuseEval-promptguard-explicit-seed42-run1"
    """
    parts = [
        cfg.data.dataset_name,
        cfg.model.model_name,
        cfg.data.source_style,
    ]

    # Add seed
    seed = cfg.pipeline.get("seed", 42)
    parts.append(f"seed{seed}")

    # Add run index if requested and present
    if include_run_index and run_index > 1:
        parts.append(f"run{run_index}")

    return "-".join(str(p) for p in parts)


def opposite_style(style: str) -> str:
    """Get the opposite style (explicit <-> implicit).

    Args:
        style: "explicit" or "implicit"

    Returns:
        Opposite style

    Raises:
        ValueError if input style is invalid
    """
    if style == "explicit":
        return "implicit"
    elif style == "implicit":
        return "explicit"
    else:
        raise ValueError(f"Invalid style: {style}. Must be 'explicit' or 'implicit'.")


def to_serializable(obj):
    """Convert OmegaConf DictConfig/ListConfig to plain Python objects.

    This ensures proper JSON serialization by recursively converting
    all OmegaConf objects to their Python equivalents.

    Args:
        obj: Object to convert (can be DictConfig, ListConfig, or any Python object)

    Returns:
        Plain Python object (dict, list, or primitive)

    Examples:
        >>> from omegaconf import DictConfig
        >>> cfg = DictConfig({"a": 1, "b": {"c": 2}})
        >>> to_serializable(cfg)
        {'a': 1, 'b': {'c': 2}}
    """
    from omegaconf import DictConfig, ListConfig, OmegaConf

    if isinstance(obj, (DictConfig, ListConfig)):
        # Use OmegaConf to convert with interpolation resolution
        return OmegaConf.to_container(obj, resolve=True)
    elif isinstance(obj, dict):
        # Recursively convert dict values
        return {key: to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # Recursively convert list/tuple items
        return [to_serializable(item) for item in obj]
    else:
        # Return primitives as-is
        return obj
