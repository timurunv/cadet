# Scripts Directory

This directory contains command-line tools for the CADET project.

## Main Scripts

### `run_experiments.py`

**Unified experiment runner for all CADET framework models.**

This is the recommended way to run experiments. It provides a single interface for:

- Simple baselines (BERT, RoBERTa, DistilBERT, BART)
- LLM Guard models (PromptGuard, LlamaGuard, ShieldGemma)
- CADET model

**Single experiment:**

```bash
# CADET model
python scripts/run_experiment.py -cn=cadet

# Simple baseline
python scripts/run_experiment.py -cn=simple_baselines

# LLM Guard
python scripts/run_experiment.py -cn=llm_guard

# Override parameters
python scripts/run_experiment.py -cn=cadet \
  data.dataset_name=DynaHate \
  data.source_style=implicit
```

**Ad-hoc testing with limited samples:**

```bash
# Quick testing with 100 samples
python scripts/run_experiment.py -cn=cadet +adhoc.n_samples=100
```

**Multirun mode (multiple experiments):**

```bash
# Use predefined multirun configs
python scripts/run_experiment.py -cn=cadet_multirun
python scripts/run_experiment.py -cn=simple_baselines_multirun
python scripts/run_experiment.py -cn=llm_guard_multirun

# Or specify multirun from command line
python scripts/run_experiment.py -cn=cadet -m \
  data.dataset_name=AbuseEval,DynaHate

# Combine multirun with ad-hoc testing
python scripts/run_experiment.py -cn=cadet_multirun +adhoc.n_samples=50
```

## Utility Scripts

### `cleanup_experiments.py`

Clean up failed experiment results.

**Basic usage:**

```bash
# Dry-run mode (preview what would be deleted, default behavior)
python scripts/cleanup_experiments.py

# Actually delete failed experiments
python scripts/cleanup_experiments.py --no-dry-run
```

**Advanced usage:**

```bash
# Use custom results directory
python scripts/cleanup_experiments.py --results-path /path/to/results

# Also delete running experiments (use with caution)
python scripts/cleanup_experiments.py --include-running --no-dry-run
```

## Configuration

All scripts use Hydra for configuration management:

- **Config files**: `configs/` - See [configs/README.md](../configs/README.md) for detailed documentation
- **Override syntax**: `key.subkey=value`
- **Multirun flag**: `--multirun` or `-m`
- **Config selection**: `-cn=CONFIG_NAME` or `--config-name=CONFIG_NAME`

**Available configs:**

- `cadet.yaml` - CADET model (single experiment)
- `cadet_multirun.yaml` - CADET multirun (sweep datasets/styles)
- `cadet_ablation.yaml` - CADET ablation study (20+ loss combinations)
- `simple_baselines.yaml` - Simple baselines (default: DistilBERT)
- `simple_baselines_multirun.yaml` - Simple baselines multirun
- `llm_guard.yaml` - LLM Guard (default: PromptGuard)
- `llm_guard_multirun.yaml` - LLM Guard multirun

For comprehensive configuration documentation including model-specific parameters, template variables,
and multirun examples, see [configs/README.md](../configs/README.md).
