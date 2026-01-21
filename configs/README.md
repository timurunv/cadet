# Configuration Files

This directory contains Hydra configuration files for all CADET framework experiments.

For usage instructions on running experiments, see [scripts/README.md](../scripts/README.md).

## Overview

The CADET framework supports three types of models:

1. **CADET Model**: Full causal disentanglement model with multi-objective training
2. **Simple Baselines**: Fine-tuned transformer models (BERT, RoBERTa, DistilBERT, BART)
3. **LLM Guard**: Inference-only pre-trained safety models (PromptGuard, LlamaGuard, ShieldGemma)

## Configuration Files

### CADET Model Configurations

| Config | Purpose | Key Features |
|--------|---------|-------------|
| `cadet.yaml` | Single CADET experiment | Default CADET configuration with full multi-objective training |
| `cadet_multirun.yaml` | Multiple CADET experiments | Sweep over datasets and styles |
| `cadet_ablation.yaml` | Ablation study | Comprehensive loss component ablation (20+ configurations) |

### Simple Baselines Configurations

| Config | Purpose | Key Features |
|--------|---------|-------------|
| `simple_baselines.yaml` | Single baseline experiment | Default: DistilBERT, 3 epochs, threshold optimization |
| `simple_baselines_multirun.yaml` | Multiple baseline experiments | Sweep over 4 models × datasets × styles |

### LLM Guard Configurations

| Config | Purpose | Key Features |
|--------|---------|-------------|
| `llm_guard.yaml` | Single LLM Guard experiment | Default: PromptGuard, inference-only |
| `llm_guard_multirun.yaml` | Multiple LLM Guard experiments | Sweep over 3 models × datasets × styles |

### Default Configuration

| Config | Purpose |
|--------|---------|
| `default.yaml` | Base defaults, Inherited by all configs |

## Configuration Structure

All configurations follow this structure:

```yaml
# Model configuration
model:
  model_name: "cadet"  # or bert, roberta, distilbert, bart, promptguard, llamaguard, shieldgemma
  # Model-specific hyperparameters...

# Data configuration
data:
  dataset_name: "AbuseEval"  # AbuseEval, DynaHate, Implicit-Hate-Corpus, IsHate
  source_style: "explicit"    # explicit or implicit
  target_style: null          # Auto-set to opposite of source_style
  data_path: "Shuwan/cadet-datasets"

# Training configuration (CADET and Simple Baselines only)
training:
  # Training hyperparameters...

# Evaluation configuration
evaluation:
  metrics:
    - "accuracy"
    - "precision"
    - "recall"
    - "f1"
    - "auc_roc"
    - "aupr"

# Pipeline configuration
pipeline:
  run_name: "{model_name}-{dataset_name}-{source_style}-seed{seed}-{timestamp}"
  output_path: "results/"
  seed: 42
```

## Model-Specific Details

### CADET Model

**Supported models:**

- `cadet`: Full causal disentanglement model

**Key configuration sections:**

```yaml
model:
  model_name: "cadet"
  encoder_checkpoint: "bert-base-uncased"
  decoder_checkpoint: "facebook/bart-base"
  encoder_tokenizer: "bert-base-uncased"
  decoder_tokenizer: "facebook/bart-base"
  style_dim: 2        # Style latent dimension (explicit/implicit)
  conf_dim: 5         # Confounder latent dimension
  orth_dim: 5         # Orthogonal latent dimension
  use_confounder_for_prediction: true
  style_tau: 0.5      # Gumbel-Softmax temperature for style
  target_tau: 0.5     # Gumbel-Softmax temperature for target

training:
  batch_size: 16
  max_epochs: 20
  patience: 5

  # Multi-objective loss weights
  loss_weights:
    reconstruction: 0.5
    hate: 2.0
    target: 0.5
    style: 1.0
    kl: 0.1
    orthogonality: 3.0
    counterfactual: 0.5
    cycle: 0.5
    adversarial: 1.0

  # Progressive training schedule
  progressive:
    reconstruction_ramp_epochs: 5
    kl_start_epoch: 2
    kl_full_epoch: 6
    orth_start_epoch: 3
    adv_start_epoch: 2
    no_early_stop_before_epoch: 5

  # Loss ablation (optional)
  drop_losses: ""  # Empty string = use all losses
                   # Or: "reconstruction-kl" to drop multiple losses
```

**Ablation configurations:**

The `cadet_ablation.yaml` config runs comprehensive ablation studies by dropping loss components:
`hate, target, style, orthogonality, adversarial, reconstruction, counterfactual, cycle, kl`

### Simple Baselines

**Supported models:**

- `bert`: BERT-base-uncased (110M parameters)
- `roberta`: RoBERTa-base (125M parameters)
- `distilbert`: DistilBERT-base-uncased (66M parameters)
- `bart`: BART-base (140M parameters)

**Key configuration sections:**

```yaml
model:
  model_name: "distilbert"  # bert, roberta, distilbert, or bart

  hyperparams:
    learning_rate: 5e-5
    batch_size: 20
    eval_batch_size: 40
    gradient_accumulation_steps: 1
    num_epochs: 3
    warmup_ratio: 0.1
    weight_decay: 0.01
    threshold_range: [0.05, 0.95]
    n_thresholds: 19

evaluation:
  save_embeddings: true  # Extract embeddings for t-SNE visualization
```

### LLM Guard Models

**Supported models:**

- `promptguard`: Meta Prompt-Guard-86M (86M parameters, fast)
- `llamaguard`: Meta Llama-Guard-3-8B (8B parameters, high quality)
- `shieldgemma`: Google ShieldGemma-2B (2B parameters, alternative)

**Key configuration sections:**

```yaml
model:
  model_name: "promptguard"  # promptguard, llamaguard, or shieldgemma
  batch_size: 10  # Inference batch size

# Note: No training section (inference-only)
```

**Authentication requirement:**

LLM Guard models require Hugging Face authentication:

```bash
export HF_TOKEN=hf_...your_token...
# Or create .env file with HF_TOKEN=hf_...
```

## Multirun Configurations

Multirun configs use Hydra's sweep functionality to run multiple experiments:

```yaml
hydra:
  mode: MULTIRUN
  sweeper:
    params:
      model.model_name: bert,roberta,distilbert,bart
      data.dataset_name: AbuseEval,DynaHate
      data.source_style: explicit,implicit
```

This creates: 4 models × 2 datasets × 2 styles = **16 experiments**

## Template Variables

Run names support dynamic template substitution:

| Variable | Description | Example |
|----------|-------------|---------|
| `{model_name}` | Model name | `cadet`, `bert`, `promptguard` |
| `{dataset_name}` | Dataset name | `AbuseEval`, `DynaHate` |
| `{source_style}` | Source training style | `explicit`, `implicit` |
| `{target_style}` | Target evaluation style | `implicit`, `explicit` |
| `{seed}` | Random seed | `42` |
| `{timestamp}` | Full timestamp | `20260119_143022` |
| `{date}` | Date only | `2026-01-19` |
| `{time}` | Time only | `143022` |
| `{year}`, `{month}`, `{day}` | Date components | `2026`, `01`, `19` |
| `{hour}`, `{minute}`, `{second}` | Time components | `14`, `30`, `22` |

**Template examples:**

```yaml
# Descriptive naming
run_name: "{model_name}-{dataset_name}-{source_style}-seed{seed}-{timestamp}"
# Result: cadet-AbuseEval-explicit-seed42-20260119_143022

# Simple naming
run_name: "{model_name}_{date}"
# Result: bert_2026-01-19

# Ablation naming
run_name: "{model_name}-{source_style}-ablation-{timestamp}"
# Result: cadet-explicit-ablation-20260119_143022
```

## Output Structure

All experiments save results to:

```bash
{output_path}/{run_name}/
├── config/
│   └── config.yaml              # Frozen experiment configuration
├── predictions/
│   ├── test_predictions.csv     # Predictions with probabilities
│   └── test_predictions.json    # JSON format
├── metrics/
│   └── metrics.json             # All computed metrics
├── models/
│   ├── embeddings.npz           # Model embeddings (for visualization)
│   ├── latents.npz              # CADET latents (M, T, S, U)
│   ├── training_losses.csv      # Per-batch training losses
│   └── checkpoints/             # Model checkpoints
├── visualizations/              # All plots and figures
│   ├── roc_curve.png
│   ├── pr_curve.png
│   ├── tsne_plot.png            # General t-SNE
│   ├── causal_alignment.png     # CADET: Variable predictiveness
│   ├── latent_tsne.png          # CADET: Multi-latent visualization
│   └── training_losses.png      # CADET: Loss curves
├── reports/
│   └── evaluation_report.txt    # Human-readable summary
└── logs/
    └── experiment.log           # Execution logs
```

## Datasets

All configurations support these datasets:

- **AbuseEval**: Explicit and implicit abuse detection
- **DynaHate**: Dynamic hate speech with multiple rounds
- **Implicit-Hate-Corpus**: Specialized implicit hate corpus
- **IsHate**: Binary hate speech classification

See [docs/DATASETS.md](../docs/DATASETS.md) for detailed dataset information.

## Common Overrides

Override any configuration parameter from the command line:

```bash
# Change dataset
python scripts/run_experiments.py -cn=cadet data.dataset_name=DynaHate

# Change style
python scripts/run_experiments.py -cn=cadet data.source_style=implicit

# Change seed
python scripts/run_experiments.py -cn=cadet seed=123

# Change output path
python scripts/run_experiments.py -cn=cadet pipeline.output_path=/scratch/results

# Multiple overrides
python scripts/run_experiments.py -cn=cadet \
  data.dataset_name=DynaHate \
  data.source_style=implicit \
  seed=123

# Ad-hoc testing with limited samples
python scripts/run_experiments.py -cn=cadet +adhoc.n_samples=100
```

## See Also

- [scripts/README.md](../scripts/README.md) - How to run experiments
- [docs/EVALUATIONS.md](../docs/EVALUATIONS.md) - Evaluation metrics and analysis
- [docs/DATASETS.md](../docs/DATASETS.md) - Dataset documentation
- [docs/DESIGN.md](../docs/DESIGN.md) - Architecture and design principles
