# Evaluation Framework and Metrics

This document provides a comprehensive overview of the evaluation metrics, analyses, and
methodologies used in the CADET framework for hate speech detection across explicit and implicit styles.

## Classification-based Metrics

The CADET framework provides two main functions for calculating classification metrics,
separated by their input requirements:

### Basic Classification Metrics

Use `calculate_classification_metrics()` for metrics that require discrete 0/1 predictions:

```python
from cadet.evaluation import calculate_classification_metrics

# Calculate all basic metrics at once
metrics = calculate_classification_metrics(y_true, y_pred)

# Access individual metrics
accuracy = metrics['accuracy']
precision = metrics['precision']
recall = metrics['recall']
f1 = metrics['f1']
```

Returns: `accuracy`, `precision`, `recall`, `f1`

### Probability-based Metrics

Use `calculate_probability_metrics()` for metrics that require probability outputs:

```python
from cadet.evaluation import calculate_probability_metrics

# Calculate probability-based metrics
prob_metrics = calculate_probability_metrics(y_true, y_proba)

# Access metrics
auc_roc = prob_metrics['auc_roc']
aupr = prob_metrics['aupr']
```

Returns: `auc_roc`, `aupr`

### Visualization Functions

Plot ROC and Precision-Recall curves with custom titles and save options:

```python
from cadet.evaluation import plot_roc_curve, plot_precision_recall_curve

# Create ROC curve
roc_fig = plot_roc_curve(
    y_true, y_proba,
    title="ROC Curve - Explicit->Implicit",
    save_path="results/roc_curve.png"
)

# Create Precision-Recall curve
pr_fig = plot_precision_recall_curve(
    y_true, y_proba,
    title="PR Curve - Implicit->Explicit",
    save_path="results/pr_curve.png"
)
```

### Usage Examples

```python
from cadet.evaluation import calculate_classification_metrics, calculate_probability_metrics

# For models that output both predictions and probabilities
def evaluate_model(model, test_loader):
    all_preds = []
    all_probs = []
    all_true = []

    for batch in test_loader:
        logits = model(batch['input'])
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_true.extend(batch['labels'].cpu().numpy())

    # Calculate all metrics
    basic_metrics = calculate_classification_metrics(all_true, all_preds)
    prob_metrics = calculate_probability_metrics(all_true, all_probs)

    # Combine results
    all_metrics = {**basic_metrics, **prob_metrics}
    return all_metrics
```

## Threshold Optimization

The framework provides comprehensive threshold optimization tools for binary classification tasks.

### Finding Optimal Thresholds

Use `optimize_threshold()` to find the best decision threshold for a specific metric:

```python
from cadet.evaluation import optimize_threshold

# Optimize for F1 macro score
best_threshold, best_f1 = optimize_threshold(
    y_true_val,
    y_proba_val,
    metric='f1_macro',
    thresholds=np.linspace(0.05, 0.95, 19)
)
```

**Supported metrics:**

- `f1_macro`: Macro F1-score (default, balances precision and recall across classes)
- `f1`: Binary F1-score
- `precision`: Precision score
- `recall`: Recall score
- `accuracy`: Overall accuracy

**Methodology:**

- **Range**: Test thresholds from 0.05 to 0.95 in 19 steps (configurable)
- **Optimization Target**: User-specified metric
- **Validation**: Use cross-style validation set (opposite style from training)

**Benefits:**

- Adapts to dataset-specific characteristics
- Improves minority class performance
- Prevents overfitting to default 0.5 threshold
- Maximizes chosen evaluation metric

## Representation Analysis

### t-SNE Visualization

The framework provides comprehensive tools for visualizing high-dimensional embeddings using t-SNE.

#### Basic t-SNE Visualization

```python
from cadet.evaluation import (
    prepare_tsne_embeddings,
    plot_tsne_visualization
)

# Prepare t-SNE embeddings
tsne_embeddings, labels = prepare_tsne_embeddings(
    embeddings,
    labels,
    perplexity=30,
    n_iter=1000,
    use_pca=True,
    pca_components=50
)

# Generate visualization
fig = plot_tsne_visualization(
    tsne_embeddings,
    labels,
    label_names=['Non-Hate', 'Hate'],
    title='t-SNE: Hate Detection Embeddings',
    save_path='tsne_plot.png'
)
```

**Features:**

- Automatic PCA preprocessing for high-dimensional data
- Configurable perplexity and iteration parameters
- Custom label names and styling
- Save to file or return figure object

## CADET-Specific Evaluation

The CADET model provides specialized causal analysis capabilities to verify proper disentanglement
of latent variables.

### Causal Alignment Verification

The `CausalEvaluator` verifies that the motivation variable (M) is the strongest predictor of hate
labels, indicating proper causal alignment.

```python
from cadet.evaluation import CausalEvaluator

evaluator = CausalEvaluator(
    output_path="./results",
    enable_embeddings=True,
    num_counterfactual_samples=5
)

# Evaluate with causal analysis
results = evaluator.evaluate_with_causal_analysis(
    predictions_file="test_predictions.csv",
    latents_file="latents.npz",
    generate_visualizations=True
)

# Check causal alignment
causal_analysis = results['causal_analysis']
print(f"M->Hate accuracy: {causal_analysis['hate_from_M']:.4f}")
print(f"T->Hate accuracy: {causal_analysis['hate_from_T']:.4f}")
print(f"S->Hate accuracy: {causal_analysis['hate_from_style']:.4f}")
print(f"U->Hate accuracy: {causal_analysis['hate_from_U']:.4f}")
print(f"Properly aligned: {causal_analysis['properly_aligned']}")
```

### Causal Analysis Metrics

The evaluator trains linear probes to measure how well each latent variable predicts hate labels:

- **M (Motivation)**: Should have **highest** accuracy predicting hate (primary causal path)
- **T (Target)**: Should predict hate only through M (weaker than M)
- **S (Style)**: Should predict hate only through M (weaker than M)
- **U (Confounder)**: Should **not** predict hate after debiasing (lowest accuracy)

**Proper alignment criteria:**

```text
hate_from_M > hate_from_T and
hate_from_M > hate_from_style and
hate_from_M > hate_from_U
```

### Visualization Outputs

When `generate_visualizations=True`, the evaluator produces:

1. **causal_alignment.png**: Bar chart comparing prediction accuracy from each latent variable
2. **latent_tsne.png**: 2×2 grid of t-SNE plots for zm, zt, z_style, zu
3. **training_losses.png**: Loss curves with progressive weighting annotations
4. **tsne_plot.png**: General embeddings visualization (from embeddings.npz)

### Example Output

```python
{
    'accuracy': 0.8234,
    'precision': 0.7956,
    'recall': 0.8512,
    'f1': 0.8224,
    'auc_roc': 0.8891,
    'aupr': 0.8734,
    'causal_analysis': {
        'hate_from_M': 0.8234,      # Highest - proper alignment
        'hate_from_T': 0.6145,      # Lower than M
        'hate_from_style': 0.5823,  # Lower than M
        'hate_from_U': 0.5234,      # Lowest - successful debiasing
        'properly_aligned': True
    },
    'causal_alignment_plot_path': 'visualizations/causal_alignment.png',
    'latent_tsne_plot_path': 'visualizations/latent_tsne.png',
    'training_losses_plot_path': 'visualizations/training_losses.png',
    'tsne_plot_path': 'visualizations/tsne_plot.png'
}
```

## Running Experiments

The CADET framework uses a unified experiment runner:

```bash
# Run single experiment
python scripts/run_experiments.py -cn=cadet

# Run with different configurations
python scripts/run_experiments.py -cn=simple_baselines
python scripts/run_experiments.py -cn=llm_guard
```

See [scripts/README.md](../scripts/README.md) for detailed usage.

**Pipeline-Specific Evaluators:**

- **CADET**: `CausalEvaluator` (causal alignment verification)
- **Simple Baselines**: `EnhancedEvaluator` (t-SNE visualization)
- **LLM Guard**: `SimpleEvaluator` (classification metrics)

## Output Directory Structure

All evaluators save results to:

```bash
{output_path}/{run_name}/
├── metrics/
│   └── metrics.json              # All computed metrics
├── predictions/
│   └── test_predictions.csv      # Predictions with probabilities
├── models/
│   ├── embeddings.npz            # Model embeddings (Simple Baselines, CADET)
│   ├── latents.npz               # CADET latents (M, T, S, U)
│   ├── training_losses.csv       # Per-batch losses (CADET)
│   └── checkpoints/              # Model checkpoints
├── visualizations/
│   ├── roc_curve.png             # ROC curve
│   ├── pr_curve.png              # Precision-Recall curve
│   ├── tsne_plot.png             # t-SNE embeddings (Enhanced/Causal evaluators)
│   ├── causal_alignment.png      # CADET only: Variable predictiveness
│   ├── latent_tsne.png           # CADET only: Multi-latent visualization
│   └── training_losses.png       # CADET only: Loss curves
└── reports/
    └── evaluation_report.txt     # Human-readable summary
```

## Evaluator Classes

The CADET framework provides three levels of evaluators for different use cases.

### SimpleEvaluator

Basic evaluator for classification metrics without embeddings analysis.

```python
from cadet.evaluation import SimpleEvaluator

evaluator = SimpleEvaluator(
    output_path="./results",
    metrics=["accuracy", "precision", "recall", "f1", "auc_roc", "aupr"]
)

# Evaluate predictions
results = evaluator.evaluate(predictions_file="test_predictions.csv")
```

**Computes:**

- Classification metrics: accuracy, precision, recall, F1
- Confusion matrix components: TP, FP, TN, FN
- Probability metrics: AUC-ROC, AUPR

**Generates:**

- `metrics.json`: All computed metrics
- `roc_curve.png`: ROC curve plot
- `pr_curve.png`: Precision-Recall curve plot

### EnhancedEvaluator

Extends SimpleEvaluator with t-SNE visualization capabilities.

```python
from cadet.evaluation import EnhancedEvaluator

evaluator = EnhancedEvaluator(
    output_path="./results",
    metrics=["accuracy", "precision", "recall", "f1", "auc_roc", "aupr"],
    enable_embeddings=True
)

# Evaluate with embeddings
results = evaluator.evaluate_with_embeddings(
    predictions_file="test_predictions.csv",
    embeddings_file="embeddings.npz"
)
```

**Additional outputs:**

- `tsne_plot.png`: t-SNE visualization of embeddings
- Embedding quality metrics (silhouette score, class separation)

### CausalEvaluator

Specialized evaluator for CADET model with causal analysis.

```python
from cadet.evaluation import CausalEvaluator

evaluator = CausalEvaluator(
    output_path="./results",
    enable_embeddings=True,
    num_counterfactual_samples=5
)

# Full causal evaluation
results = evaluator.evaluate_with_causal_analysis(
    predictions_file="test_predictions.csv",
    latents_file="latents.npz",
    generate_visualizations=True
)
```

**Additional capabilities:**

- Causal alignment verification (M, T, S, U → Hate predictiveness)
- Multi-latent t-SNE visualizations
- Training loss curve plotting
- Causal alignment bar charts
