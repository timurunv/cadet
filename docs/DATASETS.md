# Datasets

We use four hate speech dataset for CADET

- [AbuseEval/](https://github.com/tommasoc80/AbuseEval)
- [DynaHate/](https://github.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset)
- [Implicit-Hate-Corpus/](https://github.com/SALT-NLP/implicit-hate)
- [IsHate/](https://huggingface.co/datasets/BenjaminOcampo/ISHate)

The modified version is available at [Hugging Face](https://huggingface.co/datasets/Shuwan/cadet-datasets):

Each dataset contains the following files:

- `{style}_hate_{split}.csv`, style in {implicit, explicit}, split in {train, test}
- and their `with_targets` variants: `*_with_targets.csv`

> [!Warning]
> `with_targets` has the same data as the counterpart, but with additional columns
> for target group and confidence.

## Field Descriptions

Across datasets, the following fields are standardized:

- `text_id`: Unique identifier for each text sample (int64, sequential index)
- `text`: Input text content (string)
- `hate_label`: Binary hate label (0=non-hate, 1=hate)
- `avg`: Average Perspective API toxicity score (float, 0.0-1.0)
- `style`: Binary toxicity derived from `avg` (0=non-toxic, 1=toxic)
- `true_style`: Style from file naming (0=implicit, 1=explicit)
- `target`: Target demographic group (string)
- `target_conf`: Confidence score for target annotation (float, 0.0-1.0)

## How `avg` Is Calculated

We used the Perspective API to fetch toxicity scores for the following attributes,
then averaged them per example:

- TOXICITY, SEVERE_TOXICITY, IDENTITY_ATTACK, INSULT, PROFANITY, THREAT

The resulting mean is stored in the `avg` column.

> [!Warning]
> However, we still named it as style in the dataset due to past implementaitons.
> We call the true style label `true_style` instead.

## Data Processing Logic

All datasets are processed to Hugging Face `Dataset` format and saved in `data/processed/`.

### 1) Label standardization

- Columns `label`, `hateful_layer`, or `hate_label` are normalized to `hate_label`
- `hate_label` is cast to a `ClassLabel` with names `["non_hate", "hate"]`

### 2) Toxicity and Style separation

- Toxicity is derived from `avg` using a threshold:
  - If `avg` is missing, default to 0.0
  - `avg >= 0.4` => `style = 1` (explicit)
  - `avg < 0.4` => `style = 0` (implicit)
- `true_style` is set strictly from the dataset file name, not from `avg`:
  - Files named `explicit_*` set `style = 1`
  - Files named `implicit_*` set `style = 0`
  - `true_style` is from the original dataset (Except for Dynahate)

> [!Warning]
> However, we still named it as style in the dataset due to past implementaitons.
> We call the true style label `true_style` instead.

### 3) Cross-style evaluation setup

For cross-style evaluation (implicit <-> explicit):

- Training uses the specified source style (from file naming)
- Validation/Test uses the opposite style

## GPT_Analysis.csv

### Structure

The synthetic evaluation dataset contains triples of related examples:

| text                                        | label |  style   |
| ------------------------------------------- | :---: | :------: |
| "black women are so terrifying"             |   1   | explicit |
| "melanin-endowed women make my skin crawl"  |   1   | implicit |
| "I want to understand and overcome anxiety" |   0   | non-hate |

## Processed Datasets

After processing, datasets are saved in HuggingFace Arrow format under `data/processed/`:

```text
data/processed/
├── AbuseEval/
│   ├── explicit-train/
│   ├── explicit-test/
│   ├── implicit-train/
│   └── implicit-test/
├── DynaHate/
│   └── ... (same structure)
├── Implicit-Hate-Corpus/
│   └── ... (same structure)
└── IsHate/
    └── ... (same structure)
```

Each subdirectory contains Hugging Face Dataset files:

- `dataset_info.json`: Dataset metadata
- `data-*.arrow`: Arrow format data files
- `state.json`: Processing state information

## Usage

### Basic Usage

```python
from cadet.datasets import HateSpeechDataset

# Load a single dataset split
loader = HateSpeechDataset(
    dataset="AbuseEval",
    style="explicit",
    split="train"
)
dataset = loader.data  # Returns a single Dataset object
print(f"Loaded {len(dataset)} samples")

# Load multiple splits
loader = HateSpeechDataset(
    dataset="AbuseEval",
    style=["explicit", "implicit"],
    split="train"
)
datasets = loader.data  # Returns a DatasetDict
print(f"Available splits: {list(datasets.keys())}")
print(f"Explicit train: {len(datasets['explicit-train'])} samples")
print(f"Implicit train: {len(datasets['implicit-train'])} samples")

# Load all combinations (default behavior)
loader = HateSpeechDataset(dataset="AbuseEval")
all_data = loader.data  # Returns DatasetDict with all style-split combinations
print(f"All splits: {list(all_data.keys())}")
```

### Hugging Face Hub Usage

For experimental use, datasets can be loaded from Hugging Face Hub:

```python
from cadet.datasets import HateSpeechDataset

# Load from HF Hub
loader = HateSpeechDataset(
    dataset="AbuseEval",
    root="Shuwan/cadet-datasets",
    style="explicit",
    split="train"
)
```

### Using the helper function

```python
from cadet.datasets import get_hate_speech_datasets

# Get train/test datasets for cross-style evaluation
# Train on explicit, test on implicit
train_dataset, test_dataset = get_hate_speech_datasets(
    dataset="AbuseEval",
    source_style="explicit"
)

print(f"Train samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Both are single Dataset objects ready for training
sample = train_dataset[0
print(f"Sample: {sample}")
```

> [!NOTE]
> It's allowed to set `source_style` and `target_style` the same for in-domain experiments.

## Data Structure

Each sample in the Hugging Face Dataset contains:

```python
{
    'text_id': int,        # Unique sequential identifier (int64)
    'text': str,           # Raw text
    'hate_label': int,     # Binary hate label (0=non-hate, 1=hate)
    'avg': float,          # Average Perspective API toxicity score
    'style': int,          # Binary style (0=implicit, 1=explicit)
    'true_style': int,     # Binary style from file (0=implicit, 1=explicit)
    'target': str,         # Target demographic (if available)
    'target_conf': float   # Target confidence (if available)
}
```

### Available Datasets and Constants

```python
from cadet.datasets import HateSpeechDataset

# Available datasets
print(HateSpeechDataset.AVAILABLE_DATASETS)
# ['AbuseEval', 'DynaHate', 'Implicit-Hate-Corpus', 'IsHate']

# Available styles
print(HateSpeechDataset.AVAILABLE_STYLES)
# ['implicit', 'explicit']

# Available splits
print(HateSpeechDataset.AVAILABLE_SPLITS)
# ['train', 'test']
```

## Preprocessing Raw Datasets

Run the command below to preprocess all datasets.

```bash
python scripts/preprocess_all_datasets.py
```

See [scripts/README.md](/scripts/README.md) for detailed usage options.
