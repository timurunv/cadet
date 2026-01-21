# Implicit-Explicit Hate Speech Corpus (IE-HSC) Data Cards

> **Dataset:** Implicit-Explicit Hate Speech Corpus (IE-HSC)
>
> **Version:** v1.0
>
> **Last updated:** 2026-01-21

Implicit-Explicit Hate Speech Corpus (IE-HSC) is a dataset contains implicit/ explicit style hate
speech samples from multiple sources. It is designed for research on hate speech detection,
especially cross-style generalization between implicit and explicit hate speech.

> [!WARNING]
> IE-HSC contains hateful, harassing, and otherwise offensive language, including identity-based
> attacks and slurs. This content may be distressing and can cause harm if misused.
>
> Use this dataset responsibly. Follow your organization's ethics and safety policies, minimize
> exposure to raw text, and avoid redistributing offensive examples unless necessary for research.
> IE-HSC is intended for research and educational use only.
>
> These notes describe intended use and precautions; they do not replace the applicable licenses
> and terms of the underlying datasets.

IE-HSC is based on four open-source datasets:

- [AbuseEval (Zampieri, Marcos, et al. 2019; Caselli, Tommaso, et al. 2020)][abuseeval]
- [DynaHate (Vidgen, Bertie, et al. 2021)][dynahate]
- [Implicit-Hate-Corpus (ElSherief, Mai, et al. 2021)][implicit-hate-corpus]
- [IsHate (Ocampo, Nicolás Benjamín, et al. 2023)][ishate]

## Dataset Access

IE-HSC is hosted at [🤗 Hugging Face][hf-gated-datasets].

We also provide small samples in `data/samples/` for testing and debugging without downloading
the full dataset.

## Datasets Overview

| Dataset              | Split          | Total Count | Hate Ratio | Style Ratio |
| -------------------- | -------------- | ----------: | ---------: | ----------: |
| AbuseEval            | explicit_test  |         589 |      98.1% |       17.8% |
| AbuseEval            | explicit_train |       2,353 |      98.8% |       17.4% |
| AbuseEval            | implicit_test  |       2,060 |      14.0% |        0.5% |
| AbuseEval            | implicit_train |       8,238 |      14.7% |        0.6% |
| DynaHate             | explicit_test  |       2,206 |      65.6% |      100.0% |
| DynaHate             | explicit_train |       8,823 |      64.1% |      100.0% |
| DynaHate             | implicit_test  |       6,021 |      49.6% |        0.0% |
| DynaHate             | implicit_train |      24,084 |      50.2% |        0.0% |
| Implicit-Hate-Corpus | explicit_test  |         301 |      70.8% |       64.5% |
| Implicit-Hate-Corpus | explicit_train |       1,201 |      72.8% |       64.9% |
| Implicit-Hate-Corpus | implicit_test  |       4,028 |      35.5% |        3.5% |
| Implicit-Hate-Corpus | implicit_train |      16,110 |      36.2% |        4.0% |
| IsHate               | explicit_test  |       2,732 |      86.3% |       51.5% |
| IsHate               | explicit_train |      10,928 |      86.0% |       52.8% |
| IsHate               | implicit_test  |       9,673 |      67.4% |        0.6% |
| IsHate               | implicit_train |      38,689 |      67.2% |        0.6% |

## Data Structure

Each sample in the Dataset contains:

- `text_id`: ID for each text sample (int64, sequential index)
- `text`: Input text content (string)
- `hate_label`: Binary hate label (0=non-hate, 1=hate)
- `avg`: Average Perspective API toxicity score (float, 0.0-1.0)
- `style`: Binary toxicity derived from `avg` (0=non-toxic, 1=toxic)
- `true_style`: Style from file naming (0=implicit, 1=explicit)
- `target`: Target demographic group (string)
- `target_conf`: Confidence score for target annotation (float, 0.0-1.0)

## Provenance

1. Label standardization

   - Columns `label`, `hateful_layer`, or `hate_label` are normalized to `hate_label`
   - `hate_label` is cast to a `ClassLabel` with names `["non_hate", "hate"]`

2. How `avg` Is Calculated
   - We used the Perspective API to fetch toxicity scores for the following attributes, then averaged
  them per example: `TOXICITY, SEVERE_TOXICITY, IDENTITY_ATTACK, INSULT, PROFANITY, THREAT`

   - The resulting mean is stored in the `avg` column.

3. Toxicity and Style

   - Toxicity is derived from `avg` using a threshold:
     - If `avg` is missing, default to 0.0
     - `avg >= 0.4` => `style = 1` (explicit)
     - `avg < 0.4` => `style = 0` (implicit)
   - `true_style` is set strictly from the dataset file name, not from `avg`:
     - Files named `explicit_*` set `true_style = 1`
     - Files named `implicit_*` set `true_style = 0`
     - `true_style` is from the original dataset (except for DynaHate, in which `style` equals `true_style`)

## Usage

By default, the dataset is loaded from the Hugging Face Hub. Local loading is also supported.

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

# Using the helper function
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
sample = train_dataset[0]
print(f"Sample: {sample}")
```

> [!NOTE]
> It's allowed to set `source_style` and `target_style` the same for in-domain experiments.

## Intended Use

IE-HSC is intended for **research and educational use only** (non-commercial), such as:

- Studying implicit vs. explicit hate speech
- Training and evaluating hate/toxicity classifiers
- Benchmarking robustness, bias mitigation, and safety interventions

### Out-of-Scope / Misuse

- Using the data to generate, amplify, or target hateful content
- Deploying models trained on IE-HSC in user-facing or high-stakes settings without additional
  validation, safeguards, and human oversight
- Treating `target` labels or toxicity scores as ground truth about individuals or communities

### Ethical Considerations

- Labels and target annotations may be noisy and reflect the biases of source datasets and
  automated scoring tools.
- If you share examples (papers, demos, model cards), include content warnings and avoid quoting
  slurs or targeting language unless strictly necessary.

## LICENSE

IE-HSC is licensed under the [CC BY-NC-SA 4.0 license][cc-by-nc-sa-4.0].
Because IE-HSC is derived from multiple sources, please also review and comply with the licenses
and terms of the underlying datasets.
The upstream datasets may use different licenses (e.g., CC BY-NC-SA 4.0, CC BY 4.0, MIT, BSL-1.0).

## Citation

If you use IE-HSC in your research, please cite:

```bibtex
@article{zhao2025causality,
  title={Causality Guided Representation Learning for Cross-Style Hate Speech Detection},
  author={Zhao, Chengshuai and Wan, Shu and Sheth, Paras and Patwa, Karan and Candan, K Sel{\c{c}}uk and Liu, Huan},
  journal={arXiv preprint arXiv:2510.07707},
  year={2025}
}
```

<!-- link references -->
[abuseeval]: https://github.com/tommasoc80/AbuseEval
[cc-by-nc-sa-4.0]: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en/
[dynahate]: https://github.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset
[hf-gated-datasets]: https://huggingface.co/docs/hub/en/datasets-gated
[implicit-hate-corpus]: https://github.com/SALT-NLP/implicit-hate
[ishate]: https://huggingface.co/datasets/BenjaminOcampo/ISHate
