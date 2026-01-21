# Causality Guided Representation Learning for Cross-Style Hate Speech Detection

<div align="center">

**CADET**: **C**ausal-**A**ware **D**isentanglement for **E**xplicit/Implicit **T**ransfer

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2510.07707)
[![Conference](https://img.shields.io/badge/WWW-2026-blue.svg)](https://www2026.thewebconf.org/accepted.html)
[![Dataset](https://img.shields.io/badge/🤗-Dataset-yellow.svg)](https://huggingface.co/datasets/Shuwan/cadet-datasets)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)

</div>

---

CADET is a comprehensive framework for hate speech detection research with focus on cross-style
generalization between explicit and implicit hate speech.

![CADET](/docs/figures/CADET.png)

**:trophy: Accepted at [The Web Conference 2026](https://www2026.thewebconf.org/accepted.html) (WWW'26)!**

## Quick Start

```bash
# Clone and setup
git clone https://github.com/Shu-Wan/cadet.git
cd cadet
uv sync --all-groups

# Run CADET experiment
uv run python scripts/run_experiments.py
```

## Installation

**Requirements:** Python 3.12+, uv, CUDA-compatible GPU (recommended)

```bash
# Clone and setup
git clone https://github.com/Shu-Wan/cadet.git
cd cadet
uv sync --all-groups

# Preprocess datasets
uv run python scripts/preprocess_all_datasets.py

# Set up Hugging Face authentication (for LLM Guard models)
export HF_TOKEN="your_hf_token_here"
```

## Usage

### Running Experiments

All experiments use the unified runner `run_experiments.py`:

```bash
# Single experiments (select config)
uv run python scripts/run_experiments.py -cn=cadet
uv run python scripts/run_experiments.py -cn=simple_baselines
uv run python scripts/run_experiments.py -cn=llm_guard

# Override parameters
uv run python scripts/run_experiments.py -cn=cadet \
    data.dataset_name=DynaHate \
    data.source_style=implicit

# Multirun mode (predefined configs)
uv run python scripts/run_experiments.py -cn=cadet_multirun
uv run python scripts/run_experiments.py -cn=simple_baselines_multirun
uv run python scripts/run_experiments.py -cn=llm_guard_multirun

# Multirun from command line
uv run python scripts/run_experiments.py -cn=cadet -m \
    data.dataset_name=AbuseEval,DynaHate \
    data.source_style=explicit,implicit

# Ad-hoc testing with limited samples
uv run python scripts/run_experiments.py -cn=cadet +adhoc.n_samples=100
```

For detailed configuration options, see [configs/README.md](configs/README.md).

## Documentation

| Document | Description |
|----------|-------------|
| [configs/README.md](configs/README.md) | Configuration files and options |
| [scripts/README.md](scripts/README.md) | Running experiments and utility scripts |
| [CADET.md](docs/CADET.md) | CADET model architecture and causal analysis |
| [DATASETS.md](docs/DATASETS.md) | Dataset structure and preprocessing |
| [BASELINES.md](docs/BASELINES.md) | Baseline model implementations |
| [EVALUATIONS.md](docs/EVALUATIONS.md) | Evaluation metrics and analysis |
| [DESIGN.md](docs/DESIGN.md) | System architecture and design principles |
| [TRAINING.md](docs/TRAINING.md) | Training recipes and methodologies |

## Development

```bash
# Run linting
ruff check src/ scripts/

# Run pre-commit hooks
pre-commit run --all-files
```

## Usage of GenAI

This project is developed with the help of AI coding assistant tools such as GitHub Copilot and ChatGPT.

## Citation

If you use CADET in your research, please cite:

```bibtex
@article{zhao2025causality,
  title={Causality Guided Representation Learning for Cross-Style Hate Speech Detection},
  author={Zhao, Chengshuai and Wan, Shu and Sheth, Paras and Patwa, Karan and Candan, K Sel{\c{c}}uk and Liu, Huan},
  journal={arXiv preprint arXiv:2510.07707},
  year={2025}
}
```

## License

This project is licensed under the MIT License.
