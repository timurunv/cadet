# Causality Guided Representation Learning for Cross-Style Hate Speech Detection

<div align="center">

**CADET**: **C**ausal-**A**ware **D**isentanglement for **E**xplicit/Implicit **T**ransfer

[![Paper][badge-paper]][paper]
[![Conference][badge-conference]][www2026]
[![Dataset][badge-dataset]][hf-dataset]
[![DOI](https://zenodo.org/badge/1138061385.svg)](https://doi.org/10.5281/zenodo.18331975)
</div>

---

CADET is a comprehensive framework for hate speech detection research with focus on cross-style
generalization between explicit and implicit hate speech.

> [!WARNING]
> This project and associated datasets involve hate speech and other offensive content. Handle
> with care and follow your organization's ethics and safety policies.

![CADET](/docs/figures/CADET.png)

**:trophy: Accepted at [The Web Conference 2026][www2026] (WWW'26)!**

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

# Set up Hugging Face authentication (for LLM Guard models)
export HF_TOKEN="your_hf_token_here"
```

## Data

- IE-HSC (Implicit-Explicit Hate Speech Corpus) is hosted on Hugging Face:
  [Shuwan/cadet-datasets](https://huggingface.co/datasets/Shuwan/cadet-datasets)
- Data cards, schema, warnings, and access notes: [docs/DATASETS.md](docs/DATASETS.md)
- Small samples for testing/debugging: [data/samples/](data/samples/)

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
| [DATASETS.md](docs/DATASETS.md) | Dataset structure, samples, and safety notes |
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

This project is developed with the help of AI coding assistant tools such as GitHub Copilot and
ChatGPT.

## Citation

If you use CADET in your research, please cite:

```bibtex
@article{zhao2025causality,
  title={Causality Guided Representation Learning for Cross-Style Hate Speech Detection},
  author={Zhao, Chengshuai and Wan, Shu and Sheth, Paras and Patwa, Karan and Candan, K Sel{\c{c}}uk and Liu, Huan},
  journal={arXiv preprint arXiv:2510.07707},
  year={2025}
}

@dataset{cadet_dataset_2026,
  title={cadet-datasets},
  url={https://huggingface.co/datasets/Shuwan/cadet-datasets},
  DOI={10.57967/HF/7615},
  publisher={Hugging Face},
  author={Chengshuai Zhao, Shu Wan, Paras Sheth, Karan Patwa, K. Selçuk Candan, Huan Liu}, year={2026}
}

@software{shu_2026_18331976,
  author       = {Shu},
  title        = {Shu-Wan/cadet: Initial release (v1.0.0)},
  month        = jan,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.18331976},
  url          = {https://doi.org/10.5281/zenodo.18331976},
  swhid        = {swh:1:dir:ceb3fc11806a24207cb0e43fd92b9b6f106f4cc2
                   ;origin=https://doi.org/10.5281/zenodo.18331975;vi
                   sit=swh:1:snp:402265cabd7f46dfbeeadf2a29ed9ed9a173
                   48a8;anchor=swh:1:rel:014aced8c618a22ad8ffd659e14f
                   8c8e29e11440;path=Shu-Wan-cadet-678b13c
                  },
}
```

## License

This code is licensed under the MIT License.

The dataset is derived from multiple upstream sources and may be governed by different licenses
and terms than the code in this repository. Refer to the dataset card on Hugging Face and
[docs/DATASETS.md](docs/DATASETS.md) for details.

[badge-conference]: https://img.shields.io/badge/WWW-2026-blue.svg
[badge-dataset]: https://img.shields.io/badge/🤗-Dataset-yellow.svg
[badge-paper]: https://img.shields.io/badge/Paper-arXiv-b31b1b.svg
[badge-python]: https://img.shields.io/badge/Python-3.12+-blue.svg
[hf-dataset]: https://huggingface.co/datasets/Shuwan/cadet-datasets
[paper]: https://arxiv.org/abs/2510.07707
[python-downloads]: https://www.python.org/downloads/
[www2026]: https://www2026.thewebconf.org/accepted.html
