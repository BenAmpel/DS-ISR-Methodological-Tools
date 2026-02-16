---
layout: default
title: "AutoML"
parent: Topics
nav_order: 16
permalink: /Files/AutoML/
---


{::nomarkdown}
<div class="section-meta">
  <span class="meta-pill meta-reading">⏱ ~5 min read</span>
  <span class="meta-pill meta-updated">📅 Updated Feb 2026</span>
  <span class="meta-pill meta-beginner">📊 Beginner</span>
  <span class="meta-pill meta-prereq">🔑 Python basics</span>
</div>
{:/}


# [Techniques for Automated Machine Learning](https://dl.acm.org/doi/pdf/10.1145/3447556.3447567)
*Adapted from Chen et al., 2021 — updated for the current AutoML landscape.*

*Last updated: February 2026*

> **Note:** The AutoML ecosystem has consolidated since 2021. Several academic projects have been abandoned or superseded. The tools below represent the active, maintained packages most relevant to IS researchers.

> **IS Research Applications:** Rapidly prototype IS artifact classifiers without deep ML expertise; run reproducible hyperparameter searches for fair model comparison in IS papers; automate feature engineering for tabular IS datasets (survey data, log data, transaction records).

| **Python Package** | **Description** | **Status** |
|-|-|-|
| [AutoGluon](https://auto.gluon.ai) | Automated ML for tabular, text, image, and multimodal data. Consistently top performer on tabular benchmarks. | Active |
| [FLAML](https://github.com/microsoft/FLAML) | Fast & Lightweight AutoML from Microsoft Research. Low compute budget optimization with strong default performance on tabular data. | Active |
| [TPOT](https://github.com/EpistasisLab/tpot) | Tree-based pipeline optimization using genetic algorithms over Scikit-Learn pipelines. | Active |
| [MLJAR](https://github.com/mljar/mljar-supervised) | Automated supervised ML with explainability reports (SHAP, feature importance). Produces human-readable markdown reports alongside models. | Active |
| [Auto-Sklearn](https://github.com/automl/auto-sklearn) | Meta learning-based ensemble selection over Scikit-Learn. Well-studied academic baseline. | Active |
| [Auto-PyTorch](https://github.com/automl/Auto-PyTorch) | Neural architecture search in the PyTorch framework for fully automated deep learning. | Active |
| [Auto-Keras](https://github.com/keras-team/autokeras) | Accessible deep learning built on top of the Keras library. | Active |
| [Neural Network Intelligence (NNI)](https://github.com/microsoft/nni) | Toolkit to automate Feature Engineering, Neural Architecture Search, Hyperparameter Tuning, and Model Compression. | Active |
| [Optuna](https://github.com/optuna/optuna) | Efficient hyperparameter optimization with define-by-run API; integrates with PyTorch, Keras, and XGBoost. | Active |
| [Feature Tools](https://github.com/Featuretools/featuretools) | Automated feature engineering for relational and time series data. Generates descriptive feature sets from raw tables. | Active |
| [Hyperopt](https://github.com/hyperopt/hyperopt) | Hyperparameter optimization over arbitrary search spaces using Tree of Parzen Estimators (TPE). | Active |
| [Scikit-Optimize](https://github.com/scikit-optimize/scikit-optimize) | Sequential model-based optimization built over Scikit-Learn. | Maintenance |
| [Rasa](https://github.com/RasaHQ/rasa) | Open source ML framework for text- and voice-based conversational AI. | Active |

---

## Conversational / LLM-Based AutoML
The emergence of LLMs has introduced a new paradigm where natural language is used to specify, generate, and evaluate ML pipelines.

| **Tool** | **Description** |
|-|-|
| [AutoGen](https://github.com/microsoft/autogen) | Multi-agent framework where LLM agents collaborate to write, execute, and debug ML code. Enables conversational AutoML workflows. |
| [MLflow](https://mlflow.org/) | Experiment tracking, model registry, and deployment. Logs parameters, metrics, and artifacts across AutoML runs for reproducibility. |
| [Weights & Biases (W&B)](https://wandb.ai/) | Cloud-based experiment tracking with sweep (hyperparameter search) functionality. Integrates with all major ML frameworks. |

---

## Tabular Foundation Models
A major 2024–2025 development: foundation models pre-trained on large collections of tabular datasets that can be fine-tuned or prompted for new tabular tasks — analogous to LLMs for text. Particularly relevant for IS research using survey, transaction, and enterprise data.

| **Tool** | **Description** |
|-|-|
| [TabPFN](https://github.com/automl/TabPFN) | A prior-fitted network trained on synthetic tabular data. Performs in-context learning on small tabular datasets (<1000 rows) in milliseconds without fitting. State-of-the-art on small-data IS tasks. [Paper](https://arxiv.org/abs/2207.01848) |
| [TabPFN v2](https://github.com/automl/tabpfn-client) | Updated version with improved scalability and performance, competitive with gradient-boosted trees on medium-sized datasets. [Paper](https://arxiv.org/abs/2501.02945) |
| [CARTE](https://github.com/soda-inria/carte) | Foundation model for heterogeneous tabular data that handles text-heavy columns via pretrained language encoders. Strong on real-world IS datasets with mixed types. [Paper](https://arxiv.org/abs/2402.16785) |

## AutoML Surveys & Meta-Learning

- [AutoML: A Survey of the State-of-the-Art](https://arxiv.org/abs/1908.00709), 2019 - Comprehensive survey of the AutoML landscape covering NAS, HPO, and meta-learning. Classic reference for IS papers using AutoML.
- [Why does AutoML fail? A Study on AutoML Failure Modes](https://arxiv.org/abs/2301.00978), 2023 - Systematic analysis of when AutoML underperforms manual pipelines — essential reading for IS researchers reporting AutoML baselines.
- [TabZilla: A Large-Scale Benchmark of Tabular Data Methods](https://arxiv.org/abs/2305.02997), 2023 - Comprehensive benchmark comparing 19 methods on 176 datasets. Reveals that tree-based methods (XGBoost, CatBoost) still often outperform deep learning and AutoML on tabular data. [Code](https://github.com/naszilla/tabzilla)

---

**Related Sections:** [Python Tools](../PythonTools/README.md) | [Fine-Tuning](../FineTuning/README.md) | [Prompt Engineering](../Prompt-Engineering/README.md) | [AI for Research Productivity](../AI-for-Research-Productivity/README.md)
