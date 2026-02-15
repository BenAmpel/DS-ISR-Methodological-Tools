
# [Techniques for Automated Machine Learning](https://dl.acm.org/doi/pdf/10.1145/3447556.3447567)
*Adapted from Chen et al., 2021 — updated for the current AutoML landscape.*

> **Note:** The AutoML ecosystem has consolidated since 2021. Several academic projects have been abandoned or superseded. The tools below represent the active, maintained packages most relevant to IS researchers.

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
