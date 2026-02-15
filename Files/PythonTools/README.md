# Python Data Mining Tools
*A curated collection of Python tools for data processing, visualization, machine learning, and LLM orchestration in IS research.*

*Last updated: February 2026*

> For a comprehensive list of Python libraries beyond this curated selection, see the [Awesome Python List](https://github.com/vinta/awesome-python).

---

## Core Data Science Stack

| **Category** | **Tool** | **Description** | **Link** |
|-|-|-|-|
| Data Processing | Pandas | The standard DataFrame library for tabular data manipulation | [Docs](https://pandas.pydata.org/) |
| Data Processing | **Polars** | Faster-than-Pandas DataFrame library written in Rust. Drop-in replacement for large datasets; lazy evaluation and multi-threaded by default | [GitHub](https://github.com/pola-rs/polars) |
| Data Processing | Label Studio | Create labels for supervised learning tasks with an interactive UI | [GitHub](https://github.com/heartexlabs/label-studio) |
| Data Processing | Detectron 2 | Parse and understand any document with deep learning | [Guide](https://towardsdatascience.com/auto-parse-and-understand-any-document-5d72e81b0be9) |
| Data Processing | Pandas Profiling | One-line exploratory data analysis (EDA) report generation | [Guide](https://towardsdatascience.com/how-to-do-exploratory-data-analysis-with-one-line-of-code-1364e16a102e) |
| Python Efficiency | Swifter | Speed up Pandas `apply()` with automatic parallelization | [Guide](https://towardsdatascience.com/10x-times-faster-pandas-apply-in-a-single-line-change-of-code-c42cb5e82f6d) |

---

## Visualization

| **Category** | **Tool** | **Description** | **Link** |
|-|-|-|-|
| Visualization | Plotly | Interactive chart library for publication-quality figures and dashboards | [Docs](https://plotly.com/python/) |
| Visualization | **Streamlit** | Converts Python scripts into interactive web apps for rapid DSR prototype deployment. No front-end knowledge required | [Docs](https://streamlit.io/) |
| Visualization | Autoplotter | Automated exploratory data analysis (EDA) visualization tool | `pip install autoplotter` |
| Visualization | Observable Summary Table | Get dataset summaries with interactive filtering | [Product](https://observablehq.com/product) |
| Visualization | Pretty Bar Graph Tutorial | Guide on creating publication-ready bar charts in Plotly | [Tutorial](https://towardsdatascience.com/tutorial-on-building-a-professional-bar-graph-in-plotly-python-ba8e63fda048) |

---

## Machine Learning & AutoML

| **Category** | **Tool** | **Description** | **Link** |
|-|-|-|-|
| Machine Learning | Randomized Search CV | Scikit-Learn hyperparameter search — more efficient than Grid Search for high-dimensional spaces | [Scikit-Learn Docs](https://scikit-learn.org/) |
| Hyperparameter Tuning | Optuna | Efficient hyperparameter optimization with define-by-run API; integrates with PyTorch, Keras, and XGBoost | [GitHub](https://github.com/optuna/optuna) |
| Deep Learning | External Attention | A comprehensive collection of attention mechanisms implemented in PyTorch | [GitHub](https://github.com/xmu-xiaoma666/External-Attention-pytorch) |

> See the full [AutoML](../AutoML/README.md) section for a complete list of automated machine learning tools.

---

## LLM Orchestration

| **Category** | **Tool** | **Description** | **Link** |
|-|-|-|-|
| LLM Orchestration | **LangChain** | The dominant framework for building LLM applications with retrieval chains, agents, and tools | [GitHub](https://github.com/langchain-ai/langchain) |
| LLM Orchestration | **LlamaIndex** | Data framework for connecting LLMs to structured and unstructured documents. Strong for complex indexing strategies | [GitHub](https://github.com/run-llama/llama_index) |

> See the full [Prompt Engineering](../Prompt-Engineering/README.md) section for a complete list of LLM frameworks and libraries.

---

## MLOps & Experiment Tracking

| **Category** | **Tool** | **Description** | **Link** |
|-|-|-|-|
| MLOps | **MLflow** | Experiment tracking, model registry, and deployment. Logs parameters, metrics, and artifacts for reproducible ML research | [Docs](https://mlflow.org/) |
| MLOps | **Weights & Biases (W&B)** | Cloud experiment tracking with hyperparameter sweeps. Integrates with PyTorch, Keras, and HuggingFace | [Docs](https://wandb.ai/) |

---

**Related Sections:** [AutoML](../AutoML/README.md) | [Prompt Engineering](../Prompt-Engineering/README.md) | [AI for Research Productivity](../AI-for-Research-Productivity/README.md)
