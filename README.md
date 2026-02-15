# Methodological AI Tools For Design Science Information Systems Researchers

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
![GitHub](https://img.shields.io/github/last-commit/BenAmpel/DS-ISR-Methodological-Tools)
![GitHub](https://img.shields.io/github/followers/BenAmpel?style=plastic)
![GitHub](https://img.shields.io/github/stars/BenAmpel/DS-ISR-Methodological-Tools?style=social)

*A repository of tools found in top conferences to aid in method identification and application to IS research.*

*Last updated: February 2026*

- Remember to always maintain high **ethical** standards when building AI models. Read more about [ethical guidelines for AI here](Files/Ethics).
- New to machine learning? [Read this paper](https://arxiv.org/abs/2108.02497) to avoid common pitfalls and work through [hands-on tutorials](https://github.com/aladdinpersson/Machine-Learning-Collection). Find frameworks that do the heavy lifting in [AutoML](Files/AutoML).
- New to Python? [Follow these mini projects to hone your skills](https://github.com/Python-World/python-mini-projects).

---

## The Automation-of-Invention Framework
*Based on the framework introduced in Information Systems Research (2025), AI tools for IS researchers can be classified by their role in the research process:*

| Level | Role | Description | Example Tools |
|-------|------|-------------|---------------|
| **Level I** | **Copy Editor** | Drafting, polishing, and translating research artifacts | DeepL Write, GrammarlyGO, Lex |
| **Level II** | **Research Assistant** | Literature review, data collection, and processing pipelines | Elicit, Research Rabbit, LangChain, LlamaIndex |
| **Level III** | **Super-Collaborator** | Autonomous agents that simulate users, explore theory, or co-generate artifacts | Multi-Agent Systems, Generative Agents, DSPy |

> *At Level III, AI moves from assisting the researcher to actively participating in the invention process — simulating stakeholders, stress-testing theory, and generating novel design alternatives.*

---

## Topics
*More Targeted Sections for New and Exciting AI Research.*
| | | |
|-|-|-|
| [:chart_with_upwards_trend: Graphs](Files/Graphs) | [📜 LLMs & Natural Language Processing](Files/NaturalLanguageProcessing) | [:chess_pawn: Reinforcement Learning](Files/ReinforcementLearning)
| [💪 Generative Media & Synthetic Data](Files/DataGeneration)|[🤖 LLM Safety & Adversarial Defense](Files/AdversarialDefense)| [:red_circle: Anomaly Detection](Files/AnomalyDetection)
| [:snake: Python Tools](Files/PythonTools) | [:bulb: AI for Research Productivity](Files/AI-for-Research-Productivity) | [:speech_balloon: Prompt Engineering](Files/Prompt-Engineering) |
| [:eyes: Multimodal Models](Files/MultimodalModels) | [:wrench: Fine-Tuning](Files/FineTuning) | [:mag: Interpretability](Files/Interpretability) |
| [:balance_scale: Ethics](Files/Ethics) | [:triangular_ruler: Evaluation & Benchmarking](Files/Evaluation) | [:seedling: Causal Inference](Files/Causal-Inference) |
| [:high_brightness: Attention Mechanisms](https://github.com/xmu-xiaoma666/External-Attention-pytorch) | [:arrow_right: Transfer Learning](https://github.com/jindongwang/transferlearning) | [:paintbrush: Images](https://github.com/rwightman/pytorch-image-models) |

---

## Quick Start: I Want To...

| Goal | Where to Start |
|-|-|
| Review the literature on a topic | [AI for Research Productivity](Files/AI-for-Research-Productivity) → Literature Review |
| Build a chatbot that answers questions over documents | [Prompt Engineering](Files/Prompt-Engineering) → Agentic Frameworks (RAG) |
| Fine-tune a model on my domain data | [Fine-Tuning](Files/FineTuning) → When to Fine-Tune decision table |
| Generate synthetic survey respondents for IS experiments | [Generative Media & Synthetic Data](Files/DataGeneration) → Synthetic Users |
| Detect fraud or anomalous behavior in enterprise data | [Anomaly Detection](Files/AnomalyDetection) or [Graphs](Files/Graphs) → GraphRAG |
| Explain why my model made a prediction | [Interpretability](Files/Interpretability) → SHAP / Counterfactuals |
| Evaluate whether my LLM artifact actually works | [Evaluation & Benchmarking](Files/Evaluation) → RAG Evaluation / LLM-as-a-Judge |
| Understand causal effects (not just correlations) in IS data | [Causal Inference](Files/Causal-Inference) → DoWhy / EconML |
| Ensure my AI artifact is fair, ethical, and compliant | [Ethics](Files/Ethics) → Responsible AI Checklist |
| Deploy a prototype for anonymous reviewers to test | [Python Tools](Files/PythonTools) → Artifact Deployment |

---

## Learning Resources & Getting Started

### Literature Discovery
| Tool | Description |
|-|-|
| [Connected Papers](https://www.connectedpapers.com/) | Visual graph of papers related to a seed paper — rapidly orient to a new field. |
| [Elicit](https://elicit.com/) | LLM-powered research assistant that searches Semantic Scholar and builds comparison tables. |
| [Research Rabbit](https://www.researchrabbit.ai/) | Discover related papers and map forward/backward citation networks. |
| [Semantic Scholar](https://www.semanticscholar.org/) | AI-powered search engine with paper summarization and influence metrics. |

> For a full list of AI-powered literature tools, see [AI for Research Productivity](Files/AI-for-Research-Productivity).

### Foundational Learning
| Resource | Description |
|-|-|
| [ML Pitfalls Paper](https://arxiv.org/abs/2108.02497) | Practical guide to avoiding common machine learning mistakes — essential reading before starting any IS+ML project. |
| [Machine Learning Collection](https://github.com/aladdinpersson/Machine-Learning-Collection) | Hands-on PyTorch tutorials from basics through advanced architectures, with runnable notebooks. |
| [Python Mini Projects](https://github.com/Python-World/python-mini-projects) | Beginner Python projects to build fluency before tackling data pipelines. |
| [fast.ai Practical Deep Learning](https://course.fast.ai/) | The most accessible deep learning course. Top-down, code-first approach ideal for IS researchers. |
| [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course/) | Free end-to-end course on transformers, fine-tuning, and deploying NLP models with HuggingFace. |

---

## Cookbooks & End-to-End Tutorials
Practical, step-by-step guides for common IS research workflows — bridging tools into complete pipelines.

| Tutorial | Description |
|-|-|
| [LangChain RAG Cookbook](https://python.langchain.com/docs/tutorials/rag/) | Build a retrieval-augmented generation pipeline over a document corpus — end-to-end from ingestion to response. |
| [OpenAI Cookbook: Building Agents](https://cookbook.openai.com/examples/agents_sdk/multi-agent-portfolio-collaboration) | How to build multi-step tool-using agents with the OpenAI API. |
| [HuggingFace Fine-Tuning Tutorials](https://huggingface.co/docs/transformers/training) | Official step-by-step guide to fine-tuning transformers on custom datasets. |
| [RAGAS: Evaluating RAG Pipelines](https://docs.ragas.io/en/stable/getstarted/) | How to measure faithfulness, relevance, and context precision in RAG-based IS artifacts. |
| [DoWhy Causal Inference Tutorial](https://py-why.github.io/dowhy/main/example_notebooks/dowhy_simple_example.html) | End-to-end causal effect estimation from observational IS data using the four-step DoWhy framework. |
| [Streamlit App Gallery](https://streamlit.io/gallery) | Deployed examples to inspire DSR artifact prototypes — find patterns similar to your artifact design. |

---

## Contributing
Want to suggest a paper, tool, or new section? See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. Contributions that connect AI/ML methods to IS and DSR research are especially welcome.

---
