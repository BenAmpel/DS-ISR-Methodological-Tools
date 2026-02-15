<div align="center">

# Methodological AI Tools for Design Science & IS Researchers

**A curated, living reference of AI/ML methods, tools, and tutorials — drawn from top IS conferences and journals — to support rigorous design science research.**

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![Last Commit](https://img.shields.io/github/last-commit/BenAmpel/DS-ISR-Methodological-Tools?color=blue)](https://github.com/BenAmpel/DS-ISR-Methodological-Tools/commits/main)
[![Stars](https://img.shields.io/github/stars/BenAmpel/DS-ISR-Methodological-Tools?style=social)](https://github.com/BenAmpel/DS-ISR-Methodological-Tools/stargazers)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen)](CONTRIBUTING.md)

*Last updated: February 2026*

</div>

---

## What Is This?

This repository collects AI and machine learning tools that are methodologically relevant to **Information Systems (IS) researchers** working in a **Design Science Research (DSR)** paradigm. It is organized around research tasks — literature discovery, artifact construction, evaluation, deployment — not just model types.

**Who is this for?**
- IS researchers integrating AI/ML into their research designs
- PhD students learning to apply deep learning methods in organizational or behavioral contexts
- Practitioners building AI-based artifacts for IS problems

> **Ethics first.** Always maintain high ethical standards when building AI models. → [Ethical Guidelines for AI](Files/Ethics)

---

## The Automation-of-Invention Framework

*Based on the framework introduced in* Information Systems Research *(2025), AI tools for IS researchers are classified by their role in the research process:*

| Level | Role | Description | Representative Tools |
|:-----:|------|-------------|----------------------|
| **I** | **Copy Editor** | Drafting, polishing, and translating research artifacts | DeepL Write, GrammarlyGO, Lex |
| **II** | **Research Assistant** | Literature review, data collection, analysis pipelines | Elicit, Research Rabbit, LangChain, LlamaIndex |
| **III** | **Super-Collaborator** | Autonomous agents that simulate users, explore theory, co-generate design artifacts | Multi-Agent Systems, Generative Agents, DSPy |

> At **Level III**, AI moves from assisting the researcher to actively participating in the invention process — simulating stakeholders, stress-testing theory, and generating novel design alternatives.

---

## Topic Index

| | | |
|---|---|---|
| [:chart_with_upwards_trend: Graph Neural Networks](Files/Graphs) | [📜 LLMs & Natural Language Processing](Files/NaturalLanguageProcessing) | [:chess_pawn: Reinforcement Learning](Files/ReinforcementLearning) |
| [💪 Generative Media & Synthetic Data](Files/DataGeneration) | [🛡️ LLM Safety & Adversarial Defense](Files/AdversarialDefense) | [:red_circle: Anomaly Detection](Files/AnomalyDetection) |
| [:snake: Python Tools & Infrastructure](Files/PythonTools) | [:bulb: AI for Research Productivity](Files/AI-for-Research-Productivity) | [:speech_balloon: Prompt Engineering](Files/Prompt-Engineering) |
| [:eyes: Multimodal Models](Files/MultimodalModels) | [:wrench: Fine-Tuning](Files/FineTuning) | [:mag: Interpretability & Explainability](Files/Interpretability) |
| [:balance_scale: Ethics & Responsible AI](Files/Ethics) | [:triangular_ruler: Evaluation & Benchmarking](Files/Evaluation) | [:seedling: Causal Inference](Files/Causal-Inference) |
| [:high_brightness: Attention Mechanisms](https://github.com/xmu-xiaoma666/External-Attention-pytorch) | [:arrow_right: Transfer Learning](https://github.com/jindongwang/transferlearning) | [:paintbrush: Vision & Image Models](https://github.com/rwightman/pytorch-image-models) |
| [:robot: AutoML](Files/AutoML) | | |

---

## Quick Start: I Want To...

| Goal | Where to Start |
|------|----------------|
| Review the literature on a research topic | [AI for Research Productivity](Files/AI-for-Research-Productivity) → Literature Review |
| Build a chatbot that answers questions over documents | [Prompt Engineering](Files/Prompt-Engineering) → Agentic Frameworks (RAG) |
| Fine-tune a model on my domain-specific data | [Fine-Tuning](Files/FineTuning) → When to Fine-Tune decision table |
| Generate synthetic survey respondents for IS experiments | [Generative Media & Synthetic Data](Files/DataGeneration) → Synthetic Users |
| Detect fraud or anomalous behavior in enterprise data | [Anomaly Detection](Files/AnomalyDetection) · [Graphs](Files/Graphs) → GraphRAG |
| Explain why my model made a specific prediction | [Interpretability](Files/Interpretability) → SHAP / Counterfactuals |
| Evaluate whether my LLM artifact actually works | [Evaluation & Benchmarking](Files/Evaluation) → RAG Evaluation / LLM-as-a-Judge |
| Understand causal effects (not just correlations) | [Causal Inference](Files/Causal-Inference) → DoWhy / EconML |
| Ensure my AI artifact is fair, ethical, and compliant | [Ethics](Files/Ethics) → Responsible AI Checklist |
| Deploy a prototype for anonymous reviewers to test | [Python Tools](Files/PythonTools) → Artifact Deployment |

---

## Learning Resources

### New to Machine Learning?

Start here before diving into the topic sections. These resources are selected for IS researchers — not computer science students.

| Resource | Why It Matters |
|----------|----------------|
| [Avoiding ML Pitfalls (2021)](https://arxiv.org/abs/2108.02497) | Essential reading before any IS+ML project. Covers evaluation errors, data leakage, and reproducibility traps. |
| [fast.ai: Practical Deep Learning](https://course.fast.ai/) | The most accessible DL course. Top-down, code-first — ideal for researchers who want results, not theory. |
| [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course/) | Free end-to-end course on transformers, fine-tuning, and deploying NLP models. |
| [Machine Learning Collection](https://github.com/aladdinpersson/Machine-Learning-Collection) | Hands-on PyTorch tutorials from basics through advanced architectures, with runnable notebooks. |
| [Python Mini Projects](https://github.com/Python-World/python-mini-projects) | Beginner Python projects to build fluency before tackling data pipelines. |
| [AutoML Tools](Files/AutoML) | Frameworks that handle model selection and hyperparameter tuning automatically — useful when ML is not your main contribution. |

### Literature Discovery

| Tool | Description |
|------|-------------|
| [Connected Papers](https://www.connectedpapers.com/) | Visual citation graph — rapidly orient to a new field from a single seed paper. |
| [Elicit](https://elicit.com/) | LLM-powered assistant that searches Semantic Scholar and builds structured comparison tables. |
| [Research Rabbit](https://www.researchrabbit.ai/) | Map forward and backward citation networks to find related work. |
| [Semantic Scholar](https://www.semanticscholar.org/) | AI-powered search with paper summarization and influence metrics. |

> For the full list of AI-powered literature tools, see [AI for Research Productivity](Files/AI-for-Research-Productivity).

---

## Cookbooks & End-to-End Tutorials

Practical, step-by-step guides bridging individual tools into complete IS research workflows.

| Tutorial | What You'll Build |
|----------|-------------------|
| [LangChain RAG Cookbook](https://python.langchain.com/docs/tutorials/rag/) | A retrieval-augmented generation pipeline over a document corpus — end-to-end from ingestion to response. |
| [OpenAI: Building Multi-Agent Systems](https://cookbook.openai.com/examples/agents_sdk/multi-agent-portfolio-collaboration) | Multi-step, tool-using agents with the OpenAI API. |
| [HuggingFace Fine-Tuning Guide](https://huggingface.co/docs/transformers/training) | Fine-tune a transformer on a custom dataset, step by step. |
| [RAGAS: Evaluating RAG Pipelines](https://docs.ragas.io/en/stable/getstarted/) | Measure faithfulness, relevance, and context precision in RAG-based IS artifacts. |
| [DoWhy Causal Inference Tutorial](https://py-why.github.io/dowhy/main/example_notebooks/dowhy_simple_example.html) | End-to-end causal effect estimation from observational IS data using the four-step DoWhy framework. |
| [Streamlit App Gallery](https://streamlit.io/gallery) | Deployed artifact examples — find patterns similar to your DSR prototype design. |

---

## Contributing

Suggestions for papers, tools, or new sections are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Especially welcome:** contributions that connect AI/ML methods to IS theory, DSR evaluation, or organizational research contexts.

---

<div align="center">

*Maintained by [Ben Ampel](https://github.com/BenAmpel)*

</div>
