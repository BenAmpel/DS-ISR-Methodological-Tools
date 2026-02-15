---
layout: home
title: Home
nav_order: 1
permalink: /
---

<div style="text-align:center; padding: 1.5rem 0 1rem;">
  <h1 style="font-size: 1.9rem; margin-bottom: 0.5rem; border-bottom: none;">
    Methodological AI Tools<br>for Design Science &amp; IS Researchers
  </h1>
  <p style="font-size: 1.05rem; max-width: 660px; margin: 0 auto 1.2rem; color: #444;">
    A curated, living reference of AI/ML methods, tools, and tutorials —
    drawn from top IS conferences and journals — to support rigorous design science research.
  </p>
  <a href="https://awesome.re"><img src="https://awesome.re/badge.svg" alt="Awesome" style="margin: 0 3px;"></a>
  <a href="https://github.com/BenAmpel/DS-ISR-Methodological-Tools/commits/main"><img src="https://img.shields.io/github/last-commit/BenAmpel/DS-ISR-Methodological-Tools?color=blue" alt="Last Commit" style="margin: 0 3px;"></a>
  <a href="https://github.com/BenAmpel/DS-ISR-Methodological-Tools/stargazers"><img src="https://img.shields.io/github/stars/BenAmpel/DS-ISR-Methodological-Tools?style=social" alt="Stars" style="margin: 0 3px;"></a>
  <a href="{{ site.baseurl }}/contributing"><img src="https://img.shields.io/badge/contributions-welcome-brightgreen" alt="Contributions Welcome" style="margin: 0 3px;"></a>
</div>

---

## The Automation-of-Invention Framework

*Based on the framework introduced in* Information Systems Research *(2025), AI tools are classified by their role in the research process:*

{% include automation-framework.svg %}

| Level | Role | Description | Representative Tools |
|:-----:|------|-------------|----------------------|
| **I** | **Copy Editor** | Drafting, polishing, and translating research artifacts | DeepL Write, GrammarlyGO, Lex |
| **II** | **Research Assistant** | Literature review, data collection, analysis pipelines | Elicit, Research Rabbit, LangChain, LlamaIndex |
| **III** | **Super-Collaborator** | Autonomous agents that simulate users, explore theory, co-generate design artifacts | Multi-Agent Systems, Generative Agents, DSPy |

> At **Level III**, AI moves from assisting the researcher to actively participating in the invention process — simulating stakeholders, stress-testing theory, and generating novel design alternatives.

---

## DSR Research Workflow

{% include topic-map.svg %}

Browse all tool categories in the [Topic Index]({{ site.baseurl }}/topics).

---

## Quick Start: I Want To...

| Goal | Where to Start |
|------|----------------|
| Review the literature on a research topic | [AI for Research Productivity]({{ site.baseurl }}/Files/AI-for-Research-Productivity/) → Literature Review |
| Build a chatbot that answers questions over documents | [Prompt Engineering]({{ site.baseurl }}/Files/Prompt-Engineering/) → Agentic Frameworks (RAG) |
| Fine-tune a model on my domain-specific data | [Fine-Tuning]({{ site.baseurl }}/Files/FineTuning/) → When to Fine-Tune decision table |
| Generate synthetic survey respondents for IS experiments | [Generative Media & Synthetic Data]({{ site.baseurl }}/Files/DataGeneration/) → Synthetic Users |
| Detect fraud or anomalous behavior in enterprise data | [Anomaly Detection]({{ site.baseurl }}/Files/AnomalyDetection/) · [Graphs]({{ site.baseurl }}/Files/Graphs/) → GraphRAG |
| Explain why my model made a specific prediction | [Interpretability]({{ site.baseurl }}/Files/Interpretability/) → SHAP / Counterfactuals |
| Evaluate whether my LLM artifact actually works | [Evaluation & Benchmarking]({{ site.baseurl }}/Files/Evaluation/) → RAG Evaluation / LLM-as-a-Judge |
| Understand causal effects (not just correlations) | [Causal Inference]({{ site.baseurl }}/Files/Causal-Inference/) → DoWhy / EconML |
| Ensure my AI artifact is fair, ethical, and compliant | [Ethics]({{ site.baseurl }}/Files/Ethics/) → Responsible AI Checklist |
| Deploy a prototype for anonymous reviewers to test | [Python Tools]({{ site.baseurl }}/Files/PythonTools/) → Artifact Deployment |

---

## Learning Resources

### New to Machine Learning?

Start here before diving into the topic sections. These resources are selected for IS researchers — not computer science students.

| Resource | Why It Matters |
|----------|----------------|
| [Avoiding ML Pitfalls (2021)](https://arxiv.org/abs/2108.02497) | Essential reading before any IS+ML project. Covers evaluation errors, data leakage, and reproducibility traps. |
| [fast.ai: Practical Deep Learning](https://course.fast.ai/) | Top-down, code-first — the most accessible DL course for researchers who want results, not theory. |
| [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course/) | Free end-to-end course on transformers, fine-tuning, and deploying NLP models. |
| [Machine Learning Collection](https://github.com/aladdinpersson/Machine-Learning-Collection) | Hands-on PyTorch tutorials from basics through advanced architectures, with runnable notebooks. |
| [Python Mini Projects](https://github.com/Python-World/python-mini-projects) | Beginner Python projects to build fluency before tackling data pipelines. |
| [AutoML Tools]({{ site.baseurl }}/Files/AutoML/) | Frameworks that handle model selection and tuning automatically — useful when ML is not your main contribution. |

### Literature Discovery

| Tool | Description |
|------|-------------|
| [Connected Papers](https://www.connectedpapers.com/) | Visual citation graph — rapidly orient to a new field from a single seed paper. |
| [Elicit](https://elicit.com/) | LLM-powered assistant that searches Semantic Scholar and builds structured comparison tables. |
| [Research Rabbit](https://www.researchrabbit.ai/) | Map forward and backward citation networks to find related work. |
| [Semantic Scholar](https://www.semanticscholar.org/) | AI-powered search with paper summarization and influence metrics. |

> For the full list of AI-powered literature tools, see [AI for Research Productivity]({{ site.baseurl }}/Files/AI-for-Research-Productivity/).

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

> **Ethics first.** Always maintain high ethical standards when building AI models. → [Ethical Guidelines for AI]({{ site.baseurl }}/Files/Ethics/)
