---
layout: home
title: Home
nav_order: 1
permalink: /
---

{::nomarkdown}
<div class="hero-section">
  <h1>Methodological AI Tools<br>for Design Science &amp; IS Researchers</h1>
  <p class="hero-subtitle">
    A curated, living reference of AI/ML methods, tools, and tutorials —
    drawn from top IS conferences and journals — to support rigorous design science research.
  </p>
  <div class="hero-badges">
    <a href="https://awesome.re"><img src="https://awesome.re/badge.svg" alt="Awesome"></a>
    <a href="https://github.com/BenAmpel/DS-ISR-Methodological-Tools/commits/main"><img src="https://img.shields.io/github/last-commit/BenAmpel/DS-ISR-Methodological-Tools?color=blue" alt="Last Commit"></a>
    <a href="https://github.com/BenAmpel/DS-ISR-Methodological-Tools/stargazers"><img src="https://img.shields.io/github/stars/BenAmpel/DS-ISR-Methodological-Tools?style=social" alt="Stars"></a>
    <a href="{{ site.baseurl }}/contributing/"><img src="https://img.shields.io/badge/contributions-welcome-brightgreen" alt="Contributions Welcome"></a>
  </div>
  <div class="search-hint">
    🔍 Press <kbd>/</kbd> to search across all 16 topic sections
  </div>
  <div class="hero-links">
    <a href="{{ site.baseurl }}/compare/">📊 Compare Sections</a>
    <span class="hero-sep">·</span>
    <a href="{{ site.baseurl }}/changelog/">🆕 What's New</a>
    <span class="hero-sep">·</span>
    <a href="{{ site.baseurl }}/suggest/">💡 Suggest a Resource</a>
  </div>
</div>
{:/}

---

## 🔥 Featured This Month

{::nomarkdown}
<div class="featured-grid">
  <a class="featured-card" href="{{ site.baseurl }}/Files/NaturalLanguageProcessing/">
    <div class="featured-card-icon">📜</div>
    <div class="featured-card-body">
      <div class="featured-card-tag">New Content</div>
      <strong>DeepSeek-V3 & LLaMA 3 Added</strong>
      <p>The LLMs & NLP section now covers the latest frontier open-weight models, updated leaderboards, and GraphRAG.</p>
    </div>
  </a>
  <a class="featured-card" href="{{ site.baseurl }}/Files/Evaluation/">
    <div class="featured-card-icon">📐</div>
    <div class="featured-card-body">
      <div class="featured-card-tag">New Section</div>
      <strong>Evaluation & Benchmarking</strong>
      <p>Complete guide to evaluating GenAI artifacts: RAGAS, LLM-as-a-Judge, human evaluation workflows, and agent benchmarks.</p>
    </div>
  </a>
  <a class="featured-card" href="{{ site.baseurl }}/Files/Causal-Inference/">
    <div class="featured-card-icon">🌱</div>
    <div class="featured-card-body">
      <div class="featured-card-tag">Updated</div>
      <strong>Causal Inference</strong>
      <p>DoWhy, EconML, and causal graphs for IS research — when correlations aren't enough to support a design claim.</p>
    </div>
  </a>
</div>
{:/}

---

## The Automation-of-Invention Framework

*Based on the framework introduced in* Information Systems Research *(2025), AI tools are classified by their role in the research process:*

{% include automation-framework.svg %}

{::nomarkdown}
<div class="level-cards">
  <div class="level-card level-1">
    <div class="level-badge">Level I</div>
    <div class="level-role">Copy Editor</div>
    <div class="level-desc">Drafting, polishing, and translating research artifacts</div>
    <div class="level-tools">DeepL Write · GrammarlyGO · Lex · Writefull</div>
  </div>
  <div class="level-card level-2">
    <div class="level-badge">Level II</div>
    <div class="level-role">Research Assistant</div>
    <div class="level-desc">Literature review, data collection, analysis pipelines</div>
    <div class="level-tools">Elicit · Research Rabbit · LangChain · LlamaIndex</div>
  </div>
  <div class="level-card level-3">
    <div class="level-badge">Level III</div>
    <div class="level-role">Super-Collaborator</div>
    <div class="level-desc">Autonomous agents that simulate users, explore theory, co-generate artifacts</div>
    <div class="level-tools">Multi-Agent Systems · Generative Agents · DSPy</div>
  </div>
</div>
{:/}

> At **Level III**, AI moves from assisting the researcher to actively participating in the invention process — simulating stakeholders, stress-testing theory, and generating novel design alternatives.

---

## DSR Research Workflow

{% include topic-map.svg %}

---

## Browse by Topic

{::nomarkdown}
<div class="topic-cards">
  <a class="topic-card" href="{{ site.baseurl }}/Files/AI-for-Research-Productivity/">
    <span class="topic-icon">💡</span>
    <span class="topic-name">AI for Research Productivity</span>
    <span class="topic-desc">Literature discovery, writing, review automation</span>
    <span class="topic-counts">📄 12 papers · 🔧 15 tools</span>
  </a>
  <a class="topic-card" href="{{ site.baseurl }}/Files/NaturalLanguageProcessing/">
    <span class="topic-icon">📜</span>
    <span class="topic-name">LLMs & NLP</span>
    <span class="topic-desc">Large language models, RAG, transformers</span>
    <span class="topic-counts">📄 18 papers · 🔧 20 tools</span>
  </a>
  <a class="topic-card" href="{{ site.baseurl }}/Files/Prompt-Engineering/">
    <span class="topic-icon">💬</span>
    <span class="topic-name">Prompt Engineering</span>
    <span class="topic-desc">CoT, RAG, agents, structured output</span>
    <span class="topic-counts">📄 14 papers · 🔧 12 tools</span>
  </a>
  <a class="topic-card" href="{{ site.baseurl }}/Files/FineTuning/">
    <span class="topic-icon">🔧</span>
    <span class="topic-name">Fine-Tuning</span>
    <span class="topic-desc">LoRA, PEFT, domain adaptation</span>
    <span class="topic-counts">📄 16 papers · 🔧 11 tools</span>
  </a>
  <a class="topic-card" href="{{ site.baseurl }}/Files/DataGeneration/">
    <span class="topic-icon">💪</span>
    <span class="topic-name">Generative Media & Synthetic Data</span>
    <span class="topic-desc">Synthetic users, diffusion models, GANs</span>
    <span class="topic-counts">📄 20 papers · 🔧 14 tools</span>
  </a>
  <a class="topic-card" href="{{ site.baseurl }}/Files/Graphs/">
    <span class="topic-icon">📈</span>
    <span class="topic-name">Graph Neural Networks</span>
    <span class="topic-desc">GNNs, knowledge graphs, GraphRAG</span>
    <span class="topic-counts">📄 22 papers · 🔧 16 tools</span>
  </a>
  <a class="topic-card" href="{{ site.baseurl }}/Files/Interpretability/">
    <span class="topic-icon">🔍</span>
    <span class="topic-name">Interpretability</span>
    <span class="topic-desc">SHAP, LIME, counterfactuals, saliency</span>
    <span class="topic-counts">📄 15 papers · 🔧 13 tools</span>
  </a>
  <a class="topic-card" href="{{ site.baseurl }}/Files/Evaluation/">
    <span class="topic-icon">📐</span>
    <span class="topic-name">Evaluation & Benchmarking</span>
    <span class="topic-desc">RAGAS, LLM-as-a-Judge, human eval</span>
    <span class="topic-counts">📄 12 papers · 🔧 10 tools</span>
  </a>
  <a class="topic-card" href="{{ site.baseurl }}/Files/Causal-Inference/">
    <span class="topic-icon">🌱</span>
    <span class="topic-name">Causal Inference</span>
    <span class="topic-desc">DoWhy, EconML, causal graphs</span>
    <span class="topic-counts">📄 11 papers · 🔧 8 tools</span>
  </a>
  <a class="topic-card" href="{{ site.baseurl }}/Files/AnomalyDetection/">
    <span class="topic-icon">🔴</span>
    <span class="topic-name">Anomaly Detection</span>
    <span class="topic-desc">Fraud, outliers, time-series anomalies</span>
    <span class="topic-counts">📄 14 papers · 🔧 11 tools</span>
  </a>
  <a class="topic-card" href="{{ site.baseurl }}/Files/AdversarialDefense/">
    <span class="topic-icon">🛡️</span>
    <span class="topic-name">LLM Safety & Adversarial Defense</span>
    <span class="topic-desc">Jailbreaks, prompt injection, red-teaming</span>
    <span class="topic-counts">📄 16 papers · 🔧 10 tools</span>
  </a>
  <a class="topic-card" href="{{ site.baseurl }}/Files/MultimodalModels/">
    <span class="topic-icon">👁️</span>
    <span class="topic-name">Multimodal Models</span>
    <span class="topic-desc">Vision-language, audio, cross-modal</span>
    <span class="topic-counts">📄 13 papers · 🔧 12 tools</span>
  </a>
  <a class="topic-card" href="{{ site.baseurl }}/Files/ReinforcementLearning/">
    <span class="topic-icon">♟️</span>
    <span class="topic-name">Reinforcement Learning</span>
    <span class="topic-desc">RLHF, policy optimization, simulation</span>
    <span class="topic-counts">📄 14 papers · 🔧 10 tools</span>
  </a>
  <a class="topic-card" href="{{ site.baseurl }}/Files/Ethics/">
    <span class="topic-icon">⚖️</span>
    <span class="topic-name">Ethics & Responsible AI</span>
    <span class="topic-desc">Fairness, bias, compliance, checklists</span>
    <span class="topic-counts">📄 10 papers · 🔧 9 tools</span>
  </a>
  <a class="topic-card" href="{{ site.baseurl }}/Files/PythonTools/">
    <span class="topic-icon">🐍</span>
    <span class="topic-name">Python Tools & Infrastructure</span>
    <span class="topic-desc">Deployment, dashboards, pipelines</span>
    <span class="topic-counts">📄 5 papers · 🔧 22 tools</span>
  </a>
  <a class="topic-card" href="{{ site.baseurl }}/Files/AutoML/">
    <span class="topic-icon">🤖</span>
    <span class="topic-name">AutoML</span>
    <span class="topic-desc">Automated model selection, HPO</span>
    <span class="topic-counts">📄 8 papers · 🔧 12 tools</span>
  </a>
</div>
{:/}

---

## Quick Start: I Want To...

{::nomarkdown}
<div class="quickstart-grid">
  <a class="qs-card" href="{{ site.baseurl }}/Files/AI-for-Research-Productivity/">
    <span class="qs-icon">📚</span>
    <span class="qs-goal">Review the literature on a topic</span>
    <span class="qs-where">AI for Research Productivity → Literature Review</span>
    <span class="qs-badge beginner">Beginner-friendly</span>
  </a>
  <a class="qs-card" href="{{ site.baseurl }}/Files/Prompt-Engineering/">
    <span class="qs-icon">🤖</span>
    <span class="qs-goal">Build a chatbot over documents</span>
    <span class="qs-where">Prompt Engineering → RAG Frameworks</span>
    <span class="qs-badge intermediate">Intermediate</span>
  </a>
  <a class="qs-card" href="{{ site.baseurl }}/Files/FineTuning/">
    <span class="qs-icon">🎯</span>
    <span class="qs-goal">Fine-tune a model on domain data</span>
    <span class="qs-where">Fine-Tuning → When to Fine-Tune</span>
    <span class="qs-badge advanced">Advanced</span>
  </a>
  <a class="qs-card" href="{{ site.baseurl }}/Files/DataGeneration/">
    <span class="qs-icon">🧪</span>
    <span class="qs-goal">Generate synthetic survey respondents</span>
    <span class="qs-where">Generative Media → Synthetic Users</span>
    <span class="qs-badge intermediate">Intermediate</span>
  </a>
  <a class="qs-card" href="{{ site.baseurl }}/Files/AnomalyDetection/">
    <span class="qs-icon">🚨</span>
    <span class="qs-goal">Detect fraud or anomalous behavior</span>
    <span class="qs-where">Anomaly Detection or Graphs → GraphRAG</span>
    <span class="qs-badge intermediate">Intermediate</span>
  </a>
  <a class="qs-card" href="{{ site.baseurl }}/Files/Interpretability/">
    <span class="qs-icon">🔍</span>
    <span class="qs-goal">Explain a model's prediction</span>
    <span class="qs-where">Interpretability → SHAP / Counterfactuals</span>
    <span class="qs-badge beginner">Beginner-friendly</span>
  </a>
  <a class="qs-card" href="{{ site.baseurl }}/Files/Evaluation/">
    <span class="qs-icon">✅</span>
    <span class="qs-goal">Evaluate whether my LLM artifact works</span>
    <span class="qs-where">Evaluation → RAG Eval / LLM-as-a-Judge</span>
    <span class="qs-badge intermediate">Intermediate</span>
  </a>
  <a class="qs-card" href="{{ site.baseurl }}/Files/Causal-Inference/">
    <span class="qs-icon">🌿</span>
    <span class="qs-goal">Understand causal effects in IS data</span>
    <span class="qs-where">Causal Inference → DoWhy / EconML</span>
    <span class="qs-badge advanced">Advanced</span>
  </a>
  <a class="qs-card" href="{{ site.baseurl }}/Files/Ethics/">
    <span class="qs-icon">⚖️</span>
    <span class="qs-goal">Ensure my artifact is fair and compliant</span>
    <span class="qs-where">Ethics → Responsible AI Checklist</span>
    <span class="qs-badge beginner">Beginner-friendly</span>
  </a>
  <a class="qs-card" href="{{ site.baseurl }}/Files/PythonTools/">
    <span class="qs-icon">🚀</span>
    <span class="qs-goal">Deploy a prototype for reviewers</span>
    <span class="qs-where">Python Tools → Artifact Deployment</span>
    <span class="qs-badge intermediate">Intermediate</span>
  </a>
</div>
{:/}

---

## Learning Resources

### New to Machine Learning?

Start here before diving into the topic sections. Curated for IS researchers — not CS students.

{::nomarkdown}
<div class="resource-grid">
  <a class="resource-card" href="https://arxiv.org/abs/2108.02497" target="_blank">
    <div class="resource-type paper">📄 Paper</div>
    <strong>Avoiding ML Pitfalls</strong>
    <p>Essential reading before any IS+ML project. Covers evaluation errors, data leakage, and reproducibility traps.</p>
  </a>
  <a class="resource-card" href="https://course.fast.ai/" target="_blank">
    <div class="resource-type course">🎓 Course</div>
    <strong>fast.ai: Practical Deep Learning</strong>
    <p>Top-down, code-first — the most accessible DL course for researchers who want results, not theory.</p>
  </a>
  <a class="resource-card" href="https://huggingface.co/learn/nlp-course/" target="_blank">
    <div class="resource-type course">🎓 Course</div>
    <strong>HuggingFace NLP Course</strong>
    <p>Free end-to-end course on transformers, fine-tuning, and deploying NLP models.</p>
  </a>
  <a class="resource-card" href="https://github.com/aladdinpersson/Machine-Learning-Collection" target="_blank">
    <div class="resource-type notebook">📓 Notebooks</div>
    <strong>ML Collection (PyTorch)</strong>
    <p>Hands-on tutorials from basics to advanced architectures, all with runnable notebooks.</p>
  </a>
  <a class="resource-card" href="{{ site.baseurl }}/Files/AutoML/" >
    <div class="resource-type tool">🔧 Tools</div>
    <strong>AutoML Frameworks</strong>
    <p>Automated model selection and tuning — useful when ML is not your primary research contribution.</p>
  </a>
</div>
{:/}

### Literature Discovery

| Tool | What It Does |
|------|-------------|
| [Connected Papers](https://www.connectedpapers.com/) | Visual citation graph — orient to a new field from a single seed paper |
| [Elicit](https://elicit.com/) | LLM-powered research assistant with structured comparison tables |
| [Research Rabbit](https://www.researchrabbit.ai/) | Map forward/backward citation networks to discover related work |
| [Semantic Scholar](https://www.semanticscholar.org/) | AI-powered search with paper summarization and influence metrics |

> Full list of AI-powered literature tools → [AI for Research Productivity]({{ site.baseurl }}/Files/AI-for-Research-Productivity/)

---

## Cookbooks & Tutorials

{::nomarkdown}
<div class="cookbook-grid">
  <a class="cookbook-card" href="https://python.langchain.com/docs/tutorials/rag/" target="_blank">
    <div class="cookbook-step">01</div>
    <strong>LangChain RAG Pipeline</strong>
    <p>End-to-end retrieval-augmented generation from document ingestion to response.</p>
  </a>
  <a class="cookbook-card" href="https://cookbook.openai.com/examples/agents_sdk/multi-agent-portfolio-collaboration" target="_blank">
    <div class="cookbook-step">02</div>
    <strong>Multi-Agent Systems</strong>
    <p>Build multi-step, tool-using agents with the OpenAI Agents SDK.</p>
  </a>
  <a class="cookbook-card" href="https://huggingface.co/docs/transformers/training" target="_blank">
    <div class="cookbook-step">03</div>
    <strong>HuggingFace Fine-Tuning</strong>
    <p>Fine-tune any transformer on a custom dataset, step by step.</p>
  </a>
  <a class="cookbook-card" href="https://docs.ragas.io/en/stable/getstarted/" target="_blank">
    <div class="cookbook-step">04</div>
    <strong>RAGAS: Evaluate RAG Pipelines</strong>
    <p>Measure faithfulness, relevance, and context precision in LLM artifacts.</p>
  </a>
  <a class="cookbook-card" href="https://py-why.github.io/dowhy/main/example_notebooks/dowhy_simple_example.html" target="_blank">
    <div class="cookbook-step">05</div>
    <strong>DoWhy Causal Inference</strong>
    <p>End-to-end causal effect estimation from observational IS data.</p>
  </a>
  <a class="cookbook-card" href="https://streamlit.io/gallery" target="_blank">
    <div class="cookbook-step">06</div>
    <strong>Streamlit App Gallery</strong>
    <p>Deployed artifact examples to inspire DSR prototype designs.</p>
  </a>
</div>
{:/}

---

{::nomarkdown}
<div class="ethics-banner">
  ⚖️ <strong>Ethics First.</strong> Always maintain high ethical standards when building AI models. &nbsp;
  <a href="{{ site.baseurl }}/Files/Ethics/">→ Ethical Guidelines & Responsible AI Checklist</a>
</div>
{:/}
