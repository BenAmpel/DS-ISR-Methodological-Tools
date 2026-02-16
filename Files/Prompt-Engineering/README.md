---
layout: default
title: "Prompt Engineering"
parent: Topics
nav_order: 3
permalink: /Files/Prompt-Engineering/
---


{::nomarkdown}
<div class="section-meta">
  <span class="meta-pill meta-reading">⏱ ~9 min read</span>
  <span class="meta-pill meta-updated">📅 Updated Feb 2026</span>
  <span class="meta-pill meta-beginner">📊 Beginner</span>
  <span class="meta-pill meta-prereq">🔑 LLM basics</span>
</div>
{:/}

# Prompt Engineering
*Guides, Frameworks, and Libraries for Effectively Communicating with Large Language Models.*

*Last updated: February 2026*

> *Prompt engineering is the practice of crafting inputs to LLMs to elicit desired outputs reliably and consistently. As LLMs become core components of DSR artifacts, prompt engineering is an essential methodological skill for IS researchers.*

> ⭐ = open-source / free to use

| [Core Prompting Techniques](#core-prompting-techniques) | [Advanced Reasoning Frameworks](#advanced-reasoning-frameworks) | [Reasoning Models](#reasoning-models) |
| [Agentic Frameworks](#agentic-frameworks) | [Structured Output & Validation](#structured-output-validation) | [Prompt Optimization](#prompt-optimization) |
| [Prompt Security & Red Teaming](#prompt-security-red-teaming) | [Multimodal Prompting](#multimodal-prompting) | [Frameworks & Libraries](#frameworks-libraries) |
| [Meta-Prompting & Self-Refinement](#meta-prompting--self-refinement) | [Prompt Compression & Efficiency](#prompt-compression--efficiency) | [Guides & Resources](#guides-resources) |

---

### Core Prompting Techniques

| **Technique** | **Description** | **When to Use** |
|-|-|-|
| **Zero-Shot Prompting** | Ask the model to perform a task with no examples. | Simple tasks; frontier models (GPT-4o, Claude 3.5+) often match few-shot quality zero-shot. |
| **Few-Shot Prompting** | Provide 2–8 input/output examples before the task. | Format alignment, domain-specific patterns, or smaller/open-weight models. Note: excess examples can *degrade* reasoning on frontier models. |
| **System Prompting** | Define the model's role, persona, and constraints in a system message. | All production deployments; establishes consistent behavior across sessions. |
| **Role Prompting** | Assign an expert persona ("You are an experienced IS researcher..."). | Eliciting domain-specific reasoning and vocabulary. |
| **Instruction Following** | Explicit, numbered instructions for multi-step tasks. | Complex outputs requiring specific structure or sequential reasoning. |
| **XML Tagging (Claude)** | Structure prompts using semantic XML tags (`<instructions>`, `<context>`, `<example>`). | All Claude deployments; tags serve as semantic anchors, measurably improving output quality. |

> **2024 Update:** On frontier models (GPT-4o-class, Claude 3.5+, Llama 3.3+), zero-shot Chain-of-Thought often matches or exceeds few-shot CoT for reasoning tasks. The primary role of few-shot examples has shifted toward **output format alignment** rather than teaching reasoning strategy.

- **Key Papers:**
  - [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165), 2020 - Demonstrated that large language models can perform tasks from few-shot examples in the prompt alone, without fine-tuning. [Code](https://github.com/openai/gpt-3)
  - [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916), 2022 - Showed that appending "Let's think step by step" dramatically improves zero-shot reasoning — the canonical zero-shot CoT trigger. [Code](https://github.com/kojima-takeshi188/zero_shot_cot)

---

### Advanced Reasoning Frameworks

**Chain-of-Thought (CoT)**
Prompting the model to reason step-by-step before giving a final answer, dramatically improving performance on multi-step problems.

- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903), 2022 - Shows that adding "Let's think step by step" or few-shot reasoning examples substantially improves LLM performance on math, logic, and commonsense tasks. [Code](https://github.com/google-research/chain-of-thought-hub)
- [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171), 2022 - Generates multiple reasoning chains and takes a majority vote over answers, improving reliability. [Code](https://github.com/google-research/chain-of-thought-hub)
- [Chain of Draft: Thinking Faster by Writing Less](https://arxiv.org/abs/2502.18600), 2025 ⭐ open-source - Prompts the model to produce reasoning steps using *at most five words per step*. Preserves reasoning accuracy while reducing token usage by 7–10x — critical for cost-effective API deployments. No library required.
- [A Survey of Chain-of-X Paradigms for LLMs](https://aclanthology.org/2025.coling-main.719.pdf), ACL 2025 - Comprehensive taxonomy of 15+ CoT variants (CoD, Multimodal-CoT, Chain-of-Verification, Chain-of-Symbol, Chain-of-Feedback) in a unified framework.

**Tree of Thoughts (ToT)** ⭐ open-source
Extends CoT by exploring multiple reasoning branches in parallel, allowing backtracking and lookahead — mimicking human deliberate problem solving.

- [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601), 2023 - Frames problem solving as a search over a tree of intermediate thoughts, enabling the model to evaluate and abandon unproductive paths. [Code](https://github.com/princeton-nlp/tree-of-thought-llm)

**Graph of Thoughts (GoT)** ⭐ open-source
Extends ToT from tree-shaped to arbitrary graph-shaped reasoning, enabling merging of thought branches and feedback loops.

- [Graph of Thoughts: Solving Elaborate Problems with Large Language Models](https://arxiv.org/abs/2308.09687), revised Feb 2024 - Outperforms ToT by 62% on sorting tasks while reducing inference cost by over 31%. Allows intermediate results to be merged across branches — highly relevant for multi-stage IS artifact design. [Code](https://github.com/spcl/graph-of-thoughts)
- [Demystifying Chains, Trees, and Graphs of Thoughts](https://arxiv.org/abs/2401.14295), Jan 2024 - Provides a unified formal taxonomy covering CoT, ToT, GoT, and ReAct — recommended for academic framing.

**ReAct (Reasoning + Acting)** ⭐ open-source
Interleaves reasoning traces with tool use (search, code execution, API calls), allowing LLMs to gather information and act in the world based on their reasoning.

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629), 2022 - The foundational framework for LLM agents that alternate between thinking and tool use. Now the de facto backbone of virtually all LLM agent frameworks (LangChain, AutoGen, CrewAI). [Code](https://github.com/ysymyth/react)

**Step-Back Prompting** ⭐ open-source (prompt pattern)
Before answering a specific question, prompts the model to first ask and answer a more general/abstract version — "stepping back" to first principles — then uses that abstracted answer as context.

- [Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models](https://arxiv.org/abs/2310.06117), ICLR 2024 - +7% on MMLU Physics, +11% on Chemistry, +27% on TimeQA. Highly applicable to IS research artifact design: forces articulation of design principles (kernel theory) before generating artifact-level specifications.

**Skeleton-of-Thought (SoT)** ⭐ open-source
Generates a structural skeleton first, then fills in parallel content for each skeleton point via simultaneous API calls — achieving 2–5x speedup.

- [Skeleton-of-Thought: Prompting LLMs for Efficient Parallel Generation](https://arxiv.org/abs/2307.15337), ICLR 2024 - Fastest high-quality generation for structured documents (literature reviews, design specifications, reports). Requires parallel API calls or a batching framework. [Code](https://github.com/imagination-research/sot)

**Chain-of-Verification (CoVe)** ⭐ open-source (prompt pattern)
Four-step hallucination reduction pipeline: generate initial response → plan verification questions → answer those questions independently → refine original response.

- [Chain-of-Verification Reduces Hallucination in Large Language Models](https://arxiv.org/abs/2309.11495), ACL 2024 Findings - Specifically targets factual hallucination. Critical for IS artifacts generating claims about existing literature or empirical findings. Implementable as a multi-turn prompt sequence without any library.

**Reflexion** ⭐ open-source
LLM agents that verbally reflect on task feedback, storing self-reflections in an episodic memory buffer for use in subsequent trials.

- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366), 2023 - Foundational paper for agent self-improvement. Extended by the 2024 "Distilling System 2 into System 1" insight (arXiv:[2407.06023](https://arxiv.org/abs/2407.06023)): slow reflective reasoning can be distilled into fast single-pass responses. [Code](https://github.com/noahshinn/reflexion)

**Retrieval-Augmented Generation (RAG)**
See the [LLMs & NLP section](../NaturalLanguageProcessing/README.md#Retrieval-Augmented-Generation-RAG) for a full treatment of RAG as both a prompting pattern and an architecture.

---

### Agentic Frameworks
Agentic frameworks enable LLMs to plan, use tools, and execute multi-step tasks autonomously. They are the foundation for Level III (Super-Collaborator) applications in the Automation-of-Invention Framework.

| **Framework** | **Description** | **Best For** |
|-|-|-|
| [LangChain](https://github.com/langchain-ai/langchain) ⭐ | Dominant framework for LLM applications. Provides agents, chains, tools (web search, code execution, databases), and memory. 2024: LangChain Expression Language (LCEL) introduced a `\|` pipe syntax for composing chains. | Rapid prototyping of LLM pipelines with retrieval and tool use |
| [LlamaIndex](https://github.com/run-llama/llama_index) ⭐ | Data framework for connecting LLMs to structured and unstructured data. Better for complex indexing strategies than LangChain. | Enterprise document Q&A, knowledge base construction |
| [DSPy](https://github.com/stanfordnlp/dspy) ⭐ | **Declarative Self-improving Python** (v2.5+, 250+ contributors): replaces hand-crafted prompts with programmatic modules automatically optimized against a metric via **MIPROv2** or **GEPA** (RL-based). | Systematic prompt optimization; reproducible LLM artifact development |
| [AutoGen](https://github.com/microsoft/autogen) ⭐ | Multi-agent conversation framework (v0.4 rewrite in 2024) where specialized agents (coder, critic, planner) collaborate via asynchronous message passing. | Complex tasks requiring multiple expert perspectives; simulating organizational processes |
| [CrewAI](https://github.com/crewaiinc/crewai) ⭐ | Role-based multi-agent framework with task delegation and collaboration. More structured than AutoGen. | Simulating research teams; pipeline orchestration with clear roles |
| [LangGraph](https://github.com/langchain-ai/langgraph) ⭐ | Graph-based agent orchestration from LangChain (2024). Defines agent workflows as directed graphs with nodes (LLM calls) and edges (routing logic) — enables complex branching and cycles. | Stateful multi-agent workflows; production agent systems requiring fine-grained control |
| [Semantic Kernel](https://github.com/microsoft/semantic-kernel) ⭐ | Microsoft's SDK (MIT) for integrating LLMs into applications via "plugins." Strong .NET/C# support. 2024: Prompty integration for standardized `.prompty` prompt templates. | Enterprise application integration; Microsoft ecosystem |

---

### Structured Output & Validation
Reliable IS research artifacts require LLM outputs in predictable, parseable formats. These tools enforce structure either at the API level or at the token-generation level.

| **Tool** | **Description** |
|-|-|
| [Instructor](https://github.com/jxnl/instructor) ⭐ | Uses Pydantic models to force LLM outputs into validated Python objects. Works with any OpenAI-compatible API including Ollama, Groq, and Mistral. The simplest way to get structured JSON from any LLM. |
| [Outlines](https://github.com/outlines-dev/outlines) ⭐ (Apache 2.0) | Constrains LLM generation to conform to a schema **at the token level** (not post-hoc parsing) — guarantees valid JSON/regex output even when schema info is absent from the prompt. Integrated as a backend in vLLM. |
| [LM-Format-Enforcer](https://github.com/noamgat/lm-format-enforcer) ⭐ (MIT) | Token-level JSON schema and regex enforcement via logit filtering. Compatible with transformers, llama.cpp, and vLLM. |
| [Guidance](https://github.com/guidance-ai/guidance) ⭐ (MIT) | Microsoft's library for interleaving constrained generation with arbitrary Python logic. Fastest in 2024 speed benchmarks; best for complex conditional generation programs. |
| [Pydantic](https://docs.pydantic.dev/) ⭐ | Python data validation library. Used by Instructor, LangChain, and DSPy for defining output schemas. |
| **OpenAI Structured Outputs** | `response_format: {type: "json_schema"}` with `gpt-4o-2024-08-06+` guarantees schema compliance — stronger than JSON mode (valid JSON only). The API-native option for OpenAI users; no library needed. |

---

### Frameworks & Libraries

| **Library** | **Description** |
|-|-|
| [LangChain](https://github.com/langchain-ai/langchain) ⭐ | End-to-end LLM application framework (see Agentic Frameworks above). |
| [LlamaIndex](https://github.com/run-llama/llama_index) ⭐ | Data-centric LLM framework (see Agentic Frameworks above). |
| [DSPy](https://github.com/stanfordnlp/dspy) ⭐ (MIT) | Declarative LLM programming with automatic prompt optimization via MIPROv2, GEPA, and BetterTogether. See paper: [DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines](https://arxiv.org/abs/2310.03714), 2023. v2.5+ released 2024; 250+ contributors. |
| [PromptFlow](https://github.com/microsoft/promptflow) ⭐ (MIT) | Microsoft's tool for building, evaluating, and deploying LLM workflows. 2024: added **Prompty** (`.prompty` file format standardizing prompts + model metadata into one portable asset; integrates with VS Code and GitHub Actions). |
| [Haystack](https://github.com/deepset-ai/haystack) ⭐ | NLP framework for building production RAG and agent pipelines. Strong document processing capabilities. |
| [Promptfoo](https://github.com/promptfoo/promptfoo) ⭐ (MIT) | CLI + library for systematic prompt testing, A/B evaluation, and automated red teaming. Declarative YAML test configs; tests prompts against GPT, Claude, Gemini, Llama simultaneously. 200,000+ developers; 80+ Fortune 500 companies. *The* standard tool for prompt regression testing in IS artifact development. |
| [TextGrad](https://github.com/zou-group/textgrad) ⭐ (MIT) | "Automatic differentiation via text." LLMs provide natural-language feedback (analogous to gradients) to optimize components in a computation graph — prompts, code, or entire pipelines. See paper: [TextGrad: Automatic Differentiation via Text](https://arxiv.org/abs/2406.07496), *Nature*, June 2024. |

---

### Guides & Resources

- [Anthropic Prompt Engineering Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview) - Official best practices for prompting Claude, including system prompts, XML structuring, chain-of-thought, and the new extended thinking guide for Claude 3.7+.
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering) - OpenAI's official guide. Updated 2024 to include guidance for o1/o3 reasoning models.
- [Prompt Engineering Guide (DAIR.AI)](https://www.promptingguide.ai/) ⭐ open-source - Comprehensive community guide covering all major prompting techniques with examples. Includes 2024 additions: LLM reasoning models, multimodal prompting, and automated prompt optimization. [(GitHub)](https://github.com/dair-ai/prompt-engineering-guide)
- [Awesome Prompt Engineering](https://github.com/promptslab/awesome-prompt-engineering) ⭐ open-source - Curated list of prompt engineering papers, tools, and resources.
- [Learn Prompting](https://learnprompting.org/) ⭐ open-source - Beginner-friendly curriculum for prompt engineering with interactive examples. Updated 2024.
- [A Survey of Automatic Prompt Engineering: An Optimization Perspective](https://arxiv.org/abs/2502.11560), 2025 - Unified optimization-theoretic framework covering APE, OPRO, DSPy, EvoPrompt, and TextGrad. Best single reference for the automated prompt optimization landscape.

---

### Meta-Prompting & Self-Refinement
Using LLMs to generate, evaluate, and improve their own prompts — automating the prompt engineering process itself. See also the dedicated [Prompt Optimization](#prompt-optimization) section for systematic optimization frameworks (DSPy, OPRO, EvoPrompt).

- **Papers with code:**
  - [Large Language Models as Optimizers (OPRO)](https://arxiv.org/abs/2309.03409), 2023 ⭐ open-source - Uses an LLM as a black-box optimizer: describe the optimization problem in natural language and have the model iteratively propose improved solutions (or prompts). Best prompts outperform human-designed prompts by up to 8% on GSM8K. [Code](https://github.com/google-deepmind/opro)
  - [Automatic Prompt Optimization with "Gradient Descent" and Beam Search (APO)](https://arxiv.org/abs/2305.03495), 2023 ⭐ open-source - Treats prompt optimization as text-based gradient descent — LLM generates textual critiques and proposes improved prompts iteratively. [Code](https://github.com/microsoft/lmops/tree/main/prompt_optimization)
  - [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651), 2023 ⭐ open-source - Single model iteratively generates output, critiques it, then refines based on its own critique. Improves generation quality without additional training. [Code](https://github.com/madaan/self-refine)
  - [Socratic Self-Refine (SSR)](https://arxiv.org/abs/2511.10621), Nov 2024 ⭐ open-source - Decomposes refinement into Socratic questioning steps; re-evaluates sub-answers via self-consistency; refines only step-level errors. Substantially outperforms naive self-refine on reasoning tasks.
  - [System 2 Attention (S2A)](https://arxiv.org/abs/2311.11829), 2024 - Mitigates sycophancy by having the model regenerate/clean the input context before generating a response. Improves factual accuracy and reduces opinion-anchoring bias — critical when using LLMs as IS peer reviewers or evaluation judges.

---

### Prompt Compression & Efficiency
Long prompts are expensive. Compression techniques reduce token counts while preserving semantic content — critical for cost-effective RAG and agentic IS systems. See also **Chain of Draft** in [Advanced Reasoning Frameworks](#advanced-reasoning-frameworks) for reasoning-step compression.

- **Papers with code:**
  - [LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models](https://arxiv.org/abs/2310.05736), 2023 ⭐ open-source - Trains a small model (Llama-2 7B) to identify and remove redundant tokens from prompts. Achieves 20x compression with minimal quality degradation. [Code](https://github.com/microsoft/llmlingua)
  - [LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression](https://arxiv.org/abs/2403.12968), 2024 ⭐ open-source - Bidirectional token classification model (vs. LLMLingua's causal approach). More faithful to original meaning; better for task-agnostic compression of retrieved documents in RAG pipelines. [Code](https://github.com/microsoft/LLMLingua)
  - [Selective Context: Compressing Long Prompts by Selective Filtering](https://arxiv.org/abs/2304.01373), 2023 ⭐ open-source - Scores prompt segments by self-information and filters low-value content. Training-free compression — the simplest approach to implement for IS researchers. [Code](https://github.com/liyucheng09/selective_context)

---

### Prompt Evaluation & Red Teaming
Systematic evaluation of prompt effectiveness and robustness before deploying LLM-based IS artifacts.

- **Papers with code:**
  - [PromptBench: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts](https://arxiv.org/abs/2306.04528), 2023 ⭐ open-source - Evaluates LLM performance under adversarial prompt perturbations (character, word, sentence, semantic attacks). [Code](https://github.com/microsoft/promptbench)
  - [HELM: Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110), 2022 ⭐ open-source - Comprehensive benchmark framework. Standardized evaluation methodology for comparing prompting strategies across multiple metrics. [Code](https://github.com/stanford-crfm/helm)

- **Tools:**
  - [PromptFoo](https://github.com/promptfoo/promptfoo) ⭐ open-source (MIT) - CLI and library for systematic prompt testing, A/B evaluation, and automated red teaming. Declarative YAML test configs; tests prompts against GPT, Claude, Gemini, Llama simultaneously. $18.4M Series A (2025); 200,000+ developers; essential for IS researchers validating LLM-based artifact behavior.

---

### Reasoning Models
**(New in 2024–2025)** Models like OpenAI o1/o3, DeepSeek-R1, and Qwen3 perform *internal* chain-of-thought via reinforcement learning — fundamentally changing prompt engineering strategy for complex reasoning tasks.

| Strategy | Standard LLMs | Reasoning Models (o1, o3, DeepSeek-R1, Qwen3) |
|-|-|-|
| Chain-of-Thought ("think step by step") | Essential | **Counterproductive** — these models reason internally; explicit CoT adds verbosity |
| Few-shot examples | Improves reasoning | Use sparingly; only for output **format** alignment |
| Detailed step-by-step instructions | Helpful | Avoid — trust the model's internal process |
| Encouraging depth | "Think step by step" | "Take your time and think carefully" |
| Problem specification | Can be loose | **Invest heavily here** — precise specification matters more than prompting technique |
| `<think>` tags | Not applicable | Prepend `<think>\n` to enforce explicit scratchpad in DeepSeek-R1 |

> **IS/DSR Implication:** For complex DSR problem formulation, hypothesis generation, or theoretical synthesis tasks, reasoning models under minimal prompting outperform elaborate prompt engineering. Redirect effort from prompt scaffolding to *precise problem specification*.

- **Key Papers:**
  - [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948), Jan 2025 ⭐ open-source (Apache 2.0) - Full technical report for DeepSeek-R1. Demonstrates emergent self-reflection and verification via pure RL, without human-labeled reasoning trajectories. Available at [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-R1). Key distilled variants: R1-Distill-Llama-8B/70B, R1-Distill-Qwen-32B.
  - [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388), May 2025 ⭐ open-source - Introduces unified "thinking mode" (extended internal reasoning) and "non-thinking mode" in a single model. Adds a **thinking budget** parameter for allocating compute per query. [Code](https://github.com/QwenLM/QwQ)

---

### Prompt Optimization
Automated frameworks for optimizing prompts against a measurable objective — replacing manual trial-and-error with systematic search.

- **Papers with code:**
  - [DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines](https://arxiv.org/abs/2310.03714), 2023 ⭐ open-source (MIT) - Stanford's declarative LLM programming framework. Define program logic as typed Python signatures; MIPROv2 or GEPA optimizers automatically find the best prompts and few-shot examples against your metric. v2.5+ (2024), 250+ contributors. [Code](https://github.com/stanfordnlp/dspy). **Highest IS/DSR relevance**: makes prompt optimization as rigorous and reproducible as algorithmic optimization.
  - [Large Language Models are Optimizers (OPRO)](https://arxiv.org/abs/2309.03409), 2023 ⭐ open-source - LLM as black-box optimizer: describe the task in natural language; the model iteratively proposes improved solutions. Best discovered prompts outperform human-designed prompts by up to 8% on GSM8K, 50% on Big-Bench Hard tasks. Note: sensitive to optimizer model quality — fails with small models. [Code](https://github.com/google-deepmind/opro)
  - [EvoPrompt: Connecting LLMs with Evolutionary Algorithms for Prompt Optimization](https://arxiv.org/abs/2309.08532), ICLR 2024 ⭐ open-source - Applies genetic algorithms and differential evolution to prompt optimization; LLMs act as mutation/crossover operators. Outperforms human-engineered prompts by up to 25% on Big-Bench Hard; works with open-source models. [Code](https://github.com/beeevita/EvoPrompt)
  - [Large Language Models Are Human-Level Prompt Engineers (APE)](https://arxiv.org/abs/2211.01910), ICLR 2023 ⭐ open-source - The foundational automated prompt generation paper. LLMs propose candidate instructions from few input-output examples; candidates are scored and iterated. Discovered a better zero-shot CoT trigger than the human-engineered "Let's think step by step." [Code](https://github.com/keirp/automatic_prompt_engineer)
  - [TextGrad: Automatic "Differentiation" via Text](https://arxiv.org/abs/2406.07496), *Nature*, June 2024 ⭐ open-source (MIT) - Extends PyTorch autograd to text: LLMs provide natural-language gradients to co-optimize all components of a pipeline (prompts, code, agent instructions). Enables end-to-end optimization of multi-component IS artifacts. [Code](https://github.com/zou-group/textgrad)

---

### Prompt Security & Red Teaming
Security considerations for LLM-based IS artifacts that use system prompts or handle sensitive organizational data.

- **Prompt Injection** — Attacks that embed adversarial instructions in user input or retrieved documents to override system instructions.
  - [Prompt Injection attack against LLM-integrated Applications (HouYi)](https://arxiv.org/abs/2306.05499), 2023 - Tested on 36 real LLM apps; 31 were vulnerable. The most studied prompt injection framework; vendors including Notion confirmed findings.
  - **OWASP LLM Top 10 (2025):** Prompt injection is ranked **LLM01:2025** — the top vulnerability for enterprise LLM systems.
  - **IS/DSR implication:** Any IS artifact that processes external documents (papers, forms, emails) in its context is potentially vulnerable to indirect prompt injection from adversarial document content.

- **Prompt Leakage** — Extracting proprietary system prompts that encode business rules, research protocols, or organizational knowledge.
  - [PLeak: Prompt Leaking Attacks against Large Language Model Applications](https://arxiv.org/abs/2405.06823), ACM CCS 2024 - First closed-box optimization attack for system prompt extraction; automates adversarial query generation.
  - [Prompt Leakage Effect and Defense Strategies for Multi-turn LLM Interactions](https://arxiv.org/abs/2404.16251), EMNLP 2024 Industry Track - Shows leakage risk compounds over multi-turn conversations.
  - **Defense:** ProxyPrompt (arXiv:[2505.11459](https://arxiv.org/abs/2505.11459), 2025) — protects 94.70% of prompts from extraction.
  - **IS/DSR implication:** Treat system prompt confidentiality as an architectural requirement; consider prompt abstraction patterns for production artifact deployment.

- **Tools:**
  - [Promptfoo](https://github.com/promptfoo/promptfoo) ⭐ open-source - Automated red teaming for prompt injection, PII leakage, jailbreaks, and guardrail bypasses; generates adversarial test cases programmatically. See also [Prompt Evaluation & Red Teaming](#prompt-evaluation--red-teaming).
  - [Giskard](https://github.com/Giskard-AI/giskard) ⭐ open-source - LLM vulnerability scanner with automatic red-teaming; generates shareable safety reports for paper appendices. See also [Evaluation section](../Evaluation/README.md).

---

### Multimodal Prompting
Prompting techniques for models that process both images and text — increasingly relevant as IS artifacts incorporate visual data (dashboards, diagrams, documents).

- **Papers with code:**
  - [Multimodal Chain-of-Thought Reasoning in Language Models](https://arxiv.org/abs/2302.00923), updated May 2024 ⭐ open-source - Two-stage framework: generate rationale using both image and text, then infer answer from the rationale. Substantially improves visual QA accuracy. [Code](https://github.com/amazon-science/mm-cot)
  - [Compositional Chain-of-Thought Prompting for Large Multimodal Models (CVPR 2024)](https://arxiv.org/abs/2311.17076), CVPR 2024 ⭐ open-source - Generates scene graphs as an intermediate reasoning step (vs. captions) before answering — improving compositional visual understanding.
  - [Visual Sketchpad: Sketching as a Visual Chain of Thought for Multimodal Language Models](https://arxiv.org/abs/2406.09403), Jun 2024 - Prompts the model to produce visual artifacts (sketches, diagrams, annotations) as intermediate reasoning steps. Average improvement: +11.2% for GPT-4o, +23.4% for GPT-4 Turbo. Directly applicable to IS artifacts that need to reason over process diagrams or architectural schematics.
  - [Image-of-Thought Prompting for Visual Question Answering](https://arxiv.org/abs/2405.13872), May 2024 - Automatically designs visual information extraction operations based on image + question, generating visual rationales step-by-step.
  - [Visual Prompting in Multimodal Large Language Models: A Survey](https://arxiv.org/abs/2409.15310), Sep 2024 - Comprehensive survey of visual prompting techniques: annotations, arrows, region highlighting, textual descriptions. Useful reference for multimodal IS artifact design.

---

**Related Sections:** [LLMs & NLP](../NaturalLanguageProcessing/README.md) | [Fine-Tuning](../FineTuning/README.md) | [AI for Research Productivity](../AI-for-Research-Productivity/README.md) | [Evaluation](../Evaluation/README.md) | [Reinforcement Learning](../ReinforcementLearning/README.md)
