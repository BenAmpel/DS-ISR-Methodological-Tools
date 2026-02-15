# Prompt Engineering
*Guides, Frameworks, and Libraries for Effectively Communicating with Large Language Models.*

*Last updated: February 2026*

> *Prompt engineering is the practice of crafting inputs to LLMs to elicit desired outputs reliably and consistently. As LLMs become core components of DSR artifacts, prompt engineering is an essential methodological skill for IS researchers.*

| | | |
|-|-|-|
| [Core Prompting Techniques](#Core-Prompting-Techniques) | [Advanced Reasoning Frameworks](#Advanced-Reasoning-Frameworks) | [Agentic Frameworks](#Agentic-Frameworks) |
| [Structured Output & Validation](#Structured-Output--Validation) | [Frameworks & Libraries](#Frameworks--Libraries) | [Guides & Resources](#Guides--Resources) |

---

### Core Prompting Techniques

| **Technique** | **Description** | **When to Use** |
|-|-|-|
| **Zero-Shot Prompting** | Ask the model to perform a task with no examples. | Simple, well-defined tasks the model likely encountered in training. |
| **Few-Shot Prompting** | Provide 2–8 input/output examples before the task. | Tasks with specific formatting requirements or domain-specific patterns. |
| **System Prompting** | Define the model's role, persona, and constraints in a system message. | All production deployments; establishes consistent behavior. |
| **Role Prompting** | Assign an expert persona ("You are an experienced IS researcher..."). | Eliciting domain-specific reasoning and vocabulary. |
| **Instruction Following** | Explicit, numbered instructions for multi-step tasks. | Complex outputs requiring specific structure or sequential reasoning. |

- **Key Paper:**
  - [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165), 2020 - Demonstrated that large language models can perform tasks from few-shot examples in the prompt alone, without fine-tuning. [Code](https://github.com/openai/gpt-3)

---

### Advanced Reasoning Frameworks

**Chain-of-Thought (CoT)**
Prompting the model to reason step-by-step before giving a final answer, dramatically improving performance on multi-step problems.

- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903), 2022 - Shows that adding "Let's think step by step" or few-shot reasoning examples substantially improves LLM performance on math, logic, and commonsense tasks. [Code](https://github.com/google-research/chain-of-thought-hub)
- [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171), 2022 - Generates multiple reasoning chains and takes a majority vote over answers, improving reliability. [Code](https://github.com/google-research/chain-of-thought-hub)

**Tree of Thoughts (ToT)**
Extends CoT by exploring multiple reasoning branches in parallel, allowing backtracking and lookahead — mimicking human deliberate problem solving.

- [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601), 2023 - Frames problem solving as a search over a tree of intermediate thoughts, enabling the model to evaluate and abandon unproductive paths. [Code](https://github.com/princeton-nlp/tree-of-thought-llm)

**ReAct (Reasoning + Acting)**
Interleaves reasoning traces with tool use (search, code execution, API calls), allowing LLMs to gather information and act in the world based on their reasoning.

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629), 2022 - The foundational framework for LLM agents that alternate between thinking and tool use. Basis for modern agent frameworks. [Code](https://github.com/ysymyth/ReAct)

**Retrieval-Augmented Generation (RAG)**
See the [LLMs & NLP section](../NaturalLanguageProcessing/README.md#Retrieval-Augmented-Generation-RAG) for a full treatment of RAG as both a prompting pattern and an architecture.

---

### Agentic Frameworks
Agentic frameworks enable LLMs to plan, use tools, and execute multi-step tasks autonomously. They are the foundation for Level III (Super-Collaborator) applications in the Automation-of-Invention Framework.

| **Framework** | **Description** | **Best For** |
|-|-|-|
| [LangChain](https://github.com/langchain-ai/langchain) | Dominant framework for LLM applications. Provides agents, chains, tools (web search, code execution, databases), and memory. | Rapid prototyping of LLM pipelines with retrieval and tool use |
| [LlamaIndex](https://github.com/run-llama/llama_index) | Data framework for connecting LLMs to structured and unstructured data. Better for complex indexing strategies than LangChain. | Enterprise document Q&A, knowledge base construction |
| [DSPy](https://github.com/stanfordnlp/dspy) | **Declarative Self-improving Python**: replaces hand-crafted prompts with programmatic modules that are *automatically optimized* against a metric. | Systematic prompt optimization; building robust LLM pipelines without manual prompt engineering |
| [AutoGen](https://github.com/microsoft/autogen) | Multi-agent conversation framework where specialized agents (coder, critic, planner) collaborate to complete tasks. | Complex tasks requiring multiple expert perspectives; simulating organizational processes |
| [CrewAI](https://github.com/crewAIInc/crewAI) | Role-based multi-agent framework with task delegation and collaboration. More structured than AutoGen. | Simulating research teams; pipeline orchestration with clear roles |
| [Semantic Kernel](https://github.com/microsoft/semantic-kernel) | Microsoft's SDK for integrating LLMs into applications. Strong .NET and C# support alongside Python. | Enterprise application integration; Microsoft ecosystem |

---

### Structured Output & Validation
Reliable IS research artifacts require LLM outputs in predictable, parseable formats. These tools enforce structure.

| **Tool** | **Description** |
|-|-|
| [Instructor](https://github.com/jxnl/instructor) | Uses Pydantic models to force LLM outputs into validated Python objects. Works with any OpenAI-compatible API. The simplest way to get structured JSON from an LLM. |
| [Outlines](https://github.com/outlines-dev/outlines) | Constrains LLM generation to conform to a schema at the token level (not post-hoc parsing). Guarantees valid JSON/regex output. |
| [Guidance](https://github.com/guidance-ai/guidance) | Microsoft's library for constrained generation with interleaving of generation and logic. |
| [Pydantic](https://docs.pydantic.dev/) | Python data validation library. Used by Instructor and LangChain for defining output schemas. |

---

### Frameworks & Libraries

| **Library** | **Description** |
|-|-|
| [LangChain](https://github.com/langchain-ai/langchain) | End-to-end LLM application framework (see Agentic Frameworks above). |
| [LlamaIndex](https://github.com/run-llama/llama_index) | Data-centric LLM framework (see Agentic Frameworks above). |
| [DSPy](https://github.com/stanfordnlp/dspy) | Declarative LLM programming with automatic prompt optimization. See paper: [DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines](https://arxiv.org/abs/2310.03714), 2023. |
| [PromptFlow](https://github.com/microsoft/promptflow) | Microsoft's tool for building, evaluating, and deploying LLM-based workflows with visual flow editors. |
| [Haystack](https://github.com/deepset-ai/haystack) | NLP framework for building production RAG and agent pipelines. Strong document processing capabilities. |

---

### Guides & Resources

- [Anthropic Prompt Engineering Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview) - Official best practices for prompting Claude, including system prompts, XML structuring, and chain-of-thought.
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering) - OpenAI's official guide covering strategies and tactics for GPT models.
- [Prompt Engineering Guide (DAIR.AI)](https://www.promptingguide.ai/) - Comprehensive open-source guide covering all major prompting techniques with examples. [(GitHub)](https://github.com/dair-ai/Prompt-Engineering-Guide)
- [Awesome Prompt Engineering](https://github.com/promptslab/Awesome-Prompt-Engineering) - Curated list of prompt engineering papers, tools, and resources.
- [Learn Prompting](https://learnprompting.org/) - Beginner-friendly curriculum for prompt engineering with interactive examples.

---

### Meta-Prompting & Self-Refinement
Using LLMs to generate, evaluate, and improve their own prompts — automating the prompt engineering process itself.

- **Papers with code:**
  - [Large Language Models as Optimizers (OPRO)](https://arxiv.org/abs/2309.03409), 2023 - Uses an LLM as a black-box optimizer: describe the optimization problem in natural language and have the model iteratively propose improved solutions (or prompts). [Code](https://github.com/google-deepmind/opro)
  - [Automatic Prompt Optimization with "Gradient Descent" and Beam Search (APO)](https://arxiv.org/abs/2305.03495), 2023 - Treats prompt optimization as a text-based gradient descent — the LLM generates textual gradients (critiques) and proposes improved prompts. [Code](https://github.com/microsoft/LMOps/tree/main/prompt_optimization)
  - [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651), 2023 - Single model iteratively generates output, critiques it, then refines based on its own critique. Improves generation quality without additional training. [Code](https://github.com/madaan/self-refine)

---

### Prompt Compression & Efficiency
Long prompts are expensive. Compression techniques reduce token counts while preserving semantic content — critical for cost-effective RAG and agentic IS systems.

- **Papers with code:**
  - [LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models](https://arxiv.org/abs/2310.05736), 2023 - Trains a small model to identify and remove redundant tokens from prompts. Achieves 20x compression with minimal quality degradation. [Code](https://github.com/microsoft/LLMLingua)
  - [Selective Context: Compressing Long Prompts by Selective Filtering](https://arxiv.org/abs/2304.01373), 2023 - Scores prompt segments by self-information and filters low-value content. Training-free compression. [Code](https://github.com/liyucheng09/Selective_Context)

---

### Prompt Evaluation & Red Teaming
Systematic evaluation of prompt effectiveness and robustness before deploying LLM-based IS artifacts.

- **Papers with code:**
  - [PromptBench: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts](https://arxiv.org/abs/2306.04528), 2023 - Evaluates LLM performance under adversarial prompt perturbations (character, word, sentence, semantic attacks). [Code](https://github.com/microsoft/promptbench)
  - [HELM: Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110), 2022 - Comprehensive benchmark framework. Standardized evaluation methodology for comparing prompting strategies across multiple metrics. [Code](https://github.com/stanford-crfm/helm)

- **Tools:**
  - [PromptFoo](https://github.com/promptfoo/promptfoo) - Open-source prompt testing framework. Define test cases, run evals, and compare prompt variants systematically. Essential for IS researchers validating LLM-based artifact behavior.

---

**Related Sections:** [LLMs & NLP](../NaturalLanguageProcessing/README.md) | [Fine-Tuning](../FineTuning/README.md) | [AI for Research Productivity](../AI-for-Research-Productivity/README.md) | [Reinforcement Learning](../ReinforcementLearning/README.md) [LLMs & NLP](../NaturalLanguageProcessing/README.md) | [Fine-Tuning](../FineTuning/README.md) | [AI for Research Productivity](../AI-for-Research-Productivity/README.md) | [Reinforcement Learning](../ReinforcementLearning/README.md)
