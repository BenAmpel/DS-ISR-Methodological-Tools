---
layout: default
title: "Fine-Tuning"
parent: Topics
nav_order: 4
---

# Fine-Tuning Foundation Models
*Techniques, Libraries, and Best Practices for Adapting Pre-Trained LLMs and Vision Models to Domain-Specific Tasks.*

*Last updated: February 2026*

> Fine-tuning adapts a pre-trained foundation model to a specific task or domain by continuing training on task-specific data. It bridges the gap between general-purpose LLMs (listed in [LLMs & NLP](../NaturalLanguageProcessing/README.md)) and production-ready DSR artifacts. Understanding when to fine-tune vs. use RAG vs. prompt engineering is a key methodological decision for IS researchers.

| | | |
|-|-|-|
| [When to Fine-Tune vs. RAG vs. Prompting](#When-to-Fine-Tune-vs-RAG-vs-Prompting) | [Parameter-Efficient Fine-Tuning (PEFT)](#Parameter-Efficient-Fine-Tuning-PEFT) | [Instruction Tuning](#Instruction-Tuning) |
| [Full Fine-Tuning](#Full-Fine-Tuning) | [Domain Adaptation](#Domain-Adaptation) | [Tools & Libraries](#Tools--Libraries) |

---

> **IS Research Applications:** Fine-tune a sentiment classifier on domain-specific review corpora; adapt an LLM to generate realistic survey responses in a specific organizational context; train a document classifier on proprietary IS datasets; create a task-specific QA model grounded in IS theory.

---

### When to Fine-Tune vs. RAG vs. Prompting

Choosing the right adaptation strategy is as important as choosing the model. Use this decision framework:

| **Approach** | **Best When** | **Limitations** |
|-|-|-|
| **Prompt Engineering** | Task is well-defined; model already has required knowledge; quick iteration needed | Limited by context window; inconsistent on complex tasks; no learning from your data |
| **RAG** | External knowledge must be kept current; knowledge base is large; factual grounding is critical | Retrieval quality is a bottleneck; adds latency; requires vector database infrastructure |
| **Fine-Tuning** | Consistent style/format is critical; task requires specialized knowledge not in base model; high-volume inference | Requires labeled data; compute cost; risk of catastrophic forgetting; slow iteration |
| **PEFT (LoRA/QLoRA)** | Fine-tuning benefits apply but GPU memory is limited; multiple task adaptations needed | Slightly lower performance ceiling than full fine-tuning; requires more setup than prompting |

> **Rule of thumb for IS research:** Start with prompting → add RAG if factual grounding fails → fine-tune only if style/format consistency or domain vocabulary is critical.

---

### Parameter-Efficient Fine-Tuning (PEFT)
PEFT methods adapt large models by training only a small number of additional parameters, making fine-tuning accessible on consumer hardware.

**LoRA (Low-Rank Adaptation)**
The dominant PEFT method. Freezes the original model weights and injects trainable low-rank matrices into attention layers.

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685), 2021 - Introduces rank decomposition of weight update matrices. A 7B model can be fine-tuned on a single GPU with LoRA. [Code](https://github.com/microsoft/LoRA)

**QLoRA (Quantized LoRA)**
Combines 4-bit quantization with LoRA, enabling fine-tuning of 70B+ models on a single consumer GPU.

- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314), 2023 - Fine-tunes a 65B model on a single 48GB GPU to match ChatGPT performance. Made large-model fine-tuning democratically accessible. [Code](https://github.com/artidoro/qlora)

**Other PEFT Methods:**
- [Prefix Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190), 2021 - Prepends trainable continuous vectors to the input, leaving model weights frozen. [Code](https://github.com/XiangLi1999/PrefixTuning)
- [Prompt Tuning: The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691), 2021 - Learns soft prompt tokens prepended to inputs. Matches fine-tuning performance at scale with far fewer trainable parameters. [Code](https://github.com/google-research/prompt-tuning)
- [IA³: Infused Adapter by Inhibiting and Amplifying Inner Activations](https://arxiv.org/abs/2205.05638), 2022 - Scales activations with learned vectors; even more parameter-efficient than LoRA. [Code](https://github.com/r-three/t-few)

---

### Instruction Tuning
Instruction tuning fine-tunes models on (instruction, response) pairs, making them follow natural language directions reliably. It is the bridge between base language models and useful assistants.

- **Seminal Papers:**
  - [Finetuned Language Models Are Zero-Shot Learners (FLAN)](https://arxiv.org/abs/2109.01652), 2022 - Showed that fine-tuning on diverse NLP tasks phrased as instructions dramatically improves zero-shot generalization. [Code](https://github.com/google-research/FLAN)
  - [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560), 2022 - Bootstraps instruction-tuning data using the model itself, dramatically reducing human labeling costs. [Code](https://github.com/yizhongw/self-instruct)
  - [Alpaca: A Strong, Replicable Instruction-Following Model](https://crfm.stanford.edu/2023/03/13/alpaca.html), 2023 - Fine-tuned LLaMA-7B on 52K self-instruct examples for <$600. Sparked the open-source instruction-tuning ecosystem. [Code](https://github.com/tatsu-lab/stanford_alpaca)

- **IS Research Application — Synthetic Persona Generation:**
  - Fine-tune a model on domain-specific instruction pairs (e.g., "Respond as a skeptical CIO evaluating a new ERP system") to create consistent synthetic respondents for IS experiments.

---

### Full Fine-Tuning
Full fine-tuning updates all model weights and requires significant GPU memory but achieves the highest task-specific performance.

- **When appropriate:** Small models (<3B parameters); highly specialized domain vocabulary; when PEFT performance ceiling is insufficient; when training data is large and high-quality.

- **Papers with code:**
  - [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971), 2023 - The base model family most commonly full-fine-tuned for research applications. [Code](https://github.com/meta-llama/llama)
  - [Mistral 7B](https://arxiv.org/abs/2310.06825), 2023 - Highly efficient 7B model with sliding window attention. Strong base for domain-specific full fine-tuning. [Code](https://github.com/mistralai/mistral-src)

---

### Domain Adaptation
Adapting models to IS-relevant domains (healthcare, finance, legal, organizational behavior) via continued pre-training or supervised fine-tuning.

- **Papers with code:**
  - [BioMedLM](https://arxiv.org/abs/2304.15004), 2023 - GPT-2-scale model trained exclusively on biomedical literature (PubMed). Demonstrates domain-specific pre-training benefits. [Code](https://github.com/stanford-crfm/BioMedLM)
  - [FinGPT: Open-Source Financial Large Language Models](https://arxiv.org/abs/2306.06031), 2023 - Framework for fine-tuning LLMs on financial news, sentiment, and reports. Directly transferable to IS research in digital finance. [Code](https://github.com/AI4Finance-Foundation/FinGPT)
  - [LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning](https://arxiv.org/abs/2308.11462), 2023 - Benchmark for evaluating LLMs on legal reasoning tasks. Useful for IS research involving compliance, contracts, and governance. [Code](https://github.com/HazyResearch/legalbench)

---

### Tools & Libraries

| **Tool** | **Description** | **Best For** |
|-|-|-|
| [PEFT (HuggingFace)](https://github.com/huggingface/peft) | Official HuggingFace library implementing LoRA, QLoRA, prefix tuning, prompt tuning, and IA³. The standard entry point for PEFT. | All PEFT methods; integrates directly with Transformers and TRL |
| [TRL (HuggingFace)](https://github.com/huggingface/trl) | Combines supervised fine-tuning (SFT), reward modeling, and PPO/DPO RLHF in one library. | End-to-end instruction tuning + alignment pipelines |
| [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) | Configuration-driven fine-tuning framework wrapping HuggingFace + PEFT. Supports multi-GPU and FSDP with minimal code. | Production fine-tuning with minimal boilerplate |
| [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | Unified fine-tuning framework with web UI. Supports 100+ models with LoRA, QLoRA, full fine-tuning, and DPO. | Researchers who prefer GUI-based fine-tuning workflows |
| [Unsloth](https://github.com/unslothai/unsloth) | 2x faster, 70% less memory fine-tuning via hand-written CUDA kernels. Drop-in replacement for HuggingFace training. | Speed-critical fine-tuning on limited hardware |
| [MLflow](https://mlflow.org/) | Experiment tracking for fine-tuning runs — logs hyperparameters, loss curves, and model artifacts. | Reproducibility across fine-tuning experiments |
| [Weights & Biases](https://wandb.ai/) | Real-time training dashboards, model comparison, and dataset versioning. | Collaborative fine-tuning projects; comparing multiple training runs |

---

### Model Merging
Merging multiple fine-tuned models into a single model — combining task-specific expertise without additional training. Particularly useful when fine-tuning data from multiple IS domains is siloed.

- **Papers with code:**
  - [Model Soups: Averaging Weights of Multiple Fine-Tuned Models Improves Accuracy and Robustness](https://arxiv.org/abs/2203.05482), 2022 - Averaging weights of models fine-tuned from the same base improves robustness over any single model. [Code](https://github.com/mlfoundations/model-soups)
  - [TIES-Merging: Resolving Interference When Merging Models](https://arxiv.org/abs/2306.01708), 2023 - Addresses sign conflicts between merged model parameters, improving multi-task merging. [Code](https://github.com/prateeky2806/ties-merging)
  - [DARE: Language Model Arithmetic with Weight Disentanglement](https://arxiv.org/abs/2311.03099), 2023 - Prunes and rescales task vectors before merging to reduce parameter interference. [Code](https://github.com/yule-BUAA/MergeLM)

- **Tools:**
  - [MergeKit](https://github.com/arcee-ai/mergekit) - The standard toolkit for model merging. Supports linear, SLERP, TIES, DARE, and evolutionary merge strategies. Used to create most merged models on HuggingFace Hub.

---

### Alignment Beyond RLHF
New alignment methods that are simpler, more stable, or more data-efficient than standard RLHF/PPO.

- **Papers with code:**
  - [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691), 2024 - Incorporates preference alignment directly into the standard SFT loss via an odds ratio penalty. No reference model required — simpler and more memory-efficient than DPO. [Code](https://github.com/xfactlab/orpo)
  - [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734), 2024 - Simplifies DPO using average log-probability as the reward signal and a margin term. Outperforms DPO on AlpacaEval benchmarks. [Code](https://github.com/princeton-nlp/SimPO)
  - [SPIN: Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models](https://arxiv.org/abs/2401.01335), 2024 - Uses the model's own outputs as the "weak" reference for contrastive fine-tuning — iterative self-improvement without external preference data. [Code](https://github.com/uclaml/SPIN)

---

**Related Sections:** [LLMs & NLP](../NaturalLanguageProcessing/README.md) | [Reinforcement Learning](../ReinforcementLearning/README.md) | [Prompt Engineering](../Prompt-Engineering/README.md) | [AutoML](../AutoML/README.md)
