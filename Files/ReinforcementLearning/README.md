---
layout: default
title: "Reinforcement Learning"
parent: Topics
nav_order: 13
---

# Reinforcement Learning
*Papers and Repositories For Reinforcement Learning Tasks.*

*Last updated: February 2026*

| | | |
|-|-|-|
| [Reasoning Models](#Reasoning-Models) | [RLHF & Alignment](#RLHF--Alignment) | [RLVR & Verifiable Rewards](#RLVR--Verifiable-Rewards) |
| [Multi-Agent RL (MARL)](#Multi-Agent-RL-MARL) | [Tools & Environments](#Tools--Environments) | |

---

> **IS Research Applications:** Simulate complex organizational and market behaviors with MARL; fine-tune LLMs for domain-specific IS applications via RLHF; build adaptive DSR artifacts that improve through user feedback; model user behavior sequences and decision-making processes; create synthetic experimental participants using generative agents.

---

### Reasoning Models
RL has shifted from game-playing (e.g., AlphaGo) toward teaching language models to *reason* through problems before producing an answer — a paradigm known as "thinking" or "chain-of-thought RL."

- **Seminal Models:**
  - [OpenAI o1](https://openai.com/index/learning-to-reason-with-llms/), 2024 - Uses RL to train a model to produce extended internal reasoning ("thinking") traces before answering, significantly improving performance on math, coding, and scientific reasoning tasks.
  - [DeepSeek-R1](https://arxiv.org/abs/2501.12948), 2025 - Open-weight reasoning model trained with RL (GRPO) that matches or exceeds o1 on benchmarks. Demonstrates that long chain-of-thought reasoning can emerge from pure RL without supervised fine-tuning on reasoning traces. [Code](https://github.com/deepseek-ai/DeepSeek-R1)

---

### RLHF & Alignment
Reinforcement Learning from Human Feedback (RLHF) is the dominant technique for aligning LLM outputs with human preferences. It underpins ChatGPT, Claude, and Gemini.

- **Papers with code:**
  - [Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155), 2022 - The foundational RLHF paper. Fine-tunes GPT-3 using PPO against a reward model trained on human preference rankings. [Code](https://github.com/openai/following-instructions-human-feedback)
  - [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290), 2023 - Bypasses the explicit reward model, treating alignment as a classification problem on preference pairs. Simpler and more stable than PPO-based RLHF. [Code](https://github.com/eric-mitchell/direct-preference-optimization)
  - [GRPO: Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300), 2024 - The RL algorithm used to train DeepSeek models. Eliminates the critic network by normalizing rewards within a group of sampled responses, reducing memory and compute. [Code](https://github.com/deepseek-ai/DeepSeek-Math)

- **Papers without code:**
  - [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073), 2022 - Uses a set of principles (a "constitution") to guide AI self-critique and revision, replacing some human labeling with AI-generated preference data (RLAIF).

---

### RLVR & Verifiable Rewards
RL from Verifiable Rewards (RLVR) uses objective, programmatic reward signals (e.g., test suite pass/fail, math answer correctness) rather than learned reward models, making training more stable and scalable.

- **Datasets & Benchmarks:**
  - [GURU: A Comprehensive Benchmark for Medical and General Reasoning Under Uncertainty](https://arxiv.org/abs/2406.10270), NeurIPS 2025 - A large corpus spanning Math, Code, and Logic tasks designed specifically for RLVR training. Provides structured, verifiable answers for each problem, enabling scalable RL training without human labelers.
  - [MATH](https://arxiv.org/abs/2103.03874), 2021 - Competition-level mathematics problems with step-by-step solutions; a standard RLVR training and evaluation set. [Code](https://github.com/hendrycks/math)

- **Papers with code:**
  - [Let's Verify Step by Step (PRM800K)](https://arxiv.org/abs/2305.20050), 2023 - Process Reward Models (PRMs) that score each reasoning step individually, providing denser reward signals for RL training. [Code](https://github.com/openai/prm800k)

---

### Multi-Agent RL (MARL)
MARL studies how multiple agents learn and interact in a shared environment. It is particularly relevant for IS research because it enables computational simulation of social systems, markets, organizations, and user populations.

- **Paper Collections:**
  - [MARL-Papers](https://github.com/LantaoYu/MARL-Papers) - A comprehensive, curated list of multi-agent RL papers organized by topic (cooperation, competition, communication, emergent behavior). Essential starting point for MARL literature.

- **Papers with code:**
  - [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442), 2023 - Creates believable AI agents with memory, reflection, and planning that simulate human social behavior in a virtual town. Directly applicable to IS research for simulating user populations and organizational behavior. [Code](https://github.com/joonspk-research/generative_agents)
  - [QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11605), 2018 - A foundational cooperative MARL algorithm for decentralized execution with centralized training. [Code](https://github.com/oxwhirl/pymarl)

- **Papers without code:**
  - [An Overview of Multi-Agent Reinforcement Learning from Game Theoretical Perspective](https://arxiv.org/abs/2011.00583), 2021 - A comprehensive survey connecting MARL to game theory concepts (Nash equilibria, cooperative games), providing the theoretical foundation for IS applications.

---

### Tools & Environments
- **Training Frameworks:**
  - [verl (Volcano Engine Reinforcement Learning)](https://github.com/volcengine/verl) - A high-performance, production-ready RLHF/RLVR training library built for LLMs. Supports PPO, GRPO, DPO, and custom reward functions. Designed for scalability on GPU clusters.
  - [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl) - HuggingFace's library for fine-tuning and aligning LLMs with RL methods including PPO, DPO, and ORPO. The most accessible entry point for RLHF experimentation.
  - [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) - A scalable RLHF training framework supporting models >70B parameters with Ray-based distributed training.

- **Environments:**
  - [Jumanji](https://github.com/instadeepai/jumanji) - A JAX-based suite of combinatorial optimization and multi-agent RL environments. JAX acceleration makes it dramatically faster than CPU-based simulators, enabling large-scale agent population studies.
  - [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) - The standard multi-agent environment library (analogous to Gymnasium for single-agent RL). Provides competitive, cooperative, and mixed environments.

---

### Test-Time Compute & Search
A major 2024–2025 shift: rather than only scaling *training* compute, models now expend additional compute *at inference time* to search for better answers. This is the mechanism behind o1/R1-style "thinking."

- **Papers:**
  - [Scaling LLM Test-Time Compute Optimally Outperforms RLVR on Hard Reasoning](https://arxiv.org/abs/2408.03314), 2024 - Shows that best-of-N sampling with a process reward model (PRM) can match or exceed RL-trained reasoning models when inference budget is sufficient. Key trade-off: training cost vs. inference cost.
  - [Thinking LLMs: General Instruction Following with Thought Generation](https://arxiv.org/abs/2410.10630), 2024 - Fine-tunes models to generate internal "thoughts" before answering, without RL — a supervised alternative to o1-style training. [Code](https://github.com/thought-gen/thinking-llms)
  - [MCTS-based Reasoning: Enhancing LLMs with Monte Carlo Tree Search](https://arxiv.org/abs/2406.07394), 2024 - Applies MCTS to LLM decoding, treating each reasoning step as a tree node and using a value function to guide search. Applicable to complex IS research tasks requiring multi-step analysis.

- **Benchmarks for Reasoning:**
  - [AIME (American Invitational Mathematics Examination)](https://artofproblemsolving.com/wiki/index.php/AIME_Problems_and_Solutions) - Competition math benchmark used to evaluate o1/R1 reasoning capability; now standard for measuring reasoning model progress.
  - [LiveCodeBench](https://livecodebench.github.io/) - Contamination-free coding benchmark that continuously updates with new competitive programming problems. [Code](https://github.com/LiveCodeBench/LiveCodeBench)

---

### Offline RL & Decision Making from Data
RL without environment interaction — learning policies from fixed logged datasets. Relevant for IS research where real-time interaction is impossible (e.g., learning from historical user clickstreams).

- **Papers with code:**
  - [Conservative Q-Learning (CQL) for Offline Reinforcement Learning](https://arxiv.org/abs/2006.04779), 2020 - Trains a conservative Q-function that lower-bounds values on out-of-distribution actions, enabling safe offline learning. [Code](https://github.com/aviralkumar2907/CQL)
  - [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/pdf/2106.01345v2.pdf), 2021 - Reframes offline RL as conditional sequence generation (already in NLP section; also foundational here). [Code](https://github.com/kzl/decision-transformer)
  - [Offline Reinforcement Learning as One Big Sequence Modeling Problem](https://arxiv.org/abs/2106.02039), 2021 - Treats entire trajectories as sequences, enabling transformer architectures to learn from diverse offline datasets. [Code](https://github.com/JannerM/trajectory-transformer)

---

**Related Sections:** [LLMs & NLP](../NaturalLanguageProcessing/README.md) | [Fine-Tuning](../FineTuning/README.md) | [Generative Media & Synthetic Data](../DataGeneration/README.md) | [Prompt Engineering](../Prompt-Engineering/README.md)
