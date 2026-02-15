# LLM Safety & Adversarial Defense
*Papers and Repositories for Defending Against Adversarial Attacks on ML Models and Ensuring LLM Safety.*

> **Note:** The adversarial ML landscape has shifted significantly. While pixel-perturbation attacks on vision models remain relevant, the dominant safety challenges in 2025 are **LLM-specific**: jailbreaking, prompt injection, hallucination, and alignment failures. This section covers both classical adversarial defense and the emerging LLM safety literature.

| | | |
|-|-|-|
| [LLM Safety & Alignment](#LLM-Safety--Alignment) | [Jailbreaking & Red Teaming](#Jailbreaking--Red-Teaming) | [Prompt Injection](#Prompt-Injection) |
| [Trustworthy GNNs](#Trustworthy-GNNs) | [Classical Adversarial Defense](#Classical-Adversarial-Defense) | |

---

### LLM Safety & Alignment
Safety in LLMs encompasses ensuring that model outputs are helpful, harmless, and honest (the "HHH" framework). This is an active research area closely tied to RLHF and Constitutional AI.

- **Papers with code:**
  - [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073), 2022 - Anthropic's approach to alignment: the model critiques and revises its own outputs according to a set of principles (a "constitution"), reducing harmful outputs without requiring human labeling for every case. [Code](https://github.com/anthropics/hh-rlhf)
  - [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290), 2023 - Treats alignment as preference learning, bypassing explicit reward modeling. [Code](https://github.com/eric-mitchell/direct-preference-optimization)
  - [Llama Guard: LLM-based Input-Output Safeguard](https://arxiv.org/abs/2312.06674), 2023 - A fine-tuned LLaMA model that classifies inputs and outputs against a safety taxonomy. Deployable as a safety layer in production systems. [Code](https://github.com/meta-llama/PurpleLlama)

- **Papers without code:**
  - [Holistic Evaluation of Language Models (HELM)](https://arxiv.org/abs/2211.09110), 2022 - A comprehensive benchmark framework for evaluating LLMs across accuracy, calibration, robustness, fairness, bias, toxicity, and efficiency.
  - [TruthfulQA: Measuring How Models Mimic Human Falsehoods](https://arxiv.org/abs/2109.07958), 2021 - A benchmark of questions that humans answer falsely due to misconceptions; tests whether LLMs reproduce these falsehoods.

---

### Jailbreaking & Red Teaming
"Jailbreaking" refers to adversarial prompts that bypass LLM safety filters, causing models to produce harmful outputs. Red teaming is the systematic process of probing for these vulnerabilities before deployment.

- **Papers with code:**
  - [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043), 2023 - Demonstrates that a single adversarial suffix can reliably bypass safety training on multiple models, including closed-source APIs. [Code](https://github.com/llm-attacks/llm-attacks)
  - [Red Teaming Language Models with Language Models](https://arxiv.org/abs/2202.03286), 2022 - Automates red teaming by using an LLM to generate test cases targeting a victim LLM. [Code](https://github.com/anthropics/hh-rlhf)

- **Resources:**
  - [Awesome LLM Safety](https://github.com/ydyjya/Awesome-LLM-Safety) - Curated collection of papers on LLM safety, alignment, robustness, and red teaming.
  - [AI Safety Fundamentals](https://course.aisafetyfundamentals.com/) - Introductory curriculum covering alignment, interpretability, and governance.

---

### Prompt Injection
Prompt injection attacks embed malicious instructions within data that an LLM processes, causing it to follow the attacker's instructions instead of the user's. Directly relevant for IS researchers building LLM-powered artifacts that process external data.

- **Papers with code:**
  - [Prompt Injection Attacks and Defenses in LLM-Integrated Applications](https://arxiv.org/abs/2310.12815), 2023 - Systematic taxonomy of prompt injection attack types and a framework for evaluating defenses. [Code](https://github.com/liu00222/Open-Prompt-Injection)

- **Papers without code:**
  - [Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173), 2023 - Demonstrates real-world indirect prompt injection attacks via web content, emails, and documents processed by LLM agents.

---

### Trustworthy GNNs
Graph models face unique adversarial challenges including node injection, edge perturbation, and structural attacks. Trustworthiness in GNNs spans robustness, fairness, explainability, and privacy.

- **Papers with code:**
  - [Graph Structure Learning for Robust Graph Neural Networks](https://arxiv.org/pdf/2005.10203.pdf), 2020 - Protects GNNs against structural adversarial attacks by jointly learning a clean graph structure. [Code](https://github.com/ChandlerBang/Pro-GNN)
  - [Certifiable Robustness of Graph Convolutional Networks under Structure Perturbations](https://dl.acm.org/doi/10.1145/3394486.3403217), 2020 - Provides certified guarantees for GCN robustness. [Code](https://www.in.tum.de/daml/robust-gcn/)

- **Papers without code:**
  - [Trustworthy Graph Neural Networks: Aspects, Methods, and Trends](https://arxiv.org/abs/2205.07424), 2022 - Comprehensive survey covering robustness, explainability, fairness, privacy, and accountability in GNNs. Includes emerging 2025 results.

---

### Classical Adversarial Defense
The foundational literature on adversarial examples and defenses for vision and classification models.

[A Complete List of All (arXiv) Adversarial Example Papers](https://nicholas.carlini.com/writing/2019/all-adversarial-example-papers.html)

- **Papers with code:**
  - [AdvMind: Inferring Adversary Intent of Black-Box Attacks](https://arxiv.org/pdf/2006.09539.pdf), 2020 - Infers the adversary intent behind black-box adversarial attacks. [Code](https://github.com/ain-soph/trojanzoo)
  - [An Embarrassingly Simple Approach for Trojan Attack in Deep Neural Networks](https://arxiv.org/pdf/2006.08131.pdf), 2020 - Demonstrates the ease of embedding backdoor triggers in DNN systems. [Code](https://github.com/trx14/TrojanNet)

- **Papers without code:**
  - [Interpretability is a Kind of Safety: An Interpreter-based Ensemble for Adversary Defense](https://dl.acm.org/doi/abs/10.1145/3394486.3403044), 2020 - Interpreter-based ensemble framework for detecting and defending adversarial attacks.

---
