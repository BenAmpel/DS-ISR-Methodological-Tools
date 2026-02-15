---
layout: default
title: "Causal Inference"
parent: Topics
nav_order: 9
---

# Causal Inference & Causal AI
*Methods and Libraries for Moving from Correlation to Causation in IS Research.*

*Last updated: February 2026*

> IS research is inherently causal — we ask "does IT adoption *cause* performance improvement?" not "is IT adoption *correlated with* performance improvement?" Traditional ML predicts outcomes; causal inference identifies *why* outcomes occur and *what would happen* under interventions. This section bridges the econometrics tradition of IS research with modern causal AI tools.

| | | |
|-|-|-|
| [Causal Inference Frameworks](#causal-inference-frameworks) | [Causal Discovery](#causal-discovery) | [Double Machine Learning](#double-machine-learning) |
| [Heterogeneous Treatment Effects](#heterogeneous-treatment-effects) | [Tools & Libraries](#tools-libraries) | [IS Research Applications](#is-research-applications) |

---

> **IS Research Applications:** Estimate the causal effect of a new IS artifact on organizational outcomes (controlling for confounders); discover causal structure in enterprise process data; move from "predictive" to "explanatory" IS theory using observational data; satisfy reviewers demanding causal rather than correlational evidence.

---

### Causal Inference Frameworks
The foundational tools for expressing and estimating causal relationships from observational data.

**DoWhy (Microsoft)**
The leading Python library for causal inference. Implements the four-step causal reasoning workflow: Model → Identify → Estimate → Refute.

- [DoWhy: An End-to-End Library for Causal Inference](#https://arxiv.org/abs/2011.04216), 2020 - Introduces a unified causal inference workflow that makes assumptions explicit and tests them systematically. [Code](#https://github.com/py-why/dowhy)
- [DoWhy Documentation & Tutorials](#https://py-why.github.io/dowhy/) - Comprehensive guide with IS-relevant examples (A/B tests, observational studies, instrumental variables).

**CausalML (Uber)**
Meta-learner and tree-based methods for causal inference, specialized for treatment effect estimation in industry settings.

- [CausalML: A Python Package for Uplift Modeling and Causal Inference with ML](#https://arxiv.org/abs/2002.11631), 2020 - Implements S-learner, T-learner, X-learner, and R-learner meta-learners for heterogeneous treatment effects. [Code](#https://github.com/uber/causalml)

---

### Causal Discovery
Learning the causal graph structure directly from data — answering "what causes what?" before estimating effect sizes.

- **Papers with code:**
  - [DAGs with NO TEARS: Continuous Optimization for Structure Learning](#https://arxiv.org/abs/1803.01422), 2018 - Reformulates causal graph discovery as a continuous optimization problem, making it tractable for IS datasets. [Code](#https://github.com/xunzheng/notears)
  - [Causal Discovery with Score Matching on Additive Noise Models](#https://arxiv.org/abs/2302.12202), 2023 - Score-based causal discovery applicable to mixed tabular IS data. [Code](#https://github.com/py-why/causal-learn)
  - [Tigramite: Causal Discovery for Time Series](#https://arxiv.org/abs/1702.07007), 2017 - PCMCI algorithm for discovering causal relationships in time series IS data (e.g., system logs, user behavior sequences). [Code](#https://github.com/jakobrunge/tigramite)

- **Tools:**
  - [causal-learn](#https://github.com/py-why/causal-learn) - Python implementation of classic causal discovery algorithms: PC, FCI, GES, LiNGAM, and score-based methods. The standard library for causal graph discovery.

---

### Double Machine Learning
Combining high-dimensional ML with classical causal econometrics — estimating causal effects even when there are many confounders.

- **Papers with code:**
  - [Double/Debiased Machine Learning for Treatment and Structural Parameters](#https://arxiv.org/abs/1608.00060), 2018 (Chernozhukov et al.) - The foundational DML paper. Uses cross-fitting to remove regularization bias when using ML for nuisance estimation. Directly applicable to IS panel data studies. [Code](#https://github.com/doubleml/doubleml-for-py)
  - [DoubleML: An Object-Oriented Implementation of Double Machine Learning](#https://arxiv.org/abs/2103.09603), 2021 - Python and R implementation of DML with PLR, PLIV, IRM models. [Code](#https://github.com/doubleml/doubleml-for-py)

- **IS Research Note:** DML is particularly valuable for IS research using administrative or observational panel data where randomization is impossible. It allows high-dimensional controls (firm characteristics, time effects) while recovering unbiased causal estimates.

---

### Heterogeneous Treatment Effects
Estimating *who benefits most* from an IS intervention — crucial for personalized artifact design and subgroup analysis.

- **Papers with code:**
  - [Generalized Random Forests](#https://arxiv.org/abs/1610.01271), 2019 - Non-parametric forest estimator for conditional average treatment effects (CATE). [Code](#https://github.com/grf-labs/grf)
  - [Quasi-oracle Estimation of Heterogeneous Treatment Effects](#https://arxiv.org/abs/1712.04912), 2019 - The R-learner; a flexible meta-learning approach to CATE estimation. [Code](#https://github.com/xnie/rlearner)

- **Tools:**
  - [EconML (Microsoft)](#https://github.com/py-why/econml) - The primary Python library for heterogeneous treatment effects. Implements DML, DR-learner, Causal Forest, and IV estimators. Excellent documentation with IS-relevant use cases.

---

### Tools & Libraries

| **Tool** | **Description** | **Best For** |
|-|-|-|
| [DoWhy](#https://github.com/py-why/dowhy) | Four-step causal workflow (model, identify, estimate, refute). The standard starting point for IS causal analysis. | Full causal inference pipeline with explicit assumption testing |
| [EconML](#https://github.com/py-why/econml) | Heterogeneous treatment effect estimation with DML, Causal Forest, and IV. | Estimating who benefits from an IS intervention |
| [CausalML](#https://github.com/uber/causalml) | Uplift modeling and meta-learners for treatment effect estimation. | A/B test analysis and feature importance in causal models |
| [causal-learn](#https://github.com/py-why/causal-learn) | Causal discovery algorithms (PC, FCI, GES, LiNGAM). | Learning causal graph structure from IS observational data |
| [Tigramite](#https://github.com/jakobrunge/tigramite) | Time series causal discovery (PCMCI). | Causal analysis of temporal IS data (logs, usage sequences) |
| [DoubleML](#https://github.com/doubleml/doubleml-for-py) | Double/Debiased ML for high-dimensional confounding. | Observational IS panel studies with many controls |
| [grf (R)](#https://github.com/grf-labs/grf) | Generalized Random Forests for CATE estimation. | R-based IS research needing subgroup treatment effects |

---

### IS Research Applications

**Common IS Causal Questions:**
- *Does IT investment cause firm performance?* → Use DoWhy with instrumental variables or difference-in-differences.
- *Which users benefit most from a recommendation system?* → Use EconML Causal Forest for heterogeneous effects.
- *What are the causal drivers of system adoption?* → Use causal-learn to discover the causal graph from survey data.
- *Does AI-assisted decision making improve outcomes for all subgroups?* → Use DML + EconML to estimate conditional effects.

**Connecting Causal AI to IS Theory:**
Causal inference tools can help validate IS theories that make causal claims (TAM, DeLone & McLean, Resource-Based View). Rather than reporting correlations, estimate causal effects with proper identification strategies and test for effect heterogeneity across user segments, firm sizes, or industries.

**Recommended Reading:**
- [The Effect: An Introduction to Research Design and Causality](#https://theeffectbook.net/) - Free textbook connecting causal inference to empirical research design. Essential companion for IS researchers using these tools.
- [Causal Inference: The Mixtape](#https://mixtape.scunning.com/) - Accessible treatment of causal econometrics with Python/R code. Covers DiD, IV, RDD, and matching.

---

**Related Sections:** [LLMs & NLP](../NaturalLanguageProcessing/README.md) | [Interpretability](../Interpretability/README.md) | [Evaluation & Benchmarking](../Evaluation/README.md) | [Python Tools](../PythonTools/README.md) | [Ethics](../Ethics/README.md)
