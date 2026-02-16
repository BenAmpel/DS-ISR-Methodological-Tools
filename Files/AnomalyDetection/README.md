---
layout: default
title: "Anomaly Detection"
parent: Topics
nav_order: 10
permalink: /Files/AnomalyDetection/
---


{::nomarkdown}
<div class="section-meta">
  <span class="meta-pill meta-reading">⏱ ~8 min read</span>
  <span class="meta-pill meta-updated">📅 Updated Feb 2026</span>
  <span class="meta-pill meta-intermediate">📊 Intermediate</span>
  <span class="meta-pill meta-prereq">🔑 Python, Statistics</span>
</div>
{:/}

# Anomaly Detection
*Papers, Libraries, and Tools for Unsupervised and Semi-Supervised Detection of Anomalous Data Points.*

*Last updated: February 2026*

> Anomaly detection identifies data points that deviate significantly from expected patterns without requiring labeled examples of anomalies. In IS research, this encompasses fraud detection, system intrusion detection, quality control, outlier identification in surveys, and detecting unusual user behavior in digital systems.

| | | |
|-|-|-|
| [Classical Methods](#classical-methods) | [Deep Learning Methods](#deep-learning-methods) | [Time Series Anomaly Detection](#time-series-anomaly-detection) |
| [LLM-Based Anomaly Detection](#llm-based-anomaly-detection) | [Tools & Libraries](#tools-libraries) | [Benchmarks](#benchmarks) |

---

> **IS Research Applications:** Detect fraudulent transactions or fake reviews; identify unusual login patterns in security research; flag outlier survey responses; monitor data quality in longitudinal studies; detect concept drift in deployed IS artifacts.

---

### Classical Methods

- **Seminal Papers:**
  - [Isolation Forest](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4781136), 2008 - Isolates anomalies by recursively partitioning data. Anomalies require fewer partitions and thus have shorter path lengths. Remains one of the fastest and most effective methods for tabular data. [Code](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.isolationforest.html)
  - [Local Outlier Factor (LOF)](https://dl.acm.org/doi/10.1145/342009.335388), 2000 - Measures local density deviation of a data point relative to its neighbors. Effective for datasets with varying density clusters.
  - [One-Class SVM](https://proceedings.neurips.cc/paper/1999/hash/8725fb777f25776ffa9076e44fcfd776-abstract.html), 1999 - Learns a decision boundary around normal data in a high-dimensional feature space.

- **Papers with code:**
  - [Extended Isolation Forest](https://arxiv.org/abs/1811.02141), 2019 - Addresses the bias artifact of original Isolation Forest by randomizing hyperplane orientations. [Code](https://github.com/sahandha/eif)
  - [COPOD: Copula-Based Outlier Detection](https://arxiv.org/abs/2009.09463), 2020 - Uses copulas to model tail probabilities for efficient, parameter-free outlier detection. [Code](https://github.com/yzhao062/pyod)

---

### Deep Learning Methods

**Autoencoder-Based Detection**
Train an autoencoder on normal data; anomalies are identified by high reconstruction error.

- [Robust Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection](https://arxiv.org/abs/1805.06725), 2018 - Combines autoencoders with Gaussian mixture models for robust anomaly scoring. [Code](https://github.com/danieltan07/dagmm)
- [Anomaly Detection with Robust Deep Autoencoders](https://dl.acm.org/doi/abs/10.1145/3097983.3098052), 2017 - Jointly learns robust PCA and sparse encoding for anomaly detection.

**Transformer-Based Detection**
- [Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy](https://arxiv.org/abs/2110.02642), 2022 - Uses the discrepancy between point-wise and segment-wise attention associations as an anomaly criterion. [Code](https://github.com/thuml/anomaly-transformer)
- [TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series](https://arxiv.org/abs/2201.07284), 2022 - Transformer architecture with adversarial training for multivariate time series anomaly detection. [Code](https://github.com/imperial-qore/tranad)

**Flow-Based & VAE Methods**
- [Normalizing Flows for Novelty Detection in Time Series](https://arxiv.org/abs/2202.01060), 2022 - Normalizing flows model the distribution of normal data; anomalies have low likelihood.

---

### Time Series Anomaly Detection
A critical application domain for IS research: monitoring system logs, transaction streams, sensor data, and user behavioral sequences.

- **Papers with code:**
  - [MSCRED: A Multi-Scale Convolutional Recurrent Encoder-Decoder for System Failure Detection](https://arxiv.org/abs/1811.04520), 2019 - Captures temporal patterns across multiple scales for multi-sensor system monitoring. [Code](https://github.com/zhang-zhi-jie/pytorch-mscred)
  - [USAD: UnSupervised Anomaly Detection on Multivariate Time Series](https://dl.acm.org/doi/10.1145/3394486.3403392), 2020 - Fast training via adversarial autoencoders with amplified anomaly signals. [Code](https://github.com/manigalati/usad)
  - [Revisiting Time Series Outiler Detection](https://arxiv.org/abs/2009.08808), 2020 - Benchmark study comparing 20+ methods across diverse time series anomaly datasets.

- **Benchmarks:**
  - [TSB-UAD](https://github.com/thedatumorg/tsb-uad) - Comprehensive benchmark of 15 anomaly detection algorithms on 1980 time series. Essential reference for method selection.

---

### LLM-Based Anomaly Detection
LLMs bring semantic understanding to anomaly detection, enabling detection of contextually unusual patterns in text that rule-based or statistical methods would miss.

- **Papers with code:**
  - [AnomalyGPT: Detecting Industrial Anomalies using Large Vision-Language Models](https://arxiv.org/abs/2308.15366), 2023 - Uses LLaVA-style VLM to detect visual anomalies in industrial images without anomaly training data. [Code](https://github.com/casia-iva-lab/anomalygpt)
  - [LogGPT: Exploring ChatGPT for Log-Based Anomaly Detection](https://arxiv.org/abs/2309.01189), 2023 - Prompts LLMs to detect anomalous system log sequences using in-context learning. Demonstrates strong zero-shot log anomaly detection.

- **Papers without code:**
  - [Large Language Models for Anomaly Detection in Computational Workflows](https://arxiv.org/abs/2404.16337), 2024 - Systematic evaluation of LLMs for workflow anomaly detection; includes prompt strategies for IS monitoring applications.

- **IS Research Note:** LLMs can detect semantic anomalies in unstructured text (e.g., unusual review content, atypical email language) where traditional statistical methods fail due to the high dimensionality of text.

---

### Tools & Libraries

| **Tool** | **Description** | **Best For** |
|-|-|-|
| [PyOD](https://github.com/yzhao062/pyod) | The most comprehensive Python outlier detection library. 40+ algorithms including Isolation Forest, LOF, COPOD, AutoEncoders, and deep methods. Unified API across all methods. | **Primary tool for tabular anomaly detection in IS research** |
| [PyGOD](https://github.com/pygod-team/pygod) | Graph outlier detection built on PyOD principles. Detects anomalous nodes and edges in graphs. | Fraud detection in transaction networks; social network anomaly detection |
| [TimeEval](https://github.com/timeeval/timeeval) | Evaluation toolkit for time series anomaly detection algorithms with reproducible benchmarking. | Comparing time series methods for IS monitoring applications |
| [STUMPY](https://github.com/tdameritrade/stumpy) | Fast matrix profile computation for time series motif and anomaly discovery. | Detecting unusual subsequences in behavioral log data |
| [Alibi Detect](https://github.com/seldonio/alibi-detect) | Drift detection, outlier detection, and adversarial detection for production ML systems. | Monitoring deployed IS artifacts for distribution shift |
| [sklearn.ensemble.IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.isolationforest.html) | Scikit-learn's built-in Isolation Forest. No additional installation required. | Quick baseline; integration with existing sklearn pipelines |

---

### Benchmarks
- [ADBench](https://github.com/minqi824/adbench) - Anomaly detection benchmark with 57 datasets and 30 algorithms. Comprehensive comparison for tabular data.
- [Top 96 Anomaly Detection Projects](https://awesomeopensource.com/projects/anomaly-detection) - Curated collection of open-source anomaly detection projects.
- [Awesome Anomaly Detection](https://github.com/hoya012/awesome-anomaly-detection) - Curated paper list organized by domain (image, video, time series, NLP).

---

### Foundation Models for Anomaly Detection
A 2024–2025 trend: using pre-trained foundation models as zero-shot or few-shot anomaly detectors, eliminating the need for domain-specific training data.

- **Papers with code:**
  - [AnomalyGPT: Detecting Industrial Anomalies using Large Vision-Language Models](https://arxiv.org/abs/2308.15366), 2023 - Prompts GPT-4V-style models to identify visual defects with natural language descriptions. [Code](https://github.com/casia-iva-lab/anomalygpt)
  - [WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation](https://arxiv.org/abs/2303.14814), 2023 - Uses CLIP's vision-language alignment for zero-shot visual anomaly detection without any anomaly training examples. [Code](https://github.com/caoyunkang/winclip)
  - [UniFormaly: Towards Task-Agnostic Unified Framework for Visual Anomaly Detection](https://arxiv.org/abs/2307.12325), 2023 - Unified framework combining reconstruction- and embedding-based methods under one architecture. [Code](https://github.com/zhiyuanyou/uniformaly)

- **IS Research Note:** Foundation model anomaly detectors enable IS researchers to detect unusual patterns in new domains (e.g., atypical UI interactions, unusual enterprise system logs) without collecting labeled anomaly examples — a major practical advantage.

---

### Graph-Based Anomaly Detection
When IS data has inherent relational structure (transaction networks, social networks, supply chains), graph methods substantially outperform tabular approaches.

- **Papers with code:**
  - [DOMINANT: Deep Anomaly Detection on Attributed Networks](https://epubs.siam.org/doi/abs/10.1137/1.9781611975673.67), 2019 - Graph autoencoder that reconstructs both network structure and node attributes; anomalies have high reconstruction error. [Code](https://github.com/kaize0409/gcn_anomalydetection)
  - [CoLA: Contrastive Self-Supervised Learning for Graph Anomaly Detection](https://arxiv.org/abs/2103.00859), 2021 - Contrastive learning between local subgraph and global graph context for node-level anomaly detection. [Code](https://github.com/tianxiangzhao/cola)
  - [GRADATE: Graph Anomaly Detection via Multi-Scale Contrastive Learning Networks](https://arxiv.org/abs/2212.00535), 2023 - Multi-scale contrastive learning across node, subgraph, and graph levels. [Code](https://github.com/felixdjc/gradate)

- **Tools:**
  - [PyGOD](https://github.com/pygod-team/pygod) - The primary Python library for graph outlier detection. Implements 10+ algorithms with a unified API. [Paper](https://arxiv.org/abs/2204.12095)

---

**Related Sections:** [Graphs](../Graphs/README.md) | [LLM Safety & Adversarial Defense](../AdversarialDefense/README.md) | [Interpretability](../Interpretability/README.md) | [Python Tools](../PythonTools/README.md)
