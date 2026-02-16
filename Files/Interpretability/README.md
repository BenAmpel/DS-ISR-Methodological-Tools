---
layout: default
title: "Interpretability & Explainability"
parent: Topics
nav_order: 7
permalink: /Files/Interpretability/
---


{::nomarkdown}
<div class="section-meta">
  <span class="meta-pill meta-reading">⏱ ~10 min read</span>
  <span class="meta-pill meta-updated">📅 Updated Feb 2026</span>
  <span class="meta-pill meta-intermediate">📊 Intermediate</span>
  <span class="meta-pill meta-prereq">🔑 ML basics, scikit-learn</span>
</div>
{:/}

# Interpretability & Explainability
*Methods, Papers, and Tools for Understanding What Machine Learning Models Have Learned and Why They Make Predictions.*

*Last updated: February 2026*

> Interpretability is not optional for IS research — it is a methodological requirement. Peer reviewers, ethics boards, and practitioners need to understand *why* a model makes a prediction, not just *that* it does. This section covers the spectrum from local post-hoc explanations (why this prediction?) to global mechanistic understanding (what did the model learn?).

| | | |
|-|-|-|
| [Post-Hoc Explanation Methods](#post-hoc-explanation-methods) | [Attention Visualization](#attention-visualization) | [LLM Interpretability](#llm-interpretability) |
| [GNN Explainability](#gnn-explainability) | [Evaluation of Explanations](#evaluation-of-explanations) | [Tools & Libraries](#tools-libraries) |

---

> **IS Research Applications:** Justify model decisions to stakeholders in DSR artifact evaluations; satisfy IRB/ethics requirements for AI-assisted research; identify dataset biases before publication; provide feature importance evidence in theory-building; build trust in AI-powered IS artifacts.

---

### Post-Hoc Explanation Methods
Post-hoc methods explain predictions of any black-box model without modifying it.

**SHAP (SHapley Additive exPlanations)**
The gold standard for feature attribution. Uses Shapley values from cooperative game theory to fairly distribute prediction credit across input features.

- [A Unified Approach to Interpreting Model Predictions (SHAP)](https://arxiv.org/abs/1705.07874), 2017 - Introduces SHAP values as a unified framework unifying LIME, DeepLIFT, and other attribution methods. Provides both local (per-prediction) and global (dataset-level) explanations. [Code](https://github.com/shap/shap)
- [TreeSHAP: Consistent Individualized Feature Attribution for Tree Ensembles](https://arxiv.org/abs/1802.03888), 2018 - Polynomial-time exact SHAP computation for tree-based models (XGBoost, LightGBM, Random Forest). [Code](https://github.com/shap/shap)

**LIME (Local Interpretable Model-agnostic Explanations)**
Approximates the model locally around a prediction with an interpretable surrogate model.

- ["Why Should I Trust You?": Explaining the Predictions of Any Classifier (LIME)](https://arxiv.org/abs/1602.04938), 2016 - Fits a linear model to locally perturbed samples to explain individual predictions. Works for tabular, text, and image data. [Code](https://github.com/marcotcr/lime)

**Integrated Gradients**
Attributes predictions to input features by integrating gradients along the path from a baseline to the input.

- [Axiomatic Attribution for Deep Networks](https://arxiv.org/abs/1703.01365), 2017 - Introduces Integrated Gradients satisfying sensitivity and implementation invariance axioms. Preferred for neural networks over simple gradient methods. [Code](https://github.com/ankurtaly/integrated-gradients)

**Counterfactual Explanations**
Answers "what is the minimum change to the input that would change the prediction?" — particularly intuitive for human stakeholders.

- [Counterfactual Explanations without Opening the Black Box: Automated Decisions and the GDPR](https://arxiv.org/abs/1711.00399), 2018 - Introduces counterfactual explanation as the legally motivated explanation format (directly addresses GDPR Article 22 right to explanation). Highly relevant for IS research in digital governance.
- [DiCE: Diverse Counterfactual Explanations for Any ML Classifier](https://arxiv.org/abs/1905.07697), 2020 - Generates diverse, actionable counterfactuals. [Code](https://github.com/interpretml/dice)

---

### Attention Visualization
Attention weights in transformers can be visualized to show which input tokens the model attended to when producing an output. Note: attention ≠ explanation (see papers below), but visualization remains a useful diagnostic tool.

- **Tools:**
  - [BertViz](https://github.com/jessevig/bertviz) - Interactive visualization of attention in BERT, GPT, and T5. Supports head-level and layer-level attention views.
  - [Ecco](https://github.com/jalammar/ecco) - Explains LLM outputs via neuron activations, token rankings, and attention patterns. Includes Jupyter notebook integration.
  - [TransformerLens](https://github.com/neelnanda-io/transformerlens) - Mechanistic interpretability library for GPT-style models. Enables activation patching, causal tracing, and circuit analysis.

- **Caution paper:**
  - [Attention is not Explanation](https://arxiv.org/abs/1902.10186), 2019 - Demonstrates that attention weights do not reliably indicate which inputs are important for a prediction. Essential reading before using attention as an explanation in IS papers.
  - [Attention is not not Explanation](https://arxiv.org/abs/1908.04626), 2019 - Nuanced response arguing attention can provide explanations under certain conditions. Read alongside the above.

---

### LLM Interpretability
As LLMs become core DSR artifacts, understanding *what* they know and *how* they reason becomes critical for scientific validity.

**Mechanistic Interpretability**
Reverse-engineering the internal computations of neural networks to identify human-interpretable circuits and features.

- [Toy Models of Superposition](https://arxiv.org/abs/2209.11170), 2022 - Demonstrates that neural networks represent more features than they have neurons via superposition. Foundational for understanding LLM internal structure. [Code](https://github.com/anthropics/toy-models-of-superposition)
- [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features/index.html), 2023 - Uses sparse autoencoders to find monosemantic features in MLP layers of LLMs. A breakthrough in practical mechanistic interpretability. [Code](https://github.com/anthropics/sae-vis)

**Probing & Knowledge Localization**
- [ROME: Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262), 2022 - Shows that factual knowledge is stored in specific MLP layers and can be surgically edited. [Code](https://github.com/kmeng01/rome)
- [Language Models as Knowledge Bases?](https://arxiv.org/abs/1909.01066), 2019 - Evaluates how much factual knowledge is stored in LLM weights via fill-in-the-blank probes.

**Saliency & Attribution for LLMs:**
- [SHAP for Text](https://shap.readthedocs.io/en/latest/text_examples.html) - Applies SHAP values to transformer text classifiers to identify which tokens drove a prediction.

---

### GNN Explainability
See also [Graphs](../Graphs/README.md) for domain context.

- **Papers with code:**
  - [GNNExplainer: Generating Explanations for Graph Neural Networks](https://arxiv.org/abs/1903.03894), 2019 - Identifies the subgraph structure and node features most important for a GNN prediction. [Code](https://github.com/rexying/gnn-model-explainer)
  - [Interpreting Graph Neural Networks for NLP With Differentiable Edge Masking](https://openreview.net/pdf?id=wznmqa42zax), 2021 - Drops edges to identify which graph connections drive NLP predictions. [Code](https://github.com/michschli/graphmask)
  - [XGNN: Towards Model-Level Explanations of Graph Neural Networks](https://arxiv.org/pdf/2006.02587.pdf), 2020 - Graph generator that produces graph patterns maximizing a target prediction class.

---

### Evaluation of Explanations
Explanations themselves must be evaluated — faithfulness, stability, and human comprehensibility are distinct properties.

- **Papers:**
  - [Towards Faithfully Interpretable NLP Systems: On the Coherence of Attention Weights and Gradient-based Feature Importance](https://arxiv.org/abs/2009.01416), 2020 - Framework for evaluating whether explanations faithfully reflect model behavior.
  - [The (Un)reliability of Saliency Methods](https://arxiv.org/abs/1711.00867), 2017 - Shows that many gradient-based saliency methods are sensitive to irrelevant transformations.
  - [Evaluating the Faithfulness of Importance Measures in NLP by Robustness to Scrubbing](https://arxiv.org/abs/2010.00154), 2020 - Tests whether removing "important" features actually changes predictions as expected.

---

### Tools & Libraries

| **Tool** | **Description** | **Best For** |
|-|-|-|
| [SHAP](https://github.com/shap/shap) | Feature attribution via Shapley values. Works for tabular, text, and image models. | Publication-quality feature importance plots for IS papers |
| [LIME](https://github.com/marcotcr/lime) | Local surrogate model explanations. | Quick per-prediction explanations for any classifier |
| [InterpretML](https://github.com/interpretml/interpret) | Microsoft's toolkit including EBMs (Explainable Boosting Machines), SHAP, LIME, and counterfactuals. | End-to-end explainability with glass-box models |
| [Captum](https://captum.ai/) | PyTorch-native attribution library (Integrated Gradients, SHAP, Occlusion, GradCAM). | Deep learning model attribution in PyTorch |
| [TransformerLens](https://github.com/neelnanda-io/transformerlens) | Mechanistic interpretability for GPT-style models. Activation patching, causal tracing. | LLM circuit analysis and knowledge localization |
| [BertViz](https://github.com/jessevig/bertviz) | Interactive attention visualization for BERT/GPT/T5. | Visualizing attention patterns in transformer papers |
| [DiCE](https://github.com/interpretml/dice) | Counterfactual explanation generation. | Actionable recourse explanations for IS decision systems |
| [Alibi](https://github.com/seldonio/alibi) | ML model inspection library with anchors, counterfactuals, and concept drift detection. | Production monitoring of deployed IS artifacts |

---

### Sparse Autoencoders (SAEs) for LLM Interpretability
A 2024 breakthrough in mechanistic interpretability: sparse autoencoders decompose superposed LLM representations into interpretable monosemantic features.

- **Papers with code:**
  - [Scaling and Evaluating Sparse Autoencoders](https://arxiv.org/abs/2406.04093), 2024 - OpenAI's systematic study of SAE scaling; identifies millions of interpretable features in GPT-4-scale models. [Code](https://github.com/openai/sparse_autoencoder)
  - [Improving Dictionary Learning with Gated Sparse Autoencoders](https://arxiv.org/abs/2404.16014), 2024 - Introduces gated SAEs that better separate feature detection from magnitude estimation. [Code](https://github.com/eleutherai/sae-utils)
  - [Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small](https://arxiv.org/abs/2211.00593), 2022 - Foundational circuit analysis work identifying specific attention heads responsible for a syntactic task. [Code](https://github.com/redwoodresearch/easy-transformer)

- **Tools:**
  - [SAELens](https://github.com/jbloomaus/saelens) - Toolkit for training, analyzing, and visualizing sparse autoencoders on any transformer model.
  - [Neuronpedia](https://www.neuronpedia.org/) - Interactive web interface for exploring SAE features in Claude and GPT-2 models. Enables rapid exploration of what individual neurons and features represent.

---

### Concept Bottleneck Models
A glass-box approach where the model first predicts human-interpretable concepts, then uses those concepts to make a final prediction — enabling human-in-the-loop oversight.

- **Papers with code:**
  - [Concept Bottleneck Models](https://arxiv.org/abs/2007.04612), 2020 - Trains models to predict predefined human concepts as intermediate representations. Final prediction depends only on concepts, providing full interpretability. [Code](https://github.com/yewsiang/conceptbottleneck)
  - [Label-free Concept Bottleneck Models](https://arxiv.org/abs/2304.06129), 2023 - Automatically discovers concept sets from a language model without requiring manual concept annotation. [Code](https://github.com/trustworthy-ml-lab/label-free-cbm)
  - [Post-hoc Concept Bottleneck Models](https://arxiv.org/abs/2205.15480), 2022 - Retrofits concept bottleneck explanations onto any pre-trained model without retraining. [Code](https://github.com/mertyg/post-hoc-cbm)

- **IS Research Application:** Concept bottleneck models are particularly suited for IS research where domain experts need to validate model reasoning (e.g., "this loan application was rejected because concept X = high-risk, concept Y = insufficient-history").

---

**Related Sections:** [LLMs & NLP](../NaturalLanguageProcessing/README.md) | [Graphs](../Graphs/README.md) | [LLM Safety & Adversarial Defense](../AdversarialDefense/README.md) | [Ethics](../Ethics/README.md)
