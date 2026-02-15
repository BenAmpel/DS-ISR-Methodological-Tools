# Graphs
*Papers and Repositories For Constructing Novel Graphs.*

*Last updated: February 2026*

| | | |
|-|-|-|
| [Graph Neural Networks](#Graph-Neural-Networks) | [Graph Attention Networks](#Graph-Attention-Networks) | [Graph Convolutional Networks](#Graph-Convolutional-Networks)
| [GraphRAG & LLM Integration](#GraphRAG--LLM-Integration) | [Graph Prototypical Networks](#Graph-Prototypical-Networks) | [Graph Summarization](#Graph-Summarization)
| [Temporal Graphs](#Temporal-Graphs) | [Graph Embeddings](#Graph-Embeddings) | [Graphs for NLP](https://github.com/graph4ai/graph4nlp_literature)

---

> **IS Research Applications:** Model organizational social networks and information diffusion; detect fraud in transaction networks; represent knowledge bases and ontologies for enterprise systems; analyze citation networks for literature review; model supply chain relationships; build knowledge graphs for LLM-powered IS artifacts (GraphRAG).

---

### Graph Neural Networks
Graph Neural Networks (GNNs) map graph-based data into a Euclidean space using a supervised learning algorithm.
[Seminal Paper](https://repository.hkbu.edu.hk/cgi/viewcontent.cgi?article=1000&context=vprd_ja), 2007

- **Papers with code:**
  - [Interpreting Graph Neural Networks for NLP With Differentiable Edge Masking](https://openreview.net/pdf?id=WznmQa42ZAx), 2021 - Interpreting GNN predictions through dropping edges. [Code](https://github.com/MichSchli/GraphMask)
  - [Distance-wise Prototypical Graph Neural Network for Imbalanced Node Classification](https://arxiv.org/pdf/2110.12035v1.pdf), 2021 - Improving node classification through distance-wise tasks. [Code](https://github.com/yuwvandy/dpgnn)
  - [Graph Transformer Networks: Learning Meta-path Graphs to Improve GNNs](https://arxiv.org/pdf/2106.06218.pdf), 2021 - Generates new graph structures, precluding noisy connections and including useful meta-paths in an end-to-end fashion. [Code](https://github.com/seongjunyun/Graph_Transformer_Networks)
  - [Graph Structure Learning for Robust Graph Neural Networks](https://arxiv.org/pdf/2005.10203.pdf), 2020 - Protects GNNs against adversarial attacks. [Code](https://github.com/ChandlerBang/Pro-GNN)
  - [GPT-GNN: Generative Pre-Training of Graph Neural Networks](https://arxiv.org/pdf/2006.15437.pdf), 2020 - Using GPT to fine-tune a GNN improves many downstream tasks. [Code](https://github.com/acbull/GPT-GNN)
  - [GCC: Graph Contrastive Coding for Graph Neural Network Pre-Training](https://arxiv.org/pdf/2006.09963.pdf), 2020 - Pre-training a GNN on ten graph datasets improves many downstream tasks. [Code](https://github.com/THUDM/GCC)
  - [Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks](https://arxiv.org/pdf/2005.11650.pdf), 2020 - Framework for multi-variate time series data. [Code](https://github.com/THUDM/GCC)
  - [PolicyGNN: Aggregation Optimization for Graph Neural Networks](https://arxiv.org/pdf/2006.15097.pdf), 2020 - Adaptively determines the number of aggregations for each node with deep RL. [Code](https://github.com/nnzhan/MTGNN)
  - [GNNVis: Visualize Large-Scale Data by Learning a Graph Neural Network Representation](https://dl.acm.org/doi/abs/10.1145/3340531.3411987), 2020 - Computes efficient network embeddings with unseen big data. [Code](https://github.com/YajunHuang/gnnvis)
  - [Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters](https://arxiv.org/pdf/2008.08692.pdf), 2020 - Improves fraud detection systems against adversarial camouflage. [Code](https://github.com/YingtongDou/CARE-GNN)
  - [Cola-GNN: Cross-location Attention based Graph Neural Networks for Long-term ILI Prediction](https://yue-ning.github.io/docs/CIKM20-colagnn.pdf), 2020 - Forecasts influenza-like illness. [Code](https://github.com/amy-deng/colagnn)
  - [Graph Unfolding Networks](https://dl.acm.org/doi/abs/10.1145/3340531.3412141), 2020 - Uses parallel computation more efficiently than recursive neighborhood aggregation. [Code](https://github.com/GUNets/GUNets)

- **Papers without code:**
  - [Graph Neural Networks: Architectures, Applications, and Future Directions](https://ieeexplore.ieee.org/document/10643095), IEEE 2025 - Comprehensive survey of GNN architectures and emerging applications including trustworthiness and reliability.
  - [XGNN: Towards Model-Level Explanations of Graph Neural Networks](https://arxiv.org/pdf/2006.02587.pdf), 2020 - Creates model-level interpretability in GNNs.
  - [TinyGNN: Learning Efficient Graph Neural Networks](https://dl.acm.org/doi/10.1145/3394486.3403236), 2020 - Speeds up graph training without losing important information.
  - [Adaptive-Step Graph Meta-Learner for Few-Shot Graph Classification](https://arxiv.org/pdf/2003.08246.pdf), 2020 - Extracts accurate information from graph-structured data for classification.
  - [Graph Few-shot Learning with Attribute Matching](https://dl.acm.org/doi/10.1145/3340531.3411923), 2020 - Leverages attribute-level attention to capture distinct task information.
  - [Streaming Graph Neural Networks via Continual Learning](https://arxiv.org/pdf/2009.10951.pdf), 2020 - Updates model parameters automatically with constant data flow.

---

### GraphRAG & LLM Integration
The convergence of LLMs and graph methods is an active research frontier. LLMs enhance graph representations with semantic understanding, while graphs provide LLMs with structured relational context — particularly valuable for knowledge-intensive IS research artifacts.

- **GraphRAG (Graph Retrieval-Augmented Generation):**
  - [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130), 2024 - Microsoft's GraphRAG builds a knowledge graph from a corpus and uses it to answer complex, multi-hop questions requiring synthesis across many documents. Outperforms vector-only RAG on queries requiring global understanding. [Code](https://github.com/microsoft/graphrag)
  - [G-RAG: Knowledge Expansion in Material Science](https://arxiv.org/abs/2408.04558), 2024 - Domain-specific GraphRAG pipeline for scientific literature. Demonstrates application to structured knowledge domains relevant for IS research.

- **LLM-Augmented GNNs:**
  - [One for All: Towards Training One Graph Model for All Classification Tasks](https://arxiv.org/abs/2310.00149), 2023 - Uses LLMs to create unified node text embeddings, enabling a single GNN to generalize across diverse graphs without task-specific fine-tuning. [Code](https://github.com/LechengKong/OneForAll)
  - [Explanations as Features: LLM-Based Features for Text-Attributed Graphs](https://arxiv.org/abs/2305.19523), 2023 - Generates LLM-based textual explanations as node features, improving GNN performance and interpretability on text-attributed graphs. [Code](https://github.com/XiaoxinHe/TAPE)
  - [Trustworthy Graph Neural Networks: Aspects, Methods, and Trends](https://arxiv.org/abs/2205.07424), 2022 - Survey of GNN trustworthiness including robustness, explainability, fairness, and privacy — critical concerns for IS research deployments. Includes emerging results from 2025.

- **Papers without code:**
  - [Graph Neural Networks in Brain Connectivity Analysis: A Survey](https://arxiv.org/abs/2409.04833), 2025 - Reviews GNN applications in neuroscience connectivity data; methodology transferable to any relational network structure in IS (e.g., organizational networks, social networks).

---

### Graph Attention Networks
Leveraging GNNs, Graph Attention Networks (GATs) add self-attention layers to effectively capture neighborhood features.
[Seminal Paper](https://arxiv.org/pdf/1710.10903.pdf), 2018

- **Papers with code:**
  - [Graph Attention Networks over Edge Content-Based Channels](https://dl.acm.org/doi/10.1145/3394486.3403233), 2020 - Enhanced learning by utilizing latent semantic information in edge content. [Code](https://github.com/Louise-LuLin/topic-gcn)
  - [DETERRENT: Knowledge Guided Graph Attention Network for Detecting Healthcare Misinformation](http://pike.psu.edu/publications/kdd20-deterrent.pdf), 2020 - Detects misinformation in healthcare. [Code](https://github.com/cuilimeng/DETERRENT)

- **Papers without code:**
  - [DisenHAN: Disentangled Heterogeneous Graph Attention Network for Recommendation](https://dl.acm.org/doi/abs/10.1145/3340531.3411996), 2020 - Decomposes high-order connectivity between node pairs and identifies major aspects of meta relations.

---

### Graph Convolutional Networks
Leveraging GNNs, Graph Convolutional Networks (GCNs) add convolutional layers to learn layer representations through semi-supervised learning.
[Seminal Paper](https://arxiv.org/pdf/1609.02907.pdf), 2017

- **Papers with code:**
  - [GIST: Distributed Training for Large-Scale Graph Convolutional Networks](https://arxiv.org/pdf/2102.10424.pdf), 2021 - Disjointly partitions GCN parameters into smaller sub-GCNs trained independently and in parallel. [Code](https://github.com/wolfecameron/GIST)
  - [HGCN: A Heterogeneous Graph Convolutional Network-Based Deep Learning Model Toward Collective Classification](https://dl.acm.org/doi/10.1145/3394486.3403169), 2020 - Improves GCN with heterogeneity for enhanced classification performance. [Code](https://github.com/huazai1992/HGCN)
  - [Certifiable Robustness of Graph Convolutional Networks under Structure Perturbations](https://dl.acm.org/doi/10.1145/3394486.3403217), 2020 - Protects GCNs against adversarial attacks. [Code](https://www.in.tum.de/daml/robust-gcn/)
  - [Adaptive Graph Encoder for Attributed Graph Embedding](https://arxiv.org/pdf/2007.01594.pdf), 2020 - Creates better graph node embeddings. [Code](https://github.com/thunlp/AGE)

- **Papers without code:**
  - [Automated Graph Learning via Population Based Self-Tuning GCN](https://arxiv.org/pdf/2009.10951.pdf), 2021 - Self-tuning GCN with alternate training algorithm for hyperparameter optimization.
  - [Graph Structural-topic Neural Network](https://arxiv.org/pdf/2006.14278.pdf), 2020 - Improves the underlying structure of GCNs.

---

### Graph Prototypical Networks
Adds a meta-learning component to GNNs for effective few-shot learning.
[Seminal Paper](https://arxiv.org/pdf/2006.12739.pdf), 2020 - [Code](https://github.com/kaize0409/GPN)

---

### Graph Summarization
Creating a summarization of a graph to reduce data, speed graph queries, and eliminate noise.
[Seminal Paper](https://arxiv.org/pdf/1612.04883.pdf), 2018

- **Papers with code:**
  - [Incremental and Parallel Computation of Structural Graph Summaries for Evolving Graphs](https://dl.acm.org/doi/abs/10.1145/3340531.3411878), 2020 - Finds condensed representations of evolving graphs. [Code](https://github.com/t-blume/fluid-spark)
  - [Incremental Lossless Graph Summarization](https://arxiv.org/pdf/2006.09935.pdf), 2020 - Makes large graphs scalable and enables faster, effective processing. [Code](http://dmlab.kaist.ac.kr/mosso/)

---

### Temporal Graphs
Graphs that change over time.
[Seminal Paper](https://www.tandfonline.com/doi/abs/10.1080/15427951.2016.1177801), 2016

- **Papers with code:**
  - [tdGraphEmbed: Temporal Dynamic Graph-Level Embedding](https://dl.acm.org/doi/abs/10.1145/3340531.3411953), 2020 - Extends random-walk based node embedding methods to improve embeddings over time. [Code](https://github.com/moranbel/tdGraphEmbed)
  - [Local Motif Clustering on Time-Evolving Graphs](https://dl.acm.org/doi/abs/10.1145/3394486.3403081), 2020 - Tracks temporal evolution of local motif clusters using edge filtering. [Code](https://github.com/DongqiFu/L-MEGA)
  - [Inductive representation learning on temporal graphs](https://arxiv.org/pdf/2002.07962.pdf), 2020 - Uses a temporal graph attention layer to efficiently aggregate temporal-topological neighborhood features. [Code](https://drive.google.com/drive/folders/1GaH8vusCXJj4ucayfO-PyHpnNsJRkB78A)

- **Papers without code:**
  - [Continuous-Time Dynamic Graph Learning via Neural Interaction Processes](https://dl.acm.org/doi/abs/10.1145/3340531.3411946), 2020 - Captures fine-grained global and local information for temporal interaction prediction.
  - [Algorithmic Aspects of Temporal Betweenness](https://arxiv.org/pdf/2006.08668.pdf), 2020 - Systematic study of temporal betweenness variants.

---

### Graph Embeddings
Custom node embeddings in a semi-supervised fashion can improve model performance.
[Seminal Paper](http://proceedings.mlr.press/v48/yanga16.pdf), 2016

- **Papers with code:**
  - [Towards Locality-Aware Meta-Learning of Tail Node Embeddings on Networks](https://dl.acm.org/doi/10.1145/3340531.3411910), 2020 - Creates tail node embeddings for node classification and link prediction. [Code](https://github.com/smufang/meta-tail2vec)
  - [Towards Temporal Knowledge Graph Embeddings with Arbitrary Time Precision](https://dl.acm.org/doi/abs/10.1145/3340531.3412028), 2020 - Enables improved time-dependent queries. [Code](https://gitlab.com/jleblay/tokei)

- **Papers without code:**
  - [An Adaptive Embedding Framework for Heterogeneous Information Networks](https://dl.acm.org/doi/10.1145/3340531.3411989), 2020 - Improves node classification and link prediction tasks.

---

**Related Sections:** [LLMs & NLP](../NaturalLanguageProcessing/README.md) | [Anomaly Detection](../AnomalyDetection/README.md) | [LLM Safety & Adversarial Defense](../AdversarialDefense/README.md) | [Interpretability](../Interpretability/README.md)
