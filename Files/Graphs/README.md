# Graphs
*Papers and Repositories For Construcing Novel Graphs.*

| | | |
|-|-|-|
| [Graph Neural Networks](#Graph-Neural-Networks) | [Graph Attention Networks](#Graph-Attention-Networks) | [Graph Convolutional Networks](#Graph-Convolutional-Networks)
| [Graph Prototypical Networks](#Graph-Prototypical-Networks) | [Graph Summarization](#Graph-Summarization) | [Temporal Graphs](#Temporal-Graphs)
| [Graph Embeddings](#Graph-Embeddings) | [Graphs for NLP](https://github.com/graph4ai/graph4nlp_literature)

---

### Graph Neural Networks
Graph Neural Networks (GNNs) maps graph-based data into a Euclidean space using a supervised learning algorithm. 
[Seminal Paper](https://repository.hkbu.edu.hk/cgi/viewcontent.cgi?article=1000&context=vprd_ja), 2007

- **Papers with code:**
  - [Interpreting Graph Neural Networks for NLP With Differentiable Edge Masking](https://openreview.net/pdf?id=WznmQa42ZAx), 2021 - Interpreting the predictions of GNNs through dropping edges. [Code](https://github.com/MichSchli/GraphMask)
  - [Distance-wise Prototypical Graph Neural Network for Imbalanced Node Classification](https://arxiv.org/pdf/2110.12035v1.pdf), 2021 - Improving node classification through distance-wise tasks. [Code](https://github.com/yuwvandy/dpgnn)
  - [Graph Transformer Networks: Learning Meta-path Graphs to Improve GNNs](https://arxiv.org/pdf/2106.06218.pdf), 2021 - Capable of generating new graph structures, which preclude noisy connections and include useful connections (e.g., meta-paths) for tasks, while learning effective node representations on the new graphs in an end-to-end fashion. [Code](https://github.com/seongjunyun/Graph_Transformer_Networks)
  - [Graph Structure Learning for Robust Graph Neural Networks](https://arxiv.org/pdf/2005.10203.pdf), 2020 - Protect GNNs against adversarial attacks. [Code](https://github.com/ChandlerBang/Pro-GNN) 
  - [GPT-GNN: Generative Pre-Training of Graph Neural Networks](https://arxiv.org/pdf/2006.15437.pdf), 2020 - Using GPT to fine-tune a GNN improves many downstream tasks. [Code](https://github.com/acbull/GPT-GNN) 
  - [GCC: Graph Contrastive Coding for Graph Neural Network Pre-Training](https://arxiv.org/pdf/2006.09963.pdf), 2020 - Pre-training a GNN on ten graph datasets improves many downstream tasks. [Code](https://github.com/THUDM/GCC)
  - [Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks](https://arxiv.org/pdf/2005.11650.pdf), 2020 - Framework designed for multi-variate time series data. [Code](https://github.com/THUDM/GCC)
  - [PolicyGNN: Aggregation Optimization for Graph Neural Networks](https://arxiv.org/pdf/2006.15097.pdf), 2020 - To adaptively determine the number of aggregations for each node with deep RL. [Code](https://github.com/nnzhan/MTGNN)
  - [GNNVis: Visualize Large-Scale Data by Learning a Graph Neural Network Representation](https://dl.acm.org/doi/abs/10.1145/3340531.3411987), 2020 - To compute efficient network embeddings with unseen big data. [Code](https://github.com/YajunHuang/gnnvis) 
  - [Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters](https://arxiv.org/pdf/2008.08692.pdf), 2020 - To improve fraud detection systems. [Code](https://github.com/YingtongDou/CARE-GNN)
  - [Cola-GNN: Cross-location Attention based Graph Neural Networks for Long-term ILI Prediction](https://yue-ning.github.io/docs/CIKM20-colagnn.pdf), 2020 - To forecast influenza like illness. [Code](https://github.com/amy-deng/colagnn)
  - [Graph Unfolding Networks](https://dl.acm.org/doi/abs/10.1145/3340531.3412141), 2020 - Uses parallel computation, which is more efficient than the recursive neighborhood aggregation process. [Code](https://github.com/GUNets/GUNets)

- **Papers without code:**
  - [XGNN: Towards Model-Level Explanations of Graph Neural Networks](https://arxiv.org/pdf/2006.02587.pdf), 2020 - To create model-level interpretability in GNNs.
  - [TinyGNN: Learning Efficient Graph Neural Networks](https://dl.acm.org/doi/10.1145/3394486.3403236), 2020 - To speed up graph training without losing important information.
  - [Adaptive-Step Graph Meta-Learner for Few-Shot Graph Classification](https://arxiv.org/pdf/2003.08246.pdf), 2020 - Extract accurate information from graph-structured data for classification.
  - [Graph Few-shot Learning with Attribute Matching](https://dl.acm.org/doi/10.1145/3340531.3411923), 2020 - To leverage attribute-level attention mechanisms to capture distinct information of each task.
  - [Streaming Graph Neural Networks via Continual Learning](https://arxiv.org/pdf/2009.10951.pdf), 2020 - To update model parameters automatically with constant data flow.

---

### Graph Attention Networks
Leveraging GNNs, Graph Attention Networks (GATs) add self-attention layers to effectively capture neighborhood features.
[Seminal Paper](https://arxiv.org/pdf/1710.10903.pdf), 2018
- **Papers with code:**
  - [Graph Attention Networks over Edge Content-Based Channels](https://dl.acm.org/doi/10.1145/3394486.3403233), 2020 - Enhanced learning by utilizing the latent semantic information in edge content. [Code](https://github.com/Louise-LuLin/topic-gcn)  
  - [DETERRENT: Knowledge Guided Graph Attention Network for Detecting Healthcare Misinformation](http://pike.psu.edu/publications/kdd20-deterrent.pdf), 2020 - To detect misinformation in healthcare. [Code](https://github.com/cuilimeng/DETERRENT)
- **Papers without code:**
  - [DisenHAN: Disentangled Heterogeneous Graph Attention Network for Recommendation](https://dl.acm.org/doi/abs/10.1145/3340531.3411996), 2020 - To decompose high order connectivity between node pairs and identify major aspects of meta relations.
---

### Graph Convolutional Networks
Leveraging GNNs, Graph Convolutional Networks (GCNs) add convolutional layers to learn layer representations through semi-supervised learning.
[Seminal Paper](https://arxiv.org/pdf/1609.02907.pdf), 2017

- **Papers with code:**
  - [GIST: Distributed Training for Large-Scale Graph Convolutional Networks](https://arxiv.org/pdf/2102.10424.pdf), 2021 - Our proposed training methodology, called GIST, disjointly partitions the parameters of a GCN model into several, smaller sub-GCNs that are trained independently and in parallel. [Code](https://github.com/wolfecameron/GIST) 
  - [HGCN: A Heterogeneous Graph Convolutional Network-Based Deep Learning Model Toward Collective Classification](https://dl.acm.org/doi/10.1145/3394486.3403169), 2020 - Improving GCN with heterogeneity to improve classification performance. [Code](https://github.com/huazai1992/HGCN)
  - [Certifiable Robustness of Graph Convolutional Networks under Structure Perturbations](https://dl.acm.org/doi/10.1145/3394486.3403217), 2020 - To protect GCNs against adversarial attacks. [Code](https://www.in.tum.de/daml/robust-gcn/)
  - [Adaptive Graph Encoder for Attributed Graph Embedding](https://arxiv.org/pdf/2007.01594.pdf), 2020 - To create better graph node embeddings. [Code](https://github.com/thunlp/AGE)

- **Papers without code:**
  - [Automated Graph Learning via Population Based Self-Tuning GCN](https://arxiv.org/pdf/2009.10951.pdf), 2021 - A self-tuning GCN approach with an alternate training algorithm to optimize hyperparameter tuning
  - [Graph Structural-topic Neural Network](https://arxiv.org/pdf/2006.14278.pdf), 2020 - Improve the underlying structure of GCNs

---

### Graph Prototypical Networks
Adds a meta-learning component to GNNs for effective few-shot learning.
[Seminal Paper](https://arxiv.org/pdf/2006.12739.pdf), 2020 - [Code](https://github.com/kaize0409/GPN)

---

### Graph Summarization
Creating a summarization of a graph to reduce data, speed graph queries, and eliminate noise.
[Seminal Paper](https://arxiv.org/pdf/1612.04883.pdf), 2018
- **Papers with code:**
  - [Incremental and Parallel Computation of Structural Graph Summaries for Evolving Graphs](https://dl.acm.org/doi/abs/10.1145/3340531.3411878), 2020 - Finding a condensed representation of a graph. [Code](https://github.com/t-blume/fluid-spark)
  - [Incremental Lossless Graph Summarization](https://arxiv.org/pdf/2006.09935.pdf), 2020 - To make large graphs scalable, faster processing, and effective. [Code](http://dmlab.kaist.ac.kr/mosso/)

---

### Temporal Graphs
Graphs that change over time.
[Seminal Paper](https://www.tandfonline.com/doi/abs/10.1080/15427951.2016.1177801), 2016
- **Papers with code:**
  - [tdGraphEmbed: Temporal Dynamic Graph-Level Embedding](https://dl.acm.org/doi/abs/10.1145/3340531.3411953), 2020 - To extend random-walk based node embedding methods to improve embeddings. [Code](https://github.com/moranbel/tdGraphEmbed)
  - [Local Motif Clustering on Time-Evolving Graphs](https://dl.acm.org/doi/abs/10.1145/3394486.3403081), 2020 - To track the temporal evolution of the local motif cluster using edge filtering. [Code](https://github.com/DongqiFu/L-MEGA)
  - [Inductive representation learning on temporal graphs](https://arxiv.org/pdf/2002.07962.pdf), 2020 - Using a temporal graph attention layer to efficiently aggreagte temproal-topological neighborhod features. [Code](https://drive.google.com/drive/folders/1GaH8vusCXJj4ucayfO-PyHpnNsJRkB78A)

- **Papers without code:**
  - [Continuous-Time Dynamic Graph Learning via Neural Interaction Processes](https://dl.acm.org/doi/abs/10.1145/3340531.3411946), 2020 - To capture the fine-grained global and local information for temporal interaction prediction.
  - [Algorithmic Aspects of Temporal Betweenness](https://arxiv.org/pdf/2006.08668.pdf), 2020 - A systematic study of temporal betweeness variants.

---

### Graph Embeddings
Custom node embeddings in a semi-supervised fashion can improve model performance.
[Seminal Paper](http://proceedings.mlr.press/v48/yanga16.pdf), 2016
- **Papers with code:**
  - [Towards Locality-Aware Meta-Learning of Tail Node Embeddings on Networks](https://dl.acm.org/doi/10.1145/3340531.3411910), 2020 - Create tail node embedding for node classification and link prediction. [Code](https://github.com/smufang/meta-tail2vec)
  - [Towards Temporal Knowledge Graph Embeddings with Arbitrary Time Precision](https://dl.acm.org/doi/abs/10.1145/3340531.3412028), 2020 - For improved time-dependent queries. [Code](https://gitlab.com/jleblay/tokei)
- **Papers without code:**
  - [An Adaptive Embedding Framework for Heterogeneous Information Networks](https://dl.acm.org/doi/10.1145/3340531.3411989), 2020 - For improved node classification and link prediction tasks.

---
