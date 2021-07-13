# Methodological Tools
A repository of tools found in top conferences to aid in method identification and application to IS research.

## Topic
- [Graphs](#Graphs)
- [Text Classification](#Text-Classification)
- [Data Generation](#Data-Generation)
- [Adversarial Defense](#Adversarial-Defense)

---

## Graphs
*Papers and Repositories For Construcing Novel Graphs.*
- [Graph Neural Networks](#Graph-Neural-Networks)
- [Graph Attention Networks](#Graph-Attention-Networks)
- [Graph Convolutional Networks](#Graph-Convolutional-Networks)
- [Graph Prototypical Networks](#Graph-Prototypical-Networks)
- [Graph Summarization](#Graph-Summarization)
- [Temporal Graphs](#Temporal-Graphs)
- [Graph Embeddings](#Graph-Embeddings)

---

### Graph Neural Networks
[Seminal Paper](https://repository.hkbu.edu.hk/cgi/viewcontent.cgi?article=1000&context=vprd_ja), 2007
- **Papers with code:**
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
[Seminal Paper](https://arxiv.org/pdf/1710.10903.pdf), 2018
- **Papers with code:**
  - [Graph Attention Networks over Edge Content-Based Channels](https://dl.acm.org/doi/10.1145/3394486.3403233), 2020 - Enhanced learning by utilizing the latent semantic information in edge content. [Code](https://github.com/Louise-LuLin/topic-gcn)  
  - [DETERRENT: Knowledge Guided Graph Attention Network for Detecting Healthcare Misinformation](http://pike.psu.edu/publications/kdd20-deterrent.pdf), 2020 - To detect misinformation in healthcare. [Code](https://github.com/cuilimeng/DETERRENT)
- **Papers without code:**
  - [DisenHAN: Disentangled Heterogeneous Graph Attention Network for Recommendation](https://dl.acm.org/doi/abs/10.1145/3340531.3411996), 2020 - To decompose high order connectivity between node pairs and identify major aspects of meta relations.
---

### Graph Convolutional Networks
[Seminal Paper](https://arxiv.org/pdf/1609.02907.pdf), 2017
- **Papers with code:**
  - [HGCN: A Heterogeneous Graph Convolutional Network-Based Deep Learning Model Toward Collective Classification](https://dl.acm.org/doi/10.1145/3394486.3403169), 2020 - Improving GCN with heterogeneity to improve classification performance. [Code](https://github.com/huazai1992/HGCN)
  - [Certifiable Robustness of Graph Convolutional Networks under Structure Perturbations](https://dl.acm.org/doi/10.1145/3394486.3403217), 2020 - To protect GCNs against adversarial attacks. [Code](https://www.in.tum.de/daml/robust-gcn/)
  - [Adaptive Graph Encoder for Attributed Graph Embedding](https://arxiv.org/pdf/2007.01594.pdf), 2020 - To create better graph node embeddings. [Code](https://github.com/thunlp/AGE) 
- **Papers without code:**
  - [Graph Structural-topic Neural Network](https://arxiv.org/pdf/2006.14278.pdf), 2020 - Improve the underlying structure of GCNs

---
 
### Graph Prototypical Networks
[Seminal Paper](https://arxiv.org/pdf/2006.12739.pdf), 2020 - [Code](https://github.com/kaize0409/GPN)

---

### Graph Summarization
[Seminal Paper](https://arxiv.org/pdf/1612.04883.pdf), 2018
- **Papers with code:**
  - [Incremental and Parallel Computation of Structural Graph Summaries for Evolving Graphs](https://dl.acm.org/doi/abs/10.1145/3340531.3411878), 2020 - Finding a condensed representation of a graph. [Code](https://github.com/t-blume/fluid-spark)
  - [Incremental Lossless Graph Summarization](https://arxiv.org/pdf/2006.09935.pdf), 2020 - To make large graphs scalable, faster processing, and effective. [Code](http://dmlab.kaist.ac.kr/mosso/)

---

### Temporal Graphs
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
[Seminal Paper](http://proceedings.mlr.press/v48/yanga16.pdf), 2016
- **Papers with code:**
  - [Towards Locality-Aware Meta-Learning of Tail Node Embeddings on Networks](https://dl.acm.org/doi/10.1145/3340531.3411910), 2020 - Create tail node embedding for node classification and link prediction. [Code](https://github.com/smufang/meta-tail2vec)
  - [Towards Temporal Knowledge Graph Embeddings with Arbitrary Time Precision](https://dl.acm.org/doi/abs/10.1145/3340531.3412028), 2020 - For improved time-dependent queries. [Code](https://gitlab.com/jleblay/tokei)
- **Papers without code:**
  - [An Adaptive Embedding Framework for Heterogeneous Information Networks](https://dl.acm.org/doi/10.1145/3340531.3411989), 2020 - For improved node classification and link prediction tasks.

---

## Text Classification
*Papers and Repositories For Multi-Label and Multi-Class Classification Problems With Text.*
- [Transformers](#Transformers)
- [Topic Modeling](#Topic-Modeling)
- [Word Embeddings](#Word-Embeddings)
- [Knowledge Distillation](#Knowledge-Distillation) 

---

### Transformers
[Seminal Paper](https://arxiv.org/pdf/1706.03762.pdf), 2017

- **Papers with code:**
  - [Encoding word order in complex embeddings](https://arxiv.org/pdf/1912.12333.pdf), 2020 - To create a custom embedding that models global absolute positions of words and their order relationships. [Code](https://github.com/iclr-complex-order/complex-order)
  - [Tree-Structured Attention with Hierarchical Accumulation](https://arxiv.org/pdf/2002.08046.pdf), 2020 - Hierarchical Accumulation to encode parse tree structures into self-attention at a constant time complexity. [Code](https://github.com/nxphi47/tree_transformer)
  - [Lite Transformer with Long-Short Range Attention](https://arxiv.org/pdf/2004.11886.pdf), 2020 - Models local context by convolution and long-distance relationship by attention. [Code](https://github.com/mit-han-lab/lite-transformer)
  - [Transformer-XH: Multi-Evidence Reasoning with eXtra Hop Attention](https://openreview.net/pdf/d046083250740c4f9687e47d1df323759b66b5e4.pdf), 2020 - Enables intrinsic modeling of structured text by "hopping" around the document. [Code](https://github.com/microsoft/Transformer-XH)
  - [Monotonic Multihead Attention](https://arxiv.org/pdf/1909.12406.pdf), 2020 - Adds a multihead monotonic attention mechanism for latency control. [Code](https://github.com/pytorch/fairseq/tree/master/examples/simultaneous_translation)
  - [Reformer: The Efficient Transformer](https://arxiv.org/pdf/2001.04451.pdf), 2020 - Memory-efficiency and faster on long-sequences. [Code](https://github.com/google/trax/tree/master/trax/models/reformer)

- **Papers without code:**
  - [On Identifiability in Transformers](https://openreview.net/pdf?id=BJg1f6EFDB), 2020 - Introduces effective attention to improve explanatory interpretations based on attention.
  - [Are Transformers universal approximators of sequence-to-sequence functions?](https://arxiv.org/pdf/1912.10077.pdf), 2020 - Proof that fixed width self-attention layers can compute contextual mappings of the input sequences.
  - [Robustness Verification for Transformers](https://arxiv.org/pdf/2002.06622.pdf), 2020 - A robustness verification algorithm for TRansformers that shed light on the Transformer's interpretative capabilities.

---
### Topic Modeling
[Seminal Paper: LDA](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf?TB_iframe=true&width=370.8&height=658.8), 2003

- **Papers with code:**
  - To Be Continued

- **Papers without code:**
  - [Hierarchical Topic Modeling of Twitter Data for Online Analytical Processing](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8607040), 2018 - automatically mine the hierarchical dimension of tweets’ topics, which can be further employed for text OLAP on the tweets. Furthermore, thLDA uses word2vec to analyze the semantic relationships of words in tweets to obtain a more effective dimension.
---
### Word Embeddings
- **Seminal Pre-Trained Word Embeddings**
  - [Word2Vec](https://arxiv.org/pdf/1301.3781.pdf), 2013. [Code](https://github.com/tmikolov/word2vec)
  - [GloVe](https://www.aclweb.org/anthology/D14-1162.pdf), 2014. [Code](https://github.com/stanfordnlp/GloVe)

- **Papers with code:**
  - [Learning from Textual Data in Database Systems](https://dl.acm.org/doi/abs/10.1145/3340531.3412056), 2020 - An optimized algorithm for creating enhanced word embeddings. [Code](https://github.com/guenthermi/postgres-retrofit)
  
---
### Knowledge Distillation
- **Papers with code:**
  - [Big Self-Supervised Models are Strong Semi-Supervised Learners](https://arxiv.org/pdf/2006.10029.pdf), 2020 - The proposed semi-supervised learning algorithm can be summarized in three steps: unsupervised pretraining of a big ResNet model using SimCLRv2, supervised fine-tuning on a few labeled examples, and distillation with unlabeled examples for refining and transferring the task-specific knowledge.  [Code](https://github.com/google-research/simclr)
---
## Data Generation
*Papers and Repositories For Image, Textual, and Other Generation Tasks.*

[Seminal Paper](https://arxiv.org/pdf/1406.2661.pdf), 2014

- [Image Generation](#Image-Generation)
- [Text Generation](#Data-Generation)
- [Other Generation](#Other-Generation)

---

### Image Generation
- **Papers with code:**
  - [Training Generative Adversarial Networks from Incomplete Observations using Factorised Discriminators](https://arxiv.org/pdf/1905.12660.pdf), 2020 - Splits the discriminator into parts that can be independently trained with incomplete observations. [Code](https://www.dropbox.com/s/gtc7m7pc4n2yt05/source.zip?dl=1)
  - [On the "steerability" of generative adversarial networks](hhttps://arxiv.org/pdf/1907.07171.pdf), 2020 - Shifts the distribution of GANs to improve generalizability. [Code](https://ali-design.github.io/gan_steerability/)
  - [Controlling generative models with continuous factors of variations](https://arxiv.org/pdf/2001.10238.pdf), 2020 - Control the position of an object in an image. [Code](https://github.com/AntoinePlumerault/Controlling-generative-models-with-continuous-factors-of-variations)
  
- **Papers without code:**
   - [Stochastic Conditional Generative Networks with Basis Decomposition](https://arxiv.org/pdf/1909.11286.pdf), 2019 - Generate improved images for multi-mode datasets.
---
   
### Text Generation
- **Papers with code:**
  - [Neural Text Generation With Unlikelihood Training](https://arxiv.org/pdf/1908.04319.pdf), 2020 - On transformers, forces unlikely generations to be assigned lower probability by the model. [Code](https://github.com/facebookresearch/unlikelihood_training)
  - [Language GANs Falling Short](https://arxiv.org/pdf/1811.02549.pdf), 2020 - A unique quality-diversity evaluation procuedure to reduce softmax temperature. [Code](https://github.com/pclucas14/GansFallingShort)
  - [The Curious Case of Neural Text Degeneration](https://arxiv.org/pdf/1904.09751.pdf), 2020 - Nucleus sampling in a transformer can draw high quality text from neural language models. [Code](https://github.com/ari-holtzman/degen)
  - [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/pdf/1904.09675.pdf), 2020 - An evaluative metric for text generation using similarity scores. [Code](https://github.com/Tiiiger/bert_score)
  - [Plug and Play Language Models: A Simple Approach to Controlled Text Generation](https://arxiv.org/pdf/1912.02164.pdf), 2020 - Combines a pre-trained transformer language model with a attribute classifiers to guide text generation. [Code](https://github.com/uber-research/PPLM)

- **Papers without code:**
   - [Augmenting Non-Collaborative Dialog Systems with Explicit Semantic and Strategic Dialog History](https://arxiv.org/pdf/1909.13425.pdf), 2020 - Using finite state tranducers to explicity represent dialog history.
   - [Self-Adversarial Learning with Comparative Discrimination for Text Generation](https://arxiv.org/pdf/2001.11691.pdf), 2020 - Self-improvement reward mechanism allows the model to avoid collapse.

---

### Other Generation
- **Papers with code:**
  - [A Closer Look at the Optimization Landscapes of Generative Adversarial Networks](https://arxiv.org/pdf/1906.04848.pdf), 2020 - GANs exhibits significant rotations around Local Stable Stationary Points (LSSP), which can be used as a saddle point for the generator loss. [Code](https://github.com/facebookresearch/GAN-optimization-landscape)
  - [Curb-GAN: Conditional Urban Traffic Estimation through Spatio-Temporal Generative Adversarial Networks](https://dl.acm.org/doi/10.1145/3394486.3403127), 2020 - Traffic estimations in consecutive time slots based on different (unprecedented) travel demands. [Code](https://github.com/Curb-GAN/Curb-GAN)
  - [SEAL: Learning Heuristics for Community Detection with Generative Adversarial Networks](https://dl.acm.org/doi/abs/10.1145/3394486.3403154), 2020 - Creates and predicts whether a community is real or fake for enhanced detection. [Code](https://github.com/yzhang1918/kdd2020seal)

- **Papers without code:**
  - [Smoothness and Stability in GANs](https://arxiv.org/pdf/2002.04185.pdf), 2020 - Hyperparameter tuning to improve the smoothness and stability of GANs.
  - [Deep State-Space Generative Model For Correlated Time-to-Event Predictions](https://dl.acm.org/doi/pdf/10.1145/3394486.3403206), 2020 - A new general discrete-time formulation of the hazard rate function to estimate the survival distribution of patients with significantly improved accuracy.
  - [Catalysis Clustering With GAN By Incorporating Domain Knowledge](https://dl.acm.org/doi/10.1145/3394486.3403187), 2020 - Creates better unsupervised clusters based on domain-defined rules and guidelines.

---
 
## Adversarial Defense
*Papers and Repositories For Defense Against Machine Learning Adversarial Attacks.*

- **Papers with code:**
  - [AdvMind: Inferring Adversary Intent of Black-Box Attacks](https://arxiv.org/pdf/2006.09539.pdf), 2020 - Infer the adversary intent of black-box adversarial attacks. [Code](https://github.com/ain-soph/trojanzoo)
  - [An Embarrassingly Simple Approach for Trojan Attack in Deep Neural Networks](https://arxiv.org/pdf/2006.08131.pdf), 2020 - To show how easy it is to damage DNN systems. [Code](https://github.com/trx14/TrojanNet)

- **Papers without code:**
  - [Interpretability is a Kind of Safety: An Interpreter-based Ensemble for Adversary Defense](https://dl.acm.org/doi/abs/10.1145/3394486.3403044), 2020 - Interpreter-based ensemble framework for the detection and defense of adversarial attacks to a model.

---

