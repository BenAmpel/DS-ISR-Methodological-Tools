# Natural Language Processing
*Papers and Repositories For Topic Modeling and Multi-Label and Multi-Class Classification Problems With Text.*

| | | |
|-|-|-|
| [Transformers](#Transformers) | [Topic Modeling](#Topic-Modeling) | [Word Embeddings](#Word-Embeddings)
| [Knowledge Distillation](#Knowledge-Distillation) | [Custom Layers](#Custom-Layers) 

---

### Transformers
[Seminal Paper](https://arxiv.org/pdf/1706.03762.pdf), 2017

An introduction to transformers can be found [here](https://github.com/will-thompson-k/tldr-transformers).

- **Papers with code:**
  - [Fastformer: Additive Attention Can Be All You Need](https://arxiv.org/pdf/2108.09084.pdf), 2021 - efficient Transformer variant based on additive attention that can achieve effective context modeling in linear complexity. [Code](https://github.com/wilile26811249/Fastformer-PyTorch)
  - [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/pdf/2106.01345v2.pdf), 2021 - Decision Transformer simply outputs the optimal actions by leveraging a causally masked Transformer. By conditioning an autoregressive model on the desired return (reward), past states, and actions, our Decision Transformer model can generate future actions that achieve the desired return. [Code](https://github.com/kzl/decision-transformer)
  - [CoBERL: Contrastive BERT for Reinforcement Learning](https://arxiv.org/pdf/2107.05431v1.pdf), 2021 - An agent that combines a new contrastive loss and a hybrid LSTM-transformer architecture to tackle the challenge of improving data efficiency. COBERL enables efficient, robust learning from pixels across a wide range of domains. [Code](https://github.com/deepmind/dm_control)
  - [Encoding word order in complex embeddings](https://arxiv.org/pdf/1912.12333.pdf), 2020 - To create a custom embedding that models global absolute positions of words and their order relationships. [Code](https://github.com/iclr-complex-order/complex-order)
  - [Tree-Structured Attention with Hierarchical Accumulation](https://arxiv.org/pdf/2002.08046.pdf), 2020 - Hierarchical Accumulation to encode parse tree structures into self-attention at a constant time complexity. [Code](https://github.com/nxphi47/tree_transformer)
  - [Lite Transformer with Long-Short Range Attention](https://arxiv.org/pdf/2004.11886.pdf), 2020 - Models local context by convolution and long-distance relationship by attention. [Code](https://github.com/mit-han-lab/lite-transformer)
  - [Transformer-XH: Multi-Evidence Reasoning with eXtra Hop Attention](https://openreview.net/pdf/d046083250740c4f9687e47d1df323759b66b5e4.pdf), 2020 - Enables intrinsic modeling of structured text by "hopping" around the document. [Code](https://github.com/microsoft/Transformer-XH)
  - [Monotonic Multihead Attention](https://arxiv.org/pdf/1909.12406.pdf), 2020 - Adds a multihead monotonic attention mechanism for latency control. [Code](https://github.com/pytorch/fairseq/tree/master/examples/simultaneous_translation)
  - [Reformer: The Efficient Transformer](https://arxiv.org/pdf/2001.04451.pdf), 2020 - Memory-efficiency and faster on long-sequences. [Code](https://github.com/google/trax/tree/master/trax/models/reformer)

- **Papers without code:**
  - [An Attention Free Transformer](https://arxiv.org/pdf/2105.14103.pdf), 2021 - An efficient variant of Transformers that eliminates the need for dot product self attention. 
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

### Custom Layers
- **Papers with code:**
  - [The Tree Ensemble Layer: Differentiability meets Conditional Computation](https://arxiv.org/pdf/2002.07772v1.pdf), 2020 - An ensemble of differentiable decision trees (a.k.a. soft trees).  [Code](https://github.com/google-research/google-research/tree/master/tf_trees)

---
