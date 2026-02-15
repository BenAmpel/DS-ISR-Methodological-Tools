---
layout: default
title: "LLMs & Natural Language Processing"
parent: Topics
nav_order: 2
---

# LLMs & Natural Language Processing
*Papers, Models, and Repositories for Large Language Models, Text Understanding, and Generation.*

*Last updated: February 2026*

| | | |
|-|-|-|
| [Large Language Models](#Large-Language-Models) | [Retrieval-Augmented Generation (RAG)](#Retrieval-Augmented-Generation-RAG) | [Transformers](#Transformers) |
| [Topic Modeling](#Topic-Modeling) | [Knowledge Distillation](#Knowledge-Distillation) | [Custom Layers](#Custom-Layers) |
| [Hallucination & Evaluation](#Hallucination--Evaluation) | [Synthetic Data Generation](#Synthetic-Data-Generation) | |

---

> **IS Research Applications:** Build domain-specific chatbots and Q&A systems over IS literature corpora (RAG); classify open-ended survey responses at scale; detect sentiment in online reviews or social media; generate synthetic respondents for IS experiments; automate systematic literature review extraction.

---

### Large Language Models
Open-weight LLMs have replaced static word embeddings as the foundation for NLP tasks. These models can be fine-tuned, prompted, or distilled for domain-specific IS research applications.

- **Frontier Open-Weight Models:**
  - [LLaMA 3](https://ai.meta.com/blog/meta-llama-3/) (Meta, 2024) - State-of-the-art open-weight family (8B, 70B, 405B). Best general-purpose open model; widely used for fine-tuning and RAG pipelines. [Code](https://github.com/meta-llama/llama3)
  - [Gemma 2](https://blog.google/technology/developers/google-gemma-2/) (Google, 2024) - Compact, high-performance open models (2B, 9B, 27B) optimized for efficient inference. Strong performance-per-parameter ratio. [Code](https://github.com/google-deepmind/gemma)
  - [Mistral / Mixtral-8x22B](https://mistral.ai/news/mixtral-8x22b/) (Mistral AI, 2024) - A sparse mixture-of-experts model with 141B total parameters but only 39B active per token. Strong coding and reasoning performance with efficient inference. [Code](https://github.com/mistralai/mistral-inference)
  - [Qwen2](https://qwenlm.github.io/blog/qwen2/) (Alibaba, 2024) - High-performance multilingual models (0.5B–72B). Particularly strong on Chinese-language tasks and coding. [Code](https://github.com/QwenLM/Qwen2)
  - [DeepSeek-V3](https://arxiv.org/abs/2412.19437) (DeepSeek, 2025) - A 671B MoE model trained at a fraction of the cost of comparable closed models. Demonstrates the rapid commoditization of frontier LLM capabilities. [Code](https://github.com/deepseek-ai/DeepSeek-V3)

- **Model Registries & Leaderboards:**
  - [HuggingFace Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) - Standardized benchmark comparisons across open-weight models.
  - [LMSYS Chatbot Arena](https://chat.lmsys.org/) - Human preference-based rankings via blind pairwise comparisons.

---

### Retrieval-Augmented Generation (RAG)
RAG connects LLMs to external knowledge sources (databases, document corpora, APIs) at inference time, enabling factual grounding without retraining. It is the standard architecture for building LLM-powered DSR artifacts that must access domain-specific knowledge.

- **Seminal Paper:**
  - [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401), 2020 - Introduced the RAG framework combining a dense retriever (DPR) with a generative model (BART). [Code](https://github.com/huggingface/transformers/tree/main/examples/research_projects/rag)

- **Papers with code:**
  - [REALM: Retrieval-Augmented Language Model Pre-Training](https://arxiv.org/abs/2002.08909), 2020 - Integrates retrieval into the pre-training objective itself. [Code](https://github.com/google-research/language/tree/master/language/realm)
  - [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511), 2023 - Model learns to decide when to retrieve and critiques its own outputs using special tokens. [Code](https://github.com/AkariAsai/self-rag)

- **Frameworks & Tools:**
  - [LangChain](https://github.com/langchain-ai/langchain) - The dominant framework for building LLM applications with retrieval, agents, and chains. Provides prebuilt RAG pipelines, document loaders, and vector store integrations.
  - [LlamaIndex](https://github.com/run-llama/llama_index) - Data framework optimized for connecting LLMs to structured and unstructured enterprise data. Stronger than LangChain for complex document indexing strategies.
  - [Chroma](https://github.com/chroma-core/chroma) - Lightweight, embeddable vector database; the easiest entry point for local RAG development.
  - [FAISS](https://github.com/facebookresearch/faiss) - Facebook's high-performance library for efficient similarity search over dense vectors.

---

### Transformers
[Seminal Paper](https://arxiv.org/pdf/1706.03762.pdf), 2017

An introduction to transformers can be found [here](https://github.com/will-thompson-k/tldr-transformers).

- **Papers with code:**
  - [Fastformer: Additive Attention Can Be All You Need](https://arxiv.org/pdf/2108.09084.pdf), 2021 - Efficient Transformer variant based on additive attention with linear complexity. [Code](https://github.com/wilile26811249/Fastformer-PyTorch)
  - [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/pdf/2106.01345v2.pdf), 2021 - Outputs optimal actions by conditioning a masked Transformer on desired return, past states, and actions. [Code](https://github.com/kzl/decision-transformer)
  - [CoBERL: Contrastive BERT for Reinforcement Learning](https://arxiv.org/pdf/2107.05431v1.pdf), 2021 - Combines contrastive loss and hybrid LSTM-transformer architecture for data-efficient learning from pixels. [Code](https://github.com/deepmind/dm_control)
  - [Encoding word order in complex embeddings](https://arxiv.org/pdf/1912.12333.pdf), 2020 - Custom embedding that models global absolute positions and word order relationships. [Code](https://github.com/iclr-complex-order/complex-order)
  - [Tree-Structured Attention with Hierarchical Accumulation](https://arxiv.org/pdf/2002.08046.pdf), 2020 - Encodes parse tree structures into self-attention at constant time complexity. [Code](https://github.com/nxphi47/tree_transformer)
  - [Lite Transformer with Long-Short Range Attention](https://arxiv.org/pdf/2004.11886.pdf), 2020 - Models local context by convolution and long-distance relationships by attention. [Code](https://github.com/mit-han-lab/lite-transformer)
  - [Transformer-XH: Multi-Evidence Reasoning with eXtra Hop Attention](https://openreview.net/pdf/d046083250740c4f9687e47d1df323759b66b5e4.pdf), 2020 - Enables intrinsic modeling of structured text by "hopping" around documents. [Code](https://github.com/microsoft/Transformer-XH)
  - [Monotonic Multihead Attention](https://arxiv.org/pdf/1909.12406.pdf), 2020 - Adds multihead monotonic attention for latency control in simultaneous translation. [Code](https://github.com/pytorch/fairseq/tree/master/examples/simultaneous_translation)
  - [Reformer: The Efficient Transformer](https://arxiv.org/pdf/2001.04451.pdf), 2020 - Memory-efficient and faster on long sequences via locality-sensitive hashing. [Code](https://github.com/google/trax/tree/master/trax/models/reformer)

- **Papers without code:**
  - [An Attention Free Transformer](https://arxiv.org/pdf/2105.14103.pdf), 2021 - Efficient variant that eliminates dot product self-attention.
  - [On Identifiability in Transformers](https://openreview.net/pdf?id=BJg1f6EFDB), 2020 - Introduces effective attention to improve explanatory interpretations.
  - [Are Transformers universal approximators of sequence-to-sequence functions?](https://arxiv.org/pdf/1912.10077.pdf), 2020 - Proves fixed-width self-attention layers can compute contextual mappings of input sequences.
  - [Robustness Verification for Transformers](https://arxiv.org/pdf/2002.06622.pdf), 2020 - A robustness verification algorithm shedding light on Transformer interpretability.

---

### Topic Modeling
[Seminal Paper: LDA](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf?TB_iframe=true&width=370.8&height=658.8), 2003

> **IS Research Application:** Topic modeling discovers latent themes across large text corpora (e.g., app store reviews, forum posts, interview transcripts) without requiring predefined categories — enabling inductive theory building from qualitative data at scale.

- **Papers with code:**
  - [BERTopic: Leveraging BERT and c-TF-IDF to Create Easily Interpretable Topics](https://arxiv.org/abs/2203.05794), 2022 - The modern standard for topic modeling. Uses sentence embeddings (BERT/SentenceTransformers) to cluster documents, then extracts interpretable topic labels via a class-based TF-IDF variant. Dramatically more coherent and interpretable than LDA on short texts. [Code](https://github.com/MaartenGr/BERTopic)
  - [Top2Vec: Distributed Representations of Topics](https://arxiv.org/abs/2008.09470), 2020 - Jointly learns document, word, and topic vectors. Topics emerge naturally from the embedding space without requiring the number of topics to be specified in advance. [Code](https://github.com/ddangelov/Top2Vec)
  - [CTM: Combined Topic Model](https://arxiv.org/abs/2004.14914), 2021 - Combines contextual BERT embeddings with neural topic models (ProdLDA) for improved topic coherence on short texts. [Code](https://github.com/MilaNLProc/contextualized-topic-models)

- **Papers without code:**
  - [Hierarchical Topic Modeling of Twitter Data for Online Analytical Processing](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8607040), 2018 - Mines hierarchical topic dimensions of tweets using LDA + word2vec for semantic analysis.
  - [Dynamic Topic Models](https://dl.acm.org/doi/10.1145/1143844.1143859), 2006 - Extends LDA to capture how topics evolve over time — applicable to longitudinal IS research on technology discourse.

---

### Hallucination & Evaluation
LLMs frequently generate plausible-sounding but factually incorrect content ("hallucinations"). Detecting and measuring this is critical for deploying LLMs in IS research artifacts.

- **Papers with code:**
  - [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/pdf/1904.09675.pdf), 2020 - Evaluative metric for text generation using contextual similarity scores from BERT. [Code](https://github.com/Tiiiger/bert_score)
  - [FActScoring: Fine-grained Atomic Evaluation of Factual Precision](https://arxiv.org/abs/2305.14251), 2023 - Decomposes generated text into atomic claims and verifies each against a knowledge source. [Code](https://github.com/shmsw25/FActScoring)

- **Papers without code:**
  - [SHI: Sentence-level Hallucination Index](https://arxiv.org/abs/2409.07708), RANLP 2025 - A metric for detecting false attribution and hallucinated citations in LLM outputs. Particularly relevant for automated literature review tools.
  - [Survey of Hallucination in Natural Language Generation](https://arxiv.org/abs/2202.03629), 2023 - Comprehensive taxonomy of hallucination types and evaluation methods across NLG tasks.

---

### Synthetic Data Generation
LLMs can generate synthetic text data (e.g., resumes, reviews, survey responses) for IS research when real data is scarce, sensitive, or expensive to collect.

- **Papers with code:**
  - [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560), 2022 - Uses in-context examples to have an LLM generate its own instruction-following training data. [Code](https://github.com/yizhongw/self-instruct)
  - [Textbooks Are All You Need (phi-1)](https://arxiv.org/abs/2306.11644), 2023 - Demonstrates that small models trained on high-quality synthetic data outperform larger models trained on web-scraped data. [Code](https://huggingface.co/microsoft/phi-1)

---

### Knowledge Distillation
- **Papers with code:**
  - [Big Self-Supervised Models are Strong Semi-Supervised Learners](https://arxiv.org/pdf/2006.10029.pdf), 2020 - Three-step algorithm: unsupervised pretraining (SimCLRv2), supervised fine-tuning on few labels, distillation with unlabeled examples. [Code](https://github.com/google-research/simclr)

---

### Custom Layers
- **Papers with code:**
  - [The Tree Ensemble Layer: Differentiability meets Conditional Computation](https://arxiv.org/pdf/2002.07772v1.pdf), 2020 - An ensemble of differentiable decision trees (soft trees) as a neural network layer. [Code](https://github.com/google-research/google-research/tree/master/tf_trees)

---

### Text Embedding Models
Modern embedding models produce dense vectors for semantic search, clustering, and similarity tasks. They are a core component of RAG pipelines and classification systems.

- **Papers with code:**
  - [E5: Text Embeddings by Weakly-Supervised Contrastive Pre-training](https://arxiv.org/abs/2212.03533), 2022 - Trains general-purpose text embeddings using weakly supervised contrastive learning on (query, passage) pairs mined from the web. Strong baseline for most NLP tasks. [Code](https://github.com/microsoft/unilm/tree/master/e5)
  - [GTE: Generalized Text Embeddings](https://arxiv.org/abs/2308.03281), 2023 - Multi-stage contrastive training pipeline producing state-of-the-art dense embeddings on MTEB. [Code](https://huggingface.co/thenlper/gte-large)
  - [Nomic Embed: Training a Reproducible Long Context Text Embedder](https://arxiv.org/abs/2402.01613), 2024 - Fully open-source, long-context (8192 token) embedding model competitive with OpenAI's text-embedding-3. [Code](https://github.com/nomic-ai/nomic)

- **Leaderboards & Benchmarks:**
  - [MTEB: Massive Text Embedding Benchmark](https://huggingface.co/spaces/mteb/leaderboard) - The standard benchmark for comparing embedding models across retrieval, clustering, classification, and reranking tasks. Essential reference before choosing an embedding model for IS research. [Code](https://github.com/embeddings-benchmark/mteb)

- **Tools:**
  - [SentenceTransformers](https://www.sbert.net/) - The standard Python library for computing and using text embeddings. Integrates with all major embedding models and supports fine-tuning. [Code](https://github.com/UKPLab/sentence-transformers)

---

### Long-Context LLMs
Many IS research tasks require processing entire documents, codebases, or interview transcripts — exceeding standard context limits.

- **Models & Papers:**
  - [Gemini 1.5 Pro: 1M Token Context Window](https://storage.googleapis.com/deepmind-media/gemini/gemini_v1_5_report.pdf), 2024 - Demonstrates that 1M-token context enables processing of entire books, codebases, and research corpora in a single pass. Directly applicable to full-document IS research.
  - [LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753), 2024 - Extends transformer context via non-uniform positional interpolation. Enables training-efficient long-context extension of any RoPE-based model. [Code](https://github.com/microsoft/LongRoPE)
  - [RULER: What's the Real Context Size of Your Long-Context Language Models?](https://arxiv.org/abs/2404.06654), 2024 - Benchmark revealing that effective context utilization degrades well before the nominal context limit is reached. [Code](https://github.com/hsiehjackson/RULER)

---

### Agent Memory & Persistence
LLM agents for IS research artifacts need to remember past interactions, accumulate knowledge, and maintain state across sessions.

- **Papers with code:**
  - [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560), 2023 - Manages unlimited context via a tiered memory architecture (in-context, external storage, archival). Enables perpetual conversational agents with growing knowledge bases. [Code](https://github.com/cpacker/MemGPT)
  - [Cognitive Architectures for Language Agents (CoALA)](https://arxiv.org/abs/2309.02427), 2023 - A systematic framework classifying agent memory (working, episodic, semantic, procedural) and decision processes. Foundational for IS researchers designing intelligent artifact architectures.

---

**Related Sections:** [Reinforcement Learning](../ReinforcementLearning/README.md) | [Fine-Tuning](../FineTuning/README.md) | [Multimodal Models](../MultimodalModels/README.md) | [Prompt Engineering](../Prompt-Engineering/README.md) | [Graphs](../Graphs/README.md)
