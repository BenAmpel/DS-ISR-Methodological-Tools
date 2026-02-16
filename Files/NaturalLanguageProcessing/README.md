---
layout: default
title: "LLMs & Natural Language Processing"
parent: Topics
nav_order: 2
permalink: /Files/NaturalLanguageProcessing/
---


{::nomarkdown}
<div class="section-meta">
  <span class="meta-pill meta-reading">⏱ ~11 min read</span>
  <span class="meta-pill meta-updated">📅 Updated Feb 2026</span>
  <span class="meta-pill meta-intermediate">📊 Intermediate</span>
  <span class="meta-pill meta-prereq">🔑 Python, ML basics</span>
</div>
{:/}

# LLMs & Natural Language Processing
*Papers, Models, and Repositories for Large Language Models, Text Understanding, and Generation.*

*Last updated: February 2026*

| | | |
|-|-|-|
| [Large Language Models](#large-language-models) | [Retrieval-Augmented Generation (RAG)](#retrieval-augmented-generation-rag) | [Transformers](#transformers) |
| [Topic Modeling](#topic-modeling) | [Hallucination & Evaluation](#hallucination--evaluation) | [Synthetic Data Generation](#synthetic-data-generation) |
| [Knowledge Distillation](#knowledge-distillation) | [Text Embedding Models](#text-embedding-models) | [Long-Context LLMs](#long-context-llms) |
| [Agent Memory & Persistence](#agent-memory--persistence) | [Custom Layers](#custom-layers) | |

> ⭐ **Open-source** models and tools are marked throughout this section — prioritise these to reduce costs and keep data local.

---

> **IS Research Applications:** Build domain-specific chatbots and Q&A systems over IS literature corpora (RAG); classify open-ended survey responses at scale; detect sentiment in online reviews or social media; generate synthetic respondents for IS experiments; automate systematic literature review extraction.

---

### Large Language Models
Open-weight LLMs have replaced static word embeddings as the foundation for NLP tasks. These models can be fine-tuned, prompted, or distilled for domain-specific IS research applications.

- **Frontier Open-Weight Models:**
  - [LLaMA 3 / 3.1 / 3.2 / 3.3](https://ai.meta.com/blog/meta-llama-3/) (Meta, 2024) - LLaMA 3.1 (July 2024) extended to 128k context across 8B, 70B, and 405B. LLaMA 3.2 added multimodal vision (11B, 90B) and edge models (1B, 3B). LLaMA 3.3 70B matches 405B on many tasks. [Code](https://github.com/meta-llama/llama3)
  - [Gemma 2](https://arxiv.org/abs/2408.00118) (Google, 2024) - Compact, high-performance open models (2B, 9B, 27B). Uses interleaved local/global attention and distillation from Gemini. 27B competitive with models twice its size. [Code](https://github.com/google-deepmind/gemma) — Gemma 3 (2025) adds multimodality and extended context.
  - [Mistral / Mixtral-8x22B](https://mistral.ai/news/mixtral-8x22b/) (Mistral AI, 2024) - Sparse MoE with 141B total / 39B active parameters. Mistral Large 2 (123B, 128k context) and Mistral Nemo (12B, Apache 2.0) released mid-2024. [Code](https://github.com/mistralai/mistral-inference)
  - [Qwen2 / Qwen2.5](https://arxiv.org/abs/2407.10671) (Alibaba, 2024) - Strong multilingual models (0.5B–72B, Apache 2.0). Qwen2.5 (September 2024) added specialized coding (Qwen2.5-Coder-32B) and math variants. 128k context on 72B. [Code](https://github.com/QwenLM/Qwen2.5)
  - [DeepSeek-V3](https://arxiv.org/abs/2412.19437) (DeepSeek, 2024) - 671B MoE model (37B active) trained at ~$6M, demonstrating rapid commoditization of frontier LLM capabilities. [Code](https://github.com/deepseek-ai/DeepSeek-V3)
  - [DeepSeek-R1](https://arxiv.org/abs/2501.12948) (DeepSeek, 2025) ⭐ MIT license - Reasoning model trained entirely via reinforcement learning (no supervised CoT). Matches OpenAI o1 on math/coding benchmarks. Distilled R1 variants available at 1.5B–70B. [Code](https://github.com/deepseek-ai/DeepSeek-R1)
  - [Phi-4](https://arxiv.org/abs/2412.08905) (Microsoft, 2024) ⭐ MIT license - 14B model heavily trained on synthetic data. Strong STEM reasoning. Phi-3 family (3.8B–14B) achieves GPT-3.5-level performance at small sizes. [Code](https://huggingface.co/microsoft/phi-4)
  - [OLMo 2](https://arxiv.org/abs/2501.00656) (Allen AI, 2025) ⭐ fully open - The most transparent LLM: training data, code, weights, and evaluation all publicly released. 7B and 13B. Competitive with Llama 3.1. [Code](https://github.com/allenai/OLMo)
  - [Command R / R+](https://huggingface.co/CohereForAI/c4ai-command-r-plus) (Cohere, 2024) - Optimized for RAG and tool use with 128k context. Command R (35B) and Command R+ (104B). Strong citation grounding in retrieved outputs.

- **Model Registries & Leaderboards:**
  - [HuggingFace Open LLM Leaderboard v2](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) - Revamped 2024 with harder benchmarks (MMLU-Pro, GPQA, MuSR, MATH, IFEval, BBH). Replaces the original leaderboard.
  - [LMSYS Chatbot Arena](https://chat.lmsys.org/) - Human preference-based ELO rankings via blind pairwise comparisons. Expanded to include vision models.
  - [LiveBench](https://livebench.ai/) - Contamination-free benchmark updated monthly to prevent benchmark overfitting. [Paper](https://arxiv.org/abs/2406.19314)
  - [AlpacaEval 2.0](https://github.com/tatsu-lab/alpaca_eval) - Length-controlled win rates against GPT-4. [Paper](https://arxiv.org/abs/2404.04475)

---

### Retrieval-Augmented Generation (RAG)
RAG connects LLMs to external knowledge sources (databases, document corpora, APIs) at inference time, enabling factual grounding without retraining. It is the standard architecture for building LLM-powered DSR artifacts that must access domain-specific knowledge.

- **Seminal Paper:**
  - [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401), 2020 - Introduced the RAG framework combining a dense retriever (DPR) with a generative model (BART). [Code](https://github.com/huggingface/transformers/tree/main/examples/research_projects/rag)

- **Papers with code:**
  - [REALM: Retrieval-Augmented Language Model Pre-Training](https://arxiv.org/abs/2002.08909), 2020 - Integrates retrieval into the pre-training objective itself. [Code](https://github.com/google-research/language/tree/master/language/realm)
  - [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511), 2023 - Model learns to decide when to retrieve and critiques its own outputs using special tokens. [Code](https://github.com/akariasai/self-rag)
  - [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059), 2024 - Clusters and summarizes document chunks into a tree; retrieves at multiple abstraction levels. Strong on multi-hop and synthesis tasks. [Code](https://github.com/parthsarthi03/raptor)
  - [GraphRAG](https://arxiv.org/abs/2404.16130) (Microsoft, 2024) - Builds a knowledge graph from documents; uses graph community summaries for global queries. Addresses failure modes of standard vector RAG for whole-corpus questions. [Code](https://github.com/microsoft/graphrag)
  - [Corrective RAG (CRAG)](https://arxiv.org/abs/2401.15884), 2024 - Adds a retrieval evaluator that triggers web search when retrieved documents are judged irrelevant. [Code](https://github.com/HuskyInSalt/CRAG)
  - [Adaptive RAG](https://arxiv.org/abs/2403.14403), 2024 - Dynamically routes queries to no-retrieval, single-step, or multi-step retrieval based on a query complexity classifier. Reduces unnecessary retrieval overhead.
  - [HyDE: Hypothetical Document Embeddings](https://arxiv.org/abs/2212.10496), 2022 - Generates a hypothetical answer with an LLM, then embeds that for retrieval. Improves zero-shot dense retrieval; widely adopted in production RAG pipelines.

- **Surveys:**
  - [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) (Gao et al., 2024) - Comprehensive taxonomy of RAG approaches (Naive, Advanced, Modular RAG). The standard reference for RAG research.
  - [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217), 2023 - Automated RAG evaluation framework measuring faithfulness, answer relevancy, context precision, and recall. [Code](https://github.com/explodinggradients/ragas)

- **Frameworks & Tools:**
  - [LangChain](https://github.com/langchain-ai/langchain) - The dominant framework for building LLM applications with retrieval, agents, and chains. Provides prebuilt RAG pipelines, document loaders, and vector store integrations.
  - [LlamaIndex](https://github.com/run-llama/llama_index) - Data framework optimized for connecting LLMs to structured and unstructured enterprise data. Stronger than LangChain for complex document indexing strategies.
  - [RAGatouille](https://github.com/bclavie/RAGatouille) - Python library wrapping ColBERT for late-interaction (token-level) retrieval in RAG pipelines. Significantly improves recall over bi-encoder retrieval.
  - [Chroma](https://github.com/chroma-core/chroma) - Lightweight, embeddable vector database; the easiest entry point for local RAG development.
  - [FAISS](https://github.com/facebookresearch/faiss) - Facebook's high-performance library for efficient similarity search over dense vectors.
  - [pgvector](https://github.com/pgvector/pgvector) ⭐ open-source - PostgreSQL extension for vector similarity search. Production-ready with no separate infrastructure required. Best choice when your data already lives in Postgres.
  - [Qdrant](https://github.com/qdrant/qdrant) ⭐ open-source (Apache 2.0) - Rust-based vector DB with payload filtering, quantization, and sparse+dense hybrid search. Fast development cadence throughout 2024.
  - [Milvus](https://github.com/milvus-io/milvus) ⭐ open-source (Apache 2.0) - Distributed vector DB. Milvus 2.4 (2024) added sparse vector support and GPU indexing for billion-scale collections.
  - [LanceDB](https://github.com/lancedb/lancedb) ⭐ open-source (Apache 2.0) - Serverless vector DB built on the Lance columnar format. Excellent for embedded/local research use without a running server.

---

### Transformers
[Seminal Paper](https://arxiv.org/pdf/1706.03762.pdf), 2017

An introduction to transformers can be found [here](https://github.com/will-thompson-k/tldr-transformers).

- **Attention Efficiency (2023–2024):**
  - [Flash Attention 2](https://arxiv.org/abs/2307.08691), 2023 - Reorders attention computation via IO-aware tiling; 2× faster than v1. Now the standard training implementation for virtually all LLMs. [Code](https://github.com/Dao-AILab/flash-attention)
  - [Flash Attention 3](https://arxiv.org/abs/2407.08608), 2024 - H100-specific implementation using WGMMA and TMA instructions; ~2× faster than FA2 on Hopper GPUs.
  - [GQA: Grouped-Query Attention](https://arxiv.org/abs/2305.13245), 2023 - Shares key/value heads across groups of query heads, reducing KV-cache memory significantly. Used in Llama 3, Mistral, and Gemma.
  - [Ring Attention with Blockwise Transformers](https://arxiv.org/abs/2310.01889), 2023 - Distributes long-context attention across devices in a ring topology; context length limited only by total GPU memory. [Code](https://github.com/lhao499/ring-attention)

- **State Space Models & Transformer Alternatives:**
  - [Mamba](https://arxiv.org/abs/2312.00752), 2023 - Selective State Space Model with input-dependent transitions. Linear-time inference, no attention. Competitive with Transformers up to ~3B scale. [Code](https://github.com/state-spaces/mamba)
  - [Mamba-2](https://arxiv.org/abs/2405.21060), 2024 - Connects SSMs and attention via Structured State Space Duality; faster than Mamba 1. [Code](https://github.com/state-spaces/mamba)
  - [xLSTM](https://arxiv.org/abs/2405.04517), 2024 - Extended LSTM with exponential gating and matrix memory cells (mLSTM/sLSTM). Competitive with Mamba and Transformers. [Code](https://github.com/NX-AI/xlstm)
  - [Hawk / Griffin](https://arxiv.org/abs/2402.19427) (Google DeepMind, 2024) - Hybrid gated linear recurrences + local attention. Competitive with pure Transformers at lower memory cost.

- **Classic Efficient Transformer Papers (2020–2021):**
  - [Fastformer: Additive Attention Can Be All You Need](https://arxiv.org/pdf/2108.09084.pdf), 2021 - Efficient Transformer variant based on additive attention with linear complexity. [Code](https://github.com/wilile26811249/fastformer-pytorch)
  - [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/pdf/2106.01345v2.pdf), 2021 - Outputs optimal actions by conditioning a masked Transformer on desired return, past states, and actions. [Code](https://github.com/kzl/decision-transformer)
  - [Encoding word order in complex embeddings](https://arxiv.org/pdf/1912.12333.pdf), 2020 - Custom embedding that models global absolute positions and word order relationships. [Code](https://github.com/iclr-complex-order/complex-order)
  - [Lite Transformer with Long-Short Range Attention](https://arxiv.org/pdf/2004.11886.pdf), 2020 - Models local context by convolution and long-distance relationships by attention. [Code](https://github.com/mit-han-lab/lite-transformer)
  - [Reformer: The Efficient Transformer](https://arxiv.org/pdf/2001.04451.pdf), 2020 - Memory-efficient and faster on long sequences via locality-sensitive hashing. [Code](https://github.com/google/trax/tree/master/trax/models/reformer)
  - [Transformer-XH: Multi-Evidence Reasoning with eXtra Hop Attention](https://openreview.net/pdf/d046083250740c4f9687e47d1df323759b66b5e4.pdf), 2020 - Enables intrinsic modeling of structured text by "hopping" around documents. [Code](https://github.com/microsoft/transformer-xh)
  - [Monotonic Multihead Attention](https://arxiv.org/pdf/1909.12406.pdf), 2020 - Adds multihead monotonic attention for latency control in simultaneous translation. [Code](https://github.com/pytorch/fairseq/tree/master/examples/simultaneous_translation)

- **Papers without code:**
  - [An Attention Free Transformer](https://arxiv.org/pdf/2105.14103.pdf), 2021 - Efficient variant that eliminates dot product self-attention.
  - [On Identifiability in Transformers](https://openreview.net/pdf?id=bjg1f6efdb), 2020 - Introduces effective attention to improve explanatory interpretations.
  - [Are Transformers universal approximators of sequence-to-sequence functions?](https://arxiv.org/pdf/1912.10077.pdf), 2020 - Proves fixed-width self-attention layers can compute contextual mappings of input sequences.
  - [Robustness Verification for Transformers](https://arxiv.org/pdf/2002.06622.pdf), 2020 - A robustness verification algorithm shedding light on Transformer interpretability.

- **Surveys:**
  - [Efficient Large Language Models: A Survey](https://arxiv.org/abs/2312.03863), 2023 - Covers pruning, quantization, distillation, efficient architectures, and inference optimization. Comprehensive reference for deploying LLMs in resource-constrained IS research settings.

---

### Topic Modeling
[Seminal Paper: LDA](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf?tb_iframe=true&width=370.8&height=658.8), 2003

> **IS Research Application:** Topic modeling discovers latent themes across large text corpora (e.g., app store reviews, forum posts, interview transcripts) without requiring predefined categories — enabling inductive theory building from qualitative data at scale.

- **Papers with code:**
  - [BERTopic: Leveraging BERT and c-TF-IDF to Create Easily Interpretable Topics](https://arxiv.org/abs/2203.05794), 2022 - The modern standard for topic modeling. Uses sentence embeddings to cluster documents, then extracts interpretable labels via c-TF-IDF. Version 0.16+ (2024) adds LLM-based label generation, zero-shot topic modeling, and guided topics via OpenAI, Cohere, or local models. [Code](https://github.com/MaartenGr/BERTopic)
  - [Top2Vec: Distributed Representations of Topics](https://arxiv.org/abs/2008.09470), 2020 - Jointly learns document, word, and topic vectors. Topics emerge naturally without specifying the number in advance. [Code](https://github.com/ddangelov/top2vec)
  - [CTM: Combined Topic Model](https://arxiv.org/abs/2004.14914), 2021 - Combines contextual BERT embeddings with neural topic models (ProdLDA) for improved coherence on short texts. [Code](https://github.com/milanlproc/contextualized-topic-models)
  - [TopicGPT](https://arxiv.org/abs/2311.01449), 2024 - Uses LLMs to generate and verify topic assignments via a two-stage prompting pipeline. Topics represented as natural language descriptions rather than keyword lists. Outperforms LDA and BERTopic on human judgment. [Code](https://github.com/chtmp223/topicGPT)
  - [FASTopic](https://arxiv.org/abs/2405.17978), 2024 - Dual semantic alignment between document and topic embeddings. Faster than BERTopic on large corpora while maintaining competitive coherence. [Code](https://github.com/bobxwu/FASTopic)

- **Papers without code:**
  - [Is LLM a Good Topic Modeler?](https://arxiv.org/abs/2406.09413), 2024 - Systematic evaluation comparing LLM-based topic modeling (TopicGPT) against classical approaches (LDA, BERTopic) across coherence, diversity, and alignment metrics.
  - [Hierarchical Topic Modeling of Twitter Data for Online Analytical Processing](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8607040), 2018 - Mines hierarchical topic dimensions of tweets using LDA + word2vec for semantic analysis.
  - [Dynamic Topic Models](https://dl.acm.org/doi/10.1145/1143844.1143859), 2006 - Extends LDA to capture how topics evolve over time — applicable to longitudinal IS research on technology discourse.

---

### Hallucination & Evaluation
LLMs frequently generate plausible-sounding but factually incorrect content ("hallucinations"). Detecting and measuring this is critical for deploying LLMs in IS research artifacts.

- **Papers with code:**
  - [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/pdf/1904.09675.pdf), 2020 - Evaluative metric for text generation using contextual similarity scores from BERT. [Code](https://github.com/tiiiger/bert_score)
  - [FActScoring: Fine-grained Atomic Evaluation of Factual Precision](https://arxiv.org/abs/2305.14251), 2023 - Decomposes generated text into atomic claims and verifies each against a knowledge source. [Code](https://github.com/shmsw25/FActScore)
  - [SelfCheckGPT: Zero-Resource Hallucination Detection](https://arxiv.org/abs/2303.08896), 2023 - Detects hallucinations via sampling consistency: facts a model is confident about will appear consistently across multiple generations. No external knowledge base required. [Code](https://github.com/potsawee/selfcheckgpt)
  - [SAFE: Search-Augmented Factuality Evaluator](https://arxiv.org/abs/2403.18802) (Google, 2024) - Uses search to fact-check LLM responses at the atomic claim level. Part of the "Long-form Factuality" paper; sets a strong benchmark for automated factuality evaluation.

- **Evaluation Frameworks / Tools:**
  - [RAGAS](https://github.com/explodinggradients/ragas) ⭐ open-source - Automated RAG evaluation measuring faithfulness (hallucination), answer relevancy, context precision, and context recall. The standard evaluation suite for RAG pipelines. [Paper](https://arxiv.org/abs/2309.15217)
  - [DeepEval](https://github.com/confident-ai/deepeval) ⭐ open-source (Apache 2.0) - Comprehensive LLM evaluation framework. Includes hallucination, answer relevancy, faithfulness, and G-Eval metrics. Integrates with pytest for CI/CD testing of LLM pipelines.
  - [TruLens](https://github.com/truera/trulens) ⭐ open-source (MIT) - Evaluation and monitoring for LLM applications. Implements the RAG triad (context relevance, groundedness, answer relevance) as interpretable feedback functions.
  - [Giskard](https://github.com/Giskard-AI/giskard) ⭐ open-source (Apache 2.0) - LLM testing and red-teaming framework. Automated vulnerability scanning for hallucination, prompt injection, stereotypes, and sensitive topic leakage. Generates test reports.

- **Papers without code:**
  - [SHI: Sentence-level Hallucination Index](https://arxiv.org/abs/2409.07708), RANLP 2025 - A metric for detecting false attribution and hallucinated citations in LLM outputs. Particularly relevant for automated literature review tools.
  - [HaluEval 2.0](https://arxiv.org/abs/2407.10457), 2024 - 35k-sample benchmark for hallucination evaluation across tasks. Expanded coverage over v1.
  - [Survey of Hallucination in Natural Language Generation](https://arxiv.org/abs/2202.03629), 2023 - Comprehensive taxonomy of hallucination types and evaluation methods across NLG tasks.
  - [Siren's Song in the AI Ocean: A Survey on Hallucination in LLMs](https://arxiv.org/abs/2309.01219), 2023 - Updated survey covering LLM-specific hallucination, causes, and mitigation strategies.

---

### Synthetic Data Generation
LLMs can generate synthetic text data (e.g., resumes, reviews, survey responses) for IS research when real data is scarce, sensitive, or expensive to collect.

- **Papers with code:**
  - [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560), 2022 - Uses in-context examples to have an LLM generate its own instruction-following training data. Foundation for all subsequent instruction-tuning data pipelines. [Code](https://github.com/yizhongw/self-instruct)
  - [Textbooks Are All You Need (phi-1)](https://arxiv.org/abs/2306.11644), 2023 - Demonstrates that small models trained on high-quality synthetic data outperform larger models trained on web-scraped data. Direct influence on Phi-2, Phi-3, and Phi-4. [Code](https://huggingface.co/microsoft/phi-1)
  - [Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs](https://arxiv.org/abs/2406.08464), 2024 - Generates instruction-response pairs by prompting aligned LLMs with only the conversation template prefix; extracts naturally occurring instructions without manual seed writing. Produces 3M+ high-quality pairs. [Code](https://github.com/magpie-align/magpie)
  - [WizardLM: Empowering LLMs to Follow Complex Instructions](https://arxiv.org/abs/2304.12244), 2023 - Evol-Instruct method evolves simple instructions in breadth (diverse) and depth (complex) using an LLM. Applied to WizardCoder and WizardMath for domain-specific data. [Code](https://github.com/nlpxucan/WizardLM)
  - [Distilling Step-by-Step!](https://arxiv.org/abs/2305.02301), 2023 - Distills both labels and chain-of-thought rationales from LLMs; achieves better performance with less data than standard fine-tuning. Directly applicable to generating IS survey response data with reasoning traces.
  - [DataDreamer: A Tool for Synthetic Data Generation and Reproducible LLM Workflows](https://arxiv.org/abs/2402.10379), 2024 ⭐ open-source - Python library for reproducible synthetic data generation. Automatic caching, provenance tracking, and pipeline DAGs. Designed for NLP researchers. [Code](https://github.com/datadreamer-dev/DataDreamer)

- **Tools:**
  - [Distilabel](https://github.com/argilla-io/distilabel) ⭐ open-source (Apache 2.0) - Framework for synthetic data generation pipelines by Argilla. Supports multiple LLM backends, async generation, and structured pipeline DAGs. Used to generate high-quality instruction datasets for fine-tuning open models.
  - [Self-Play Fine-Tuning (SPIN)](https://arxiv.org/abs/2401.01335), 2024 - Generates synthetic training data by having the model compete against a previous version of itself, iteratively improving alignment without additional human annotations. [Code](https://github.com/uclaml/SPIN)

---

### Knowledge Distillation
Distillation transfers knowledge from large "teacher" models into smaller, deployable "student" models — critical for deploying LLMs in resource-constrained IS research settings or on local hardware.

- **Papers with code:**
  - [Big Self-Supervised Models are Strong Semi-Supervised Learners](https://arxiv.org/pdf/2006.10029.pdf), 2020 - Three-step algorithm: unsupervised pretraining (SimCLRv2), supervised fine-tuning on few labels, distillation with unlabeled examples. [Code](https://github.com/google-research/simclr)
  - [MiniLLM: Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2306.08543), 2024 - Minimizes reverse KL divergence (instead of forward KL) for LLM distillation, avoiding mode-averaging artifacts. Student models significantly outperform standard KD. [Code](https://github.com/microsoft/LMOps/tree/main/minillm)
  - [GKD: Generalized Knowledge Distillation](https://arxiv.org/abs/2306.13649) (Google DeepMind, 2024) - On-policy distillation: student generates samples, then learns from teacher's distribution over those samples. Addresses exposure bias in standard offline KD.
  - [Speculative Decoding](https://arxiv.org/abs/2211.17192) (Leviathan et al., 2023) - Uses a small draft model to propose token candidates verified in parallel by the large model. 2–3× speedup with mathematically identical outputs to the large model.
  - [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774), 2024 - Adds multiple parallel decoding heads to a single model (no separate draft model). ~2× speedup. [Code](https://github.com/FasterDecoding/Medusa)
  - [Distilling Step-by-Step!](https://arxiv.org/abs/2305.02301), 2023 - Distills both task labels and chain-of-thought rationales from LLMs, achieving better student performance with less labelled data than standard fine-tuning.
  - [TinyLlama: An Open-Source Small Language Model](https://arxiv.org/abs/2401.02385), 2024 - 1.1B model trained on 3T tokens with Llama 2 architecture. Demonstrates the performance frontier for small open models trained with scale. [Code](https://github.com/jzhang38/TinyLlama)

---

### Custom Layers
- **Papers with code:**
  - [The Tree Ensemble Layer: Differentiability meets Conditional Computation](https://arxiv.org/pdf/2002.07772v1.pdf), 2020 - An ensemble of differentiable decision trees (soft trees) as a neural network layer. [Code](https://github.com/google-research/google-research/tree/master/tf_trees)

---

### Text Embedding Models
Modern embedding models produce dense vectors for semantic search, clustering, and similarity tasks. They are a core component of RAG pipelines and classification systems.

- **Papers with code:**
  - [E5: Text Embeddings by Weakly-Supervised Contrastive Pre-training](https://arxiv.org/abs/2212.03533), 2022 - Trains general-purpose text embeddings using weakly supervised contrastive learning on (query, passage) pairs. Strong baseline for most NLP tasks. [Code](https://github.com/microsoft/unilm/tree/master/e5)
  - [E5-mistral-7B-instruct](https://arxiv.org/abs/2401.00368) (Microsoft, 2024) - Uses GPT-4 synthetic data to train instruction-following embeddings on a Mistral backbone. Strong zero-shot generalization across retrieval tasks. [HuggingFace](https://huggingface.co/intfloat/e5-mistral-7b-instruct)
  - [GTE: Generalized Text Embeddings](https://arxiv.org/abs/2308.03281), 2023 - Multi-stage contrastive training producing state-of-the-art dense embeddings on MTEB. [Code](https://huggingface.co/thenlper/gte-large)
  - [GTE-Qwen2-7B-Instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct) (Alibaba, 2024) ⭐ Apache 2.0 - Qwen2-based embedding model. Instruction-following, 7B parameters. Top MTEB scores as of late 2024.
  - [BGE-M3](https://arxiv.org/abs/2402.03216) (BAAI, 2024) ⭐ MIT license - Supports dense, sparse (learned), and multi-vector (ColBERT-style) retrieval in one model. 100+ languages, 8192 token context. The most versatile open embedding model. [Code](https://github.com/FlagOpen/FlagEmbedding)
  - [Nomic Embed v1.5](https://arxiv.org/abs/2402.01613), 2024 ⭐ fully open (Apache 2.0) - Long-context (8192 token) embedding with Matryoshka Representation Learning for variable-dimension embeddings. Competitive with OpenAI text-embedding-3. [Code](https://github.com/nomic-ai/contrastors)
  - [LLM2Vec](https://arxiv.org/abs/2404.05961) (McGill/Mila, 2024) ⭐ open-source - Converts any decoder-only LLM into a strong text encoder via masked next-token prediction and contrastive learning. Works with Llama, Mistral, and other open models. [Code](https://github.com/McGill-NLP/llm2vec)

- **Leaderboards & Benchmarks:**
  - [MTEB: Massive Text Embedding Benchmark](https://huggingface.co/spaces/mteb/leaderboard) - The standard benchmark for embedding models across retrieval, clustering, classification, and reranking. Updated in 2024 with multilingual (MMTEB), code retrieval, and long-document tasks. [Code](https://github.com/embeddings-benchmark/mteb) [Paper](https://arxiv.org/abs/2210.07316)

- **Tools:**
  - [SentenceTransformers](https://www.sbert.net/) ⭐ open-source - The standard Python library for computing and using text embeddings. Integrates with all major embedding models and supports fine-tuning. Version 3.x (2024) added support for late interaction (ColBERT), sparse, and multi-vector models. [Code](https://github.com/ukplab/sentence-transformers)

---

### Long-Context LLMs
Many IS research tasks require processing entire documents, codebases, or interview transcripts — exceeding standard context limits.

- **Models:**
  - [Gemini 1.5 Pro / Flash](https://storage.googleapis.com/deepmind-media/gemini/gemini_v1_5_report.pdf) (Google, 2024) - 1M token context (2M in API). MoE architecture with strong needle-in-a-haystack performance. Gemini 1.5 Flash offers a faster, cost-effective variant with the same 1M context.
  - [Llama 3.1](https://ai.meta.com/blog/meta-llama-3/) (Meta, 2024) - All Llama 3.1 models (8B, 70B, 405B) support 128k tokens via RoPE scaling. The standard open-weight choice for long-document IS research.
  - [Claude 3 / 3.5](https://www.anthropic.com/claude) (Anthropic, 2024) - 200k context across all Claude 3 models. Strong long-context retrieval and synthesis; Claude 3.5 Sonnet (June 2024) adds improved long-context reasoning.
  - [Mistral Large 2 / Command R+](https://mistral.ai/) (2024) - Both support 128k context and are optimized for grounded retrieval in long documents.

- **Techniques:**
  - [LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753), 2024 - Non-uniform positional interpolation for training-efficient context extension of any RoPE-based model. [Code](https://github.com/microsoft/longrope)
  - [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071), 2023 - Efficient context extension via attention scaling + NTK-aware RoPE interpolation. Adopted by many open models for context extension without full retraining.
  - [LongLoRA: Efficient Fine-tuning of Long-Context LLMs](https://arxiv.org/abs/2309.12307), 2023 - Extends Llama 2 to 100k context cheaply using shifted sparse attention. Fine-tuning long-context on a single GPU becomes feasible. [Code](https://github.com/dvlab-research/LongLoRA)
  - [Ring Attention with Blockwise Transformers](https://arxiv.org/abs/2310.01889), 2023 - Distributes attention across devices in a ring; context length limited only by total GPU memory. Used for million-token training.

- **Benchmarks:**
  - [RULER: What's the Real Context Size of Your Long-Context Language Models?](https://arxiv.org/abs/2404.06654), 2024 - Reveals effective context utilization degrades well before the nominal limit via multi-hop, aggregation, and QA tasks at varied lengths. [Code](https://github.com/hsiehjackson/ruler)
  - [HELMET: How to Evaluate Long-context Models Effectively and Thoroughly](https://arxiv.org/abs/2410.02694), 2024 - Comprehensive long-context evaluation covering hallucination, long-range dependencies, and information retrieval at 128k+ tokens.

---

### Agent Memory & Persistence
LLM agents for IS research artifacts need to remember past interactions, accumulate knowledge, and maintain state across sessions.

- **Papers with code:**
  - [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560), 2023 - Manages unlimited context via a tiered memory architecture (in-context, external storage, archival). Rebranded as **Letta** in 2024 with REST API and agent state persistence. [Code](https://github.com/letta-ai/letta)
  - [Cognitive Architectures for Language Agents (CoALA)](https://arxiv.org/abs/2309.02427), 2023 - A systematic framework classifying agent memory (working, episodic, semantic, procedural) and decision processes. Foundational for IS researchers designing intelligent artifact architectures.
  - [HippoRAG: Neurobiologically Inspired Long-Term Memory for LLMs](https://arxiv.org/abs/2405.14831), 2024 - Knowledge graph-based memory architecture inspired by the hippocampus. Strong on multi-hop QA requiring integration of facts across many documents. [Code](https://github.com/OSU-NLP-Group/HippoRAG)

- **Tools:**
  - [mem0](https://github.com/mem0ai/mem0) ⭐ open-source (Apache 2.0) - Intelligent memory layer for AI agents. Stores user preferences, facts, and conversation history with automatic conflict resolution and semantic retrieval. The most widely adopted open-source agent memory layer in 2024.
  - [Zep](https://github.com/getzep/zep) ⭐ open-source community edition (Apache 2.0) - Long-term memory for LLM applications via a temporal knowledge graph. Automatic fact extraction, hybrid retrieval, and session management.
  - [LangMem](https://github.com/langchain-ai/langmem) ⭐ open-source (MIT) - Memory abstraction layer in the LangChain/LangGraph ecosystem. Supports episodic, semantic, and procedural memory types with automatic consolidation across sessions.

- **Survey:**
  - [Survey on Memory Mechanisms in LLM-based Agents](https://arxiv.org/abs/2404.13501), 2024 - Taxonomy of memory types (sensory, short-term, long-term), storage mechanisms, and retrieval strategies for LLM agent systems.

---

**Related Sections:** [Reinforcement Learning](../ReinforcementLearning/README.md) | [Fine-Tuning](../FineTuning/README.md) | [Multimodal Models](../MultimodalModels/README.md) | [Prompt Engineering](../Prompt-Engineering/README.md) | [Graphs](../Graphs/README.md)
