---
layout: default
title: "Multimodal Models"
parent: Topics
nav_order: 12
---

# Multimodal Models
*Papers, Models, and Tools for Vision-Language Understanding, Image-Text Alignment, and Multimodal AI.*

*Last updated: February 2026*

> Multimodal models process and generate content across multiple modalities — primarily text and images, but also video, audio, and structured data. For IS researchers, these models enable analysis of screenshots, UI elements, survey stimuli, social media images, and document layouts.

| | | |
|-|-|-|
| [Vision-Language Models (VLMs)](#vision-language-models-vlms) | [Image-Text Alignment](#image-text-alignment) | [Document & UI Understanding](#document-ui-understanding) |
| [Multimodal Generation](#multimodal-generation) | [Benchmarks & Evaluation](#benchmarks-evaluation) | [Tools & Libraries](#tools-libraries) |

---

> **IS Research Applications:** Analyze user interface screenshots for usability studies; classify social media images alongside captions; extract structured data from survey forms and documents; generate consistent visual stimuli for experiments; evaluate AI-generated vs. human-created content.

---

### Vision-Language Models (VLMs)
VLMs accept both images and text as input and produce text output. They enable natural language interaction with visual content.

- **Frontier Closed Models:**
  - [GPT-4V / GPT-4o](#https://openai.com/research/gpt-4v-system-card) (OpenAI, 2023/2024) - The first widely deployed VLM API. GPT-4o extends this to real-time audio/image/text interleaving. Benchmark standard for multimodal reasoning.
  - [Gemini 1.5 Pro](#https://deepmind.google/technologies/gemini/) (Google, 2024) - Natively multimodal architecture with a 1M-token context window. Can process entire research papers with figures, long videos, and mixed-modality documents.
  - [Claude 3.5 Sonnet](#https://www.anthropic.com/claude) (Anthropic, 2024) - Strong document and screenshot understanding. Particularly effective for chart/figure interpretation and structured document extraction.

- **Open-Weight VLMs:**
  - [LLaVA: Large Language and Vision Assistant](#https://arxiv.org/abs/2304.08485), 2023 - Connected a CLIP vision encoder to LLaMA via a lightweight projection layer. Demonstrated that strong VLMs can be built by connecting pretrained vision and language encoders with minimal additional training. [Code](#https://github.com/haotian-liu/llava)
  - [LLaVA-1.5 / LLaVA-NeXT](#https://arxiv.org/abs/2310.03744), 2023/2024 - Improvements with better visual encoders and higher-resolution image handling. [Code](#https://github.com/haotian-liu/llava)
  - [InternVL2](#https://arxiv.org/abs/2404.16821), 2024 - State-of-the-art open-weight VLM family (1B–108B). Competitive with GPT-4V on many benchmarks. [Code](#https://github.com/opengvlab/internvl)
  - [Qwen-VL](#https://arxiv.org/abs/2308.12966), 2023 - Alibaba's open-weight VLM with strong document understanding and multilingual support. [Code](#https://github.com/qwenlm/qwen-vl)
  - [Phi-3 Vision](#https://arxiv.org/abs/2404.14219), 2024 - Microsoft's small (4B) but capable VLM. Runs on consumer hardware. [Code](#https://huggingface.co/microsoft/phi-3-vision-128k-instruct)

---

### Image-Text Alignment
These models learn joint representations of images and text, enabling zero-shot classification, image retrieval, and similarity scoring.

- **Seminal Paper:**
  - [CLIP: Learning Transferable Visual Models From Natural Language Supervision](#https://arxiv.org/abs/2103.00020), 2021 - Trained on 400M image-text pairs via contrastive learning. A single model can perform zero-shot classification, image retrieval, and semantic similarity scoring across any visual domain. [Code](#https://github.com/openai/clip)

- **Papers with code:**
  - [ALIGN: Scaling Up Visual and Vision-Language Representation Learning](#https://arxiv.org/abs/2102.05918), 2021 - Google's counterpart to CLIP, trained on 1.8B noisy image-text pairs using a simpler loss. [Code](#https://github.com/kakaobrain/align-rudolph)
  - [SigLIP: Sigmoid Loss for Language Image Pre-Training](#https://arxiv.org/abs/2303.15343), 2023 - Replaces contrastive softmax loss with sigmoid loss, enabling batch-size-independent training and stronger performance at smaller scales. [Code](#https://github.com/google-research/big_vision)
  - [BLIP-2: Bootstrapping Language-Image Pre-training](#https://arxiv.org/abs/2301.12597), 2023 - Bridges frozen image encoders and frozen LLMs with a lightweight Q-Former module. Efficient multimodal pretraining without end-to-end training. [Code](#https://github.com/salesforce/lavis)

---

### Document & UI Understanding
A critical application area for IS research: extracting structured information from forms, interfaces, tables, and mixed-layout documents.

- **Papers with code:**
  - [LayoutLMv3: Pre-Training for Document AI with Unified Text and Image Masking](#https://arxiv.org/abs/2204.08387), 2022 - Jointly models text, layout (bounding boxes), and image patches for document understanding. State-of-the-art on form understanding, receipt processing, and document QA. [Code](#https://github.com/microsoft/unilm/tree/master/layoutlmv3)
  - [Donut: Document Understanding Transformer](#https://arxiv.org/abs/2111.15664), 2022 - End-to-end document parsing without OCR. Reads documents as images and outputs structured JSON. [Code](#https://github.com/clovaai/donut)
  - [ScreenSpot: Vision-Language Model-based GUI Grounding](#https://arxiv.org/abs/2501.12730), 2025 - Evaluates VLMs on localizing UI elements from natural language descriptions. Directly applicable to IS research on human-computer interaction and usability. [Code](#https://github.com/njucckevin/seeclick)

- **Tools:**
  - [Surya](#https://github.com/vikparuchuri/surya) - Fast, accurate OCR and document layout analysis. Handles tables, columns, equations, and mixed languages.
  - [Marker](#https://github.com/vikparuchuri/marker) - Converts PDFs (including scanned documents) to clean Markdown using VLMs. Useful for processing academic papers and reports at scale.

---

### Multimodal Generation
Models that generate images, video, or interleaved image-text content from text prompts. See [Generative Media & Synthetic Data](../DataGeneration/README.md) for the full treatment.

- **Key Models:**
  - [Stable Diffusion / SDXL](../DataGeneration/README.md#Diffusion-Models) - Text-to-image generation.
  - [DALL-E 3](#https://openai.com/dall-e-3) - Integrated into ChatGPT for high-fidelity image generation.
  - [Sora](../DataGeneration/README.md#Text-to-Video) - Text-to-video generation.

- **Interleaved Generation:**
  - [Flamingo: a Visual Language Model for Few-Shot Learning](#https://arxiv.org/abs/2204.14198), 2022 - Processes arbitrarily interleaved images and text, enabling few-shot visual QA and captioning. [Code](#https://github.com/mlfoundations/open_flamingo)
  - [GPT-4o Image Generation](#https://openai.com/index/hello-gpt-4o/) (2024) - Native multimodal generation combining understanding and synthesis in a single model.

---

### Benchmarks & Evaluation

| **Benchmark** | **Description** |
|-|-|
| [MMBench](#https://github.com/open-compass/mmbench) | Comprehensive VLM evaluation across 20 capability dimensions including reasoning, OCR, and spatial understanding. |
| [MMMU](#https://mmmu-benchmark.github.io/) | Massive Multi-discipline Multimodal Understanding — college-level questions requiring image + text reasoning across 30 subjects. |
| [DocVQA](#https://www.docvqa.org/) | Document Visual Question Answering — questions about scanned document images. Relevant for IS document processing research. |
| [TextVQA](#https://textvqa.org/) | Questions requiring reading and reasoning about text within images. |
| [ScienceQA](#https://scienceqa.github.io/) | Multi-modal science questions with explanations. |

---

### Tools & Libraries

| **Tool** | **Description** |
|-|-|
| [Transformers (HuggingFace)](#https://huggingface.co/docs/transformers/index) | Unified API for loading and running all major open-weight VLMs (LLaVA, InternVL, Qwen-VL, etc.). |
| [LAVIS](#https://github.com/salesforce/lavis) | Salesforce's library for vision-language research. Includes BLIP, BLIP-2, InstructBLIP, and evaluation pipelines. |
| [OpenCLIP](#https://github.com/mlfoundations/open_clip) | Open-source reproduction of CLIP with additional model variants trained on LAION datasets. |
| [LLaVA-Med](#https://github.com/microsoft/llava-med) | Medical domain-adapted LLaVA. Example of domain-specific VLM fine-tuning applicable to IS healthcare research. |
| [vLLM](#https://github.com/vllm-project/vllm) | High-throughput inference engine for LLMs and VLMs. Enables efficient batch processing of images+text at scale. |

---

### Audio-Language Models
Models that process audio (speech, music, environmental sounds) alongside text, enabling transcription, understanding, and generation.

- **Models:**
  - [Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](#https://arxiv.org/abs/2212.04356), 2022 - OpenAI's state-of-the-art speech-to-text model trained on 680K hours of multilingual audio. Near-human accuracy on clean speech; strong on accented speech. Essential for IS research transcribing interviews. [Code](#https://github.com/openai/whisper)
  - [Qwen-Audio: Advancing Universal Audio Understanding via Unified Large-Scale Audio-Language Models](#https://arxiv.org/abs/2311.07919), 2023 - Extends LLMs to audio reasoning across speech, sound, and music. [Code](#https://github.com/qwenlm/qwen-audio)
  - [Gemini 1.5 Pro Native Audio](#https://deepmind.google/technologies/gemini/), 2024 - Natively processes audio input alongside text and images in a single model call — enabling transcription, sentiment analysis, and content understanding simultaneously.

- **IS Research Applications:**
  - Automated transcription and thematic coding of qualitative interviews at scale
  - Sentiment and tone analysis of earnings calls, customer service recordings, and focus groups
  - Creating multimodal research instruments that combine spoken instructions with visual stimuli

---

### Video Understanding
Models that process video as sequences of frames with temporal context — distinct from text-to-video *generation*.

- **Papers with code:**
  - [VideoLLaMA 2: Advancing Spatial-Temporal Modeling and Audio Understanding in Video-LLMs](#https://arxiv.org/abs/2406.07476), 2024 - Unified video + audio + text understanding model. Handles temporal reasoning over video content. [Code](#https://github.com/damo-nlp-sg/videollama2)
  - [Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models](#https://arxiv.org/abs/2306.05424), 2023 - Adapts LLaVA to video via temporal pooling of frame features. [Code](#https://github.com/mbzuai-oryx/video-chatgpt)

- **Tools:**
  - [Twelve Labs](#https://twelvelabs.io/) - Video understanding API that enables semantic search, summarization, and Q&A over video content. Relevant for IS research analyzing instructional or interface videos.

---

**Related Sections:** [LLMs & NLP](../NaturalLanguageProcessing/README.md) | [Generative Media & Synthetic Data](../DataGeneration/README.md) | [Fine-Tuning](../FineTuning/README.md) | [Interpretability](../Interpretability/README.md)
