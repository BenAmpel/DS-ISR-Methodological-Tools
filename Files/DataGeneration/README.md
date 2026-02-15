# Generative Media & Synthetic Data
*Papers and Repositories for Image, Video, Text, and Synthetic Data Generation.*

*Last updated: February 2026*

> **Note:** As of 2022, diffusion models have largely superseded GANs for image and video generation tasks due to superior image quality, training stability, and diversity. GAN literature remains relevant for tabular and time-series generation and for understanding the generative modeling landscape.

| | | |
|-|-|-|
| [Diffusion Models](#Diffusion-Models) | [Image Generation (GANs)](#Image-Generation-GANs) | [Text Generation](#Text-Generation) |
| [Text-to-Video](#Text-to-Video) | [Synthetic Users & Generative Agents](#Synthetic-Users--Generative-Agents) | [Research Dissemination Tools](#Research-Dissemination-Tools) |
| [Other Generation](#Other-Generation) | | |

---

> **IS Research Applications:** Generate experimental stimuli (images, videos) for behavioral IS studies without filming budgets; create synthetic survey respondents and user personas for experimental design; produce research dissemination videos from paper abstracts; generate realistic fake data for system testing; create visual vignettes for technology acceptance experiments.

---

### Diffusion Models
Diffusion models iteratively denoise a random signal to generate high-fidelity images, audio, and video. They have become the foundation of modern generative AI for visual content.

[Seminal Paper: Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239), 2020 - [Code](https://github.com/hojonathanho/diffusion)

- **Image Diffusion:**
  - [Stable Diffusion (Latent Diffusion Models)](https://arxiv.org/abs/2112.10752), 2022 - Runs diffusion in a compressed latent space rather than pixel space, dramatically reducing compute requirements while maintaining quality. The most widely deployed open-source image generation model. [Code](https://github.com/CompVis/stable-diffusion)
  - [SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952), 2023 - An enhanced Stable Diffusion architecture producing photorealistic images at 1024x1024 resolution. [Code](https://github.com/Stability-AI/generative-models)
  - [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598), 2022 - The technique enabling text-conditional control of diffusion models (used in DALL-E 2, Imagen, Stable Diffusion). [Code](https://github.com/lucidrains/classifier-free-guidance-pytorch)

- **Commercial Tools:**
  - [Midjourney](https://www.midjourney.com/) - High-quality aesthetic image generation via Discord/web interface. Widely used for creating research figures, stimuli, and presentation visuals.
  - [DALL-E 3](https://openai.com/dall-e-3) (OpenAI, 2023) - Integrated into ChatGPT; excels at following detailed text prompts with strong text rendering.
  - [Adobe Firefly](https://www.adobe.com/products/firefly.html) - Commercially safe diffusion model integrated into Adobe Creative Suite.

---

### Text-to-Video
Text-to-video models generate video clips from natural language descriptions, enabling creation of research stimuli, experimental vignettes, and instructional content without filming.

- **Models:**
  - [Sora](https://openai.com/sora) (OpenAI, 2024) - Generates up to 60-second HD videos from text prompts. Uses a diffusion transformer (DiT) architecture operating on video patches. Represents a significant step toward physical world simulation.
  - [Runway Gen-3 Alpha](https://runwayml.com/) (2024) - Production-grade text-to-video and image-to-video generation. Widely used for research video production.
  - [CogVideoX](https://arxiv.org/abs/2408.06072), 2024 - Open-weight text-to-video model. [Code](https://github.com/THUDM/CogVideo)

- **IS Research Applications:**
  - Generate experimental vignettes (e.g., showing a user interface interaction) for behavioral IS studies.
  - Create instructional videos for technology acceptance or training studies.
  - Produce research stimuli at scale without actor recruitment or filming budgets.

---

### Image Generation (GANs)
[Seminal Paper: Generative Adversarial Networks](https://arxiv.org/pdf/1406.2661.pdf), 2014

- **Papers with code:**
  - [Training Generative Adversarial Networks from Incomplete Observations using Factorised Discriminators](https://arxiv.org/pdf/1905.12660.pdf), 2020 - Splits the discriminator into parts independently trainable with incomplete observations. [Code](https://www.dropbox.com/s/gtc7m7pc4n2yt05/source.zip?dl=1)
  - [On the "steerability" of generative adversarial networks](https://arxiv.org/pdf/1907.07171.pdf), 2020 - Shifts GAN distributions to improve generalizability. [Code](https://ali-design.github.io/gan_steerability/)
  - [Controlling generative models with continuous factors of variations](https://arxiv.org/pdf/2001.10238.pdf), 2020 - Control object position in a generated image. [Code](https://github.com/AntoinePlumerault/Controlling-generative-models-with-continuous-factors-of-variations)

- **Papers without code:**
  - [Stochastic Conditional Generative Networks with Basis Decomposition](https://arxiv.org/pdf/1909.11286.pdf), 2019 - Generate improved images for multi-mode datasets.

---

### Text Generation
- **Papers with code:**
  - [Neural Text Generation With Unlikelihood Training](https://arxiv.org/pdf/1908.04319.pdf), 2020 - Forces unlikely generations to be assigned lower probability on transformers. [Code](https://github.com/facebookresearch/unlikelihood_training)
  - [Language GANs Falling Short](https://arxiv.org/pdf/1811.02549.pdf), 2020 - Quality-diversity evaluation procedure to reduce softmax temperature. [Code](https://github.com/pclucas14/GansFallingShort)
  - [The Curious Case of Neural Text Degeneration](https://arxiv.org/pdf/1904.09751.pdf), 2020 - Nucleus sampling for high-quality text from neural language models. [Code](https://github.com/ari-holtzman/degen)
  - [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/pdf/1904.09675.pdf), 2020 - Evaluative metric using contextual similarity scores. [Code](https://github.com/Tiiiger/bert_score)
  - [Plug and Play Language Models: A Simple Approach to Controlled Text Generation](https://arxiv.org/pdf/1912.02164.pdf), 2020 - Combines pretrained transformer LM with attribute classifiers to guide generation. [Code](https://github.com/uber-research/PPLM)

- **Papers without code:**
  - [Augmenting Non-Collaborative Dialog Systems with Explicit Semantic and Strategic Dialog History](https://arxiv.org/pdf/1909.13425.pdf), 2020 - Finite state transducers to explicitly represent dialog history.
  - [Self-Adversarial Learning with Comparative Discrimination for Text Generation](https://arxiv.org/pdf/2001.11691.pdf), 2020 - Self-improvement reward mechanism to avoid model collapse.

---

### Synthetic Users & Generative Agents
Generative agents simulate human users computationally, enabling IS researchers to run large-scale experiments with synthetic participants, stress-test system designs, and model organizational behavior.

- **Papers with code:**
  - [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442), 2023 - LLM-powered agents with memory, reflection, and planning that simulate human behavior in a virtual social environment. [Code](https://github.com/joonspk-research/generative_agents)
  - [Out of One, Many: Using Language Models to Simulate Human Samples](https://arxiv.org/abs/2209.06899), 2022 - Demonstrates that LLMs can simulate human survey responses conditioned on demographic profiles ("silicon sampling"). [Code](https://github.com/tobynorris/silicon-sampling)

- **Tools:**
  - [AgentSims](https://github.com/py499372727/AgentSims) - A simulation sandbox for testing LLM-based agents in customizable social environments.

---

### Research Dissemination Tools
- [Synthesia](https://www.synthesia.io/) - Generate professional avatar-based explainer videos from text scripts. Useful for recording research presentations and dissemination without on-camera recording.
- [Lumen5](https://lumen5.com/) - Converts text (blog posts, abstracts) to short video summaries automatically. Useful for social media research dissemination.
- [HeyGen](https://www.heygen.com/) - AI video generation with custom avatars, useful for creating multilingual research presentations.

---

### Other Generation
- **Papers with code:**
  - [A Closer Look at the Optimization Landscapes of Generative Adversarial Networks](https://arxiv.org/pdf/1906.04848.pdf), 2020 - GANs exhibit rotations around Local Stable Stationary Points (LSSP). [Code](https://github.com/facebookresearch/GAN-optimization-landscape)
  - [Curb-GAN: Conditional Urban Traffic Estimation through Spatio-Temporal Generative Adversarial Networks](https://dl.acm.org/doi/10.1145/3394486.3403127), 2020 - Traffic estimations based on unprecedented travel demands. [Code](https://github.com/Curb-GAN/Curb-GAN)
  - [SEAL: Learning Heuristics for Community Detection with Generative Adversarial Networks](https://dl.acm.org/doi/abs/10.1145/3394486.3403154), 2020 - Predicts whether a community is real or fake for enhanced detection. [Code](https://github.com/yzhang1918/kdd2020seal)

- **Papers without code:**
  - [Smoothness and Stability in GANs](https://arxiv.org/pdf/2002.04185.pdf), 2020 - Hyperparameter tuning to improve GAN smoothness and stability.
  - [Deep State-Space Generative Model For Correlated Time-to-Event Predictions](https://dl.acm.org/doi/pdf/10.1145/3394486.3403206), 2020 - Discrete-time hazard rate formulation to estimate patient survival distributions.
  - [Catalysis Clustering With GAN By Incorporating Domain Knowledge](https://dl.acm.org/doi/10.1145/3394486.3403187), 2020 - Unsupervised clusters based on domain-defined rules.

---

### Audio Generation
AI-generated speech and music enable creation of research stimuli, narration for video abstracts, and synthetic interview data.

- **Models & Tools:**
  - [ElevenLabs](https://elevenlabs.io/) - State-of-the-art voice cloning and text-to-speech. Widely used for research narration and creating diverse synthetic interview stimuli with consistent personas.
  - [OpenAI TTS](https://platform.openai.com/docs/guides/text-to-speech) - High-quality text-to-speech API with 6 voice variants. Cost-effective for generating spoken research stimuli.
  - [Bark](https://github.com/suno-ai/bark) - Open-source generative audio model capable of realistic speech, music, and sound effects. Runs locally. [Code](https://github.com/suno-ai/bark)
  - [Stable Audio](https://stability.ai/stable-audio) - Diffusion-based music and sound generation from text prompts.

- **Papers with code:**
  - [VoiceBox: Text-Guided Multilingual Universal Speech Generation at Scale](https://arxiv.org/abs/2306.15687), 2023 - Meta's flow-matching-based TTS model with strong zero-shot voice cloning and style transfer. [Code](https://voicebox.metademolab.com/)
  - [NaturalSpeech 3: Zero-Shot Lifelike Speech Synthesis](https://arxiv.org/abs/2403.03100), 2024 - Microsoft's factorized codec + diffusion approach achieving near-human prosody and expressiveness.

---

### 3D & Spatial Generation
Emerging tools for generating 3D objects and environments — relevant for IS research on virtual spaces, digital twins, and immersive technology studies.

- **Papers with code:**
  - [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934), 2020 - Foundational method for learning 3D scene representations from 2D images. Enables rendering novel viewpoints from photograph collections. [Code](https://github.com/bmild/nerf)
  - [Instant NGP: Instant Neural Graphics Primitives](https://arxiv.org/abs/2201.05989), 2022 - Reduces NeRF training from hours to seconds via multiresolution hash encoding. [Code](https://github.com/NVlabs/instant-ngp)
  - [Gaussian Splatting: 3D Gaussian Splatting for Real-Time Novel View Synthesis](https://arxiv.org/abs/2308.04079), 2023 - Represents scenes as explicit 3D Gaussians instead of implicit neural networks, enabling real-time rendering. Rapidly replacing NeRF in practice. [Code](https://github.com/graphdeco-inria/gaussian-splatting)

---

### Generative AI Evaluation Metrics
Assessing the quality of generative outputs is as methodologically important as the generation itself.

| **Metric** | **Modality** | **Measures** | **Reference** |
|-|-|-|-|
| FID (Fréchet Inception Distance) | Image | Distribution similarity between real and generated images | [Paper](https://arxiv.org/abs/1706.08500) |
| CLIP Score | Image+Text | Alignment between generated image and text prompt | [Paper](https://arxiv.org/abs/2104.08718) |
| FVD (Fréchet Video Distance) | Video | Temporal quality of generated video | [Paper](https://arxiv.org/abs/1812.01717) |
| BLEU / ROUGE | Text | N-gram overlap between generated and reference text | Standard NLP metrics |
| BERTScore | Text | Semantic similarity via contextual BERT embeddings | [Paper](https://arxiv.org/abs/1904.09675) |
| MOS (Mean Opinion Score) | Audio | Human-rated perceptual quality of synthesized speech | ITU-T P.800 standard |

---

**Related Sections:** [Multimodal Models](../MultimodalModels/README.md) | [LLMs & NLP](../NaturalLanguageProcessing/README.md) | [Reinforcement Learning](../ReinforcementLearning/README.md) | [AI for Research Productivity](../AI-for-Research-Productivity/README.md) [Multimodal Models](../MultimodalModels/README.md) | [LLMs & NLP](../NaturalLanguageProcessing/README.md) | [Reinforcement Learning](../ReinforcementLearning/README.md) | [AI for Research Productivity](../AI-for-Research-Productivity/README.md)
