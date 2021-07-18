# Data Generation
*Papers and Repositories For Image, Textual, and Other Generation Tasks.*

[Seminal Paper](https://arxiv.org/pdf/1406.2661.pdf), 2014

| | | |
|-|-|-|
| [Image Generation](#Image-Generation) | [Text Generation](#Data-Generation) | [Other Generation](#Other-Generation)

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
