---
layout: default
title: "AI for Research Productivity"
parent: Topics
nav_order: 1
permalink: /Files/AI-for-Research-Productivity/
---


{::nomarkdown}
<div class="section-meta">
  <span class="meta-pill meta-reading">⏱ ~10 min read</span>
  <span class="meta-pill meta-updated">📅 Updated Feb 2026</span>
  <span class="meta-pill meta-beginner">📊 Beginner</span>
  <span class="meta-pill meta-prereq">🔑 None</span>
</div>
{:/}

# AI for Research Productivity
*Tools for accelerating the research workflow: literature discovery, coding assistance, writing, and data analysis.*

*Last updated: February 2026*

> *This section maps to **Level I (Copy Editor)** and **Level II (Research Assistant)** of the Automation-of-Invention Framework (ISR, 2025).*

| | | |
|-|-|-|
| [Literature Review](#literature-review) | [AI-Assisted Coding](#ai-assisted-coding) | [Writing & Editing](#writing-editing) |
| [Data Analysis & Visualization](#data-analysis-visualization) | [Research Organization](#research-organization) | [Deep Research Agents](#deep-research--autonomous-literature-agents) |
| [AI Video & Lecture Tools](#ai-video--lecture-tools) | [Statistical Analysis](#ai-assisted-statistical-analysis) | [Qualitative Analysis](#ai-for-qualitative-analysis) |

> ⭐ **Open-source** tools are marked throughout this section — prioritise these to reduce costs and keep data local.

---

### Literature Review
Tools that accelerate systematic and scoping literature reviews by discovering, summarizing, and mapping papers.

| **Tool** | **Description** | **Best For** |
|-|-|-|
| [Elicit](https://elicit.com/) | LLM-powered research assistant that searches semantic scholar, extracts key findings from papers, and builds comparison tables. Added Notebooks (2024) for structured evidence synthesis across many papers. | Structured literature synthesis, extracting specific variables from many papers |
| [Research Rabbit](https://www.researchrabbit.ai/) | Visual citation network mapper. Upload seed papers and discover related work through forward/backward citation graphs. Added author network exploration and team-shareable Collections. | Building comprehensive literature maps, finding seminal papers |
| [Connected Papers](https://www.connectedpapers.com/) | Generates a visual graph of papers related to a seed paper based on co-citation and bibliographic coupling. | Rapid field orientation, identifying clusters of related work |
| [Semantic Scholar](https://www.semanticscholar.org/) | AI-powered search engine with TLDR summaries (via SPECTER2 embeddings) for 200M+ papers, citation counts, and influence metrics. | Discovering high-impact papers in a domain |
| [Perplexity](https://www.perplexity.ai/) | Web-search-backed LLM that provides cited answers to research questions. | Quick factual lookups, initial literature scouting |
| [Consensus](https://consensus.app/) | Searches empirical research and synthesizes findings across studies. Added study quality filters and expanded coverage into CS/physics. | Finding what research says about a specific question |
| [OpenScholar](https://github.com/AkariAsai/OpenScholar) ⭐ open-source | Retrieval-augmented LLM from the Allen AI / UW team that retrieves from a 45M+ paper datastore and generates grounded literature synthesis with inline citations. Open-source models match GPT-4 on literature review tasks. | Fully reproducible, citeable literature synthesis; privacy-sensitive research |
| [Khoj](https://github.com/khoj-ai/khoj) ⭐ open-source | Self-hostable personal AI assistant that indexes your PDFs, notes, and documents for semantic search and Q&A. Supports Obsidian and web UI. Runs with local models via Ollama. | Querying your own paper library without sending data to the cloud |
| [AnythingLLM](https://github.com/Mintplex-Labs/anything-llm) ⭐ open-source | Full-stack self-hosted RAG application. Ingest PDFs and papers into a local vector store; query with any LLM (Ollama, OpenAI, Claude). Multi-user and workspace support. | Building a private, shareable research knowledge base for a lab or project |

> **Key Papers:**
> - Asai et al. (2024). *OpenScholar: Synthesizing Scientific Literature with Retrieval-Augmented LMs.* [arXiv:2404.01869](https://arxiv.org/abs/2404.01869) — shows open models can match GPT-4 for literature review via retrieval.
> - Wang et al. (2024). *AutoSurvey: Large Language Models Can Automatically Write Related Work.* [arXiv:2406.10252](https://arxiv.org/abs/2406.10252) — benchmarks LLM pipelines for generating survey sections.

---

### AI-Assisted Coding
AI coding assistants dramatically accelerate the implementation of DSR artifacts, data pipelines, and statistical analyses.

| **Tool** | **Description** | **Best For** |
|-|-|-|
| [GitHub Copilot](https://github.com/features/copilot) | In-editor AI code completion and chat (VSCode, JetBrains, etc.). Added Copilot Workspace (2024) for multi-file, task-level planning; extended to CLI. | Day-to-day coding, boilerplate generation, code explanation |
| [Cursor](https://cursor.sh/) | AI-native code editor with multi-file context, codebase chat, and autonomous code editing. Added Composer mode (2024) for large multi-file edits and agent-style planning. | Larger refactoring tasks, understanding unfamiliar codebases |
| [Cline](https://github.com/cline/cline) (formerly Claude Dev) ⭐ open-source | Autonomous AI agent in VSCode that can read/write files, run terminal commands, and browse the web. Added MCP (Model Context Protocol) server support, enabling file system access and browser control. | Complex multi-step development tasks with oversight |
| [Claude Code](https://claude.ai/code) | Anthropic's terminal-based agentic coding tool. Can read, edit, and run code across an entire repository. | Research scripting, data analysis pipelines, rapid prototyping |
| [Replit Agent](https://replit.com/ai) | Full-stack AI coding in the browser with instant deployment. No local setup required. | Quick DSR prototype deployment, sharing running demos |
| [Continue.dev](https://github.com/continuedev/continue) ⭐ open-source | The leading open-source AI code assistant plugin for VS Code and JetBrains. Supports any LLM — local (Ollama) or cloud. Autocomplete, chat, and multi-file edit modes. | Cost-free, privacy-preserving AI coding with local models |
| [Aider](https://github.com/paul-gauthier/aider) ⭐ open-source | CLI pair-programming tool that edits source files directly using git-aware context. Works with GPT-4, Claude, Gemini, and local models. Top performer on SWE-bench. | Automated code experiments, scripting research pipelines from the terminal |
| [Tabby](https://github.com/TabbyML/tabby) ⭐ open-source | Self-hosted GitHub Copilot alternative with a local inference server. Supports fine-tuning on private codebases. | Institutions needing on-premises code completion without sending code to external APIs |

> **Key Papers:**
> - Jimenez et al. (2024). *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR 2024. [arXiv:2310.06770](https://arxiv.org/abs/2310.06770) — the definitive benchmark for AI coding agents.
> - Wang et al. (2024). *OpenDevin: An Open Platform for AI Software Developers as Generalist Agents.* [arXiv:2407.16741](https://arxiv.org/abs/2407.16741) — describes the open-source OpenHands platform for autonomous coding agents.

---

### Writing & Editing
AI writing tools assist with drafting, translating, paraphrasing, and polishing academic text.

| **Tool** | **Description** | **Best For** |
|-|-|-|
| [DeepL Write](https://www.deepl.com/write) | High-quality AI paraphrasing and translation. Best-in-class for academic tone and accuracy. | Refining phrasing, translating between languages while preserving academic register |
| [GrammarlyGO](https://www.grammarly.com/) | AI writing assistant with grammar checking, tone adjustment, and generative suggestions. Context-aware of document-level tone and discipline-specific vocabulary. | Copyediting, tone consistency, academic style |
| [Lex](https://lex.page/) | AI-native document editor with in-line generation, continuation, and brainstorming. | Drafting sections, overcoming writer's block |
| [Writefull](https://writefull.com/) | Academic language feedback trained on published papers. Integrates with Overleaf. Added Title Generator and Abstract Generator fine-tuned on 200M+ published papers. | Checking academic phrasing in LaTeX manuscripts |
| [Paperpal](https://paperpal.com/) | AI academic writing assistant specifically trained on scientific literature. Added Manuscript Check for validating journal submission requirements; deeper Overleaf integration. | Technical language suggestions, journal submission preparation |
| [LanguageTool](https://github.com/languagetool-org/languagetool) ⭐ open-source | Self-hostable grammar, style, and spell checker supporting 30+ languages. Integrates with VS Code and Overleaf via browser extension. Community edition is fully free. | Privacy-preserving grammar checking without cloud dependency; non-English academic writing |
| [Ollama](https://github.com/ollama/ollama) ⭐ open-source | Run Llama 3, Mistral, Gemma, and other LLMs locally with a single command. Standard infrastructure in 2024 for private, institution-safe LLM writing and editing assistance. | Institutions where manuscripts cannot leave local systems; zero-cost LLM writing workflows |
| [Quivr](https://github.com/QuivrHQ/quivr) ⭐ open-source | Open-source RAG platform — upload your paper corpus, then generate AI-assisted writing grounded in your own documents. Self-hostable. | Writing related-work or discussion sections that stay grounded in a specific literature set |

> **Key Papers:**
> - Liang et al. (2024). *Can Large Language Models Provide Useful Feedback on Research Papers?* [arXiv:2404.01268](https://arxiv.org/abs/2404.01268) — comprehensive evaluation of LLM feedback quality on academic drafts.
> - D'Arcy et al. (2024). *MARG: Multi-Agent Review Generation for Scientific Papers.* [arXiv:2401.04259](https://arxiv.org/abs/2401.04259) — multi-agent pipeline with specialized agents critiquing methods, related work, and novelty.

---

### Data Analysis & Visualization
AI-assisted tools for exploratory data analysis, statistical testing, and figure generation.

| **Tool** | **Description** | **Best For** |
|-|-|-|
| [Julius AI](https://julius.ai/) | Upload datasets (CSV, Excel) and ask natural language questions; generates Python/R code and charts. Added persistent Projects memory and multi-dataset joins via natural language. | Rapid EDA without coding, sharing analyses with non-technical collaborators |
| [ChatGPT Advanced Data Analysis](https://chat.openai.com/) | Code interpreter that runs Python in a sandboxed environment for data analysis and figure generation. | Statistical analysis, figure generation from uploaded data |
| [Streamlit](https://streamlit.io/) | Converts Python scripts into interactive web apps in minutes. Added `st.chat_message` and `st.data_editor` (2024) making LLM-powered data apps significantly easier to build. | Deploying DSR artifact prototypes, interactive research dashboards |
| [Plotly / Dash](https://plotly.com/) | Interactive chart library (Plotly) and web app framework (Dash) for Python. Dash 2.17+ added AI-assisted callbacks and AG Grid for large datasets. | Publication-quality interactive figures, research dashboards |
| [PandasAI](https://github.com/Sinaptik-AI/pandas-ai) ⭐ open-source | Adds a natural language interface directly to Pandas DataFrames. Ask questions in plain English; get executable Python code and charts in return. Supports local LLMs. | In-notebook conversational data analysis; reproducible EDA with auditable code output |
| [LIDA](https://github.com/microsoft/lida) ⭐ open-source | Microsoft Research tool for automatic generation of data visualizations from natural language goals. Generates, executes, and iteratively refines chart code. | Rapidly prototyping visualizations; exploring which chart type best answers a research question |
| [PyGWalker](https://github.com/Kanaries/pygwalker) ⭐ open-source | Turns Pandas DataFrames into a Tableau-style drag-and-drop visual analysis interface inside Jupyter notebooks. Added AI-assisted chart recommendations in 2024. | Interactive exploratory analysis within a familiar notebook environment |

> **Key Papers:**
> - Dibia (2023). *LIDA: A Tool for Automatic Generation of Grammar-Agnostic Visualizations and Infographics using Large Language Models.* [arXiv:2303.02927](https://arxiv.org/abs/2303.02927) — describes the LIDA system for LLM-driven visualization generation.
> - Zhang et al. (2024). *Data-Copilot: Bridging Billions of Data and Humans with Autonomous Workflow.* [arXiv:2306.07209](https://arxiv.org/abs/2306.07209) — LLM agent for autonomous data retrieval, analysis, and visualization across heterogeneous sources.

---

### Research Organization
Tools for managing notes, references, and knowledge bases.

| **Tool** | **Description** | **Best For** |
|-|-|-|
| [Zotero](https://www.zotero.org/) ⭐ open-source | Open-source reference manager with browser extension and Word/LaTeX integration. Zotero 7 (2024) brought a complete UI rewrite and improved PDF reader. | Citation management, PDF organization, shared group libraries |
| [Obsidian](https://obsidian.md/) | Local-first markdown note-taking with bidirectional links and graph view. The "Smart Connections" and "Copilot" community plugins add LLM Q&A over your vault. | Synthesis notes, connecting ideas across papers |
| [Notion AI](https://www.notion.so/product/ai) | AI-enhanced workspace for notes, project management, and databases. Added workspace-level Q&A, AI database views, and auto-fill properties. | Lab wikis, project tracking, collaborative documentation |
| [Logseq](https://logseq.com/) ⭐ open-source | Open-source outliner with bidirectional links; privacy-focused Obsidian alternative. Database version (SQLite backend) in beta 2024 enables structured queries on large graphs. | Personal research journals, networked literature notes |
| [ZoteroGPT](https://github.com/MuiseDestiny/zotero-gpt) ⭐ open-source | Zotero plugin adding LLM-powered Q&A over your entire Zotero library. Supports OpenAI, Claude, and local models. Built on the new Zotero 7 plugin API. | Conversational search across your personal paper collection |
| [Paperlib](https://github.com/Future-Scholars/paperlib) ⭐ open-source | Open-source academic paper management with metadata scraping, smart folders, and AI-assisted paper search and recommendations. A modern, researcher-focused Zotero alternative. | Researchers wanting a lighter, more opinionated paper manager with built-in AI recommendations |
| [SiYuan](https://github.com/siyuan-note/siyuan) ⭐ open-source | Local-first, self-hostable PKM with block-level references, database views, and PDF annotation. Rapidly growing in 2024 with strong international community. | Structured research notes with database-style organization; self-hosted lab knowledge bases |

> **Key Papers:**
> - Edge et al. (2024). *From Local to Global: A Graph RAG Approach to Query-Focused Summarization.* [arXiv:2404.16130](https://arxiv.org/abs/2404.16130) — foundational paper for GraphRAG, directly applicable to building LLM-searchable PKM knowledge graphs over large paper collections.

---

### Deep Research & Autonomous Literature Agents
A new category of tools (emerging 2024–2025) that can autonomously conduct multi-step research tasks — searching, reading, synthesizing, and reporting — with minimal human guidance.

| **Tool** | **Description** | **Best For** |
|-|-|-|
| [Google Gemini Deep Research](https://gemini.google.com/app) | Gemini-powered agent that autonomously conducts web research over 5–30 minutes, producing structured multi-page reports with citations. 1M token context window enables whole-corpus reasoning. | Initial literature surveys, competitive landscape analysis |
| [NotebookLM](https://notebooklm.google.com/) | Google's AI notebook grounding all responses in your uploaded documents. Added Audio Overview (podcast-style synthesis), mind-map generation, and expanded source limit to 50 sources per notebook (2024). | Processing large paper collections; creating podcast-style summaries of your own papers |
| [OpenAI Deep Research](https://openai.com/index/introducing-deep-research/) | o3-based agent that autonomously browses the web for hours, synthesizes findings, and produces detailed research reports. Strong on technical topics. | Complex multi-step research questions requiring deep web synthesis |
| [Perplexity Deep Research](https://www.perplexity.ai/hub/blog/introducing-deep-research) | Multi-step research agent (launched early 2025) running dozens of sub-queries and synthesizing structured reports with inline citations. | Rapid factual research with full source attribution |
| [GPT-Researcher](https://github.com/assafelovic/gpt-researcher) ⭐ open-source | The most prominent open-source autonomous research agent (10k+ GitHub stars). Conducts multi-step web research, synthesizes findings, and outputs a structured report with citations. Supports local LLMs. | Self-hosted deep research without commercial API costs; fully reproducible research reports |
| [STORM](https://github.com/stanford-oval/storm) ⭐ open-source | Stanford system that autonomously writes Wikipedia-style articles on any topic by simulating expert conversations and multi-perspective question asking. Produces structured outlines before writing. | Generating comprehensive literature survey drafts; exploring a new topic systematically |
| [AutoGen](https://github.com/microsoft/autogen) ⭐ open-source | Microsoft multi-agent framework enabling research agents where specialized sub-agents handle search, summarization, critique, and synthesis. Version 0.4 (late 2024) brought major architectural improvements. | Building custom multi-agent research pipelines; automating iterative literature synthesis |
| [CrewAI](https://github.com/crewAIInc/crewAI) ⭐ open-source | Framework for orchestrating teams of AI agents with defined roles (researcher, writer, critic). One of the most popular frameworks in 2024 for building custom autonomous literature agents. | Rapid prototyping of custom multi-agent research workflows |

> **Key Papers:**
> - Shao et al. (2024). *Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models (STORM).* [arXiv:2402.14207](https://arxiv.org/abs/2402.14207) — introduces the STORM pipeline for autonomous survey generation via multi-perspective question asking.
> - Zhuge et al. (2024). *Agent-as-a-Judge: Evaluate Agents with Agents.* [arXiv:2410.10934](https://arxiv.org/abs/2410.10934) — addresses reliable evaluation of autonomous research agents, a critical gap for deployment.

---

### AI Video & Lecture Tools
Tools for processing, summarizing, and creating educational content from video lectures and research talks.

| **Tool** | **Description** | **Best For** |
|-|-|-|
| [Otter.ai](https://otter.ai/) | Real-time transcription and AI meeting notes with speaker identification. Added AI "Channels" for organized meeting summaries and real-time action item extraction. | Transcribing qualitative interviews, conference talks, and focus groups |
| [Tactiq](https://tactiq.io/) | AI meeting transcription with GPT-4-powered summary templates and CRM integrations. | Lab meetings, research seminars, advisor sessions |
| [Descript](https://www.descript.com/) | Video/audio editor where you edit by editing the transcript text. Added AI voice cloning ("Overdub"), filler-word removal, and eye contact correction in 2024. | Editing research presentation recordings and video abstracts |
| [VideoAsk](https://www.videoask.com/) | Async video-based survey/interview tool. Useful for IS research collecting video responses from participants. | Asynchronous qualitative data collection |
| [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) ⭐ open-source | 4× faster reimplementation of OpenAI Whisper using CTranslate2. The standard 2024 transcription backend for research pipelines — processes a 1-hour lecture in ~2 minutes on CPU. | Fast, local, free transcription of interviews, lectures, and focus group recordings |
| [Buzz](https://github.com/chidiwilliams/buzz) ⭐ open-source | Cross-platform desktop app for transcribing and translating audio/video using Whisper models locally. No data leaves your machine. Simple GUI suitable for non-technical researchers. | Transcribing sensitive interviews locally without any cloud service |
| [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) ⭐ open-source | C++ port of Whisper enabling real-time transcription on CPU and Apple Silicon (Metal). Minimal dependencies; easily embedded in research automation scripts. | Integrating transcription into custom research pipelines; low-latency real-time transcription |

> **Key Papers:**
> - Radford et al. (2023). *Robust Speech Recognition via Large-Scale Weak Supervision (Whisper).* [arXiv:2212.04356](https://arxiv.org/abs/2212.04356) — foundational paper for all open-source lecture transcription tools active in 2024-2025.
> - Zhang et al. (2023). *Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding.* [arXiv:2306.02858](https://arxiv.org/abs/2306.02858) — enables Q&A directly over lecture video content, going beyond transcription to comprehension.

---

### AI-Assisted Statistical Analysis
Tools that bring natural language interfaces to statistical analysis — lowering the barrier for IS researchers conducting quantitative studies.

| **Tool** | **Description** | **Best For** |
|-|-|-|
| [Julius AI](https://julius.ai/) | Upload data and ask questions in natural language; generates and executes Python/R code with charts. Added persistent project memory and multi-dataset join operations via natural language. | Rapid EDA and hypothesis testing without coding |
| [JASP](https://jasp-stats.org/) ⭐ open-source | Free, open-source statistics software with Bayesian analysis. JASP 0.18+ added Bayesian network analysis modules and improved SEM integration via R's `lavaan`. Increasingly used in IS behavioral research. | Bayesian hypothesis testing; survey data analysis; reproducible statistical reporting |
| [Statsbot](https://statsbot.co/) | Natural language interface to databases and analytics. Connect to your research data warehouse and ask questions in English. | Querying large IS datasets without SQL expertise |
| [PyGWalker](https://github.com/Kanaries/pygwalker) ⭐ open-source | Turns Pandas DataFrames into an interactive Tableau-style visual analysis interface within Jupyter notebooks. Added AI-assisted chart recommendations in 2024. | Interactive exploratory statistical analysis in a familiar notebook environment |
| [gptstudio](https://github.com/MichelNivard/gptstudio) ⭐ open-source | RStudio add-in that integrates ChatGPT/Claude directly into the R IDE for LLM-assisted statistical coding, model specification, and output interpretation. Active in the R stats community in 2024. | R-based researchers wanting in-IDE LLM assistance without leaving their statistical workflow |
| [Bambi](https://github.com/bambinos/bambi) ⭐ open-source | High-level Bayesian modeling in Python built on PyMC. Formula-based syntax similar to R's `lme4`; enables LLM-assisted Bayesian model specification via natural language prompting. | Bayesian multilevel models for IS survey/panel data; researchers familiar with R formula syntax |

> **Key Papers:**
> - Korinek (2023). *Language Models and Cognitive Automation for Economic Research.* NBER Working Paper 30957. Evaluates LLM accuracy on statistical reasoning and hypothesis testing; identifies reliability concerns alongside productivity gains.
> - Nori et al. (2024). *Can Large Language Models be Used to Provide Trustworthy Statistical Analysis?* [arXiv:2402.02012](https://arxiv.org/abs/2402.02012) — systematic benchmarking of LLM statistical reasoning reliability across common IS research methods.

---

### AI for Qualitative Analysis
DSR frequently involves qualitative interviews, think-aloud protocols, and thematic analysis. These tools bring AI assistance to grounded theory and qualitative coding workflows.

| **Tool** | **Description** | **Best For** |
|-|-|-|
| [Atlas.ti 24](https://atlasti.com/) | Industry-standard qualitative analysis software. Added AI-powered Concept Map generation, AI-assisted code suggestions, and GPT-4 integration for memo writing. | Grounded theory, thematic analysis of IS interview data |
| [NVivo 15](https://lumivero.com/products/nvivo/) | Qualitative data analysis with built-in AI auto-coding using sentiment and topic patterns. Added improved transcription via Azure Speech in 2024. | Mixed-method IS research combining survey and interview data |
| [Taguette](https://www.taguette.org/) ⭐ open-source | Free, open-source qualitative coding tool. Actively developed in 2024. Can be paired with local LLMs (via Ollama) for AI-assisted code suggestion with no data leaving the institution. | Privacy-sensitive IS research where data cannot leave your institution |
| [Reduct.Video](https://reduct.video/) | Video-based qualitative coding — transcribe and highlight video interviews, tag segments, and export for analysis. Added AI-generated highlight reels and improved cross-transcript search. | Think-aloud protocols, screen recordings, focus group videos |
| [QualCoder](https://github.com/ccbogel/QualCoder) ⭐ open-source | Open-source qualitative data analysis software (Python/Qt) supporting coding of text, images, audio, and video. The most feature-complete open-source NVivo alternative. Actively developed through 2024. | Full-featured qualitative analysis without commercial licensing costs |
| [Label Studio](https://github.com/HumanSignal/label-studio) ⭐ open-source | Open-source data labeling platform that researchers adapt for qualitative coding. Supports multi-annotator workflows, inter-annotator agreement metrics, and ML-assisted pre-labeling. | Collaborative multi-coder workflows; computing inter-rater reliability at scale |
| [CATMA](https://catma.de/) ⭐ open-source | Web-based open-source platform for collaborative text annotation and analysis. Supports both structured (tagset-based) and free-form annotation; used in digital humanities and IS research. | Web-based collaborative qualitative annotation without local installation |

- **LLM-Assisted Coding Workflow:** Use [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) to transcribe interviews locally → use a local LLM (via Ollama) with a grounded theory prompt to generate initial code suggestions → import into QualCoder, Atlas.ti, or NVivo for human review and saturation analysis. See [Multimodal Models](../MultimodalModels/README.md) for additional transcription options.

- **IS Research Note:** AI-generated codes should be treated as *initial suggestions*, not final codes. Validate with a second human coder and report inter-rater reliability (Cohen's κ ≥ 0.70 is the standard threshold for IS qualitative research).

> **Key Papers:**
> - Shaib et al. (2024). *CHIME: LLM-Assisted Hierarchical Organization of Scientific Studies for a Literature Review.* [arXiv:2407.06734](https://arxiv.org/abs/2407.06734) — addresses AI-assisted qualitative synthesis using hierarchical coding structures.
> - Xiao et al. (2023). *Supporting Qualitative Analysis with Large Language Models: Combining Codebook Themes and LLM-Generated Codes.* [arXiv:2306.00003](https://arxiv.org/abs/2306.00003) — evaluates LLM coding against human coders; finds moderate agreement with significant reliability variance across code types.

---

**Related Sections:** [Prompt Engineering](../Prompt-Engineering/README.md) | [LLMs & NLP](../NaturalLanguageProcessing/README.md) | [Python Tools](../PythonTools/README.md) | [Ethics](../Ethics/README.md) | [Evaluation & Benchmarking](../Evaluation/README.md)
