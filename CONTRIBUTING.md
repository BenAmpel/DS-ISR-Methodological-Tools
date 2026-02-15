# Contributing to DS-ISR Methodological Tools

Thank you for your interest in contributing! This repository curates AI/ML tools and papers specifically for Design Science Information Systems (DSR/IS) researchers. Contributions that add value for this audience are warmly welcome.

---

## What to Contribute

**Good contributions include:**
- New papers (published or arXiv preprint) with a brief description of IS research relevance
- Tools or libraries with active maintenance and clear applicability to IS research
- New section proposals for topics not currently covered (see open gaps below)
- Corrections to broken links, outdated descriptions, or factual errors
- IS Research Application notes that connect technical content to DSR methodology

**Out of scope:**
- Papers without clear relevance to IS or DSR research
- Abandoned repositories (last commit >2 years ago) unless historically significant
- Commercial tools without a free tier or academic license
- Duplicate entries already covered in another section

---

## How to Contribute

### Option 1: Open an Issue (Recommended for suggestions)
1. Go to [Issues](https://github.com/BenAmpel/DS-ISR-Methodological-Tools/issues)
2. Click **New Issue**
3. Use one of the templates:
   - **Paper Suggestion**: Title, year, link, one-sentence IS relevance description
   - **Tool Suggestion**: Name, link, category, one-sentence description
   - **Section Proposal**: Proposed section name, rationale, 3+ seed papers/tools
   - **Broken Link / Correction**: Section, current text, corrected text

### Option 2: Submit a Pull Request
1. Fork the repository
2. Create a branch: `git checkout -b add-[topic]-[brief-description]`
3. Make your changes following the style guide below
4. Submit a pull request with a clear description of what was added and why

---

## Style Guide

### Paper Entries
Follow this format consistently:
```markdown
- [Paper Title](https://arxiv.org/abs/XXXX.XXXXX), YEAR - One sentence describing what the paper does and its IS relevance. [Code](https://github.com/author/repo)
```

- Use the arXiv abstract page URL where available (not the PDF direct link)
- Include `[Code](url)` only if an official or widely-used implementation exists
- Keep descriptions to one sentence; prioritize IS applicability over technical detail
- Add to **Papers with code** or **Papers without code** subsections as appropriate

### Tool Entries
Follow this format for table rows:
```markdown
| [Tool Name](https://tool-url.com) | Brief description of what it does. | Active/Maintenance/Archived |
```

Or for inline lists:
```markdown
- [Tool Name](https://github.com/author/repo) - Brief description focused on IS research use case.
```

### Section Headers
- Each section README should start with a title, subtitle, and `*Last updated: [Month Year]*`
- Include a navigation table at the top linking to subsections
- Include an `> **IS Research Applications:**` callout block after the navigation table
- End with a `**Related Sections:**` footer line

### IS Research Applications Callout
Every section should have a callout box explaining relevance to IS researchers:
```markdown
> **IS Research Applications:** [2-4 concrete use cases connecting the section to DSR/IS methodology]
```

---

## Section Template

When proposing a new section, use this template for the README:

```markdown
# Section Name
*Brief subtitle describing the section scope.*

*Last updated: [Month Year]*

> Brief paragraph explaining the section and its importance for IS research.

| | | |
|-|-|-|
| [Subsection 1](#Subsection-1) | [Subsection 2](#Subsection-2) | [Subsection 3](#Subsection-3) |

---

> **IS Research Applications:** [concrete use cases]

---

### Subsection 1
[Description]

- **Seminal Paper:**
  - [Paper](url), YEAR - Description. [Code](url)

- **Papers with code:**
  - [Paper](url), YEAR - Description. [Code](url)

- **Papers without code:**
  - [Paper](url), YEAR - Description.

---

### Tools & Libraries

| **Tool** | **Description** | **Best For** |
|-|-|-|
| [Tool](url) | Description | Use case |

---

**Related Sections:** [Section](../Section/README.md) | [Section](../Section/README.md)
```

---

## Priority Gaps (as of February 2026)

The following areas are under-represented and high-value contributions:

| Area | Suggested Content |
|-|-|
| **Topic Modeling** | BERTopic, CTM (Combined Topic Model), NMF |
| **Causal Inference** | DoWhy, EconML, causal graphs for IS research |
| **Survey & Measurement** | AI-assisted scale development, LLM survey generation |
| **Federated Learning** | Privacy-preserving ML for distributed IS research |
| **Knowledge Graphs** | Wikidata, construction pipelines, IS ontologies |

---

## Code of Conduct

This repository follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/). Be respectful, constructive, and focused on advancing IS research quality.

---

## Questions?

Open an issue with the **Question** label or contact the maintainer via GitHub.
