# IRG: Iterative Reasoning-Generation for Text-to-Image Synthesis

> A production-grade Multi-Agent AI system that iteratively refines Text-to-Image generation through autonomous reasoning and quality evaluation. Now fully Dockerized with CI/CD.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688)](https://fastapi.tiangolo.com/)
[![Gemini](https://img.shields.io/badge/LLM-Gemini%20API-4285F4)](https://ai.google.dev/)
[![Stability AI](https://img.shields.io/badge/Image-Stability%20AI%20SDXL-blueviolet)](https://stability.ai/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey)](https://creativecommons.org/licenses/by-nc/4.0/)

---

## Abstract

Standard text-to-image models are single-shot systems — they generate an image and stop. **IRG** replaces that with an autonomous **Think → Generate → Critique → Refine** loop, where multiple AI agents collaborate to produce progressively higher-quality outputs.

The project has evolved through two phases: an academic thesis foundation using fine-tuned local models (Phase 1), and a fully productionized, lightweight, multi-agent cloud API system (Phase 2).

---

## Phase 1: Academic Thesis (Qwen-IRG)

The foundational research phase involved fine-tuning a small-parameter LLM to perform visual diagnostics on consumer-grade hardware.

### Technical Approach

- **Feature-Aware Diagnostics**: Translates statistical features (mean, variance, max) into structured textual descriptions for LLM reasoning.
- **Low-Rank Adaptation (LoRA)**: Fine-tuned Qwen-2.5 3B on 4,000 synthetic reasoning traces to master object decomposition and spatial reasoning.
- **Adaptive Denoising Schedule**: Dynamically decaying denoising with linearly increasing guidance scales to prevent semantic drift.

### Experimental Results (Phase 1)

| Metric | Baseline (SDXL) | IRG 2-iter | IRG 4-iter |
|---|---|---|---|
| Compositional Accuracy | 0.3497 | **0.3768** (+7.74%) | 0.3651 |
| Aesthetic Score | 0.612 | 0.624 | **0.631** (+3.08%) |

---

## Phase 2: Production Multi-Agent System (Gemini-IRG)

The architecture was elevated to a fully autonomous multi-agent cloud API service, optimized for speed and reliability.

### System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   IRG Multi-Agent Pipeline               │
│                                                          │
│  User Prompt                                            │
│       │                                                  │
│       ▼                                                  │
│  ┌──────────┐    ┌─────────────────────────────────┐    │
│  │ RAGService│───►│ Historical Cases (Vector Search) │   │
│  └────┬─────┘    └─────────────────────────────────┘    │
│       │ Context                                          │
│       ▼                                                  │
│  ┌─────────────┐                                         │
│  │ ExpertAgent │ ◄── Gemini API                          │
│  │ (Analyzer)  │  Analyzes image stats, crafts refined   │
│  └──────┬──────┘  prompt via few-shot RAG context        │
│         │                                                │
│         ▼                                                │
│  ┌──────────────┐                                        │
│  │ ImageService │ ◄── Stability AI SDXL API              │
│  │ (Generator)  │  Text2Img → Img2Img refinement         │
│  └──────┬───────┘                                        │
│         │                                                │
│         ▼                                                │
│  ┌─────────────┐                                         │
│  │ CriticAgent │ ◄── Gemini API                          │
│  │  (Quality   │  Evaluates image stats → ACCEPT/REFINE  │
│  │   Gate)     │  Early stopping if quality ≥ threshold  │
│  └──────┬──────┘                                         │
│         │                                                │
│    ┌────┴────┐                                           │
│    │  ACCEPT │──► Return refined image + metadata        │
│    │  REFINE │──► Loop back to ExpertAgent               │
│    └─────────┘                                           │
└─────────────────────────────────────────────────────────┘
```

### Key Features

- **Multi-Agent Orchestration**: ExpertAgent (Reasoning) and CriticAgent (Evaluation) collaborate in a closed-loop.
- **Semantic Vector RAG**: Uses `sentence-transformers` for intelligent case retrieval from high-quality historical refinements.
- **Production Optimization**: Replaced heavy PyTorch/CLIP dependencies with **lightweight NumPy-based analysis**, reducing Docker image size by ~80% and speeding up startup.
- **Async Execution**: FastAPI server with non-blocking execution for heavy AI tasks.

---

## Setup & Installation

### Option A: Docker (Recommended)

The project is automatically built and stored in the **GitHub Container Registry (GHCR)**.

1. **Login to GHCR**:
   ```bash
   echo $CR_PAT | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
   ```
2. **Pull and Run**:
   ```bash
   docker run -d -p 8000:8000 --name irg-app --env-file .env ghcr.io/dungna13/irg-thesis:latest
   ```

### Option B: Local Installation

1. **Clone & Install**:
   ```bash
   git clone https://github.com/dungna13/IRG-Thesis.git
   cd IRG-Thesis
   pip install -r requirements.txt
   ```
2. **Configure Environment**: Create `.env` with `GEMINI_API_KEY` and `STABILITY_API_KEY`.
3. **Run**: `python main.py`

---

## Tech Stack

| Layer | Technology |
|---|---|
| **LLM Reasoning** | Google Gemini 2.0 Flash / 1.5 Pro |
| **Image Generation** | Stability AI `stable-diffusion-xl-1024-v1-0` |
| **Vector RAG** | `sentence-transformers` (all-MiniLM-L6-v2) |
| **Image Analysis** | Lightweight NumPy-based Statistical Profiling |
| **API Framework** | FastAPI + Uvicorn (Asynchronous) |
| **CI/CD** | GitHub Actions + Docker (GHCR) |

---

## Models Used

[![Qwen](https://img.shields.io/badge/Phase1%20LLM-Qwen--2.5--3B-orange)](https://huggingface.co/Qwen/Qwen2.5-3B)
[![SDXL Base](https://img.shields.io/badge/Phase1%20Image-stable--diffusion--xl--base--1.0-purple)](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
[![Gemini](https://img.shields.io/badge/Phase2%20LLM-Gemini--2.0--Flash-4285F4)](https://ai.google.dev/)
[![SDXL 1.0](https://img.shields.io/badge/Phase2%20Image-SDXL--1.0--Cloud-blueviolet)](https://platform.stability.ai/)

---

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — academic and non-commercial use only.

Copyright © 2025 Ngô Anh Dũng.
