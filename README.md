# IRG: Iterative Reasoning-Generation for Text-to-Image Synthesis

> A production-grade Multi-Agent AI system that applies closed-loop feedback refinement to Text-to-Image generation via cloud APIs.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688)](https://fastapi.tiangolo.com/)
[![Gemini](https://img.shields.io/badge/LLM-Gemini%20API-4285F4)](https://ai.google.dev/)
[![Stability AI](https://img.shields.io/badge/Image-Stability%20AI%20SDXL-blueviolet)](https://stability.ai/)
[![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey)](https://creativecommons.org/licenses/by-nc/4.0/)

---

## Overview

Standard text-to-image models are single-shot systems — they generate an image and stop. **IRG** replaces that with an autonomous **Think → Generate → Critique → Refine** loop, where multiple AI agents collaborate to produce progressively higher-quality outputs.

The system is built as a REST API service and is designed for production deployment. It demonstrates practical AI engineering skills including: multi-agent orchestration, vector-based RAG, asynchronous API design, and cloud AI integration.

---

## System Architecture

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
│  │ (Analyzer)  │  Analyzes CLIP stats, crafts refined    │
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
│  │  (Quality   │  Evaluates CLIP stats → ACCEPT/REFINE   │
│  │   Gate)     │  Early stopping if quality ≥ threshold  │
│  └──────┬──────┘                                         │
│         │                                                │
│    ┌────┴────┐                                           │
│    │  ACCEPT │──► Return refined image + metadata        │
│    │  REFINE │──► Loop back to ExpertAgent               │
│    └─────────┘                                           │
└─────────────────────────────────────────────────────────┘
```

### Key Components

| Component | File | Role |
|---|---|---|
| **ExpertAgent** | `src/agents/expert_agent.py` | Analyzes image stats, produces refined prompts |
| **CriticAgent** | `src/agents/critic_agent.py` | Quality gate: ACCEPT or request another REFINE |
| **RAGService** | `src/services/rag_service.py` | Vector search over 244 historical refinement cases |
| **ImageService** | `src/services/image_service.py` | Calls Stability AI for Text2Img / Img2Img |
| **IRGWorkflow** | `src/core/workflow.py` | Orchestrates the closed-loop multi-agent pipeline |
| **API Layer** | `src/api/routes.py` | FastAPI REST endpoints (async, non-blocking) |

### Agent Interaction Flow

```
Iteration N:
  ExpertAgent → analyzes CLIP(mean, std, max) → produces refined_prompt
  ImageService → refines image via Img2Img API
  CriticAgent  → scores result [0.0–1.0]
               → ACCEPT (score ≥ 0.75): early stop, return result
               → REFINE (score < 0.75): proceed to Iteration N+1
```

---

## Experimental Results

Benchmarked against standard single-shot SDXL generation on compositionally complex prompts:

| Metric | Baseline (SDXL) | IRG 2-iter | IRG 4-iter |
|---|---|---|---|
| Compositional Accuracy | 0.3497 | **0.3768** (+7.74%) | 0.3651 |
| Aesthetic Score | 0.612 | 0.624 | **0.631** (+3.08%) |

> **Finding**: The 2-iteration loop yields the best compositional accuracy; 4 iterations maximizes aesthetic quality. CriticAgent's early stopping prevents over-refinement and semantic drift.

---

## Project Structure

```
IRG-Thesis/
├── src/
│   ├── agents/
│   │   ├── expert_agent.py       # ExpertAgent: prompt diagnosis & refinement
│   │   └── critic_agent.py       # CriticAgent: quality gate (ACCEPT/REFINE)
│   ├── services/
│   │   ├── gemini_service.py     # Gemini API client with retry logic
│   │   ├── image_service.py      # Stability AI API (Text2Img + Img2Img)
│   │   └── rag_service.py        # Vector RAG (sentence-transformers)
│   ├── core/
│   │   └── workflow.py           # Multi-agent orchestrator
│   ├── api/
│   │   └── routes.py             # FastAPI endpoints
│   └── config.py                 # Centralized configuration
├── dataset_final_v3.csv          # 244 historical refinement cases (RAG corpus)
├── Kaggle_IRG_Thesis.ipynb       # Cloud notebook for data generation
├── main.py                       # Server entry point
├── requirements.txt
└── .env                          # API keys (not committed)
```

---

## Setup & Installation

### Prerequisites
- Python >= 3.10
- API keys for [Google Gemini](https://aistudio.google.com/app/apikey) and [Stability AI](https://platform.stability.ai/)

### 1. Clone & Install

```bash
git clone https://github.com/your-username/IRG-Thesis.git
cd IRG-Thesis
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
STABILITY_API_KEY=your_stability_api_key_here
GEMINI_MODEL=gemini-2.0-flash
```

### 3. Run the Server

```bash
python main.py
```

Server starts at `http://localhost:8000`. Interactive API docs available at `http://localhost:8000/docs`.

---

## API Reference

### `POST /refine`

Initiates the multi-agent refinement pipeline for a given text prompt.

**Request:**
```json
{
  "prompt": "a photo of a cat sitting on a windowsill",
  "iterations": 2
}
```

**Response:**
```json
{
  "request_id": "f3d8c1a2-...",
  "status": "completed",
  "total_iterations": 1,
  "final_refined_prompt": "A high-quality, soft-focus photo of a cat...",
  "execution_time_seconds": 14.3,
  "iterations_summary": [
    {
      "iteration": 0,
      "issues": "none",
      "actions": "none",
      "refined_prompt": "A photo of a cat..."
    },
    {
      "iteration": 1,
      "issues": "[std:High, max:Blown]",
      "actions": "Apply blur filter, reduce highlights",
      "refined_prompt": "A soft-focus photo of a cat, controlled highlights...",
      "clip_mean": 0.5612,
      "clip_std": 0.1823,
      "clip_max": 0.9441
    }
  ],
  "output_dir": "./output/f3d8c1a2-.../"
}
```

> **Note**: The `total_iterations` may be less than the requested value if `CriticAgent` accepts the quality early.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM Reasoning | Google Gemini API |
| Image Generation | Stability AI SDXL 1.0 |
| Vector RAG | `sentence-transformers` (all-MiniLM-L6-v2) |
| API Framework | FastAPI + Uvicorn (async) |
| Image Analysis | CLIP (ViT-B/32) — via Pillow/NumPy |

---

## License

This project is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — for academic and non-commercial use only.

Copyright © 2025 Ngô Anh Dũng.
