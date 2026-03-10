# IRG: Iterative Reasoning-Generation for Text-to-Image Synthesis

> A closed-loop multi-agent AI system that iteratively refines Text-to-Image generation through autonomous reasoning and quality evaluation.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688)](https://fastapi.tiangolo.com/)
[![Gemini](https://img.shields.io/badge/LLM-Gemini%20API-4285F4)](https://ai.google.dev/)
[![Stability AI](https://img.shields.io/badge/Image-Stability%20AI%20SDXL-blueviolet)](https://stability.ai/)
[![Qwen](https://img.shields.io/badge/Phase1%20LLM-Qwen--2.5--3B-orange)](https://huggingface.co/Qwen/Qwen2.5-3B)
[![SDXL Base](https://img.shields.io/badge/Phase1%20Image-stable--diffusion--xl--base--1.0-purple)](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
[![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey)](https://creativecommons.org/licenses/by-nc/4.0/)

---

## Abstract

Standard text-to-image models are single-shot systems — they generate an image and stop. **IRG** replaces that with an autonomous **Think → Generate → Critique → Refine** loop, where multiple AI agents collaborate to produce progressively higher-quality outputs.

The project has evolved through two phases: an academic thesis foundation using fine-tuned local models (Phase 1), and a fully productionized multi-agent cloud API system (Phase 2).

---

## Phase 1: Academic Thesis (Qwen-IRG)

The foundational research phase involved fine-tuning a small-parameter LLM to perform visual diagnostics on consumer-grade hardware.

### Technical Approach

- **Feature-Aware Diagnostics**: Translates CLIP-extracted statistical features (mean, variance, max) into structured textual descriptions the LLM can reason about.
- **Low-Rank Adaptation (LoRA)**: Fine-tuned Qwen-2.5 3B on 4,000 synthetic reasoning traces to master object decomposition, spatial reasoning, and attribute binding.
- **Adaptive Denoising Schedule**: Dynamically decaying denoising with linearly increasing guidance scales to prevent semantic drift during Img2Img inference.

### Experimental Results (Phase 1)

Benchmarked against standard single-shot SDXL generation on compositionally complex prompts:

| Metric | Baseline (SDXL) | IRG 2-iter | IRG 4-iter |
|---|---|---|---|
| Compositional Accuracy | 0.3497 | **0.3768** (+7.74%) | 0.3651 |
| Aesthetic Score | 0.612 | 0.624 | **0.631** (+3.08%) |

> **Finding**: 2-iteration loop yields the best compositional accuracy; 4 iterations maximizes aesthetic quality.

### Phase 1 Reproduction

1. Navigate to `Kaggle_IRG_Thesis.ipynb` for cloud execution (recommended: dual T4 GPU on Kaggle).
2. Execute notebook sections in order: Dataset Generation → LoRA Fine-tuning → Inference → Benchmark.
3. Requires 16GB+ VRAM for 4-bit quantized Qwen-2.5-3B and FP16 SDXL coexistence.

---

## Phase 2: Production Multi-Agent System (Gemini-IRG)

The architecture was elevated to overcome the contextual limitations of small-parameter models, transitioning to a fully autonomous multi-agent cloud API service.

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
├── Kaggle_IRG_Thesis.ipynb       # Phase 1: cloud notebook for fine-tuning & data gen
├── main.py                       # Phase 2: server entry point
├── requirements.txt
└── .env                          # API keys (not committed)
```

---

## Setup & Installation (Phase 2)

### Prerequisites
- Python >= 3.10
- API keys: [Google Gemini](https://aistudio.google.com/app/apikey) and [Stability AI](https://platform.stability.ai/)

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
GEMINI_MODEL=gemini-2.0-flash ( or model llm you want)
```

### 3. Run the Server

```bash
python main.py
```

Interactive API docs: `http://localhost:8000/docs`

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

> `total_iterations` may be less than requested if `CriticAgent` accepts quality early.

---

## Tech Stack

| | Phase 1 (Thesis) | Phase 2 (Production) |
|---|---|---|
| **LLM** | Qwen-2.5 3B (LoRA fine-tuned) | Google Gemini API |
| **Image Gen** | Stable Diffusion XL (local) | Stability AI `stable-diffusion-xl-1024-v1-0` |
| **RAG** | — | sentence-transformers + cosine similarity |
| **API** | — | FastAPI + Uvicorn (async) |
| **Hardware** | Dual NVIDIA T4 (Kaggle) | Cloud APIs (no GPU needed) |

---

## Models Used

[![Qwen](https://img.shields.io/badge/Phase%201-Qwen--2.5--3B%20(LoRA)-orange?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyQzYuNDggMiAyIDYuNDggMiAxMnM0LjQ4IDEwIDEwIDEwIDEwLTQuNDggMTAtMTBTMTcuNTIgMiAxMiAyem0wIDE4Yy00LjQxIDAtOC0zLjU5LTgtOHMzLjU5LTggOC04IDggMy41OSA4IDgtMy41OSA4LTggOHoiLz48L3N2Zz4=)](https://huggingface.co/Qwen/Qwen2.5-3B)
[![SDXL](https://img.shields.io/badge/Phase%202-stable--diffusion--xl--1024--v1--0-blueviolet?logo=stability-ai)](https://platform.stability.ai/docs/api-reference)

---

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — academic and non-commercial use only.

Copyright © 2025 Ngô Anh Dũng.
