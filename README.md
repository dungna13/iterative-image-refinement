# IRG: Iterative Reasoning-Generation for Text-to-Image Synthesis

> A production-grade Multi-Agent AI system that iteratively refines Text-to-Image generation through autonomous reasoning and quality evaluation. Fully Dockerized with CI/CD integration.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688)](https://fastapi.tiangolo.com/)
[![Gemini](https://img.shields.io/badge/LLM-gemini--3.1--flash--lite--preview-4285F4)](https://ai.google.dev/)
[![Stability AI](https://img.shields.io/badge/Image-SDXL--1.0--Cloud-blueviolet)](https://stability.ai/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Qwen](https://img.shields.io/badge/Phase1%20LLM-Qwen--2.5--3B-orange)](https://huggingface.co/Qwen/Qwen2.5-3B)
[![SDXL Base](https://img.shields.io/badge/Phase1%20Image-stable--diffusion--xl--base--1.0-purple)](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)

---

## Abstract

Standard text-to-image models are single-shot systems — they generate an image and stop. **IRG** replaces that with an autonomous **Think → Generate → Critique → Refine** loop, where multiple AI agents collaborate to produce progressively higher-quality outputs.

The project has evolved through two phases:
1.  **Phase 1 (Academic)**: A thesis foundation focusing on fine-tuning small-parameter local models (Qwen-2.5-3B).
2.  **Phase 2 (Production)**: A streamlined, lightweight multi-agent cloud API system leveraging Gemini 3.1 and Stability AI.

---

## Phase 1: Academic Thesis (Qwen-IRG)

The foundational research phase involved fine-tuning a small-parameter LLM to perform complex visual diagnostics on consumer-grade hardware.

### Technical Approach
-   **Feature-Aware Diagnostics**: Translates statistical features (mean, variance, max) into structured textual descriptions for LLM reasoning.
-   **Low-Rank Adaptation (LoRA)**: Fine-tuned Qwen-2.5 3B on 4,000 synthetic reasoning traces to master object decomposition, spatial reasoning, and attribute binding.
-   **Adaptive Denoising Schedule**: Dynamically decaying denoising with linearly increasing guidance scales to prevent semantic drift during refinement.

### Experimental Results
| Metric | Baseline (SDXL) | IRG 2-iter | IRG 4-iter |
| :--- | :--- | :--- | :--- |
| **Compositional Accuracy** | 0.3497 | **0.3768** (+7.74%) | 0.3651 |
| **Aesthetic Score** | 0.612 | 0.624 | **0.631** (+3.08%) |

---

## Phase 2: Production Multi-Agent System (Gemini-IRG)

The architecture was elevated to a fully autonomous multi-agent cloud API service, optimized for industrial deployment.

### System Architecture
```text
┌─────────────────────────────────────────────────────────┐
│                   IRG Multi-Agent Pipeline               │
│                                                          │
│  User Prompt                                            │
│       │                                                  │
│       ▼                                                  │
│  ┌──────────┐    ┌─────────────────────────────────┐    │
│  │RAGService│───►│ Historical Cases (Vector Search) │   │
│  └────┬─────┘    └─────────────────────────────────┘    │
│       │ Context                                          │
│       ▼                                                  │
│  ┌─────────────┐                                         │
│  │ ExpertAgent │ ◄── Google gemini-3.1-flash-lite-preview│
│  │ (Analyzer)  │  Analyzes image stats, crafts refined   │
│  └──────┬──────┘  prompt via few-shot RAG context        │
│         │                                                │
│         ▼                                                │
│  ┌──────────────┐                                        │
│  │ ImageService │ ◄── Stability stable-diffusion-xl-1024-v1.0
│  │ (Generator)  │  Text2Img → Img2Img refinement         │
│  └──────┬───────┘                                        │
│         │                                                │
│         ▼                                                │
│  ┌─────────────┐                                         │
│  │ CriticAgent │ ◄── Google gemini-3.1-flash-lite-preview│
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

### Key Components
-   **ExpertAgent**: The "Brain". Diagnoses issues and suggests prompt improvements.
-   **CriticAgent**: The "Eye". Evaluates quality using statistical thresholds and decides when to stop.
-   **RAGService**: The "Memory". Uses `sentence-transformers` for semantic retrieval of high-quality historical examples.
-   **ImageService**: The "Executioner". Interfaces with Stability AI Cloud for premium image synthesis.

---

## Project Structure
```text
IRG-Thesis/
├── src/
│   ├── agents/
│   │   ├── expert_agent.py       # Prompt diagnosis & refinement
│   │   └── critic_agent.py       # Quality gate (ACCEPT/REFINE)
│   ├── services/
│   │   ├── gemini_service.py     # Gemini SDK integration
│   │   ├── image_service.py      # Stability AI API & NumPy Analyzer
│   │   └── rag_service.py        # Vector RAG (all-MiniLM-L6-v2)
│   ├── core/
│   │   └── workflow.py           # Multi-agent orchestrator
│   ├── api/
│   │   └── routes.py             # FastAPI REST endpoints
│   └── config.py                 # Pydantic Configuration
├── .github/workflows/            # GitHub Actions (CI/CD)
├── dataset_final_v3.csv          # RAG historical corpus
├── main.py                       # Uvicorn entry point
├── Dockerfile                    # Production image definition
└── requirements.txt              # Dependency manifest
```

---

## Setup & Installation

### Option 1: Docker (Fastest & Cleanest) 
The project is fully containerized and published to the **GitHub Container Registry (GHCR)**.

#### 1. Login to GitHub Container Registry
Generate a **Personal Access Token (classic)** with `read:packages` scope, then run:
```bash
docker login ghcr.io -u YOUR_GITHUB_USERNAME
```
*(Paste your PAT when prompted for a password).*

#### 2. Pull the Image
```bash
docker pull ghcr.io/dungna13/irg-thesis:latest
```

#### 3. Run with Environment Variables
Create a local `.env` file first, then run:
```bash
docker run -d -p 8000:8000 --name irg-app --env-file .env ghcr.io/dungna13/irg-thesis:latest
```
Access the system at `http://localhost:8000/docs`.

---

### Option 2: Local Development
1.  **Clone & Install**:
    ```bash
    git clone https://github.com/dungna13/IRG-Thesis.git
    cd IRG-Thesis
    pip install -r requirements.txt
    ```
2.  **Configure `.env`**:
    ```env
    GEMINI_API_KEY=AIza...
    STABILITY_API_KEY=sk-...
    GEMINI_MODEL=gemini-3.1-flash-lite-preview
    ```
3.  **Run**: `python main.py`

---

## API Documentation
### `POST /refine`
Starts the multi-agent refinement loop.

**Request Body:**
```json
{
  "prompt": "a photo of a cat sitting on a windowsill",
  "iterations": 2
}
```

**Response Highlights:**
- `total_iterations`: Actual rounds performed (may be less if early stopped).
- `iterations_summary`: Detailed log of issues, actions, and CLIP-derived stats.
- `output_dir`: Location of generated images.

---

## Tech Stack
| Category | Technology |
| :--- | :--- |
| **Orchestration** | Python 3.11, FastAPI (Async) |
| **LLMs** | Google `gemini-3.1-flash-lite-preview` |
| **T2I Model** | Stability AI `stable-diffusion-xl-1024-v1-0` |
| **RAG Engine** | `sentence-transformers` (all-MiniLM-L6-v2) |
| **Packaging** | Docker (python:3.11-slim) |
| **CI/CD** | GitHub Actions & GHCR |

---

## 📜 License & Copyright
Licensed under **CC BY-NC 4.0** (Non-Commercial Research).
Copyright © 2025 **Ngô Anh Dũng**.
