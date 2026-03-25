<div align="center">

# 🎨 IRG — Iterative Reasoning-Generation
### An Autonomous Multi-Agent System That Refines AI Images Automatically

**Give it a prompt. Watch it think, generate, critique, and perfect.**

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Google Gemini](https://img.shields.io/badge/LLM-Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)
[![Stability AI](https://img.shields.io/badge/Image-SDXL%201.0-blueviolet?style=for-the-badge)](https://stability.ai/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey?style=for-the-badge)](https://creativecommons.org/licenses/by-nc/4.0/)

[**Quickstart**](#-quickstart-2-minutes) • [**API Docs**](#-api) • [**Demo**](#-demo) • [**Architecture**](#️-how-it-works)

</div>

---

## 🤔 The Problem

Standard text-to-image models are **one-shot and dumb**. You type a prompt, get one image, and if it's wrong — you manually tweak words and try again. And again. And again.

**IRG automates that entire loop.** It acts like an AI art director:

```
Your Prompt → Think → Generate → Critique → Refine → ... → Perfect Image ✅
```

No manual prompt engineering. No guessing. Just describe what you want.

---

## 🎬 Demo

### 📺 Video Demo

[https://github.com/user-attachments/assets/video_demo.mp4](https://github.com/user-attachments/assets/0182f60d-0d8b-4cdd-a648-3194bda74b92)

> *Upload `assets/demo_video.mp4` to GitHub and replace the link above. GitHub will auto-embed it.*

---

### 🌃 Example Run: Cyberpunk City

**Prompt:** *"futuristic cyberpunk city at night, neon reflections on wet streets"*
**Result:** 3 iterations in 201s | CriticAgent detected **Blown-High** highlights and autonomously refined lighting.

<details>
<summary><b>🧠 Full Agent Reasoning Log (Click to expand)</b></summary>

#### Iteration 0 → Initial Generation
**Refined Prompt:** *"A futuristic cyberpunk city at night, balanced atmospheric lighting, soft neon reflections on wet asphalt, moderate contrast, diffused glow, cinematic composition, 8k resolution."*

#### Iteration 1 → Blown Highlights Detected
| Metric | Value | Status |
|--------|-------|--------|
| mean | 0.2977 | ✅ Optimal |
| std | 0.1771 | ✅ Optimal |
| max | **0.9935** | ⚠️ **Blown-High** |

- **Expert Action:** *"Apply a subtle highlight recovery or luminosity mask to bring the maximum pixel intensity below 0.97."*
- **Critic Verdict:** `REFINE` (score=0.65) — *"Max value exceeds 0.97, indicating blown-out highlights."*

#### Iteration 2 → Precision Adjustment
| Metric | Value | Status |
|--------|-------|--------|
| mean | 0.2991 | ✅ Optimal |
| std | 0.1772 | ✅ Optimal |
| max | **0.9922** | ⚠️ **High** |

- **Expert Action:** *"-2% exposure to pull the peak highlights back below the threshold."*
- **Critic Verdict:** `REFINE` (score=0.65) — *"Highlights still reaching 0.99, need more attenuation."*

#### Iteration 3 → Final Output
| Metric | Value | Status |
|--------|-------|--------|
| mean | 0.2948 | ✅ Optimal |
| std | 0.1864 | ✅ Optimal |
| max | **0.9922** | ⚠️ Stable |

- **Final Prompt:** *"A futuristic cyberpunk city at night, vibrant neon reflections on rain-slicked streets, cinematic lighting with controlled luminosity, deep shadows, balanced highlights without overexposure, sharp detail, 8k resolution, photorealistic."*

</details>

---

### 🐱 Example Run: Orange Cat Portrait

**Prompt:** *"a fluffy orange cat sitting on a windowsill, warm afternoon sunlight"*
**Result:** 2 iterations in 118s | CriticAgent detected overexposed fur highlights.

<details>
<summary><b>🧠 Full Agent Reasoning Log (Click to expand)</b></summary>

#### Iteration 0 → Initial Generation
**Refined Prompt:** *"A photorealistic portrait of a fluffy orange cat sitting on a windowsill, soft diffused warm afternoon sunlight, gentle shadows, balanced exposure, cinematic depth of field, natural color grading."*

#### Iteration 1 → White Point Correction
| Metric | Value | Status |
|--------|-------|--------|
| mean | 0.4518 | ✅ Optimal |
| std | 0.2232 | ✅ Optimal |
| max | **0.9922** | ⚠️ **Blown-High** |

- **Expert Action:** *"-5% adjust global exposure; -8% adjust white point to prevent clipping."*
- **Critic Verdict:** `REFINE` (score=0.65) — *"Max value indicates blown highlights that need correction."*

#### Iteration 2 → Final Output
| Metric | Value | Status |
|--------|-------|--------|
| mean | 0.4512 | ✅ Optimal |
| std | 0.2083 | ✅ Optimal |
| max | **0.9935** | ⚠️ Stable |

- **Final Prompt:** *"A hyper-realistic, soft-focus photograph of a fluffy orange cat sitting on a windowsill during warm afternoon sunlight, balanced exposure with controlled specular highlights, natural diffused lighting, detailed fur texture, 8k, professional color grading with soft highlight roll-off."*

</details>

---

## ✨ Key Features

- 🧠 **Autonomous Agentic Loop** — Think → Generate → Critique → Refine, no human in the loop
- 👁️ **Dual-Agent Architecture** — ExpertAgent diagnoses, CriticAgent quality-gates
- 🔍 **RAG-Enhanced Prompting** — Learns from past high-quality generations using semantic search
- ⚡ **Early Stopping** — Stops automatically when quality threshold is met (no wasted API calls)
- 📦 **Production Docker Image** — Fully containerized, CI/CD via GitHub Actions
- 🔌 **Drop-in REST API** — `POST /refine` and you're done, integrates into any pipeline
- 🎨 **Gradio Web UI** — Browser-based demo for visual interaction
- 🔬 **Research Foundation** — Built on LoRA fine-tuning + compositional reasoning research

---

## ⚡ Quickstart (2 minutes)

### Prerequisites
You need two free(ish) API keys:
- [Google AI Studio](https://aistudio.google.com/) → Gemini API Key (free tier available)
- [Stability AI](https://platform.stability.ai/) → Stability API Key ($10 free credit on signup)

### Option 1: Docker — Recommended

```bash
# 1. Clone
git clone https://github.com/dungna13/IRG-Thesis.git && cd IRG-Thesis

# 2. Configure
cp .env.example .env
# Edit .env with your two API keys

# 3. Run
docker compose up
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) — interactive API is ready.

### Option 2: Local Python

```bash
git clone https://github.com/dungna13/IRG-Thesis.git
cd IRG-Thesis
pip install -r requirements.txt

# Configure
cp .env.example .env   # add your API keys

# Run API server
python main.py

# Or run Gradio UI
pip install gradio
python app_gradio.py
```

### `.env` Template

```env
GEMINI_API_KEY=AIza...         # From Google AI Studio
STABILITY_API_KEY=sk-...       # From Stability AI Platform
GEMINI_MODEL=gemini-2.0-flash-lite
```

---

## 🔌 API

### `POST /refine` — Refine an image iteratively

```bash
curl -X POST http://localhost:8000/refine \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a knight fighting a dragon at golden hour", "iterations": 3}'
```

**Response:**
```json
{
  "total_iterations": 2,
  "final_quality_score": 0.81,
  "output_dir": "./outputs/run_20250326_143201/",
  "iterations_summary": [
    {
      "iteration": 1,
      "issues_found": ["poor lighting", "missing shield detail"],
      "action": "REFINE",
      "quality_score": 0.67,
      "refined_prompt": "a medieval knight in gleaming armor holding a shield..."
    },
    {
      "iteration": 2,
      "issues_found": [],
      "action": "ACCEPT",
      "quality_score": 0.81
    }
  ]
}
```

> Full interactive docs at [http://localhost:8000/docs](http://localhost:8000/docs) (Swagger UI)

---

## 🏗️ How It Works

```
User Prompt
      │
      ▼
  ┌──────────┐    ┌──────────────────────────────────┐
  │RAGService│───►│ Historical Successes (Vector DB)  │
  └────┬─────┘    └──────────────────────────────────┘
       │ Few-shot context
       ▼
  ┌─────────────┐
  │ ExpertAgent │◄── Gemini LLM
  │  (The Brain)│    Diagnoses issues, crafts refined prompt
  └──────┬──────┘
         │
         ▼
  ┌──────────────┐
  │ ImageService │◄── Stability AI SDXL
  │ (Generator)  │    Text→Image then Image→Image
  └──────┬───────┘
         │
         ▼
  ┌─────────────┐
  │ CriticAgent │◄── Gemini LLM
  │  (The Eye)  │    Scores quality → ACCEPT or REFINE
  └──────┬──────┘
         │
    ┌────┴────┐
    │ ACCEPT  │──► Return final image + full iteration log
    │ REFINE  │──► Loop back to ExpertAgent
    └─────────┘
```

| Agent | Role |
|-------|------|
| **ExpertAgent** | The "Brain" — Translates image statistics into language, crafts improved prompts |
| **CriticAgent** | The "Eye" — Evaluates outputs against quality thresholds, triggers early stop |
| **RAGService** | The "Memory" — Semantic retrieval of historical prompts for few-shot context |
| **ImageService** | The "Executioner" — Interfaces with Stability AI for T2I and I2I refinement |

---

## 📁 Project Structure

```
IRG-Thesis/
├── src/
│   ├── agents/
│   │   ├── expert_agent.py       # Prompt diagnosis & refinement logic
│   │   └── critic_agent.py       # Quality gate (ACCEPT / REFINE)
│   ├── services/
│   │   ├── gemini_service.py     # Gemini API integration
│   │   ├── image_service.py      # Stability AI + statistical analysis
│   │   └── rag_service.py        # Vector RAG (all-MiniLM-L6-v2)
│   ├── core/
│   │   └── workflow.py           # Multi-agent orchestrator
│   └── api/
│       └── routes.py             # FastAPI REST endpoints
├── assets/                       # Demo video
├── app_gradio.py                 # Gradio Web UI
├── dataset_final_v3.csv          # RAG historical corpus
├── .env.example                  # Environment variable template
├── docker-compose.yml            # One-command setup
├── Dockerfile                    # Production image
├── main.py                       # Entry point
└── requirements.txt              # Dependencies
```

---

## 🧪 Research Background

This project originated as a university thesis exploring whether a **small LLM (3B parameters)** could learn to reason about image quality and iteratively improve generation outcomes — on **consumer hardware**, without cloud dependencies.

### Phase 1 Results (Local Qwen-2.5 3B + LoRA)

| Metric | Baseline SDXL | IRG 2-iter | IRG 4-iter |
|--------|:---:|:---:|:---:|
| Compositional Accuracy | 0.3497 | **0.3768** (+7.74%) | 0.3651 |
| Aesthetic Score | 0.612 | 0.624 | **0.631** (+3.08%) |

**Key innovations:**
- **Feature-Aware Diagnostics** — Converts image statistics (mean, variance) into structured text for LLM reasoning
- **LoRA Fine-tuning** — 4,000 synthetic reasoning traces on a 3B parameter model
- **Adaptive Denoising** — Dynamically decays noise strength to prevent semantic drift

### Phase 2 (Current — Cloud API)
Architecture elevated to a full cloud API service using Gemini + Stability AI for production-grade reliability and speed.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **API Framework** | FastAPI (async) |
| **LLMs** | Google Gemini 2.0 Flash Lite |
| **Image Generation** | Stability AI SDXL 1.0 |
| **RAG Engine** | sentence-transformers `all-MiniLM-L6-v2` |
| **Web UI** | Gradio |
| **Containerization** | Docker + docker-compose |
| **CI/CD** | GitHub Actions → GHCR |
| **Language** | Python 3.11 |

---

## 🗺️ Roadmap

- [x] Multi-Agent Pipeline (ExpertAgent + CriticAgent)
- [x] RAG-enhanced prompt refinement
- [x] Docker + CI/CD
- [x] Gradio Web UI
- [ ] HuggingFace Spaces deployment — Free public demo
- [ ] Negative prompt support — Better control over unwanted elements
- [ ] Generation gallery — SQLite-backed history with image browser
- [ ] Local model option — Ollama integration for fully offline runs
- [ ] ComfyUI node — Integration into popular image workflow tool

---

## 📋 Changelog

### v3.0.0 (Current)
- Multi-Agent Architecture: ExpertAgent + CriticAgent dual-agent pipeline
- RAG Service with semantic vector search (`all-MiniLM-L6-v2`)
- Gemini Cloud LLM + Stability AI Cloud for production reliability
- FastAPI async REST API + Gradio Web UI
- Full Docker containerization with GitHub Actions CI/CD

### v2.0.0 (Academic Thesis)
- LoRA fine-tuning of Qwen-2.5-3B on 4,000 synthetic reasoning traces
- +7.74% compositional accuracy, +3.08% aesthetic score over baseline SDXL

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repo
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Commit using [Conventional Commits](https://www.conventionalcommits.org/): `git commit -m "feat: add your feature"`
4. Open a Pull Request

---

## 📜 License

Licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (Non-Commercial Research).
Copyright © 2025 **Ngô Anh Dũng**

---

<div align="center">

**If this project helped you, please ⭐ star the repo — it helps others find it!**

</div>
