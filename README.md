# Qwen-IRG (V1) & Gemini-IRG (V2): Visual Reasoning and Autonomous Refinement for Image Generation

## Project Overview

This research project presents a comprehensive pipeline for enhancing image generation capabilities through the reasoning power of Large Language Models (LLMs). The project consists of two major phases:

1. **V1.0 (Graduation Thesis - Qwen-IRG)**: A fine-tuned Qwen model (using LoRA/QLoRA) acting as an expert visual reasoning assistant, trained on a custom synthetic dataset to provide detailed visual reasoning, prompt refinement, and quality diagnostics.
2. **V2.0 (Autonomous Architecture - Gemini-IRG)**: An evolution of the thesis work that introduces an Autonomous Multi-Agent workflow and Retrieval-Augmented Generation (RAG) using Gemini and Stability AI for a fully closed-loop self-correcting system.

By integrating quantitative feature awareness (interpreting brightness, contrast, highlights) and multi-iteration reasoning, the system bridges the gap between user intent and generated image fidelity, offering robust corrective feedback on composition, lighting, style, and technical attributes.

---

## 🔬 V1.0: Core Thesis Work (Qwen-IRG)

The foundation of this project is built upon fine-tuning open-source LLMs to mimic the critical eye of an art director or photographer.

### Key Innovations
* **Synthetic Dataset Generation**: Developed a robust system for generating high-quality training data that includes multi-iteration refinement sequences, feature interpretation, and problem-solving scenarios (e.g., diagnosing underexposure).
* **Feature-Aware Refinement**: Simulates a feedback loop where the LLM analyzes theoretical image features (derived from CLIP statistics) to propose concrete corrective actions.
* **Fine-tuned Qwen Model**: Trained a specialized reasoning model (Qwen 2.5) using PEFT/LoRA to decompose constraints into granular visual prompts for SDXL.

### System Architecture
![System Architecture](assets/system_architecture.png)
*(Fig: The Interleaving Reasoning-Generation system architecture illustrating the iterative workflow)*

### Empirical Findings
The modular reasoning approach was rigorously evaluated against standard single-shot generation (Base SDXL). The iterative loop consistently outperformed no-reasoning baselines:

* **Compositional Accuracy**: The 2-iteration reasoning loop achieves a **+7.74% improvement** in compositional accuracy (from 0.3497 to 0.3768) compared to the zero-shot baseline.
    ![Compositional Accuracy](assets/compositional_accuracy.png)
* **Aesthetic Improvement**: The refinement process yields monotonic improvements in visual quality, increasing the aesthetic score by up to **+3.08%** over the base model.
    ![Aesthetic Improvement](assets/aesthetic_improvement.png)

### Qualitative Examples
![Counting Task](assets/counting_task.png)
*(Fig: Visual comparison of precise counting task success achieved through iterative reasoning)*

---

## 🌟 V2.0: Autonomous Multi-Agent & RAG Architecture (Recent Evolution)

To push the boundaries of traditional prompt engineering, the V1.0 concept was fundamentally re-architected into a **closed-loop autonomous system**:

* **Multi-Agent Workflow**: Moved away from static generation to a fully automated pipeline. The workflow automatically evaluates image statistics via `image_service` and feeds them to an `ExpertAgent` (Gemini) for dynamic, heuristic adjustments.
* **Retrieval-Augmented Generation (RAG)**: Injects high-quality historical refinement cases as few-shot context to the LLM. Before generating a prompt, the system queries past successful cases to guide the Expert Agent's reasoning format and decision-making.
* **Structured Heuristic Diagnostics**: Implemented strict Regex parsing to force the LLM to output predictable formats containing precise actionable steps (`Diagnosis -> Actions -> Refined Prompt`).

---

## 📂 Repository Structure

* **`Workflow-CODE/` (V1.0 - Thesis Fine-tuning & Data)**
    * `irg-1-dataset-generation.ipynb`: Generates the comprehensive training dataset, simulating CLIP features.
    * `irg-2-qwen-finetuning.ipynb`: Implements the fine-tuning pipeline for Qwen using PEFT/LoRA.
    * `irg-imagegeneration.ipynb`: Demonstrates the baseline image generation process.
    * `phase3-benchmark.ipynb`: Benchmarking suite to evaluate the model's improvement metrics via GenEval.
    * `Final_check_2.pdf`: The complete thesis report documentation containing full methodology and experimental data.
* **`src/` (V2.0 - Core Architecture)**
    * `core/workflow.py`: The orchestrator handling the Multi-Agent loop and multi-iteration reasoning.
    * `agents/expert_agent.py`: The Gemini LLM Agent enforcing heuristic rules for diagnostics and refinement.
    * `services/rag_service.py`: Handles contextual data retrieval from historical runs (`dataset_final_v3.csv`).
    * `services/image_service.py`: Evaluates and generates images using the Stability AI API.

## 🛠️ Installation and Requirements

### Prerequisites
* Python 3.8 or higher
* CUDA-compatible GPU (Recommended: NVIDIA T4 or better for V1.0 fine-tuning)
* Valid API Keys for Gemini and Stability AI (for V2.0)

### Dependencies
Install the required packages based on the phase you are running:
```bash
# General / V2.0 Dependencies
pip install google-generativeai requests pandas numpy pillow

# V1.0 Fine-tuning Dependencies
pip install torch transformers peft accelerate datasets
```

## 🚀 Usage

### Phase 1: V1.0 Dataset Generation & Fine-tuning
1. **Dataset Generation**: Run `Workflow-CODE/irg-1-dataset-generation.ipynb` to create the training dataset. The output is a JSON/CSV file containing the structured reasoning sequences.
2. **Model Fine-Tuning**: Execute `Workflow-CODE/irg-2-qwen-finetuning.ipynb` to train the Qwen model. Ensure you have the base Qwen model downloaded. The script supports QLoRA for memory-efficient training on consumer GPUs.
3. **Inference & Benchmarking**: Use `irg-imagegeneration.ipynb` and `phase3-benchmark.ipynb` to run the baseline evaluation suite.

### Phase 2: V2.0 Autonomous Multi-Agent Generation
1. Configure your `.env` file with your `GEMINI_API_KEY` and `STABILITY_API_KEY`.
2. Ensure you have a valid RAG dataset (`dataset_final_v3.csv`) available.
3. Run the application via the updated multi-agent orchestrator:
   ```python
   # Example usage using the V2.0 architecture
   python main.py
   ```
   The `IRGWorkflow` will automatically retrieve historical RAG context, formulate an initial prompt, generate an image via Stability AI, extract its statistical properties, and iteratively refine it without human intervention.

## 🙏 Acknowledgments

This research makes use of the Qwen language model series by Alibaba Cloud and the PEFT library by Hugging Face. Special thanks to the open-source community for the tools enabling efficient LLM fine-tuning, and to FPT University for academic guidance.

## 📄 License

Copyright (c) 2025 Ngô Anh Dũng. This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. You may not use this work for commercial purposes.
