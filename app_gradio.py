"""
app_gradio.py — Gradio Web UI for IRG Multi-Agent Image Refinement

Launch: python app_gradio.py
Opens a browser-based demo at http://localhost:7860
"""

import os
import time
import gradio as gr
from src.config import PipelineConfig
from src.services.gemini_service import GeminiService
from src.services.image_service import ImageService
from src.services.rag_service import RAGService
from src.agents.expert_agent import ExpertAgent
from src.agents.critic_agent import CriticAgent
from src.core.workflow import IRGWorkflow

# ── Initialize Pipeline ──────────────────────────────────────────────
config = PipelineConfig()
gemini_service = GeminiService(config.gemini)
image_service = ImageService(config.image_gen)
rag_service = RAGService(config.rag_dataset_file)
expert_agent = ExpertAgent(gemini_service)
critic_agent = CriticAgent(gemini_service)
workflow = IRGWorkflow(expert_agent, image_service, rag_service, critic_agent)

# Load services at startup
gemini_service.load()
image_service.load()
rag_service.load()


def run_refinement(prompt: str, iterations: int, progress=gr.Progress()):
    """Run the IRG refinement pipeline and return results for the Gradio UI."""
    if not prompt.strip():
        raise gr.Error("Please enter a prompt!")

    iterations = int(iterations)
    t_start = time.time()

    progress(0, desc="🚀 Starting IRG pipeline...")

    # Run the multi-agent workflow
    results = workflow.run_refinement(prompt, iterations=iterations)

    elapsed = round(time.time() - t_start, 2)

    # ── Build outputs ─────────────────────────────────────────────
    images = []
    log_lines = []

    for res in results:
        i = res["iteration"]
        img = res["image"]
        resp = res.get("response", {})
        critic = res.get("critic")
        stats = res.get("stats", {})

        # Collect image
        images.append((img, f"Iteration {i}"))

        # Build log
        if i == 0:
            log_lines.append(f"### Iteration 0 — Initial Generation")
            if resp.get("refined_prompt"):
                log_lines.append(f"**Refined Prompt:** {resp['refined_prompt']}")
        else:
            log_lines.append(f"---\n### Iteration {i}")
            if stats:
                log_lines.append(
                    f"📊 Stats: mean={stats.get('mean', 0):.4f}, "
                    f"std={stats.get('std', 0):.4f}, "
                    f"max={stats.get('max', 0):.4f}"
                )
            if resp.get("issues") and resp["issues"] != "none":
                log_lines.append(f"🔍 **Issues:** {resp['issues']}")
            if resp.get("actions") and resp["actions"] != "none":
                log_lines.append(f"🔧 **Actions:** {resp['actions']}")
            if resp.get("refined_prompt"):
                log_lines.append(f"✏️ **Refined Prompt:** {resp['refined_prompt']}")
            if critic:
                verdict = critic.get("verdict", "?")
                score = critic.get("score", 0)
                reason = critic.get("reason", "")
                emoji = "✅" if verdict == "ACCEPT" else "🔄"
                log_lines.append(f"{emoji} **Critic:** {verdict} (score={score:.2f}) — {reason}")

        progress((i + 1) / (iterations + 1), desc=f"Iteration {i} complete")

    # Summary
    final_prompt = results[-1].get("response", {}).get("refined_prompt", prompt)
    total_iters = len(results) - 1
    summary = (
        f"## ✅ Complete\n"
        f"- **Iterations:** {total_iters}\n"
        f"- **Time:** {elapsed}s\n"
        f"- **Final Prompt:** {final_prompt}\n"
    )

    log_text = summary + "\n\n" + "\n\n".join(log_lines)

    return images, log_text


# ── Gradio Interface ──────────────────────────────────────────────────

css = """
.gradio-container { max-width: 1200px !important; }
#title { text-align: center; margin-bottom: 0; }
#subtitle { text-align: center; color: #666; margin-top: 0; font-size: 1.1rem; }
"""

with gr.Blocks(
    title="IRG — Iterative Reasoning-Generation",
) as demo:

    gr.Markdown("# 🎨 IRG — Iterative Reasoning-Generation", elem_id="title")
    gr.Markdown(
        "Autonomous multi-agent AI that **thinks, generates, critiques, and refines** images iteratively.",
        elem_id="subtitle",
    )

    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(
                label="📝 Your Prompt",
                placeholder="e.g. a medieval knight fighting a dragon at golden hour...",
                lines=3,
            )
            iterations_slider = gr.Slider(
                minimum=1,
                maximum=5,
                value=2,
                step=1,
                label="🔄 Refinement Iterations",
                info="More iterations = better quality, but slower & more API calls",
            )
            run_btn = gr.Button("🚀 Generate & Refine", variant="primary", size="lg")

            gr.Markdown("### 💡 Try These Prompts")
            gr.Examples(
                examples=[
                    ["a fluffy orange cat sitting on a windowsill, warm afternoon sunlight", 2],
                    ["a medieval knight fighting a dragon at golden hour, cinematic", 3],
                    ["futuristic cyberpunk city at night, neon reflections on wet streets", 3],
                    ["a serene Japanese garden with cherry blossoms, watercolor painting style", 2],
                    ["astronaut riding a horse on Mars, realistic photography", 2],
                ],
                inputs=[prompt_input, iterations_slider],
            )

        with gr.Column(scale=2):
            gallery = gr.Gallery(
                label="🖼️ Refinement Progress",
                columns=3,
                height=500,
                object_fit="contain",
            )
            log_output = gr.Markdown(label="📋 Agent Log")

    run_btn.click(
        fn=run_refinement,
        inputs=[prompt_input, iterations_slider],
        outputs=[gallery, log_output],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        share=False,
        theme=gr.themes.Soft(
            primary_hue="violet",
            secondary_hue="blue",
        ),
        css=css,
    )
