import re
import logging
from PIL import Image
from ..agents.expert_agent import ExpertAgent
from ..agents.critic_agent import CriticAgent
from ..services.image_service import ImageService
from ..services.rag_service import RAGService

logger = logging.getLogger(__name__)

class IRGWorkflow:
    def __init__(self, expert_agent: ExpertAgent, image_service: ImageService, rag_service: RAGService, critic_agent: CriticAgent = None):
        self.expert = expert_agent
        self.image_service = image_service
        self.rag = rag_service
        self.critic = critic_agent  # Optional: enables closed-loop early stopping

    def parse_response(self, text: str) -> dict:
        """Bóc tách phản hồi từ Expert Agent."""
        data = {"issues": "none", "actions": "none", "refined_prompt": "", "full_text": text}
        
        diag_match = re.search(r"DIAGNOSIS:\s*(.*?)(?=ACTIONS:|$)", text, re.I | re.S)
        act_match = re.search(r"ACTIONS:\s*(.*?)(?=REFINED PROMPT:|$)", text, re.I | re.S)
        prompt_match = re.search(r"REFINED PROMPT:\s*(.*)", text, re.I | re.S)
        
        if diag_match: data["issues"] = diag_match.group(1).strip()
        if act_match: data["actions"] = act_match.group(1).strip()
        if prompt_match: data["refined_prompt"] = prompt_match.group(1).strip()
        
        return data

    def run_refinement(self, prompt: str, iterations: int = 2):
        """Luồng điều phối chính: ExpertAgent → Refine → CriticAgent → (loop)."""
        results = []

        # 0. RAG: Lấy ngữ cảnh lịch sử
        logger.info("Workflow: Retrieving historical context (RAG)...")
        print("Workflow: Retrieving historical context (RAG)...")
        rag_context = self.rag.query(prompt)

        # 1. Expert phân tích ban đầu
        logger.info("Workflow: Starting initialization...")
        print("Workflow: Starting initialization...")
        init_res = self.expert.analyze_initial_prompt(prompt, rag_context=rag_context)
        init_data = self.parse_response(init_res)

        # 2. Sinh ảnh gốc
        current_prompt = init_data["refined_prompt"] or prompt
        current_image = self.image_service.generate(current_prompt, seed=42)

        results.append({
            "iteration": 0,
            "image": current_image,
            "response": init_data
        })

        for i in range(1, iterations + 1):
            logger.info(f"Workflow: Iteration {i}/{iterations}...")
            print(f"Workflow: Iteration {i}/{iterations}...")

            # Đo CLIP stats
            stats = self.image_service.get_stats(current_image)

            # Expert đưa ra quyết định tinh chỉnh
            refinement_res = self.expert.analyze_feedback(prompt, stats, i, rag_context=rag_context)
            refinement_data = self.parse_response(refinement_res)

            # Thực hiện tinh chỉnh
            current_prompt = refinement_data["refined_prompt"] or current_prompt
            current_image = self.image_service.refine(
                image=current_image,
                prompt=current_prompt,
                strength=0.35,
                seed=42 + i
            )

            # CriticAgent: đánh giá chất lượng sau refinement
            critic_verdict = None
            if self.critic:
                refined_stats = self.image_service.get_stats(current_image)
                critic_verdict = self.critic.evaluate(prompt, refined_stats, i)
                verdict = critic_verdict["verdict"]
                score = critic_verdict["score"]
                print(f"CriticAgent → {verdict} (score={score:.2f}): {critic_verdict['reason']}")

            results.append({
                "iteration": i,
                "image": current_image,
                "response": refinement_data,
                "stats": stats,
                "critic": critic_verdict
            })

            # Early stopping: nếu Critic ACCEPT → không cần vòng lặp tiếp
            if critic_verdict and not critic_verdict["should_continue"]:
                logger.info(f"CriticAgent ACCEPTED at iteration {i}. Early stopping.")
                print(f"✅ CriticAgent: Quality threshold reached. Stopping early at iteration {i}.")
                break

        return results
