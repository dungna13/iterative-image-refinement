from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import uuid
import asyncio
import time
import logging
from ..config import PipelineConfig
from ..services.gemini_service import GeminiService
from ..services.image_service import ImageService
from ..services.rag_service import RAGService
from ..agents.expert_agent import ExpertAgent
from ..agents.critic_agent import CriticAgent
from ..core.workflow import IRGWorkflow

logger = logging.getLogger(__name__)

app = FastAPI(title="IRG-Thesis Multi-Agent API", version="3.0.0")

# Cấu hình Singletons
config = PipelineConfig()
gemini_service = GeminiService(config.gemini)
image_service = ImageService(config.image_gen)
rag_service = RAGService(config.rag_dataset_file)
expert_agent = ExpertAgent(gemini_service)
critic_agent = CriticAgent(gemini_service)  # NEW
workflow = IRGWorkflow(expert_agent, image_service, rag_service, critic_agent)

class RefinementRequest(BaseModel):
    prompt: str
    iterations: Optional[int] = 2

class IterationSummary(BaseModel):
    iteration: int
    issues: Optional[str] = None
    actions: Optional[str] = None
    refined_prompt: Optional[str] = None
    clip_mean: Optional[float] = None
    clip_std: Optional[float] = None
    clip_max: Optional[float] = None

class RefinementResponse(BaseModel):
    request_id: str
    status: str
    total_iterations: int
    final_refined_prompt: str
    execution_time_seconds: float
    iterations_summary: List[IterationSummary]
    output_dir: str

@app.on_event("startup")
async def startup_event():
    gemini_service.load()
    image_service.load()
    rag_service.load()

@app.get("/")
def read_root():
    return {"status": "online", "version": app.version, "engine": config.image_gen.engine_id}

@app.post("/refine", response_model=RefinementResponse)
async def start_refinement(request: RefinementRequest):
    """Endpoint khởi chạy quy trình Agentic Refinement (Non-blocking)."""
    request_id = str(uuid.uuid4())
    t_start = time.time()

    try:
        # Đẩy blocking workflow vào thread pool → không block event loop
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: workflow.run_refinement(request.prompt, iterations=request.iterations)
        )

        # Lưu ảnh kết quả
        output_dir = os.path.join(config.output_dir, request_id)
        os.makedirs(output_dir, exist_ok=True)
        for res in results:
            res["image"].save(os.path.join(output_dir, f"iter_{res['iteration']}.png"))

        # Tổng hợp thống kê từng vòng
        summary = []
        final_prompt = request.prompt
        for res in results:
            stats = res.get("stats", {})
            resp = res.get("response", {})
            if resp.get("refined_prompt"):
                final_prompt = resp["refined_prompt"]
            summary.append(IterationSummary(
                iteration=res["iteration"],
                issues=resp.get("issues"),
                actions=resp.get("actions"),
                refined_prompt=resp.get("refined_prompt"),
                clip_mean=round(stats.get("mean", 0), 4) if stats else None,
                clip_std=round(stats.get("std", 0), 4) if stats else None,
                clip_max=round(stats.get("max", 0), 4) if stats else None,
            ))

        exec_time = round(time.time() - t_start, 2)
        logger.info(f"[{request_id}] Completed {len(results)-1} iterations in {exec_time}s")

        return RefinementResponse(
            request_id=request_id,
            status="completed",
            total_iterations=len(results) - 1,
            final_refined_prompt=final_prompt,
            execution_time_seconds=exec_time,
            iterations_summary=summary,
            output_dir=output_dir
        )

    except Exception as e:
        logger.error(f"[{request_id}] Error in /refine: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
