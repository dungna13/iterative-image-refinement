import logging
from ..services.gemini_service import GeminiService

logger = logging.getLogger(__name__)

IRG_CRITIC_SYSTEM = (
    "Act: IRG Critic. Task: Evaluate image quality stats after each refinement iteration.\n"
    "Rules for CLIP stats optimal ranges:\n"
    "- mean: 0.28–0.62 = Optimal | <0.28 = Under | >0.62 = Over\n"
    "- std:  0.12–0.28 = Optimal | <0.12 = Low-Con | >0.28 = High-Con\n"
    "- max:  0.70–0.97 = Optimal | <0.70 = No-High | >0.97 = Blown\n"
    "Response format (STRICT JSON-like, one line each):\n"
    "VERDICT: [ACCEPT or REFINE]\n"
    "SCORE: [0.0–1.0, overall quality]\n"
    "REASON: [one sentence explaining decision]"
)


class CriticAgent:
    """
    Evaluates the quality of a generated image based on its CLIP statistics.
    Acts as a quality gate — decides whether refinement should continue (REFINE)
    or if the current output is good enough (ACCEPT).
    """

    ACCEPT_THRESHOLD = 0.75  # SCORE >= this → stop early, no more refinement needed

    def __init__(self, gemini_service: GeminiService):
        self.gemini = gemini_service

    def evaluate(self, prompt: str, stats: dict, iteration: int) -> dict:
        """
        Evaluate image quality and return verdict.

        Returns:
            {
                "verdict": "ACCEPT" | "REFINE",
                "score": float,
                "reason": str,
                "should_continue": bool
            }
        """
        s = (
            f"mean={stats['mean']:.4f}, "
            f"std={stats['std']:.4f}, "
            f"max={stats['max']:.4f}"
        )
        msg = (
            f"Iteration {iteration} result for prompt: '{prompt}'\n"
            f"CLIP Stats: {s}\n"
            f"Task: Evaluate if this meets quality standards."
        )

        raw = self.gemini.generate(prompt=msg, context=IRG_CRITIC_SYSTEM)
        return self._parse_verdict(raw)

    def _parse_verdict(self, text: str) -> dict:
        """Parse structured response from Gemini into a usable dict."""
        import re
        result = {"verdict": "REFINE", "score": 0.5, "reason": text, "should_continue": True}

        verdict_match = re.search(r"VERDICT:\s*(ACCEPT|REFINE)", text, re.I)
        score_match = re.search(r"SCORE:\s*([0-9.]+)", text, re.I)
        reason_match = re.search(r"REASON:\s*(.+)", text, re.I)

        if verdict_match:
            result["verdict"] = verdict_match.group(1).upper()
        if score_match:
            result["score"] = float(score_match.group(1))
        if reason_match:
            result["reason"] = reason_match.group(1).strip()

        result["should_continue"] = (
            result["verdict"] == "REFINE"
            and result["score"] < self.ACCEPT_THRESHOLD
        )

        logger.info(
            f"CriticAgent → VERDICT={result['verdict']} | "
            f"SCORE={result['score']} | REASON={result['reason']}"
        )
        return result
