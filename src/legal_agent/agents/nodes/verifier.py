from __future__ import annotations
from ...llm.prompts import VERIFIER_SYSTEM, VERIFIER_USER, format_evidence
from ...logging_config import get_logger
from ..state import AgentState, trace_entry
from .base import AgentContext

logger = get_logger(__name__)


class VerifierNode:
    name = "verify"

    def __init__(self, context: AgentContext) -> None:
        self.context = context

    def __call__(self, state: AgentState) -> dict:
        retrieved = state.get("retrieved") or []
        threshold = self.context.settings.grounding_threshold

        if not retrieved:
            return {
                "grounding_score": 0.0,
                "is_sufficient": False,
                "verifier_feedback": "Không truy xuất được điều khoản nào liên quan.",
                "trace": [trace_entry(self.name, grounding_score=0.0, is_sufficient=False,
                                      reason="empty_evidence")],
            }

        payload = self.context.llm.complete_json(
            VERIFIER_SYSTEM,
            VERIFIER_USER.format(question=state["question"],
                                 evidence=format_evidence(retrieved)),
            task="verifier",
            default={"grounding_score": 0.0, "is_sufficient": False,
                     "missing_information": "Không đánh giá được bằng chứng.",
                     "rewritten_query": ""},
        )
        score = _clamp(payload.get("grounding_score"))
        sufficient = bool(payload.get("is_sufficient", score >= threshold)) and score >= threshold
        feedback = str(payload.get("missing_information") or "").strip()
        rewritten = str(payload.get("rewritten_query") or "").strip()

        update: dict = {
            "grounding_score": score,
            "is_sufficient": sufficient,
            "verifier_feedback": feedback,
            "trace": [trace_entry(self.name, grounding_score=score, is_sufficient=sufficient,
                                  missing=feedback, rewritten_query=rewritten)],
        }
        if not sufficient and rewritten:
            update["search_query"] = rewritten
        logger.info("Verifier: score=%.2f sufficient=%s", score, sufficient)
        return update


def _clamp(value: object) -> float:
    try:
        score = float(value) 
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))
