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

        unmatched = _named_documents_missing(state.get("doc_title_hints") or [], retrieved)
        if unmatched:
            names = ", ".join(unmatched)
            logger.info("Câu hỏi nêu đích danh %s nhưng kho không có bằng chứng từ đó.",
                        names)
            return {
                "grounding_score": 0.0,
                "is_sufficient": False,
                "verifier_feedback": f"Kho tri thức không có văn bản được hỏi đích danh: "
                                     f"{names}.",
                "trace": [trace_entry(self.name, grounding_score=0.0, is_sufficient=False,
                                      reason="named_document_missing",
                                      missing_documents=unmatched)],
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
        current_query = state.get("search_query") or state["question"]
        changed = not sufficient and _materially_different(rewritten, current_query)
        update["query_changed"] = changed
        if changed:
            update["search_query"] = rewritten
        logger.info("Verifier: score=%.2f sufficient=%s", score, sufficient)
        return update


def _clamp(value: object) -> float:
    try:
        score = float(value) 
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


def _named_documents_missing(hints: list[str], retrieved) -> list[str]:
    if not hints or not retrieved:
        return list(hints)
    available = [
        f"{item.chunk.doc_title} {item.chunk.doc_number}".strip().lower()
        for item in retrieved
    ]
    unmatched = [hint for hint in hints if not _matches_any(hint.lower(), available)]
    return unmatched if len(unmatched) == len(hints) else []


def _matches_any(hint: str, available: list[str]) -> bool:
    return any(hint in name or name in hint for name in available)


def _materially_different(candidate: str, current: str) -> bool:
    if not candidate:
        return False
    current_terms = _terms(current)
    candidate_terms = _terms(candidate)
    if not candidate_terms or candidate_terms == current_terms:
        return False
    overlap = len(candidate_terms & current_terms) / max(len(candidate_terms), 1)
    return overlap < 0.9


def _terms(text: str) -> set[str]:
    import re

    return set(re.sub(r"[^\w\s]", " ", text.lower(), flags=re.UNICODE).split())
