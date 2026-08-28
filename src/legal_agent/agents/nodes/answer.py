from __future__ import annotations

from ...domain.citation import Citation
from ...llm.prompts import ANSWER_SYSTEM, ANSWER_USER, format_evidence, format_graph_context
from ...logging_config import get_logger
from ..state import AgentState, trace_entry
from .base import AgentContext

logger = get_logger(__name__)

_REFUSAL_MARKERS = (
    "không đủ căn cứ pháp lý",
    "không đủ căn cứ",
    "không tìm thấy quy định",
)


class AnswerNode:
    name = "answer"

    def __init__(self, context: AgentContext) -> None:
        self.context = context

    def __call__(self, state: AgentState) -> dict:
        retrieved = state.get("retrieved") or []
        if not retrieved:
            return {
                "answer": "",
                "refusal_reason": "Không còn bằng chứng nào sau bước kiểm tra hiệu lực.",
                "trace": [trace_entry(self.name, skipped="empty_evidence")],
            }

        draft = self.context.llm.complete(
            ANSWER_SYSTEM,
            ANSWER_USER.format(
                question=state["question"],
                evidence=format_evidence(retrieved),
                graph_context=format_graph_context(state.get("graph_notes") or []),
            ),
            task="answer",
        ).strip()

        if self._is_refusal(draft):
            logger.info("Answer agent tự từ chối vì bằng chứng không đủ.")
            return {
                "answer": draft,
                "refusal_reason": "Mô hình xác định bằng chứng không đủ để kết luận.",
                "citations": [],
                "trace": [trace_entry(self.name, self_refused=True)],
            }

        citations = Citation.parse_cited(draft)
        logger.info("Answer: %d ký tự, %d trích dẫn", len(draft), len(citations))
        return {
            "answer": draft,
            "citations": [citation.model_dump() for citation in citations],
            "trace": [trace_entry(self.name, length=len(draft),
                                  citations=[citation.render() for citation in citations])],
        }

    @staticmethod
    def _is_refusal(draft: str) -> bool:
        lowered = draft.lower()
        return not draft or any(marker in lowered for marker in _REFUSAL_MARKERS)
