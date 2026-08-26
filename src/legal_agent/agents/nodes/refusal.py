from __future__ import annotations

from ...llm.prompts import DISCLAIMER, REFUSAL_TEMPLATE
from ...logging_config import get_logger
from ..state import AgentState, trace_entry

logger = get_logger(__name__)


class RefusalNode:
    name = "refuse"

    def __call__(self, state: AgentState) -> dict:
        reason = (state.get("refusal_reason")
                  or state.get("verifier_feedback")
                  or "Không tìm thấy điều khoản còn hiệu lực nào đủ căn cứ cho câu hỏi.")
        excluded = state.get("excluded_chunks") or []
        message = REFUSAL_TEMPLATE.format(reason=reason)
        if excluded:
            listed = "\n".join(f"- {item['citation']} ({item['reason']})" for item in excluded)
            message += f"\n\nCác điều khoản đã bị loại vì lý do hiệu lực:\n{listed}"
        message += f"\n\n{DISCLAIMER}"

        logger.info("Refusal: %s", reason)
        return {
            "answer": message,
            "status": "refused",
            "refusal_reason": reason,
            "citations": [],
            "trace": [trace_entry(self.name, reason=reason, excluded=len(excluded))],
        }
