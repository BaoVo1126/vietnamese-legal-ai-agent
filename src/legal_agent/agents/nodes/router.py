from __future__ import annotations
from ...domain.enums import QueryIntent
from ...llm.prompts import ROUTER_SYSTEM, ROUTER_USER
from ...logging_config import get_logger
from ..state import AgentState, trace_entry
from .base import AgentContext

logger = get_logger(__name__)

_DEFAULT_ROUTE = {
    "intent": QueryIntent.HOI_DAP_KHAI_NIEM.value,
    "rewritten_query": "",
    "sub_queries": [],
    "doc_numbers": [],
    "doc_titles": [],
    "dieu_hints": [],
    "reasoning": "",
}


class RouterNode:
    name = "router"

    def __init__(self, context: AgentContext) -> None:
        self.context = context

    def __call__(self, state: AgentState) -> dict:
        question = state["question"]
        payload = self.context.llm.complete_json(
            ROUTER_SYSTEM, ROUTER_USER.format(question=question),
            task="router", default=dict(_DEFAULT_ROUTE),
        )
        intent = self._parse_intent(payload.get("intent"))
        rewritten = (payload.get("rewritten_query") or question).strip()
        sub_queries = [str(item).strip() for item in payload.get("sub_queries", [])
                       if str(item).strip()]

        logger.info("Router: intent=%s | query=%r", intent.value, rewritten)
        return {
            "intent": intent.value,
            "rewritten_query": rewritten,
            "search_query": rewritten,
            "sub_queries": sub_queries[:3],
            "doc_hints": [str(item) for item in payload.get("doc_numbers", []) if item],
            "doc_title_hints": [str(item).strip()
                                for item in payload.get("doc_titles", []) if item],
            "dieu_hints": [str(item) for item in payload.get("dieu_hints", []) if item],
            "trace": [trace_entry(self.name, intent=intent.value, rewritten_query=rewritten,
                                 sub_queries=sub_queries[:3],
                                 reasoning=payload.get("reasoning", ""))],
        }

    @staticmethod
    def _parse_intent(raw: object) -> QueryIntent:
        try:
            return QueryIntent(str(raw))
        except ValueError:
            logger.warning("Intent không hợp lệ từ LLM: %r - dùng mặc định.", raw)
            return QueryIntent.HOI_DAP_KHAI_NIEM
