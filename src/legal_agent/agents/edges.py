from __future__ import annotations
from langgraph.graph import END
from ..config import Settings
from ..domain.enums import QueryIntent
from ..logging_config import get_logger
from .state import AgentState

logger = get_logger(__name__)


def route_after_router(state: AgentState) -> str:
    if state.get("intent") == QueryIntent.NGOAI_PHAM_VI.value:
        return "refuse"
    return "retrieve"


def make_route_after_verifier(settings: Settings):
    def route_after_verifier(state: AgentState) -> str:
        if state.get("is_sufficient"):
            return "answer"
        if state.get("attempts", 0) >= settings.max_retrieval_attempts:
            return "refuse"
        if not state.get("retrieved"):
            logger.info("Bỏ qua retry: lượt truy xuất không trả về bằng chứng nào.")
            return "refuse"
        if not state.get("query_changed"):
            logger.info("Bỏ qua retry: truy vấn viết lại không khác truy vấn vừa dùng.")
            return "refuse"
        logger.info("Self-correction: thử truy xuất lại (lần %d/%d).",
                    state.get("attempts", 0) + 1, settings.max_retrieval_attempts)
        return "retrieve"

    return route_after_verifier


def make_route_after_citation_check(settings: Settings):
    def route_after_citation_check(state: AgentState) -> str:
        if state.get("status") == "answered":
            return END
        if state.get("attempts", 0) < settings.max_retrieval_attempts:
            logger.info("Post-hoc gate thất bại - truy xuất lại với truy vấn gốc.")
            return "retrieve"
        return "refuse"

    return route_after_citation_check
