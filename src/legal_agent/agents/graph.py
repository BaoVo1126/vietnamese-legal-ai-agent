from __future__ import annotations
import time
from collections.abc import Callable
from langgraph.graph import END, START, StateGraph
from ..logging_config import get_logger
from .edges import (
    make_route_after_citation_check,
    make_route_after_verifier,
    route_after_router,
)
from .nodes import (
    AnswerNode,
    CitationCheckerNode,
    HybridRetrievalNode,
    KnowledgeGraphNode,
    RefusalNode,
    RouterNode,
    VerifierNode,
)
from .state import AgentState, AgentContext

logger = get_logger(__name__)


def instrument(name: str, node: Callable) -> Callable:
    def timed(state):
        started = time.perf_counter()
        update = node(state) or {}
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)
        entries = update.get("trace")
        if entries:
            update["trace"] = [{**entry, "elapsed_ms": elapsed_ms} for entry in entries]
        else:
            update["trace"] = [{"node": name, "elapsed_ms": elapsed_ms}]
        return update

    timed.__name__ = f"{name}_node"
    return timed


def build_agent_graph(context: AgentContext, checkpointer=None):
    graph = StateGraph(AgentState)

    graph.add_node("router", instrument("router", RouterNode(context)))
    graph.add_node("retrieve", instrument("retrieve", HybridRetrievalNode(context)))
    graph.add_node("kg_validate", instrument("kg_validate", KnowledgeGraphNode(context)))
    graph.add_node("verify", instrument("verify", VerifierNode(context)))
    graph.add_node("answer", instrument("answer", AnswerNode(context)))
    graph.add_node("citation_check", instrument("citation_check",
                                                CitationCheckerNode(context)))
    graph.add_node("refuse", instrument("refuse", RefusalNode()))

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", route_after_router,
                                {"retrieve": "retrieve", "refuse": "refuse"})
    graph.add_edge("retrieve", "kg_validate")
    graph.add_edge("kg_validate", "verify")
    graph.add_conditional_edges(
        "verify", make_route_after_verifier(context.settings),
        {"answer": "answer", "retrieve": "retrieve", "refuse": "refuse"},
    )
    graph.add_edge("answer", "citation_check")
    graph.add_conditional_edges(
        "citation_check", make_route_after_citation_check(context.settings),
        {END: END, "retrieve": "retrieve", "refuse": "refuse"},
    )
    graph.add_edge("refuse", END)

    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("LangGraph đã compile: 7 nodes, 3 conditional edges.")
    return compiled
