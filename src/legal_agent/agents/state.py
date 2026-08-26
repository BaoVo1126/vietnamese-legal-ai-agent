from __future__ import annotations
import operator
from typing import Annotated, Any, Literal, TypedDict
from ..domain.chunk import RetrievedChunk

RunStatus = Literal["running", "answered", "refused"]


class AgentState(TypedDict, total=False):
    question: str
    session_id: str
    as_of: str | None                  

    intent: str
    rewritten_query: str
    sub_queries: list[str]
    doc_hints: list[str]
    dieu_hints: list[str]

    search_query: str                    
    retrieved: list[RetrievedChunk]

    graph_notes: list[str]
    graph_verdicts: list[dict[str, Any]]
    excluded_chunks: list[dict[str, Any]]

    grounding_score: float
    is_sufficient: bool
    verifier_feedback: str
    attempts: int

    answer: str
    citations: list[dict[str, Any]]
    claims: list[dict[str, Any]]
    claim_verdicts: list[dict[str, Any]]
    support_ratio: float

    status: RunStatus
    refusal_reason: str
    trace: Annotated[list[dict[str, Any]], operator.add]


def initial_state(question: str, session_id: str = "", as_of: str | None = None) -> AgentState:
    return AgentState(
        question=question.strip(),
        session_id=session_id,
        as_of=as_of,
        search_query=question.strip(),
        attempts=0,
        retrieved=[],
        graph_notes=[],
        graph_verdicts=[],
        excluded_chunks=[],
        citations=[],
        claims=[],
        claim_verdicts=[],
        status="running",
        trace=[],
    )


def trace_entry(node: str, **details: Any) -> dict[str, Any]:
    return {"node": node, **details}
