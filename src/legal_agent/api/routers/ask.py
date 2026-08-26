from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from ...agents.service import AgentAnswer, LegalAgentService
from ...domain.citation import Citation
from ...logging_config import get_logger
from ..deps import get_agent_service
from ..schemas import AskRequest, AskResponse

logger = get_logger(__name__)
router = APIRouter(tags=["ask"])


@router.post("/ask", response_model=AskResponse, summary="Hỏi đáp pháp luật có trích dẫn")
def ask(request: AskRequest,
        service: LegalAgentService = Depends(get_agent_service)) -> AskResponse:
    try:
        answer = service.ask(request.question, session_id=request.session_id,
                             as_of=request.as_of)
    except Exception as error:  
        logger.exception("Lỗi khi xử lý câu hỏi")
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {error}") from error
    return _to_response(answer, include_trace=request.include_trace)


def _to_response(answer: AgentAnswer, include_trace: bool) -> AskResponse:
    return AskResponse(
        question=answer.question,
        answer=answer.answer,
        status=answer.status,
        intent=answer.intent,
        is_refusal=answer.is_refusal,
        citations=[Citation.model_validate(item).render() for item in answer.citations],
        evidence=answer.evidence,
        graph_notes=answer.graph_notes,
        excluded_chunks=answer.excluded_chunks,
        grounding_score=answer.grounding_score,
        support_ratio=answer.support_ratio,
        attempts=answer.attempts,
        refusal_reason=answer.refusal_reason,
        latency_ms=answer.latency_ms,
        trace=answer.trace if include_trace else None,
    )
