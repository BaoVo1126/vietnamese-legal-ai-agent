from __future__ import annotations

from fastapi import APIRouter, Depends

from ...agents.service import LegalAgentService
from ...config import get_settings
from ..deps import get_agent_service
from ..schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Trạng thái hệ thống")
def health(service: LegalAgentService = Depends(get_agent_service)) -> HealthResponse:
    settings = get_settings()
    try:
        indexed = service.vector_store.count()
    except Exception: 
        indexed = 0
    return HealthResponse(
        status="ok",
        profile=settings.app_profile,
        llm_backend=settings.llm_backend,
        embedding_backend=settings.embedding_backend,
        graph_backend=settings.graph_backend,
        qdrant_mode=settings.qdrant_mode,
        indexed_chunks=indexed,
        graph_documents=len(service.graph_store.all_documents()),
        tracing=getattr(service, "tracing", {}),
    )


@router.get("/live", summary="Liveness probe")
def live() -> dict:
    return {"status": "alive"}
