from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from ...agents.service import LegalAgentService
from ..deps import get_agent_service
from ..schemas import MetricsResponse

router = APIRouter(tags=["monitoring"])


@router.get("/metrics", response_model=MetricsResponse,
            summary="Chỉ số chất lượng & độ trễ theo cửa sổ gần nhất")
def metrics(limit: int | None = Query(None, ge=1, le=10000,
                                      description="Chỉ tính N lượt hỏi gần nhất"),
            service: LegalAgentService = Depends(get_agent_service)) -> MetricsResponse:
    return MetricsResponse(**service.metrics(limit=limit).as_dict())


@router.get("/runs", summary="Nhật ký các lượt hỏi gần nhất")
def runs(limit: int = Query(20, ge=1, le=500),
         service: LegalAgentService = Depends(get_agent_service)) -> list[dict]:
    return service.recorder.read_all(limit=limit)[::-1]
