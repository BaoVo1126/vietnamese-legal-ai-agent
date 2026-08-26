from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, Query
from ...agents.service import LegalAgentService
from ...logging_config import get_logger
from ..deps import get_agent_service
from ..schemas import IngestRequest, IngestResponse

logger = get_logger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/ingest", response_model=IngestResponse, summary="Chạy lại pipeline ingestion")
def ingest(request: IngestRequest,
           service: LegalAgentService = Depends(get_agent_service)) -> IngestResponse:
    try:
        report = service.bootstrap(force_ingest=True)
    except Exception as error:  
        logger.exception("Ingestion thất bại")
        raise HTTPException(status_code=500, detail=str(error)) from error
    _ = request.recreate
    return IngestResponse(**{key: value for key, value in report.items()
                             if key in IngestResponse.model_fields})


@router.get("/documents", summary="Danh sách văn bản trong Knowledge Graph")
def documents(service: LegalAgentService = Depends(get_agent_service)) -> list[dict]:
    return [
        {
            "doc_number": document.doc_number,
            "title": document.title,
            "doc_type": document.doc_type.value,
            "effect_status": document.effect_status.value,
            "effective_date": document.effective_date,
            "expiry_date": document.expiry_date,
            "relations": len(document.relations),
        }
        for document in service.graph_store.all_documents()
    ]


@router.get("/validate", summary="Kiểm tra hiệu lực một văn bản/điều luật")
def validate(doc_number: str = Query(..., description="Số hiệu, vd 68/2014/QH13"),
             dieu: str | None = Query(None, description="Số Điều, nếu cần"),
             as_of: str | None = Query(None, description="Ngày tham chiếu YYYY-MM-DD"),
             service: LegalAgentService = Depends(get_agent_service)) -> dict:
    reference_date = service.context.as_of_date(as_of)
    return service.graph_store.validate(doc_number, dieu, reference_date).as_dict()
