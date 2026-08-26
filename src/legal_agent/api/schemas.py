from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=2000,
                          description="Câu hỏi pháp luật bằng tiếng Việt")
    session_id: str = Field("", max_length=64)
    as_of: str | None = Field(None, description="Ngày tham chiếu hiệu lực (YYYY-MM-DD)")
    include_trace: bool = Field(False, description="Trả về trace của LangGraph để debug")


class EvidenceItem(BaseModel):
    citation: str
    effect_status: str
    score: float
    source: str
    graph_note: str = ""
    text: str
    node_path: str = ""


class ExcludedItem(BaseModel):
    citation: str
    reason: str


class AskResponse(BaseModel):
    question: str
    answer: str
    status: str = Field(..., description="answered | refused")
    intent: str = ""
    is_refusal: bool = False
    citations: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    graph_notes: list[str] = Field(default_factory=list)
    excluded_chunks: list[ExcludedItem] = Field(default_factory=list)
    grounding_score: float = 0.0
    support_ratio: float = 0.0
    attempts: int = 0
    refusal_reason: str = ""
    latency_ms: float = 0.0
    trace: list[dict[str, Any]] | None = None


class MetricsResponse(BaseModel):
    total_runs: int = 0
    answered: int = 0
    refused: int = 0
    errors: int = 0
    refusal_rate: float = 0.0
    retry_rate: float = 0.0
    avg_grounding: float = 0.0
    avg_support: float = 0.0
    avg_evidence: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_max_ms: float = 0.0
    node_latency_ms: dict[str, float] = Field(default_factory=dict)
    intents: dict[str, int] = Field(default_factory=dict)
    top_cited_documents: list[tuple[str, int]] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    profile: str
    llm_backend: str
    embedding_backend: str
    graph_backend: str
    qdrant_mode: str
    indexed_chunks: int = 0
    graph_documents: int = 0
    tracing: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    recreate: bool = Field(True, description="Xoá và tạo lại collection trước khi nạp")


class IngestResponse(BaseModel):
    mode: str
    documents: int = 0
    chunks: int = 0
    warnings: int = 0
    doc_numbers: list[str] = Field(default_factory=list)
