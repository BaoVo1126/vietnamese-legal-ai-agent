from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

from .citation import Citation
from .enums import DocumentType, EffectStatus


class LegalChunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_number: str
    doc_title: str = ""
    doc_type: DocumentType = DocumentType.KHAC

    chuong: str = Field("", description="Nhãn chương, e.g. 'Chương II. Thành lập doanh nghiệp'")
    muc: str = ""
    dieu: str | None = None
    dieu_title: str = ""
    khoan: str | None = None
    diem: str | None = None
    node_path: str = Field("", description="Đường dẫn cấu trúc đầy đủ, phục vụ debug/UI")

    text: str = Field(..., description="Nội dung pháp lý thuần, dùng để hiển thị & verify")
    context_header: str = Field("", description="Ngữ cảnh cha, chỉ ghép vào text khi embed")

    effect_status: EffectStatus = EffectStatus.KHONG_XAC_DINH
    effective_date: date | None = None
    expiry_date: date | None = None
    issuing_body: str = ""
    field_of_law: str = ""

    @property
    def doc_key(self) -> str:
        return self.doc_number or self.doc_title

    @property
    def citation(self) -> Citation:
        return Citation(
            doc_number=self.doc_number,
            doc_title=self.doc_title,
            dieu=self.dieu,
            khoan=self.khoan,
            diem=self.diem,
        )

    @property
    def embed_text(self) -> str:
        return f"{self.context_header}\n{self.text}".strip()

    def to_evidence_block(self, index: int) -> str:
        status = self.effect_status.display_name
        lines = [f"[{index}] {self.citation.render()} ({status})"]
        if self.context_header:
            lines.append(f"Ngữ cảnh: {self.context_header}")
        lines.append(f"Nội dung: {self.text}")
        return "\n".join(lines)

    def to_payload(self) -> dict:
        payload = self.model_dump(mode="json")
        payload["citation"] = self.citation.render()
        payload["doc_key"] = self.doc_key
        return payload

    @classmethod
    def from_payload(cls, payload: dict) -> LegalChunk:
        data = {k: v for k, v in payload.items() if k in cls.model_fields}
        return cls.model_validate(data)


class RetrievedChunk(BaseModel):
    chunk: LegalChunk
    dense_score: float | None = None
    sparse_score: float | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None
    fusion_score: float = 0.0
    rerank_score: float | None = None
    graph_boost: float = 0.0
    source: str = Field("hybrid", description="dense | sparse | hybrid | graph_expansion")
    graph_note: str = Field("", description="Ghi chú từ KG, e.g. 'được hướng dẫn bởi ...'")

    @property
    def final_score(self) -> float:
        base = self.rerank_score if self.rerank_score is not None else self.fusion_score
        return base + self.graph_boost

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id
