from __future__ import annotations

import hashlib
import re
from datetime import date

from pydantic import BaseModel, Field

from .enums import DocumentType, EffectStatus, RelationType


def make_doc_id(identity: str) -> str:
    slug = re.sub(r"[^\w]+", "-", identity, flags=re.UNICODE).strip("-")
    return slug or "doc-" + hashlib.sha1(identity.encode("utf-8")).hexdigest()[:10]


class LegalRelation(BaseModel):
    source_doc_id: str
    target_doc_id: str
    relation: RelationType
    source_dieu: str | None = None
    target_dieu: str | None = None
    evidence: str = Field("", description="Câu văn gốc dùng để suy ra quan hệ")
    confidence: float = Field(1.0, ge=0.0, le=1.0)

    model_config = {"frozen": True}

    @property
    def key(self) -> tuple[str, str, str, str, str]:
        return (
            self.source_doc_id,
            self.target_doc_id,
            self.relation.value,
            self.source_dieu or "",
            self.target_dieu or "",
        )


class LegalDocumentMeta(BaseModel):
    doc_id: str
    doc_number: str = Field("", description="Số hiệu, e.g. '59/2020/QH14'")
    doc_type: DocumentType = DocumentType.KHAC
    title: str = ""
    issuing_body: str = Field("", description="Cơ quan ban hành")
    signer: str = ""
    issued_date: date | None = None
    effective_date: date | None = None
    expiry_date: date | None = None
    effect_status: EffectStatus = EffectStatus.KHONG_XAC_DINH
    field_of_law: str = Field("", description="Lĩnh vực, e.g. 'Doanh nghiệp'")
    source_path: str = ""
    relations: list[LegalRelation] = Field(default_factory=list)

    @property
    def doc_key(self) -> str:
        return self.doc_number or self.title

    @property
    def display_name(self) -> str:
        title = self.title or self.doc_type.display_name
        return f"{title} {self.doc_number}".strip()

    def status_as_of(self, today: date | None = None) -> EffectStatus:
        if self.effect_status in {EffectStatus.HET_HIEU_LUC, EffectStatus.HET_HIEU_LUC_MOT_PHAN}:
            return self.effect_status
        today = today or date.today()
        if self.expiry_date and self.expiry_date <= today:
            return EffectStatus.HET_HIEU_LUC
        if self.effective_date and self.effective_date > today:
            return EffectStatus.CHUA_CO_HIEU_LUC
        if self.effective_date and self.effective_date <= today:
            return EffectStatus.CON_HIEU_LUC
        return self.effect_status
