from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from typing import Protocol, runtime_checkable

from ..domain.document import LegalDocumentMeta, LegalRelation
from ..domain.enums import EffectStatus, RelationType


@dataclass
class GraphVerdict:
    doc_number: str
    status: EffectStatus
    replaced_by: list[str] = field(default_factory=list)
    amended_by: list[str] = field(default_factory=list)
    guided_by: list[tuple[str, str | None]] = field(default_factory=list)
    note: str = ""

    @property
    def is_citable(self) -> bool:
        return self.status.is_citable

    def as_dict(self) -> dict:
        return {
            "doc_number": self.doc_number,
            "status": self.status.value,
            "status_label": self.status.display_name,
            "replaced_by": self.replaced_by,
            "amended_by": self.amended_by,
            "guided_by": [
                {"doc_number": number, "dieu": dieu} for number, dieu in self.guided_by
            ],
            "note": self.note,
        }


@runtime_checkable
class LegalGraphStore(Protocol):
    def clear(self) -> None: ...

    def upsert_document(self, meta: LegalDocumentMeta) -> None: ...

    def upsert_relation(self, relation: LegalRelation) -> None: ...

    def get_document(self, doc_number: str) -> LegalDocumentMeta | None: ...

    def all_documents(self) -> list[LegalDocumentMeta]: ...

    def neighbours(self, doc_number: str, relation: RelationType) -> list[LegalRelation]: ...

    def incoming(self, doc_number: str, relation: RelationType) -> list[LegalRelation]: ...

    def validate(self, doc_number: str, dieu: str | None = None,
                 as_of: date | None = None) -> GraphVerdict: ...

    def close(self) -> None: ...
