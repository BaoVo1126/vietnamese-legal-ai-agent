from __future__ import annotations
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

from ..domain.document import LegalDocumentMeta, LegalRelation, make_doc_id
from ..domain.enums import EffectStatus, RelationType
from ..logging_config import get_logger
from .base import GraphVerdict

logger = get_logger(__name__)


class MemoryGraphStore:
    def __init__(self) -> None:
        self._documents: dict[str, LegalDocumentMeta] = {}
        self._number_to_id: dict[str, str] = {}
        self._out_edges: dict[str, list[LegalRelation]] = defaultdict(list)
        self._in_edges: dict[str, list[LegalRelation]] = defaultdict(list)

    #  write
    def clear(self) -> None:
        self._documents.clear()
        self._number_to_id.clear()
        self._out_edges.clear()
        self._in_edges.clear()

    def upsert_document(self, meta: LegalDocumentMeta) -> None:
        self._documents[meta.doc_id] = meta
        for key in (meta.doc_number, meta.title):
            if key:
                self._number_to_id[key] = meta.doc_id

    def upsert_relation(self, relation: LegalRelation) -> None:
        if relation not in self._out_edges[relation.source_doc_id]:
            self._out_edges[relation.source_doc_id].append(relation)
        if relation not in self._in_edges[relation.target_doc_id]:
            self._in_edges[relation.target_doc_id].append(relation)

    #  read
    def get_document(self, doc_key: str) -> LegalDocumentMeta | None:
        doc_id = self._number_to_id.get(doc_key) or make_doc_id(doc_key)
        return self._documents.get(doc_id)

    def all_documents(self) -> list[LegalDocumentMeta]:
        return list(self._documents.values())

    def neighbours(self, doc_number: str, relation: RelationType) -> list[LegalRelation]:
        doc_id = self._number_to_id.get(doc_number, make_doc_id(doc_number))
        return [edge for edge in self._out_edges[doc_id] if edge.relation is relation]

    def incoming(self, doc_number: str, relation: RelationType) -> list[LegalRelation]:
        doc_id = self._number_to_id.get(doc_number, make_doc_id(doc_number))
        return [edge for edge in self._in_edges[doc_id] if edge.relation is relation]

    def _number_of(self, doc_id: str) -> str:
        document = self._documents.get(doc_id)
        return document.doc_key if document else doc_id.replace("-", "/")

    def validate(self, doc_number: str, dieu: str | None = None,
                 as_of: date | None = None) -> GraphVerdict:
        document = self.get_document(doc_number)
        status = document.status_as_of(as_of) if document else EffectStatus.KHONG_XAC_DINH

        replaced_by = [self._number_of(edge.source_doc_id)
                       for edge in self.incoming(doc_number, RelationType.THAY_THE)]
        amended_by = [self._number_of(edge.source_doc_id)
                      for edge in self.incoming(doc_number, RelationType.SUA_DOI)]
        guided_by = [
            (self._number_of(edge.source_doc_id), edge.target_dieu)
            for edge in self.incoming(doc_number, RelationType.HUONG_DAN)
            if dieu is None or edge.target_dieu is None or edge.target_dieu == dieu
        ]

        if replaced_by and status is EffectStatus.KHONG_XAC_DINH:
            status = EffectStatus.HET_HIEU_LUC

        notes: list[str] = []
        if replaced_by:
            notes.append("bị thay thế bởi " + ", ".join(replaced_by))
        if amended_by:
            notes.append("bị sửa đổi bởi " + ", ".join(amended_by))
        if guided_by:
            notes.append("được hướng dẫn bởi " + ", ".join(
                f"{number} (Điều {target})" if target else number
                for number, target in guided_by))

        return GraphVerdict(doc_number=doc_number, status=status, replaced_by=replaced_by,
                            amended_by=amended_by, guided_by=guided_by,
                            note="; ".join(notes))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "documents": [doc.model_dump(mode="json") for doc in self._documents.values()],
            "relations": [
                edge.model_dump(mode="json")
                for edges in self._out_edges.values() for edge in edges
            ],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Đã lưu KG snapshot (%d docs) -> %s", len(self._documents), path)

    def load(self, path: Path) -> bool:
        if not path.exists():
            logger.warning("Không tìm thấy KG snapshot tại %s", path)
            return False
        payload = json.loads(path.read_text(encoding="utf-8"))
        for raw_document in payload.get("documents", []):
            self.upsert_document(LegalDocumentMeta.model_validate(raw_document))
        for raw_relation in payload.get("relations", []):
            self.upsert_relation(LegalRelation.model_validate(raw_relation))
        logger.info("Đã nạp KG (%d docs, %d edges) từ %s", len(self._documents),
                    sum(len(edges) for edges in self._out_edges.values()), path)
        return True

    def close(self) -> None: 
        return None
