from __future__ import annotations
from datetime import date
from ..domain.document import LegalDocumentMeta, LegalRelation
from ..domain.enums import DocumentType, EffectStatus, RelationType
from ..logging_config import get_logger
from .base import GraphVerdict

logger = get_logger(__name__)

_MERGE_DOCUMENT = """
MERGE (d:VanBan {doc_id: $doc_id})
SET d.doc_number = $doc_number, d.doc_key = $doc_key, d.title = $title,
    d.doc_type = $doc_type,
    d.effect_status = $effect_status, d.effective_date = $effective_date,
    d.expiry_date = $expiry_date, d.issuing_body = $issuing_body,
    d.field_of_law = $field_of_law
"""

_MERGE_RELATION = """
MERGE (source:VanBan {doc_id: $source_doc_id})
MERGE (target:VanBan {doc_id: $target_doc_id})
MERGE (source)-[edge:%s {target_dieu: $target_dieu}]->(target)
SET edge.evidence = $evidence, edge.confidence = $confidence
"""

_VALIDATE = """
OPTIONAL MATCH (d:VanBan {doc_key: $doc_key})
OPTIONAL MATCH (replacer:VanBan)-[:THAY_THE]->(d)
OPTIONAL MATCH (amender:VanBan)-[:SUA_DOI]->(d)
OPTIONAL MATCH (guide:VanBan)-[g:HUONG_DAN]->(d)
  WHERE $dieu IS NULL OR g.target_dieu IS NULL OR g.target_dieu = $dieu
RETURN d.effect_status AS status, d.effective_date AS effective_date,
       d.expiry_date AS expiry_date,
       collect(DISTINCT replacer.doc_key) AS replaced_by,
       collect(DISTINCT amender.doc_key) AS amended_by,
       collect(DISTINCT [guide.doc_key, g.target_dieu]) AS guided_by
"""


class Neo4jGraphStore:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j") -> None:
        from neo4j import GraphDatabase  
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database
        self.ensure_constraints()

    def ensure_constraints(self) -> None:
        with self._driver.session(database=self._database) as session:
            session.run(
                "CREATE CONSTRAINT van_ban_doc_id IF NOT EXISTS "
                "FOR (d:VanBan) REQUIRE d.doc_id IS UNIQUE"
            )
            session.run(
                "CREATE INDEX van_ban_doc_key IF NOT EXISTS "
                "FOR (d:VanBan) ON (d.doc_key)"
            )

    def clear(self) -> None:
        with self._driver.session(database=self._database) as session:
            session.run("MATCH (d:VanBan) DETACH DELETE d")

    def upsert_document(self, meta: LegalDocumentMeta) -> None:
        with self._driver.session(database=self._database) as session:
            session.run(_MERGE_DOCUMENT, {
                "doc_id": meta.doc_id,
                "doc_number": meta.doc_number,
                "doc_key": meta.doc_key,
                "title": meta.title,
                "doc_type": meta.doc_type.value,
                "effect_status": meta.effect_status.value,
                "effective_date": meta.effective_date.isoformat() if meta.effective_date else None,
                "expiry_date": meta.expiry_date.isoformat() if meta.expiry_date else None,
                "issuing_body": meta.issuing_body,
                "field_of_law": meta.field_of_law,
            })

    def upsert_relation(self, relation: LegalRelation) -> None:
        query = _MERGE_RELATION % relation.relation.value
        with self._driver.session(database=self._database) as session:
            session.run(query, {
                "source_doc_id": relation.source_doc_id,
                "target_doc_id": relation.target_doc_id,
                "target_dieu": relation.target_dieu,
                "evidence": relation.evidence,
                "confidence": relation.confidence,
            })

    def get_document(self, doc_key: str) -> LegalDocumentMeta | None:
        query = "MATCH (d:VanBan {doc_key: $doc_key}) RETURN d LIMIT 1"
        with self._driver.session(database=self._database) as session:
            record = session.run(query, {"doc_key": doc_key}).single()
        return _record_to_meta(record["d"]) if record else None

    def all_documents(self) -> list[LegalDocumentMeta]:
        with self._driver.session(database=self._database) as session:
            records = session.run("MATCH (d:VanBan) RETURN d").data()
        return [_record_to_meta(record["d"]) for record in records]

    def neighbours(self, doc_key: str, relation: RelationType) -> list[LegalRelation]:
        query = (
            f"MATCH (s:VanBan {{doc_key: $doc_key}})-[e:{relation.value}]->(t:VanBan) "
            "RETURN s.doc_id AS source, t.doc_id AS target, e.target_dieu AS target_dieu, "
            "e.evidence AS evidence, e.confidence AS confidence"
        )
        return self._edges(query, doc_key, relation)

    def incoming(self, doc_key: str, relation: RelationType) -> list[LegalRelation]:
        query = (
            f"MATCH (s:VanBan)-[e:{relation.value}]->(t:VanBan {{doc_key: $doc_key}}) "
            "RETURN s.doc_id AS source, t.doc_id AS target, e.target_dieu AS target_dieu, "
            "e.evidence AS evidence, e.confidence AS confidence"
        )
        return self._edges(query, doc_key, relation)

    def _edges(self, query: str, doc_key: str,
               relation: RelationType) -> list[LegalRelation]:
        with self._driver.session(database=self._database) as session:
            rows = session.run(query, {"doc_key": doc_key}).data()
        return [
            LegalRelation(
                source_doc_id=row["source"], target_doc_id=row["target"], relation=relation,
                target_dieu=row.get("target_dieu"), evidence=row.get("evidence") or "",
                confidence=row.get("confidence") if row.get("confidence") is not None else 1.0,
            )
            for row in rows
        ]

    def validate(self, doc_key: str, dieu: str | None = None,
                 as_of: date | None = None) -> GraphVerdict:
        with self._driver.session(database=self._database) as session:
            record = session.run(_VALIDATE, {"doc_key": doc_key, "dieu": dieu}).single()
        if record is None or record["status"] is None:
            return GraphVerdict(doc_number=doc_key, status=EffectStatus.KHONG_XAC_DINH,
                                note="văn bản chưa có trong Knowledge Graph")

        meta = LegalDocumentMeta(
            doc_id=doc_key,
            doc_number=doc_key,
            effect_status=_to_status(record["status"]),
            effective_date=_to_date(record["effective_date"]),
            expiry_date=_to_date(record["expiry_date"]),
        )
        replaced_by = [value for value in record["replaced_by"] if value]
        amended_by = [value for value in record["amended_by"] if value]
        guided_by = [(pair[0], pair[1]) for pair in record["guided_by"] if pair and pair[0]]

        status = meta.status_as_of(as_of)
        if replaced_by and status is EffectStatus.KHONG_XAC_DINH:
            status = EffectStatus.HET_HIEU_LUC

        notes = []
        if replaced_by:
            notes.append("bị thay thế bởi " + ", ".join(replaced_by))
        if amended_by:
            notes.append("bị sửa đổi bởi " + ", ".join(amended_by))
        if guided_by:
            notes.append("được hướng dẫn bởi " + ", ".join(
                f"{number} (Điều {target})" if target else number
                for number, target in guided_by))

        return GraphVerdict(doc_number=doc_key, status=status, replaced_by=replaced_by,
                            amended_by=amended_by, guided_by=guided_by, note="; ".join(notes))

    def close(self) -> None:
        self._driver.close()


def _to_status(value: str | None) -> EffectStatus:
    try:
        return EffectStatus(value) if value else EffectStatus.KHONG_XAC_DINH
    except ValueError:
        return EffectStatus.KHONG_XAC_DINH


def _to_date(value) -> date | None:
    if not value:
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def _record_to_meta(node) -> LegalDocumentMeta:
    try:
        doc_type = DocumentType(node.get("doc_type"))
    except ValueError:
        doc_type = DocumentType.KHAC
    return LegalDocumentMeta(
        doc_id=node.get("doc_id"),
        doc_number=node.get("doc_number") or "",
        title=node.get("title") or "",
        doc_type=doc_type,
        effect_status=_to_status(node.get("effect_status")),
        effective_date=_to_date(node.get("effective_date")),
        expiry_date=_to_date(node.get("expiry_date")),
        issuing_body=node.get("issuing_body") or "",
        field_of_law=node.get("field_of_law") or "",
    )
