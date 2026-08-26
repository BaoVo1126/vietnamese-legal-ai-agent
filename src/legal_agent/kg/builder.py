from __future__ import annotations
from ..config import Settings, get_settings
from ..domain.document import LegalDocumentMeta
from ..domain.enums import EffectStatus, RelationType
from ..logging_config import get_logger
from .base import LegalGraphStore
from .memory_store import MemoryGraphStore

logger = get_logger(__name__)

_REPEALING_RELATIONS = {RelationType.THAY_THE, RelationType.BAI_BO}
_DECLARED_REPEAL_STATUSES = {EffectStatus.HET_HIEU_LUC,
                             EffectStatus.HET_HIEU_LUC_MOT_PHAN}


class KnowledgeGraphBuilder:
    def __init__(self, store: LegalGraphStore) -> None:
        self.store = store

    def build(self, documents: list[LegalDocumentMeta],
              reset: bool = False) -> LegalGraphStore:
        if reset:
            clear = getattr(self.store, "clear", None)
            if callable(clear):
                clear()
                logger.info("Đã xoá graph cũ trước khi dựng lại.")
        by_number = {doc.doc_number: doc for doc in documents if doc.doc_number}
        by_id = {doc.doc_id: doc for doc in documents}

        for document in documents:
            self.store.upsert_document(document)

        edge_count = 0
        for document in documents:
            for relation in document.relations:
                self.store.upsert_relation(relation)
                self.store.upsert_relation(
                    relation.model_copy(update={
                        "source_doc_id": relation.target_doc_id,
                        "target_doc_id": relation.source_doc_id,
                        "relation": relation.relation.inverse,
                        "source_dieu": relation.target_dieu,
                        "target_dieu": relation.source_dieu,
                    })
                )
                edge_count += 2

        repealed = self._propagate_repeals(documents, by_id)
        logger.info("KG: %d documents, %d edges, %d văn bản bị suy ra hết hiệu lực",
                    len(documents), edge_count, repealed)
        _ = by_number 
        return self.store

    def _propagate_repeals(self, documents: list[LegalDocumentMeta],
                           by_id: dict[str, LegalDocumentMeta]) -> int:
        """Mark a target as repealed when a document in force replaces it."""
        repealed = 0
        for document in documents:
            if document.effect_status is EffectStatus.HET_HIEU_LUC:
                continue
            for relation in document.relations:
                if relation.relation not in _REPEALING_RELATIONS:
                    continue
                target = by_id.get(relation.target_doc_id)
                if target is None or target.effect_status in _DECLARED_REPEAL_STATUSES:
                    continue
                target.effect_status = EffectStatus.HET_HIEU_LUC
                target.expiry_date = target.expiry_date or document.effective_date
                self.store.upsert_document(target)
                repealed += 1
                logger.info("%s bị %s thay thế -> đánh dấu hết hiệu lực",
                            target.doc_number, document.doc_number)
        return repealed


def build_graph_store(settings: Settings | None = None) -> LegalGraphStore:
    settings = settings or get_settings()
    if settings.graph_backend == "neo4j":
        try:
            from .neo4j_store import Neo4jGraphStore

            return Neo4jGraphStore(uri=settings.neo4j_uri, user=settings.neo4j_user,
                                   password=settings.neo4j_password,
                                   database=settings.neo4j_database)
        except Exception as error:
            logger.warning("Không kết nối được Neo4j (%s) - dùng MemoryGraphStore.", error)
    store = MemoryGraphStore()
    snapshot = settings.abs_graph_snapshot_path
    if snapshot.exists():
        store.load(snapshot)
    return store
