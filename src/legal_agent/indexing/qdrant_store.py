from __future__ import annotations

import uuid

from qdrant_client import QdrantClient, models

from ..config import Settings, get_settings
from ..domain.chunk import LegalChunk
from ..domain.enums import EffectStatus
from ..logging_config import get_logger

logger = get_logger(__name__)

_NAMESPACE = uuid.UUID("6f1d7e2e-8a7b-5c31-9f0a-2f3a4b5c6d7e")


def point_id_for(chunk_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


class QdrantVectorStore:
    def __init__(self, client: QdrantClient, collection: str, vector_size: int,
                 supports_payload_index: bool = False) -> None:
        self.client = client
        self.collection = collection
        self.vector_size = vector_size
        self.supports_payload_index = supports_payload_index

    @classmethod
    def from_settings(cls, vector_size: int, settings: Settings | None = None
                      ) -> QdrantVectorStore:
        settings = settings or get_settings()
        if settings.qdrant_mode == "server":
            client = QdrantClient(url=settings.qdrant_url,
                                  api_key=settings.qdrant_api_key or None)
            logger.info("Qdrant server: %s", settings.qdrant_url)
            return cls(client, settings.qdrant_collection, vector_size,
                       supports_payload_index=True)
        client = QdrantClient(":memory:")
        logger.info("Qdrant in-memory (profile MVP).")
        return cls(client, settings.qdrant_collection, vector_size)

    def ensure_collection(self, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False
        if not exists:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=self.vector_size,
                                                   distance=models.Distance.COSINE),
            )
            for field, schema in () if not self.supports_payload_index else (
                ("doc_number", models.PayloadSchemaType.KEYWORD),
                ("doc_key", models.PayloadSchemaType.KEYWORD),
                ("doc_id", models.PayloadSchemaType.KEYWORD),
                ("effect_status", models.PayloadSchemaType.KEYWORD),
                ("dieu", models.PayloadSchemaType.KEYWORD),
                ("field_of_law", models.PayloadSchemaType.KEYWORD),
            ):
                try:
                    self.client.create_payload_index(self.collection, field, schema)
                except Exception as error: 
                    logger.debug("Bỏ qua payload index %s: %s", field, error)
            logger.info("Đã tạo collection %s (dim=%d)", self.collection, self.vector_size)

    def upsert(self, chunks: list[LegalChunk], vectors: list[list[float]],
               batch_size: int = 128) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("Số lượng chunk và vector không khớp.")
        points = [
            models.PointStruct(id=point_id_for(chunk.chunk_id), vector=vector,
                               payload=chunk.to_payload())
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        for start in range(0, len(points), batch_size):
            self.client.upsert(self.collection, points=points[start:start + batch_size])
        logger.info("Đã upsert %d chunks vào %s", len(points), self.collection)
        return len(points)

    def search(self, vector: list[float], top_k: int = 20,
               only_in_force: bool = False,
               doc_keys: list[str] | None = None) -> list[tuple[LegalChunk, float]]:
        conditions: list[models.Condition] = []
        if only_in_force:
            citable = [status.value for status in EffectStatus if status.is_citable]
            conditions.append(models.FieldCondition(
                key="effect_status", match=models.MatchAny(any=citable)))
        if doc_keys:
            conditions.append(models.FieldCondition(
                key="doc_key", match=models.MatchAny(any=list(doc_keys))))
        query_filter = models.Filter(must=conditions) if conditions else None

        response = self.client.query_points(
            collection_name=self.collection, query=vector, limit=top_k,
            query_filter=query_filter, with_payload=True,
        )
        return [(LegalChunk.from_payload(point.payload), float(point.score))
                for point in response.points]

    def fetch_by_chunk_ids(self, chunk_ids: list[str]) -> list[LegalChunk]:
        if not chunk_ids:
            return []
        records = self.client.retrieve(self.collection,
                                       ids=[point_id_for(cid) for cid in chunk_ids],
                                       with_payload=True)
        return [LegalChunk.from_payload(record.payload) for record in records]

    def fetch_by_citation(self, doc_key: str, dieu: str | None = None,
                          limit: int = 20) -> list[LegalChunk]:
        conditions: list[models.Condition] = [
            models.FieldCondition(key="doc_key", match=models.MatchValue(value=doc_key))
        ]
        if dieu:
            conditions.append(
                models.FieldCondition(key="dieu", match=models.MatchValue(value=dieu)))
        records, _ = self.client.scroll(self.collection,
                                        scroll_filter=models.Filter(must=conditions),
                                        limit=limit, with_payload=True)
        return [LegalChunk.from_payload(record.payload) for record in records]

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count
