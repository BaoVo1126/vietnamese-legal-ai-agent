from __future__ import annotations
from ..config import Settings, get_settings
from ..domain.chunk import LegalChunk, RetrievedChunk
from ..indexing.bm25_index import BM25Index
from ..indexing.embedder import Embedder
from ..indexing.qdrant_store import QdrantVectorStore
from ..logging_config import get_logger

logger = get_logger(__name__)


class HybridRetriever:
    def __init__(self, vector_store: QdrantVectorStore, bm25_index: BM25Index,
                 embedder: Embedder, reranker=None, settings: Settings | None = None) -> None:
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.embedder = embedder
        self.reranker = reranker
        self.settings = settings or get_settings()

    def retrieve(self, query: str, top_n: int | None = None, only_in_force: bool = False,
                 doc_keys: list[str] | None = None) -> list[RetrievedChunk]:
        top_n = top_n or self.settings.rerank_top_n
        dense_hits = self._dense(query, only_in_force, doc_keys)
        sparse_hits = self._sparse(query, doc_keys)
        fused = self.fuse(dense_hits, sparse_hits, k=self.settings.rrf_k)
        if not fused:
            logger.info("Không có kết quả nào cho truy vấn: %r", query)
            return []
        if self.reranker is None:
            return fused[:top_n]
        return self.reranker.rerank(query, fused[: self.settings.retrieval_top_k_dense
                                                 + self.settings.retrieval_top_k_sparse], top_n)
    
    def _dense(self, query: str, only_in_force: bool,
               doc_keys: list[str] | None) -> list[tuple[LegalChunk, float]]:
        vector = self.embedder.embed_query(query)
        return self.vector_store.search(vector, top_k=self.settings.retrieval_top_k_dense,
                                        only_in_force=only_in_force, doc_keys=doc_keys)

    def _sparse(self, query: str,
                doc_keys: list[str] | None) -> list[tuple[LegalChunk, float]]:
        pairs = self.bm25_index.search(query, top_k=self.settings.retrieval_top_k_sparse)
        if not pairs:
            return []
        chunks = {chunk.chunk_id: chunk
                  for chunk in self.vector_store.fetch_by_chunk_ids([cid for cid, _ in pairs])}
        results: list[tuple[LegalChunk, float]] = []
        for chunk_id, score in pairs:
            chunk = chunks.get(chunk_id)
            if chunk is None:
                continue
            if doc_keys and chunk.doc_key not in doc_keys:
                continue
            results.append((chunk, score))
        return results

    @staticmethod
    def fuse(dense_hits: list[tuple[LegalChunk, float]],
             sparse_hits: list[tuple[LegalChunk, float]], k: int = 60) -> list[RetrievedChunk]:
        merged: dict[str, RetrievedChunk] = {}

        for rank, (chunk, score) in enumerate(dense_hits, start=1):
            merged[chunk.chunk_id] = RetrievedChunk(
                chunk=chunk, dense_score=score, dense_rank=rank,
                fusion_score=1.0 / (k + rank), source="dense",
            )

        for rank, (chunk, score) in enumerate(sparse_hits, start=1):
            existing = merged.get(chunk.chunk_id)
            if existing is None:
                merged[chunk.chunk_id] = RetrievedChunk(
                    chunk=chunk, sparse_score=score, sparse_rank=rank,
                    fusion_score=1.0 / (k + rank), source="sparse",
                )
            else:
                existing.sparse_score = score
                existing.sparse_rank = rank
                existing.fusion_score += 1.0 / (k + rank)
                existing.source = "hybrid"

        return sorted(merged.values(), key=lambda item: item.fusion_score, reverse=True)
