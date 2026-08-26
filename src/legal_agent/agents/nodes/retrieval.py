from __future__ import annotations
from ...domain.chunk import RetrievedChunk
from ...logging_config import get_logger
from ..state import AgentState, trace_entry
from .base import AgentContext

logger = get_logger(__name__)

_EXACT_MATCH_BOOST = 0.5


class HybridRetrievalNode:
    name = "retrieve"

    def __init__(self, context: AgentContext) -> None:
        self.context = context

    def __call__(self, state: AgentState) -> dict:
        settings = self.context.settings
        queries = self._queries(state)
        doc_hints = state.get("doc_hints") or []

        merged: dict[str, RetrievedChunk] = {}
        for query in queries:
            for item in self._retrieve_one(query, doc_hints):
                self._merge(merged, item)

        for item in self._exact_citation_hits(state):
            self._merge(merged, item)

        results = sorted(merged.values(), key=lambda item: item.final_score, reverse=True)
        results = results[: settings.rerank_top_n]
        attempts = state.get("attempts", 0) + 1

        logger.info("Retrieval lần %d: %d queries -> %d chunks", attempts, len(queries),
                    len(results))
        return {
            "retrieved": results,
            "attempts": attempts,
            "trace": [trace_entry(self.name, attempt=attempts, queries=queries,
                                  hits=[item.chunk.citation.render() for item in results],
                                  scores=[round(item.final_score, 4) for item in results])],
        }

    @staticmethod
    def _queries(state: AgentState) -> list[str]:
        candidates = [state.get("search_query") or state["question"]]
        candidates.extend(state.get("sub_queries") or [])
        seen: set[str] = set()
        queries: list[str] = []
        for candidate in candidates:
            key = candidate.strip().lower()
            if key and key not in seen:
                seen.add(key)
                queries.append(candidate.strip())
        return queries

    def _retrieve_one(self, query: str, doc_hints: list[str]) -> list[RetrievedChunk]:
        if doc_hints:
            scoped = self.context.retriever.retrieve(query, doc_keys=doc_hints)
            if scoped:
                return scoped
            logger.info("Không có kết quả trong phạm vi %s - mở rộng toàn corpus.", doc_hints)
        return self.context.retriever.retrieve(query)

    def _exact_citation_hits(self, state: AgentState) -> list[RetrievedChunk]:
        doc_hints = state.get("doc_hints") or []
        dieu_hints = state.get("dieu_hints") or []
        if not doc_hints or not dieu_hints:
            return []
        store = self.context.retriever.vector_store
        hits: list[RetrievedChunk] = []
        for doc_number in doc_hints:
            for dieu in dieu_hints:
                for chunk in store.fetch_by_citation(doc_number, dieu):
                    hits.append(RetrievedChunk(chunk=chunk, source="exact_citation",
                                               fusion_score=_EXACT_MATCH_BOOST,
                                               rerank_score=_EXACT_MATCH_BOOST))
        if hits:
            logger.info("Nạp trực tiếp %d chunk theo trích dẫn tường minh.", len(hits))
        return hits

    @staticmethod
    def _merge(merged: dict[str, RetrievedChunk], item: RetrievedChunk) -> None:
        existing = merged.get(item.chunk_id)
        if existing is None or item.final_score > existing.final_score:
            merged[item.chunk_id] = item
