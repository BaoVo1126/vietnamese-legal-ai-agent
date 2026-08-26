from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from ..config import Settings, get_settings
from ..indexing.bm25_index import BM25Index
from ..indexing.embedder import build_embedder
from ..indexing.qdrant_store import QdrantVectorStore
from ..kg.builder import build_graph_store
from ..llm.vllm_client import build_llm_client
from ..logging_config import get_logger
from ..monitoring.run_logger import RunRecorder, Stopwatch
from ..monitoring.tracing import tracing_status
from ..retrieval.hybrid import HybridRetriever
from ..retrieval.reranker import build_reranker
from .graph import build_agent_graph
from .nodes import AgentContext
from .state import initial_state

logger = get_logger(__name__)


@dataclass
class AgentAnswer:
    question: str
    answer: str
    status: str
    intent: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    graph_notes: list[str] = field(default_factory=list)
    excluded_chunks: list[dict[str, Any]] = field(default_factory=list)
    grounding_score: float = 0.0
    support_ratio: float = 0.0
    attempts: int = 0
    refusal_reason: str = ""
    latency_ms: float = 0.0
    trace: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_refusal(self) -> bool:
        return self.status == "refused"


class LegalAgentService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.embedder = build_embedder(self.settings)
        self.vector_store = QdrantVectorStore.from_settings(
            vector_size=self._vector_size(), settings=self.settings)
        self.bm25_index = BM25Index()
        self.graph_store = build_graph_store(self.settings)
        self.llm = build_llm_client(self.settings)
        self.retriever = HybridRetriever(
            vector_store=self.vector_store, bm25_index=self.bm25_index,
            embedder=self.embedder, reranker=build_reranker(self.settings),
            settings=self.settings,
        )
        self.context = AgentContext(llm=self.llm, retriever=self.retriever,
                                    graph_store=self.graph_store, settings=self.settings)
        self.graph = build_agent_graph(self.context)
        self.recorder = RunRecorder(self.settings.abs_run_log_path,
                                    enabled=self.settings.enable_run_log)
        self.tracing = tracing_status()
        self._ready = False


    def bootstrap(self, force_ingest: bool = False) -> dict:
        from ..ingestion.pipeline import IngestionPipeline

        needs_ingest = force_ingest or self.settings.qdrant_mode == "memory"
        if not needs_ingest:
            self.vector_store.ensure_collection(recreate=False)
            loaded = self.bm25_index.load(self.settings.abs_bm25_index_path)
            count = self.vector_store.count()
            needs_ingest = count == 0 or not loaded
            if not needs_ingest:
                self._ready = True
                logger.info("Đã nạp KB sẵn có: %d chunks, BM25 %d docs", count,
                            self.bm25_index.size)
                return {"mode": "loaded", "chunks": count, "bm25": self.bm25_index.size}

        pipeline = IngestionPipeline(settings=self.settings, embedder=self.embedder,
                                     vector_store=self.vector_store,
                                     graph_store=self.graph_store)
        result = pipeline.run()
        self.bm25_index.build(result.chunks)
        self._ready = True
        logger.info("Bootstrap bằng ingestion: %s", result.report)
        return {"mode": "ingested", **result.report}

    def ask(self, question: str, session_id: str = "", as_of: str | None = None) -> AgentAnswer:
        if not self._ready:
            self.bootstrap()
        with Stopwatch() as stopwatch:
            try:
                state = self.graph.invoke(
                    initial_state(question, session_id=session_id, as_of=as_of))
            except Exception as error:
                self.recorder.record(AgentAnswer(question=question, answer="",
                                                 status="error"),
                                     latency_ms=stopwatch_elapsed(stopwatch),
                                     session_id=session_id, error=str(error))
                raise
        answer = self._to_answer(question, state)
        answer.latency_ms = round(stopwatch.elapsed_ms, 1)
        self.recorder.record(answer, latency_ms=stopwatch.elapsed_ms, session_id=session_id)
        return answer

    @staticmethod
    def _to_answer(question: str, state: dict) -> AgentAnswer:
        evidence = [
            {
                "citation": item.chunk.citation.render(),
                "effect_status": item.chunk.effect_status.value,
                "score": round(item.final_score, 4),
                "source": item.source,
                "graph_note": item.graph_note,
                "text": item.chunk.text,
                "node_path": item.chunk.node_path,
            }
            for item in (state.get("retrieved") or [])
        ]
        return AgentAnswer(
            question=question,
            answer=state.get("answer", ""),
            status=state.get("status", "refused"),
            intent=state.get("intent", ""),
            citations=state.get("citations") or [],
            evidence=evidence,
            graph_notes=state.get("graph_notes") or [],
            excluded_chunks=state.get("excluded_chunks") or [],
            grounding_score=float(state.get("grounding_score", 0.0)),
            support_ratio=float(state.get("support_ratio", 0.0)),
            attempts=int(state.get("attempts", 0)),
            refusal_reason=state.get("refusal_reason", ""),
            trace=state.get("trace") or [],
        )

    def _vector_size(self) -> int:
        dim = getattr(self.embedder, "dim", 0)
        return dim or len(self.embedder.embed_query("khởi tạo"))

    def close(self) -> None:
        self.graph_store.close()

    def metrics(self, limit: int | None = None):
        from ..monitoring.metrics import summarise

        return summarise(self.recorder.read_all(limit=limit))


def stopwatch_elapsed(stopwatch: Stopwatch) -> float:
    import time

    return getattr(stopwatch, "elapsed_ms", 0.0) or (
        (time.perf_counter() - stopwatch._start) * 1000.0)
