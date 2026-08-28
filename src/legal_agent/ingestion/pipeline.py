from __future__ import annotations
import hashlib
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from __future__ import annotations
import hashlib
import pickle
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Settings, get_settings
from ..domain.chunk import LegalChunk
from ..domain.document import LegalDocumentMeta
from ..indexing.bm25_index import BM25Index
from ..indexing.embedder import Embedder, build_embedder
from ..indexing.qdrant_store import QdrantVectorStore
from ..kg.base import LegalGraphStore
from ..kg.builder import KnowledgeGraphBuilder, build_graph_store
from ..logging_config import get_logger
from .chunker import LegalChunkBuilder
from .loaders import RawDocument, load_directory
from .parser import ParsedDocument, StructureAwareParser

logger = get_logger(__name__)


def _fingerprint(source_dir: Path) -> str:
    parts = []
    for path in sorted(source_dir.glob("*")):
        if path.is_file():
            stat = path.stat()
            parts.append(f"{path.name}:{stat.st_size}:{int(stat.st_mtime)}")
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
    return f"{len(parts)}-{digest[:16]}"


@dataclass
class IngestionResult:
    documents: list[LegalDocumentMeta] = field(default_factory=list)
    chunks: list[LegalChunk] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    vector_store: QdrantVectorStore | None = None
    bm25_index: BM25Index | None = None
    graph_store: LegalGraphStore | None = None
    embedder: Embedder | None = None

    @property
    def report(self) -> dict:
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "warnings": len(self.warnings),
            "doc_numbers": [document.doc_number for document in self.documents],
        }


class IngestionPipeline:
    def __init__(self, settings: Settings | None = None, embedder: Embedder | None = None,
                 vector_store: QdrantVectorStore | None = None,
                 graph_store: LegalGraphStore | None = None) -> None:
        self.settings = settings or get_settings()
        self.parser = StructureAwareParser(
            relation_extractor=self._build_relation_extractor())
        self.chunk_builder = LegalChunkBuilder(self.settings)
        self.embedder = embedder or build_embedder(self.settings)
        self.vector_store = vector_store or QdrantVectorStore.from_settings(
            vector_size=self._vector_size(), settings=self.settings)
        self.graph_store = graph_store or build_graph_store(self.settings)

    def _build_relation_extractor(self):
        """Bộ khai thác quan hệ, kèm fallback LLM nếu cấu hình bật."""
        from .relation_extractor import RelationExtractor

        if not self.settings.kg_llm_fallback:
            return RelationExtractor()
        from ..llm.vllm_client import build_llm_client

        logger.info("Bật fallback LLM cho khai thác quan hệ Knowledge Graph.")
        return RelationExtractor(llm=build_llm_client(self.settings))

    def run(self, source_dir: Path | None = None, recreate: bool = True,
            use_cache: bool = True) -> IngestionResult:
        source_dir = source_dir or self.settings.abs_raw_data_dir
        fingerprint = _fingerprint(source_dir)
        cached = self._load_cache(fingerprint) if use_cache else None
        if cached is not None:
            documents, chunks = cached
            logger.info("Dùng cache corpus: %d văn bản, %d chunks",
                        len(documents), len(chunks))
            return self._index(documents, chunks, recreate=recreate,
                               fingerprint=fingerprint)

        raw_documents = load_directory(source_dir)
        result = self.run_on(raw_documents, recreate=recreate, fingerprint=fingerprint)
        if use_cache:
            self._save_cache(fingerprint, result.documents, result.chunks)
        return result

    @property
    def _cache_path(self) -> Path:
        return self.settings.abs_processed_data_dir / "corpus_cache.pkl"

    def _load_cache(self, fingerprint: str):
        path = self._cache_path
        if not path.exists():
            return None
        try:
            with path.open("rb") as handle:
                payload = pickle.load(handle)
        except Exception as error:  
            logger.warning("Không đọc được cache corpus: %s", error)
            return None
        if payload.get("fingerprint") != fingerprint:
            logger.info("Vân tay thư mục nguồn đã đổi - dựng lại corpus.")
            return None
        return payload["documents"], payload["chunks"]

    def _save_cache(self, fingerprint: str, documents, chunks) -> None:
        path = self._cache_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump({"fingerprint": fingerprint, "documents": documents,
                         "chunks": chunks}, handle)
        logger.info("Đã lưu cache corpus -> %s", path)

    def run_on(self, raw_documents: list[RawDocument], recreate: bool = True,
               fingerprint: str = "") -> IngestionResult:
        result = IngestionResult(embedder=self.embedder)

        parsed_documents: list[ParsedDocument] = []
        for raw in raw_documents:
            parsed = self.parser.parse(raw.text, source_path=raw.source_path)
            parsed_documents.append(parsed)
            result.documents.append(parsed.meta)
            result.warnings.extend(f"{parsed.meta.doc_number}: {w}" for w in parsed.warnings)

        KnowledgeGraphBuilder(self.graph_store).build(result.documents, reset=True)

        for parsed in parsed_documents:
            result.chunks.extend(self.chunk_builder.build(parsed))

        self._index_dense(result.chunks, recreate=recreate)
        result.vector_store = self.vector_store
        result.bm25_index = self._index_sparse(result.chunks, fingerprint)
        result.graph_store = self.graph_store
        self._persist_graph()

        logger.info("Ingestion hoàn tất: %s", result.report)
        return result

    def _index(self, documents, chunks, recreate: bool, fingerprint: str) -> IngestionResult:
        result = IngestionResult(embedder=self.embedder, documents=list(documents),
                                 chunks=list(chunks))
        KnowledgeGraphBuilder(self.graph_store).build(result.documents, reset=True)
        self._index_dense(result.chunks, recreate=recreate)
        result.vector_store = self.vector_store
        result.bm25_index = self._index_sparse(result.chunks, fingerprint)
        result.graph_store = self.graph_store
        self._persist_graph()
        logger.info("Ingestion (cache) hoàn tất: %s", result.report)
        return result

    def _index_dense(self, chunks: list[LegalChunk], recreate: bool) -> None:
        self.vector_store.ensure_collection(recreate=recreate)
        if not chunks:
            return
        vectors = self.embedder.embed_documents([chunk.embed_text for chunk in chunks])
        self.vector_store.upsert(chunks, vectors)

    def _index_sparse(self, chunks: list[LegalChunk], fingerprint: str = "") -> BM25Index:
        index = BM25Index()
        path = self.settings.abs_bm25_index_path
        if fingerprint and index.load(path, fingerprint=fingerprint):
            return index
        index.build(chunks)
        index.save(path, fingerprint=fingerprint)
        return index

    def _persist_graph(self) -> None:
        save = getattr(self.graph_store, "save", None)
        if callable(save):
            save(self.settings.abs_graph_snapshot_path)

    def _vector_size(self) -> int:
        dim = getattr(self.embedder, "dim", 0)
        if dim:
            return dim
        return len(self.embedder.embed_query("khởi tạo"))

from ..config import Settings, get_settings
from ..domain.chunk import LegalChunk
from ..domain.document import LegalDocumentMeta
from ..indexing.bm25_index import BM25Index
from ..indexing.embedder import Embedder, build_embedder
from ..indexing.qdrant_store import QdrantVectorStore
from ..kg.base import LegalGraphStore
from ..kg.builder import KnowledgeGraphBuilder, build_graph_store
from ..logging_config import get_logger
from .chunker import LegalChunkBuilder
from .loaders import RawDocument, load_directory
from .parser import ParsedDocument, StructureAwareParser

logger = get_logger(__name__)


def _fingerprint(source_dir: Path) -> str:
    parts = []
    for path in sorted(source_dir.glob("*")):
        if path.is_file():
            stat = path.stat()
            parts.append(f"{path.name}:{stat.st_size}:{int(stat.st_mtime)}")
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
    return f"{len(parts)}-{digest[:16]}"


@dataclass
class IngestionResult:
    documents: list[LegalDocumentMeta] = field(default_factory=list)
    chunks: list[LegalChunk] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    vector_store: QdrantVectorStore | None = None
    bm25_index: BM25Index | None = None
    graph_store: LegalGraphStore | None = None
    embedder: Embedder | None = None

    @property
    def report(self) -> dict:
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "warnings": len(self.warnings),
            "doc_numbers": [document.doc_number for document in self.documents],
        }


class IngestionPipeline:
    def __init__(self, settings: Settings | None = None, embedder: Embedder | None = None,
                 vector_store: QdrantVectorStore | None = None,
                 graph_store: LegalGraphStore | None = None) -> None:
        self.settings = settings or get_settings()
        self.parser = StructureAwareParser(
            relation_extractor=self._build_relation_extractor())
        self.chunk_builder = LegalChunkBuilder(self.settings)
        self.embedder = embedder or build_embedder(self.settings)
        self.vector_store = vector_store or QdrantVectorStore.from_settings(
            vector_size=self._vector_size(), settings=self.settings)
        self.graph_store = graph_store or build_graph_store(self.settings)

    def _build_relation_extractor(self):
        """Bộ khai thác quan hệ, kèm fallback LLM nếu cấu hình bật."""
        from .relation_extractor import RelationExtractor

        if not self.settings.kg_llm_fallback:
            return RelationExtractor()
        from ..llm.vllm_client import build_llm_client

        logger.info("Bật fallback LLM cho khai thác quan hệ Knowledge Graph.")
        return RelationExtractor(llm=build_llm_client(self.settings))

    def run(self, source_dir: Path | None = None, recreate: bool = True,
            use_cache: bool = True) -> IngestionResult:
        source_dir = source_dir or self.settings.abs_raw_data_dir
        fingerprint = _fingerprint(source_dir)
        cached = self._load_cache(fingerprint) if use_cache else None
        if cached is not None:
            documents, chunks = cached
            logger.info("Dùng cache corpus: %d văn bản, %d chunks",
                        len(documents), len(chunks))
            return self._index(documents, chunks, recreate=recreate,
                               fingerprint=fingerprint)

        raw_documents = load_directory(source_dir)
        result = self.run_on(raw_documents, recreate=recreate, fingerprint=fingerprint)
        if use_cache:
            self._save_cache(fingerprint, result.documents, result.chunks)
        return result

    @property
    def _cache_path(self) -> Path:
        return self.settings.abs_processed_data_dir / "corpus_cache.pkl"

    def _load_cache(self, fingerprint: str):
        path = self._cache_path
        if not path.exists():
            return None
        try:
            with path.open("rb") as handle:
                payload = pickle.load(handle)
        except Exception as error:  
            logger.warning("Không đọc được cache corpus: %s", error)
            return None
        if payload.get("fingerprint") != fingerprint:
            logger.info("Vân tay thư mục nguồn đã đổi - dựng lại corpus.")
            return None
        return payload["documents"], payload["chunks"]

    def _save_cache(self, fingerprint: str, documents, chunks) -> None:
        path = self._cache_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump({"fingerprint": fingerprint, "documents": documents,
                         "chunks": chunks}, handle)
        logger.info("Đã lưu cache corpus -> %s", path)

    def run_on(self, raw_documents: list[RawDocument], recreate: bool = True,
               fingerprint: str = "") -> IngestionResult:
        result = IngestionResult(embedder=self.embedder)

        parsed_documents: list[ParsedDocument] = []
        for raw in raw_documents:
            parsed = self.parser.parse(raw.text, source_path=raw.source_path)
            parsed_documents.append(parsed)
            result.documents.append(parsed.meta)
            result.warnings.extend(f"{parsed.meta.doc_number}: {w}" for w in parsed.warnings)

        KnowledgeGraphBuilder(self.graph_store).build(result.documents, reset=True)

        for parsed in parsed_documents:
            result.chunks.extend(self.chunk_builder.build(parsed))

        self._index_dense(result.chunks, recreate=recreate)
        result.vector_store = self.vector_store
        result.bm25_index = self._index_sparse(result.chunks, fingerprint)
        result.graph_store = self.graph_store
        self._persist_graph()

        logger.info("Ingestion hoàn tất: %s", result.report)
        return result

    def _index(self, documents, chunks, recreate: bool, fingerprint: str) -> IngestionResult:
        result = IngestionResult(embedder=self.embedder, documents=list(documents),
                                 chunks=list(chunks))
        KnowledgeGraphBuilder(self.graph_store).build(result.documents, reset=True)
        self._index_dense(result.chunks, recreate=recreate)
        result.vector_store = self.vector_store
        result.bm25_index = self._index_sparse(result.chunks, fingerprint)
        result.graph_store = self.graph_store
        self._persist_graph()
        logger.info("Ingestion (cache) hoàn tất: %s", result.report)
        return result

    def _index_dense(self, chunks: list[LegalChunk], recreate: bool) -> None:
        self.vector_store.ensure_collection(recreate=recreate)
        if not chunks:
            return
        vectors = self.embedder.embed_documents([chunk.embed_text for chunk in chunks])
        self.vector_store.upsert(chunks, vectors)

    def _index_sparse(self, chunks: list[LegalChunk], fingerprint: str = "") -> BM25Index:
        index = BM25Index()
        path = self.settings.abs_bm25_index_path
        if fingerprint and index.load(path, fingerprint=fingerprint):
            return index
        index.build(chunks)
        index.save(path, fingerprint=fingerprint)
        return index

    def _persist_graph(self) -> None:
        save = getattr(self.graph_store, "save", None)
        if callable(save):
            save(self.settings.abs_graph_snapshot_path)

    def _vector_size(self) -> int:
        dim = getattr(self.embedder, "dim", 0)
        if dim:
            return dim
        return len(self.embedder.embed_query("khởi tạo"))
