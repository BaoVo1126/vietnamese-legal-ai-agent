from __future__ import annotations

import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from ..domain.chunk import LegalChunk
from ..logging_config import get_logger
from .tokenizer import VietnameseTokenizer

logger = get_logger(__name__)


class BM25Index:
    def __init__(self, tokenizer: VietnameseTokenizer | None = None) -> None:
        self.tokenizer = tokenizer or VietnameseTokenizer()
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []
        self._corpus_tokens: list[list[str]] = []

    def build(self, chunks: list[LegalChunk]) -> None:
        self._chunk_ids = [chunk.chunk_id for chunk in chunks]
        self._corpus_tokens = [self.tokenizer.tokenize(chunk.embed_text) for chunk in chunks]
        if not self._corpus_tokens:
            logger.warning("BM25: corpus rỗng.")
            self._bm25 = None
            return
        self._bm25 = BM25Okapi(self._corpus_tokens)
        logger.info("BM25 index: %d documents", len(self._chunk_ids))

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Return ``(chunk_id, score)`` pairs ordered by descending BM25 score."""
        if self._bm25 is None or not self._chunk_ids:
            return []
        tokens = self.tokenizer.tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(zip(self._chunk_ids, scores, strict=True),
                        key=lambda pair: pair[1], reverse=True)
        return [(chunk_id, float(score)) for chunk_id, score in ranked[:top_k] if score > 0.0]

    def save(self, path: Path, fingerprint: str = "") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump({"chunk_ids": self._chunk_ids, "corpus": self._corpus_tokens,
                         "fingerprint": fingerprint}, handle)
        logger.info("Đã lưu BM25 index -> %s", path)

    def load(self, path: Path, fingerprint: str = "") -> bool:
        if not path.exists():
            logger.warning("Không tìm thấy BM25 index tại %s", path)
            return False
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if fingerprint and payload.get("fingerprint") != fingerprint:
            logger.info("BM25 index cũ không khớp corpus hiện tại - dựng lại.")
            return False
        self._chunk_ids = payload["chunk_ids"]
        self._corpus_tokens = payload["corpus"]
        self._bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None
        logger.info("Đã nạp BM25 index (%d documents) từ %s", len(self._chunk_ids), path)
        return True

    @property
    def size(self) -> int:
        return len(self._chunk_ids)
