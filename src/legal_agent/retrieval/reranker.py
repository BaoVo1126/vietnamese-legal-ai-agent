from __future__ import annotations
from typing import Protocol, runtime_checkable
from ..config import Settings, get_settings
from ..domain.chunk import RetrievedChunk
from ..logging_config import get_logger

logger = get_logger(__name__)


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[RetrievedChunk],
               top_n: int) -> list[RetrievedChunk]: ...


class LexicalOverlapReranker:
    def __init__(self, tokenizer=None) -> None:
        from ..indexing.tokenizer import VietnameseTokenizer

        self.tokenizer = tokenizer or VietnameseTokenizer()

    def rerank(self, query: str, candidates: list[RetrievedChunk],
               top_n: int) -> list[RetrievedChunk]:
        query_terms = set(self.tokenizer.tokenize(query))
        for candidate in candidates:
            passage_terms = set(self.tokenizer.tokenize(candidate.chunk.embed_text))
            if query_terms and passage_terms:
                overlap = len(query_terms & passage_terms) / len(query_terms)
            else:
                overlap = 0.0
            candidate.rerank_score = round(overlap, 6) + candidate.fusion_score * 1e-3
        return sorted(candidates, key=lambda item: item.final_score, reverse=True)[:top_n]


class CrossEncoderReranker:
    def __init__(self, model_name: str, use_fp16: bool = False) -> None:
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from FlagEmbedding import FlagReranker

            logger.info("Loading reranker %s ...", self.model_name)
            self._model = FlagReranker(self.model_name, use_fp16=self.use_fp16)
        return self._model

    def rerank(self, query: str, candidates: list[RetrievedChunk],
               top_n: int) -> list[RetrievedChunk]:
        if not candidates:
            return []
        model = self._ensure_model()
        pairs = [[query, candidate.chunk.embed_text] for candidate in candidates]
        scores = model.compute_score(pairs, normalize=True)
        if not isinstance(scores, list):
            scores = [scores]
        for candidate, score in zip(candidates, scores, strict=False):
            candidate.rerank_score = float(score)
        return sorted(candidates, key=lambda item: item.final_score, reverse=True)[:top_n]


def build_reranker(settings: Settings | None = None) -> Reranker:
    settings = settings or get_settings()
    if settings.reranker_backend == "flag_embedding":
        return CrossEncoderReranker(settings.reranker_model)
    logger.info("Dùng LexicalOverlapReranker (profile MVP).")
    return LexicalOverlapReranker()
