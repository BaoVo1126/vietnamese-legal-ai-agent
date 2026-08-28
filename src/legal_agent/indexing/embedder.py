from __future__ import annotations

import hashlib
import math
from typing import Protocol, runtime_checkable

from ..config import Settings, get_settings
from ..logging_config import get_logger

logger = get_logger(__name__)


@runtime_checkable
class Embedder(Protocol):
    dim: int
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class HashingEmbedder:
    def __init__(self, dim: int = 512, tokenizer=None) -> None:
        from .tokenizer import VietnameseTokenizer

        self.dim = dim
        self.tokenizer = tokenizer or VietnameseTokenizer()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = self.tokenizer.tokenize(text)
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, fallback_model: str | None = None,
                 device: str = "cpu", batch_size: int = 16) -> None:
        self.model_name = model_name
        self.fallback_model = fallback_model
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self.dim = 0

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        from sentence_transformers import SentenceTransformer

        for candidate in (self.model_name, self.fallback_model):
            if not candidate:
                continue
            try:
                logger.info("Loading embedding model %s ...", candidate)
                self._model = SentenceTransformer(candidate, device=self.device)
                self.model_name = candidate
                self.dim = self._model.get_sentence_embedding_dimension()
                return self._model
            except Exception as error:  # pragma: no cover - depends on local cache
                logger.warning("Không tải được model %s: %s", candidate, error)
        raise RuntimeError("Không tải được bất kỳ embedding model nào đã cấu hình.")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure_model()
        vectors = model.encode(texts, batch_size=self.batch_size, convert_to_numpy=True,
                               normalize_embeddings=True, show_progress_bar=False)
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def build_embedder(settings: Settings | None = None) -> Embedder:
    """Factory driven by ``EMBEDDING_BACKEND``."""
    settings = settings or get_settings()
    if settings.embedding_backend == "sentence_transformers":
        return SentenceTransformerEmbedder(
            model_name=settings.embedding_model,
            fallback_model=settings.embedding_fallback_model,
            device=settings.embedding_device,
            batch_size=settings.embedding_batch_size,
        )
    logger.info("Dùng HashingEmbedder (profile MVP, không cần tải model).")
    return HashingEmbedder(dim=min(settings.embedding_dim, 512))
