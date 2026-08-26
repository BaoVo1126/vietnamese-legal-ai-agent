from __future__ import annotations
import re
import unicodedata
from functools import lru_cache
from ..logging_config import get_logger

logger = get_logger(__name__)

_PUNCTUATION_RE = re.compile(r"[^\w\s_]", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")

STOPWORDS = {
    "và", "của", "có", "được", "cho", "các", "những", "là", "trong", "khi", "này",
    "đó", "với", "theo", "tại", "về", "từ", "đến", "một", "hoặc", "thì", "mà", "như",
    "bị", "sẽ", "đã", "cũng", "nếu", "nên", "vào", "ra", "trên", "dưới", "sau", "trước",
}


@lru_cache(maxsize=1)
def _load_segmenter():
    try:
        from underthesea import word_tokenize  

        logger.info("Vietnamese segmenter: underthesea")
        return lambda text: word_tokenize(text, format="text").split()
    except Exception: 
        pass
    try:
        from pyvi import ViTokenizer  # type: ignore

        logger.info("Vietnamese segmenter: pyvi")
        return lambda text: ViTokenizer.tokenize(text).split()
    except Exception: 
        logger.warning("Không tìm thấy underthesea/pyvi - dùng tách từ theo khoảng trắng.")
        return lambda text: text.split()


@lru_cache(maxsize=65536)
def _segment_cached(normalised_text: str) -> tuple[str, ...]:
    """Bọc bộ tách từ bằng cache; trả tuple để giá trị cache là bất biến."""
    return tuple(_load_segmenter()(normalised_text))


class VietnameseTokenizer:
    def __init__(self, remove_stopwords: bool = True, keep_legal_numbers: bool = True) -> None:
        self.remove_stopwords = remove_stopwords
        self.keep_legal_numbers = keep_legal_numbers

    def tokenize(self, text: str) -> list[str]:
        normalised = self.normalise(text)
        if not normalised:
            return []
        raw_tokens = _segment_cached(normalised)
        tokens: list[str] = []
        for token in raw_tokens:
            token = re.sub(r"_+", "_", token).strip("_ ")
            if not token:
                continue
            if self.remove_stopwords and token.replace("_", " ") in STOPWORDS:
                continue
            tokens.append(token)
        return tokens

    def normalise(self, text: str) -> str:
        """Lower-case, strip punctuation, but preserve document numbers as terms."""
        text = unicodedata.normalize("NFC", text).lower()
        if self.keep_legal_numbers:
            text = re.sub(r"(\d{1,4})/(\d{4})/([a-zđ][a-zđ0-9\-/]*)",
                          lambda m: f"{m.group(1)}_{m.group(2)}_{m.group(3).replace('-', '_')}",
                          text)
        text = _PUNCTUATION_RE.sub(" ", text)
        return _WHITESPACE_RE.sub(" ", text).strip()
