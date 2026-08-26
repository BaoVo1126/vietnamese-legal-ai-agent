from __future__ import annotations
import re
import unicodedata

from pydantic import BaseModel, Field, model_validator


class Citation(BaseModel):
    doc_number: str = Field("", description="Số hiệu văn bản, e.g. '59/2020/QH14'")
    doc_title: str = Field("", description="Tên văn bản, e.g. 'Luật Doanh nghiệp'")
    dieu: str | None = Field(None, description="Số Điều")
    khoan: str | None = Field(None, description="Số Khoản")
    diem: str | None = Field(None, description="Ký hiệu Điểm, e.g. 'a'")

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def _require_identity(self) -> Citation:
        if not self.doc_number.strip() and not self.doc_title.strip():
            raise ValueError("Citation phải có doc_number hoặc doc_title")
        return self

    @property
    def doc_key(self) -> str:
        return _normalise(self.doc_number or self.doc_title)

    def render(self, with_title: bool = True) -> str:
        parts: list[str] = []
        if self.dieu:
            parts.append(f"Điều {self.dieu}")
        if self.khoan:
            parts.append(f"Khoản {self.khoan}")
        if self.diem:
            parts.append(f"Điểm {self.diem}")
        if with_title:
            doc = f"{self.doc_title} {self.doc_number}".strip()
        else:
            doc = self.doc_number or self.doc_title
        parts.append(doc)
        return ", ".join(part for part in parts if part)

    def __str__(self) -> str:
        return self.render()

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (
            self.doc_key,
            _normalise(self.dieu or ""),
            _normalise(self.khoan or ""),
            _normalise(self.diem or ""),
        )

    def covers(self, other: Citation) -> bool:
        if self.doc_key != other.doc_key:
            return False
        pairs = ((self.dieu, other.dieu), (self.khoan, other.khoan),
                 (self.diem, other.diem))
        for mine, theirs in pairs:
            if mine is None:
                continue
            if theirs is None or _normalise(mine) != _normalise(theirs):
                return False
        return True

    @classmethod
    def parse_cited(cls, text: str) -> list[Citation]:
        citations: list[Citation] = []
        seen: set[tuple[str, str, str, str]] = set()
        for match in _PARENTHESES_RE.finditer(text):
            for citation in cls.parse_all(match.group(1)):
                if citation.key not in seen:
                    seen.add(citation.key)
                    citations.append(citation)
        return citations

    @classmethod
    def parse_all(cls, text: str) -> list[Citation]:
        citations: list[Citation] = []
        seen: set[tuple[str, str, str, str]] = set()
        cursor = 0
        for anchor in _iter_document_anchors(text):
            window = text[max(cursor, anchor.start - 200): anchor.start]
            cursor = anchor.end
            citation = cls(
                doc_number=anchor.doc_number,
                doc_title=anchor.doc_title or _extract_title(window, text, anchor.start),
                dieu=_last(_DIEU_RE.findall(window)),
                khoan=_last(_KHOAN_RE.findall(window)),
                diem=_last(_DIEM_RE.findall(window)),
            )
            if citation.key not in seen:
                seen.add(citation.key)
                citations.append(citation)
        return citations


class _Anchor:
    __slots__ = ("doc_number", "doc_title", "end", "start")

    def __init__(self, start: int, end: int, doc_number: str = "",
                 doc_title: str = "") -> None:
        self.start = start
        self.end = end
        self.doc_number = doc_number
        self.doc_title = doc_title


def _iter_document_anchors(text: str):
    anchors = [
        _Anchor(match.start(), match.end(), doc_number=match.group(0))
        for match in _DOC_NUMBER_RE.finditer(text)
    ]
    for match in _TITLED_DOC_RE.finditer(text):
        if any(0 <= anchor.start - match.end() <= 8 or
               anchor.start <= match.start() < anchor.end
               for anchor in anchors):
            continue
        anchors.append(_Anchor(match.start(), match.end(),
                               doc_title=re.sub(r"\s+", " ", match.group(0)).strip()))
    anchors.sort(key=lambda anchor: anchor.start)
    return anchors


def _normalise(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip().lower()


def _last(values: list[str]) -> str | None:
    return values[-1].strip() if values else None


_PARENTHESES_RE = re.compile(r"\(([^()]{0,400})\)")
_DOC_NUMBER_RE = re.compile(r"\b\d{1,4}/\d{4}/[A-ZĐ][A-ZĐ0-9\-/]*\b")
_TITLED_DOC_RE = re.compile(r"Hiến pháp(?:\s+năm)?\s+(?:19|20)\d{2}", re.UNICODE)
_DIEU_RE = re.compile(r"[Đđ]iều\s+(\d+[a-zđ]?)", re.UNICODE)
_KHOAN_RE = re.compile(r"[Kk]hoản\s+(\d+[a-zđ]?)", re.UNICODE)
_DIEM_RE = re.compile(r"[Đđ]iểm\s+([a-zđ]{1,2})\b", re.UNICODE)
_TITLE_STOPWORDS = (
    "số|đã|và|thì|là|của|này|đó|khi|do|được|có|hoặc|với|theo|tại|về|nêu|trên|nay|bị|hết"
)
_TITLE_WORD = rf"(?:\s+(?!(?:{_TITLE_STOPWORDS})\b)[A-ZĐÀ-Ỹa-zà-ỹ]+)"
_DOC_KEYWORD = (
    r"(?:Bộ luật|Luật|Pháp lệnh|Nghị định|Nghị quyết|Thông tư|Quyết định|Hiến pháp)"
)

_TITLE_BEFORE_RE = re.compile(rf"({_DOC_KEYWORD}{_TITLE_WORD}{{0,6}})\s*(?:số\s*)?$",
                              re.UNICODE)
_TITLE_AFTER_RE = re.compile(rf"^\s*({_DOC_KEYWORD}{_TITLE_WORD}{{0,6}})", re.UNICODE)


def _extract_title(window: str, text: str, doc_start: int) -> str:
    match = _TITLE_BEFORE_RE.search(window.rstrip())
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip()
    tail = text[doc_start:]
    number = _DOC_NUMBER_RE.match(tail)
    after = _TITLE_AFTER_RE.search(tail[len(number.group(0)):]) if number else None
    return re.sub(r"\s+", " ", after.group(1)).strip() if after else ""
