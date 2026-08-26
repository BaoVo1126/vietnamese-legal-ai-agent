from __future__ import annotations
import re
from datetime import date
from ..domain.document import LegalDocumentMeta, make_doc_id
from ..domain.enums import DocumentType, EffectStatus
from ..logging_config import get_logger
from . import patterns as P

logger = get_logger(__name__)

_FRONT_MATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)

_FIELD_OF_LAW_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("doanh nghiệp", "Doanh nghiệp"),
    ("đầu tư", "Đầu tư"),
    ("lao động", "Lao động"),
    ("thuế", "Thuế"),
    ("đất đai", "Đất đai"),
    ("dân sự", "Dân sự"),
    ("hình sự", "Hình sự"),
    ("hành chính", "Hành chính"),
    ("sở hữu trí tuệ", "Sở hữu trí tuệ"),
)

_TITLE_LOWER_STARTS = {"về", "quy", "hướng", "sửa", "quy_định"}


def split_front_matter(text: str) -> tuple[dict[str, str], str]:
    match = _FRONT_MATTER_RE.match(text)
    if not match:
        return {}, text
    front_matter: dict[str, str] = {}
    for line in match.group(1).split("\n"):
        if ":" in line:
            key, _, value = line.partition(":")
            front_matter[key.strip().lower()] = value.strip()
    return front_matter, text[match.end():]


class MetadataExtractor:
    def __init__(self, today: date | None = None) -> None:
        self._today = today

    def extract(self, header_text: str, full_text: str, source_path: str = "",
                front_matter: dict[str, str] | None = None) -> LegalDocumentMeta:
        front_matter = front_matter or {}
        search_space = header_text or full_text[:2000]

        doc_number = front_matter.get("doc_number") or self._extract_doc_number(
            search_space, full_text
        )
        title = sanitize_title(front_matter.get("title") or
                               self._extract_title(search_space))
        doc_type = P.infer_document_type(doc_number, title)
        issued_date = P.parse_vn_date(front_matter.get("issued_date", "")) or \
            self._extract_issued_date(search_space)
        effective_date = P.parse_vn_date(front_matter.get("effective_date", "")) or \
            self._extract_effective_date(full_text)
        expiry_date = P.parse_vn_date(front_matter.get("expiry_date", "")) or \
            self._extract_self_expiry(full_text)

        meta = LegalDocumentMeta(
            doc_id=make_doc_id(doc_number or title or source_path),
            doc_number=doc_number,
            doc_type=doc_type,
            title=title or doc_type.display_name,
            issuing_body=front_matter.get("issuing_body") or self._extract_issuing_body(
                search_space
            ),
            signer=self._extract_signer(full_text),
            issued_date=issued_date,
            effective_date=effective_date,
            expiry_date=expiry_date,
            field_of_law=front_matter.get("field_of_law") or self._infer_field_of_law(title),
            source_path=source_path,
        )
        meta.effect_status = self._resolve_effect_status(meta, front_matter, full_text)
        return meta

    @staticmethod
    def _extract_doc_number(header: str, full_text: str) -> str:
        for text in (header, full_text[:4000]):
            labelled = P.DOC_NUMBER_LABELLED_RE.search(text)
            if labelled:
                return labelled.group(1)
            plain = P.DOC_NUMBER_RE.search(text)
            if plain:
                return plain.group(1)
        return ""

    def _extract_title(self, header: str) -> str:
        lines = [line.strip() for line in header.split("\n") if line.strip()]
        for index, line in enumerate(lines):
            match = P.TITLE_LINE_RE.match(line)
            if not match or P.DOC_NUMBER_RE.search(line):
                continue
            parts = [match.group(1).strip(), match.group(2).strip()]
            for follower in lines[index + 1:]:
                if not self._is_title_continuation(follower):
                    break
                parts.append(follower)
            return self._prettify_title(" ".join(part for part in parts if part))
        return ""

    @staticmethod
    def _is_title_continuation(line: str) -> bool:
        if not line or len(line) > 160 or P.DOC_NUMBER_RE.search(line):
            return False
        if line.endswith((".", ";", ":")) or line.startswith("-"):
            return False
        letters = [ch for ch in line if ch.isalpha()]
        return bool(letters) and all(ch.isupper() for ch in letters)

    @staticmethod
    def _prettify_title(raw_title: str) -> str:
        cleaned = re.sub(r"\s+", " ", raw_title).strip(" -–:.")
        if not cleaned:
            return ""
        words = cleaned.split(" ")
        keyword_length = 2 if words[0].upper() in {"BỘ", "NGHỊ", "PHÁP", "THÔNG", "QUYẾT"} else 1
        keyword = " ".join(words[:keyword_length]).capitalize()
        remainder = words[keyword_length:]
        if not remainder:
            return keyword
        rest = " ".join(remainder).lower()
        first_word = rest.split(" ")[0]
        if first_word not in _TITLE_LOWER_STARTS:
            rest = rest[:1].upper() + rest[1:]
        return f"{keyword} {rest}".strip()

    @staticmethod
    def _extract_issued_date(header: str) -> date | None:
        return P.parse_vn_date(header)

    @staticmethod
    def _extract_effective_date(full_text: str) -> date | None:
        for sentence in _iter_sentences(full_text):
            if "hiệu lực" not in sentence.lower():
                continue
            if not _is_self_referential(sentence):
                continue
            match = P.EFFECTIVE_DATE_RE.search(sentence)
            if match:
                return P.parse_vn_date(match.group(1))
        return None

    @staticmethod
    def _extract_self_expiry(full_text: str) -> date | None:
        for sentence in _iter_sentences(full_text):
            lowered = sentence.lower()
            if "hết hiệu lực" not in lowered or not _is_self_referential(sentence):
                continue
            match = P.EXPIRY_DATE_RE.search(sentence)
            if match:
                return P.parse_vn_date(match.group(1))
        return None

    @staticmethod
    def _extract_issuing_body(header: str) -> str:
        upper_header = header.upper()
        for body in P.ISSUING_BODIES:
            if body in upper_header:
                return _ISSUING_BODY_DISPLAY.get(body, body.capitalize())
        return ""

    @staticmethod
    def _extract_signer(full_text: str) -> str:
        match = P.SIGNER_RE.search(full_text)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _infer_field_of_law(title: str) -> str:
        lowered = title.lower()
        for keyword, label in _FIELD_OF_LAW_KEYWORDS:
            if keyword in lowered:
                return label
        return ""

    def _resolve_effect_status(self, meta: LegalDocumentMeta,
                               front_matter: dict[str, str], full_text: str) -> EffectStatus:
        declared = front_matter.get("effect_status", "").strip().lower()
        if declared:
            try:
                return EffectStatus(declared)
            except ValueError:
                logger.warning("Front-matter effect_status không hợp lệ: %r", declared)
        head = full_text[:1200].lower()
        if "văn bản này đã hết hiệu lực" in head or "trạng thái: hết hiệu lực" in head:
            return EffectStatus.HET_HIEU_LUC
        if "hết hiệu lực một phần" in head:
            return EffectStatus.HET_HIEU_LUC_MOT_PHAN
        return meta.status_as_of(self._today)


_ISSUING_BODY_DISPLAY = {
    "QUỐC HỘI": "Quốc hội",
    "ỦY BAN THƯỜNG VỤ QUỐC HỘI": "Ủy ban Thường vụ Quốc hội",
    "CHÍNH PHỦ": "Chính phủ",
    "THỦ TƯỚNG CHÍNH PHỦ": "Thủ tướng Chính phủ",
    "BỘ TÀI CHÍNH": "Bộ Tài chính",
    "BỘ KẾ HOẠCH VÀ ĐẦU TƯ": "Bộ Kế hoạch và Đầu tư",
    "BỘ TƯ PHÁP": "Bộ Tư pháp",
    "BỘ CÔNG THƯƠNG": "Bộ Công Thương",
    "NGÂN HÀNG NHÀ NƯỚC VIỆT NAM": "Ngân hàng Nhà nước Việt Nam",
}

_SELF_REFERENCE_TERMS = (
    "luật này", "bộ luật này", "nghị định này", "thông tư này", "quyết định này",
    "pháp lệnh này", "nghị quyết này", "văn bản này",
)


def sanitize_title(title: str) -> str:
    if not title:
        return ""
    cleaned = re.sub(r"\s*(?:số\s*)?\d{1,4}/\d{4}/[A-ZĐ][A-ZĐ0-9\-/]*", " ", title)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;.-")
    return re.sub(r"\s+số$", "", cleaned).strip()


def _iter_sentences(text: str):
    for block in re.split(r"[\n]+", text):
        for sentence in re.split(r"(?<=[.;])\s+", block):
            sentence = sentence.strip()
            if sentence:
                yield sentence


def _is_self_referential(sentence: str) -> bool:
    lowered = sentence.lower()
    return any(term in lowered for term in _SELF_REFERENCE_TERMS)


__all__ = ["DocumentType", "MetadataExtractor", "split_front_matter"]
