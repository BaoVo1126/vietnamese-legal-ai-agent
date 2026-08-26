from __future__ import annotations

import re
import unicodedata
from datetime import date

from ..domain.enums import ISSUER_SUFFIX_TO_TYPE, DocumentType

PHAN_RE = re.compile(
    r"^\s*PHẦN\s+(?:THỨ\s+)?([IVXLCDM]+|\d+|[A-ZĐÀ-Ỹ]+)\s*[.:\-–]?\s*(.*)$",
    re.IGNORECASE | re.UNICODE,
)
CHUONG_RE = re.compile(
    r"^\s*CHƯƠNG\s+([IVXLCDM]+|\d+)\s*[.:\-–]?\s*(.*)$",
    re.IGNORECASE | re.UNICODE,
)
MUC_RE = re.compile(
    r"^\s*MỤC\s+([IVXLCDM]+|\d+)\s*[.:\-–]?\s*(.*)$",
    re.IGNORECASE | re.UNICODE,
)
TIEU_MUC_RE = re.compile(
    r"^\s*TIỂU\s+MỤC\s+([IVXLCDM]+|\d+)\s*[.:\-–]?\s*(.*)$",
    re.IGNORECASE | re.UNICODE,
)
DIEU_RE = re.compile(r"^\s*Điều\s+(\d+[a-zđ]?)\s*([.:\-–])?\s*(.*)$", re.UNICODE)
KHOAN_RE = re.compile(r"^\s*(\d+[a-zđ]?)\s*\.\s+(.*)$", re.UNICODE)
DIEM_RE = re.compile(r"^\s*([a-zđ]{1,2})\s*\)\s+(.*)$", re.UNICODE)

DIEU_FALSE_POSITIVE_PREFIXES = (
    "của", "này", "nêu", "trên", "được", "quy", "khoản", "và", "đến", "hoặc", "thì",
)

DOC_NUMBER_RE = re.compile(r"\b(\d{1,4}/\d{4}/[A-ZĐ][A-ZĐ0-9\-/]*)\b")
DOC_NUMBER_LABELLED_RE = re.compile(
    r"(?:Số|số)\s*[:.]?\s*(\d{1,4}/\d{4}/[A-ZĐ][A-ZĐ0-9\-/]*)", re.UNICODE
)
DATE_VN_RE = re.compile(r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})", re.IGNORECASE)
DATE_SLASH_RE = re.compile(r"ngày\s+(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})", re.IGNORECASE)
DATE_BARE_RE = re.compile(r"\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})\b")

_DATE_TAIL = r"(ngày\s+\d{1,2}(?:\s+tháng\s+\d{1,2}\s+năm\s+|[/\-]\d{1,2}[/\-])\d{4})"
EFFECTIVE_DATE_RE = re.compile(
    r"(?:có\s+hiệu\s+lực(?:\s+thi\s+hành)?|hiệu\s+lực\s+thi\s+hành)"
    r"[^.\n]{0,60}?" + _DATE_TAIL,
    re.IGNORECASE | re.UNICODE,
)
EXPIRY_DATE_RE = re.compile(
    r"hết\s+hiệu\s+lực[^.\n]{0,60}?" + _DATE_TAIL,
    re.IGNORECASE | re.UNICODE,
)

ISSUING_BODIES = (
    "ỦY BAN THƯỜNG VỤ QUỐC HỘI",
    "QUỐC HỘI",
    "THỦ TƯỚNG CHÍNH PHỦ",
    "CHÍNH PHỦ",
    "BỘ TÀI CHÍNH",
    "BỘ KẾ HOẠCH VÀ ĐẦU TƯ",
    "BỘ TƯ PHÁP",
    "BỘ CÔNG THƯƠNG",
    "NGÂN HÀNG NHÀ NƯỚC VIỆT NAM",
)

TITLE_LINE_RE = re.compile(
    r"^\s*(BỘ LUẬT|LUẬT|PHÁP LỆNH|NGHỊ ĐỊNH|NGHỊ QUYẾT|THÔNG TƯ|QUYẾT ĐỊNH)\b(.*)$",
    re.UNICODE,
)

SIGNER_RE = re.compile(
    r"(?:CHỦ TỊCH QUỐC HỘI|TM\. CHÍNH PHỦ|THỦ TƯỚNG CHÍNH PHỦ|THỦ TƯỚNG"
    r"|KT\. BỘ TRƯỞNG|BỘ TRƯỞNG)\s*\n+\s*([A-ZĐÀ-Ỹ][A-Za-zĐđà-ỹÀ-Ỹ ]{3,40})",
    re.UNICODE,
)

RELATION_CUES: dict[str, tuple[str, ...]] = {
    "THAY_THE": ("thay thế cho", "thay thế"),
    "SUA_DOI": ("sửa đổi, bổ sung một số điều của", "sửa đổi, bổ sung", "sửa đổi"),
    "HUONG_DAN": (
        "quy định chi tiết thi hành",
        "quy định chi tiết",
        "hướng dẫn thi hành",
        "hướng dẫn",
    ),
    "BAI_BO": ("bãi bỏ", "hủy bỏ"),
    "CAN_CU": ("căn cứ",),
}

DIEU_OF_DOC_RE = re.compile(
    r"Điều\s+(\d+[a-zđ]?)[^.\n]{0,40}?của\s+([^,;.\n]{0,60}?\d{1,4}/\d{4}/[A-ZĐ][A-ZĐ0-9\-/]*)",
    re.UNICODE,
)

ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def normalise_text(raw: str) -> str:
    text = unicodedata.normalize("NFC", raw)
    text = text.replace("\u00a0", " ").replace("\u00ad", "").replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def roman_to_int(value: str) -> int:
    value = value.strip().upper()
    if not value or any(ch not in ROMAN_VALUES for ch in value):
        return 0
    total, previous = 0, 0
    for char in reversed(value):
        current = ROMAN_VALUES[char]
        total = total - current if current < previous else total + current
        previous = max(previous, current)
    return total


def parse_vn_date(text: str) -> date | None:
    if not text:
        return None
    match = (DATE_VN_RE.search(text) or DATE_SLASH_RE.search(text)
             or DATE_BARE_RE.search(text))
    if not match:
        return None
    day, month, year = (int(group) for group in match.groups())
    try:
        return date(year, month, day)
    except ValueError:
        return None


def infer_document_type(doc_number: str, title: str = "") -> DocumentType:
    if doc_number:
        suffix = doc_number.rsplit("/", 1)[-1].upper()
        for key, doc_type in sorted(
            ISSUER_SUFFIX_TO_TYPE.items(), key=lambda kv: len(kv[0]), reverse=True
        ):
            if suffix.startswith(key.upper()):
                return doc_type
    upper_title = title.upper()
    for keyword, doc_type in (
        ("BỘ LUẬT", DocumentType.BO_LUAT),
        ("HIẾN PHÁP", DocumentType.HIEN_PHAP),
        ("PHÁP LỆNH", DocumentType.PHAP_LENH),
        ("NGHỊ ĐỊNH", DocumentType.NGHI_DINH),
        ("NGHỊ QUYẾT", DocumentType.NGHI_QUYET),
        ("THÔNG TƯ", DocumentType.THONG_TU),
        ("QUYẾT ĐỊNH", DocumentType.QUYET_DINH),
        ("LUẬT", DocumentType.LUAT),
    ):
        if keyword in upper_title:
            return doc_type
    return DocumentType.KHAC


def looks_like_dieu_heading(number: str, separator: str | None, remainder: str) -> bool:
    remainder = remainder.strip()
    if separator in {".", ":"} or not remainder:
        return True
    first_word = remainder.split()[0].lower().strip(",.;:")
    return first_word not in DIEU_FALSE_POSITIVE_PREFIXES
