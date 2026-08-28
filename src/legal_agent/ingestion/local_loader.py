from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

from ..domain.enums import EffectStatus
from ..logging_config import get_logger
from .hf_corpus import EFFECT_STATUS_MAP, front_matter, html_to_text

logger = get_logger(__name__)

MIN_DOCUMENT_CHARS = 500


@dataclass
class LocalDocumentSpec:
    label: str
    title: str
    doc_number: str = ""
    effect_status: str = "Còn hiệu lực"
    issued_date: str = ""
    effective_date: str = ""
    expiry_date: str = ""
    issuing_body: str = ""
    field_of_law: str = ""

    def as_record(self, source: str) -> dict:
        return {
            "label": self.label,
            "title": self.title,
            "so_ky_hieu": self.doc_number,
            "tinh_trang_hieu_luc": self.effect_status,
            "ngay_ban_hanh": self.issued_date,
            "ngay_co_hieu_luc": self.effective_date,
            "ngay_het_hieu_luc": self.expiry_date,
            "co_quan_ban_hanh": self.issuing_body,
            "field_of_law": self.field_of_law,
            "id": f"local:{source}",
        }

    def validate(self) -> None:
        if not self.title.strip():
            raise ValueError("Phải có tiêu đề văn bản để trích dẫn được.")
        if self.effect_status not in EFFECT_STATUS_MAP:
            raise ValueError(
                f"Trạng thái hiệu lực không hợp lệ: {self.effect_status!r}. "
                f"Chọn một trong: {sorted(EFFECT_STATUS_MAP)}"
            )


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".html", ".htm"}:
        return html_to_text(path.read_text(encoding="utf-8", errors="replace"))
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        return _pdf_to_text(path)
    raise ValueError(f"Chưa hỗ trợ định dạng {suffix}. Dùng .html, .txt hoặc .pdf.")


def _pdf_to_text(path: Path) -> str:
    from docling.document_converter import DocumentConverter

    logger.info("Chuyển PDF bằng Docling: %s", path.name)
    result = DocumentConverter().convert(str(path))
    return result.document.export_to_markdown()


def ingest_local_document(path: Path, spec: LocalDocumentSpec, out_dir: Path) -> Path:
    spec.validate()
    text = extract_text(path)
    if len(text) < MIN_DOCUMENT_CHARS:
        raise ValueError(f"Nội dung quá ngắn ({len(text)} ký tự) - kiểm tra lại file nguồn.")

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{_slugify(spec.label)}.txt"
    target.write_text(front_matter(spec.as_record(path.name)) + text, encoding="utf-8")
    status = EFFECT_STATUS_MAP.get(spec.effect_status, EffectStatus.KHONG_XAC_DINH)
    logger.info("Đã ghi %s (%d ký tự, %s)", target.name, len(text), status.value)
    return target


def _slugify(label: str) -> str:
    import re
    import unicodedata

    normalised = unicodedata.normalize("NFD", label)
    ascii_label = "".join(ch for ch in normalised if unicodedata.category(ch) != "Mn")
    ascii_label = ascii_label.replace("Đ", "D").replace("đ", "d")
    return re.sub(r"[^a-zA-Z0-9]+", "_", ascii_label).strip("_").lower()
