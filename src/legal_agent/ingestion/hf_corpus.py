from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ..domain.enums import DocumentType, EffectStatus
from ..logging_config import get_logger

if TYPE_CHECKING:  
    import pandas as pd

logger = get_logger(__name__)

DATASET_REPO = "th1nhng0/vietnamese-legal-documents"
METADATA_FILE = "data/metadata.parquet"
RELATIONSHIPS_FILE = "data/relationships.parquet"
CONTENT_FILE = "data/content.parquet"

MIN_DOCUMENT_CHARS = 500

CORE_DOC_TYPES = ("Hiến pháp", "Bộ luật", "Luật")
EXCLUDED_DOC_TYPES = ("Bản dịch văn bản",)

EFFECT_STATUS_MAP: dict[str, EffectStatus] = {
    "Còn hiệu lực": EffectStatus.CON_HIEU_LUC,
    "Hết hiệu lực toàn bộ": EffectStatus.HET_HIEU_LUC,
    "Hết hiệu lực một phần": EffectStatus.HET_HIEU_LUC_MOT_PHAN,
    "Chưa xác định": EffectStatus.KHONG_XAC_DINH,
    "Ngưng hiệu lực": EffectStatus.KHONG_XAC_DINH,
    "Không còn phù hợp": EffectStatus.KHONG_XAC_DINH,
}

_STATUS_RANK = {
    "Còn hiệu lực": 0,
    "Hết hiệu lực một phần": 1,
    "Ngưng hiệu lực": 2,
    "Chưa xác định": 3,
    "Không còn phù hợp": 4,
    "Hết hiệu lực toàn bộ": 5,
}

DOC_TYPE_MAP: dict[str, DocumentType] = {
    "Hiến pháp": DocumentType.HIEN_PHAP,
    "Bộ luật": DocumentType.BO_LUAT,
    "Luật": DocumentType.LUAT,
    "Pháp lệnh": DocumentType.PHAP_LENH,
    "Nghị định": DocumentType.NGHI_DINH,
    "Nghị quyết": DocumentType.NGHI_QUYET,
    "Thông tư": DocumentType.THONG_TU,
    "Quyết định": DocumentType.QUYET_DINH,
}


@dataclass(frozen=True)
class PriorityRule:
    label: str
    field_of_law: str
    so_ky_hieu: str | None = None
    title_contains: str | None = None
    doc_types: tuple[str, ...] = CORE_DOC_TYPES

    def matches(self, row) -> bool:
        if str(row.loai_van_ban) not in self.doc_types:
            return False
        if self.so_ky_hieu and str(row.so_ky_hieu).strip() != self.so_ky_hieu:
            return False
        return not (self.title_contains
                    and self.title_contains.lower() not in str(row.title).lower())

PRIORITY_DOCUMENTS: tuple[PriorityRule, ...] = (
    PriorityRule("Hiến pháp 2013", "Hiến pháp", title_contains="2013",
                 doc_types=("Hiến pháp",)),
    PriorityRule("Bộ luật Hình sự 2015", "Hình sự", so_ky_hieu="100/2015/QH13"),
    PriorityRule("Luật sửa đổi BLHS 2017", "Hình sự", so_ky_hieu="12/2017/QH14"),
    PriorityRule("Bộ luật Dân sự 2015", "Dân sự", so_ky_hieu="91/2015/QH13"),
    PriorityRule("Bộ luật Lao động 2019", "Lao động", so_ky_hieu="45/2019/QH14"),
    PriorityRule("Luật Xử lý vi phạm hành chính 2012", "Hành chính",
                 so_ky_hieu="15/2012/QH13"),
    PriorityRule("Luật sửa đổi XLVPHC 2020", "Hành chính", so_ky_hieu="67/2020/QH14"),
    PriorityRule("Luật Doanh nghiệp 2020", "Doanh nghiệp", so_ky_hieu="59/2020/QH14"),
    PriorityRule("Luật Doanh nghiệp 2014", "Doanh nghiệp", so_ky_hieu="68/2014/QH13"),
    PriorityRule("Luật Hôn nhân và gia đình 2014", "Hôn nhân gia đình",
                 so_ky_hieu="52/2014/QH13"),
    PriorityRule("Luật Đất đai 2024", "Đất đai", so_ky_hieu="31/2024/QH15"),
    PriorityRule("Luật Bảo hiểm xã hội 2024", "Bảo hiểm xã hội", so_ky_hieu="41/2024/QH15"),
    PriorityRule("Nghị định 01/2021 đăng ký doanh nghiệp", "Doanh nghiệp",
                 so_ky_hieu="01/2021/NĐ-CP", doc_types=("Nghị định",)),
    PriorityRule("Nghị định 100/2019 xử phạt giao thông đường bộ", "Giao thông",
                 so_ky_hieu="100/2019/NĐ-CP", doc_types=("Nghị định",)),
)


@dataclass
class SelectionReport:
    selected: list[dict] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        total = len(self.selected) + len(self.missing)
        return round(len(self.selected) / total, 4) if total else 0.0


class HFLegalCorpusLoader:
    def __init__(self, repo_id: str = DATASET_REPO) -> None:
        self.repo_id = repo_id
        self._metadata = None
        self._relationships = None

    def metadata(self) -> pd.DataFrame:
        if self._metadata is None:
            self._metadata = self._read_parquet(METADATA_FILE)
            logger.info("Metadata: %d văn bản", len(self._metadata))
        return self._metadata

    def relationships(self) -> pd.DataFrame:
        if self._relationships is None:
            self._relationships = self._read_parquet(RELATIONSHIPS_FILE)
            logger.info("Relationships: %d cạnh", len(self._relationships))
        return self._relationships

    def _read_parquet(self, filename: str) -> pd.DataFrame:
        import pandas as pd
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(self.repo_id, filename, repo_type="dataset")
        return pd.read_parquet(path)

    def select_priority(self, rules: tuple[PriorityRule, ...] = PRIORITY_DOCUMENTS
                        ) -> SelectionReport:
        frame = self.metadata()
        frame = frame[~frame.loai_van_ban.isin(EXCLUDED_DOC_TYPES)]
        report = SelectionReport()

        for rule in rules:
            matches = sorted((row for row in frame.itertuples() if rule.matches(row)),
                             key=self._preference_key)
            if not matches:
                report.missing.append(rule.label)
                logger.warning("Thiếu trong corpus: %s", rule.label)
                continue
            candidates = [self._to_record(rule, row) for row in matches]
            entry = dict(candidates[0])
            entry["candidates"] = candidates
            report.selected.append(entry)
        logger.info("Priority selection: %d/%d văn bản (thiếu: %s)",
                    len(report.selected), len(rules), report.missing or "không")
        return report

    @staticmethod
    def _to_record(rule: PriorityRule, row) -> dict:
        return {
            "label": rule.label,
            "field_of_law": rule.field_of_law,
            "id": str(row.id),
            "title": str(row.title),
            "so_ky_hieu": _clean_doc_number(row.so_ky_hieu),
            "loai_van_ban": str(row.loai_van_ban),
            "ngay_ban_hanh": _clean(row.ngay_ban_hanh),
            "ngay_co_hieu_luc": _clean(row.ngay_co_hieu_luc),
            "ngay_het_hieu_luc": _clean(row.ngay_het_hieu_luc),
            "co_quan_ban_hanh": _clean(row.co_quan_ban_hanh),
            "nguoi_ky": _clean(row.nguoi_ky),
            "tinh_trang_hieu_luc": _clean(row.tinh_trang_hieu_luc),
        }

    @staticmethod
    def _preference_key(row) -> tuple[int, int]:
        status_rank = _STATUS_RANK.get(str(row.tinh_trang_hieu_luc), 9)
        return (status_rank, -len(str(row.title)))

    def fetch_html(self, doc_ids: list[str]) -> dict[str, str]:
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem

        targets = {str(doc_id) for doc_id in doc_ids}
        found: dict[str, str] = {}
        filesystem = HfFileSystem()
        remote_path = f"datasets/{self.repo_id}/{CONTENT_FILE}"

        with filesystem.open(remote_path, "rb") as handle:
            parquet = pq.ParquetFile(handle)
            for index in range(parquet.num_row_groups):
                ids = [str(value) for value in
                       parquet.read_row_group(index, columns=["id"]).column("id").to_pylist()]
                if not targets & set(ids):
                    continue
                table = parquet.read_row_group(index)
                for doc_id, html in zip(table.column("id").to_pylist(),
                                        table.column("content_html").to_pylist(),
                                        strict=True):
                    if str(doc_id) in targets:
                        found[str(doc_id)] = html or ""
                logger.info("row-group %d/%d -> đã lấy %d/%d văn bản",
                            index + 1, parquet.num_row_groups, len(found), len(targets))
                if len(found) == len(targets):
                    break
        missing = targets - set(found)
        if missing:
            logger.warning("Không tìm thấy nội dung cho id: %s", sorted(missing))
        return found

    def write_raw(self, records: list[dict], html_by_id: dict[str, str],
                  out_dir: Path) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for record in records:
            resolved = self._resolve_content(record, html_by_id)
            if resolved is None:
                logger.error("Bỏ qua %s: không ứng viên nào có nội dung dùng được.",
                             record["label"])
                continue
            candidate, text = resolved
            merged = record
            if candidate["id"] != record["id"]:
                logger.warning("%s: bản ghi ưu tiên (id=%s) rỗng nội dung, lấy nội dung "
                               "từ bản dự phòng id=%s.", record["label"], record["id"],
                               candidate["id"])
                merged = merge_metadata(record, candidate)
            path = out_dir / f"{_slug(record['label'])}.txt"
            path.write_text(front_matter(merged) + text, encoding="utf-8")
            written.append(path)
            logger.info("Đã ghi %s (%d ký tự)", path.name, len(text))
        return written

    @staticmethod
    def _resolve_content(record: dict, html_by_id: dict[str, str]
                         ) -> tuple[dict, str] | None:
        for candidate in record.get("candidates", [record]):
            html = html_by_id.get(candidate["id"])
            if not html:
                continue
            text = html_to_text(html)
            if len(text) >= MIN_DOCUMENT_CHARS:
                return candidate, text
            logger.debug("%s: ứng viên id=%s chỉ có %d ký tự.",
                         record["label"], candidate["id"], len(text))
        return None


def merge_metadata(preferred: dict, content_source: dict) -> dict:
    merged = dict(preferred)
    merged["source_id"] = content_source.get("id", merged.get("id"))
    statuses = [preferred.get("tinh_trang_hieu_luc", ""),
                content_source.get("tinh_trang_hieu_luc", "")]
    merged["tinh_trang_hieu_luc"] = max(
        (status for status in statuses if status),
        key=lambda status: _STATUS_RANK.get(status, 9), default="")
    return merged


def front_matter(record: dict) -> str:
    status = EFFECT_STATUS_MAP.get(record.get("tinh_trang_hieu_luc", ""),
                                   EffectStatus.KHONG_XAC_DINH)
    fields = {
        "doc_number": record.get("so_ky_hieu", ""),
        "title": record.get("title", ""),
        "effect_status": status.value,
        "issued_date": record.get("ngay_ban_hanh", ""),
        "effective_date": record.get("ngay_co_hieu_luc", ""),
        "expiry_date": record.get("ngay_het_hieu_luc", ""),
        "issuing_body": record.get("co_quan_ban_hanh", ""),
        "field_of_law": record.get("field_of_law", ""),
        "source_id": record.get("id", ""),
    }
    lines = [f"{key}: {value}" for key, value in fields.items() if value]
    return "---\n" + "\n".join(lines) + "\n---\n"


def html_to_text(html: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    for tag in soup.find_all(["br", "p", "div", "tr", "td", "li", "h1", "h2", "h3"]):
        tag.append("\n")
    text = soup.get_text()
    text = text.replace("\u00a0", " ").replace("\u00ad", "")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    cleaned = [line for line in lines if line]
    return "\n".join(cleaned)


def _clean(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"none", "nan", ""} else text


def _clean_doc_number(value) -> str:
    text = _clean(value)
    normalised = text.lower().replace(":", "").strip()
    if normalised in {"không số", "khong so", "không có số"}:
        return ""
    return text.removeprefix("Số:").strip()


def _slug(label: str) -> str:
    import unicodedata

    ascii_label = unicodedata.normalize("NFD", label)
    ascii_label = "".join(ch for ch in ascii_label if unicodedata.category(ch) != "Mn")
    ascii_label = ascii_label.replace("Đ", "D").replace("đ", "d")
    return re.sub(r"[^a-zA-Z0-9]+", "_", ascii_label).strip("_").lower()
