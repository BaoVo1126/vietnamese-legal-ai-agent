from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legal_agent.config import get_settings
from legal_agent.domain.enums import NodeLevel
from legal_agent.ingestion.hf_corpus import EFFECT_STATUS_MAP
from legal_agent.ingestion.local_loader import (
    LocalDocumentSpec,
    ingest_local_document,
)
from legal_agent.ingestion.parser import StructureAwareParser
from legal_agent.logging_config import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="Nạp văn bản tải sẵn vào kho")
    parser.add_argument("path", help="File .html, .txt hoặc .pdf đã tải về")
    parser.add_argument("--label", required=True, help="Nhãn ngắn, dùng đặt tên file")
    parser.add_argument("--title", required=True, help="Tên văn bản dùng để trích dẫn")
    parser.add_argument("--doc-number", default="", help="Số hiệu; để trống nếu không có")
    parser.add_argument("--status", default="Còn hiệu lực",
                        choices=sorted(EFFECT_STATUS_MAP))
    parser.add_argument("--issued", default="", help="Ngày ban hành dd/mm/yyyy")
    parser.add_argument("--effective", default="", help="Ngày có hiệu lực dd/mm/yyyy")
    parser.add_argument("--expiry", default="", help="Ngày hết hiệu lực dd/mm/yyyy")
    parser.add_argument("--issuing-body", default="", help="Cơ quan ban hành")
    parser.add_argument("--field", default="", help="Lĩnh vực")
    parser.add_argument("--out", default=None, help="Thư mục kho (mặc định data/raw)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    source = Path(args.path)
    if not source.exists():
        print(f"Không tìm thấy file: {source}")
        return 1

    spec = LocalDocumentSpec(
        label=args.label, title=args.title, doc_number=args.doc_number,
        effect_status=args.status, issued_date=args.issued,
        effective_date=args.effective, expiry_date=args.expiry,
        issuing_body=args.issuing_body, field_of_law=args.field,
    )
    try:
        target = ingest_local_document(source, spec, Path(args.out) if args.out
                                       else get_settings().abs_raw_data_dir)
    except ValueError as error:
        print(f"Lỗi: {error}")
        return 1

    parsed = StructureAwareParser().parse(target.read_text(encoding="utf-8"), str(target))
    print(f"\nĐã ghi: {target}")
    print(f"  số hiệu   : {parsed.meta.doc_number or '(không có)'}")
    print(f"  tiêu đề   : {parsed.meta.title}")
    print(f"  hiệu lực  : {parsed.meta.effect_status.value}")
    print(f"  cấu trúc  : {parsed.stats}")
    if not parsed.root.find_all(NodeLevel.DIEU):
        print("Không tách được Điều nào - kiểm tra lại file nguồn trước khi dùng.")
        return 1
    print("\nChạy lại ingestion để cập nhật index: python scripts/run_ingestion.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
