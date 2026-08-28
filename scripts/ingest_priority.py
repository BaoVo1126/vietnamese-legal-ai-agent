from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legal_agent.config import get_settings
from legal_agent.ingestion.hf_corpus import (
    PRIORITY_DOCUMENTS,
    HFLegalCorpusLoader,
)
from legal_agent.ingestion.parser import StructureAwareParser
from legal_agent.logging_config import setup_logging


def verify_parse(paths: list[Path]) -> dict:
    parser = StructureAwareParser()
    rows, failures = [], []
    for path in paths:
        parsed = parser.parse(path.read_text(encoding="utf-8"), source_path=str(path))
        stats = parsed.stats
        rows.append({"file": path.name, "doc_number": parsed.meta.doc_number,
                     "title": parsed.meta.title,
                     "status": parsed.meta.effect_status.value, **stats})
        if stats["dieu"] == 0:
            failures.append(path.name)
    return {"documents": rows, "zero_article_documents": failures,
            "fallback_rate": round(len(failures) / len(rows), 4) if rows else 0.0}


def main() -> int:
    parser = argparse.ArgumentParser(description="Nạp văn bản pháp luật nền tảng")
    parser.add_argument("--dry-run", action="store_true",
                        help="Chỉ đối chiếu danh sách, không tải nội dung")
    parser.add_argument("--out", default=None, help="Thư mục ghi văn bản thô")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Chỉ nạp các nhãn khớp (khớp chuỗi con, không phân biệt hoa "
                             "thường) - dùng để bổ sung văn bản còn thiếu mà không tải lại "
                             "toàn bộ")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    settings = get_settings()
    out_dir = Path(args.out) if args.out else settings.abs_raw_data_dir

    loader = HFLegalCorpusLoader()
    rules = PRIORITY_DOCUMENTS
    if args.only:
        wanted = [term.lower() for term in args.only]
        rules = tuple(rule for rule in rules
                      if any(term in rule.label.lower() for term in wanted))
        if not rules:
            print(f"Không có nhãn nào khớp {args.only}")
            return 1
    report = loader.select_priority(rules)
    print("\n=== ĐỐI CHIẾU DANH SÁCH ƯU TIÊN ===")
    for record in report.selected:
        print(f" {record['label']:<45} id={record['id']:<8} "
              f"{record['so_ky_hieu'] or '(không số hiệu)':<16} "
              f"{record['tinh_trang_hieu_luc']}")
    for label in report.missing:
        print(f"   {label:<45} KHÔNG CÓ TRONG CORPUS")
    print(f"  -> coverage: {report.coverage:.0%}")

    if args.dry_run:
        return 0

    print("\n=== TẢI NỘI DUNG (row-group pruning) ===")
    candidate_ids = [candidate["id"] for record in report.selected
                     for candidate in record.get("candidates", [record])]
    html_by_id = loader.fetch_html(candidate_ids)
    written = loader.write_raw(report.selected, html_by_id, out_dir)
    print(f"  -> đã ghi {len(written)} file vào {out_dir}")

    print("\n=== KIỂM TRA PARSE (số liệu quyết định bước 3) ===")
    verification = verify_parse(written)
    for row in verification["documents"]:
        print(f"  {row['file']:<45} điều={row['dieu']:<5} khoản={row['khoan']:<5} "
              f"điểm={row['diem']:<5} {row['status']}")
    print(f"  -> tỷ lệ parse ra 0 Điều: {verification['fallback_rate']:.1%} "
          f"{verification['zero_article_documents']}")

    output = settings.abs_processed_data_dir / "priority_ingestion_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"selection": report.selected,
                                  "missing": report.missing, **verification},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nBáo cáo -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
