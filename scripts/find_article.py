from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legal_agent.config import get_settings
from legal_agent.domain.enums import NodeLevel
from legal_agent.ingestion.parser import StructureAwareParser
from legal_agent.logging_config import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="Tìm Điều chứa các cụm từ")
    parser.add_argument("terms", nargs="+", help="Các cụm từ phải cùng xuất hiện")
    parser.add_argument("--doc", default=None, help="Giới hạn theo số hiệu văn bản")
    parser.add_argument("--limit", type=int, default=6)
    args = parser.parse_args()

    setup_logging("ERROR")
    structure_parser = StructureAwareParser()
    hits = 0
    for path in sorted(get_settings().abs_raw_data_dir.glob("*.txt")):
        parsed = structure_parser.parse(path.read_text(encoding="utf-8"), str(path))
        meta = parsed.meta
        if args.doc and meta.doc_number != args.doc:
            continue
        for node in parsed.root.find_all(NodeLevel.DIEU):
            text = node.full_text()
            lowered = text.lower()
            if all(term.lower() in lowered for term in args.terms):
                identity = meta.doc_number or meta.title
                print(f"  Điều {node.number:<5} | {identity:<18} | {meta.title[:32]:<32} "
                      f"| {node.title[:44]}")
                print(f"      {text[:150].replace(chr(10), ' ')}")
                hits += 1
                if hits >= args.limit:
                    return 0
    if not hits:
        print("(không tìm thấy)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
