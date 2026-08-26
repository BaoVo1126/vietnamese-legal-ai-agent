from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legal_agent.config import get_settings
from legal_agent.ingestion.pipeline import IngestionPipeline
from legal_agent.logging_config import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Nạp văn bản pháp luật vào Knowledge Base")
    parser.add_argument("--source", default=None, help="Thư mục chứa văn bản thô")
    parser.add_argument("--no-recreate", action="store_true",
                        help="Không xoá collection cũ trước khi nạp")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    settings = get_settings()
    if settings.qdrant_mode == "memory":
        print("⚠️  QDRANT_MODE=memory: index chỉ tồn tại trong tiến trình này.\n"
              "    Đặt QDRANT_MODE=server để ingestion có tác dụng lâu dài.")

    source = Path(args.source) if args.source else None
    result = IngestionPipeline(settings).run(source_dir=source, recreate=not args.no_recreate)
    print(json.dumps(result.report, ensure_ascii=False, indent=2))
    for warning in result.warnings:
        print(f"  ! {warning}")


if __name__ == "__main__":
    main()
