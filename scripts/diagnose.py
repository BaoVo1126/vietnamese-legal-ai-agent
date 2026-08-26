from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legal_agent.agents import LegalAgentService
from legal_agent.config import get_settings
from legal_agent.evaluation.diagnostics import CANARY_PROBES, FailureDiagnoser
from legal_agent.logging_config import setup_logging

SYMBOL = {"PASS": "✅", "FAIL": "❌", "BLOCKED": "⛔"}


def render(diagnosis) -> None:
    print("=" * 92)
    print(f" {diagnosis.probe.question}")
    print(f"   nhãn: {diagnosis.probe.label} | "
          f"kỳ vọng: Điều {diagnosis.probe.expected_dieu} của văn bản chứa "
          f"{list(diagnosis.probe.doc_keywords)}")
    print("-" * 92)
    for result in diagnosis.layers:
        print(f"{SYMBOL[result.verdict.value]} [{result.layer.value}] {result.detail}")
        top_k = result.evidence.get("top_k")
        if top_k:
            print("      top-5 Hybrid Retrieval:")
            for row in top_k:
                print(f"        {row['rank']}. {row['citation']}")
                print(f"           dense={row['dense']} sparse={row['sparse']} "
                      f"fusion={row['fusion']} rerank={row['rerank']} ({row['source']})")
        if result.evidence.get("corpus"):
            print(f"      KB hiện có: {result.evidence['corpus']}")
        if result.evidence.get("citations"):
            print(f"      trích dẫn: {result.evidence['citations']}")
        if result.evidence.get("missing_terms"):
            print(f"      thiếu cụm từ: {result.evidence['missing_terms']}")
    print("-" * 92)
    print(f"  TẦNG LỖI: {diagnosis.failing_layer.value}   "
          f"(status câu trả lời: {diagnosis.answer_status})")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Chẩn đoán 4 tầng lỗi của pipeline")
    parser.add_argument("--json", default=None, help="Ghi báo cáo JSON ra file")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--log-level", default="ERROR")
    args = parser.parse_args()

    setup_logging(args.log_level)
    service = LegalAgentService(get_settings())
    print("Bootstrap:", service.bootstrap())
    print()

    diagnoser = FailureDiagnoser(service, top_k=args.top_k)
    results = []
    for probe in CANARY_PROBES:
        diagnosis = diagnoser.diagnose(probe)
        render(diagnosis)
        results.append(diagnosis.as_dict())

    summary = {}
    for item in results:
        summary[item["failing_layer"]] = summary.get(item["failing_layer"], 0) + 1
    print("TỔNG HỢP TẦNG LỖI:", json.dumps(summary, ensure_ascii=False))

    if args.json:
        Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
        print(f"Đã ghi báo cáo -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
