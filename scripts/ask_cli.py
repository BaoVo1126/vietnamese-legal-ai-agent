from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legal_agent.agents import LegalAgentService
from legal_agent.config import get_settings
from legal_agent.logging_config import setup_logging

DEMO_QUESTIONS = [
    "Ai không có quyền thành lập và quản lý doanh nghiệp tại Việt Nam?",
    "Thủ tục đăng ký doanh nghiệp theo Điều 26 Luật Doanh nghiệp 59/2020/QH14 "
    "gồm những phương thức nào?",
    "Nghị định nào đang hướng dẫn Điều 26 của Luật Doanh nghiệp 59/2020/QH14 và còn hiệu lực?",
    "Luật Doanh nghiệp 68/2014/QH13 còn hiệu lực không?",
    "Cách nấu phở bò ngon nhất là gì?",
]


def render(answer, show_trace: bool) -> None:
    print("=" * 88)
    print(f" {answer.question}")
    print(f"   intent={answer.intent} | status={answer.status} | "
          f"grounding={answer.grounding_score:.2f} | support={answer.support_ratio:.2f} | "
          f"attempts={answer.attempts}")
    print("-" * 88)
    print(answer.answer)
    if answer.evidence:
        print("-" * 88)
        print("Bằng chứng đã dùng:")
        for item in answer.evidence:
            print(f"  · [{item['score']:.4f}] {item['citation']} "
                  f"({item['effect_status']}, {item['source']})")
    if show_trace:
        print("-" * 88)
        print(json.dumps(answer.trace, ensure_ascii=False, indent=2))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Hỏi trợ lý pháp luật Việt Nam")
    parser.add_argument("question", nargs="*", help="Câu hỏi")
    parser.add_argument("--demo", action="store_true", help="Chạy bộ câu hỏi mẫu")
    parser.add_argument("--trace", action="store_true", help="In toàn bộ trace của graph")
    parser.add_argument("--as-of", default=None, help="Ngày tham chiếu hiệu lực (YYYY-MM-DD)")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    setup_logging(args.log_level)
    service = LegalAgentService(get_settings())
    print("Bootstrap:", service.bootstrap())

    questions = DEMO_QUESTIONS if args.demo else [" ".join(args.question)]
    if not questions or not questions[0]:
        parser.error("Cần truyền câu hỏi hoặc dùng --demo")
    for question in questions:
        render(service.ask(question, as_of=args.as_of), show_trace=args.trace)


if __name__ == "__main__":
    main()
