from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legal_agent.agents import LegalAgentService
from legal_agent.config import get_settings
from legal_agent.evaluation import (
    EvaluationRunner,
    diagnose_failures,
    render_markdown,
    save_report,
)
from legal_agent.logging_config import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="Đánh giá trợ lý pháp luật")
    parser.add_argument("--dataset", default=None, help="Đường dẫn golden set (.jsonl)")
    parser.add_argument("--regression", action="store_true",
                        help="Dùng bộ regression nhiều lĩnh vực trong "
                             "legal_agent.evaluation.datasets.legal_qa_regression")
    parser.add_argument("--tag", default=None, help="Chỉ chạy các case mang tag này")
    parser.add_argument("--diagnose-failures", action="store_true",
                        help="Quy mỗi case chưa đạt về đúng một tầng lỗi a/b/c/d")
    parser.add_argument("--fail-under", type=float, default=0.0,
                        help="Thoát mã lỗi nếu pass_rate thấp hơn ngưỡng (dùng cho CI)")
    parser.add_argument("--no-save", action="store_true", help="Không ghi file báo cáo")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    setup_logging(args.log_level)
    settings = get_settings()

    service = LegalAgentService(settings)
    print("Bootstrap:", service.bootstrap())

    def progress(index: int, total: int, result) -> None:
        mark = "PASS" if result.passed else "FAIL"
        print(f"[{index:>2}/{total}] {mark}  {result.case_id}  "
              f"(status={result.actual_status}, cit_recall={result.citation_recall})")

    runner = EvaluationRunner(service)
    if args.regression:
        from legal_agent.evaluation.datasets import as_eval_cases

        cases = as_eval_cases()
        if args.tag:
            cases = [case for case in cases if args.tag in case.tags]
        print(f"Regression set: {len(cases)} case")
        print()
        report = runner.run(cases, progress=progress)
    else:
        dataset = Path(args.dataset) if args.dataset else settings.abs_eval_dataset_path
        report = runner.run_from_path(dataset, progress=progress)

    print()
    print(render_markdown(report))

    if args.diagnose_failures and report.failures():
        _print_failure_layers(service, report, regression=args.regression)

    if not args.no_save:
        json_path, markdown_path = save_report(report, settings.abs_eval_report_dir)
        print(f"\nĐã lưu: {json_path.name}, {markdown_path.name}")

    aggregate = report.aggregate()
    print()
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))

    if args.fail_under and aggregate.get("pass_rate", 0.0) < args.fail_under:
        print(f"\n pass_rate {aggregate.get('pass_rate')} < ngưỡng {args.fail_under}")
        return 1
    return 0


def _print_failure_layers(service, report, regression: bool) -> None:
    """In bảng quy tầng lỗi cho từng case chưa đạt."""
    if regression:
        from legal_agent.evaluation.datasets import as_eval_cases
        from legal_agent.evaluation.datasets.legal_qa_regression import answer_terms

        cases = as_eval_cases()
        terms_for = answer_terms
    else:
        from legal_agent.evaluation.dataset import load_golden_set

        cases = load_golden_set(get_settings().abs_eval_dataset_path)
        terms_for = None

    rows, tally = diagnose_failures(service, report, cases, terms_for)
    print()
    print("=== TẦNG LỖI CỦA CÁC CASE CHƯA ĐẠT ===")
    for row in rows:
        print(f"  {row['case_id']:<34} {row['layer']:<14} {row['detail'][:70]}")
    print(f"  -> phân bố tầng lỗi: {json.dumps(tally, ensure_ascii=False)}")


if __name__ == "__main__":
    raise SystemExit(main())
