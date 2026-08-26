from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from ..logging_config import get_logger
from .dataset import EvalCase, load_golden_set
from .metrics import EvalReport, score_case

logger = get_logger(__name__)


class EvaluationRunner:
    def __init__(self, service) -> None:
        self.service = service

    def run(self, cases: list[EvalCase], progress=None) -> EvalReport:
        report = EvalReport()
        for index, case in enumerate(cases, start=1):
            logger.info("Eval [%d/%d] %s", index, len(cases), case.case_id)
            answer = self.service.ask(case.question, session_id=f"eval::{case.case_id}")
            report.results.append(score_case(case, answer))
            if progress is not None:
                progress(index, len(cases), report.results[-1])
        return report

    def run_from_path(self, path: Path, progress=None) -> EvalReport:
        return self.run(load_golden_set(path), progress=progress)


def save_report(report: EvalReport, directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    json_path = directory / f"eval-{stamp}.json"
    markdown_path = directory / f"eval-{stamp}.md"
    json_path.write_text(json.dumps(report.as_dict(), ensure_ascii=False, indent=2),
                         encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    logger.info("Đã lưu báo cáo eval -> %s", markdown_path)
    return json_path, markdown_path


def render_markdown(report: EvalReport) -> str:
    aggregate = report.aggregate()
    lines = ["# Báo cáo đánh giá - Trợ lý Pháp luật Việt Nam", "",
             f"Thời điểm: {datetime.now(UTC).isoformat(timespec='seconds')}", "",
             "## Chỉ số tổng hợp", "", "| Chỉ số | Giá trị |", "|---|---|"]
    labels = {
        "total_cases": "Số case",
        "pass_rate": "Tỷ lệ đạt toàn phần",
        "status_accuracy": "Đúng hành vi (trả lời/từ chối)",
        "intent_accuracy": "Đúng intent",
        "retrieval_recall": "Retrieval recall",
        "citation_recall": "Citation recall",
        "citation_precision": "Citation precision",
        "stale_citation_rate": "Tỷ lệ trích dẫn văn bản hết hiệu lực",
        "avg_grounding": "Grounding score trung bình",
        "avg_support_ratio": "Support ratio trung bình",
        "retry_rate": "Tỷ lệ phải self-correct",
        "avg_latency_ms": "Độ trễ trung bình (ms)",
    }
    for key, label in labels.items():
        if key in aggregate:
            lines.append(f"| {label} | {aggregate[key]} |")

    lines += ["", "## Chi tiết từng case", "",
              "| Case | Trạng thái | Đạt | Retr. recall | Cit. recall | Cit. prec. | Ghi chú |",
              "|---|---|---|---|---|---|---|"]
    for result in report.results:
        note = "; ".join(filter(None, [
            "trích dẫn hết hiệu lực: " + ", ".join(result.stale_citations)
            if result.stale_citations else "",
            "trích dẫn bị cấm: " + ", ".join(result.forbidden_hits)
            if result.forbidden_hits else "",
            "thiếu: " + ", ".join(result.missing_citations)
            if result.missing_citations and result.expected_status == "answered" else "",
        ])) or "-"
        lines.append(
            f"| {result.case_id} | {result.actual_status} | "
            f"{'✅' if result.passed else '❌'} | {result.retrieval_recall} | "
            f"{result.citation_recall} | {result.citation_precision} | {note} |"
        )

    failures = report.failures()
    if failures:
        lines += ["", "## Case chưa đạt", ""]
        for result in failures:
            lines += [f"### {result.case_id}", f"- Câu hỏi: {result.question}",
                      f"- Kỳ vọng: `{result.expected_status}` / Thực tế: "
                      f"`{result.actual_status}`",
                      f"- Lý do từ chối: {result.refusal_reason or '-'}",
                      f"- Thiếu trích dẫn: {result.missing_citations or '-'}", ""]
    return "\n".join(lines)


def diagnose_failures(service, report, cases, answer_terms_fn=None
                      ) -> tuple[list[dict], dict[str, int]]:
    from .diagnostics import FailureDiagnoser, probe_from_case

    by_id = {case.case_id: case for case in cases}
    diagnoser = FailureDiagnoser(service)
    rows: list[dict] = []
    tally: dict[str, int] = {}

    for result in report.failures():
        case = by_id.get(result.case_id)
        if case is None or not case.expected_citations:
            rows.append({"case_id": result.case_id, "layer": "khong_chan_doan",
                         "detail": "case không có nhãn vàng để đối chiếu"})
            continue
        terms = tuple(answer_terms_fn(result.case_id)) if answer_terms_fn else ()
        diagnosis = diagnoser.diagnose(probe_from_case(case, terms))
        layer = diagnosis.failing_layer.value
        tally[layer] = tally.get(layer, 0) + 1
        detail = next((entry.detail for entry in diagnosis.layers
                       if entry.verdict.value == "FAIL"), "")
        rows.append({"case_id": result.case_id, "layer": layer, "detail": detail})
    return rows, tally
