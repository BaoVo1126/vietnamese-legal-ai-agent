from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..domain.citation import Citation
from ..logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class EvalCase:
    case_id: str
    question: str
    expected_citations: list[str] = field(default_factory=list)
    forbidden_citations: list[str] = field(default_factory=list)
    expected_status: str = "answered"         
    expected_intent: str = ""                 
    allow_stale_citations: bool = False
    tags: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def parsed_expected(self) -> list[Citation]:
        return _parse_each(self.expected_citations)

    @property
    def parsed_forbidden(self) -> list[Citation]:
        return _parse_each(self.forbidden_citations)

    @classmethod
    def from_dict(cls, raw: dict, index: int) -> EvalCase:
        return cls(
            case_id=str(raw.get("case_id") or f"case-{index:03d}"),
            question=raw["question"],
            expected_citations=list(raw.get("expected_citations", [])),
            forbidden_citations=list(raw.get("forbidden_citations", [])),
            expected_status=raw.get("expected_status", "answered"),
            expected_intent=raw.get("expected_intent", ""),
            allow_stale_citations=bool(raw.get("allow_stale_citations", False)),
            tags=list(raw.get("tags", [])),
            note=raw.get("note", ""),
        )


def load_golden_set(path: Path) -> list[EvalCase]:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy golden set tại {path}")
    cases: list[EvalCase] = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cases.append(EvalCase.from_dict(json.loads(line), index))
    logger.info("Đã nạp %d eval case từ %s", len(cases), path)
    return cases


def _parse_each(values: list[str]) -> list[Citation]:
    citations: list[Citation] = []
    for value in values:
        parsed = Citation.parse_all(value)
        if parsed:
            citations.append(parsed[0])
        else:
            logger.warning("Không parse được trích dẫn kỳ vọng: %r", value)
    return citations
