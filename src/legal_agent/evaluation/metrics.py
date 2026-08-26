from __future__ import annotations
from dataclasses import asdict, dataclass, field
from statistics import mean

from ..domain.citation import Citation
from .dataset import EvalCase


@dataclass
class CaseResult:
    case_id: str
    question: str
    tags: list[str] = field(default_factory=list)
    expected_status: str = ""
    actual_status: str = ""
    status_correct: bool = False
    intent_correct: bool | None = None
    retrieval_recall: float = 0.0
    citation_recall: float = 0.0
    citation_precision: float = 0.0
    stale_citations: list[str] = field(default_factory=list)
    forbidden_hits: list[str] = field(default_factory=list)
    grounding_score: float = 0.0
    support_ratio: float = 0.0
    attempts: int = 0
    latency_ms: float = 0.0
    missing_citations: list[str] = field(default_factory=list)
    refusal_reason: str = ""

    @property
    def passed(self) -> bool:
        if not self.status_correct:
            return False
        if self.forbidden_hits or self.stale_citations:
            return False
        if self.expected_status == "refused":
            return True
        return self.citation_recall >= 1.0 and self.citation_precision >= 1.0

    def as_dict(self) -> dict:
        data = asdict(self)
        data["passed"] = self.passed
        return data


@dataclass
class EvalReport:
    results: list[CaseResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    def aggregate(self) -> dict:
        if not self.results:
            return {}
        answerable = [r for r in self.results if r.expected_status == "answered"]
        graded_intents = [r for r in self.results if r.intent_correct is not None]
        summary = {
            "total_cases": self.total,
            "pass_rate": _ratio(sum(1 for r in self.results if r.passed), self.total),
            "status_accuracy": _ratio(sum(1 for r in self.results if r.status_correct),
                                      self.total),
            "retrieval_recall": _mean([r.retrieval_recall for r in answerable]),
            "citation_recall": _mean([r.citation_recall for r in answerable]),
            "citation_precision": _mean([r.citation_precision for r in answerable]),
            "stale_citation_rate": _ratio(sum(1 for r in self.results if r.stale_citations),
                                          self.total),
            "avg_grounding": _mean([r.grounding_score for r in self.results]),
            "avg_support_ratio": _mean([r.support_ratio for r in self.results]),
            "retry_rate": _ratio(sum(1 for r in self.results if r.attempts > 1), self.total),
            "avg_latency_ms": _mean([r.latency_ms for r in self.results]),
        }
        if graded_intents:
            summary["intent_accuracy"] = _mean(
                [1.0 if r.intent_correct else 0.0 for r in graded_intents])
        return summary

    def failures(self) -> list[CaseResult]:
        return [result for result in self.results if not result.passed]

    def as_dict(self) -> dict:
        return {"aggregate": self.aggregate(),
                "cases": [result.as_dict() for result in self.results]}


def score_case(case: EvalCase, answer) -> CaseResult:
    expected = case.parsed_expected
    forbidden = case.parsed_forbidden

    retrieved = [Citation.parse_all(item["citation"])[0]
                 for item in (answer.evidence or [])
                 if Citation.parse_all(item["citation"])]
    emitted = [Citation.model_validate(item) for item in (answer.citations or [])]

    retrieval_hits = [target for target in expected
                      if any(_overlaps(candidate, target) for candidate in retrieved)]
    citation_hits = [target for target in expected
                     if any(_overlaps(candidate, target) for candidate in emitted)]
    supported = [candidate for candidate in emitted
                 if any(source.covers(candidate) for source in retrieved)]

    stale = [] if case.allow_stale_citations else [
        item["citation"] for item in (answer.evidence or [])
        if item.get("effect_status") == "het_hieu_luc"
        and _is_cited(item["citation"], emitted)
    ]
    forbidden_hits = [target.render() for target in forbidden
                      if any(_overlaps(candidate, target) for candidate in emitted)]

    return CaseResult(
        case_id=case.case_id,
        question=case.question,
        tags=list(case.tags),
        expected_status=case.expected_status,
        actual_status=answer.status,
        status_correct=answer.status == case.expected_status,
        intent_correct=(answer.intent == case.expected_intent) if case.expected_intent
                       else None,
        retrieval_recall=_ratio(len(retrieval_hits), len(expected)) if expected else 1.0,
        citation_recall=_ratio(len(citation_hits), len(expected)) if expected else 1.0,
        citation_precision=_ratio(len(supported), len(emitted)) if emitted else
                           (1.0 if case.expected_status == "refused" else 0.0),
        stale_citations=stale,
        forbidden_hits=forbidden_hits,
        grounding_score=answer.grounding_score,
        support_ratio=answer.support_ratio,
        attempts=answer.attempts,
        latency_ms=getattr(answer, "latency_ms", 0.0),
        missing_citations=[target.render() for target in expected
                           if target not in citation_hits],
        refusal_reason=answer.refusal_reason,
    )


def _overlaps(left: Citation, right: Citation) -> bool:
    return left.covers(right) or right.covers(left)


def _is_cited(citation_text: str, emitted: list[Citation]) -> bool:
    parsed = Citation.parse_all(citation_text)
    if not parsed:
        return False
    return any(_overlaps(candidate, parsed[0]) for candidate in emitted)


def _ratio(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _mean(values: list[float]) -> float:
    return round(mean(values), 4) if values else 0.0
