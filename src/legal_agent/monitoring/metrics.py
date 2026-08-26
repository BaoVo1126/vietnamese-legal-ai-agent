from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field
from statistics import mean
from typing import Any


@dataclass
class MetricsSummary:
    total_runs: int = 0
    answered: int = 0
    refused: int = 0
    errors: int = 0
    refusal_rate: float = 0.0
    retry_rate: float = 0.0
    avg_grounding: float = 0.0
    avg_support: float = 0.0
    avg_evidence: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_max_ms: float = 0.0
    node_latency_ms: dict[str, float] = field(default_factory=dict)
    intents: dict[str, int] = field(default_factory=dict)
    top_cited_documents: list[tuple[str, int]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_runs": self.total_runs,
            "answered": self.answered,
            "refused": self.refused,
            "errors": self.errors,
            "refusal_rate": self.refusal_rate,
            "retry_rate": self.retry_rate,
            "avg_grounding": self.avg_grounding,
            "avg_support": self.avg_support,
            "avg_evidence": self.avg_evidence,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_max_ms": self.latency_max_ms,
            "node_latency_ms": self.node_latency_ms,
            "intents": self.intents,
            "top_cited_documents": self.top_cited_documents,
        }


def summarise(records: list[dict[str, Any]]) -> MetricsSummary:
    if not records:
        return MetricsSummary()

    latencies = sorted(float(record.get("latency_ms", 0.0)) for record in records)
    statuses = Counter(record.get("status", "error") for record in records)
    answered = statuses.get("answered", 0)
    refused = statuses.get("refused", 0)
    errors = len(records) - answered - refused

    node_totals: dict[str, list[float]] = {}
    for record in records:
        for node, elapsed in (record.get("node_latency_ms") or {}).items():
            node_totals.setdefault(node, []).append(float(elapsed))

    cited = Counter(
        number for record in records for number in (record.get("citations") or []) if number
    )

    return MetricsSummary(
        total_runs=len(records),
        answered=answered,
        refused=refused,
        errors=errors,
        refusal_rate=round(refused / len(records), 4),
        retry_rate=round(sum(1 for r in records if r.get("retried")) / len(records), 4),
        avg_grounding=round(mean(float(r.get("grounding_score", 0.0)) for r in records), 4),
        avg_support=round(mean(float(r.get("support_ratio", 0.0)) for r in records), 4),
        avg_evidence=round(mean(float(r.get("evidence_count", 0)) for r in records), 2),
        latency_p50_ms=percentile(latencies, 0.50),
        latency_p95_ms=percentile(latencies, 0.95),
        latency_max_ms=round(latencies[-1], 1),
        node_latency_ms={node: round(mean(values), 1)
                         for node, values in sorted(node_totals.items())},
        intents=dict(Counter(r.get("intent", "") for r in records).most_common()),
        top_cited_documents=cited.most_common(5),
    )


def percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    index = max(0, min(len(sorted_values) - 1,
                       round(fraction * (len(sorted_values) - 1))))
    return round(sorted_values[index], 1)
