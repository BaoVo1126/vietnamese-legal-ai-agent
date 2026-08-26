from __future__ import annotations
import json
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..logging_config import get_logger

logger = get_logger(__name__)

_LOCK = threading.Lock()


class RunRecorder:
    def __init__(self, path: Path, enabled: bool = True) -> None:
        self.path = path
        self.enabled = enabled
        if enabled:
            path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, answer, latency_ms: float, session_id: str = "",
               error: str = "") -> dict[str, Any]:
        node_timings = _node_timings(getattr(answer, "trace", []) or [])
        record = {
            "run_id": str(uuid.uuid4()),
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "session_id": session_id,
            "question": getattr(answer, "question", ""),
            "status": getattr(answer, "status", "error"),
            "intent": getattr(answer, "intent", ""),
            "latency_ms": round(latency_ms, 1),
            "grounding_score": getattr(answer, "grounding_score", 0.0),
            "support_ratio": getattr(answer, "support_ratio", 0.0),
            "attempts": getattr(answer, "attempts", 0),
            "retried": getattr(answer, "attempts", 0) > 1,
            "evidence_count": len(getattr(answer, "evidence", []) or []),
            "excluded_count": len(getattr(answer, "excluded_chunks", []) or []),
            "citations": [item.get("doc_number", "") for item in
                          (getattr(answer, "citations", []) or [])],
            "answer_chars": len(getattr(answer, "answer", "") or ""),
            "refusal_reason": getattr(answer, "refusal_reason", ""),
            "node_latency_ms": node_timings,
            "error": error,
        }
        self._append(record)
        return record

    def _append(self, record: dict[str, Any]) -> None:
        if not self.enabled:
            return
        line = json.dumps(record, ensure_ascii=False)
        try:
            with _LOCK, self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError as error:  
            logger.warning("Không ghi được run log: %s", error)

    def read_all(self, limit: int | None = None) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records[-limit:] if limit else records


class Stopwatch:
    def __enter__(self) -> Stopwatch:
        self._start = time.perf_counter()
        self.elapsed_ms = 0.0
        return self

    def __exit__(self, *exc_info) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0


def _node_timings(trace: list[dict[str, Any]]) -> dict[str, float]:
    timings: dict[str, float] = {}
    for entry in trace:
        elapsed = entry.get("elapsed_ms")
        if elapsed is None:
            continue
        node = entry.get("node", "unknown")
        timings[node] = round(timings.get(node, 0.0) + float(elapsed), 1)
    return timings
