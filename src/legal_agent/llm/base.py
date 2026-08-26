from __future__ import annotations
import json
import re
from typing import Any, Protocol, runtime_checkable

from ..logging_config import get_logger

logger = get_logger(__name__)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


@runtime_checkable
class LLMClient(Protocol):
    def complete(self, system: str, user: str, *, task: str = "generic",
                 temperature: float | None = None, max_tokens: int | None = None) -> str: ...

    def complete_json(self, system: str, user: str, *, task: str = "generic",
                      default: dict | None = None, **kwargs: Any) -> dict: ...


def extract_json(raw: str) -> dict | None:
    if not raw:
        return None
    candidates: list[str] = []
    fenced = _JSON_FENCE_RE.search(raw)
    if fenced:
        candidates.append(fenced.group(1))
    stripped = raw.strip()
    candidates.append(stripped)
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end > start:
        candidates.append(stripped[start:end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    logger.warning("Không parse được JSON từ phản hồi LLM: %.160s", raw)
    return None


class BaseLLMClient:
    def complete_json(self, system: str, user: str, *, task: str = "generic",
                      default: dict | None = None, **kwargs: Any) -> dict:
        raw = self.complete(system, user, task=task, **kwargs)
        parsed = extract_json(raw)
        if parsed is None:
            logger.warning("Task %s trả về JSON không hợp lệ - dùng giá trị mặc định.", task)
            return dict(default or {})
        return parsed

    def complete(self, system: str, user: str, *, task: str = "generic",
                 temperature: float | None = None,
                 max_tokens: int | None = None) -> str:  
        raise NotImplementedError
