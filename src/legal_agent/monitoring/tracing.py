from __future__ import annotations

import os

from ..logging_config import get_logger

logger = get_logger(__name__)


def tracing_status() -> dict[str, object]:
    enabled = os.getenv("LANGSMITH_TRACING", "").lower() in {"1", "true", "yes"}
    has_key = bool(os.getenv("LANGSMITH_API_KEY"))
    project = os.getenv("LANGSMITH_PROJECT", "legal-agent-vn")
    status = {
        "enabled": enabled,
        "configured": enabled and has_key,
        "project": project,
    }
    if enabled and not has_key:
        logger.warning("LANGSMITH_TRACING bật nhưng thiếu LANGSMITH_API_KEY - "
                       "sẽ không có trace nào được gửi.")
    elif enabled:
        logger.info("LangSmith tracing: bật (project=%s)", project)
    return status
