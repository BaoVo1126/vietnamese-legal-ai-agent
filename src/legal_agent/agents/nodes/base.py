from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from ...config import Settings
from ...kg.base import LegalGraphStore
from ...llm.base import LLMClient
from ...retrieval.hybrid import HybridRetriever


@dataclass
class AgentContext:
    llm: LLMClient
    retriever: HybridRetriever
    graph_store: LegalGraphStore
    settings: Settings

    def as_of_date(self, value: str | None) -> date | None:
        if not value:
            return None
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
