from __future__ import annotations

from ...domain.chunk import RetrievedChunk
from ...domain.enums import QueryIntent
from ...logging_config import get_logger
from ..state import AgentState, trace_entry
from .base import AgentContext

logger = get_logger(__name__)

_GRAPH_BOOST = 0.15
_MAX_EXPANSION_DOCS = 2
_MAX_EXPANSION_CHUNKS = 3

_HISTORY_INTENTS = {QueryIntent.HIEU_LUC_VAN_BAN.value, QueryIntent.SO_SANH_DOI_CHIEU.value}


class KnowledgeGraphNode:
    name = "kg_validate"

    def __init__(self, context: AgentContext) -> None:
        self.context = context

    def __call__(self, state: AgentState) -> dict:
        retrieved: list[RetrievedChunk] = list(state.get("retrieved") or [])
        if not retrieved:
            return {"trace": [trace_entry(self.name, skipped="không có bằng chứng")]}

        as_of = self.context.as_of_date(state.get("as_of"))
        keep_repealed = state.get("intent") in _HISTORY_INTENTS

        verdicts: dict[str, dict] = {}
        kept: list[RetrievedChunk] = []
        excluded: list[dict] = []
        notes: list[str] = []

        for item in retrieved:
            chunk = item.chunk
            verdict = self.context.graph_store.validate(chunk.doc_key, chunk.dieu, as_of)
            verdicts.setdefault(chunk.doc_key, verdict.as_dict())

            if verdict.is_citable or keep_repealed:
                if not verdict.is_citable:
                    item.graph_note = f"{chunk.doc_key} {verdict.status.display_name}"
                if verdict.guided_by or verdict.replaced_by:
                    item.graph_boost = _GRAPH_BOOST
                kept.append(item)
            else:
                excluded.append({
                    "citation": chunk.citation.render(),
                    "reason": f"{verdict.status.display_name}"
                              + (f" ({verdict.note})" if verdict.note else ""),
                })

        for doc_number, verdict in verdicts.items():
            note = self._note_for(doc_number, verdict)
            if note:
                notes.append(note)

        expansion = self._expand(state, verdicts, {item.chunk_id for item in kept},
                                 as_of=as_of, keep_repealed=keep_repealed)
        kept.extend(expansion)
        kept.sort(key=lambda item: item.final_score, reverse=True)

        logger.info("KG: giữ %d, loại %d, mở rộng %d", len(kept) - len(expansion),
                    len(excluded), len(expansion))
        return {
            "retrieved": kept,
            "graph_verdicts": list(verdicts.values()),
            "graph_notes": notes,
            "excluded_chunks": excluded,
            "trace": [trace_entry(self.name, kept=len(kept), excluded=excluded, notes=notes)],
        }

    @staticmethod
    def _note_for(doc_number: str, verdict: dict) -> str:
        """One human-readable sentence per document, injected into the answer prompt."""
        parts = [f"{doc_number}: {verdict['status_label']}"]
        if verdict["replaced_by"]:
            parts.append("bị thay thế bởi " + ", ".join(verdict["replaced_by"]))
        if verdict["amended_by"]:
            parts.append("bị sửa đổi bởi " + ", ".join(verdict["amended_by"]))
        if verdict["guided_by"]:
            parts.append("được hướng dẫn bởi " + ", ".join(
                f"{entry['doc_number']}"
                + (f" (Điều {entry['dieu']})" if entry.get("dieu") else "")
                for entry in verdict["guided_by"]))
        return "; ".join(parts) if len(parts) > 1 or verdict["status"] != "con_hieu_luc" else ""

    def _expand(self, state: AgentState, verdicts: dict[str, dict], already: set[str],
                as_of=None, keep_repealed: bool = False) -> list[RetrievedChunk]:
        targets: list[str] = []
        for verdict in verdicts.values():
            targets.extend(entry["doc_number"] for entry in verdict["guided_by"])
            targets.extend(verdict["replaced_by"])
        unique_targets = list(dict.fromkeys(number for number in targets if number))
        if not unique_targets:
            return []

        query = state.get("search_query") or state["question"]
        expansion: list[RetrievedChunk] = []
        for doc_number in unique_targets[:_MAX_EXPANSION_DOCS]:
            hits = self.context.retriever.retrieve(query, top_n=_MAX_EXPANSION_CHUNKS,
                                                   doc_keys=[doc_number])
            for hit in hits:
                if hit.chunk_id in already:
                    continue
                hit_verdict = self.context.graph_store.validate(
                    hit.chunk.doc_key, hit.chunk.dieu, as_of)
                if not hit_verdict.is_citable and not keep_repealed:
                    logger.info("Bỏ qua chunk mở rộng %s: %s", hit.chunk.citation.render(),
                                hit_verdict.status.display_name)
                    continue
                hit.source = "graph_expansion"
                hit.graph_boost = _GRAPH_BOOST
                hit.graph_note = f"bổ sung qua Knowledge Graph từ {doc_number}"
                expansion.append(hit)
                already.add(hit.chunk_id)
        if expansion:
            logger.info("KG expansion: +%d chunk từ %s", len(expansion),
                        unique_targets[:_MAX_EXPANSION_DOCS])
        return expansion
