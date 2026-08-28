from __future__ import annotations
from ...domain.citation import Citation
from ...llm.prompts import (
    CLAIM_EXTRACTION_SYSTEM,
    CLAIM_EXTRACTION_USER,
    CLAIM_VERIFICATION_SYSTEM,
    CLAIM_VERIFICATION_USER,
    DISCLAIMER,
    format_evidence,
)
from ...logging_config import get_logger
from ..state import AgentState, trace_entry
from .base import AgentContext

logger = get_logger(__name__)

_VERDICT_WEIGHTS = {"supported": 1.0, "partially_supported": 0.5, "unsupported": 0.0}


class CitationCheckerNode:
    name = "citation_check"

    def __init__(self, context: AgentContext) -> None:
        self.context = context

    def __call__(self, state: AgentState) -> dict:
        draft = (state.get("answer") or "").strip()
        retrieved = state.get("retrieved") or []
        if not draft or not retrieved:
            return {
                "support_ratio": 0.0,
                "refusal_reason": state.get("refusal_reason")
                                  or "Không có câu trả lời nháp để kiểm chứng.",
                "trace": [trace_entry(self.name, skipped=True)],
            }

        evidence_citations = [item.chunk.citation for item in retrieved]
        cited = Citation.parse_cited(draft)
        unsupported_citations = [
            citation.render() for citation in cited
            if not any(evidence.covers(citation) for evidence in evidence_citations)
        ]

        claims = self._extract_claims(draft)
        verdicts = self._verify_claims(claims, retrieved)
        support_ratio = _support_ratio(verdicts)
        threshold = self.context.settings.claim_support_threshold

        passed = not unsupported_citations and support_ratio >= threshold and bool(cited)
        reason = self._failure_reason(cited, unsupported_citations, support_ratio, threshold)

        update: dict = {
            "claims": claims,
            "claim_verdicts": verdicts,
            "support_ratio": support_ratio,
            "trace": [trace_entry(self.name, passed=passed, support_ratio=support_ratio,
                                  unsupported_citations=unsupported_citations,
                                  claims=len(claims))],
        }
        if passed:
            update["answer"] = self._finalise(draft, retrieved, state)
            update["status"] = "answered"
            update["refusal_reason"] = ""
        else:
            update["refusal_reason"] = reason
            update["search_query"] = state["question"]
            update["doc_hints"] = []
            logger.warning("Post-hoc gate KHÔNG đạt: %s", reason)
        return update

    def _extract_claims(self, draft: str) -> list[dict]:
        payload = self.context.llm.complete_json(
            CLAIM_EXTRACTION_SYSTEM, CLAIM_EXTRACTION_USER.format(answer=draft),
            task="claim_extraction", default={"claims": []},
        )
        claims: list[dict] = []
        for item in payload.get("claims", []):
            text = str(item.get("text", "")).strip() if isinstance(item, dict) else str(item)
            if len(text) >= 15:
                claims.append({"text": text,
                               "citation": str(item.get("citation", "")).strip()
                               if isinstance(item, dict) else ""})
        return claims

    def _verify_claims(self, claims: list[dict], retrieved) -> list[dict]:
        if not claims:
            return []
        rendered = "\n".join(f"[{index}] {claim['text']}"
                             for index, claim in enumerate(claims))
        payload = self.context.llm.complete_json(
            CLAIM_VERIFICATION_SYSTEM,
            CLAIM_VERIFICATION_USER.format(evidence=format_evidence(retrieved),
                                           claims=rendered),
            task="claim_verification", default={"verdicts": []},
        )
        verdicts: list[dict] = []
        for item in payload.get("verdicts", []):
            if not isinstance(item, dict):
                continue
            verdict = str(item.get("verdict", "unsupported"))
            index = item.get("index")
            verdicts.append({
                "index": index,
                "verdict": verdict if verdict in _VERDICT_WEIGHTS else "unsupported",
                "reason": str(item.get("reason", "")),
                "claim": claims[index]["text"] if isinstance(index, int)
                         and 0 <= index < len(claims) else "",
            })
        missing = len(claims) - len(verdicts)
        for offset in range(missing):
            verdicts.append({"index": len(verdicts) + offset, "verdict": "unsupported",
                             "reason": "Không nhận được phán quyết cho luận điểm này.",
                             "claim": ""})
        return verdicts

    @staticmethod
    def _failure_reason(cited: list[Citation], unsupported: list[str],
                        support_ratio: float, threshold: float) -> str:
        if not cited:
            return ("Câu trả lời không có trích dẫn đúng định dạng "
                    "(Điều …, Khoản …, Tên văn bản Số hiệu) để kiểm chứng.")
        if unsupported:
            return ("Câu trả lời trích dẫn văn bản không có trong bằng chứng: "
                    + ", ".join(unsupported))
        if support_ratio < threshold:
            return (f"Tỷ lệ luận điểm được chứng minh chỉ đạt {support_ratio:.0%} "
                    f"(ngưỡng {threshold:.0%}).")
        return ""

    @staticmethod
    def _finalise(draft: str, retrieved, state: AgentState) -> str:
        sources = "\n".join(
            f"- {item.chunk.citation.render()} "
            f"[{item.chunk.effect_status.display_name}]"
            + (f" - {item.graph_note}" if item.graph_note else "")
            for item in retrieved
        )
        sections = [draft, f"**Căn cứ pháp lý đã đối chiếu:**\n{sources}"]
        notes = state.get("graph_notes") or []
        if notes:
            sections.append("**Ghi chú hiệu lực (Knowledge Graph):**\n"
                            + "\n".join(f"- {note}" for note in notes))
        sections.append(DISCLAIMER)
        return "\n\n".join(sections)


def _support_ratio(verdicts: list[dict]) -> float:
    if not verdicts:
        return 0.0
    total = sum(_VERDICT_WEIGHTS.get(verdict["verdict"], 0.0) for verdict in verdicts)
    return round(total / len(verdicts), 3)
