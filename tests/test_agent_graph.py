from __future__ import annotations
import json
import pytest
from legal_agent.agents.graph import build_agent_graph
from legal_agent.agents.nodes.base import AgentContext
from legal_agent.agents.state import initial_state
from legal_agent.domain.enums import EffectStatus, QueryIntent
from legal_agent.kg.base import GraphVerdict
from legal_agent.llm.base import BaseLLMClient

from .conftest import make_retrieved

ANSWER_WITH_GOOD_CITATION = (
    "Các đối tượng không có quyền thành lập doanh nghiệp được liệt kê tại khoản 2 "
    "(Điều 17, Khoản 2, Luật Doanh nghiệp 59/2020/QH14)."
)
ANSWER_WITH_FABRICATED_CITATION = (
    "Doanh nghiệp phải nộp lệ phí môn bài hằng năm "
    "(Điều 4, Khoản 1, Nghị định 139/2016/NĐ-CP)."
)


class ScriptedLLM(BaseLLMClient):
    def __init__(self, *, intent: str = QueryIntent.HOI_DAP_KHAI_NIEM.value,
                 grounding: float | list[float] = 0.9,
                 answer: str = ANSWER_WITH_GOOD_CITATION,
                 claim_verdict: str = "supported") -> None:
        self.intent = intent
        self.grounding = grounding if isinstance(grounding, list) else [grounding]
        self.answer = answer
        self.claim_verdict = claim_verdict
        self.calls: list[str] = []

    def complete(self, system: str, user: str, *, task: str = "generic", **kwargs) -> str:
        self.calls.append(task)
        if task == "router":
            return json.dumps({"intent": self.intent, "rewritten_query": "truy vấn đã viết lại",
                               "sub_queries": [], "doc_numbers": [], "dieu_hints": []})
        if task == "verifier":
            index = min(self.calls.count("verifier") - 1, len(self.grounding) - 1)
            score = self.grounding[index]
            return json.dumps({"grounding_score": score, "is_sufficient": score >= 0.6,
                               "missing_information": "thiếu quy định cốt lõi",
                               "rewritten_query": "truy vấn tốt hơn"})
        if task == "answer":
            return self.answer
        if task == "claim_extraction":
            return json.dumps({"claims": [{"text": self.answer, "citation": ""}]})
        if task == "claim_verification":
            return json.dumps({"verdicts": [{"index": 0, "verdict": self.claim_verdict,
                                             "reason": "kiểm chứng giả lập"}]})
        return ""


class FakeVectorStore:
    def fetch_by_citation(self, doc_key, dieu=None, limit=20):
        return []


class FakeRetriever:
    def __init__(self, results=None) -> None:
        self.results = results if results is not None else [make_retrieved()]
        self.vector_store = FakeVectorStore()
        self.calls = 0

    def retrieve(self, query, top_n=None, only_in_force=False, doc_keys=None):
        self.calls += 1
        if doc_keys:
            return [item for item in self.results if item.chunk.doc_key in doc_keys]
        return list(self.results)


class FakeGraphStore:
    def __init__(self, status: EffectStatus = EffectStatus.CON_HIEU_LUC,
                 guided_by=None, replaced_by=None) -> None:
        self.status = status
        self.guided_by = guided_by or []
        self.replaced_by = replaced_by or []
        self.queried: list[str] = []

    def validate(self, doc_number, dieu=None, as_of=None) -> GraphVerdict:
        self.queried.append(doc_number)
        return GraphVerdict(doc_number=doc_number, status=self.status,
                            replaced_by=list(self.replaced_by),
                            guided_by=list(self.guided_by))

    def upsert_document(self, meta): ...
    def upsert_relation(self, relation): ...
    def get_document(self, doc_number): return None
    def all_documents(self): return []
    def neighbours(self, doc_number, relation): return []
    def incoming(self, doc_number, relation): return []
    def close(self): ...


def run(settings, llm=None, retriever=None, graph_store=None, question="Câu hỏi thử nghiệm?"):
    context = AgentContext(llm=llm or ScriptedLLM(), retriever=retriever or FakeRetriever(),
                           graph_store=graph_store or FakeGraphStore(), settings=settings)
    graph = build_agent_graph(context)
    return graph.invoke(initial_state(question)), context


class TestHappyPath:
    def test_grounded_question_is_answered_with_citation_and_disclaimer(self, settings):
        state, _ = run(settings)
        assert state["status"] == "answered"
        assert "Điều 17, Khoản 2, Luật Doanh nghiệp 59/2020/QH14" in state["answer"]
        assert "Căn cứ pháp lý đã đối chiếu" in state["answer"]
        assert "không thay thế ý kiến tư vấn" in state["answer"]
        assert state["attempts"] == 1

    def test_trace_records_every_node(self, settings):
        state, _ = run(settings)
        visited = [entry["node"] for entry in state["trace"]]
        assert visited == ["router", "retrieve", "kg_validate", "verify", "answer",
                           "citation_check"]


class TestRouting:
    def test_out_of_scope_question_never_reaches_retrieval(self, settings):
        retriever = FakeRetriever()
        state, _ = run(settings, llm=ScriptedLLM(intent=QueryIntent.NGOAI_PHAM_VI.value),
                       retriever=retriever)
        assert state["status"] == "refused"
        assert retriever.calls == 0
        assert [entry["node"] for entry in state["trace"]] == ["router", "refuse"]


class TestSelfCorrection:
    def test_weak_grounding_triggers_one_retry_then_succeeds(self, settings):
        retriever = FakeRetriever()
        llm = ScriptedLLM(grounding=[0.2, 0.9])
        state, _ = run(settings, llm=llm, retriever=retriever)
        assert retriever.calls == 2
        assert state["attempts"] == 2
        assert state["status"] == "answered"

    def test_rewritten_query_is_fed_back_into_retrieval(self, settings):
        state, _ = run(settings, llm=ScriptedLLM(grounding=[0.2, 0.9]))
        assert state["search_query"] == "truy vấn tốt hơn"

    def test_persistently_weak_grounding_refuses_within_the_attempt_budget(self, settings):
        retriever = FakeRetriever()
        state, _ = run(settings, llm=ScriptedLLM(grounding=[0.1]), retriever=retriever)
        assert state["status"] == "refused"
        assert retriever.calls == settings.max_retrieval_attempts
        assert "không đủ căn cứ pháp lý" in state["answer"].lower()

    def test_empty_retrieval_refuses_without_calling_the_answer_agent(self, settings):
        llm = ScriptedLLM()
        state, _ = run(settings, llm=llm, retriever=FakeRetriever(results=[]))
        assert state["status"] == "refused"
        assert "answer" not in llm.calls


class TestPostHocGate:
    def test_fabricated_citation_is_rejected(self, settings):
        state, _ = run(settings, llm=ScriptedLLM(answer=ANSWER_WITH_FABRICATED_CITATION))
        assert state["status"] == "refused"
        assert "139/2016/NĐ-CP" in state["refusal_reason"]

    def test_unsupported_claims_below_threshold_are_rejected(self, settings):
        state, _ = run(settings, llm=ScriptedLLM(claim_verdict="unsupported"))
        assert state["status"] == "refused"
        assert "luận điểm" in state["refusal_reason"].lower()
        assert state["support_ratio"] == 0.0

    def test_answer_without_any_citation_is_rejected(self, settings):
        state, _ = run(settings, llm=ScriptedLLM(answer="Doanh nghiệp được tự do kinh doanh."))
        assert state["status"] == "refused"
        assert "trích dẫn" in state["refusal_reason"].lower()


class TestVersionAwareness:
    def test_repealed_evidence_is_dropped_for_a_normal_question(self, settings):
        state, _ = run(settings, graph_store=FakeGraphStore(
            status=EffectStatus.HET_HIEU_LUC, replaced_by=["59/2020/QH14"]))
        assert state["retrieved"] == [] or all(
            item.chunk.effect_status.is_citable for item in state["retrieved"])
        assert state["excluded_chunks"]
        assert state["status"] == "refused"

    def test_repealed_evidence_is_kept_for_a_validity_question(self, settings):
        llm = ScriptedLLM(intent=QueryIntent.HIEU_LUC_VAN_BAN.value)
        state, _ = run(settings, llm=llm,
                       graph_store=FakeGraphStore(status=EffectStatus.HET_HIEU_LUC,
                                                  replaced_by=["59/2020/QH14"]))
        assert state["retrieved"], "câu hỏi về hiệu lực phải giữ lại văn bản hết hiệu lực"
        assert state["excluded_chunks"] == []
        assert any("hết hiệu lực" in note for note in state["graph_notes"])

    def test_graph_notes_reach_the_final_answer(self, settings):
        state, _ = run(settings, graph_store=FakeGraphStore(
            guided_by=[("01/2021/NĐ-CP", "26")]))
        assert state["status"] == "answered"
        assert "01/2021/NĐ-CP" in state["answer"]


@pytest.mark.parametrize("verdict,expected", [("supported", 1.0),
                                              ("partially_supported", 0.5),
                                              ("unsupported", 0.0)])
def test_support_ratio_weighting(settings, verdict, expected):
    state, _ = run(settings, llm=ScriptedLLM(claim_verdict=verdict))
    assert state["support_ratio"] == expected
