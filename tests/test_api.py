from __future__ import annotations
import pytest
from fastapi.testclient import TestClient

from legal_agent.agents.service import AgentAnswer
from legal_agent.api.deps import get_agent_service
from legal_agent.api.main import app


class FakeVectorStore:
    def count(self) -> int:
        return 42


class FakeService:
    def __init__(self, answer: AgentAnswer) -> None:
        self.answer = answer
        self.vector_store = FakeVectorStore()
        self.graph_store = type("G", (), {"all_documents": staticmethod(lambda: [])})()
        self.asked: list[str] = []

    def ask(self, question, session_id="", as_of=None) -> AgentAnswer:
        self.asked.append(question)
        return self.answer

    def close(self) -> None: ...


ANSWERED = AgentAnswer(
    question="Ai không có quyền thành lập doanh nghiệp?",
    answer="Theo quy định ... (Điều 17, Khoản 2, Luật Doanh nghiệp 59/2020/QH14)",
    status="answered", intent="hoi_dap_khai_niem",
    citations=[{"doc_number": "59/2020/QH14", "doc_title": "Luật Doanh nghiệp",
                "dieu": "17", "khoan": "2", "diem": None}],
    evidence=[{"citation": "Điều 17, Khoản 2, Luật Doanh nghiệp 59/2020/QH14",
               "effect_status": "con_hieu_luc", "score": 0.91, "source": "hybrid",
               "graph_note": "", "text": "Nội dung khoản 2.", "node_path": "…"}],
    grounding_score=0.9, support_ratio=1.0, attempts=1,
    trace=[{"node": "router"}],
)

REFUSED = AgentAnswer(
    question="Cách nấu phở?", answer="Tôi không đủ căn cứ pháp lý rõ ràng...",
    status="refused", intent="ngoai_pham_vi",
    refusal_reason="Câu hỏi ngoài phạm vi pháp luật.",
)


@pytest.fixture()
def client_for():
    def _build(answer: AgentAnswer) -> tuple[TestClient, FakeService]:
        service = FakeService(answer)
        app.dependency_overrides[get_agent_service] = lambda: service
        return TestClient(app), service

    yield _build
    app.dependency_overrides.clear()


class TestAsk:
    def test_answered_question(self, client_for):
        client, service = client_for(ANSWERED)
        response = client.post("/ask", json={"question": "Ai không có quyền thành lập?"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "answered"
        assert payload["is_refusal"] is False
        assert payload["citations"] == [
            "Điều 17, Khoản 2, Luật Doanh nghiệp 59/2020/QH14"]
        assert payload["trace"] is None 
        assert service.asked == ["Ai không có quyền thành lập?"]

    def test_refusal_is_a_200_response_not_an_error(self, client_for):
        client, _ = client_for(REFUSED)
        response = client.post("/ask", json={"question": "Cách nấu phở bò ngon nhất?"})
        assert response.status_code == 200
        assert response.json()["status"] == "refused"
        assert response.json()["is_refusal"] is True

    def test_trace_is_returned_on_request(self, client_for):
        client, _ = client_for(ANSWERED)
        response = client.post("/ask", json={"question": "Câu hỏi đủ dài để hợp lệ?",
                                             "include_trace": True})
        assert response.json()["trace"] == [{"node": "router"}]

    def test_too_short_question_is_rejected_by_validation(self, client_for):
        client, _ = client_for(ANSWERED)
        assert client.post("/ask", json={"question": "hi"}).status_code == 422


class TestHealth:
    def test_health_reports_backends(self, client_for):
        client, _ = client_for(ANSWERED)
        payload = client.get("/health").json()
        assert payload["status"] == "ok"
        assert payload["indexed_chunks"] == 42
        assert {"profile", "llm_backend", "graph_backend"} <= payload.keys()

    def test_liveness_probe_does_not_touch_the_service(self, client_for):
        client, _ = client_for(ANSWERED)
        assert client.get("/live").json() == {"status": "alive"}
