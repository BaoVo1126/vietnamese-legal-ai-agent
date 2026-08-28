from __future__ import annotations

import pytest

from legal_agent.agents.service import LegalAgentService

pytestmark = pytest.mark.slow

CONSTITUTION_QUESTION = ("Theo Hiến pháp 2013, cơ quan nào là cơ quan quyền lực nhà nước "
                         "cao nhất của nước Cộng hòa xã hội chủ nghĩa Việt Nam?")
CRIMINAL_QUESTION = ("Người từ đủ bao nhiêu tuổi trở lên phải chịu trách nhiệm hình sự "
                     "về mọi tội phạm theo Bộ luật Hình sự 2015?")


@pytest.fixture(scope="module")
def service(settings) -> LegalAgentService:
    agent = LegalAgentService(settings)
    agent.bootstrap()
    return agent


def test_ingestion_indexes_the_real_corpus(service):
    documents = service.graph_store.all_documents()
    assert len(documents) >= 10
    assert service.vector_store.count() > 1000
    assert service.bm25_index.size == service.vector_store.count()
    numbers = {document.doc_number for document in documents}
    assert {"100/2015/QH13", "91/2015/QH13"} <= numbers


def test_document_without_a_so_hieu_is_indexed_and_citable(service):
    constitution = next(
        (document for document in service.graph_store.all_documents()
         if "Hiến pháp" in document.title), None)
    assert constitution is not None
    assert constitution.doc_number == ""
    assert constitution.doc_key == constitution.title

    answer = service.ask(CONSTITUTION_QUESTION)
    assert answer.status == "answered"
    assert any("Hiến pháp" in item["citation"] for item in answer.evidence)
    assert "Quốc hội" in answer.answer


def test_answer_leads_with_a_direct_answer_and_a_citation(service):
    answer = service.ask(CRIMINAL_QUESTION)
    assert answer.status == "answered"
    assert answer.answer.startswith("**Đáp án:")
    assert "16 tuổi" in answer.answer
    assert any("Điều 12" in item["citation"] for item in answer.evidence)
    assert "không thay thế ý kiến tư vấn" in answer.answer


def test_partially_repealed_code_stays_citable(service):
    verdict = service.graph_store.validate("100/2015/QH13")
    assert verdict.is_citable, verdict.status
    answer = service.ask(CRIMINAL_QUESTION)
    assert any("100/2015/QH13" in item["citation"] for item in answer.evidence)


def test_every_cited_document_is_in_force_enough_to_cite(service):
    answer = service.ask("Người thành niên là người từ bao nhiêu tuổi?")
    assert answer.status == "answered"
    for item in answer.evidence:
        assert item["effect_status"] != "het_hieu_luc", item["citation"]


def test_out_of_corpus_question_is_refused(service):
    answer = service.ask("Điều kiện chào bán chứng khoán lần đầu ra công chúng theo "
                         "Luật Chứng khoán là gì?")
    assert answer.status == "refused"
    assert "không đủ căn cứ" in answer.answer.lower()
