from __future__ import annotations
import pytest
from legal_agent.agents.service import AgentAnswer
from legal_agent.config import get_settings
from legal_agent.evaluation.dataset import EvalCase, load_golden_set
from legal_agent.evaluation.metrics import EvalReport, score_case

DIEU_7_KHOAN_1 = "Điều 7, Khoản 1, Luật Doanh nghiệp 59/2020/QH14"
DIEU_7_OLD_LAW = "Điều 7, Khoản 1, Luật Doanh nghiệp 68/2014/QH13"


def answer_with(citations: list[str], evidence: list[tuple[str, str]],
                status: str = "answered") -> AgentAnswer:
    from legal_agent.domain.citation import Citation

    return AgentAnswer(
        question="Câu hỏi?", answer="Trả lời.", status=status,
        citations=[Citation.parse_all(text)[0].model_dump() for text in citations],
        evidence=[{"citation": citation, "effect_status": effect_status, "score": 0.9,
                   "source": "hybrid", "graph_note": "", "text": "", "node_path": ""}
                  for citation, effect_status in evidence],
        grounding_score=0.9, support_ratio=1.0, attempts=1,
    )


class TestGoldenSet:
    def test_project_golden_set_loads(self):
        cases = load_golden_set(get_settings().abs_eval_dataset_path)
        assert len(cases) >= 10
        assert {case.expected_status for case in cases} == {"answered", "refused"}

    def test_expected_citations_parse_into_exactly_one_citation_each(self):
        for case in load_golden_set(get_settings().abs_eval_dataset_path):
            assert len(case.parsed_expected) == len(case.expected_citations)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_golden_set(tmp_path / "nope.jsonl")


class TestScoreCase:
    def test_perfect_run_passes(self):
        case = EvalCase(case_id="c1", question="q", expected_citations=[DIEU_7_KHOAN_1])
        result = score_case(case, answer_with([DIEU_7_KHOAN_1],
                                              [(DIEU_7_KHOAN_1, "con_hieu_luc")]))
        assert result.retrieval_recall == 1.0
        assert result.citation_recall == 1.0
        assert result.citation_precision == 1.0
        assert result.passed

    def test_retrieved_but_not_cited_fails_on_citation_recall(self):
        case = EvalCase(case_id="c2", question="q", expected_citations=[DIEU_7_KHOAN_1])
        other = "Điều 8, Khoản 1, Luật Doanh nghiệp 59/2020/QH14"
        result = score_case(case, answer_with([other], [(DIEU_7_KHOAN_1, "con_hieu_luc"),
                                                        (other, "con_hieu_luc")]))
        assert result.retrieval_recall == 1.0
        assert result.citation_recall == 0.0
        assert not result.passed

    def test_citation_outside_evidence_lowers_precision(self):
        case = EvalCase(case_id="c3", question="q", expected_citations=[DIEU_7_KHOAN_1])
        invented = "Điều 4, Khoản 1, Nghị định 139/2016/NĐ-CP"
        result = score_case(case, answer_with([DIEU_7_KHOAN_1, invented],
                                              [(DIEU_7_KHOAN_1, "con_hieu_luc")]))
        assert result.citation_precision == 0.5
        assert not result.passed

    def test_citing_repealed_text_fails_by_default(self):
        case = EvalCase(case_id="c4", question="q", expected_citations=[DIEU_7_OLD_LAW])
        result = score_case(case, answer_with([DIEU_7_OLD_LAW],
                                              [(DIEU_7_OLD_LAW, "het_hieu_luc")]))
        assert result.stale_citations
        assert not result.passed

    def test_validity_question_may_cite_repealed_text(self):
        case = EvalCase(case_id="c5", question="q", expected_citations=[DIEU_7_OLD_LAW],
                        allow_stale_citations=True)
        result = score_case(case, answer_with([DIEU_7_OLD_LAW],
                                              [(DIEU_7_OLD_LAW, "het_hieu_luc")]))
        assert result.stale_citations == []
        assert result.passed

    def test_forbidden_citation_fails_even_when_everything_else_is_right(self):
        case = EvalCase(case_id="c6", question="q", expected_citations=[DIEU_7_KHOAN_1],
                        forbidden_citations=[DIEU_7_OLD_LAW])
        result = score_case(case, answer_with([DIEU_7_KHOAN_1, DIEU_7_OLD_LAW],
                                              [(DIEU_7_KHOAN_1, "con_hieu_luc"),
                                               (DIEU_7_OLD_LAW, "con_hieu_luc")]))
        assert result.forbidden_hits == [DIEU_7_OLD_LAW]
        assert not result.passed

    def test_expected_refusal_passes_without_citations(self):
        case = EvalCase(case_id="c7", question="q", expected_status="refused")
        result = score_case(case, answer_with([], [], status="refused"))
        assert result.status_correct and result.passed

    def test_answering_when_refusal_was_expected_fails(self):
        case = EvalCase(case_id="c8", question="q", expected_status="refused")
        result = score_case(case, answer_with([DIEU_7_KHOAN_1],
                                              [(DIEU_7_KHOAN_1, "con_hieu_luc")]))
        assert not result.status_correct and not result.passed

    def test_article_level_evidence_covers_a_clause_level_expectation(self):
        case = EvalCase(case_id="c9", question="q", expected_citations=[DIEU_7_KHOAN_1])
        article = "Điều 7, Luật Doanh nghiệp 59/2020/QH14"
        result = score_case(case, answer_with([article], [(article, "con_hieu_luc")]))
        assert result.retrieval_recall == 1.0
        assert result.citation_recall == 1.0


class TestReport:
    def test_aggregate_and_failures(self):
        case = EvalCase(case_id="c1", question="q", expected_citations=[DIEU_7_KHOAN_1])
        good = score_case(case, answer_with([DIEU_7_KHOAN_1],
                                            [(DIEU_7_KHOAN_1, "con_hieu_luc")]))
        bad = score_case(case, answer_with([], [(DIEU_7_KHOAN_1, "con_hieu_luc")]))
        report = EvalReport(results=[good, bad])
        aggregate = report.aggregate()
        assert aggregate["total_cases"] == 2
        assert aggregate["pass_rate"] == 0.5
        assert [result.case_id for result in report.failures()] == ["c1"]

    def test_empty_report(self):
        assert EvalReport().aggregate() == {}
