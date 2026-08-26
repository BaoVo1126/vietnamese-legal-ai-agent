from __future__ import annotations
from legal_agent.agents.service import AgentAnswer
from legal_agent.monitoring.metrics import percentile, summarise
from legal_agent.monitoring.run_logger import RunRecorder


def make_answer(status: str = "answered", **kwargs) -> AgentAnswer:
    defaults = dict(
        question="Câu hỏi thử nghiệm?", answer="Trả lời.", status=status,
        intent="hoi_dap_khai_niem", grounding_score=0.9, support_ratio=1.0, attempts=1,
        evidence=[{"citation": "Điều 7, Khoản 1, Luật Doanh nghiệp 59/2020/QH14"}],
        citations=[{"doc_number": "59/2020/QH14", "dieu": "7", "khoan": "1"}],
        trace=[{"node": "router", "elapsed_ms": 2.0},
               {"node": "retrieve", "elapsed_ms": 30.0},
               {"node": "retrieve", "elapsed_ms": 20.0}],
    )
    defaults.update(kwargs)
    return AgentAnswer(**defaults)


class TestRunRecorder:
    def test_record_round_trip(self, tmp_path):
        recorder = RunRecorder(tmp_path / "runs.jsonl")
        recorder.record(make_answer(), latency_ms=123.456, session_id="s1")
        records = recorder.read_all()
        assert len(records) == 1
        record = records[0]
        assert record["status"] == "answered"
        assert record["latency_ms"] == 123.5
        assert record["session_id"] == "s1"
        assert record["citations"] == ["59/2020/QH14"]

    def test_node_latency_accumulates_across_retries(self, tmp_path):
        recorder = RunRecorder(tmp_path / "runs.jsonl")
        record = recorder.record(make_answer(), latency_ms=10.0)
        assert record["node_latency_ms"] == {"router": 2.0, "retrieve": 50.0}

    def test_retry_flag_follows_attempts(self, tmp_path):
        recorder = RunRecorder(tmp_path / "runs.jsonl")
        assert recorder.record(make_answer(attempts=1), latency_ms=1.0)["retried"] is False
        assert recorder.record(make_answer(attempts=2), latency_ms=1.0)["retried"] is True

    def test_reading_a_missing_log_is_not_an_error(self, tmp_path):
        assert RunRecorder(tmp_path / "absent.jsonl").read_all() == []

    def test_disabled_recorder_keeps_returning_the_record(self, tmp_path):
        path = tmp_path / "runs.jsonl"
        recorder = RunRecorder(path, enabled=False)
        assert recorder.record(make_answer(), latency_ms=1.0)["status"] == "answered"
        assert not path.exists()


class TestSummarise:
    def _records(self, tmp_path):
        recorder = RunRecorder(tmp_path / "runs.jsonl")
        recorder.record(make_answer(), latency_ms=100.0)
        recorder.record(make_answer(status="refused", grounding_score=0.2,
                                    support_ratio=0.0, attempts=2), latency_ms=300.0)
        recorder.record(make_answer(), latency_ms=200.0)
        return recorder.read_all()

    def test_core_rates(self, tmp_path):
        summary = summarise(self._records(tmp_path))
        assert summary.total_runs == 3
        assert summary.answered == 2 and summary.refused == 1
        assert summary.refusal_rate == round(1 / 3, 4)
        assert summary.retry_rate == round(1 / 3, 4)

    def test_latency_percentiles(self, tmp_path):
        summary = summarise(self._records(tmp_path))
        assert summary.latency_p50_ms == 200.0
        assert summary.latency_max_ms == 300.0

    def test_top_cited_documents(self, tmp_path):
        summary = summarise(self._records(tmp_path))
        assert summary.top_cited_documents == [("59/2020/QH14", 3)]

    def test_empty_log_gives_a_zeroed_summary(self):
        summary = summarise([])
        assert summary.total_runs == 0 and summary.refusal_rate == 0.0

    def test_percentile_edges(self):
        values = [1.0, 2.0, 3.0, 4.0]
        assert percentile([], 0.5) == 0.0
        assert percentile(values, 0.0) == 1.0
        assert percentile(values, 1.0) == 4.0
