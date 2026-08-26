from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ..domain.citation import Citation
from ..logging_config import get_logger

logger = get_logger(__name__)


class Layer(StrEnum):
    CORPUS = "a_corpus"
    PARSE = "b_parse"
    RETRIEVAL = "c_retrieval"
    GENERATION = "d_generation"
    NONE = "none"


class Verdict(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"


@dataclass
class LayerResult:
    layer: Layer
    verdict: Verdict
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiagnosisProbe:
    question: str
    doc_number: str = ""
    doc_keywords: tuple[str, ...] = ()
    expected_dieu: str | None = None
    answer_terms: tuple[str, ...] = ()
    label: str = ""


@dataclass
class Diagnosis:
    probe: DiagnosisProbe
    layers: list[LayerResult] = field(default_factory=list)
    answer_status: str = ""
    answer_text: str = ""

    @property
    def failing_layer(self) -> Layer:
        for result in self.layers:
            if result.verdict is Verdict.FAIL:
                return result.layer
        return Layer.NONE

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.probe.question,
            "label": self.probe.label,
            "failing_layer": self.failing_layer.value,
            "answer_status": self.answer_status,
            "layers": [
                {"layer": result.layer.value, "verdict": result.verdict.value,
                 "detail": result.detail, **result.evidence}
                for result in self.layers
            ],
        }


class FailureDiagnoser:
    def __init__(self, service, top_k: int = 5) -> None:
        self.service = service
        self.top_k = top_k

    def diagnose(self, probe: DiagnosisProbe) -> Diagnosis:
        diagnosis = Diagnosis(probe=probe)

        corpus = self._check_corpus(probe)
        diagnosis.layers.append(corpus)
        if corpus.verdict is Verdict.FAIL:
            diagnosis.layers.extend(self._blocked_from(Layer.PARSE, corpus))
            self._run_answer(diagnosis)
            return diagnosis

        parse = self._check_parse(probe, corpus.evidence["doc_numbers"])
        diagnosis.layers.append(parse)
        if parse.verdict is Verdict.FAIL:
            diagnosis.layers.extend(self._blocked_from(Layer.RETRIEVAL, parse))
            self._run_answer(diagnosis)
            return diagnosis

        retrieval = self._check_retrieval(probe, parse.evidence["target_chunk_ids"])
        diagnosis.layers.append(retrieval)
        if retrieval.verdict is Verdict.FAIL:
            diagnosis.layers.extend(self._blocked_from(Layer.GENERATION, retrieval))
            self._run_answer(diagnosis)
            return diagnosis

        diagnosis.layers.append(self._check_generation(probe, diagnosis))
        return diagnosis

    def _check_corpus(self, probe: DiagnosisProbe) -> LayerResult:
        documents = self.service.graph_store.all_documents()
        if probe.doc_number:
            matched = [document for document in documents
                       if document.doc_number == probe.doc_number]
            wanted = [probe.doc_number]
        else:
            matched = [
                document for document in documents
                if any(keyword.lower() in f"{document.title} {document.doc_number}".lower()
                       for keyword in probe.doc_keywords)
            ]
            wanted = list(probe.doc_keywords)
        catalogue = [f"{document.doc_number} - {document.title}" for document in documents]
        if matched:
            return LayerResult(
                Layer.CORPUS, Verdict.PASS,
                f"Tìm thấy {len(matched)} văn bản khớp {wanted}",
                {"doc_numbers": [document.doc_number or document.title
                                 for document in matched],
                 "matched": [document.title for document in matched]},
            )
        return LayerResult(
            Layer.CORPUS, Verdict.FAIL,
            f"Không có văn bản nào khớp {wanted} trong KB "
            f"({len(documents)} văn bản).",
            {"doc_numbers": [], "corpus": catalogue},
        )

    def _check_parse(self, probe: DiagnosisProbe, doc_numbers: list[str]) -> LayerResult:
        store = self.service.vector_store
        candidates = []
        for doc_number in doc_numbers:
            candidates.extend(store.fetch_by_citation(doc_number, probe.expected_dieu,
                                                      limit=50))
        if not candidates:
            return LayerResult(
                Layer.PARSE, Verdict.FAIL,
                f"Không tách được Điều {probe.expected_dieu} trong {doc_numbers} "
                "- regex parser có thể đã gộp/bỏ sót điều luật.",
                {"target_chunk_ids": []},
            )
        if probe.answer_terms:
            grounded = [chunk for chunk in candidates
                        if all(term.lower() in chunk.text.lower()
                               for term in probe.answer_terms)]
            if not grounded:
                return LayerResult(
                    Layer.PARSE, Verdict.FAIL,
                    f"Điều {probe.expected_dieu} có tồn tại nhưng nội dung không chứa "
                    f"{list(probe.answer_terms)} - nhiều khả năng chunk bị cắt sai.",
                    {"target_chunk_ids": [],
                     "found": [chunk.citation.render() for chunk in candidates][:5]},
                )
            candidates = grounded
        return LayerResult(
            Layer.PARSE, Verdict.PASS,
            f"Tách đúng {len(candidates)} chunk cho Điều {probe.expected_dieu}",
            {"target_chunk_ids": [chunk.chunk_id for chunk in candidates],
             "citations": [chunk.citation.render() for chunk in candidates][:5]},
        )

    def _check_retrieval(self, probe: DiagnosisProbe,
                         target_chunk_ids: list[str]) -> LayerResult:
        hits = self.service.retriever.retrieve(probe.question, top_n=self.top_k)
        ranked = [
            {
                "rank": rank,
                "citation": hit.chunk.citation.render(),
                "dense": _round(hit.dense_score),
                "sparse": _round(hit.sparse_score),
                "fusion": round(hit.fusion_score, 5),
                "rerank": _round(hit.rerank_score),
                "source": hit.source,
            }
            for rank, hit in enumerate(hits, start=1)
        ]
        found = [hit.chunk_id for hit in hits if hit.chunk_id in set(target_chunk_ids)]
        if found:
            return LayerResult(Layer.RETRIEVAL, Verdict.PASS,
                               f"Chunk đúng nằm trong top-{self.top_k}",
                               {"top_k": ranked, "hit_ids": found})
        return LayerResult(
            Layer.RETRIEVAL, Verdict.FAIL,
            f"Chunk đúng KHÔNG lọt top-{self.top_k} dù đã có trong index.",
            {"top_k": ranked, "expected_ids": target_chunk_ids[:5]},
        )


    def _check_generation(self, probe: DiagnosisProbe, diagnosis: Diagnosis) -> LayerResult:
        self._run_answer(diagnosis)
        answer = diagnosis.answer_text
        if diagnosis.answer_status == "refused":
            return LayerResult(Layer.GENERATION, Verdict.FAIL,
                               "Bằng chứng đúng có trong context nhưng hệ thống vẫn từ chối.",
                               {"answer": answer[:400]})
        cited = [citation.render() for citation in Citation.parse_cited(answer)]
        expected_hit = any(
            probe.expected_dieu and f"Điều {probe.expected_dieu}" in citation
            for citation in cited
        )
        missing = [term for term in probe.answer_terms if term.lower() not in answer.lower()]
        if expected_hit and not missing:
            return LayerResult(Layer.GENERATION, Verdict.PASS,
                               "Câu trả lời dùng đúng điều luật và trích dẫn đúng.",
                               {"citations": cited})
        return LayerResult(
            Layer.GENERATION, Verdict.FAIL,
            "Bằng chứng đúng có trong context nhưng câu trả lời không dùng/trích sai.",
            {"citations": cited, "answer": answer[:400], "missing_terms": missing},
        )

    def _run_answer(self, diagnosis: Diagnosis) -> None:
        if diagnosis.answer_text:
            return
        answer = self.service.ask(diagnosis.probe.question, session_id="diagnose")
        diagnosis.answer_status = answer.status
        diagnosis.answer_text = answer.answer

    @staticmethod
    def _blocked_from(first: Layer, cause: LayerResult) -> list[LayerResult]:
        order = [Layer.CORPUS, Layer.PARSE, Layer.RETRIEVAL, Layer.GENERATION]
        reason = f"bị chặn bởi lỗi tầng {cause.layer.value}"
        return [LayerResult(layer, Verdict.BLOCKED, reason)
                for layer in order[order.index(first):]]


def _round(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None

CANARY_PROBES = (
    DiagnosisProbe(
        question="Theo Hiến pháp 2013, cơ quan nào là cơ quan quyền lực nhà nước cao nhất "
                 "của nước Cộng hòa xã hội chủ nghĩa Việt Nam?",
        doc_keywords=("hiến pháp",),
        expected_dieu="69",
        answer_terms=("quốc hội",),
        label="canary-hien-phap-2013",
    ),
    DiagnosisProbe(
        question="Người từ đủ bao nhiêu tuổi trở lên phải chịu trách nhiệm hình sự về mọi "
                 "tội phạm theo Bộ luật Hình sự 2015?",
        doc_keywords=("hình sự",),
        expected_dieu="12",
        answer_terms=("16 tuổi",),
        label="canary-blhs-2015",
    ),
)


def probe_from_case(case, answer_terms: tuple[str, ...] = ()) -> DiagnosisProbe:
    expected = case.parsed_expected
    if not expected:
        return DiagnosisProbe(question=case.question, label=case.case_id)
    citation = expected[0]
    keyword = _identity_keyword(citation.doc_title or citation.doc_number)
    return DiagnosisProbe(
        question=case.question,
        doc_number=citation.doc_number,
        doc_keywords=(keyword,) if keyword else (),
        expected_dieu=citation.dieu,
        answer_terms=answer_terms,
        label=case.case_id,
    )


def _identity_keyword(identity: str) -> str:
    cleaned = re.sub(r"\s*(?:số\s*)?\d{1,4}/\d{4}/[A-ZĐ][A-ZĐ0-9\-/]*", "", identity)
    cleaned = re.sub(r"^(?:Bộ luật|Luật|Nghị định|Thông tư|Pháp lệnh|Nghị quyết)\s+",
                     "", cleaned.strip(), flags=re.IGNORECASE)
    return cleaned.strip(" ,.;")
