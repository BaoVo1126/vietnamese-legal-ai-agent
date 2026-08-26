from __future__ import annotations
from datetime import date
from legal_agent.domain.enums import EffectStatus, RelationType
from legal_agent.kg.builder import KnowledgeGraphBuilder
from legal_agent.kg.memory_store import MemoryGraphStore
from legal_agent.retrieval.hybrid import HybridRetriever

from .conftest import make_chunk


class TestKnowledgeGraph:
    def _store(self, parsed_law, parsed_decree) -> MemoryGraphStore:
        return KnowledgeGraphBuilder(MemoryGraphStore()).build(
            [parsed_law.meta, parsed_decree.meta])

    def test_repeal_is_propagated_to_the_replaced_document(self, parsed_law, parsed_decree):
        old_law = parsed_law.meta.model_copy(update={
            "doc_id": "68-2014-QH13", "doc_number": "68/2014/QH13", "relations": [],
            "effect_status": EffectStatus.CON_HIEU_LUC, "expiry_date": None,
        })
        store = KnowledgeGraphBuilder(MemoryGraphStore()).build(
            [parsed_law.meta, parsed_decree.meta, old_law])
        verdict = store.validate("68/2014/QH13")
        assert verdict.status is EffectStatus.HET_HIEU_LUC
        assert verdict.is_citable is False
        assert "59/2020/QH14" in verdict.replaced_by

    def test_multi_hop_guidance_lookup_is_article_scoped(self, parsed_law, parsed_decree):
        store = self._store(parsed_law, parsed_decree)
        assert store.validate("59/2020/QH14", dieu="26").guided_by == [
            ("01/2021/NĐ-CP", "26")]
        assert store.validate("59/2020/QH14", dieu="7").guided_by == []

    def test_inverse_edges_are_created(self, parsed_law, parsed_decree):
        store = self._store(parsed_law, parsed_decree)
        assert store.neighbours("59/2020/QH14", RelationType.DUOC_HUONG_DAN_BOI)

    def test_status_is_evaluated_against_the_reference_date(self, parsed_law, parsed_decree):
        store = self._store(parsed_law, parsed_decree)
        assert store.validate("59/2020/QH14", as_of=date(2020, 12, 31)).status is \
            EffectStatus.CHUA_CO_HIEU_LUC
        assert store.validate("59/2020/QH14", as_of=date(2021, 6, 1)).status is \
            EffectStatus.CON_HIEU_LUC

    def test_unknown_document_is_not_silently_trusted(self):
        verdict = MemoryGraphStore().validate("99/2099/QH99")
        assert verdict.status is EffectStatus.KHONG_XAC_DINH
        assert verdict.is_citable is False

    def test_snapshot_round_trip(self, tmp_path, parsed_law, parsed_decree):
        store = self._store(parsed_law, parsed_decree)
        path = tmp_path / "graph.json"
        store.save(path)
        restored = MemoryGraphStore()
        assert restored.load(path) is True
        assert restored.validate("59/2020/QH14", dieu="26").guided_by == [
            ("01/2021/NĐ-CP", "26")]


class TestReciprocalRankFusion:
    def test_document_found_by_both_legs_outranks_single_leg_hits(self):
        shared = make_chunk(chunk_id="shared")
        dense_only = make_chunk(chunk_id="dense-only")
        sparse_only = make_chunk(chunk_id="sparse-only")

        fused = HybridRetriever.fuse(
            dense_hits=[(dense_only, 0.99), (shared, 0.50)],
            sparse_hits=[(shared, 12.0), (sparse_only, 3.0)],
            k=60,
        )
        assert fused[0].chunk_id == "shared"
        assert fused[0].source == "hybrid"
        assert fused[0].fusion_score == 1 / 62 + 1 / 61

    def test_scores_are_rank_based_not_score_based(self):
        first = make_chunk(chunk_id="a")
        second = make_chunk(chunk_id="b")
        fused = HybridRetriever.fuse(dense_hits=[(first, 1000.0), (second, 0.001)],
                                     sparse_hits=[], k=60)
        assert [item.chunk_id for item in fused] == ["a", "b"]
        assert fused[0].fusion_score == 1 / 61


class TestAmendingLawSemantics:
    AMENDING_TEXT = (
        "Sửa đổi, bổ sung một số điều của Bộ luật Hình sự số 100/2015/QH13.\n"
        "Thay thế cụm từ tại khoản 2 Điều 12 của Bộ luật Hình sự số 100/2015/QH13.\n"
        "Một số điều của Bộ luật Hình sự số 100/2015/QH13 hết hiệu lực."
    )

    def _amending_meta(self):
        from legal_agent.domain.document import LegalDocumentMeta

        return LegalDocumentMeta(
            doc_id="12-2017-QH14", doc_number="12/2017/QH14",
            title="Luật Sửa đổi, bổ sung một số điều của Bộ luật Hình sự",
            effect_status=EffectStatus.CON_HIEU_LUC)

    def test_amending_law_emits_sua_doi_not_thay_the(self):
        from legal_agent.ingestion.relation_extractor import RelationExtractor

        relations = RelationExtractor().extract(self._amending_meta(), self.AMENDING_TEXT)
        kinds = {relation.relation for relation in relations}
        assert RelationType.SUA_DOI in kinds
        assert RelationType.THAY_THE not in kinds

    def test_word_replacement_is_not_document_replacement(self):
        from legal_agent.domain.document import LegalDocumentMeta
        from legal_agent.ingestion.relation_extractor import RelationExtractor

        ordinary = LegalDocumentMeta(doc_id="x", doc_number="10/2020/NĐ-CP",
                                     title="Nghị định về một việc")
        text = "Thay thế cụm từ “cơ quan” bằng “đơn vị” tại Nghị định số 99/2015/NĐ-CP."
        kinds = {relation.relation
                 for relation in RelationExtractor().extract(ordinary, text)}
        assert RelationType.THAY_THE not in kinds

    def test_partial_repeal_is_never_escalated_to_full_repeal(self, parsed_law):
        code = parsed_law.meta.model_copy(update={
            "doc_id": "100-2015-QH13", "doc_number": "100/2015/QH13",
            "title": "Bộ luật Hình sự", "relations": [],
            "effect_status": EffectStatus.HET_HIEU_LUC_MOT_PHAN, "expiry_date": None,
        })
        amending = self._amending_meta().model_copy(update={
            "relations": [__import__("legal_agent.domain.document", fromlist=["x"])
                          .LegalRelation(source_doc_id="12-2017-QH14",
                                         target_doc_id="100-2015-QH13",
                                         relation=RelationType.THAY_THE)],
        })
        store = KnowledgeGraphBuilder(MemoryGraphStore()).build([amending, code])
        verdict = store.validate("100/2015/QH13")
        assert verdict.status is EffectStatus.HET_HIEU_LUC_MOT_PHAN
        assert verdict.is_citable, "bộ luật gốc phải vẫn trích dẫn được"


class TestGraphReset:

    def test_rebuild_drops_documents_that_left_the_corpus(self, parsed_law, parsed_decree):
        store = MemoryGraphStore()
        stale = parsed_law.meta.model_copy(update={
            "doc_id": "99-1999-QH10", "doc_number": "99/1999/QH10",
            "title": "Luật đã gỡ khỏi corpus", "relations": [],
            "effect_status": EffectStatus.CON_HIEU_LUC})
        KnowledgeGraphBuilder(store).build([parsed_law.meta, stale], reset=True)
        assert store.get_document("99/1999/QH10") is not None

        KnowledgeGraphBuilder(store).build([parsed_law.meta, parsed_decree.meta], reset=True)
        assert store.get_document("99/1999/QH10") is None
        assert store.validate("99/1999/QH10").status is EffectStatus.KHONG_XAC_DINH

    def test_without_reset_the_graph_accumulates(self, parsed_law):
        store = MemoryGraphStore()
        stale = parsed_law.meta.model_copy(update={
            "doc_id": "99-1999-QH10", "doc_number": "99/1999/QH10", "relations": []})
        KnowledgeGraphBuilder(store).build([stale])
        KnowledgeGraphBuilder(store).build([parsed_law.meta])
        assert store.get_document("99/1999/QH10") is not None
