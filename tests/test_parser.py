from __future__ import annotations
from datetime import date
from legal_agent.domain.enums import DocumentType, EffectStatus, NodeLevel, RelationType
from legal_agent.ingestion.parser import StructureAwareParser
from legal_agent.ingestion.patterns import looks_like_dieu_heading, normalise_text


class TestStructure:
    def test_builds_full_hierarchy(self, parsed_law):
        assert parsed_law.stats["chuong"] == 2
        assert [node.number for node in parsed_law.iter_dieu()] == ["7", "17", "217"]

    def test_khoan_and_diem_nest_under_their_dieu(self, parsed_law):
        dieu_17 = next(node for node in parsed_law.iter_dieu() if node.number == "17")
        khoan_numbers = [child.number for child in dieu_17.children]
        assert khoan_numbers == ["1", "2"]
        khoan_2 = dieu_17.children[1]
        assert [child.number for child in khoan_2.children] == ["a", "b"]
        assert all(child.level is NodeLevel.DIEM for child in khoan_2.children)

    def test_dieu_title_is_captured(self, parsed_law):
        dieu_7 = parsed_law.iter_dieu()[0]
        assert dieu_7.title == "Quyền của doanh nghiệp"
        assert dieu_7.heading == "Điều 7. Quyền của doanh nghiệp"

    def test_chuong_title_on_next_line_is_stitched(self, parsed_law):
        chuong = parsed_law.root.find_all(NodeLevel.CHUONG)[0]
        assert chuong.title == "NHỮNG QUY ĐỊNH CHUNG"

    def test_cross_reference_is_not_treated_as_heading(self):
        assert looks_like_dieu_heading("12", ".", "Quyền của doanh nghiệp") is True
        assert looks_like_dieu_heading("12", None, "của Luật này được sửa đổi") is False
        assert looks_like_dieu_heading("218", ".", "Quy định chuyển tiếp") is True

    def test_normalise_text_collapses_pdf_artefacts(self):
        raw = "Điều\u00a01.  Phạm vi\r\n\r\n\r\n  điều chỉnh  "
        assert normalise_text(raw) == "Điều 1. Phạm vi\n\nđiều chỉnh"


class TestMetadata:
    def test_core_fields(self, parsed_law):
        meta = parsed_law.meta
        assert meta.doc_number == "59/2020/QH14"
        assert meta.doc_id == "59-2020-QH14"
        assert meta.doc_type is DocumentType.LUAT
        assert meta.title == "Luật Doanh nghiệp"
        assert meta.issuing_body == "Quốc hội"
        assert meta.issued_date == date(2020, 6, 17)
        assert meta.effective_date == date(2021, 1, 1)
        assert meta.effect_status is EffectStatus.CON_HIEU_LUC

    def test_effective_date_only_from_self_referential_sentence(self, parsed_law):
        assert parsed_law.meta.expiry_date is None

    def test_front_matter_overrides_heuristics(self):
        text = ("---\n"
                "effect_status: het_hieu_luc\n"
                "field_of_law: Doanh nghiệp\n"
                "---\n"
                "Luật số: 68/2014/QH13\n\nLUẬT\nDOANH NGHIỆP\n\n"
                "Điều 1. Phạm vi\nNội dung.\n")
        meta = StructureAwareParser().parse(text).meta
        assert meta.effect_status is EffectStatus.HET_HIEU_LUC
        assert meta.field_of_law == "Doanh nghiệp"


class TestRelations:
    def test_replacement_relation_is_mined(self, parsed_law):
        relations = {(rel.relation, rel.target_doc_id) for rel in parsed_law.meta.relations}
        assert (RelationType.THAY_THE, "68-2014-QH13") in relations

    def test_article_level_guidance_relation(self, parsed_decree):
        guidance = [rel for rel in parsed_decree.meta.relations
                    if rel.relation is RelationType.HUONG_DAN]
        assert len(guidance) == 1
        assert guidance[0].target_doc_id == "59-2020-QH14"
        assert guidance[0].target_dieu == "26"
