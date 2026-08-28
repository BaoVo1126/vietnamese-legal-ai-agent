from __future__ import annotations

from datetime import date
from typing import ClassVar

import pytest

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


class TestDuplicateRecordMerge:
    PREFERRED: ClassVar[dict] = {"label": "Luật Doanh nghiệp 2020", "id": "142847",
                 "title": "Luật Doanh nghiệp", "so_ky_hieu": "59/2020/QH14",
                 "tinh_trang_hieu_luc": "Còn hiệu lực", "field_of_law": "Doanh nghiệp"}
    WITH_CONTENT: ClassVar[dict] = {"label": "Luật Doanh nghiệp 2020", "id": "142881",
                    "title": "doanh nghiệp", "so_ky_hieu": "59/2020/QH14",
                    "tinh_trang_hieu_luc": "Hết hiệu lực một phần",
                    "field_of_law": "Doanh nghiệp"}

    def test_title_comes_from_the_better_record(self):
        from legal_agent.ingestion.hf_corpus import merge_metadata

        merged = merge_metadata(self.PREFERRED, self.WITH_CONTENT)
        assert merged["title"] == "Luật Doanh nghiệp"
        assert merged["source_id"] == "142881"

    def test_effect_status_takes_the_more_conservative_record(self):
        from legal_agent.ingestion.hf_corpus import merge_metadata

        merged = merge_metadata(self.PREFERRED, self.WITH_CONTENT)
        assert merged["tinh_trang_hieu_luc"] == "Hết hiệu lực một phần"

    def test_front_matter_carries_the_merged_values(self):
        from legal_agent.ingestion.hf_corpus import front_matter, merge_metadata

        rendered = front_matter(merge_metadata(self.PREFERRED, self.WITH_CONTENT))
        assert "title: Luật Doanh nghiệp" in rendered
        assert "effect_status: het_hieu_luc_mot_phan" in rendered


class TestLocalDocumentIngestion:
    HTML = (
        "<html><body>"
        "<p>QUỐC HỘI</p><p>Luật số: 31/2024/QH15</p>"
        "<p>LUẬT</p><p>ĐẤT ĐAI</p>"
        "<p>Chương I</p><p>QUY ĐỊNH CHUNG</p>"
        "<p>Điều 1. Phạm vi điều chỉnh</p>"
        "<p>Luật này quy định về chế độ sở hữu đất đai, quyền hạn và trách nhiệm của "
        "Nhà nước đại diện chủ sở hữu toàn dân về đất đai và thống nhất quản lý về đất "
        "đai, chế độ quản lý và sử dụng đất đai, quyền và nghĩa vụ của công dân, người "
        "sử dụng đất đối với đất đai thuộc lãnh thổ của nước Cộng hòa xã hội chủ nghĩa "
        "Việt Nam.</p>"
        "<p>Điều 2. Đối tượng áp dụng</p>"
        "<p>1. Cơ quan nhà nước thực hiện quyền hạn và trách nhiệm đại diện chủ sở hữu "
        "toàn dân về đất đai, thực hiện nhiệm vụ thống nhất quản lý nhà nước về đất đai.</p>"
        "<p>2. Người sử dụng đất theo quy định của Luật này.</p>"
        "</body></html>"
    )

    def _spec(self, **overrides):
        from legal_agent.ingestion.local_loader import LocalDocumentSpec

        values = dict(label="Luật Đất đai 2024", title="Luật Đất đai",
                      doc_number="31/2024/QH15", effect_status="Còn hiệu lực",
                      effective_date="01/08/2024", field_of_law="Đất đai")
        values.update(overrides)
        return LocalDocumentSpec(**values)

    def test_html_document_is_ingested_and_parsed(self, tmp_path):
        from legal_agent.ingestion.local_loader import ingest_local_document
        from legal_agent.ingestion.parser import StructureAwareParser

        source = tmp_path / "dat-dai.html"
        source.write_text(self.HTML, encoding="utf-8")
        target = ingest_local_document(source, self._spec(), tmp_path / "raw")

        parsed = StructureAwareParser().parse(target.read_text(encoding="utf-8"))
        assert parsed.meta.doc_number == "31/2024/QH15"
        assert parsed.meta.title == "Luật Đất đai"
        assert parsed.meta.effect_status is EffectStatus.CON_HIEU_LUC
        assert [node.number for node in parsed.iter_dieu()] == ["1", "2"]

    def test_invalid_effect_status_is_rejected(self, tmp_path):
        from legal_agent.ingestion.local_loader import ingest_local_document

        source = tmp_path / "x.html"
        source.write_text(self.HTML, encoding="utf-8")
        with pytest.raises(ValueError, match="Trạng thái hiệu lực"):
            ingest_local_document(source, self._spec(effect_status="chắc là còn"),
                                  tmp_path / "raw")

    def test_document_without_a_title_is_rejected(self, tmp_path):
        from legal_agent.ingestion.local_loader import ingest_local_document

        source = tmp_path / "x.html"
        source.write_text(self.HTML, encoding="utf-8")
        with pytest.raises(ValueError, match="tiêu đề"):
            ingest_local_document(source, self._spec(title="  "), tmp_path / "raw")

    def test_too_short_document_is_rejected(self, tmp_path):
        from legal_agent.ingestion.local_loader import ingest_local_document

        source = tmp_path / "x.html"
        source.write_text("<html><body><p>Ngắn quá.</p></body></html>", encoding="utf-8")
        with pytest.raises(ValueError, match="quá ngắn"):
            ingest_local_document(source, self._spec(), tmp_path / "raw")

    def test_unsupported_format_is_rejected(self, tmp_path):
        from legal_agent.ingestion.local_loader import extract_text

        source = tmp_path / "x.docx"
        source.write_bytes(b"binary")
        with pytest.raises(ValueError, match="Chưa hỗ trợ"):
            extract_text(source)
