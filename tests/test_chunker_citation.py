from __future__ import annotations

from legal_agent.domain.citation import Citation
from legal_agent.ingestion.chunker import LegalChunkBuilder


class TestChunkingPolicy:
    def test_chunk_granularity_is_the_khoan(self, parsed_law, settings):
        chunks = LegalChunkBuilder(settings).build(parsed_law)
        dieu_17 = [chunk for chunk in chunks if chunk.dieu == "17"]
        assert sorted(chunk.khoan for chunk in dieu_17) == ["1", "2"]

    def test_dieu_without_khoan_becomes_one_chunk(self, parsed_decree, settings):
        chunks = LegalChunkBuilder(settings).build(parsed_decree)
        assert len(chunks) == 1
        assert chunks[0].dieu == "1" and chunks[0].khoan is None

    def test_diem_stay_with_their_khoan(self, parsed_law, settings):
        chunks = LegalChunkBuilder(settings).build(parsed_law)
        khoan_2 = next(chunk for chunk in chunks
                       if chunk.dieu == "17" and chunk.khoan == "2")
        assert "a) Cơ quan nhà nước" in khoan_2.text
        assert "b) Cán bộ, công chức" in khoan_2.text

    def test_context_header_carries_ancestors_but_not_the_body(self, parsed_law, settings):
        chunks = LegalChunkBuilder(settings).build(parsed_law)
        chunk = next(c for c in chunks if c.dieu == "17" and c.khoan == "1")
        assert "Luật Doanh nghiệp 59/2020/QH14" in chunk.context_header
        assert "Chương I" in chunk.context_header
        assert chunk.context_header not in chunk.text
        assert chunk.embed_text.startswith(chunk.context_header)

    def test_chunk_ids_are_unique(self, parsed_law, settings):
        chunks = LegalChunkBuilder(settings).build(parsed_law)
        assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)

    def test_every_chunk_maps_to_exactly_one_citation(self, parsed_law, settings):
        for chunk in LegalChunkBuilder(settings).build(parsed_law):
            citation = chunk.citation
            assert citation.doc_number == "59/2020/QH14"
            assert citation.dieu is not None
            assert "Điều" in citation.render()


class TestCitation:
    def test_render_canonical_order(self):
        citation = Citation(doc_number="59/2020/QH14", doc_title="Luật Doanh nghiệp",
                            dieu="17", khoan="2", diem="b")
        assert citation.render() == ("Điều 17, Khoản 2, Điểm b, "
                                     "Luật Doanh nghiệp 59/2020/QH14")

    def test_parse_multiple_citations_without_bleeding(self):
        text = ("Theo Điều 12, Khoản 2, Luật Doanh nghiệp 59/2020/QH14 và "
                "Khoản 1 Điều 5 Nghị định 01/2021/NĐ-CP.")
        rendered = [citation.render() for citation in Citation.parse_all(text)]
        assert rendered == ["Điều 12, Khoản 2, Luật Doanh nghiệp 59/2020/QH14",
                            "Điều 5, Khoản 1, Nghị định 01/2021/NĐ-CP"]

    def test_covers_is_directional(self):
        article = Citation(doc_number="59/2020/QH14", dieu="12")
        clause = Citation(doc_number="59/2020/QH14", dieu="12", khoan="2")
        assert article.covers(clause) is True
        assert clause.covers(article) is False

    def test_parse_cited_ignores_numbers_quoted_inside_statutory_text(self):
        draft = ("2. Luật Doanh nghiệp số 68/2014/QH13 hết hiệu lực kể từ ngày Luật này "
                 "có hiệu lực. (Điều 217, Khoản 2, Luật Doanh nghiệp 59/2020/QH14)")
        cited = [citation.doc_number for citation in Citation.parse_cited(draft)]
        assert cited == ["59/2020/QH14"]


class TestCitationRoundTrip:
    def test_title_containing_the_document_number_does_not_double_it(self):
        from legal_agent.ingestion.metadata_extractor import sanitize_title
        citation = Citation(doc_number="100/2015/QH13",
                            doc_title=sanitize_title("Bộ luật Hình sự số 100/2015/QH13"),
                            dieu="12", khoan="1")
        rendered = citation.render()
        assert rendered.count("100/2015/QH13") == 1
        assert Citation.parse_all(rendered) == [citation]

    def test_amending_law_title_does_not_leak_the_amended_document(self):
        from legal_agent.ingestion.metadata_extractor import sanitize_title

        title = sanitize_title("Luật Sửa đổi, bổ sung một số điều của Bộ luật Hình sự "
                               "số 100/2015/QH13 số 12/2017/QH14")
        citation = Citation(doc_number="12/2017/QH14", doc_title=title, dieu="1", khoan="3")
        parsed = Citation.parse_all(citation.render())
        assert [item.doc_number for item in parsed] == ["12/2017/QH14"]

    def test_round_trip_for_a_document_without_a_number(self):
        citation = Citation(doc_title="Hiến pháp năm 2013", dieu="69")
        assert Citation.parse_all(citation.render()) == [citation]

    def test_round_trip_survives_the_parenthesised_form(self):
        citation = Citation(doc_number="91/2015/QH13", doc_title="Bộ luật Dân sự",
                            dieu="651", khoan="1", diem="a")
        assert Citation.parse_cited(f"Nội dung nào đó ({citation.render()}).") == [citation]


class TestCitationVariants:
    def _numbers(self, text: str) -> list[str]:
        return [citation.render() for citation in Citation.parse_cited(text)]

    def test_parenthesised_form(self):
        assert self._numbers("Nội dung (Điều 17, Khoản 2, Luật Doanh nghiệp "
                             "59/2020/QH14).") == [
            "Điều 17, Khoản 2, Luật Doanh nghiệp 59/2020/QH14"]

    def test_cue_theo_without_parentheses(self):
        assert self._numbers("Theo Điều 12 Bộ luật Hình sự 100/2015/QH13, người từ đủ "
                             "16 tuổi...") == ["Điều 12, Bộ luật Hình sự 100/2015/QH13"]

    def test_cue_quy_dinh_tai_with_lowercase_components(self):
        assert self._numbers("Được quy định tại khoản 2 Điều 20 Bộ luật Dân sự "
                             "91/2015/QH13.") == [
            "Điều 20, Khoản 2, Bộ luật Dân sự 91/2015/QH13"]

    def test_abbreviated_form_with_dots(self):
        """"Đ.12 K.2" - dấu chấm trong viết tắt không được cắt mất mệnh đề."""
        assert self._numbers("Căn cứ Đ.12 K.2 Bộ luật Hình sự 100/2015/QH13.") == [
            "Điều 12, Khoản 2, Bộ luật Hình sự 100/2015/QH13"]

    def test_abbreviated_form_without_dots(self):
        assert self._numbers("Theo Đ12 K2 Bộ luật Hình sự 100/2015/QH13.") == [
            "Điều 12, Khoản 2, Bộ luật Hình sự 100/2015/QH13"]

    def test_enumeration_with_va_yields_one_citation_per_article(self):
        rendered = self._numbers("Theo Điều 12 và Điều 13 Bộ luật Hình sự "
                                 "100/2015/QH13 thì...")
        assert rendered == ["Điều 12, Bộ luật Hình sự 100/2015/QH13",
                            "Điều 13, Bộ luật Hình sự 100/2015/QH13"]

    def test_enumeration_with_commas(self):
        rendered = self._numbers("Căn cứ Điều 12, 13 và 14 Bộ luật Hình sự "
                                 "100/2015/QH13.")
        assert [item.split(",")[0] for item in rendered] == ["Điều 12", "Điều 13",
                                                             "Điều 14"]

    def test_range_is_expanded(self):
        rendered = self._numbers("Theo Điều 12 đến Điều 15 Bộ luật Hình sự "
                                 "100/2015/QH13.")
        assert [item.split(",")[0] for item in rendered] == ["Điều 12", "Điều 13",
                                                             "Điều 14", "Điều 15"]

    def test_absurdly_wide_range_keeps_only_the_endpoints(self):
        """Khoảng quá rộng chỉ giữ hai đầu mút, tránh sinh hàng trăm trích dẫn."""
        rendered = self._numbers("Theo Điều 1 đến Điều 200 Bộ luật Dân sự 91/2015/QH13.")
        assert [item.split(",")[0] for item in rendered] == ["Điều 1", "Điều 200"]

    def test_document_identified_only_by_title(self):
        assert self._numbers("Theo Điều 69 Hiến pháp năm 2013, Quốc hội là...") == [
            "Điều 69, Hiến pháp năm 2013"]

    def test_number_quoted_inside_statutory_text_is_not_a_citation(self):
        assert self._numbers("Nội dung: Luật này thay thế Luật Doanh nghiệp số "
                             "68/2014/QH13.") == []

    def test_answer_without_any_citation(self):
        assert self._numbers("Doanh nghiệp được tự do kinh doanh.") == []
