from __future__ import annotations
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legal_agent.config import Settings
from legal_agent.domain.chunk import LegalChunk, RetrievedChunk
from legal_agent.domain.enums import EffectStatus

SAMPLE_LAW = """QUỐC HỘI
--------
Luật số: 59/2020/QH14
Hà Nội, ngày 17 tháng 6 năm 2020

LUẬT
DOANH NGHIỆP

Chương I
NHỮNG QUY ĐỊNH CHUNG

Điều 7. Quyền của doanh nghiệp
1. Tự do kinh doanh ngành, nghề mà luật không cấm.
2. Tự chủ kinh doanh và lựa chọn hình thức tổ chức kinh doanh.

Điều 17. Quyền thành lập doanh nghiệp
1. Tổ chức, cá nhân có quyền thành lập doanh nghiệp, trừ trường hợp quy định tại khoản 2 Điều này.
2. Các đối tượng sau đây không có quyền thành lập doanh nghiệp:
a) Cơ quan nhà nước sử dụng tài sản nhà nước để thu lợi riêng;
b) Cán bộ, công chức theo quy định của Luật Cán bộ, công chức.

Chương X
ĐIỀU KHOẢN THI HÀNH

Điều 217. Hiệu lực thi hành
1. Luật này có hiệu lực thi hành từ ngày 01 tháng 01 năm 2021.
2. Luật Doanh nghiệp số 68/2014/QH13 hết hiệu lực kể từ ngày Luật này có hiệu lực thi hành.
"""

SAMPLE_DECREE = """CHÍNH PHỦ
--------
Số: 01/2021/NĐ-CP
Hà Nội, ngày 04 tháng 01 năm 2021

NGHỊ ĐỊNH
VỀ ĐĂNG KÝ DOANH NGHIỆP

Chương I
QUY ĐỊNH CHUNG

Điều 1. Phạm vi điều chỉnh
Nghị định này quy định chi tiết Điều 26 của Luật Doanh nghiệp số 59/2020/QH14.
"""


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings(app_profile="mvp", llm_backend="stub", embedding_backend="stub",
                    reranker_backend="stub", qdrant_mode="memory", graph_backend="memory",
                    max_retrieval_attempts=2, grounding_threshold=0.6,
                    claim_support_threshold=0.6)


@pytest.fixture()
def parsed_law():
    from legal_agent.ingestion.parser import StructureAwareParser

    return StructureAwareParser().parse(SAMPLE_LAW, source_path="test://law")


@pytest.fixture()
def parsed_decree():
    from legal_agent.ingestion.parser import StructureAwareParser

    return StructureAwareParser().parse(SAMPLE_DECREE, source_path="test://decree")


def make_chunk(chunk_id: str = "c1", dieu: str = "17", khoan: str | None = "2",
               text: str = "Các đối tượng sau đây không có quyền thành lập doanh nghiệp.",
               status: EffectStatus = EffectStatus.CON_HIEU_LUC) -> LegalChunk:
    """Build a chunk without going through the parser - used by node-level tests."""
    return LegalChunk(chunk_id=chunk_id, doc_id="59-2020-QH14", doc_number="59/2020/QH14",
                      doc_title="Luật Doanh nghiệp", dieu=dieu, khoan=khoan, text=text,
                      context_header="Điều 17. Quyền thành lập doanh nghiệp",
                      effect_status=status)


def make_retrieved(**kwargs) -> RetrievedChunk:
    return RetrievedChunk(chunk=make_chunk(**kwargs), fusion_score=0.9, rerank_score=0.9)
