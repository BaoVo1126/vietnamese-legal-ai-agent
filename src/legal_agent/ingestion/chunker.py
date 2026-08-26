from __future__ import annotations
from ..config import Settings, get_settings
from ..domain.chunk import LegalChunk
from ..domain.document import LegalDocumentMeta
from ..domain.enums import NodeLevel
from ..domain.node import LegalNode
from ..logging_config import get_logger
from .parser import ParsedDocument

logger = get_logger(__name__)


class LegalChunkBuilder:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def build(self, parsed: ParsedDocument) -> list[LegalChunk]:
        chunks: list[LegalChunk] = []
        for chuong, muc, dieu in self._iter_dieu_with_context(parsed.root):
            chunks.extend(self._chunks_for_dieu(parsed.meta, chuong, muc, dieu))
        chunks = self._deduplicate_ids(chunks)
        logger.info("%s -> %d chunks", parsed.meta.doc_number, len(chunks))
        return chunks

    @staticmethod
    def _deduplicate_ids(chunks: list[LegalChunk]) -> list[LegalChunk]:
        seen: dict[str, int] = {}
        for index, chunk in enumerate(chunks):
            count = seen.get(chunk.chunk_id, 0)
            seen[chunk.chunk_id] = count + 1
            if count:
                logger.warning("Trùng chunk_id %s - thêm hậu tố.", chunk.chunk_id)
                chunks[index] = chunk.model_copy(
                    update={"chunk_id": f"{chunk.chunk_id}#{count + 1}"})
        return chunks

    @staticmethod
    def _iter_dieu_with_context(root: LegalNode):
        stack: list[tuple[LegalNode | None, LegalNode | None, LegalNode]] = []

        def walk(node: LegalNode, chuong: LegalNode | None, muc: LegalNode | None) -> None:
            for child in node.children:
                if child.level is NodeLevel.CHUONG:
                    walk(child, child, None)
                elif child.level in {NodeLevel.MUC, NodeLevel.TIEU_MUC}:
                    walk(child, chuong, child)
                elif child.level is NodeLevel.DIEU:
                    stack.append((chuong, muc, child))
                else:
                    walk(child, chuong, muc)

        walk(root, None, None)
        return stack

    def _chunks_for_dieu(self, meta: LegalDocumentMeta, chuong: LegalNode | None,
                         muc: LegalNode | None, dieu: LegalNode) -> list[LegalChunk]:
        khoan_nodes = [child for child in dieu.children if child.level is NodeLevel.KHOAN]
        header = self._context_header(meta, chuong, muc, dieu)
        node_path = self._node_path(meta, chuong, muc, dieu)

        if not khoan_nodes:
            text = self._dieu_text(dieu)
            if not text:
                return []
            return [self._make_chunk(meta, chuong, muc, dieu, khoan=None, text=text,
                                     header=header, node_path=node_path, suffix="")]

        chunks: list[LegalChunk] = []
        for khoan in khoan_nodes:
            chunks.extend(
                self._chunks_for_khoan(meta, chuong, muc, dieu, khoan, header, node_path)
            )
        return chunks

    def _chunks_for_khoan(self, meta, chuong, muc, dieu, khoan, header, node_path):
        full_text = khoan.full_text().strip()
        if not full_text:
            return []
        khoan_path = f"{node_path} > Khoản {khoan.number}"
        if len(full_text) <= self.settings.max_chunk_chars:
            return [self._make_chunk(meta, chuong, muc, dieu, khoan=khoan.number,
                                     text=full_text, header=header,
                                     node_path=khoan_path, suffix=f".k{khoan.number}")]
        return self._split_khoan_by_diem(meta, chuong, muc, dieu, khoan, header, khoan_path)

    def _split_khoan_by_diem(self, meta, chuong, muc, dieu, khoan, header, khoan_path):
        chapeau = khoan.own_text_with_label().strip()
        diem_nodes = [child for child in khoan.children if child.level is NodeLevel.DIEM]
        if not diem_nodes:
            return [self._make_chunk(meta, chuong, muc, dieu, khoan=khoan.number,
                                     text=khoan.full_text().strip(), header=header,
                                     node_path=khoan_path, suffix=f".k{khoan.number}")]

        chunks: list[LegalChunk] = []
        group: list[LegalNode] = []
        budget = max(self.settings.max_chunk_chars - len(chapeau), 400)

        def flush(part_index: int) -> None:
            if not group:
                return
            body = "\n".join(node.own_text_with_label() for node in group)
            diem_label = group[0].number if len(group) == 1 else None
            chunks.append(self._make_chunk(
                meta, chuong, muc, dieu, khoan=khoan.number,
                text=f"{chapeau}\n{body}".strip(), header=header,
                node_path=f"{khoan_path} > Điểm {group[0].number}-{group[-1].number}",
                suffix=f".k{khoan.number}.p{part_index}", diem=diem_label,
            ))

        part = 1
        size = 0
        for node in diem_nodes:
            node_size = len(node.own_text_with_label())
            if group and size + node_size > budget:
                flush(part)
                part += 1
                group, size = [], 0
            group.append(node)
            size += node_size
        flush(part)
        return chunks

    @staticmethod
    def _dieu_text(dieu: LegalNode) -> str:
        return dieu.full_text().strip()

    @staticmethod
    def _context_header(meta: LegalDocumentMeta, chuong: LegalNode | None,
                        muc: LegalNode | None, dieu: LegalNode) -> str:
        parts = [meta.display_name]
        if chuong is not None:
            parts.append(chuong.heading)
        if muc is not None:
            parts.append(muc.heading)
        parts.append(dieu.heading)
        chapeau = dieu.text.strip()
        header = " > ".join(part for part in parts if part)
        return f"{header}\n{chapeau}".strip() if chapeau else header

    @staticmethod
    def _node_path(meta: LegalDocumentMeta, chuong: LegalNode | None,
                   muc: LegalNode | None, dieu: LegalNode) -> str:
        parts = [meta.doc_number or meta.doc_id]
        if chuong is not None:
            parts.append(chuong.label)
        if muc is not None:
            parts.append(muc.label)
        parts.append(dieu.label)
        return " > ".join(parts)

    @staticmethod
    def _make_chunk(meta: LegalDocumentMeta, chuong: LegalNode | None,
                    muc: LegalNode | None, dieu: LegalNode, khoan: str | None,
                    text: str, header: str, node_path: str, suffix: str,
                    diem: str | None = None) -> LegalChunk:
        return LegalChunk(
            chunk_id=f"{meta.doc_id}::d{dieu.number}{suffix}",
            doc_id=meta.doc_id,
            doc_number=meta.doc_number,
            doc_title=meta.title,
            doc_type=meta.doc_type,
            chuong=chuong.heading if chuong is not None else "",
            muc=muc.heading if muc is not None else "",
            dieu=dieu.number,
            dieu_title=dieu.title,
            khoan=khoan,
            diem=diem,
            node_path=node_path,
            text=text,
            context_header=header,
            effect_status=meta.effect_status,
            effective_date=meta.effective_date,
            expiry_date=meta.expiry_date,
            issuing_body=meta.issuing_body,
            field_of_law=meta.field_of_law,
        )
