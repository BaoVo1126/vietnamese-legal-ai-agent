from __future__ import annotations

from dataclasses import dataclass, field

from ..domain.document import LegalDocumentMeta
from ..domain.enums import NODE_LEVEL_DEPTH, NodeLevel
from ..domain.node import LegalNode
from ..logging_config import get_logger
from . import patterns as P

logger = get_logger(__name__)

_KHOAN_PARENTS = {NodeLevel.DIEU}
_DIEM_PARENTS = {NodeLevel.KHOAN, NodeLevel.DIEU}


@dataclass
class ParsedDocument:
    meta: LegalDocumentMeta
    root: LegalNode
    header_text: str = ""
    warnings: list[str] = field(default_factory=list)

    def iter_dieu(self) -> list[LegalNode]:
        return self.root.find_all(NodeLevel.DIEU)

    @property
    def stats(self) -> dict[str, int]:
        return {
            "chuong": len(self.root.find_all(NodeLevel.CHUONG)),
            "muc": len(self.root.find_all(NodeLevel.MUC)),
            "dieu": len(self.root.find_all(NodeLevel.DIEU)),
            "khoan": len(self.root.find_all(NodeLevel.KHOAN)),
            "diem": len(self.root.find_all(NodeLevel.DIEM)),
        }


@dataclass
class _Match:
    level: NodeLevel
    number: str
    title: str
    body: str = ""


class StructureAwareParser:
    def __init__(self, metadata_extractor=None, relation_extractor=None) -> None:
        from .metadata_extractor import MetadataExtractor
        from .relation_extractor import RelationExtractor

        self.metadata_extractor = metadata_extractor or MetadataExtractor()
        self.relation_extractor = relation_extractor or RelationExtractor()

    def parse(self, raw_text: str, source_path: str = "") -> ParsedDocument:
        from .metadata_extractor import split_front_matter

        front_matter, text = split_front_matter(P.normalise_text(raw_text) + "\n")
        text = text.strip()
        lines = text.split("\n")
        body_start = self._find_body_start(lines)
        header_text = "\n".join(lines[:body_start])

        meta = self.metadata_extractor.extract(
            header_text=header_text,
            full_text=text,
            source_path=source_path,
            front_matter=front_matter,
        )
        root = LegalNode(level=NodeLevel.VAN_BAN, number=meta.doc_number, title=meta.title)
        warnings: list[str] = []

        self._build_tree(lines, body_start, root, warnings)

        meta.relations = self.relation_extractor.extract(meta=meta, full_text=text)
        parsed = ParsedDocument(meta=meta, root=root, header_text=header_text,
                                warnings=warnings)
        if not parsed.iter_dieu():
            warnings.append("Không nhận diện được Điều nào - kiểm tra định dạng file nguồn.")
        logger.info("Parsed %s -> %s", meta.doc_number or source_path, parsed.stats)
        return parsed

    def _build_tree(self, lines: list[str], start: int, root: LegalNode,
                    warnings: list[str]) -> None:
        stack: list[LegalNode] = [root]
        index = start
        while index < len(lines):
            raw_line = lines[index]
            line = raw_line.strip()
            if not line:
                index += 1
                continue

            match = self._classify(line, stack)
            if match is None:
                self._append_text(stack[-1], line)
                index += 1
                continue

            title = match.title
            if not title and match.level in {NodeLevel.PHAN, NodeLevel.CHUONG,
                                             NodeLevel.MUC, NodeLevel.TIEU_MUC,
                                             NodeLevel.DIEU}:
                title, consumed = self._lookahead_title(lines, index + 1, match.level)
                index += consumed

            node = LegalNode(level=match.level, number=match.number, title=title,
                             text=match.body, line_start=index)
            parent = self._resolve_parent(stack, match.level, warnings)
            parent.add_child(node)
            stack.append(node)
            index += 1

    @staticmethod
    def _resolve_parent(stack: list[LegalNode], level: NodeLevel,
                        warnings: list[str]) -> LegalNode:
        depth = NODE_LEVEL_DEPTH[level]
        while len(stack) > 1 and NODE_LEVEL_DEPTH[stack[-1].level] >= depth:
            stack.pop()
        parent = stack[-1]
        if level is NodeLevel.KHOAN and parent.level not in _KHOAN_PARENTS:
            warnings.append(f"Khoản {level} xuất hiện ngoài Điều (parent={parent.label}).")
        return parent

    @staticmethod
    def _append_text(node: LegalNode, line: str) -> None:
        node.text = f"{node.text} {line}".strip() if node.text else line

    def _classify(self, line: str, stack: list[LegalNode]) -> _Match | None:
        match = P.PHAN_RE.match(line)
        if match:
            return _Match(NodeLevel.PHAN, match.group(1), match.group(2).strip())

        match = P.TIEU_MUC_RE.match(line)
        if match:
            return _Match(NodeLevel.TIEU_MUC, match.group(1), match.group(2).strip())

        match = P.CHUONG_RE.match(line)
        if match:
            return _Match(NodeLevel.CHUONG, match.group(1), match.group(2).strip())

        match = P.MUC_RE.match(line)
        if match:
            return _Match(NodeLevel.MUC, match.group(1), match.group(2).strip())

        match = P.DIEU_RE.match(line)
        if match:
            number, separator, remainder = match.group(1), match.group(2), match.group(3)
            if P.looks_like_dieu_heading(number, separator, remainder):
                return _Match(NodeLevel.DIEU, number, remainder.strip())

        open_levels = {node.level for node in stack}
        match = P.DIEM_RE.match(line)
        if match and open_levels & _DIEM_PARENTS:
            return _Match(NodeLevel.DIEM, match.group(1), title="", body=match.group(2).strip())

        match = P.KHOAN_RE.match(line)
        if match and open_levels & _KHOAN_PARENTS:
            return _Match(NodeLevel.KHOAN, match.group(1), title="", body=match.group(2).strip())

        return None

    def _is_structural(self, line: str) -> bool:
        if any(rgx.match(line) for rgx in (P.PHAN_RE, P.CHUONG_RE, P.MUC_RE, P.TIEU_MUC_RE)):
            return True
        match = P.DIEU_RE.match(line)
        return bool(match and P.looks_like_dieu_heading(match.group(1), match.group(2),
                                                        match.group(3)))

    def _lookahead_title(self, lines: list[str], index: int,
                         level: NodeLevel) -> tuple[str, int]:
        offset = 0
        while index + offset < len(lines) and not lines[index + offset].strip():
            offset += 1
        if index + offset >= len(lines):
            return "", 0
        candidate = lines[index + offset].strip()
        if not candidate or len(candidate) > 200 or self._is_structural(candidate):
            return "", 0
        if P.KHOAN_RE.match(candidate) or P.DIEM_RE.match(candidate):
            return "", 0
        if candidate.endswith((".", ";", ":")):
            return "", 0
        return candidate, offset + 1

    @staticmethod
    def _find_body_start(lines: list[str]) -> int:
        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            if P.PHAN_RE.match(stripped) or P.CHUONG_RE.match(stripped):
                return index
            match = P.DIEU_RE.match(stripped)
            if match and P.looks_like_dieu_heading(match.group(1), match.group(2),
                                                   match.group(3)):
                return index
        return 0
