from __future__ import annotations

from collections.abc import Iterator

from pydantic import BaseModel, Field

from .enums import NODE_LEVEL_DEPTH, NodeLevel


class LegalNode(BaseModel):
    level: NodeLevel
    number: str = Field("", description="Số thứ tự, e.g. '12', 'I', 'a'")
    title: str = Field("", description="Tiêu đề, e.g. 'Quyền của doanh nghiệp'")
    text: str = Field("", description="Nội dung trực tiếp của node (không gồm con)")
    children: list[LegalNode] = Field(default_factory=list)
    line_start: int = 0

    @property
    def depth(self) -> int:
        return NODE_LEVEL_DEPTH[self.level]

    @property
    def label(self) -> str:
        if self.level is NodeLevel.VAN_BAN:
            return self.title or "Văn bản"
        return f"{self.level.label} {self.number}".strip()

    @property
    def heading(self) -> str:
        return f"{self.label}. {self.title}".strip(". ") if self.title else self.label
    
    def add_child(self, child: LegalNode) -> LegalNode:
        self.children.append(child)
        return child

    def iter_descendants(self) -> Iterator[LegalNode]:
        for child in self.children:
            yield child
            yield from child.iter_descendants()

    def find_all(self, level: NodeLevel) -> list[LegalNode]:
        return [node for node in self.iter_descendants() if node.level is level]

    def full_text(self) -> str:
        parts: list[str] = []
        if self.heading and self.level in {NodeLevel.DIEU, NodeLevel.CHUONG, NodeLevel.MUC}:
            parts.append(self.heading)
        own = self.own_text_with_label()
        if own:
            parts.append(own)
        for child in self.children:
            child_text = child.full_text()
            if child_text:
                parts.append(child_text)
        return "\n".join(parts)

    def own_text_with_label(self) -> str:
        body = self.text.strip()
        if self.level is NodeLevel.KHOAN:
            return f"{self.number}. {body}".strip()
        if self.level is NodeLevel.DIEM:
            return f"{self.number}) {body}".strip()
        return body

    def char_count(self) -> int:
        return len(self.full_text())

LegalNode.model_rebuild()
