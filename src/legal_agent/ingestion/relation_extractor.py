from __future__ import annotations

import re

from ..domain.document import LegalDocumentMeta, LegalRelation, make_doc_id
from ..domain.enums import RelationType
from ..logging_config import get_logger
from . import patterns as P

logger = get_logger(__name__)

_CUE_TO_RELATION: dict[str, tuple[RelationType, float]] = {
    "thay thế cho": (RelationType.THAY_THE, 0.95),
    "thay thế": (RelationType.THAY_THE, 0.95),
    "sửa đổi, bổ sung một số điều của": (RelationType.SUA_DOI, 0.95),
    "sửa đổi, bổ sung": (RelationType.SUA_DOI, 0.9),
    "quy định chi tiết thi hành": (RelationType.HUONG_DAN, 0.95),
    "quy định chi tiết": (RelationType.HUONG_DAN, 0.9),
    "hướng dẫn thi hành": (RelationType.HUONG_DAN, 0.95),
    "hướng dẫn": (RelationType.HUONG_DAN, 0.8),
    "bãi bỏ": (RelationType.BAI_BO, 0.9),
    "căn cứ": (RelationType.CAN_CU, 0.85),
}
_CUES_LONGEST_FIRST = sorted(_CUE_TO_RELATION, key=len, reverse=True)

_EXPIRY_CUE = "hết hiệu lực"

_WORD_REPLACEMENT_RE = re.compile(
    r"thay\s+thế\s+(?:các\s+)?(?:cụm\s+từ|từ\s+ngữ|từ|chữ|dấu|điểm|khoản|điều)\b",
    re.UNICODE,
)
_AMENDING_TITLE_CUES = ("sửa đổi", "bổ sung một số điều")


class RelationExtractor:
    def extract(self, meta: LegalDocumentMeta, full_text: str) -> list[LegalRelation]:
        relations: dict[tuple, LegalRelation] = {}
        own_number = meta.doc_number
        is_amending = any(cue in meta.title.lower() for cue in _AMENDING_TITLE_CUES)

        for sentence in _iter_sentences(full_text):
            foreign_numbers = [
                number for number in P.DOC_NUMBER_RE.findall(sentence)
                if number != own_number
            ]
            if not foreign_numbers:
                continue

            dieu_by_doc = {
                doc_fragment.split()[-1]: dieu
                for dieu, doc_fragment in P.DIEU_OF_DOC_RE.findall(sentence)
            }

            for relation_type, confidence in self._relations_in(sentence, is_amending):
                for number in foreign_numbers:
                    relation = LegalRelation(
                        source_doc_id=meta.doc_id,
                        target_doc_id=make_doc_id(number),
                        relation=relation_type,
                        target_dieu=dieu_by_doc.get(number),
                        evidence=_shorten(sentence),
                        confidence=confidence,
                    )
                    relations.setdefault(relation.key, relation)
        logger.debug("%s -> %d relations", meta.doc_number, len(relations))
        return list(relations.values())

    @staticmethod
    def _relations_in(sentence: str,
                      is_amending: bool = False) -> list[tuple[RelationType, float]]:
        lowered = sentence.lower()
        word_replacement = bool(_WORD_REPLACEMENT_RE.search(lowered))
        found: list[tuple[RelationType, float]] = []
        for cue in _CUES_LONGEST_FIRST:
            if cue not in lowered:
                continue
            relation, confidence = _CUE_TO_RELATION[cue]
            if relation is RelationType.THAY_THE and (word_replacement or is_amending):
                continue
            if relation not in {existing for existing, _ in found}:
                found.append((relation, confidence))
        if (_EXPIRY_CUE in lowered and not is_amending and not word_replacement
                and not any(r is RelationType.THAY_THE for r, _ in found)):
            found.append((RelationType.THAY_THE, 0.85))
        return found


def _iter_sentences(text: str):
    for block in re.split(r"\n+", text):
        for sentence in re.split(r"(?<=[.;])\s+", block):
            sentence = sentence.strip()
            if sentence:
                yield sentence


def _shorten(sentence: str, limit: int = 300) -> str:
    sentence = re.sub(r"\s+", " ", sentence).strip()
    return sentence if len(sentence) <= limit else sentence[: limit - 1] + "…"
