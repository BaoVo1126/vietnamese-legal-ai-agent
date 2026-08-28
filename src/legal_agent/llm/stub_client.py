from __future__ import annotations
import json
import re

from ..logging_config import get_logger
from .base import BaseLLMClient

logger = get_logger(__name__)

_DOC_NUMBER_RE = re.compile(r"\b\d{1,4}/\d{4}/[A-ZĐ][A-ZĐ0-9\-/]*\b")
_DIEU_RE = re.compile(r"[Đđ]iều\s+(\d+[a-zđ]?)")
_EVIDENCE_BLOCK_RE = re.compile(r"^\[(\d+)\]\s*(.+?)\s*\((.+?)\)\s*$", re.MULTILINE)

_INTENT_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("hieu_luc_van_ban", ("còn hiệu lực", "hết hiệu lực", "thay thế", "hướng dẫn",
                          "áp dụng văn bản nào", "hiệu lực")),
    ("che_tai_xu_phat", ("phạt", "chế tài", "xử phạt", "vi phạm", "trách nhiệm hình sự")),
    ("thu_tuc_hanh_chinh", ("hồ sơ", "thủ tục", "trình tự", "đăng ký", "thời hạn",
                            "bao lâu", "nộp")),
    ("so_sanh_doi_chieu", ("so sánh", "khác nhau", "khác gì", "đối chiếu", "trước đây")),
    ("tra_cuu_dieu_khoan", ("điều ", "khoản ", "điểm ")),
    ("hoi_dap_khai_niem", ("là gì", "được hiểu", "khái niệm", "định nghĩa")),
)

_LIST_QUESTION_RE = re.compile(
    r"gồm những|bao gồm|liệt kê|những gì|nào sau đây|(?:những|các)(?:\s+\S+){1,4}\s+nào",
    re.UNICODE | re.IGNORECASE,
)
_MAX_LIST_BLOCKS = 5


def _is_list_question(question: str) -> bool:
    return bool(_LIST_QUESTION_RE.search(question))


def _article_key(citation: str) -> str:
    match = re.search(r"(Điều\s+\d+[a-zđ]?)", citation)
    document = citation.split(",")[-1].strip()
    return f"{match.group(1) if match else citation}|{document}"


_NON_EVIDENTIAL_TERMS = {
    "gì", "nào", "thế_nào", "như_thế_nào", "ra_sao", "sao", "bao_nhiêu", "bao_lâu",
    "ai", "đâu", "mấy", "hiểu", "được_hiểu", "hiện_hành", "hiện_nay", "cụ_thể",
    "xin_hỏi", "cho_hỏi", "vui_lòng", "giải_thích", "nêu",
}


class RuleBasedStubLLM(BaseLLMClient):
    def __init__(self, tokenizer=None) -> None:
        from ..indexing.tokenizer import VietnameseTokenizer

        self.tokenizer = tokenizer or VietnameseTokenizer()

    def complete(self, system: str, user: str, *, task: str = "generic",
                 temperature: float | None = None, max_tokens: int | None = None) -> str:
        handler = {
            "router": self._route,
            "verifier": self._verify,
            "answer": self._answer,
            "claim_extraction": self._extract_claims,
            "claim_verification": self._verify_claims,
        }.get(task)
        if handler is None:
            logger.debug("Stub LLM: task %s không có handler - trả về chuỗi rỗng.", task)
            return ""
        return handler(user)

    def _route(self, user: str) -> str:
        question = _section(user, 'Câu hỏi của người dùng:') or user
        lowered = question.lower()
        intent = "hoi_dap_khai_niem"
        for candidate, keywords in _INTENT_KEYWORDS:
            if any(keyword in lowered for keyword in keywords):
                intent = candidate
                break
        sub_queries = [part.strip() for part in re.split(r"\bvà\b|\?|;", question)
                       if len(part.strip()) > 12]
        return json.dumps({
            "intent": intent,
            "rewritten_query": question.strip().strip('"'),
            "sub_queries": sub_queries[:3] if len(sub_queries) > 1 else [],
            "doc_numbers": _DOC_NUMBER_RE.findall(question),
            "doc_titles": _named_documents(question),
            "dieu_hints": _DIEU_RE.findall(question),
            "reasoning": f"Khớp từ khoá cho intent {intent}.",
        }, ensure_ascii=False)

    def _verify(self, user: str) -> str:
        question = _section(user, "CÂU HỎI:", "BẰNG CHỨNG ĐÃ TRUY XUẤT:")
        evidence = _section(user, "BẰNG CHỨNG ĐÃ TRUY XUẤT:")
        score = self._coverage(question, evidence)
        citations = [match.group(2) for match in _EVIDENCE_BLOCK_RE.finditer(evidence)][:3]
        return json.dumps({
            "grounding_score": score,
            "is_sufficient": score >= 0.6,
            "missing_information": "" if score >= 0.6 else
                                   "Bằng chứng chưa nêu trực tiếp quy định được hỏi.",
            "rewritten_query": "" if score >= 0.6 else _keyword_query(question),
            "relevant_citations": citations,
        }, ensure_ascii=False)

    def _answer(self, user: str) -> str:
        question = _section(user, "CÂU HỎI:", "BẰNG CHỨNG (chỉ được dùng phần này):")
        evidence = _section(user, "BẰNG CHỨNG (chỉ được dùng phần này):",
                            "THÔNG TIN HIỆU LỰC TỪ KNOWLEDGE GRAPH")
        blocks = _parse_evidence_blocks(evidence)
        if not blocks:
            return "Không đủ căn cứ pháp lý rõ ràng để trả lời câu hỏi này."

        if _is_list_question(question):
            return self._answer_list(question, blocks)

        primary = blocks[0]
        lines = [
            f"**Đáp án: {_first_sentences(primary['text'], limit=1)}**",
            f"Căn cứ: ({primary['citation']})",
        ]
        explanations = [
            f"{_first_sentences(block['text'], limit=1)} ({block['citation']})"
            for block in blocks[1:3]
        ]
        if explanations:
            lines.append("Giải thích: " + " ".join(explanations))
        return "\n\n".join(lines)

    def _answer_list(self, question: str, blocks: list[dict]) -> str:
        groups: dict[str, list[dict]] = {}
        for block in blocks:
            groups.setdefault(_article_key(block["citation"]), []).append(block)

        def score(item):
            _, members = item
            combined = " ".join(member["text"] for member in members)
            best_rank = min(blocks.index(member) for member in members)
            return (self._coverage(question, combined) + 0.03 * min(len(members), 4),
                    -best_rank)

        _, chosen = max(groups.items(), key=score)
        chosen = chosen[:_MAX_LIST_BLOCKS]
        lines = [
            f"**Đáp án: {_first_sentences(chosen[0]['text'], limit=1)}**",
            "Căn cứ: " + ", ".join(f"({block['citation']})" for block in chosen),
        ]
        details = [f"{_first_sentences(block['text'], limit=1)} ({block['citation']})"
                   for block in chosen[1:]]
        if details:
            lines.append("Giải thích: " + " ".join(details))
        return "\n\n".join(lines)

    def _extract_claims(self, user: str) -> str:
        answer = _section(user, "CÂU TRẢ LỜI CẦN TÁCH:")
        claims = []
        for line in answer.split("\n"):
            line = line.strip("-• ").strip()
            if len(line) < 20:
                continue
            citation_match = re.search(r"\(([^()]*\d{1,4}/\d{4}/[A-ZĐ][^()]*)\)", line)
            claims.append({
                "text": line,
                "citation": citation_match.group(1) if citation_match else "",
            })
        return json.dumps({"claims": claims}, ensure_ascii=False)

    def _verify_claims(self, user: str) -> str:
        evidence = _section(user, "BẰNG CHỨNG GỐC:", "CÁC LUẬN ĐIỂM CẦN KIỂM CHỨNG:")
        claims_block = _section(user, "CÁC LUẬN ĐIỂM CẦN KIỂM CHỨNG:")
        verdicts = []
        for line in claims_block.split("\n"):
            match = re.match(r"\s*\[(\d+)\]\s*(.+)", line)
            if not match:
                continue
            index, claim_text = int(match.group(1)), match.group(2)
            coverage = self._coverage(claim_text, evidence)
            verdict = ("supported" if coverage >= 0.7 else
                       "partially_supported" if coverage >= 0.4 else "unsupported")
            verdicts.append({"index": index, "verdict": verdict, "evidence_index": None,
                             "reason": f"Độ phủ từ vựng với bằng chứng: {coverage:.2f}."})
        return json.dumps({"verdicts": verdicts}, ensure_ascii=False)
    
    def _coverage(self, query: str, text: str) -> float:
        query_terms = set(self.tokenizer.tokenize(query))
        content_terms = query_terms - _NON_EVIDENTIAL_TERMS
        query_terms = content_terms or query_terms
        text_terms = set(self.tokenizer.tokenize(text))
        if not query_terms or not text_terms:
            return 0.0
        text_syllables = {syllable for term in text_terms for syllable in term.split("_")}
        covered = sum(
            1 for term in query_terms
            if term in text_terms or set(term.split("_")) <= text_syllables
        )
        return round(covered / len(query_terms), 3)


# Từ không bao giờ thuộc về tên văn bản - chúng kết thúc phần tên.
_NAME_STOPWORDS = (
    r"này|số|năm|là|gì|thì|và|có|quy|được|cho|tại|theo|của|do|khi|nào|không|hiện|thế"
)
_NAMED_DOC_RE = re.compile(
    r"(?:Bộ luật|Luật|Pháp lệnh|Nghị định|Thông tư|Nghị quyết|Quyết định|Hiến pháp)"
    # Tên văn bản tiếng Việt chỉ viết hoa từ đầu: "Luật Chứng khoán", "Bộ luật Hình sự".
    rf"(?:\s+(?!(?:{_NAME_STOPWORDS})\b)[A-ZĐÀ-Ỹ][a-zà-ỹ]*"
    rf"(?:\s+(?!(?:{_NAME_STOPWORDS})\b)[a-zà-ỹ]+){{0,5}})?",
    re.UNICODE,
)


def _named_documents(question: str) -> list[str]:
    """Tên các văn bản được gọi đích danh trong câu hỏi.

    "theo Luật Chứng khoán" là một ràng buộc về phạm vi, không phải từ khoá tìm kiếm:
    nếu kho không có văn bản đó thì trả lời bằng một luật khác là sai bản chất, dù trích
    dẫn có hợp lệ đến đâu.
    """
    found = []
    for match in _NAMED_DOC_RE.finditer(question):
        name = match.group(0).strip()
        if len(name.split()) >= 2 and name not in found:
            found.append(name)
    return found


def _section(text: str, header: str, *stops: str) -> str:
    start = text.find(header)
    if start == -1:
        return ""
    remainder = text[start + len(header):].lstrip()

    fence = '"""'
    if remainder.startswith(fence):
        end = remainder.find(fence, len(fence))
        if end != -1:
            return remainder[len(fence):end].strip()

    end = len(remainder)
    for stop in (*stops, "Trả về JSON", "Hãy trả lời câu hỏi"):
        position = remainder.find(stop)
        if position != -1:
            end = min(end, position)
    return remainder[:end].strip().strip('"').strip()


def _parse_evidence_blocks(evidence: str) -> list[dict]:
    blocks: list[dict] = []
    matches = list(_EVIDENCE_BLOCK_RE.finditer(evidence))
    for position, match in enumerate(matches):
        end = matches[position + 1].start() if position + 1 < len(matches) else len(evidence)
        body = evidence[match.end():end].strip()
        blocks.append({
            "index": int(match.group(1)),
            "citation": match.group(2).strip(),
            "status": match.group(3).strip(),
            "text": _statutory_text(body),
        })
    return blocks


def _statutory_text(body: str) -> str:
    marker = "Nội dung:"
    position = body.find(marker)
    if position != -1:
        return body[position + len(marker):].strip()
    lines = [line for line in body.split("\n") if line.strip()]
    return lines[-1] if lines else ""


_ENUMERATION_PREFIX_RE = re.compile(r"^\s*(?:\d+[a-zđ]?\s*\.|[a-zđ]{1,2}\s*\))\s*")
_HEADING_LINE_RE = re.compile(
    r"^\s*(?:Điều|Chương|Mục|Tiểu mục|Phần)\s+[\dIVXLCDM]+[a-zđ]?\s*[.:]?\s*[^.;]{0,90}$",
    re.UNICODE,
)

def _first_sentences(text: str, limit: int = 2) -> str:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    content = [line for line in lines if not _HEADING_LINE_RE.match(line)] or lines
    body = _ENUMERATION_PREFIX_RE.sub("", content[0] if content else "")
    sentences = re.split(r"(?<=[.;])\s+", body)
    return " ".join(sentences[:limit]).strip()


def _keyword_query(question: str) -> str:
    cleaned = re.sub(r"\b(là gì|như thế nào|ra sao|không|ạ|vậy|thế)\b", " ", question,
                     flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip(" ?.")
