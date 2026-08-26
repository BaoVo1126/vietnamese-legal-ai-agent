from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legal_agent.config import get_settings
from legal_agent.domain.citation import Citation
from legal_agent.domain.enums import NodeLevel
from legal_agent.evaluation.datasets.legal_qa_regression import (
    REGRESSION_CASES,
)
from legal_agent.ingestion.parser import StructureAwareParser
from legal_agent.logging_config import setup_logging


def load_corpus(raw_dir: Path) -> dict[str, dict]:
    """Parse mọi văn bản thô -> {doc_key: {dieu -> text}}."""
    parser = StructureAwareParser()
    corpus: dict[str, dict] = {}
    for path in sorted(raw_dir.glob("*.txt")):
        parsed = parser.parse(path.read_text(encoding="utf-8"), source_path=str(path))
        meta = parsed.meta
        articles = {
            node.number: node.full_text()
            for node in parsed.root.find_all(NodeLevel.DIEU)
        }
        key = meta.doc_number or meta.title
        corpus[key] = {"title": meta.title, "doc_number": meta.doc_number,
                       "status": meta.effect_status.value, "articles": articles,
                       "file": path.name}
    return corpus


def resolve_document(corpus: dict[str, dict], citation: Citation) -> dict | None:
    if citation.doc_number:
        return corpus.get(citation.doc_number)
    wanted = (citation.doc_title or "").lower()
    if not wanted:
        return None
    for record in corpus.values():
        title = record["title"].lower()
        if wanted in title or title in wanted:
            return record
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Kiểm tra nhãn vàng của bộ regression")
    parser.add_argument("--raw", default=None)
    parser.add_argument("--log-level", default="ERROR")
    args = parser.parse_args()

    setup_logging(args.log_level)
    raw_dir = Path(args.raw) if args.raw else get_settings().abs_raw_data_dir
    corpus = load_corpus(raw_dir)
    print(f"Corpus: {len(corpus)} văn bản, "
          f"{sum(len(record['articles']) for record in corpus.values())} Điều\n")

    ok = bad = skipped = 0
    problems: list[str] = []

    for case in REGRESSION_CASES:
        expected = case.get("expected_citations", [])
        if not expected:
            skipped += 1
            continue
        terms = case.get("answer_terms", [])
        for raw_citation in expected:
            parsed = Citation.parse_all(raw_citation)
            if not parsed:
                problems.append(f"{case['case_id']}: không parse được {raw_citation!r}")
                bad += 1
                continue
            citation = parsed[0]
            document = resolve_document(corpus, citation)
            if document is None:
                problems.append(f"{case['case_id']}: KHÔNG CÓ văn bản cho {raw_citation!r}")
                bad += 1
                continue
            if citation.dieu is None:
                ok += 1
                continue
            article = document["articles"].get(citation.dieu)
            if article is None:
                problems.append(f"{case['case_id']}: {document['file']} không có Điều "
                                f"{citation.dieu}")
                bad += 1
                continue
            missing = [term for term in terms if term.lower() not in article.lower()]
            if missing:
                hits = [number for number, text in document["articles"].items()
                        if all(term.lower() in text.lower() for term in terms)]
                problems.append(
                    f"{case['case_id']}: Điều {citation.dieu} ({document['file']}) KHÔNG "
                    f"chứa {missing}; Điều chứa đủ cụm từ: {hits[:6] or 'không có'}")
                bad += 1
                continue
            ok += 1
            print(f"   {case['case_id']:<28} {raw_citation}")

    print()
    for problem in problems:
        print(f"   {problem}")
    print(f"\nTổng: {ok} nhãn đúng, {bad} nhãn sai/thiếu, {skipped} case không có nhãn.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
