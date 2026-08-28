from __future__ import annotations
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..logging_config import get_logger

if TYPE_CHECKING:  
    import pandas as pd

logger = get_logger(__name__)

_TEXT_FIELDS = ("text", "content", "noi_dung", "body", "content_html", "raw_text", "full_text")
_TITLE_FIELDS = ("title", "name", "doc_title")

DATASET_NAME = "th1nhng0/vietnamese-legal-documents"


@dataclass
class RawDocument:
    text: str
    source_path: str
    raw_id: Any | None = None
    title: str | None = None
    source_metadata: dict[str, Any] = field(default_factory=dict)


def load_directory(directory: Path,
                   patterns: tuple[str, ...] = ("*.txt", "*.json", "*.jsonl"),
                   ) -> list[RawDocument]:
    documents: list[RawDocument] = []
    for pattern in patterns:
        for path in sorted(directory.glob(pattern)):
            documents.extend(load_file(path))
    logger.info("Đã nạp %d văn bản thô từ thư mục cục bộ %s", len(documents), directory)
    return documents


def load_file(path: Path) -> list[RawDocument]:
    if path.suffix.lower() == ".txt":
        return [RawDocument(text=path.read_text(encoding="utf-8"), source_path=str(path))]
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload if isinstance(payload, list) else [payload]
        return list(_records_to_documents(records, str(path)))
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        return list(_records_to_documents(records, str(path)))
    logger.warning("Bỏ qua định dạng chưa hỗ trợ: %s", path)
    return []


def load_hf_dataset(
    dataset_name: str = DATASET_NAME, 
    config_name: str = "legacy", 
    split: str = "train",
    text_field: str | None = None, 
    limit: int | None = None
) -> list[RawDocument]:
    from datasets import load_dataset

    documents: list[RawDocument] = []

    if config_name == "legacy":
        logger.info("Đang tải config legacy/metadata và legacy/content từ %s...",
                    dataset_name)
        meta_ds = load_dataset(dataset_name, "legacy", split="metadata")
        meta_df = meta_ds.to_pandas()

        content_ds = load_dataset(dataset_name, "legacy", split="content")
        content_df = content_ds.to_pandas()

        join_key = "id" if "id" in meta_df.columns and "id" in content_df.columns else None
        if not join_key:
            raise ValueError(
                "Không tìm thấy cột join chung 'id'. "
                f"Cột meta: {meta_df.columns.tolist()} | "
                f"Content: {content_df.columns.tolist()}"
            )

        merged = meta_df.merge(content_df, on=join_key, suffixes=("_meta", "_content"))
        if limit:
            merged = merged.head(limit)

        for index, row in merged.iterrows():
            text = row.get("content") or row.get("raw_text") or row.get("text") or ""
            if not isinstance(text, str) or not text.strip():
                continue
            
            raw_id = row.get("id")
            title = row.get("title") or row.get("name")
            skip = {"id", "content", "raw_text", "text", "title", "name"}
            metadata = {key: value for key, value in row.items() if key not in skip}
            
            documents.append(
                RawDocument(
                    text=text,
                    source_path=f"hf://{dataset_name}@{config_name}#{raw_id or index}",
                    raw_id=raw_id,
                    title=title,
                    source_metadata=metadata
                )
            )
    else:
        dataset = load_dataset(dataset_name, config_name, split=split)
        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))
        
        for index, record in enumerate(dataset):
            text = record.get(text_field) if text_field else _first_text_field(record)
            if not text:
                continue
            
            raw_id = record.get("id") or record.get("doc_id")
            title = _first_title_field(record)
            metadata = {key: value for key, value in record.items()
                        if key not in _TEXT_FIELDS and key not in _TITLE_FIELDS}

            documents.append(
                RawDocument(
                    text=text,
                    source_path=f"hf://{dataset_name}@{config_name}#{index}",
                    raw_id=raw_id,
                    title=title,
                    source_metadata=metadata
                )
            )

    logger.info("Đã nạp %d văn bản từ HF dataset %s (config=%s)",
                len(documents), dataset_name, config_name)
    return documents


def load_hf_relationships(dataset_name: str = DATASET_NAME) -> pd.DataFrame:
    import pandas as pd
    from datasets import load_dataset

    logger.info("Đang tải relationships cho Knowledge Graph...")
    try:
        rel_ds = load_dataset(dataset_name, "relationships", split="data")
        return rel_ds.to_pandas()
    except Exception as e:
        logger.warning("Không tải được relationships (config có thể không tồn tại ở "
                       "một số phiên bản): %s", e)
        return pd.DataFrame()


def _records_to_documents(records: list[dict], source_path: str) -> Iterator[RawDocument]:
    for index, record in enumerate(records):
        text = _first_text_field(record)
        if not text:
            logger.warning("Bản ghi #%d trong %s không có trường nội dung.", index, source_path)
            continue
        
        raw_id = record.get("id") or record.get("raw_id")
        title = _first_title_field(record)
        metadata = {key: value for key, value in record.items()
                    if key not in _TEXT_FIELDS and key not in _TITLE_FIELDS}

        yield RawDocument(
            text=text,
            source_path=f"{source_path}#{index}",
            raw_id=raw_id,
            title=title,
            source_metadata=metadata
        )


def _first_text_field(record: dict) -> str:
    for field_name in _TEXT_FIELDS:
        value = record.get(field_name)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _first_title_field(record: dict) -> str | None:
    for field_name in _TITLE_FIELDS:
        value = record.get(field_name)
        if isinstance(value, str) and value.strip():
            return value
    return None
from ..logging_config import get_logger

if TYPE_CHECKING:  
    import pandas as pd

logger = get_logger(__name__)

_TEXT_FIELDS = ("text", "content", "noi_dung", "body", "content_html", "raw_text", "full_text")
_TITLE_FIELDS = ("title", "name", "doc_title")

DATASET_NAME = "th1nhng0/vietnamese-legal-documents"


@dataclass
class RawDocument:
    text: str
    source_path: str
    raw_id: Any | None = None
    title: str | None = None
    source_metadata: dict[str, Any] = field(default_factory=dict)


def load_directory(directory: Path,
                   patterns: tuple[str, ...] = ("*.txt", "*.json", "*.jsonl"),
                   ) -> list[RawDocument]:
    documents: list[RawDocument] = []
    for pattern in patterns:
        for path in sorted(directory.glob(pattern)):
            documents.extend(load_file(path))
    logger.info("Đã nạp %d văn bản thô từ thư mục cục bộ %s", len(documents), directory)
    return documents


def load_file(path: Path) -> list[RawDocument]:
    if path.suffix.lower() == ".txt":
        return [RawDocument(text=path.read_text(encoding="utf-8"), source_path=str(path))]
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload if isinstance(payload, list) else [payload]
        return list(_records_to_documents(records, str(path)))
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        return list(_records_to_documents(records, str(path)))
    logger.warning("Bỏ qua định dạng chưa hỗ trợ: %s", path)
    return []


def load_hf_dataset(
    dataset_name: str = DATASET_NAME, 
    config_name: str = "legacy", 
    split: str = "train",
    text_field: str | None = None, 
    limit: int | None = None
) -> list[RawDocument]:
    from datasets import load_dataset

    documents: list[RawDocument] = []

    if config_name == "legacy":
        logger.info("Đang tải config legacy/metadata và legacy/content từ %s...",
                    dataset_name)
        meta_ds = load_dataset(dataset_name, "legacy", split="metadata")
        meta_df = meta_ds.to_pandas()

        content_ds = load_dataset(dataset_name, "legacy", split="content")
        content_df = content_ds.to_pandas()

        join_key = "id" if "id" in meta_df.columns and "id" in content_df.columns else None
        if not join_key:
            raise ValueError(
                "Không tìm thấy cột join chung 'id'. "
                f"Cột meta: {meta_df.columns.tolist()} | "
                f"Content: {content_df.columns.tolist()}"
            )

        merged = meta_df.merge(content_df, on=join_key, suffixes=("_meta", "_content"))
        if limit:
            merged = merged.head(limit)

        for index, row in merged.iterrows():
            text = row.get("content") or row.get("raw_text") or row.get("text") or ""
            if not isinstance(text, str) or not text.strip():
                continue
            
            raw_id = row.get("id")
            title = row.get("title") or row.get("name")
            skip = {"id", "content", "raw_text", "text", "title", "name"}
            metadata = {key: value for key, value in row.items() if key not in skip}
            
            documents.append(
                RawDocument(
                    text=text,
                    source_path=f"hf://{dataset_name}@{config_name}#{raw_id or index}",
                    raw_id=raw_id,
                    title=title,
                    source_metadata=metadata
                )
            )
    else:
        dataset = load_dataset(dataset_name, config_name, split=split)
        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))
        
        for index, record in enumerate(dataset):
            text = record.get(text_field) if text_field else _first_text_field(record)
            if not text:
                continue
            
            raw_id = record.get("id") or record.get("doc_id")
            title = _first_title_field(record)
            metadata = {key: value for key, value in record.items()
                        if key not in _TEXT_FIELDS and key not in _TITLE_FIELDS}

            documents.append(
                RawDocument(
                    text=text,
                    source_path=f"hf://{dataset_name}@{config_name}#{index}",
                    raw_id=raw_id,
                    title=title,
                    source_metadata=metadata
                )
            )

    logger.info("Đã nạp %d văn bản từ HF dataset %s (config=%s)",
                len(documents), dataset_name, config_name)
    return documents


def load_hf_relationships(dataset_name: str = DATASET_NAME) -> pd.DataFrame:
    import pandas as pd
    from datasets import load_dataset

    logger.info("Đang tải relationships cho Knowledge Graph...")
    try:
        rel_ds = load_dataset(dataset_name, "relationships", split="data")
        return rel_ds.to_pandas()
    except Exception as e:
        logger.warning("Không tải được relationships (config có thể không tồn tại ở "
                       "một số phiên bản): %s", e)
        return pd.DataFrame()


def _records_to_documents(records: list[dict], source_path: str) -> Iterator[RawDocument]:
    for index, record in enumerate(records):
        text = _first_text_field(record)
        if not text:
            logger.warning("Bản ghi #%d trong %s không có trường nội dung.", index, source_path)
            continue
        
        raw_id = record.get("id") or record.get("raw_id")
        title = _first_title_field(record)
        metadata = {key: value for key, value in record.items()
                    if key not in _TEXT_FIELDS and key not in _TITLE_FIELDS}

        yield RawDocument(
            text=text,
            source_path=f"{source_path}#{index}",
            raw_id=raw_id,
            title=title,
            source_metadata=metadata
        )


def _first_text_field(record: dict) -> str:
    for field_name in _TEXT_FIELDS:
        value = record.get(field_name)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _first_title_field(record: dict) -> str | None:
    for field_name in _TITLE_FIELDS:
        value = record.get(field_name)
        if isinstance(value, str) and value.strip():
            return value
    return None
