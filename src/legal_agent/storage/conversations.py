from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..logging_config import get_logger

logger = get_logger(__name__)

MAX_TITLE_CHARS = 60

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    payload         TEXT,
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON messages(conversation_id, id);
CREATE INDEX IF NOT EXISTS idx_conversations_updated
    ON conversations(updated_at DESC);
"""


@dataclass(frozen=True)
class ConversationSummary:
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int


@dataclass(frozen=True)
class StoredMessage:
    role: str
    content: str
    payload: dict[str, Any] | None
    created_at: datetime


class ConversationStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(_SCHEMA)
        logger.info("Kho hội thoại: %s", path)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            yield connection
            connection.commit()
        finally:
            connection.close()

    def create(self, title: str = "") -> str:
        conversation_id = uuid.uuid4().hex
        now = _now()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (conversation_id, make_title(title), now, now),
            )
        return conversation_id

    def append(self, conversation_id: str, role: str, content: str,
               payload: dict[str, Any] | None = None) -> None:
        now = _now()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO messages (conversation_id, role, content, payload, "
                "created_at) VALUES (?, ?, ?, ?, ?)",
                (conversation_id, role, content,
                 json.dumps(payload, ensure_ascii=False) if payload else None, now),
            )
            connection.execute("UPDATE conversations SET updated_at = ? WHERE id = ?",
                               (now, conversation_id))

    def rename(self, conversation_id: str, title: str) -> None:
        with self._connect() as connection:
            connection.execute("UPDATE conversations SET title = ? WHERE id = ?",
                               (make_title(title), conversation_id))

    def delete(self, conversation_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM messages WHERE conversation_id = ?",
                               (conversation_id,))
            connection.execute("DELETE FROM conversations WHERE id = ?",
                               (conversation_id,))

    def list_recent(self, limit: int = 60) -> list[ConversationSummary]:
        """Hội thoại mới cập nhật gần nhất trước; bỏ qua hội thoại chưa có lượt nói nào."""
        query = (
            "SELECT c.id, c.title, c.created_at, c.updated_at, "
            "       COUNT(m.id) AS message_count "
            "FROM conversations c LEFT JOIN messages m ON m.conversation_id = c.id "
            "GROUP BY c.id HAVING message_count > 0 "
            "ORDER BY c.updated_at DESC, c.rowid DESC LIMIT ?"
        )
        with self._connect() as connection:
            rows = connection.execute(query, (limit,)).fetchall()
        return [
            ConversationSummary(
                id=row["id"], title=row["title"],
                created_at=_parse(row["created_at"]), updated_at=_parse(row["updated_at"]),
                message_count=row["message_count"],
            )
            for row in rows
        ]

    def messages(self, conversation_id: str) -> list[StoredMessage]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT role, content, payload, created_at FROM messages "
                "WHERE conversation_id = ? ORDER BY id",
                (conversation_id,),
            ).fetchall()
        return [
            StoredMessage(
                role=row["role"], content=row["content"],
                payload=json.loads(row["payload"]) if row["payload"] else None,
                created_at=_parse(row["created_at"]),
            )
            for row in rows
        ]

    def count(self) -> tuple[int, int]:
        with self._connect() as connection:
            conversations = connection.execute(
                "SELECT COUNT(*) FROM conversations").fetchone()[0]
            messages = connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        return conversations, messages


def make_title(text: str) -> str:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return "Hội thoại mới"
    if len(cleaned) <= MAX_TITLE_CHARS:
        return cleaned
    cut = cleaned[:MAX_TITLE_CHARS].rsplit(" ", 1)[0]
    return (cut or cleaned[:MAX_TITLE_CHARS]).rstrip(" ,.;:") + "..."


def group_by_day(conversations: list[ConversationSummary]) -> list[tuple[str, list]]:
    today = datetime.now().astimezone().date()
    buckets: dict[str, list[ConversationSummary]] = {}
    order: list[str] = []
    for conversation in conversations:
        delta = (today - conversation.updated_at.date()).days
        if delta <= 0:
            label = "Hôm nay"
        elif delta == 1:
            label = "Hôm qua"
        elif delta < 7:
            label = "7 ngày qua"
        elif delta < 30:
            label = "30 ngày qua"
        else:
            label = "Cũ hơn"
        if label not in buckets:
            buckets[label] = []
            order.append(label)
        buckets[label].append(conversation)
    return [(label, buckets[label]) for label in order]


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="microseconds")


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value)
