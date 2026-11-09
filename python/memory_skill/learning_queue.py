"""Persistent learning-queue store.

Replaces the passive gap detector: when the classifier routes a user turn
to ``skill`` or ``mission``, an item is enqueued here.  The weaver renders
open items as explicit directives (``[待学习]`` / ``[待拆解]``) that the
main agent must act on — check existing skill, learn via web search, teach
the result back, or decompose a mission into steps.

Item lifecycle: ``open`` → ``done`` / ``skipped``.  A ``skill`` item is
marked done once the agent has taught/updated the skill and the user has
confirmed; a ``mission`` item is marked done once it has been decomposed
into steps with required skills resolved.
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

logger = logging.getLogger("memory_skill.learning_queue")


_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS learning_queue (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,            -- 'skill' | 'mission'
    query       TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'open',  -- open | done | skipped
    detail      TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL
);
"""


@dataclass(frozen=True)
class QueueItem:
    """One pending learning item."""
    id: str
    kind: str          # "skill" | "mission"
    query: str
    status: str        # "open" | "done" | "skipped"
    detail: str = ""
    created_at: datetime = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.created_at is None:
            object.__setattr__(self, "created_at", datetime.now(UTC))


class LearningQueue:
    """SQLite-backed queue of skills/missions awaiting agent action."""

    def __init__(self, db_path: str) -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=10)
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.executescript(_CREATE_SQL)

    # ── Public API ──────────────────────────────────────────────────────

    def enqueue(self, kind: str, query: str, detail: str = "") -> QueueItem | None:
        """Add an open item if no identical open item exists (dedup).

        Returns the new item, or ``None`` when a matching open item
        already exists (or *query* is empty).
        """
        query = (query or "").strip()
        if not query or kind not in ("skill", "mission"):
            return None
        if self._has_open(kind, query):
            return None
        item_id = f"lq_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        item = QueueItem(
            id=item_id, kind=kind, query=query,
            status="open", detail=detail[:500],
            created_at=datetime.now(UTC),
        )
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO learning_queue "
                "(id, kind, query, status, detail, created_at) "
                "VALUES (?, ?, ?, 'open', ?, ?)",
                (item.id, item.kind, item.query, item.detail,
                 item.created_at.timestamp()),
            )
            self._conn.commit()
        except Exception as exc:
            logger.warning("Failed to enqueue %r: %s", query[:40], exc)
            return None
        return item

    def open_items(self, kind: str | None = None) -> list[QueueItem]:
        """Return open items, oldest first (FIFO)."""
        sql = "SELECT id, kind, query, status, detail, created_at FROM learning_queue WHERE status = 'open'"
        args: tuple = ()
        if kind:
            sql += " AND kind = ?"
            args = (kind,)
        sql += " ORDER BY created_at ASC"
        rows = self._conn.execute(sql, args).fetchall()
        return [self._row_to_item(r) for r in rows]

    def mark(self, item_id: str, status: str) -> bool:
        """Transition an item to ``done`` or ``skipped``. Idempotent."""
        if status not in ("done", "skipped"):
            return False
        cur = self._conn.execute(
            "UPDATE learning_queue SET status = ? WHERE id = ? AND status = 'open'",
            (status, item_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def count_open(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM learning_queue WHERE status = 'open'"
        ).fetchone()
        return int(row[0]) if row else 0

    def all(self, limit: int = 50) -> list[QueueItem]:
        rows = self._conn.execute(
            "SELECT id, kind, query, status, detail, created_at "
            "FROM learning_queue ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_item(r) for r in rows]

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    # ── Helpers ─────────────────────────────────────────────────────────

    def _has_open(self, kind: str, query: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM learning_queue WHERE kind = ? AND query = ? AND status = 'open'",
            (kind, query),
        ).fetchone()
        return row is not None

    @staticmethod
    def _row_to_item(row: tuple) -> QueueItem:
        return QueueItem(
            id=row[0], kind=row[1], query=row[2], status=row[3],
            detail=row[4], created_at=datetime.fromtimestamp(row[5], tz=UTC),
        )
