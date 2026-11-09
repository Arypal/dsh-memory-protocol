"""Pending candidate store — SQLite-backed, isolated from the retrieval pool.

Candidates produced by distill() live here until the main agent accepts
(reject) them.  They are intentionally NOT in the learned store: they must
never appear in retrieval results or weave injection.  The agent reviews
them via memory_pending and promotes accepted ones through the normal
teach_skill / structured-ingest path (which keeps the ADR-0002 rule that
the agent owns learning).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


@dataclass
class PendingCandidate:
    candidate_id: str
    topic: str
    summary: str
    evidence: list[str] = field(default_factory=list)
    suggested: str = "chat"
    confidence: float = 0.5
    status: str = "open"  # open | accepted | rejected
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


def _now_ts() -> float:
    return datetime.now(UTC).timestamp()


class PendingStore:
    """SQLite-backed store of distill candidates awaiting agent review."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=10)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pending_candidates (
                candidate_id TEXT PRIMARY KEY,
                topic TEXT NOT NULL,
                summary TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                suggested TEXT NOT NULL,
                confidence REAL NOT NULL,
                status TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def add_candidate(self, candidate, dialogue_store) -> tuple[bool, str]:
        """Insert a candidate after validating its evidence ids exist.

        Anti-hallucination guard: every evidence id must resolve to a real
        dialogue turn, otherwise the candidate is rejected (ADR-0002).
        """
        valid_ids = {t.id for t in dialogue_store.get_recent(2000)}
        bad = [eid for eid in candidate.evidence if eid not in valid_ids]
        if bad:
            return False, f"evidence not found: {bad[:3]}"

        cand_id = (
            f"dc_{datetime.now(UTC):%Y%m%d_%H%M%S}_"
            f"{abs(hash(candidate.topic)) & 0xFFFF:04x}"
        )
        try:
            self._conn.execute(
                """
                INSERT INTO pending_candidates
                (candidate_id, topic, summary, evidence_json, suggested,
                 confidence, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 'open', ?)
                """,
                (
                    cand_id,
                    candidate.topic,
                    candidate.summary,
                    json.dumps(candidate.evidence, ensure_ascii=False),
                    candidate.suggested,
                    candidate.confidence,
                    _now_ts(),
                ),
            )
            self._conn.commit()
            return True, cand_id
        except sqlite3.IntegrityError:
            return False, "duplicate candidate"

    def list_open(self, limit: int = 20) -> list[PendingCandidate]:
        return self._query("status = 'open'", limit)

    def mark(self, candidate_id: str, status: str) -> bool:
        """Transition a candidate's status; False when no change happened."""
        if status not in ("accepted", "rejected"):
            return False
        cur = self._conn.execute(
            "UPDATE pending_candidates SET status = ? "
            "WHERE candidate_id = ? AND status <> ?",
            (status, candidate_id, status),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def count_open(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM pending_candidates WHERE status = 'open'"
        ).fetchone()
        return int(row[0]) if row else 0

    def _query(self, where: str, limit: int) -> list[PendingCandidate]:
        rows = self._conn.execute(
            f"SELECT * FROM pending_candidates WHERE {where} "
            f"ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        out: list[PendingCandidate] = []
        for r in rows:
            try:
                evidence = json.loads(r[3])
            except json.JSONDecodeError:
                evidence = []
            out.append(PendingCandidate(
                candidate_id=r[0],
                topic=r[1],
                summary=r[2],
                evidence=evidence,
                suggested=r[4],
                confidence=r[5],
                status=r[6],
                created_at=datetime.fromtimestamp(r[7], tz=UTC),
            ))
        return out
