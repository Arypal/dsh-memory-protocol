"""Skill Writer — teach and update skill memories.

Active-learning loop's write side.  After the agent learns a topic (via
web search or user instruction), it persists the knowledge with
``teach_skill`` and can later revise it with ``update_skill``.

Differences from the legacy learning pipeline:
  - No crawl/synthesize inside: the agent prepares the content (it has
    better web access than an internal crawler).
  - ``teach_skill`` stores at high confidence (weight 0.85) so taught
    skills outrank noisy dialogue fragments.
  - ``update_skill`` REPLACES the entry's content — it is a teaching
    correction, not a semantic-merge.  This makes skills writable so the
    user can teach or correct the agent.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

logger = logging.getLogger("memory_skill.skill_writer")

_TEACH_WEIGHT = 0.85


class SkillWriter:
    """Persist / rewrite ``category=skill`` entries."""

    def __init__(self, ms):
        """``ms`` is a MemorySystem (needs ingest_skill_ex + learned_store)."""
        self._ms = ms

    # ── Public API ──────────────────────────────────────────────────────

    def teach_skill(self, title: str, content: str,
                    source_urls: list[str] | None = None) -> dict:
        """Store a taught skill at high confidence.

        Returns ``{"status": "stored", "entry_id": ...}`` or
        ``{"status": "error", "reason": ...}``.
        """
        title = (title or "").strip()
        content = (content or "").strip()
        if not title or not content:
            return {"status": "error", "reason": "title and content required"}
        if not source_urls:
            # 内部排障 / 操作经验没有外部 URL。允许空，但打上内部来源标记，
            # 与"外部学的技能"区分开（外部学的仍必须给 URL）。
            source_urls = ["internal:agent-experience"]

        try:
            receipt = self._ms.ingest_skill(title, content, source_urls=source_urls)
        except Exception as exc:
            logger.exception("teach_skill failed for %r", title)
            return {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}

        entry_id = receipt.entry_id
        # Bump weight to teaching confidence (high, not evictable-last)
        try:
            self._ms.learned_store.set_weight(entry_id, _TEACH_WEIGHT)
        except Exception as exc:
            logger.warning("teach_skill weight bump failed: %s", exc)

        self._mark_queue_done("skill", title)

        return {"status": "stored", "entry_id": entry_id, "title": title}

    def update_skill(self, entry_id: str, content: str) -> dict:
        """Replace a skill entry's content — a user/agent correction.

        Unlike the legacy dedup-merge, this is a direct rewrite: the new
        content fully replaces the old, re-embedded so future retrieval
        hits the corrected version.

        Returns ``{"status": "updated", "entry_id": ...}`` or an error.
        """
        entry_id = (entry_id or "").strip()
        content = (content or "").strip()
        if not entry_id or not content:
            return {"status": "error", "reason": "entry_id and content required"}

        try:
            self._ms.learned_store.update(entry_id, content)
        except KeyError:
            return {"status": "error", "reason": f"entry {entry_id} not found"}
        except Exception as exc:
            logger.exception("update_skill failed for %s", entry_id)
            return {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}

        return {"status": "updated", "entry_id": entry_id}

    # ── Helpers ─────────────────────────────────────────────────────────

    def _mark_queue_done(self, kind: str, query: str) -> None:
        """Close matching open learning-queue items after a successful write."""
        queue = getattr(self._ms, "learning_queue", None)
        if queue is None:
            return
        try:
            for item in queue.open_items(kind=kind):
                if item.query.strip() == query.strip():
                    queue.mark(item.id, "done")
        except Exception as exc:
            logger.warning("learning_queue mark-done failed: %s", exc)

        pending = self._ms.protocol.open_gaps()
        for gap in list(pending):
            if gap.lower() in query.lower() or query.lower() in gap.lower():
                self._ms.protocol.mark_gap_filled(gap)
