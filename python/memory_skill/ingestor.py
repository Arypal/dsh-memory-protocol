"""
Ingestor — write pipeline orchestrator for the Memory Skill.

Routes dialogue turns and screen observations to the appropriate stores:
- SawRingBuffer (always, for short-term context)
- DialogueStore (dialogue turns)
- LearnedStore (embed + store with entity tags)

Dialogue pipeline:
  1. Store in SawRingBuffer (always)
  2. Store in DialogueStore (always)
  3. Entity extraction via regex → tag MemoryEntry
  4. Embed + store in LearnedStore (always)

Screen pipeline:
  1. NoiseFilter check → if noise: skip (only SawRingBuffer)
  2. If kept: check for errors → tag as high importance → LearnedStore with high weight
  3. If kept: normal screen → LearnedStore with low weight
"""

from __future__ import annotations

import re
from memory_skill.classification_helpers import _looks_like_question, _ScreenNoiseFilter
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from memory_skill.contracts import (
    DialogueStoreProtocol,
    DialogueTurn,
    EmbedderProtocol,
    IngestReceipt,
    LearnedStoreProtocol,
    MemoryEntry,
    SawBufferProtocol,
    SawEntry,
    TreeManagerProtocol,
)
import numpy as np

if TYPE_CHECKING:
    from memory_skill.contracts import MemorySkillConfig


# ═══════════════════════════════════════════════════════════════════════════════
# Regex-based entity extraction (v1 — simple, no spaCy)
# ═══════════════════════════════════════════════════════════════════════════════

# Proper names: capitalized words (at least 2 letters to avoid single-letter noise)
_PATTERN_PROPER: re.Pattern[str] = re.compile(r"\b[A-Z][a-z]{1,}\b")

# Common English words that look like proper names but aren't
_STOP_WORDS: frozenset[str] = frozenset({
    "I", "A", "An", "The", "It", "He", "She", "We", "They", "You",
    "This", "That", "These", "Those", "My", "Your", "His", "Her",
    "Our", "Their", "Its", "No", "Yes", "Not", "So", "If", "Or",
    "And", "But", "For", "Nor", "Yet", "To", "In", "On", "At",
    "By", "From", "With", "Is", "Be", "Are", "Was", "Were", "Been",
    "Do", "Does", "Did", "Has", "Have", "Had", "Can", "Could",
    "Will", "Would", "Shall", "Should", "May", "Might", "Must",
    "What", "When", "Where", "Which", "Who", "Whom", "Whose",
    "Why", "How", "All", "Each", "Every", "Both", "Few", "More",
    "Most", "Other", "Some", "Such", "Only", "Own", "Same",
    "Then", "Than", "Too", "Very", "Just", "Now", "Here", "There",
    "Also", "Even", "Still", "Again", "Already", "Always", "Never",
    "Often", "Sometimes", "Usually", "One", "Two", "Three", "First",
    "Last", "New", "Old", "Good", "Great", "Big", "Small", "Large",
    "High", "Low", "Long", "Short", "Right", "Left", "Best", "Better",
    "Today", "Tomorrow", "Yesterday", "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday", "January", "February",
    "March", "April", "June", "July", "August", "September",
    "October", "November", "December",
})


def _extract_entities(text: str) -> list[str]:
    """Extract named entity candidates via simple regex (v1).
    
    Matches capitalized words that are not common English stop words.
    Returns deduplicated, case-preserved list.
    """
    if not text:
        return []

    matches = _PATTERN_PROPER.findall(text)
    seen: set[str] = set()
    entities: list[str] = []
    for match in matches:
        if match not in _STOP_WORDS and match not in seen:
            seen.add(match)
            entities.append(match)
    return entities


# ── 情绪加权：arousal 代理值 ──────────────────────────────────────────────
# 情绪唤醒（0-1）让记忆衰减更慢。取值优先级：心情系统状态（2 小时内有效）
# → 文本情绪词/标点密度。没有 QQ 心情状态时纯文本规则也能工作。

_EMO_WORDS: tuple[str, ...] = (
    "开心", "高兴", "快乐", "幸福", "难过", "伤心", "生气", "愤怒",
    "崩溃", "焦虑", "害怕", "惊喜", "感动", "失望", "遗憾", "委屈",
    "孤独", "兴奋", "激动", "后悔", "心疼", "烦死", "累死", "哭了",
    "笑死", "好爱", "讨厌", "气死", "吓人", "紧张", "难受", "爽",
)


def _arousal_score(text: str) -> float:
    """估算一条记忆的情绪唤醒强度（0-1）。"""
    if not text:
        return 0.0
    score = 0.0

    # 1) 心情系统状态（QQ 通道写入，桌面可共享；2 小时内才可信）
    try:
        import json
        import os
        import time

        path = os.environ.get("MEMORY_MOOD_STATE", "")
        if path and os.path.exists(path) and (time.time() - os.path.getmtime(path)) <= 7200:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            mood = float(data.get("mood", 50))
            score = max(score, min(1.0, abs(mood - 50.0) / 50.0))
    except Exception:
        pass

    # 2) 文本情绪词密度
    hits = sum(1 for w in _EMO_WORDS if w in text)
    score = max(score, min(1.0, hits / 3.0))

    # 3) 强标点（连发感叹号/问号）
    if text.count("!") + text.count("！") >= 2:
        score = max(score, 0.4)
    if text.count("?") + text.count("？") >= 3:
        score = max(score, 0.3)

    return round(score, 3)


# ═══════════════════════════════════════════════════════════════════════════════
# Text chunking (V3 — mitigates 512-token embedding truncation with bge-large)
# ═══════════════════════════════════════════════════════════════════════════════

# Character-based token approximation: ~4 chars per English word-token.
_CHARS_PER_TOKEN: int = 4



# Sentence-boundary patterns to prefer when splitting chunks.
# ═══════════════════════════════════════════════════════════════════════════════
# Duplicate-ingest window
# ═══════════════════════════════════════════════════════════════════════════════

# A retried ingest arrives seconds after the first with a fresh timestamp, so
# the (role, content, timestamp) UNIQUE constraint never fires. Identical
# content inside this window is a duplicate trigger, not a real re-utterance.
_DUP_WINDOW_SEC: float = 120.0


# ═══════════════════════════════════════════════════════════════════════════════
# Ingestor
# ═══════════════════════════════════════════════════════════════════════════════


class Ingestor:
    """Write pipeline orchestrator.

    Routes dialogue and screen observations through stores with noise
    filtering and entity extraction.

    Parameters
    ----------
    config: MemorySkillConfig
    saw_buffer: SawBufferProtocol
    dialogue_store: DialogueStoreProtocol
    learned_store: LearnedStoreProtocol
    embedder: EmbedderProtocol
    tree: TreeManagerProtocol | None
    """

    def __init__(
        self,
        config: MemorySkillConfig,
        saw_buffer: SawBufferProtocol,
        dialogue_store: DialogueStoreProtocol,
        learned_store: LearnedStoreProtocol,
        embedder: EmbedderProtocol,
        tree: TreeManagerProtocol | None = None,
        learning_queue=None,
    ) -> None:
        self._config = config
        self._saw_buffer = saw_buffer
        self._dialogue_store = dialogue_store
        self._learned_store = learned_store
        self._noise_filter = _ScreenNoiseFilter()
        self._embedder = embedder
        self._tree = tree
        self._learning_queue = learning_queue

        # Monotonic heartbeat counter for SawRingBuffer entries
        self._heartbeat: int = 0

        # Last-kept screen frame for noise filter comparison
        self._last_kept_screen: str | None = None

    # ── Public API ────────────────────────────────────────────────────────

    def ingest_dialogue(self, turn: DialogueTurn, category: str | None = None,
                         extra_metadata: dict | None = None,
                         scope: str = "web") -> IngestReceipt:
        """Ingest a dialogue turn — all content goes to both stores.

        1. Store in SawRingBuffer (always)
        2. Store in DialogueStore (always)
        3. Embed → semantic dedup → merge (weight+0.05) or insert (weight=0.5)
        4. Store in LearnedStore (always)

        Returns an ``IngestReceipt`` describing what happened (see ADR-0001):
        the stored entry id, whether it was deduped, the resulting weight,
        and per-stage status. Gap detection is no longer triggered here —
        the ingest pipeline (MemorySystem.ingest) drives it under ``enrich``.
        """
        now = datetime.now()

        # 0. Idempotency guard — must run before any store writes.
        if self._is_recent_duplicate(turn.content):
            return IngestReceipt(
                entry_id=f"dialogue:{turn.id}",
                deduped=True,
                weight=0.5,
                staged={},
                timestamp=now,
            )

        # 1. Store in SawRingBuffer (always)
        self._saw_buffer.put(self._make_saw_entry(turn.content, now))

        # 2. Store in DialogueStore (always)
        self._dialogue_store.insert(turn, scope=scope)

        # 3. Entity extraction
        entities = _extract_entities(turn.content)

        # 4. Embed + semantic deduplication
        # 两段式判重：余弦 0.85 只负责"召回候选"，再由 difflib 精确判定文本
        # 重合度（>= 0.90 才算同一条）。单用余弦会把"同一话题"误判成同一条，
        # 旧实现还会顺手覆盖掉原有内容——身份类事实就是这样被顶掉的。
        embedding = self._embedder.embed(turn.content)
        dup = self._learned_store.find_duplicate(embedding, threshold=0.85)
        if dup:
            import difflib
            _ratio = difflib.SequenceMatcher(
                None,
                (dup.get("content") or "").strip(),
                turn.content.strip(),
            ).ratio()
            if _ratio < 0.90:
                dup = None

        entry_id = f"dialogue:{turn.id}"
        meta = {
            "role": turn.role,
            "source": "dialogue",
            "turn_id": turn.id,
            "saw_index": turn.saw_index,
            "scope": scope,
        }
        # 情绪加权：记录情绪唤醒强度，供检索的时间衰减调制（强度 > 0.15 才写）
        _arousal = _arousal_score(turn.content)
        if _arousal >= 0.15:
            meta["arousal"] = _arousal
        if turn.partner:
            meta["partner"] = turn.partner
        if extra_metadata:
            meta.update(extra_metadata)

        if dup:
            # Cross-category merge would clobber the structured category
            # (taught skill absorbed into a dialogue fragment becomes
            # invisible to category-scoped check_skill) — dedup only
            # within the same category.
            dup_cat = dup.get("category") or self._config.namespace
            target_cat = category or self._config.namespace
            if dup_cat != target_cat:
                dup = None

        if dup:
            try:
                # Merge by BOOSTING ONLY — never rewrite the stored document.
                # Rewriting content under the dedup target's id made the id point
                # at different text in Chroma than in dialogue_turns, silently
                # destroying facts (the "大橘" regression).
                new_weight = self._learned_store.boost_weight(
                    dup["entry_id"], delta=0.05
                )
                deduped = True
                entry_id = dup["entry_id"]
                weight = new_weight
            except Exception:
                # dup referenced a stale id (entry deleted from chroma but
                # still indexed elsewhere) — fall through to a fresh write.
                import logging
                logging.getLogger(__name__).warning(
                    "Dedup target %s missing; writing fresh entry", dup["entry_id"])
                dup = None
        if not dup:
            # New entry with uniform weight
            entry = MemoryEntry(
                id=entry_id,
                content=turn.content,
                created_at=now,
                updated_at=now,
                weight=0.5,
                category=category or self._config.namespace,
                tags=entities,
                metadata=meta,
            )
            self._learned_store.insert(entry)
            deduped = False
            weight = 0.5

        return IngestReceipt(
            entry_id=entry_id,
            deduped=deduped,
            weight=weight,
            staged={},
            timestamp=now,
        )

    def ingest_screen(self, frame_text: str) -> None:
        """Ingest a screen observation frame — noise-filter then route.
        
        1. Always store in SawRingBuffer
        2. NoiseFilter check:
           - If noise: only SawRingBuffer, skip LearnedStore
           - If kept and error: LearnedStore with high weight (0.9)
           - If kept and normal: LearnedStore with low weight (0.3)
        """
        now = datetime.now(UTC)

        # 1. Always store in SawRingBuffer
        self._saw_buffer.put(self._make_saw_entry(frame_text, now))

        # 2. Noise filter check
        if self._last_kept_screen is not None:
            if not self._noise_filter.should_keep(frame_text, self._last_kept_screen):
                return MemoryEnvelope(
                    type="saw", entries=[], truncated=False,
                    total_candidates=0, timestamp=now,
                )

        # Update last-kept screen (kept frames only)
        self._last_kept_screen = frame_text

        # Determine weight based on error detection
        has_error = self._noise_filter.is_error_frame(frame_text)
        weight = 0.9 if has_error else 0.3

        entry_id = f"screen:{self._heartbeat}"
        entry = MemoryEntry(
            id=entry_id,
            content=frame_text,
            created_at=now,
            updated_at=now,
            weight=weight,
            category=self._config.namespace,
            tags=["error"] if has_error else ["observation"],
            metadata={
                "source": "screen",
                "heartbeat_index": self._heartbeat,
                "has_error": has_error,
            },
        )
        self._learned_store.insert(entry)

    @property
    def gaps(self) -> list:
        """Open learning-queue items (skills/missions awaiting agent action)."""
        if self._learning_queue is None:
            return []
        try:
            return self._learning_queue.open_items()
        except Exception:
            return []

    # ── Private helpers ───────────────────────────────────────────────────

    def _is_recent_duplicate(self, content: str) -> bool:
        """Return True when identical content was stored inside the window.

        The dialogue table dedups on ``(role, content, timestamp)``, but a
        retried ingest arrives seconds later with a *new* timestamp, so that
        UNIQUE constraint never fires and the same sentence lands twice
        (observed: 3 copies of one turn 8s/3s apart). Query by content plus a
        time window instead.

        Fails open: any storage error means "not a duplicate" so a broken
        check can never silently swallow a real turn.
        """
        if not content:
            return False
        try:
            import sqlite3
            import time as _time

            conn = sqlite3.connect(self._config.db_path, timeout=1.0)
            try:
                row = conn.execute(
                    "SELECT 1 FROM dialogue_turns "
                    "WHERE content = ? AND timestamp >= ? LIMIT 1",
                    (content, _time.time() - _DUP_WINDOW_SEC),
                ).fetchone()
                return row is not None
            finally:
                conn.close()
        except Exception:
            return False

    def _make_saw_entry(self, content: str, timestamp: datetime) -> SawEntry:
        """Create a SawEntry with a monotonic heartbeat_index."""
        self._heartbeat += 1
        return SawEntry(
            heartbeat_index=self._heartbeat,
            content=content,
            timestamp=timestamp,
        )
