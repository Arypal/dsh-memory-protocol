"""
Memory Skill — Observation Consolidation.

Automatically synthesises higher-level knowledge from raw dialogue turns,
without requiring a cloud LLM.

V6: Multi-dimension structured output — groups observations by type
(preferences, events, tech_stack, relationships, plans) with evidence.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import UTC

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────

_MIN_MENTIONS: int = 5
_MIN_SIGNALS: int = 2
_OBSERVATION_WEIGHT: float = 0.6


# ═══════════════════════════════════════════════════════
# ObservationConsolidator
# ═══════════════════════════════════════════════════════

class ObservationConsolidator:
    """Uses TextRank to discover conversation topics and synthesise observations."""

    def __init__(self, dialogue_store, learned_store,
                 min_mentions: int = _MIN_MENTIONS) -> None:
        self._dialogue_store = dialogue_store
        self._learned_store = learned_store
        self._min_mentions = min_mentions

    def _extract_keywords(self, texts: list[str], topk: int = 30) -> dict[str, int]:
        """TextRank keyword extraction from dialogue texts."""
        try:
            from textrank4zh import TextRank4Keyword
            all_text = "\n".join(texts)
            tr4w = TextRank4Keyword()
            tr4w.analyze(all_text, window=5, lower=True,
                          vertex_source="all_filters",
                          edge_source="no_stop_words")
            result = {}
            for item in tr4w.get_keywords(topk):
                word = item.word
                if len(word) >= 2 and not word.isdigit():
                    result[word] = int(item.weight * 100)
            return result
        except ImportError:
            return {}

    def run(self) -> int:
        turns = self._dialogue_store.get_recent(200)
        if len(turns) < self._min_mentions:
            return 0

        texts = [t.content for t in turns]
        keywords = self._extract_keywords(texts)
        if not keywords:
            return 0

        # Filter keywords that appear frequently enough
        all_text = "\n".join(t.content.lower() for t in turns)
        frequent = {kw for kw, _ in keywords.items() if all_text.count(kw) >= self._min_mentions}

        # Group evidence by keyword
        kw_evidence: dict[str, list[str]] = defaultdict(list)
        for turn in turns:
            for kw in frequent:
                if kw in turn.content:
                    kw_evidence[kw].append(turn.content[:200])

        # Generate observations
        generated = 0
        for kw, evidence in kw_evidence.items():
            if len(evidence) >= _MIN_SIGNALS:
                content = _build_observation(kw, evidence)
                self._store_observation(kw, content)
                generated += 1

        # Cross-keyword summary
        if len(kw_evidence) >= 3:
            summary = _build_overview(kw_evidence, turns)
            if summary:
                self._store_observation("overview", summary)
                generated += 1

        return generated

    def _store_observation(self, keyword: str, content: str) -> None:
        from datetime import datetime

        from memory_skill.contracts import MemoryEntry

        entry = MemoryEntry(
            id=f"obs:{keyword.replace(' ', '_')}",
            content=content,
            category="observation",
            weight=_OBSERVATION_WEIGHT,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            tags=["consolidated", "auto_generated"],
            metadata={"keyword": keyword, "source": "textrank"},
            is_system=True,
        )
        try:
            self._learned_store.insert(entry)
        except Exception as e:
            logger.debug("Observation synthesis failed: %s", e)


# ── Content builders ──────────────────────────────────

def _build_observation(keyword: str, evidence: list[str]) -> str:
    snippets = [e[:100] for e in evidence[:3]]
    parts = [f"关于「{keyword}」的对话:"]
    for s in snippets:
        parts.append(f"  · {s}")
    return "\n".join(parts)


def _build_overview(kw_evidence: dict, turns) -> str:
    parts = ["对话概况:"]
    top_kw = sorted(kw_evidence.items(), key=lambda x: len(x[1]), reverse=True)[:5]
    for kw, evidence in top_kw:
        parts.append(f"  {kw}: 提到{len(evidence)}次")
    last = [t.content[:60] for t in turns[-3:]]
    if last:
        parts.append(f"  最近: {' | '.join(last)}")
    return "\n".join(parts) if len(parts) > 1 else ""
