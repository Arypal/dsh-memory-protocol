"""Structured ingest helpers — pure storage, no LLM.

All LLM-based decisions (classification, title generation, conclusion
extraction, mission decomposition) are the main agent's responsibility.
This module only writes structured entries to the learned store.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime as dt

_logger = logging.getLogger(__name__)


def _ingest_structured(ms, title: str, content: str, category: str,
                       source_urls: list[str] | None = None) -> object:
    from memory_skill.contracts import DialogueTurn
    now = dt.now(UTC)
    turn = DialogueTurn(
        id=f"{category}_{now:%Y%m%d_%H%M%S}_{hash(content) & 0xFFFF:04x}",
        role="system",
        content=content,
        timestamp=now,
        saw_index=0,
    )
    extra = {}
    if source_urls:
        extra["source_urls"] = source_urls
    result = ms.ingestor.ingest_dialogue(turn, category=category, extra_metadata=extra)
    if ms.tree:
        try:
            if category == "skill":
                from memory_skill.tree_classifier import classify_skill_path
                path = classify_skill_path(title)
                if path:
                    ms.tree.add_skill_node(path)
            else:
                branch_short = {"pref": "pref", "pers": "pers",
                                "mission": "task", "conclusion": "task"}.get(category)
                if branch_short:
                    root = "user" if category == "pref" else "assistant"
                    ms.tree.add_node(content=content, memory_id=turn.id,
                                     root=root, branch=branch_short, timestamp=now)
        except Exception:
            pass
    return result


def ingest_skill_ex(ms, title: str, content: str,
                    source_urls: list[str] | None = None) -> object:
    return _ingest_structured(ms, title, content, "skill", source_urls=source_urls)


def ingest_mission_ex(ms, title: str, content: str) -> object:
    return _ingest_structured(ms, title, content, "mission")


def ingest_pref(ms, key: str, value: str) -> object:
    return _ingest_structured(ms, key, f"{key}: {value}", "pref")


def ingest_pers(ms, trait: str) -> object:
    existing = ms.retriever.retrieve("all", limit=10, filters={"category": "pers"})
    cards = [e for e in existing.entries if e.content.startswith("# ")]
    if cards:
        current = max(cards, key=lambda e: len(e.content)).content
        if f"- {trait}" not in current:
            dup = ms.learned_store.find_duplicate(ms.embedder.embed(current), threshold=0.85)
            if dup:
                return None
            updated = current.rstrip() + f"\n- {trait}\n"
            ms.learned_store.update(cards[0].id, updated)
            return cards[0]
    return _ingest_structured(ms, trait, f"- {trait}", "pers")
