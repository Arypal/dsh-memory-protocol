"""Skill Registry — answers "do we already know this skill?"

The active-learning loop's first step: before searching the web for a
topic, the agent checks whether the skill is already in memory.  This
module performs that check by querying the ``skill`` category with the
semantic retriever and applying the same answerability logic the
capability registry uses — but scoped to skill entries only.

Returns a tri-state verdict:
  - ``known``    — a skill entry matches strongly (sem ≥ strong threshold)
  - ``partial``  — related skill entries exist but none is a strong match
  - ``unknown``  — nothing relevant in the skill category

The agent uses the verdict to decide whether to learn (unknown / partial)
or to reuse / update (known / partial) the existing skill memory.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("memory_skill.skill_registry")


# Answerability thresholds — single source of truth is capability_registry
# (KNOWN-ISSUES #1). Imported here so a calibration change lands once.
from memory_skill.capability_registry import _SEM_CORROBORATED, _SEM_STRONG

_PARTIAL_FLOOR = 0.45


class SkillRegistry:
    """Check whether a named skill is already known to memory."""

    def __init__(self, retriever, capability_registry=None):
        """``retriever`` provides ``retrieve(query, filters=...)``;
        ``capability_registry`` (optional) adds answerability logic.
        """
        self._retriever = retriever
        self._capability = capability_registry

    def check_skill(self, name: str) -> dict:
        """Return a verdict dict for *name*:

        ``{"status": "known"|"partial"|"unknown", "confidence": float,
           "matches": [ {entry_id, content_head, semantic_score, weight} ]}``
        """
        name = (name or "").strip()
        if not name:
            return {"status": "unknown", "confidence": 0.0, "matches": []}

        try:
            result = self._retriever.retrieve(
                name, limit=5, filters={"category": "skill"},
            )
            entries = list(result.entries)
        except Exception as exc:
            logger.warning("Skill check retrieval failed: %s", exc)
            entries = []

        if not entries:
            return {"status": "unknown", "confidence": 0.0, "matches": []}

        # Top semantic score among skill entries
        top_score = max(e.semantic_score or 0.0 for e in entries)
        top = max(entries, key=lambda e: (e.semantic_score or 0.0, len(e.content)))

        matches = [
            {
                "entry_id": e.id,
                "content_head": (e.content or "")[:120],
                "semantic_score": e.semantic_score or 0.0,
                "weight": e.weight,
            }
            for e in entries[:3]
        ]

        # Known: strong semantic match, or corroborated by token overlap
        if top_score >= _SEM_STRONG:
            return {"status": "known", "confidence": round(top_score, 3), "matches": matches}

        if top_score >= _SEM_CORROBORATED and self._capability is not None:
            can, _conf = self._capability.can_answer(name)
            if can:
                return {"status": "known", "confidence": round(top_score, 3), "matches": matches}

        if top_score >= _PARTIAL_FLOOR:
            return {"status": "partial", "confidence": round(top_score, 3), "matches": matches}

        return {"status": "unknown", "confidence": round(top_score, 3), "matches": matches}
