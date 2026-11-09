"""
Memory Skill — Reflect: idle-time memory review and consolidation.

The Reflect module is the "stop and think" step.  Unlike weave (which
runs before every agent turn) or evolve (which runs after feedback),
reflect is a lazy, idle-time process that reviews recent dialogue,
synthesises observations, adjusts weights, and surfaces notable
patterns that the agent might want to mention.

Design:
  - Triggered by Room when conversation naturally pauses
  - Non-blocking, no impact on agent response latency
  - Returns a human-readable summary of what it found
  - Stateful: tracks when it last ran to avoid redundant work
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

from memory_skill.contracts import utcnow

_logger = logging.getLogger(__name__)
_MIN_DIALOGUE_FOR_REFLECT: int = 10


@dataclass
class ReflectReport:
    """What the idle-time reflection discovered."""

    new_observations: int = 0
    evolved_count: int = 0
    nudge_items: list[str] = field(default_factory=list)
    ran_at: datetime = field(default_factory=utcnow)

    @property
    def is_empty(self) -> bool:
        return (
            self.new_observations == 0
            and self.evolved_count == 0
            and not self.nudge_items
        )

    def summary(self) -> str:
        parts: list[str] = []
        if self.new_observations:
            parts.append(f"学到了 {self.new_observations} 条新知识")
        if self.evolved_count:
            parts.append(f"调整了 {self.evolved_count} 条记忆的权重")
        if self.nudge_items:
            parts.append(f"有 {len(self.nudge_items)} 条重要记忆等待被提起")
        return "; ".join(parts) if parts else "无新发现"


# ── Constants ────────────────────────────────────────

_NUDGE_WEIGHT_THRESHOLD: float = 0.80
_NUDGE_MAX_ITEMS: int = 5
_MIN_DIALOGUE_FOR_REFLECT: int = 10  # don't reflect on tiny conversations


def reflect(dialogue_store, learned_store):
    """Run idle-time memory reflection.

    Called when conversation naturally pauses.  Runs consolidation
    and nudge detection in sequence.  Returns a summary
    of what was discovered.
    """
    report = ReflectReport()

    # ── Guard: don't reflect on trivial amounts of dialogue ──
    try:
        dialogue_count = dialogue_store.count()
    except Exception as e:
        _logger.debug("Dialogue count failed: %s", e)
        dialogue_count = 0
    if dialogue_count < _MIN_DIALOGUE_FOR_REFLECT:
        return report

    # ── Step 1: Consolidate observations from recent dialogue ──
    try:
        from memory_skill.observation import ObservationConsolidator
        report.new_observations = ObservationConsolidator(
            dialogue_store, learned_store,
        ).run()
    except Exception as e:
        _logger.debug("Consolidation failed: %s", e)

    # ── Step 2: Nudge detection ──
    try:
        entries = learned_store.search(
            "",
            limit=_NUDGE_MAX_ITEMS * 3,
            filters={"weight": {"$gte": _NUDGE_WEIGHT_THRESHOLD}},
        )
        # Filter: only truly high-weight, non-system entries
        high = [
            e for e in entries
            if e.weight >= _NUDGE_WEIGHT_THRESHOLD and not e.is_system
        ]
        # Sort by weight descending
        high.sort(key=lambda e: e.weight, reverse=True)
        for e in high[:_NUDGE_MAX_ITEMS]:
            snippet = e.content[:120]
            if e.metadata and e.metadata.get("partner"):
                snippet = f"[与{e.metadata['partner']}] {snippet}"
            report.nudge_items.append(snippet)
    except Exception as e:
        _logger.debug("Nudge detection failed: %s", e)
        pass

    return report
