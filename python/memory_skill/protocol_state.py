"""ProtocolState — owns the active-learning protocol state.

The three fields that used to live on ``MemorySystem`` as magic attributes
(``_classify_pending`` / ``_pending_gaps`` / ``_mission_pending_check``) now
live here, with the write operations that mutate them. ProtocolGate reads
these through the state; tools/skill_writer mutate them through the API
methods instead of poking private fields. MemorySystem keeps a single
``protocol`` reference.

The state machine (see ProtocolGate for enforcement):
  classify(mission)         -> mission_pending_check=True, pending_gaps={gaps}
  classify(other)           -> mission_pending_check=False, pending_gaps={}
  check_skill() / search()  -> mission_pending_check=False
  mark_weave(user_message)  -> classify_pending=user_message (next turn should classify)

Since 0.4.1 the gate is *advisory*: nothing here blocks weave() or tool
calls. The state only drives a one-line reminder in the injected context,
so a missed classification can never stall a conversation.

``classify_pending_since`` makes the reminder expire: the MCP server process
is long-lived and shared by every DSH session, so a pending item left behind
by an old session must not keep nagging a brand-new one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

# A pending classification older than this is treated as abandoned.
DEFAULT_PENDING_TTL_SECONDS = 15 * 60


@dataclass
class ProtocolState:
    """Protocol state owned by a single module, mutated only through its API."""

    classify_pending: str | None = None
    pending_gaps: set = field(default_factory=set)
    mission_pending_check: bool = False
    classify_pending_since: float = 0.0

    def mark_classified(self, category: str, gaps=None) -> None:
        """Record a classification. A mission arms the mission reminder."""
        self.classify_pending = None
        self.classify_pending_since = 0.0
        if category == "mission":
            self.mission_pending_check = True
            self.pending_gaps = {str(g).strip() for g in (gaps or []) if str(g).strip()}
        else:
            self.mission_pending_check = False
            self.pending_gaps.clear()

    def mark_skill_checked(self) -> None:
        """A mission has consulted existing skills (check_skill or search)."""
        self.mission_pending_check = False

    def mark_gap_filled(self, skill_name: str) -> None:
        """teach_skill succeeded: drop the skill from pending gaps."""
        if skill_name in self.pending_gaps:
            self.pending_gaps.discard(skill_name)

    def mark_weave(self, user_message: str) -> None:
        """Record that this turn was woven (next turn may classify it)."""
        self.classify_pending = user_message
        self.classify_pending_since = time.time()

    def pending_age(self) -> float:
        """Seconds since the pending classification was recorded (0 if none)."""
        if not self.classify_pending or not self.classify_pending_since:
            return 0.0
        return max(0.0, time.time() - self.classify_pending_since)

    def expire_stale(self, ttl_seconds: float = DEFAULT_PENDING_TTL_SECONDS) -> bool:
        """Drop an abandoned pending item; returns True when it was cleared.

        A long-idle process (or a pending left by another conversation) must
        not keep reminding the current one.
        """
        if self.classify_pending and self.pending_age() > ttl_seconds:
            self.classify_pending = None
            self.classify_pending_since = 0.0
            return True
        return False

    def open_gaps(self) -> set:
        """Snapshot of the still-unfilled skill gaps."""
        return set(self.pending_gaps)
