"""Protocol Gate — advisory reminders before/after weave().

Extracted from _compose.MemorySystem.weave() so the gate logic is testable
in isolation and the weave() method is pure context assembly.

0.4.1 behaviour change: the gate **never raises**. It returns a one-line
notice that the weaver renders as a soft ``[协议提醒]`` block. Rationale:
the previous hard gate turned a context-injection call into a failure that
blocked every non-memory tool, so a stale/unclassified previous turn (which
happens in normal use, and across DSH sessions sharing one MCP process) cost
the agent 3–4 extra tool calls before it could answer anything. Classification
and active learning stay available — they are simply no longer life-or-death.

The gate reads and writes protocol state through a single ``ProtocolState``
owner (``system.protocol``), never through magic attributes on MemorySystem.
Tools and the skill writer mutate the same state via the API methods on
``ProtocolState``.

Reminder rules (checked in order, first match wins):
  1. classify-pending — the previous turn was not classified (expires after TTL).
  2. mission-check     — a mission has not consulted existing skills yet.
  3. gap-pending       — a mission has unfulfilled skill gaps.
"""

from __future__ import annotations


class ProtocolGate:
    """Produces advisory protocol notices; never raises."""

    def __init__(self, system):
        self._state = system.protocol

    def check(self, user_message: str) -> str | None:
        """Return an advisory notice for the current turn, or None."""
        # A pending item inherited from a long-idle process / other session
        # is stale — drop it instead of nagging the new conversation.
        self._state.expire_stale()

        if self._state.classify_pending and user_message != self._state.classify_pending:
            return (
                "上一轮消息还没分类。方便的话调一次 memory_classify"
                "(chat|skill|mission|pref|pers)；不分类不影响继续对话。"
            )

        if self._state.mission_pending_check:
            return (
                "当前任务被标记为 mission，但还没查过已有技能。"
                "可以先 memory_check_skill 或 memory_search 去重；"
                "不查也不影响继续对话。"
            )

        if self._state.pending_gaps:
            gaps = ", ".join(sorted(self._state.pending_gaps)[:8])
            return f"下列技能还没补齐，可考虑 websearch → memory_teach_skill：{gaps}"

        return None

    def after_weave(self, user_message: str) -> None:
        """Record that this turn was woven (sets the next-turn reminder)."""
        self._state.mark_weave(user_message)
