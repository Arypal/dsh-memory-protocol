"""Memory Skill adapter — bridges Room Agent ↔ memory-skill.

Implements ``MemoryProtocol`` via direct Python import (no subprocess).
Auto-weaves context before each agent turn, auto-ingests after each response.

V2: Token-aware context cap, auto outcome detection, needs_second_pass,
    dynamic partner routing, scene_summary support.

Usage::

    from memory_skill.room_adapter import MemorySkillAdapter

    memory = MemorySkillAdapter(agent_name="my_agent", max_context_chars=800)
    agent = Agent(name="MyAgent", personality="...", memory=memory)
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from room.message import Message
from room.protocols import MemoryProtocol

if TYPE_CHECKING:
    pass

logger = logging.getLogger("room.memory")


class MemorySkillAdapter(MemoryProtocol):
    """Wraps memory-skill via direct Python import for Room Agent integration.

    Parameters
    ----------
    agent_name:
        The agent's identity (e.g. ``"my_agent"``).  Used for namespace
        isolation — all memories are scoped to this agent.
    partner:
        Default partner for conversations (e.g. ``"user"``).
        Can be overridden per-message via dynamic routing.
    db_path:
        Path to the memory-skill SQLite database.
    max_context_chars:
        Max characters for the woven context block (default 3000, ~750 tokens).
        Raised from 800 to accommodate tier2 facts (4×~600c) + emotion + nudge.
    """

    def __init__(
        self,
        agent_name: str = "",
        partner: str = "user",
        db_path: str = os.getenv("MEMORY_DB_PATH", "opencode_memory.db"),
        max_context_chars: int = 3000,
        display_name: str = "",
        memory_depth: str = "full",  # "off" | "full" (tier2 only, tier1 removed — redundant with history)
    ) -> None:
        self.agent_name = agent_name  # namespace key for ChromaDB filtering
        self.display_name = display_name or agent_name  # weaver label
        self._memory_depth = memory_depth
        self._last_injected_unit: str = ""  # avoid repeating same memory fragment
        self.partner = partner
        self.db_path = db_path
        self._max_context_chars = max_context_chars
        self._messages: list[Message] = []
        self._pending_second_pass: bool = False
        self._last_retrieval: dict | None = None
        self._last_clean = datetime.fromtimestamp(0, tz=UTC)

        # ── Lazy-load memory skill on first use ─────────
        self._skill = None

    # ── Internal ────────────────────────────────────────

    def _ensure_skill(self):
        """Lazy-load MemorySkill — only import on first use."""
        if self._skill is not None:
            return
        import site
        site.addsitedir(site.getusersitepackages())
        _mem_path = os.getenv("MEMORY_PROJECT_PATH", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.path.insert(0, _mem_path)
        from memory_skill import MemorySkill, MemorySkillConfig

        self._skill = MemorySkill(MemorySkillConfig(
            db_path=self.db_path,
            agent_name=self.agent_name,
        ))
        mode = self._skill.ensure_embedder_loaded()
        logger.info("Memory skill loaded for agent=%s db=%s mode=%s",
                     self.agent_name, self.db_path, mode)

    # ── Dynamic partner resolution ─────────────────────

    def _resolve_partner(self, message: Message | None = None,
                         explicit: str | None = None) -> str:
        """Resolve partner identity: explicit > message.target > default."""
        if explicit:
            return explicit
        if message and message.target and message.target != self.agent_name:
            return message.target
        return self.partner

    # ── MemoryProtocol ──────────────────────────────────

    async def add(self, message: Message, partner: str | None = None) -> None:
        """Store a message in memory, with auto-maintenance."""
        self._messages.append(message)
        try:
            self._ensure_skill()
            from memory_skill import DialogueTurn

            # Use Message.role (Role enum) to determine speaker role
            role = "assistant" if message.role.value in ("agent", "assistant") else "user"
            p = self._resolve_partner(message, partner)
            turn = DialogueTurn(
                id=f"room_{datetime.now(UTC).timestamp():.0f}",
                role=role,
                content=message.content,
                timestamp=datetime.now(UTC),
                partner=p,
            )
            self._skill.ingest(turn)

            # Auto-maintenance: trigger periodically (with cooldown)
            n = self._skill.count_turns()
            now = datetime.now(UTC)
            if n % 10 == 0:
                self._skill.consolidate()
            if n % 50 == 0 and (now - self._last_clean).total_seconds() > 300:
                self._skill.clean()
                self._last_clean = now
        except Exception as e:
            logger.warning("Ingest failed: %s", e)

    async def get(self, limit: int = 20) -> list[Message]:
        """Get recent in-memory messages."""
        return self._messages[-limit:]

    async def clear(self) -> None:
        """Clear in-memory history."""
        self._messages.clear()

    # ── Extended API (beyond MemoryProtocol) ────────────

    async def weave(
        self,
        message: str,
        partner: str | None = None,
        scene_summary: str = "",
    ) -> str:
        """Assemble layered memory context + defer feedback.

        Returns the prompt block.  Sets ``needs_second_pass`` if the
        weaver detected high-weight memories that warrant a follow-up
        injection after the agent's next response.
        """
        try:
            self._ensure_skill()

            # ── Memory depth gate ──
            if self._memory_depth == "off":
                return ""

            p = partner or self.partner
            ctx = self._skill.weave(
                user_message=message,
                partner=p,
                scene_summary=scene_summary,
            )
            block = ctx.to_prompt_block()

            # ── Rotate: skip if same fragment as last turn ──
            if block and "记忆片段" in block:
                first_unit = block.split("[记忆片段]")[1].split("\n")[0].strip() if "[记忆片段]" in block else ""
                if first_unit and first_unit == self._last_injected_unit:
                    # Same memory — skip to let other memories surface
                    block = block.replace(
                        block[block.index("[记忆片段]"):].strip(), "").strip()
                else:
                    self._last_injected_unit = first_unit

            # ── Display name swap: namespace → readable label ──
            if self.display_name != self.agent_name:
                block = block.replace(f"{self.agent_name}: ", f"{self.display_name}: ")

            # ── Token-aware context cap ──
            if block and len(block) > self._max_context_chars:
                block = block[:self._max_context_chars].rsplit("\n", 1)[0] + "\n..."

            # ── Needs second pass tracking ──
            if ctx.needs_second_pass:
                self._pending_second_pass = True
                if block:
                    block += "\n\n（你想起了更多相关的事...）"

            # ── Defer feedback: store retrieval info for after agent responds ──
            try:
                envelope = self._skill.retrieve(message, limit=5, partner=p)
                entries = envelope.entries[:3]
                if entries:
                    self._last_retrieval = {
                        "query": message,
                        "mem_ids": [e.id for e in entries if e.id],
                        "results": [e.content[:200] for e in entries],
                    }
            except Exception:
                self._last_retrieval = None

            return block if block else ""
        except Exception as e:
            logger.debug("Weave failed: %s", e)
            return ""

    async def feedback_after_response(self, final_response: str) -> str:
        """Call after agent responds to provide outcome feedback.

        Uses ``auto_detect_outcome`` from memory module to determine
        whether the retrieved memories were actually useful.
        """
        if not self._last_retrieval or not self._last_retrieval["mem_ids"]:
            return "ok"
        try:
            self._ensure_skill()
            from memory_skill.feedback import auto_detect_outcome

            outcome = auto_detect_outcome(
                query=self._last_retrieval["query"],
                search_results=[
                    {"id": mid, "content": content}
                    for mid, content in zip(
                        self._last_retrieval["mem_ids"],
                        self._last_retrieval["results"],
                    )
                ],
                final_response=final_response,
            )
            # Boost weights for cited memories
            for mid in self._last_retrieval["mem_ids"][:3]:
                try:
                    self._skill.boost_weight(mid)
                except Exception:
                    pass
            logger.debug("Feedback: %s for %d mems", outcome,
                         len(self._last_retrieval["mem_ids"]))
            self._last_retrieval = None
            return outcome
        except Exception as e:
            logger.debug("Feedback failed (fallback positive): %s", e)
            # Fallback: mark as positive
            if self._last_retrieval and self._last_retrieval["mem_ids"]:
                for mid in self._last_retrieval["mem_ids"][:3]:
                    try:
                        self._skill.boost_weight(mid)
                    except Exception:
                        pass
            self._last_retrieval = None
            return "positive"

    async def search(
        self,
        query: str,
        limit: int = 5,
        partner: str | None = None,
    ) -> str:
        """Search memories — the ONE tool the agent consciously uses."""
        try:
            self._ensure_skill()
            results = self._skill.retrieve(
                query,
                limit=limit,
                partner=partner or self.partner,
            )
            if not results.entries:
                return ""
            lines = []
            for e in results.entries[:limit]:
                p = e.metadata.get("partner", "") if e.metadata else ""
                tag = f"[与{p}] " if p else ""
                # Give the full memory, not just a snippet
                snippet = e.content if len(e.content) < 400 else e.content[:400] + "..."
                lines.append(f"{tag}{snippet}")
                # Add surrounding dialogue context if available
                ctx = self._context_around(e)
                if ctx:
                    lines.append(f"  （对话上下文: {ctx}）")
            return "\n".join(lines)
        except Exception:
            return ""

    def _context_around(self, entry) -> str:
        """Fetch 1-2 turns of surrounding dialogue for a memory entry."""
        try:
            turn_id = entry.metadata.get("turn_id", "") if entry.metadata else ""
            if not turn_id:
                return ""
            # Try to get nearby turns from dialogue store
            turn = self._skill.get_turn(turn_id)
            if not turn:
                return ""
            recent = self._skill.get_recent_turns(10)
            # Find the turn's position and grab ±1 neighbors
            target_idx = None
            for i, t in enumerate(recent):
                if t.id == turn_id:
                    target_idx = i
                    break
            if target_idx is None:
                return ""
            ctx_turns = recent[max(0, target_idx-1):min(len(recent), target_idx+2)]
            ctx_lines = []
            for t in ctx_turns:
                role = "用户" if t.role == "user" else "助手"
                ctx_lines.append(f"{role}: {t.content[:60]}")
            return " | ".join(ctx_lines)
        except Exception:
            return ""

    # ── Heartbeat / second pass ────────────────────────

    @property
    def needs_second_pass(self) -> bool:
        """True when high-weight memories warrant a follow-up injection."""
        return self._pending_second_pass

    def clear_second_pass(self) -> None:
        """Clear the second-pass flag after handling it."""
        self._pending_second_pass = False

    # ── Idle reflection ────────────────────────────────

    async def reflect(self) -> str:
        """Run idle-time memory reflection.  Returns a human-readable summary.

        Called by Room when conversation naturally pauses.  Runs
        consolidation, evolution, and nudge detection.
        """
        try:
            self._ensure_skill()
            report = self._skill.reflect()
            return report.summary()
        except Exception as e:
            return f"reflect: {e}"

    # ── Health ──────────────────────────────────────────

    def status(self) -> str:
        """Health check."""
        try:
            self._ensure_skill()
            h = self._skill.health()
            mode = h['embedder']['mode']
            if h.get("embedder", {}).get("degraded"):
                mode += " DEGRADED"
            return (
                f"mode={mode} "
                f"learned={h['learned_store']['entry_count']} "
                f"dialogue={self._skill.count_turns()}"
            )
        except Exception as e:
            return f"error: {e}"
