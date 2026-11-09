"""MCP tool definitions and handlers for Memory Skill.

Exposes 4 tools over the MCP JSON-RPC protocol:
  - ``memory_search``   — semantic search across dialogue + learned memory
  - ``memory_ingest``   — ingest a dialogue turn into the memory system
  - ``memory_status``   — health summary (entry counts, model status)
  - ``memory_feedback`` — record retrieval outcome + boost cited memories
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memory_skill.skill import MemorySkill


from memory_skill.tools import TOOL_SCHEMAS

logger = logging.getLogger("memory_skill.mcp")

# ═══════════════════════════════════════════════════════════════════════════════
# Protocol Injection (delivered with every status call + tool descriptions)
# ═══════════════════════════════════════════════════════════════════════════════

MEMORY_PROTOCOL = """Memory Skill — usage notes (v6: the DSH plugin drives the protocol).

Context injection and turn persistence are AUTOMATIC — do not spend tool calls
on them:
  • The plugin calls memory_weave at each turn's pre-step and injects the
    returned context (tier1/tier2/nudge + soft [协议提醒]).
  • The plugin calls memory_ingest at turn-stopping with the turn's user and
    assistant texts. Do NOT re-ingest the same message yourself.
  • memory_classify is optional bookkeeping for the active-learning loop; it
    never blocks a reply. Skip it when the turn is plain chat.

Use tools when they actually add information:
  • memory_search — a specific fact/history gap the injected context missed.
  • memory_fact_add / memory_facts / memory_fact_timeline — L1 fact cards and
    their evidence timeline (cat name, paths, preferences, conclusions).
  • memory_teach_skill — a reusable skill you just learned (needs source_urls).
  • memory_ingest — only for content OUTSIDE the current turn (notes, pasted
    docs) that should become long-term memory.
  • memory_feedback — after a search, cite the entries you actually used.

If weave returns a protocol_notice, treat it as a suggestion, not a blocker.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Tool Definitions (MCP schema format)
# ═══════════════════════════════════════════════════════════════════════════════

TOOLS: list[dict[str, Any]] = [
    {
        "name": name,
        "description": meta["description"],
        "inputSchema": meta["inputSchema"],
    }
    for name, meta in sorted(TOOL_SCHEMAS.items())
]


# ToolHandler
# ═══════════════════════════════════════════════════════════════════════════════

# KNOWN-ISSUES #8: full-length assistant_content (e.g. a whole summary block)
# makes the auto-ingest embedding slow enough to trip MCP -32001 timeouts.
# Cap what gets embedded at ingest time; retrieval/weave still sees the
# full query where it matters.
#
# Head+tail retention: Chinese replies often put the conclusion at the END,
# so naive head-only truncation would drop it (KNOWN-ISSUES #9 "结论散落").
# Keep the first `_HEAD` chars + last `_TAIL` chars, ellipsis in between.

class ToolHandler:
    """Dispatches MCP tool calls to standalone handler functions in tools.py."""

    def __init__(self, skill: MemorySkill) -> None:
        self._skill = skill

    def handle(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        from memory_skill.tools import DISPATCH
        handler = DISPATCH.get(tool_name)
        if handler is None:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            return handler(self._skill, arguments)
        except Exception as exc:
            logger.exception("Tool %s failed", tool_name)
            return {"error": f"{type(exc).__name__}: {exc}"}
