"""MCP Server for Memory Skill — stdio JSON-RPC.

Start with::

    python -m memory_skill.mcp_server

Or install as an MCP server in Claude Desktop / OpenCode config.

The server uses the standard ``mcp`` SDK with ``Server`` + ``stdio_server``.
It is stateless across connections — each session creates a fresh
``MemorySkill`` instance.

Environment variables:
  ``MEMORY_SKILL_DB_PATH`` — override the default DB path (default: ``memory.db``)
  ``MEMORY_SKILL_REWRITE`` — enable MiniCPM query rewriting (default: ``false``)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading

# ── Ensure memory_skill is on sys.path ──────────────────────────────────────
# When launched via MCP, PYTHONPATH is not reliably set by the host.
# Use the location of THIS file to infer the project root.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── Stdio protection ───────────────────────────────────────────────────────
# The MCP protocol multiplexes JSON-RPC over stdio: stdout MUST carry only
# valid JSON-RPC messages. Redirect stdout → stderr during heavy imports
# to prevent diagnostic output from corrupting the protocol stream.
_REAL_STDOUT = sys.stdout
_REAL_STDOUT_FD = None
try:
    _REAL_STDOUT_FD = os.dup(1)
    os.dup2(2, 1)
except (OSError, AttributeError):
    pass
sys.stdout = sys.stderr

# ── Configure logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("memory_skill.mcp_server")


def _restore_stdout() -> None:
    """Restore real stdout for MCP JSON-RPC output."""
    global _REAL_STDOUT, _REAL_STDOUT_FD
    if _REAL_STDOUT_FD is not None:
        try:
            os.dup2(_REAL_STDOUT_FD, 1)
            os.close(_REAL_STDOUT_FD)
        except OSError:
            pass
        _REAL_STDOUT_FD = None
    sys.stdout = _REAL_STDOUT


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


async def main() -> None:
    """Start the Memory Skill MCP server over stdio."""
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool

    from memory_skill.contracts import MemorySkillConfig
    from memory_skill.mcp_tools import MEMORY_PROTOCOL, TOOLS, ToolHandler
    from memory_skill.skill import MemorySkill

    # Force UTF-8 on stdio for cross-platform compatibility
    for stream in (sys.stdin, sys.stdout):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, OSError):
                pass

    # ── Build config from environment ──────────────────────────────────────
    db_path = os.environ.get("MEMORY_SKILL_DB_PATH", "memory.db")

    config = MemorySkillConfig(
        db_path=db_path,
    )

    logger.info(
        "Memory Skill MCP Server starting — db=%s",
        db_path,
    )

    # ── Load skill in background (ONNX model takes ~30s) ──────────────────
    # The MCP protocol requires the server to respond to initialize within
    # a few seconds.  Load the model asynchronously so the server can
    # accept the handshake immediately.
    skill: MemorySkill | None = None
    handler: ToolHandler | None = None

    _load_lock = asyncio.Lock()

    async def _ensure_loaded() -> ToolHandler:
        """构建 skill 并预热嵌入模型，只做一次，且不阻塞事件循环。"""
        nonlocal skill, handler
        if handler is not None:
            return handler
        async with _load_lock:
            if handler is None:
                import time as _time

                _t0 = _time.monotonic()

                def _build():
                    s = MemorySkill(config)
                    h = ToolHandler(s)
                    # 真正初始化 ONNX session，避免第一次查询承担加载耗时
                    try:
                        s.embedder.embed("warmup")
                    except Exception:
                        pass
                    return s, h

                skill, handler = await asyncio.to_thread(_build)
                logger.info(
                    "Memory Skill loaded in %.1fs — db=%s",
                    _time.monotonic() - _t0, db_path,
                )
        return handler

    # Restore stdout BEFORE starting the MCP server loop.
    _restore_stdout()


    # ── Build MCP Server ───────────────────────────────────────────────────
    server = Server(
        name="memory-skill",
        version="4.0.0",
        instructions=MEMORY_PROTOCOL,
    )

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """Return the tool manifest."""
        return [
            Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t["inputSchema"],
            )
            for t in TOOLS
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """Handle a tool invocation (offloaded to thread pool)."""
        try:
            import time as _ct
            _logdir = os.path.dirname(
                os.environ.get('MEMORY_SKILL_DB_PATH', 'D:\\DSH\\memory.db')
            ) or '.'
            with open(os.path.join(_logdir, 'tool_calls.log'), 'a',
                      encoding='utf-8') as _cf:
                _cf.write('%0.3f,%s\n' % (_ct.time(), name))
        except Exception:
            pass
        h = await _ensure_loaded()
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, h.handle, name, arguments)
        return [
            TextContent(
                type="text",
                text=json.dumps(result, indent=2, ensure_ascii=False, default=str),
            )
        ]

    # ── Run ────────────────────────────────────────────────────────────────
    async with stdio_server() as (read, write):
        await server.run(
            read,
            write,
            server.create_initialization_options(),
        )


def _preload_model(db_path: str) -> None:
    """主线程同步预热嵌入模型，避免第一次工具调用承担加载耗时。"""
    import time as _time

    _t0 = _time.monotonic()
    try:
        from memory_skill.contracts import MemorySkillConfig
        from memory_skill.embedder import Embedder

        Embedder(MemorySkillConfig(db_path=db_path)).embed("warmup")
        logger.info("embedder preloaded in %.1fs", _time.monotonic() - _t0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("embedder preload failed: %s", exc)


if __name__ == "__main__":
    _preload_model(os.environ.get("MEMORY_SKILL_DB_PATH", "memory.db"))
    asyncio.run(main())


def serve():
    asyncio.run(main())
