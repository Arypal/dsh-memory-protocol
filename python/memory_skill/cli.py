"""CLI entry point for Memory Skill.

Usage::

    memory status                    # Health check
    memory search "Python project"   # Search memories
    memory ingest "hello world"      # Ingest a message
    memory weave "what now?"         # Weave context

Install: ``pip install memory-skill``, then ``memory --help``.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime

import click

from memory_skill import DialogueTurn, MemorySkill, MemorySkillConfig

# ── Logging setup ──────────────────────────────────────────
LOG_LEVEL = logging.getLevelName(
    os.getenv("MEMORY_SKILL_LOG_LEVEL", "WARNING").upper()
)
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("memory_skill.cli")


# ── Shared options ─────────────────────────────────────────

def _get_skill(db_path: str, agent: str | None) -> MemorySkill:
    """Create a MemorySkill instance with common options."""
    return MemorySkill(MemorySkillConfig(
        db_path=db_path,
        agent_name=agent or "",
    ))


# ── CLI group ──────────────────────────────────────────────

@click.group()
@click.option("--db", default="memory.db", show_default=True,
              envvar="MEMORY_SKILL_DB_PATH",
              help="Path to SQLite database file")
@click.option("--agent", default=None,
              envvar="MEMORY_SKILL_AGENT",
              help="Agent name for namespace isolation (e.g. 'my_agent')")
@click.pass_context
def main(ctx: click.Context, db: str, agent: str | None) -> None:
    """Memory Skill — long-term memory for AI agents."""
    ctx.ensure_object(dict)
    ctx.obj["db"] = db
    ctx.obj["agent"] = agent or ""


@main.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show memory system health and statistics."""
    skill = _get_skill(ctx.obj["db"], ctx.obj["agent"])
    h = skill.health()

    click.echo(f"Memory Skill v{__import__('memory_skill').__version__}")
    click.echo(f"DB: {ctx.obj['db']}")
    click.echo(f"Agent: {ctx.obj['agent'] or '(default)'}")
    click.echo(f"Embedder: {h['embedder']['mode']} ({h['embedder']['dim']}-dim)")
    if h.get("embedder", {}).get("degraded"):
        click.echo("⚠ EMBEDDER DEGRADED — 无 ONNX 模型，语义检索/去重/学习判定已降级。"
                   "运行 ./download_model.sh 修复。")
    click.echo(f"Learned store: {h['learned_store']['entry_count']} entries")
    click.echo(f"Dialogue: {skill.count_turns()} turns")


@main.command()
@click.option("--port", "-p", default=8888, show_default=True,
              help="Port for the proxy to listen on")
@click.pass_context
def proxy(ctx: click.Context, port: int) -> None:
    """Start transparent memory proxy (OpenAI-compatible).

    The proxy sits between the agent and the LLM, automatically
    injecting memory context before each request and ingesting
    both sides of the conversation after each response.

    Requires ``DEEPSEEK_API_KEY`` in environment.
    """
    from memory_skill.transparent_proxy import TransparentProxy

    api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    api_key = os.getenv("DEEPSEEK_API_KEY", "")

    if not api_key:
        click.echo("DEEPSEEK_API_KEY environment variable is required.", err=True)
        return

    skill = _get_skill(ctx.obj["db"], ctx.obj["agent"])
    TransparentProxy(skill, api_base, api_key, port).start(block=True)


@main.command()
@click.argument("query")
@click.option("--limit", "-n", default=10, show_default=True,
              help="Max results")
@click.option("--partner", "-p", default=None,
              help="Partner for object-tagged namespace")
@click.option("--json", "as_json", is_flag=True,
              help="Output as JSON")
@click.pass_context
def search(ctx: click.Context, query: str, limit: int,
           partner: str | None, as_json: bool) -> None:
    """Search memories by semantic relevance."""
    skill = _get_skill(ctx.obj["db"], ctx.obj["agent"])
    results = skill.retrieve(query, limit=limit, partner=partner)

    if as_json:
        entries = [
            {"id": e.id, "content": e.content[:200],
             "weight": round(e.weight, 2),
             "partner": (e.metadata or {}).get("partner", ""),
             "category": e.category}
            for e in results.entries
        ]
        click.echo(json.dumps({
            "query": query,
            "count": len(entries),
            "results": entries,
        }, ensure_ascii=False, indent=2))
    else:
        if not results.entries:
            click.echo("(no memories found)")
            return
        for e in results.entries:
            p = e.metadata.get("partner", "") if e.metadata else ""
            tag = f" [与{p}]" if p else ""
            click.echo(f"[{e.id}]{tag} w={e.weight:.2f}")
            click.echo(f"  {e.content[:200]}")
            click.echo()


@main.command()
@click.argument("content")
@click.option("--role", "-r", default="user", show_default=True,
              type=click.Choice(["user", "assistant", "system"]))
@click.option("--partner", "-p", default=None,
              help="Partner for object-tagged namespace")
@click.pass_context
def ingest(ctx: click.Context, content: str, role: str,
           partner: str | None) -> None:
    """Ingest a message into memory."""
    skill = _get_skill(ctx.obj["db"], ctx.obj["agent"])

    turn = DialogueTurn(
        id=f"cli_{datetime.now(UTC).timestamp():.0f}",
        role=role, content=content,
        timestamp=datetime.now(UTC),
        partner=partner,
    )
    skill.ingest(turn)
    n = skill.count_turns()
    click.echo(f"ok ({n} turns total)")


@main.command("teach-skill")
@click.argument("title")
@click.argument("content", required=False)
@click.option("--url", "-u", multiple=True,
              help="Source URL used for learning (repeatable)")
@click.pass_context
def teach_skill(ctx: click.Context, title: str, content: str,
                url: tuple[str, ...]) -> None:
    """Teach a skill: title + content (or read content from stdin)."""
    skill = _get_skill(ctx.obj["db"], ctx.obj["agent"])
    if not content:
        content = click.get_text_stream("stdin").read().strip()
    result = skill.ingest_skill_ex(title, content, source_urls=list(url) or None)
    click.echo(f"stored: {result.entry_id}")


@main.command()
@click.pass_context
def gaps(ctx: click.Context) -> None:
    """Show pending learning-queue items (skills / missions)."""
    skill = _get_skill(ctx.obj["db"], ctx.obj["agent"])
    gaps = skill.gaps
    if not gaps:
        click.echo("(queue empty)")
        return
    for g in gaps[-20:]:
        click.echo(f"[{g.kind}/{g.status}] {g.query[:80]}")


@main.command()
@click.argument("message", default="")
@click.option("--scene", "-s", default="",
              help="Scene summary for tier1 context")
@click.option("--partner", "-p", default=None,
              help="Partner for object-tagged namespace")
@click.pass_context
def weave(ctx: click.Context, message: str, scene: str,
          partner: str | None) -> None:
    """Assemble layered memory context for agent injection."""
    skill = _get_skill(ctx.obj["db"], ctx.obj["agent"])
    ctx_result = skill.weave(
        user_message=message,
        scene_summary=scene,
        partner=partner,
    )
    block = ctx_result.to_prompt_block()
    if block:
        click.echo(block)
    else:
        click.echo("(empty — no memories yet)")


if __name__ == "__main__":
    main()
