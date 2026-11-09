"""Standalone tool handler functions — extracted from mcp_tools.ToolHandler.

Each function is a pure handler: it takes a MemorySkill instance and
arguments, returns a result dict. No class state, no dispatch logic.
ToolHandler.handle() delegates here via a name→function mapping.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("memory_skill.tools")


# ── Lazy helpers (module-level singletons) ────────────────────────────

# Per-instance caches (WeakKeyDictionary so skills can be GC'd).
from weakref import WeakKeyDictionary

_registries: WeakKeyDictionary = WeakKeyDictionary()
_writers: WeakKeyDictionary = WeakKeyDictionary()


def _get_registry(skill):
    # Per-instance cache so multiple MemorySystem instances (test dbs, room
    # agents) never share one registry bound to the first skill's tree.
    if skill not in _registries:
        from memory_skill.capability_registry import CapabilityRegistry
        from memory_skill.skill_registry import SkillRegistry
        capability = CapabilityRegistry(skill.tree, skill.retriever)
        _registries[skill] = SkillRegistry(skill.retriever, capability)
    return _registries[skill]


def _get_writer(skill):
    if skill not in _writers:
        from memory_skill.skill_writer import SkillWriter
        _writers[skill] = SkillWriter(skill)
    return _writers[skill]


# ── Handler functions ─────────────────────────────────────────────────

def handle_weave(skill, args: dict[str, Any]) -> dict[str, Any]:
    """Assemble memory context. Read-only: storing turns is the ingestor's job.

    0.4.1: weave no longer ingests ``user_message``/``assistant_content`` on
    the way through. The old side effect duplicated every turn (plugin
    pre-step weave + turn-stopping ingest + any explicit model weave), which
    polluted L3 with 2–3 copies per message. Ingestion now has exactly one
    writer: ``memory_ingest`` (called by the plugin at turn-stopping).
    """
    user_message = args.get("user_message", "")
    assistant_content = args.get("assistant_content", "")

    try:
        ctx = skill.weave(
            user_message=user_message,
            scene_summary=args.get("scene_summary", ""),
            partner=args.get("partner"),
            scope=args.get("scope", "web"),
        )
        block = ctx.to_prompt_block()
        return {
            "tier1_context": ctx.tier1_context,
            "tier2_context": ctx.tier2_context,
            "memory_nudge": ctx.memory_nudge,
            "prompt_block": block,
            # ``text`` is what the DSH plugin extracts for prompt injection;
            # without it the auto-injected context was silently empty.
            "text": block,
            "protocol_notice": ctx.protocol_notice,
            "is_empty": ctx.is_empty,
        }
    except Exception as exc:
        logger.exception("weave failed")
        return {"error": f"{type(exc).__name__}: {exc}"}


def _parse_time_hints(query: str):
    """从自然语言查询里解析时间范围 → (since_days, date_from, date_to)。

    即使客户端不传时间参数，问"上周/昨天/最近3天/10-26"也能限定时间窗。
    只裁对话条目；事实卡/技能是长期知识，不受影响。
    """
    import re
    from datetime import date, timedelta

    q = (query or "").strip()
    today = date.today()

    def ds(dt) -> str:
        return dt.strftime("%Y-%m-%d")

    m = re.search(r"(?:最近|近)\s*(\d{1,3})\s*天", q)
    if m:
        return int(m.group(1)), None, None
    if re.search(r"今天|今日", q):
        return None, ds(today), ds(today)
    if re.search(r"昨天|昨日", q):
        y = today - timedelta(days=1)
        return None, ds(y), ds(y)
    if "前天" in q:
        y = today - timedelta(days=2)
        return None, ds(y), ds(y)
    if "上周" in q:
        monday = today - timedelta(days=today.weekday())
        return None, ds(monday - timedelta(days=7)), ds(monday - timedelta(days=1))
    if "上个月" in q or "上月" in q:
        first = today.replace(day=1)
        end = first - timedelta(days=1)
        return None, ds(end.replace(day=1)), ds(end)
    m = re.search(r"(?<![\d-])(\d{4})-(\d{1,2})-(\d{1,2})(?![\d-])", q)
    if m:
        d1 = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
        return None, d1, d1
    m = re.search(r"(?<![\d-])(\d{1,2})-(\d{1,2})(?![\d-])", q)
    if m:
        d1 = f"{today.year}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
        return None, d1, d1
    m = re.search(r"(\d{1,2})月(\d{1,2})[日号]", q)
    if m:
        d1 = f"{today.year}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
        return None, d1, d1
    return None, None, None


def handle_search(skill, args: dict[str, Any]) -> dict[str, Any]:
    query = args.get("query", "")
    if not query.strip():
        return {"error": "query is required"}
    # A mission that searches memory has checked for existing knowledge.
    skill.protocol.mark_skill_checked()
    try:
        # 时间窗：显式参数优先；没给就从查询文本里解析（"上周/昨天/10-26"）
        since_days = args.get("since_days")
        date_from = args.get("date_from")
        date_to = args.get("date_to")
        if since_days is None and not date_from and not date_to:
            since_days, date_from, date_to = _parse_time_hints(query)
        result = skill.retrieve(
            query, limit=args.get("limit", 10),
            scope=args.get("scope", "all"),
            since_days=since_days,
            date_from=date_from,
            date_to=date_to,
        )
        from datetime import UTC, datetime
        query_id = f"q_{datetime.now(UTC):%Y%m%d_%H%M%S}_{hash(query) & 0xFFFF:04x}"
        return {
            "query_id": query_id,
            "total_candidates": result.total_candidates,
            "results": [
                {
                    "entry_id": e.id,
                    "content": e.content,
                    "semantic_score": getattr(e, "semantic_score", None),
                    "weight": e.weight,
                    "metadata": getattr(e, "metadata", None),
                }
                for e in result.entries
            ],
        }
    except Exception as exc:
        logger.exception("search failed")
        return {"error": f"{type(exc).__name__}: {exc}"}


def handle_ingest(skill, args: dict[str, Any]) -> dict[str, Any]:
    from memory_skill.contracts import DialogueTurn, clip_auto_ingest
    from datetime import UTC, datetime

    content = args.get("content", "")
    role = args.get("role", "user")
    if not content.strip():
        return {"error": "content is required"}
    try:
        turn = DialogueTurn(
            id=f"mcp_{datetime.now(UTC):%Y%m%d_%H%M%S}_{hash(content) & 0xFFFF:04x}",
            role=role,
            content=clip_auto_ingest(content),
            timestamp=datetime.now(UTC),
        )
        receipt = skill.ingest(turn, enrich=False,
                               scope=args.get("scope", "web"))
        return {
            "entry_id": receipt.entry_id,
            "deduped": receipt.deduped,
            "weight": receipt.weight,
        }
    except Exception as exc:
        logger.exception("ingest failed")
        return {"error": f"{type(exc).__name__}: {exc}"}


def _integrity_report(skill) -> dict:
    """扫描事实卡与近期对话，统计通不过 GCM 校验的条数（篡改可见）。"""
    bad = 0
    total = 0
    try:
        facts = skill.facts.list_facts(include_inactive=True)
        total += len(facts)
        bad += sum(1 for f in facts
                   if str(f.get("current_value", "")).startswith("[! 通不过"))
    except Exception:
        pass
    try:
        turns = skill.dialogue_store.get_recent(300)
        total += len(turns)
        bad += sum(1 for t in turns
                   if str(getattr(t, "content", "")).startswith("[! 通不过"))
    except Exception:
        pass
    try:
        from memory_skill import secure
        key = secure.key_path()
    except Exception:
        key = None
    return {
        "checked": total,
        "corrupted": bad,
        "key_path": key,
        "status": "ok" if bad == 0 else "corrupted",
    }


def handle_status(skill, args: dict[str, Any]) -> dict[str, Any]:
    h = skill.health()
    if h.get("embedder", {}).get("degraded"):
        h["warning"] = (
            "⚠ EMBEDDER DEGRADED — 语义检索/去重/学习判定已降级为 SHA-256 hash"
            "（无 ONNX 模型）。修复: 运行 ./download_model.sh 下载 bge-large-en-v1.5，"
            "或 export MEMORY_MODEL_PATH=<模型路径> 后重启。"
        )
    h["integrity"] = _integrity_report(skill)
    return h


def handle_feedback(skill, args: dict[str, Any]) -> dict[str, Any]:
    query_id = args.get("query_id", "")
    outcome = args.get("outcome", "auto")
    cited_ids = args.get("cited_ids", [])
    if not query_id:
        return {"error": "query_id is required"}
    try:
        boosted: list[str] = []
        for entry_id in cited_ids:
            skill.learned_store.boost_weight(entry_id)
            boosted.append(entry_id)
        return {
            "status": "feedback_recorded",
            "query_id": query_id,
            "outcome": outcome,
            "boosted": boosted,
            "cited": cited_ids,
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def handle_check_skill(skill, args: dict[str, Any]) -> dict[str, Any]:
    skill_name = args.get("skill_name", "")
    if not skill_name.strip():
        return {"error": "skill_name is required"}
    # Checking existing skills satisfies the mission gate.
    skill.protocol.mark_skill_checked()
    try:
        return _get_registry(skill).check_skill(skill_name)
    except Exception as exc:
        logger.exception("check_skill failed for %r", skill_name)
        return {"error": f"{type(exc).__name__}: {exc}"}


def handle_teach_skill(skill, args: dict[str, Any]) -> dict[str, Any]:
    return _get_writer(skill).teach_skill(
        title=args.get("title", ""),
        content=args.get("content", ""),
        source_urls=args.get("source_urls"),
    )


def handle_update_skill(skill, args: dict[str, Any]) -> dict[str, Any]:
    entry_id = args.get("entry_id", "")
    if not entry_id.strip():
        return {"error": "entry_id is required"}
    try:
        return _get_writer(skill).update_skill(entry_id, args.get("content", ""))
    except Exception as exc:
        logger.exception("update_skill failed for %s", entry_id)
        return {"error": f"{type(exc).__name__}: {exc}"}


def handle_classify(skill, args: dict[str, Any]) -> dict[str, Any]:
    category = args.get("category", "chat")
    note = args.get("note", "")
    gaps = args.get("gaps")
    skill.protocol.mark_classified(category, gaps)
    _enqueue_if_learning(skill, category, note, gaps)
    return {"status": "classified", "category": category, "note": note}


def _enqueue_if_learning(skill, category: str, note: str, gaps) -> None:
    """Enqueue skill/mission items into the learning queue.

    The classifier routing a turn to ``skill`` or ``mission`` is the only
    write-side entry of the active-learning loop (see learning_queue docs).
    Without this, [待学习]/[待拆解] directives never appear for new turns.
    Dedup happens inside ``enqueue`` (identical open items are skipped).
    """
    if category not in ("skill", "mission"):
        return
    query = (note or "").strip()
    if not query:
        return
    queue = getattr(skill, "learning_queue", None)
    if queue is None:
        return
    detail = ""
    if category == "mission" and gaps:
        detail = "所需技能: " + ", ".join(str(g) for g in gaps if g)
    try:
        queue.enqueue(category, query, detail=detail)
    except Exception as exc:
        logger.warning("learning_queue enqueue failed: %s", exc)


def handle_learning_queue(skill, args: dict[str, Any]) -> dict[str, Any]:
    gaps = skill.gaps
    return {
        "count": len(gaps),
        "items": [
            {
                "kind": g.kind,
                "query": getattr(g, "query", "")[:120],
                "detail": getattr(g, "detail", "")[:200],
                "status": getattr(g, "status", "open"),
                "id": getattr(g, "id", ""),
            }
            for g in gaps[-20:]
        ],
    }


def handle_conclusions(skill, args: dict[str, Any]) -> dict[str, Any]:
    limit = args.get("limit", 10)
    try:
        entries = skill.learned_store.list_by_category("conclusion", limit=limit)
        return {
            "conclusions": [
                {
                    "id": e.id,
                    "title": getattr(e, "title", ""),
                    "summary": getattr(e, "summary", e.content[:120]),
                    "content": e.content,
                    "weight": e.weight,
                }
                for e in entries
            ],
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def handle_learning_mark(skill, args: dict[str, Any]) -> dict[str, Any]:
    """Mark a learning-queue item done/skipped (e.g. a completed mission).

    Missions are never auto-closed by teach_skill (that only matches
    ``kind=skill``); the agent closes them explicitly once the mission has
    been decomposed or executed. Requires the item id from
    ``memory_learning_queue``.
    """
    item_id = args.get("item_id", "")
    status = args.get("status", "done")
    if not item_id.strip():
        return {"error": "item_id is required"}
    if status not in ("done", "skipped"):
        return {"error": f"status must be 'done' or 'skipped', got {status!r}"}
    queue = getattr(skill, "learning_queue", None)
    if queue is None:
        return {"error": "learning_queue not available (tree disabled?)"}
    try:
        ok = queue.mark(item_id.strip(), status)
        return {"status": "marked" if ok else "not_found_or_already_closed",
                "item_id": item_id.strip(), "new_status": status}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def handle_distill(skill, args: dict[str, Any]) -> dict[str, Any]:
    """Turn dialogue fragments into reviewable candidate cards.

    Pure decision-support: compresses fragments into topic/summary/evidence
    candidates and writes them to the pending store.  Nothing is asserted
    or promoted — the agent reviews via memory_pending and decides.

    ``offset``/``limit`` walk the history window by window (default: newest
    60 turns).  Pass ``offset=60`` next to reach older memories.
    """
    pending = getattr(skill, "pending_store", None)
    if pending is None:
        return {"error": "pending_store not available (tree disabled?)"}
    from memory_skill.distill import distill as _distill
    try:
        return _distill(
            skill.dialogue_store, pending,
            since_days=args.get("since_days", 7),
            offset=args.get("offset", 0),
            limit=args.get("limit", 60),
        )
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def handle_pending(skill, args: dict[str, Any]) -> dict[str, Any]:
    """List open distill candidates awaiting agent review."""
    pending = getattr(skill, "pending_store", None)
    if pending is None:
        return {"error": "pending_store not available (tree disabled?)"}
    try:
        items = pending.list_open(limit=args.get("limit", 20))
        return {
            "count": len(items),
            "items": [
                {
                    "candidate_id": c.candidate_id,
                    "topic": c.topic,
                    "summary": c.summary,
                    "evidence": c.evidence,
                    "suggested": c.suggested,
                    "confidence": c.confidence,
                    "created_at": c.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                }
                for c in items
            ],
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def handle_pending_mark(skill, args: dict[str, Any]) -> dict[str, Any]:
    """Accept or reject a distill candidate.

    Accepting a ``conclusion``/``pref``/``pers`` candidate auto-promotes it
    to the structured store (the agent's decision is the review gate).
    Accepting a ``skill`` candidate only records the decision — the agent
    must still teach it via ``memory_teach_skill`` with source_urls (the
    ADR-0002 anti-hallucination gate; distill evidence ids are dialogue
    ids, not URLs).
    """
    pending = getattr(skill, "pending_store", None)
    if pending is None:
        return {"error": "pending_store not available (tree disabled?)"}
    candidate_id = args.get("candidate_id", "")
    status = args.get("status", "")
    if not candidate_id.strip():
        return {"error": "candidate_id is required"}
    if status not in ("accepted", "rejected"):
        return {"error": f"status must be 'accepted' or 'rejected', got {status!r}"}

    cand = None
    for c in pending.list_open(limit=100):
        if c.candidate_id == candidate_id.strip():
            cand = c
            break
    if cand is None:
        return {"error": "candidate not found or already closed"}
    if not pending.mark(candidate_id.strip(), status):
        return {"status": "not_found_or_already_closed",
                "candidate_id": candidate_id.strip()}

    promoted: str | None = None
    if status == "accepted" and cand.suggested in ("conclusion", "pref", "pers"):
        try:
            from memory_skill.memory_extract import _ingest_structured
            result = _ingest_structured(
                skill, cand.topic, cand.summary, cand.suggested,
            )
            promoted = getattr(result, "entry_id", None) or cand.suggested
        except Exception as exc:
            return {"status": "marked",
                    "candidate_id": candidate_id.strip(),
                    "promotion_error": f"{type(exc).__name__}: {exc}"}

    extra: dict[str, Any] = {}
    if promoted:
        extra["promoted_to"] = promoted
    if status == "accepted" and cand.suggested == "skill":
        extra["note"] = "已接受但需 memory_teach_skill(带 source_urls) 转正"
    return {"status": "marked",
            "candidate_id": candidate_id.strip(),
            **extra}


def handle_toggle(skill, args):
    """记忆开关：pause / resume / status，写 memory/paused.json。

    JS 插件每轮读该文件（3 秒缓存）：global=true 或 scopes 命中当前通道时，
    该通道不注入、不写入（数据一条不删）。
    """
    import json as _json

    action = (args.get("action") or "status").strip().lower()
    scope = (args.get("scope") or "all").strip() or "all"
    try:
        from memory_skill import secure
        path = os.path.join(secure.memory_dir(), "paused.json")
    except Exception:
        path = os.path.join(os.path.expanduser("~"), ".dsh", "memory", "paused.json")
    data = {"global": False, "scopes": []}
    try:
        with open(path, encoding="utf-8") as f:
            loaded = _json.load(f)
        if isinstance(loaded, dict):
            data["global"] = bool(loaded.get("global"))
            scopes = loaded.get("scopes")
            data["scopes"] = [str(s) for s in scopes] if isinstance(scopes, list) else []
    except Exception:
        pass
    if action == "status":
        return {"status": "ok", "file": path, "global": data["global"],
                "scopes": data["scopes"],
                "paused": bool(data["global"]) or bool(data["scopes"])}
    if action in ("pause", "resume"):
        on = action == "pause"
        if scope == "all":
            data["global"] = on
        else:
            s = set(data["scopes"])
            if on:
                s.add(scope)
            else:
                s.discard(scope)
            data["scopes"] = sorted(s)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                _json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, path)
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "reason": str(exc)}
        return {"status": "ok", "action": action, "global": data["global"],
                "scopes": data["scopes"], "file": path}
    return {"status": "error", "reason": "action must be pause|resume|status"}


def handle_prune(skill, args):
    """手动触发对话清理（L3），清理前会生成区间摘要（L4）。"""
    return skill.prune_dialogue(
        ttl_days=args.get("ttl_days"),
        max_turns=args.get("max_turns"),
        dry_run=bool(args.get("dry_run", False)),
    )


def handle_episodes(skill, args):
    """列出区间摘要（L4）。"""
    return {"status": "ok", "episodes": skill.episodes(int(args.get("limit", 20)))}


def handle_fact_add(skill, args):
    """建/更新事实卡（自动别名 + 证据）。"""
    return skill.ingest_fact(
        category=args.get("category", "pers"),
        key=args.get("key", ""),
        value=args.get("value", ""),
        aliases=args.get("aliases") or [],
        quote=args.get("quote", ""),
        turn_id=args.get("turn_id"),
        source_said_at=args.get("source_said_at"),
        scope=args.get("scope", "web"),
    )


def handle_fact_timeline(skill, args):
    """按 key / fact_id 查看事实、历史值与证据时间线。"""
    target = args.get("key") or args.get("fact_id") or ""
    return skill.fact_timeline(target, int(args.get("limit", 50)),
                               scope=args.get("scope", "web"))


def handle_facts(skill, args):
    """列出全部事实卡（可按 category 过滤）。"""
    return {"status": "ok",
            "facts": skill.facts.list_facts(
                args.get("category"),
                scope=args.get("scope", "web"),
                include_inactive=bool(args.get("include_inactive", False)))}


# ── Dispatch map ───────────────────────────────────────────────────────

DISPATCH = {
    "memory_weave": handle_weave,
    "memory_search": handle_search,
    "memory_ingest": handle_ingest,
    "memory_status": handle_status,
    "memory_feedback": handle_feedback,
    "memory_check_skill": handle_check_skill,
    "memory_teach_skill": handle_teach_skill,
    "memory_update_skill": handle_update_skill,
    "memory_classify": handle_classify,
    "memory_learning_queue": handle_learning_queue,
    "memory_learning_mark": handle_learning_mark,
    "memory_distill": handle_distill,
    "memory_pending": handle_pending,
    "memory_pending_mark": handle_pending_mark,
    "memory_conclusions": handle_conclusions,
    "memory_fact_add": handle_fact_add,
    "memory_fact_timeline": handle_fact_timeline,
    "memory_facts": handle_facts,
    "memory_prune": handle_prune,
    "memory_episodes": handle_episodes,
    "memory_toggle": handle_toggle,
}

# ── Tool schemas (single source of truth) ────────────────────────
# Each tool's MCP schema lives next to its handler. mcp_tools.TOOLS
# is derived from this map so adding a tool means editing ONE place.
# 记忆分区参数（scope）：四域作用域 + 跨区检索。
_SCOPE_PROP = {
    "type": "string",
    "description": ("记忆分区/作用域：web（Web 主记忆）｜platform（QQ/抖音等平台）｜"
                    "world（全局可见）｜topic:名（一条长期线）｜file:路径（某个文件）｜"
                    "session:名（本次会话）｜all（跨区检索，仅 Web 会话）。默认 web。"),
    "default": "web",
}

TOOL_SCHEMAS: dict[str, dict] = {
    "memory_toggle": {
        "description": "记忆开关（按通道/全局）：pause 暂停、resume 恢复、status 查询。暂停后该通道不注入也不写入（数据一条不删），3 秒内生效；适合「这段别记」的场景。",
        "inputSchema": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["pause", "resume", "status"], "default": "status", "description": "pause 暂停 / resume 恢复 / status 查询当前状态。"},
            "scope": {"type": "string", "default": "all", "description": "all=全局；web=Web 会话；platform=QQ/抖音等平台。"},
        }},
    },
    "memory_prune": {
        "description": "清理过期的对话流水（L3）：超过 ttl_days 或超出 max_turns 的旧对话会被删除，删除前自动压成一条区间摘要（L4）。事实卡与证据时间线不受影响。支持 dry_run 预览。",
        "inputSchema": {"type": "object", "properties": {
            "ttl_days": {"type": "integer", "description": "保留天数，0 表示不按时间淘汰。默认取配置（90）。"},
            "max_turns": {"type": "integer", "description": "最多保留条数，0 表示不按条数淘汰。默认取配置（2000）。"},
            "dry_run": {"type": "boolean", "description": "只预览将删除多少条，不实际删除。"}
        }},
    },
    "memory_episodes": {
        "description": "列出对话区间摘要（L4）：每段包含时间范围、涉及实体、条数，以及摘要文本（有 LLM key 时为模型摘要，否则为规则摘要）。",
        "inputSchema": {"type": "object", "properties": {
            "limit": {"type": "integer", "description": "返回条数（默认 20）。"}
        }},
    },
    "memory_fact_add": {
        "description": "建立或更新一张事实卡（方案 C 的 L1），并追加一条证据（L2）。同 category+key 再次写入会更新 current_value 并保留历史值；别名（别名数组 + 自动生成的中英变体）会写入别名索引，供跨语种精确命中。",
        "inputSchema": {"type": "object", "properties": {
            "category": {"type": "string", "description": "分类：pers（人格/身份）、pref（偏好）、conclusion（结论）等。", "default": "pers"},
            "key": {"type": "string", "description": "稳定键名，建议英文 snake_case，例如 cat_name、favorite_editor。"},
            "value": {"type": "string", "description": "当前值，例如 大橘、vscode。"},
            "aliases": {"type": "array", "items": {"type": "string"}, "description": "额外别名（可选），例如 ['Daju', '橘猫']；系统还会自动加 key 的中英变体。"},
            "quote": {"type": "string", "description": "证据原话（用户当时怎么说的），留空则用 value。"},
            "turn_id": {"type": "string", "description": "可选的出处 turn id。"},
            "scope": _SCOPE_PROP
        }, "required": ["category", "key", "value"]},
    },
    "memory_fact_timeline": {
        "description": "查看一张事实卡的详情：当前值、历史值时间线、以及全部证据（完整原话 + 时间戳 + 出处）。可用 key 或 fact_id 查询。",
        "inputSchema": {"type": "object", "properties": {
            "key": {"type": "string", "description": "事实键名，例如 cat_name。"},
            "fact_id": {"type": "string", "description": "完整 id，例如 fact:pers:cat_name（与 key 二选一）。"},
            "limit": {"type": "integer", "description": "返回证据条数（默认 50）。"},
            "scope": _SCOPE_PROP
        }},
    },
    "memory_facts": {
        "description": "列出所有事实卡（可按 category 过滤），用于盘点已知结论。默认只返回 active；被新卡取代或已完成（status=done）的旧卡需要 include_inactive=true 才显示。",
        "inputSchema": {"type": "object", "properties": {
            "category": {"type": "string", "description": "可选分类过滤。"},
            "scope": _SCOPE_PROP,
            "include_inactive": {"type": "boolean", "default": False, "description": "是否包含已取代/已完成的旧卡（status=done）。"}
        }},
    },
    "memory_check_skill": {
        "description": "Check whether a skill is already known to memory (scoped to the skill category). Returns 'known' | 'partial' | 'unknown' with matching skill entries. Use BEFORE learning a new topic \u2014 if known, reuse or update the existing skill instead of re-learning.",
        "inputSchema": {"type": "object", "properties": {"skill_name": {"type": "string", "description": "The skill name to check, e.g. 'Docker Compose'."}}, "required": ["skill_name"]},
    },
    "memory_classify": {
        "description": "Classify the previous user message (chat|skill|mission|pref|pers). Optional but useful: it feeds the learning queue and lets weave tailor its reminders. It never blocks tool use — a missing classification only produces a soft reminder in the injected context.",
        "inputSchema": {"type": "object", "properties": {"category": {"type": "string", "description": "Classification: 'chat' | 'skill' | 'mission' | 'pref' | 'pers'", "enum": ["chat", "skill", "mission", "pref", "pers"]}, "note": {"type": "string", "description": "Optional note: skill name, mission summary, etc."}, "gaps": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}], "description": "Required skill names for a mission (blocks weave until taught)"}}, "required": ["category"]},
    },
    "memory_conclusions": {
        "description": "List reusable conclusion entries extracted from assistant replies (category=conclusion, title + summary + evidence format). Queries the store directly by category \u2014 NOT semantic search \u2014 so the result is complete and accurate. Use to verify what conclusions the memory system has distilled.",
        "inputSchema": {"type": "object", "properties": {"limit": {"type": "integer", "default": 10, "description": "Max conclusions to return (newest first)."}}},
    },
    "memory_distill": {
        "description": "Turn dialogue fragments into reviewable candidate cards (topic/summary/evidence/suggested). Pure decision-support \u2014 nothing is promoted; review via memory_pending. Walks history window by window: offset=0 is newest 60 turns, pass offset=60 to reach older memories.",
        "inputSchema": {"type": "object", "properties": {"since_days": {"type": "integer", "description": "Look back this many days (default 7)."}, "offset": {"type": "integer", "description": "Skip this many newest turns (default 0)."}, "limit": {"type": "integer", "description": "Window size per call (default 60)."}}},
    },
    "memory_feedback": {
        "description": "Record feedback on a previous memory_search result. Provide the query_id from the search response, the outcome ('positive', 'negative', 'neutral', or 'auto' for automatic detection), and the list of memory IDs you cited or found useful. To enable auto-detection, also pass search_results and final_response. This boosts the weights of cited memories to improve future retrieval.\n\n",
        "inputSchema": {"type": "object", "properties": {"query_id": {"type": "string", "description": "The query_id from a previous memory_search response. Used to correlate feedback with the search."}, "outcome": {"type": "string", "description": "How useful the search results were: 'positive' (found what I needed), 'negative' (results were irrelevant), 'neutral' (mixed or unsure), or 'auto' (let the system auto-detect).", "enum": ["positive", "negative", "neutral", "auto"]}, "cited_ids": {"type": "array", "description": "List of memory entry IDs from the search results that you used or cited in your response.", "items": {"type": "string"}}, "search_results": {"type": "array", "description": "Optional: the search results from the previous memory_search call (for auto-detection). Each item should have 'id' and 'content' fields.", "items": {"type": "object"}}, "final_response": {"type": "string", "description": "Optional: the final response the agent produced using the search results (for auto-detection)."}}, "required": ["query_id"]},
    },
    "memory_ingest": {
        "description": "Save a dialogue turn into the memory system for future retrieval. Call this after significant interactions \u2014 user questions, important answers, decisions made, facts learned. Each turn is indexed for both semantic search and full-text retrieval.\n\n",
        "inputSchema": {"type": "object", "properties": {"content": {"type": "string", "description": "The content to save \u2014 typically the user's question or your response that should be remembered."}, "role": {"type": "string", "default": "user", "description": "Speaker role: 'user', 'assistant', or 'system'.", "enum": ["user", "assistant", "system"]}, "scope": _SCOPE_PROP}, "required": ["content"]},
    },
    "memory_learning_mark": {
        "description": "Mark a learning-queue item as done or skipped (e.g. a completed mission). Missions are never auto-closed \u2014 the agent closes them explicitly after decomposition/execution. Pass the item id from memory_learning_queue.",
        "inputSchema": {"type": "object", "properties": {"item_id": {"type": "string", "description": "The learning-queue item id (lq_...)."}, "status": {"type": "string", "description": "'done' (default) or 'skipped'.", "enum": ["done", "skipped"]}}, "required": ["item_id"]},
    },
    "memory_learning_queue": {
        "description": "List pending learning items: skills the classifier flagged as not-yet-mastered and missions awaiting decomposition. Open items are rendered by weave as [\u5f85\u5b66\u4e60]/[\u5f85\u62c6\u89e3] directives \u2014 act on them and confirm results with the user.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    "memory_pending": {
        "description": "List open distill candidates awaiting agent review. After reviewing, accept or reject via memory_pending_mark; promote valuable content with memory_teach_skill (skills).",
        "inputSchema": {"type": "object", "properties": {"limit": {"type": "integer", "description": "Max candidates to return (default 20)."}}},
    },
    "memory_pending_mark": {
        "description": "Accept or reject a distill candidate. Accepted candidates must still be promoted by the agent (teach_skill / structured ingest); this only records the review decision.",
        "inputSchema": {"type": "object", "properties": {"candidate_id": {"type": "string", "description": "Candidate id from memory_pending (dc_...)."}, "status": {"type": "string", "enum": ["accepted", "rejected"]}}, "required": ["candidate_id", "status"]},
    },
    "memory_search": {
        "description": "Search agent memory for relevant past conversations and learned knowledge. Returns ranked results with content, relevance scores, and memory IDs. Use this BEFORE responding to any question about past events, user preferences, or previously discussed topics.",
        "inputSchema": {"type": "object", "properties": {"query": {"type": "string", "description": "Natural language search query. Be specific \u2014 include names, topics, dates for best results."}, "limit": {"type": "integer", "default": 10, "description": "Maximum number of results to return (1-100).", "minimum": 1, "maximum": 100}, "scope": _SCOPE_PROP, "since_days": {"type": "integer", "description": "Only recall dialogue from the last N days (facts/skills are never time-filtered)."}, "date_from": {"type": "string", "description": "Start date YYYY-MM-DD (local time, inclusive)."}, "date_to": {"type": "string", "description": "End date YYYY-MM-DD (local time, inclusive)."}}, "required": ["query"]},
    },
    "memory_status": {
        "description": "Get a health summary of the memory system: entry counts, embedder status, and configuration. Call this on first wake-up to understand the memory system state.\n\n",
        "inputSchema": {"type": "object", "properties": {}},
    },
    "memory_teach_skill": {
        "description": "Persist a taught skill at high confidence. Use AFTER learning a topic (via web search or user instruction): pass the skill title and the learned content. Returns the stored entry_id. The user may later correct it via memory_update_skill.",
        "inputSchema": {"type": "object", "properties": {"title": {"type": "string", "description": "Skill title, e.g. 'Docker Compose \u591a\u5bb9\u5668\u90e8\u7f72'."}, "content": {"type": "string", "description": "The learned knowledge (markdown ok)."}, "source_urls": {"type": "array", "description": "Source URLs used for learning \u2014 required, teaching must be backed by external references, not guesswork.", "items": {"type": "string"}}}, "required": ["title", "content", "source_urls"]},
    },
    "memory_update_skill": {
        "description": "Rewrite a skill entry's content \u2014 a correction from the user or the agent after learning. Directly replaces the stored content (not a semantic merge), so skills stay writable for teaching.",
        "inputSchema": {"type": "object", "properties": {"entry_id": {"type": "string", "description": "The skill entry id to update."}, "content": {"type": "string", "description": "The corrected skill content (markdown ok)."}}, "required": ["entry_id", "content"]},
    },
    "memory_weave": {
        "description": "Assemble layered memory context for the current conversation. The DSH plugin calls this automatically at each turn's pre-step and injects the result; calling it yourself is only needed when you want a fresh/different slice. Read-only (it does NOT ingest). tier1: scene perception, tier2: deep memory (gated), nudge: high-importance reminders. Returns prompt_block plus a soft protocol_notice.",
        "inputSchema": {"type": "object", "properties": {"user_message": {"type": "string", "description": "The user's current message."}, "scene_summary": {"type": "string", "description": "Optional: what the user is doing now."}, "scope": _SCOPE_PROP}},
    },
}
