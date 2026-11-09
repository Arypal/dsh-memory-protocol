"""Distill — turn unclassified dialogue fragments into reviewable candidates.

Phase 2 of the C plan: the main agent owns every learning decision
(ADR-0002), but it needs readable *material* to decide on.  distill()
compresses recent default-category fragments into candidate cards
(topic + summary + evidence references + a *suggested* category) and
writes them to the pending store.  Nothing here asserts knowledge:

- the LLM is instructed to summarise existing turns only, never add facts
- every ``evidence`` id must exist in the dialogue store, else the
  candidate is rejected at write time (anti-hallucination guard)
- suggested category is advisory — accept/reject stays with the agent
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "deepseek-chat"
_DEFAULT_API_BASE = "https://api.deepseek.com"
_MAX_FRAGMENTS = 60
_MAX_EVIDENCE = 6
_MAX_SUMMARY_CHARS = 220


@dataclass
class DistillCandidate:
    """A compressed, evidence-backed candidate for the agent to review."""

    topic: str
    summary: str
    evidence: list[str] = field(default_factory=list)
    suggested: str = "chat"
    confidence: float = 0.5


def _get_env(key: str, default: str) -> str:
    import os
    return os.getenv(key, default)


def _load_env() -> None:
    try:
        from pathlib import Path
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent / ".env",
                    override=False)
        # pip 部署时包旁没有 .env——从标准用户位置补充（~/.config/memory-skill/.env）
        load_dotenv(Path.home() / ".config" / "memory-skill" / ".env",
                    override=False)
    except Exception:
        pass


def _call_llm(prompt: str) -> str | None:
    """Call DeepSeek once (no internal retry loop — caller decides)."""
    from memory_skill._llm_utils import call_llm
    _load_env()
    return call_llm(
        api_base=_get_env("IMPORTANCE_API_BASE", _DEFAULT_API_BASE),
        api_key=_get_env("IMPORTANCE_API_KEY", ""),
        model=_get_env("IMPORTANCE_MODEL", _DEFAULT_MODEL),
        prompt=prompt,
        # Reasoning models (deepseek-v4-flash) burn tokens on reasoning
        # before emitting content — 1200 was exhausted by reasoning alone,
        # truncating content to empty (finish_reason=length). 4k gives
        # room for both.
        max_tokens=4000,
        temperature=0.0,
        timeout=45.0,
        retries=0,
    )


def _fragment_batches(dialogue_store, since_days: int = 7,
                      offset: int = 0, limit: int = _MAX_FRAGMENTS,
                      max_batches: int = 4) -> list[list]:
    """Group turns into small batches for the LLM.

    Uses the dialogue store directly (fragments live there as raw turns).
    Returns batches of (turn_id, content) tuples, newest batches first.

    ``offset``/``limit`` let the caller walk the whole history window by
    window (e.g. offset=0 → newest 60, then offset=60 → the next 60), so
    the distill strategy reaches OLD memories too, not just the newest
    turns.  ``max_batches`` caps how many LLM calls one invocation makes.
    """
    from datetime import timedelta

    cutoff = datetime.now(UTC) - timedelta(days=since_days)
    # get_recent returns newest-last (ascending). offset=0 → the newest
    # `limit` turns; offset=N → skip the N newest, take the next `limit`.
    turns = dialogue_store.get_recent(offset + limit)
    start = max(len(turns) - limit - offset, 0)
    end = max(len(turns) - offset, 0)
    window = turns[start:end]
    recent = [t for t in window if t.timestamp >= cutoff]
    if not recent:
        return []

    batches: list[list] = []
    batch_size = 15
    for i in range(0, len(recent), batch_size):
        batches.append([(t.id, t.content[:200]) for t in recent[i:i + batch_size]])
    return batches[:max_batches]


def _build_prompt(batch: list) -> str:
    items = "\n".join(
        f"- [{tid}] {content}" for tid, content in batch
    )
    return (
        "你是记忆提炼助手。下面是若干段对话记录(带 id)。\n"
        "任务:找出其中值得长期记住的主题(可复用的技能/偏好/结论)。\n"
        "严格规则:\n"
        "1. 只总结这些记录里已有的内容,绝不添加记录中没有的事实\n"
        "2. 每条候选的 evidence 只能引用上面出现的 [id],引用不存在的 id 无效\n"
        "3. 无明确主题时返回空数组\n"
        f"4. summary 不超过 {_MAX_SUMMARY_CHARS} 字\n"
        "输出 JSON 数组,元素格式:\n"
        '{"topic": "主题", "summary": "总结", "evidence": ["id1","id2"], '
        '"suggested": "skill|pref|conclusion|chat", "confidence": 0.0-1.0}\n\n'
        f"对话记录:\n{items}"
    )


def _parse_candidates(raw: str | None) -> list[DistillCandidate]:
    if not raw:
        return []
    from memory_skill._llm_utils import parse_json_response
    parsed = parse_json_response(raw)
    if not isinstance(parsed, list):
        return []
    candidates: list[DistillCandidate] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        topic = str(item.get("topic", "")).strip()
        summary = str(item.get("summary", "")).strip()
        if not topic or not summary:
            continue
        evidence = [str(e) for e in (item.get("evidence") or []) if e]
        suggested = item.get("suggested", "chat")
        if suggested not in ("skill", "pref", "conclusion", "chat"):
            suggested = "chat"
        try:
            confidence = float(item.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        candidates.append(DistillCandidate(
            topic=topic,
            summary=summary[: _MAX_SUMMARY_CHARS],
            evidence=evidence[:_MAX_EVIDENCE],
            suggested=suggested,
            confidence=max(0.0, min(1.0, confidence)),
        ))
    return candidates


def distill(dialogue_store, pending_store,
            since_days: int = 7, offset: int = 0,
            limit: int = _MAX_FRAGMENTS,
            llm_caller=None) -> dict:
    """Distill fragments into pending candidates.

    Returns ``{"candidates": [...], "rejected": n, "stored": n}``.
    LLM failure (no api key / bad response) yields empty candidates with
    ``"error"`` set — the agent decides whether to retry or skip.

    ``offset``/``limit`` walk the history window by window so old memories
    get distilled too; the caller passes the next offset after each call
    until ``note == "no fragments"``.

    ``llm_caller`` (optional) injects the prompt→text LLM boundary for
    tests; defaults to the shared ``_llm_utils`` implementation.
    """
    if llm_caller is None:
        llm_caller = _call_llm

    batches = _fragment_batches(dialogue_store, since_days, offset, limit)
    if not batches:
        return {"candidates": [], "rejected": 0, "stored": 0, "note": "no fragments"}

    stored: list[DistillCandidate] = []
    rejected = 0
    error: str | None = None
    for batch in batches:
        raw = llm_caller(_build_prompt(batch))
        if raw is None:
            error = "LLM call failed (missing credentials?)"
            break
        for cand in _parse_candidates(raw):
            ok, msg = pending_store.add_candidate(cand, dialogue_store)
            if ok:
                stored.append(cand)
            else:
                rejected += 1
                logger.debug("Rejected candidate %r: %s", cand.topic, msg)

    return {
        "candidates": [asdict(c) for c in stored],
        "rejected": rejected,
        "stored": len(stored),
        "note": error or "ok",
    }
