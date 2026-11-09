"""
Memory Skill — Weaver: automatic context injection for agents.

V9: Conversation-unit tier2 with surrounding dialogue.
  tier1: Scene perception + recent dialogue turns (~80t, always injected)
  tier2: Retrieved conversation units (user→agent→...) with context (~300t)
  emotion: Derived affinity from partner feedback history (1 line)
  nudge: High-weight reminders with behavioral intensity (💡/⚠)

Depth auto-selection:
  compact (<3 turns, no stored history):    tier1 only
  standard (3-10 turns OR has stored data): tier1 + tier2 + emotion + nudge
  deep (>10 turns):                         same + heartbeat check

Key V9 changes from V8:
  - Tier2 format: flat fact list → conversation units with surrounding dialogue
  - Each unit shows: User→Agent→Agent→... (full turn context)
  - Unit boundary: starts at preceding user message, ends at next user message
  - V8 features retained: depth gate, emotion context, anti-hallucination guard
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Protocol, TYPE_CHECKING

from memory_skill.contracts import (
    DialogueStoreProtocol,
    LearnedStoreProtocol,
    SawBufferProtocol,
    TreeManagerProtocol,
)
from memory_skill.capability_registry import (
    _SEM_CORROBORATED,
    token_overlap,
)

if TYPE_CHECKING:
    from memory_skill.retriever import Retriever

_logger = logging.getLogger(__name__)


# ── Narrow protocols: each builder takes only what it needs ──

class RetrievalSource(Protocol):
    def retrieve(self, query: str, limit: int = 10, filters=None): ...


class DialogueSource(Protocol):
    def get_recent(self, n: int = 5): ...
    def get_by_id(self, turn_id: str): ...
    def count(self) -> int: ...
    def count_recent(self, minutes: int = 5) -> int: ...


class MemorySource(Protocol):
    def search(self, query: str, limit: int = 10, filters=None): ...


@dataclass
class WeaverStores:
    saw_buffer: SawBufferProtocol
    dialogue_store: DialogueStoreProtocol
    learned_store: LearnedStoreProtocol
    retriever: Retriever
    agent_name: str
    namespace: str
    tree: TreeManagerProtocol | None = None
    gaps: list = field(default_factory=list)
    pending_store: object = None
    facts: object = None
    degraded: bool = False
    degraded_reason: str | None = None


# ── 记忆分区包装：weave 内部所有读取自动带上 scope ──────────────────────────
# 平台会话（QQ/抖音）只看 platform 区；Web 用 web 区；显式 scope="all" 跨区。

class _ScopedDialogue:
    def __init__(self, inner, scope: str) -> None:
        self._inner = inner
        self._scope = scope

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def get_recent(self, n: int = 5):
        return self._inner.get_recent(n, scope=self._scope)

    def count_recent(self, minutes: int = 5) -> int:
        return self._inner.count_recent(minutes, scope=self._scope)


class _ScopedRetriever:
    def __init__(self, inner, scope: str) -> None:
        self._inner = inner
        self._scope = scope

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def retrieve(self, query: str, limit: int = 10, filters=None):
        return self._inner.retrieve(query, limit=limit, filters=filters,
                                    scope=self._scope)


class _ScopedFacts:
    def __init__(self, inner, scope: str) -> None:
        self._inner = inner
        self._scope = scope

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def list_facts(self, category: str | None = None):
        """多域可见：world + 会话自身 + 已绑定域（四域作用域）。"""
        try:
            scopes = self._inner.visible_scopes(self._scope)
        except Exception:
            scopes = None
        return self._inner.list_facts(category, scope=self._scope, scopes=scopes)


@dataclass
class WeaveContext:
    """Layered memory context ready for agent system prompt injection.

    tier1_context: Always injected (~80 tokens). Scene perception.
    tier2_context: Deep memory retrieval (~150 tokens). Gated by depth.
    memory_nudge: High-weight reminders. Weight >= 0.85 threshold.
    needs_second_pass: True when high-weight memories warrant a follow-up
        injection after the agent's next response (heartbeat-style).
    """

    time_context: str = ""
    tone_context: str = ""
    topic_context: str = ""
    lesson_context: str = ""
    tier1_context: str = ""
    tier2_context: str = ""
    memory_nudge: str = ""
    tree_context: str = ""
    tree_nav: str = ""
    skill_context: str = ""
    mission_context: str = ""
    pref_context: str = ""
    pers_context: str = ""
    conclusion_context: str = ""
    gap_context: str = ""
    title_preview: str = ""
    historic_hint: str = ""
    pending_context: str = ""
    embedder_banner: str = ""
    needs_second_pass: bool = False
    # Advisory protocol reminder (never blocking) — set by MemorySystem.weave.
    protocol_notice: str = ""

    def to_prompt_block(self) -> str:
        # Soft advisory block — rendered first, but explicitly non-blocking.
        # 0.4.1: protocol state no longer blocks tool calls, so this is a
        # reminder, not a mandatory directive (see protocol_gate docstring).
        directives: list[str] = []
        if self.protocol_notice:
            directives.append(f"[协议提醒] {self.protocol_notice}（可忽略，不影响继续对话）")
        if self.gap_context:
            directives.append(self.gap_context)

        parts: list[str] = []
        if directives:
            parts.append("═══ 记忆系统提示（非强制，可忽略）═══\n"
                         + "\n".join(directives)
                         + "\n═══ 提示结束（以下为背景记忆，仅供参考）═══")
        if self.embedder_banner:
            parts.append(self.embedder_banner)
        if self.time_context:
            parts.append(self.time_context)
        if self.tone_context:
            parts.append(self.tone_context)
        if self.topic_context:
            parts.append(self.topic_context)
        if self.lesson_context:
            parts.append(self.lesson_context)
        # 待审核提醒属于"要行动的"内容，优先级高于对话，放在事实卡区，
        # 这样注入预算裁剪时不会先被丢掉。
        if self.pending_context:
            parts.append(self.pending_context)
        # tier1（最近对话）与 tier2（相关记忆）都要注入：群聊/私聊是不同会话时，
        # 最近对话是理解上下文的关键，不能因为 tier2 存在就被整段跳过。
        if self.tier1_context:
            parts.append(_clip_lines(self.tier1_context, _TIER1_PROMPT_CHARS))
        if self.tier2_context:
            parts.append(self.tier2_context)
        if self.memory_nudge:
            parts.append(self.memory_nudge)
        if self.skill_context:
            parts.append(self.skill_context)
        if self.mission_context:
            parts.append(self.mission_context)
        if self.pref_context:
            parts.append(self.pref_context)
        if self.pers_context:
            parts.append(self.pers_context)
        if self.conclusion_context:
            parts.append(self.conclusion_context)
        if self.title_preview:
            parts.append(self.title_preview)
        if self.historic_hint:
            parts.append(self.historic_hint)
        if self.tree_context:
            parts.append(self.tree_context)
        if self.tree_nav:
            parts.append(self.tree_nav)
        return _fit_prompt_budget(parts)

    @property
    def is_empty(self) -> bool:
        return not (self.tier1_context or self.tier2_context
                    or self.memory_nudge)


# ── Constants ────────────────────────────────────────

_NUDGE_WEIGHT_THRESHOLD: float = 0.85
_NUDGE_CRITICAL_THRESHOLD: float = 0.95
_NUDGE_MAX_ITEMS: int = 3
_CHARS_PER_TOKEN: int = 4
_TIER1_MAX_CHARS: int = 80 * 20
_TIER2_MAX_CHARS: int = 300 * 4  # was 150→300 chars per memory
_NUDGE_MAX_CHARS: int = 30 * 4
# 最近对话条数：群聊/私聊是不同会话，抓多几条才能覆盖另一端的近期上下文
_MAX_RECENT_TURNS: int = 16
# 注入硬预算：每轮注入的字符上限。超出时从低优先级（parts 末尾）往前裁剪，
# 避免记忆卡一多就把上下文无限撑大（实测曾到 2780 字）。
_MAX_PROMPT_CHARS: int = 1800
# "最近对话"在预算里分到的额度；超了就按行从最早的开始丢。
_TIER1_PROMPT_CHARS: int = 700


def _clip_lines(text: str, limit: int) -> str:
    """把多行文本按行裁到 limit 字：保留首行（块标题），从末尾往前留。"""
    if len(text) <= limit:
        return text
    lines = text.split("\n")
    if not lines:
        return ""
    head, rest = lines[0], lines[1:]
    kept: list[str] = []
    total = len(head)
    for line in reversed(rest):
        if total + len(line) + 1 > limit:
            break
        kept.insert(0, line)
        total += len(line) + 1
    return "\n".join([head] + kept) if kept else text[:limit]


def _fit_prompt_budget(parts: list[str], limit: int | None = None) -> str:
    """按 parts 顺序（越靠后优先级越低）裁到硬上限，最后兜底硬截断。"""
    cap = limit or _MAX_PROMPT_CHARS
    if not parts:
        return ""
    block = "\n\n".join(parts)
    if len(block) <= cap:
        return block
    kept = list(parts)
    while len(kept) > 1 and len("\n\n".join(kept)) > cap:
        kept.pop()
    block = "\n\n".join(kept)
    return block[:cap] if len(block) > cap else block


def _fmt_local(ts: datetime | None, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """把时间转成本地时区再格式化（库里 dialogue/记忆的时间都是 UTC aware）。"""
    if ts is None:
        return "????-??-?? ??:??:??"
    try:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        return ts.astimezone().strftime(fmt)
    except Exception:
        return str(ts)


# ── 软遗忘表达（Soft Forgetting）───────────────────────────────────────────
# 只对"回忆型"条目（对话/摘要）做模糊化：事实卡、技能、教训是结论，不贴标签。
# 口径与 retriever._time_decay 一致：S_eff = S × (1 + 0.5 × arousal)。

def _retrievability(entry) -> float:
    """条目当前的可提取度 R = exp(-Δt / S_eff)（0-1）。"""
    import math

    meta = getattr(entry, "metadata", None) or {}
    try:
        s = float(meta.get("attention_stability") or 4.1667)
    except Exception:
        s = 4.1667
    try:
        age_days = (datetime.now(UTC) - entry.updated_at).total_seconds() / 86400.0
    except Exception:
        age_days = 0.0
    age_days = max(age_days, 0.0)
    try:
        arousal = float(meta.get("arousal") or 0.0)
    except Exception:
        arousal = 0.0
    s_eff = s * (1.0 + 0.5 * min(1.0, arousal))
    if s_eff <= 0:
        return 0.0
    return math.exp(-age_days / s_eff)


def _soft_forget_marker(entry) -> str:
    """"（印象里）/（记得大概是）"前缀；高可提取度或非回忆类条目返回空串。"""
    meta = getattr(entry, "metadata", None) or {}
    eid = str(getattr(entry, "id", ""))
    is_episode = meta.get("source") == "episode"
    is_dialogue = eid.startswith("dialogue:")
    if not (is_dialogue or is_episode):
        return ""
    r = _retrievability(entry)
    if r >= 0.55:
        return ""
    if r >= 0.25:
        return "（印象里）"
    return "（记得大概是）"


# ── weave() ──────────────────────────────────────────

def _shared_cjk_words(query: str, text: str) -> bool:
    """查询与教训文本是否有共享的中文词（长度 >= 2）。"""
    try:
        import jieba

        def words(s: str) -> set:
            return {
                w for w in jieba.cut(s)
                if len(w) >= 2 and all("\u4e00" <= ch <= "\u9fff" for ch in w)
            }

        return bool(words(query) & words(text))
    except Exception:
        return True  # 分词不可用时放行，避免误杀


def _pick_lessons(stores, facts, query: str, limit: int = 3,
                  min_sim: float = 0.30, direct_sim: float = 0.45,
                  scope: str = "web") -> list[str]:
    """按当前消息检索相关教训；相关不足时用最近更新的补足。

    教训卡存在两处：facts 表（权威）与向量库（可检索，id 前缀 fact:lesson:）。
    """
    picked: list[str] = []

    try:
        embedder = getattr(getattr(stores, "learned_store", None), "_embedder", None)
        collection = getattr(getattr(stores, "learned_store", None), "_collection", None)
        if embedder is not None and collection is not None and query:
            vector = embedder.embed(query)
            res = collection.query(query_embeddings=[vector], n_results=8)
            ids = (res.get("ids") or [[]])[0]
            dists = (res.get("distances") or [[]])[0]
            for entry_id, dist in zip(ids, dists):
                entry_id = str(entry_id)
                if not (
                    entry_id.startswith("fact:lesson:")
                    or entry_id.startswith("fact:platform:lesson:")
                ):
                    continue
                if scope != "all":
                    entry_scope = (
                        "platform" if entry_id.startswith("fact:platform:") else "web"
                    )
                    if entry_scope != scope:
                        continue
                similarity = 1.0 - float(dist)
                if similarity < min_sim:
                    continue
                fact = facts.get(entry_id)
                if not fact:
                    continue
                value = fact["current_value"]
                # 一律要求共享词：教训是短文本，纯向量匹配不可靠
                if not _shared_cjk_words(query, value):
                    continue
                if value not in picked:
                    picked.append(value)
                if len(picked) >= limit:
                    break
    except Exception:
        pass

    # 没有相关教训就返回空 —— 不拿时间序凑数（用户要求）。
    return picked[:limit]


def weave(
    stores: WeaverStores,
    user_message: str = "",
    scene_summary: str = "",
    *,
    partner: str | None = None,
    scope: str = "web",
) -> WeaveContext:
    """Assemble layered memory context from all stores.

    Depth is automatically chosen:
      compact (recent <3 turns AND 0 total stored): tier1 only (~80 tokens)
      standard (3-10 recent OR has stored history): tier1 + tier2 (~300 tokens)
      deep (>10 recent): tier1 + tier2 + heartbeat check

    Note: "recent" = last 5 minutes (via count_recent(), not total stored).
    "total stored" = all turns ever (via count()).  These are now DIFFERENT,
    fixing the bug where a new session with old stored data jumped straight
    to deep depth.
    """
    if scope != "all":
        stores = replace(
            stores,
            retriever=_ScopedRetriever(stores.retriever, scope),
            dialogue_store=_ScopedDialogue(stores.dialogue_store, scope),
            facts=_ScopedFacts(stores.facts, scope) if stores.facts is not None else None,
        )
    now = datetime.now()
    ctx = WeaveContext(time_context=f"[现在时间] {now.strftime('%Y-%m-%d %H:%M:%S')}")

    # 进行中的话题 + 教训（category=topic / lesson 的事实卡）
    try:
        _facts = getattr(stores, "facts", None)
        if _facts is not None:
            _topics = _facts.list_facts("topic")[:2]
            if _topics:
                ctx.topic_context = "[进行中] " + "；".join(
                    t["current_value"] for t in _topics
                )
            _lessons = _pick_lessons(stores, _facts, user_message, limit=3,
                                     scope=scope)
            if _lessons:
                ctx.lesson_context = "[教训] " + "；".join(_lessons)
    except Exception:
        pass

    # 近期状态（由 tone_snapshot.py 定期写入的事实卡）
    try:
        _stores_facts = getattr(stores, "facts", None)
        if _stores_facts is not None:
            _tone = _stores_facts.get("fact:context:recent_tone")
            if _tone and _tone.get("current_value"):
                ctx.tone_context = f"[近期状态] {_tone['current_value']}"
    except Exception:
        pass
    ns = _resolve_namespace(stores.agent_name, stores.namespace, partner)
    turn_count = _count_recent_turns(stores.dialogue_store)

    ctx.tier1_context = _build_tier1(
        stores.dialogue_store, stores.saw_buffer, stores.agent_name, scene_summary)

    # Directives must render even for brand-new sessions: pending learning /
    # decomposition items are the whole point of a fresh mission, and a
    # no-history early return would silently drop them.
    ctx.gap_context = _build_gap_context(stores.gaps)
    ctx.pending_context = _build_pending_context(stores.pending_store)

    if stores.degraded:
        ctx.embedder_banner = (
            "⚠ [EMBEDDER DEGRADED] 语义检索/去重/学习判定已降级为 SHA-256 hash（无 ONNX 模型）"
            + (f" — {stores.degraded_reason}" if stores.degraded_reason else "")
            + "。修复: 运行 ./download_model.sh 或 export MEMORY_MODEL_PATH=… 后重启。"
        )

    total_stored = _count_all_dialogue(stores.dialogue_store)
    if turn_count < 3 and total_stored == 0:
        return ctx  # no history at all — tier1 only

    if turn_count < 3:
        turn_count = 3

    _apply_standard_blocks(ctx, stores, user_message, ns, partner)

    # deep depth adds the heartbeat check: surface a second-pass nudge when a
    # high-weight memory is relevant to the current message.
    if turn_count > 10 and _has_high_weight(stores.learned_store, user_message):
        ctx.needs_second_pass = True
    return ctx


def _apply_standard_blocks(
    ctx: WeaveContext,
    stores: WeaverStores,
    user_message: str,
    ns: str,
    partner: str | None,
) -> None:
    """Populate the standard context slots shared by every non-compact weave.

    One call site instead of two copy-pasted branches; adding a context slot
    now means editing a single line here rather than both branches.
    """
    if user_message:
        ctx.tier2_context = _build_tier2(
            stores.retriever, stores.dialogue_store, stores.agent_name, user_message, ns)
        ctx.historic_hint = _build_historic_hint(stores.retriever, user_message)
    ctx.memory_nudge = _build_nudge(stores.learned_store, user_message)
    ctx.skill_context = _build_skill_context(stores.retriever, user_message)
    ctx.mission_context = _build_mission_context(stores.retriever, user_message)
    ctx.pref_context = _build_pref_context(stores.retriever)
    ctx.pers_context = _build_pers_context(stores.retriever)
    ctx.conclusion_context = _build_conclusion_context(stores.retriever)
    ctx.title_preview = _build_title_preview(stores.retriever)
    ctx.tree_context = _build_tree_context(stores.tree, user_message)
    ctx.tree_nav = _build_tree_nav(stores.tree, user_message)


# ── Tier builders ────────────────────────────────────

def _build_tier1(dialogue: DialogueSource, saw: SawBufferProtocol,
                 agent_name: str, scene_summary: str) -> str:
    parts: list[str] = []
    if scene_summary:
        parts.append(f"[当前场景] {scene_summary}")
    else:
        entries = saw.get_all()
        if entries:
            parts.append(f"[当前感知] {entries[-1].content[:_TIER1_MAX_CHARS]}")
    turns = dialogue.get_recent(_MAX_RECENT_TURNS)
    if turns:
        agent_label = agent_name or "助手"
        lines = [f"[{_fmt_local(t.timestamp)}] {'用户' if t.role == 'user' else agent_label}: {t.content[:80]}"
                 for t in turns[-_MAX_RECENT_TURNS:]]
        parts.append("[最近对话]\n" + "\n".join(lines))
    result = "\n".join(parts)
    return result[:_TIER1_MAX_CHARS + 100]


def _build_tier2(retriever: RetrievalSource, dialogue: DialogueSource,
                 agent_name: str, user_message: str, ns: str = "") -> str:
    """Build tier2 context — conversation units with surrounding dialogue.

    V9: Each retrieved fact is expanded into a conversation unit that shows
    the surrounding dialogue context (user → agent → agent → ...), not just
    the isolated fact.  This gives the agent the full conversational context
    of the memory, not a decontextualized snippet.
    """
    if not user_message:
        return ""
    try:
        if ns and ns != "default":
            envelope = retriever.retrieve(
                user_message, limit=4, filters={"category": ns})
        else:
            # Isolate unclassified fragments: semantic leg searches
            # structured memory only; BM25 leg still supplies the raw
            # dialogue units needed for context expansion.
            envelope = retriever.retrieve(
                user_message, limit=4,
                filters={"category": {"$ne": "default"}})
    except Exception as e:
        _logger.warning("Tier2 retrieval failed: %s", e)
        return ""
    if not envelope.entries:
        return ""

    units: list[str] = []
    seen_units: set[str] = set()
    ds = dialogue

    injected_ids: list[str] = []
    for e in envelope.entries:  # skip non-dialogue entries (e.g. persona card) to find a buildable unit
        turn_id = e.metadata.get("turn_id", "") if e.metadata else ""
        if not turn_id:
            # L4 摘要/其它非对话条目：以"模糊记忆"呈现（软遗忘表达）
            if (e.metadata or {}).get("source") == "episode":
                ts = _fmt_local(e.created_at)
                marker = _soft_forget_marker(e)
                units.append(f"[{ts}] {marker}（翻到当时大概是）{e.content[:300]}")
                injected_ids.append(e.id)
            continue

        # Get the target turn and surrounding dialogue
        target_turn = ds.get_by_id(turn_id)
        if not target_turn:
            continue

        # Find surrounding turns (±2 turns from target)
        recent = ds.get_recent(500)  # fetch enough for surrounding context
        target_idx = None
        for i, t in enumerate(recent):
            if t.id == turn_id:
                target_idx = i
                break
        if target_idx is None:
            continue

        # Build unit: always start with a user message, end with the next user
        # Walk backward to find the preceding user turn
        unit_start = target_idx
        while unit_start > 0:
            if recent[unit_start - 1].role == "user":
                unit_start -= 1
                break
            unit_start -= 1

        # Walk forward to find the next user turn (inclusive)
        unit_end = target_idx + 1
        while unit_end < len(recent):
            if recent[unit_end].role == "user":
                if unit_end > target_idx:
                    unit_end += 1  # include ending user turn
                break
            unit_end += 1
        unit_end = min(unit_end, len(recent))

        # Build unit lines (cap: 3 turns, 50 chars each)
        unit_lines: list[str] = []
        max_turns = min(unit_end, len(recent))
        for i in range(unit_start, min(unit_start + 3, max_turns)):
            t = recent[i]
            if t.role == "user":
                label = "User"
            elif t.role == "assistant":
                # Infer speaker from turn ID prefix (namespace)
                # e.g. "st_agent_a_123" → agent_a, "st_user_456" → user
                ns_match = t.id.split("_")[1] if t.id.startswith("st_") and t.id.count("_") >= 2 else ""
                label = ns_match if ns_match else (agent_name or "助手")
            else:
                label = t.role
            snippet = t.content[:50]
            unit_lines.append(f"{label}: {snippet}")

        unit_text = "\n".join(unit_lines)
        unit_key = unit_text[:80]
        if unit_key in seen_units:
            continue
        seen_units.add(unit_key)

        ts = _fmt_local(e.created_at)
        marker = _soft_forget_marker(e)
        units.append(f"[{ts}] {marker}[记忆片段]\n{unit_text}")
        injected_ids.append(e.id)

    # 注意力权重：被实际注入的条目做一次再巩固（越接近遗忘时增益越大）
    if injected_ids:
        _reinforce = getattr(retriever, "reinforce_injected", None)
        if _reinforce is not None:
            try:
                _reinforce(injected_ids)
            except Exception:
                pass

    if units:
        header = "以下是你经历的对话片段。你只能使用这些信息，不得编造:"
        return header + "\n\n" + "\n\n".join(units) + \
               "\n\n（如果你不确定某条信息是否在上面列出——就不要提及它。）"
    return ""


def _build_nudge(memory: MemorySource, user_message: str = "") -> str:
    """Build nudge with behavioral intensity based on weight.

    V7 upgrade:
      weight >= 0.95 → "⚠ 务必...——这件事很重要"
      weight >= 0.85 → "💡 可以...——让对话更自然"

    V9 (ADR-0001 candidate 8): nudge entries are gated on relevance to the
    current ``user_message`` via distinctive-token overlap — an unrelated
    high-weight entry (e.g. a stale "pip install failed" from a past
    session) is no longer injected every turn as noise.  When no
    ``user_message`` is given, the gate is skipped (keep historical
    behaviour so empty-context weaves still surface critical items).
    """
    try:
        entries = memory.search(
            "", limit=_NUDGE_MAX_ITEMS * 3,
            filters={"weight": {"$gte": _NUDGE_WEIGHT_THRESHOLD},
                     "category": {"$ne": "default"}})
    except Exception as e:
        _logger.debug("Nudge retrieval failed: %s", e)
        return ""

    # Isolate unclassified fragments: nudge should surface structured
    # memory (skills/prefs/pers), not raw dialogue noise.
    high = [e for e in entries
            if e.weight >= _NUDGE_WEIGHT_THRESHOLD
            and e.category != "default"]
    if not high:
        return ""

    if user_message.strip():
        high = [e for e in high if token_overlap(user_message, e.content)]
        if not high:
            return ""

    # Sort by weight descending
    high.sort(key=lambda e: e.weight, reverse=True)

    lines: list[str] = []
    for e in high[:_NUDGE_MAX_ITEMS]:
        snippet = e.content[:_NUDGE_MAX_CHARS]
        ts = _fmt_local(e.created_at)
        if e.weight >= _NUDGE_CRITICAL_THRESHOLD:
            lines.append(f"[{ts}] ⚠ {snippet}——务必主动提及，这件事很重要")
        else:
            lines.append(f"[{ts}] 💡 {snippet}——可以自然提到，让对话更顺畅")
    return " | ".join(lines) if lines else ""


# ── Helpers ──────────────────────────────────────────

def _resolve_namespace(agent_name: str, namespace: str, partner: str | None) -> str:
    if agent_name:
        return agent_name
    return namespace


def _count_all_dialogue(dialogue: DialogueSource) -> int:
    """Count total stored dialogue turns (across all sessions)."""
    try:
        return dialogue.count()
    except Exception as e:
        _logger.debug("Total dialogue count failed: %s", e)
        return 0


def _count_recent_turns(dialogue: DialogueSource) -> int:
    """Count turns from the last 5 minutes — approximates "current session"."""
    try:
        return dialogue.count_recent(minutes=5)
    except Exception as e:
        _logger.debug("Recent turn count failed: %s", e)
        return 0


def _has_high_weight(memory: MemorySource, user_message: str = "") -> bool:
    """True when a relevant high-weight entry exists (needs_second_pass gate).

    Mirrors the candidate-8 relevance gate of ``_build_nudge`` so the deep
    branch's second-pass flag agrees with what nudge would actually surface.
    """
    try:
        entries = memory.search(
            "", limit=5,
            filters={"weight": {"$gte": _NUDGE_WEIGHT_THRESHOLD},
                     "category": {"$ne": "default"}})
        high = [e for e in entries
                if e.weight >= _NUDGE_WEIGHT_THRESHOLD
                and e.category != "default"]
        if not high:
            return False
        if user_message.strip():
            return any(token_overlap(user_message, e.content) for e in high)
        return True
    except Exception as e:
        _logger.debug("High-weight check failed: %s", e)
        return False


def _build_tree_context(tree, user_message: str) -> str:
    """Build tree context from the memory tree for agent injection.

    Delegates to TreeManager.get_context() which finds relevant branches
    by label similarity to the query and returns siblings + parent labels.
    """
    if not tree or not user_message:
        return ""
    try:
        return tree.get_context(user_message)
    except Exception as e:
        _logger.debug("Tree context retrieval failed: %s", e)
        return ""


def _build_tree_nav(tree, user_message: str) -> str:
    """Build deterministic tree context in parallel with RRF retrieval.

    Uses the LLM-free navigation path so weave() never triggers an internal
    LLM call (ADR-0002: weave is pure context assembly). Agents that want
    LLM-selected branches call TreeManager.navigate() explicitly via a tool.
    """
    if not tree or not user_message:
        return ""
    try:
        return tree.navigate_without_llm()
    except Exception as e:
        _logger.debug("Tree navigation failed: %s", e)
        return ""


def _safe_retrieve(
    retriever: RetrievalSource,
    query: str,
    category: object,
    limit: int,
):
    """Retrieve with the category filter, tolerating backend failures.

    The category-slot builders share this scaffold (guard + filter +
    empty-check) and differ only in how they render their hits. One
    implementation, six call sites.
    """
    try:
        result = retriever.retrieve(query, limit=limit, filters={"category": category})
    except Exception:
        return []
    return result.entries if result.entries else []


def _build_skill_context(retriever: RetrievalSource, user_message: str) -> str:
    """Retrieve relevant skill titles for agent awareness."""
    if not user_message:
        return ""
    entries = _safe_retrieve(retriever, user_message, "skill", 5)
    titles = []
    for e in entries[:5]:
        t = e.content.split('\n')[0].lstrip('# ')[:40]
        if t and t not in titles:
            titles.append(t)
    if not titles:
        return ""
    return "[已掌握的技能]\n  · " + "\n  · ".join(titles)


def _build_mission_context(retriever: RetrievalSource, user_message: str) -> str:
    entries = _safe_retrieve(retriever, user_message, "mission", 2)
    if not entries:
        return ""
    import re
    lines = ["[当前任务]"]
    for e in entries[:2]:
        title = e.content.split('\n')[0].lstrip('# ')[:50]
        lines.append(f"\n→ {title}")
        steps = re.split(r'\n##\s*', '\n' + e.content)
        for step_block in steps[1:]:
            step_lines = step_block.strip().split('\n')
            header = step_lines[0]
            m = re.match(r'(.+?)\s*\[(done|doing|pending)\]', header)
            if not m:
                continue
            step_name, step_status = m.group(1).strip(), m.group(2)
            marks = {'done': '✓', 'doing': '●', 'pending': '○'}
            mark = marks.get(step_status, '○')
            skill_name = ""
            for sl in step_lines:
                if sl.strip().startswith('skill:'):
                    skill_name = sl.split(':', 1)[1].strip()
                    break
            skill_hint = ""
            if skill_name:
                has = retriever.retrieve(skill_name, limit=5, filters={"category": "skill"})
                found = False
                for sk in has.entries:
                    if skill_name.lower() in sk.content.lower()[:100]:
                        found = True
                        break
                skill_hint = f" → {skill_name} {'✅' if found else '⚠'}"
            lines.append(f"  {mark} {step_name}{skill_hint}")
    return "\n".join(lines)


def _build_gap_context(gaps: list) -> str:
    """Render open learning-queue items as explicit agent directives.

    Unlike the legacy passive gap display, these are actionable commands
    for the main agent: learn an unknown skill, or decompose a mission
    into steps before executing it.  The agent is expected to act on them
    (check_skill → web search → teach_skill; mission_decompose) and to
    confirm the result with the user.
    """
    if not gaps:
        return ""
    skill_items = [g for g in gaps if getattr(g, "kind", "") == "skill"]
    mission_items = [g for g in gaps if getattr(g, "kind", "") == "mission"]
    other = [g for g in gaps if getattr(g, "kind", "") not in ("skill", "mission")]

    lines: list[str] = []

    def _query(g) -> str:
        return getattr(g, "query", "") or str(g)[:80]

    if skill_items:
        lines.append("[待学习] 先 websearch 搜索 → 找到权威来源 → "
                     "memory_teach_skill(必须带 source_urls) → 反馈用户确认：")
        for g in skill_items[:5]:
            lines.append(f"  📚 {_query(g)}")

    if mission_items:
        lines.append("[待拆解] 以下任务尚未分解为步骤（由主 agent 自行分析拆解）：")
        for g in mission_items[:5]:
            lines.append(f"  📋 {_query(g)}")

    if other:
        lines.append("[待处理] " + "; ".join(_query(g) for g in other[:3]))

    # These directives are enforced: skipping them causes the next weave to
    # be rejected by ProtocolGate, so the agent must act before continuing.
    lines.append("⚠ 上述指令由协议硬门强制执行——不执行则下一轮记忆注入被拒绝。")

    return "\n".join(lines)


def _build_pending_context(pending_store) -> str:
    """Surface distill candidates awaiting agent review (fixes agent drift)."""
    if pending_store is None:
        return ""
    try:
        items = pending_store.list_open(limit=5)
    except Exception:
        return ""
    if not items:
        return ""
    lines = ["[待审核提炼] 以下对话碎片候选待主 agent 判断是否值得沉淀："]
    for c in items:
        lines.append(f"  · {c.topic}（建议 {c.suggested}）"
                     f"——memory_pending 查看证据")
    return "\n".join(lines)


def _build_pref_context(retriever: RetrievalSource) -> str:
    entries = _safe_retrieve(retriever, "all", "pref", 10)
    if not entries:
        return ""
    lines = ["[用户偏好]"]
    for e in entries[-10:]:
        lines.append(f"  · {e.content}")
    return "\n".join(lines)


def _build_pers_context(retriever: RetrievalSource) -> str:
    entries = _safe_retrieve(retriever, "all", "pers", 10)
    if not entries:
        return ""
    cards = [e for e in entries if e.content.startswith('# ')]
    latest = max(cards, key=lambda e: len(e.content)) if cards else entries[-1]
    return "[人格特征]\n  · " + latest.content


def _build_conclusion_context(retriever: RetrievalSource) -> str:
    """Dedicated injection channel for conclusion entries.

    Conclusions (knowledge/root-cause judgments) previously shared the
    unfiltered [近期记忆] slot and had no guaranteed exposure.  This
    gives them their own slot, newest first.
    """
    entries = _safe_retrieve(retriever, "all", "conclusion", 10)
    if not entries:
        return ""
    lines = ["[历史结论]"]
    for e in entries[-3:]:
        title = e.metadata.get("title", "") if hasattr(e, "metadata") and e.metadata else ""
        content = title if title else e.content[:60].replace("\n", " ")
        lines.append(f"  · {content}")
    return "\n".join(lines)


def _build_title_preview(retriever: RetrievalSource) -> str:
    try:
        result = retriever.retrieve(
            "all", limit=5, filters={"category": {"$ne": "default"}})
    except Exception:
        return ""
    entries = result.entries if result.entries else []
    if not entries:
        return ""
    lines = ["[近期记忆]"]
    for e in entries[:5]:
        title = e.metadata.get("title", "") if hasattr(e, "metadata") and e.metadata else ""
        content = title if title else e.content[:50].replace("\n", " ")
        lines.append(f"  · {content}")
    return "\n".join(lines)


# Entries ingested in the last N minutes are this session's own turns —
# hinting at them would tell the agent "you just said that". Skip them.
_HISTORIC_HINT_RECENT_SKIP_MINUTES: int = 10

# Cap hint content — a trigger, not a dump.
_HISTORIC_HINT_MAX_TITLE: int = 40


def _build_historic_hint(retriever: RetrievalSource, user_message: str) -> str:
    """Detect a strongly-related past memory and hint the agent to
    actively `memory_search` it — WITHOUT injecting its content.

    Passive weave context is noise when the agent has no need for it;
    a hint that *named* past work (title + date) creates a genuine
    information need, so the agent's own memory_search result is
    treated as relevant rather than ignored.

    Relevance requires BOTH a strong semantic score AND a shared
    distinctive token — same double-corroboration as KNOWN-ISSUES #1.
    Pure semantic thresholds misfire on short Chinese sentences (bge
    inflates cosine for token-sparse queries), so a bare 0.72 cosine
    would hint at unrelated recent turns ("今天天气怎么样" → "重启了").
    """
    if not user_message:
        return ""
    try:
        result = retriever.retrieve(user_message, limit=6)
    except Exception as e:
        _logger.debug("Historic hint retrieval failed: %s", e)
        return ""
    if not result.entries:
        return ""

    now = datetime.now(UTC)
    for e in result.entries:
        if not e.semantic_score or e.semantic_score < _SEM_CORROBORATED:
            continue
        if e.created_at and (now - e.created_at).total_seconds() < (
            _HISTORIC_HINT_RECENT_SKIP_MINUTES * 60
        ):
            continue
        if not token_overlap(user_message, e.content):
            continue
        title = e.content.split("\n")[0].lstrip("# ")[:_HISTORIC_HINT_MAX_TITLE]
        date = _fmt_local(e.created_at, "%m-%d") if e.created_at else "过去"
        return (
            f"[历史相关] 你在 {date} 处理过「{title}」——"
            "需要当时的细节/结论吗?用 memory_search 主动检索"
        )
    return ""
