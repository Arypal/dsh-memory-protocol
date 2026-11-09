"""Composition factory — assembles all stores into a MemorySystem.

Replaces the god-object MemorySkill class with a plain dataclass
whose store attributes are public.  Entry points can access stores
directly; helper methods remain for convenience.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime

from memory_skill.contracts import (
    DialogueStoreProtocol,
    EmbedderProtocol,
    LearnedStoreProtocol,
    MemorySkillConfig,
    SawBufferProtocol,
    TreeManagerProtocol,
)
from memory_skill.protocol_state import ProtocolState

_logger = logging.getLogger(__name__)


class ClassificationRequired(Exception):
    """Raised by weave() when the previous turn has not been classified."""


class GapRequired(Exception):
    """Raised by weave() when classified mission has unfulfilled skill gaps."""


class SkillCheckRequired(Exception):
    """Raised by weave() when a classified mission has not yet checked existing skills."""


@dataclass(eq=False)
class MemorySystem:
    """All memory stores in one place — no hidden state.

    ``eq=False`` keeps identity semantics (and a working ``__hash__``):
    instances are used as ``WeakKeyDictionary`` keys in ``tools.py``, and
    weakrefs delegate hashing to the referent.  Value equality across the
    composite stores is meaningless and was never relied upon.
    """

    # All fields have defaults — the real constructor is __init__
    config: MemorySkillConfig | None = None
    embedder: object = None
    saw_buffer: object = None
    dialogue_store: object = None
    learned_store: object = None
    tree: object = None
    ingestor: object = None
    retriever: object = None
    learning_queue: object = None
    pending_store: object = None
    facts: object = None
    alias_index: object = None
    entity_stack: object = None
    retention: object = None
    protocol: object = None
    _composed_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Backward-compatible view of the protocol state. New code should go
    # through ``self.protocol``; these properties keep existing callers and
    # tests working while the migration lands.
    @property
    def _classify_pending(self):
        return self.protocol.classify_pending if self.protocol else None

    @_classify_pending.setter
    def _classify_pending(self, value):
        if self.protocol:
            self.protocol.classify_pending = value

    @property
    def _pending_gaps(self):
        return self.protocol.pending_gaps if self.protocol else set()

    @_pending_gaps.setter
    def _pending_gaps(self, value):
        if self.protocol:
            self.protocol.pending_gaps = value

    @property
    def _mission_pending_check(self):
        return self.protocol.mission_pending_check if self.protocol else False

    @_mission_pending_check.setter
    def _mission_pending_check(self, value):
        if self.protocol:
            self.protocol.mission_pending_check = value

    def __init__(self, config: MemorySkillConfig | None = None, **overrides):
        """Create via ``MemorySystem(config)`` or ``MemorySystem(field=value).``

        When called with a single config, auto-composes all stores.
        Use keyword args to override specific fields (for testing).
        """
        if config is not None:
            ms = _build_system(config)
            self.__dict__.update(ms.__dict__)
        # Apply overrides (from dataclass field defaults)
        for k, v in overrides.items():
            setattr(self, k, v)
        if "_composed_at" not in self.__dict__:
            self._composed_at = datetime.now(UTC)

    # ── Convenience methods (thin delegates, no new logic) ──────────────

    def ingest(self, turn, *, enrich: bool = True, report: bool = False,
               scope: str = "web") -> object:
        """Persist a dialogue turn.  Pure storage — no LLM stages.

        Classification, title generation, conclusion extraction, and
        mission decomposition are the main agent's responsibility.
        This method only writes the turn to both stores (dialogue + learned).
        """
        receipt = self.ingestor.ingest_dialogue(turn, scope=scope)
        try:
            self.entity_stack.tick()
        except Exception:
            pass
        self._link_fact_mentions(turn)

        # 每 50 轮检查一次保留策略（阈值宽松时是空操作）
        self._ingest_count = getattr(self, "_ingest_count", 0) + 1
        if self._ingest_count % 50 == 0:
            try:
                self.retention.prune()
            except Exception:
                pass
        return receipt

    def retrieve(self, query: str, limit: int = 10, filters=None,
                  partner: str | None = None, scope: str = "web",
                  since_days: int | None = None,
                  date_from: str | None = None,
                  date_to: str | None = None):
        if partner and not filters:
            ns = self._ns_for(partner)
            filters = {"category": ns}
        # 代词消解：把锚定实体拼进查询，让"它是不是病了"能查到"大橘"
        expanded = query
        try:
            anchor = self.entity_stack.resolve(query)
            if anchor and anchor["name"] not in query:
                expanded = f"{anchor['name']} {query}"
        except Exception:
            pass
        return self.retriever.retrieve(expanded, limit=limit, filters=filters,
                                       scope=scope,
                                       since_days=since_days,
                                       date_from=date_from,
                                       date_to=date_to)

    def boost_weight(self, entry_id: str, delta: float = 0.05, cap: float = 0.95) -> float:
        """Boost a memory entry's weight (delegates to learned_store)."""
        return self.learned_store.boost_weight(entry_id, delta, cap)

    def ingest_screen(self, entry) -> object:
        return self.ingestor.ingest_screen(
            entry.text if hasattr(entry, 'text') else str(entry),
        )

    def weave(self, user_message: str, scene_summary: str = "",
              partner: str | None = None, scope: str = "web"):
        """Assemble layered memory context (convenience wrapper).

        The protocol gate is advisory: a missing classification or an open
        mission/gap only adds a ``[协议提醒]`` line to the context. Weaving
        never fails because of protocol state (see protocol_gate docstring).
        """
        from memory_skill.protocol_gate import ProtocolGate
        from memory_skill.weaver import WeaverStores, weave

        gate = ProtocolGate(self)
        notice = gate.check(user_message)

        stores = WeaverStores(
            saw_buffer=self.saw_buffer,
            dialogue_store=self.dialogue_store,
            learned_store=self.learned_store,
            retriever=self.retriever,
            agent_name=self.config.agent_name,
            namespace=self.config.namespace,
            tree=self.tree,
            gaps=self.gaps,
            pending_store=getattr(self, 'pending_store', None),
            facts=getattr(self, 'facts', None),
            degraded=self.embedder.degraded,
            degraded_reason=self.embedder.reason,
        )
        ctx = weave(stores, user_message, scene_summary, partner=partner,
                    scope=scope)
        if notice:
            ctx.protocol_notice = notice
        gate.after_weave(user_message)
        return ctx

    def health(self) -> dict:
        self.embedder.embed("health")  # trigger lazy-load
        return {
            "status": "healthy",
            "model_ok": not self.embedder.degraded,
            "embedder": {
                "loaded": getattr(self.embedder, '_load_attempted', True),
                "mode": self.embedder.mode,
                "dim": self.config.embedding_dim,
                "degraded": self.embedder.degraded,
                "reason": self.embedder.reason,
                "model_path": self.config.model_path,
            },
            "saw_buffer": {
                # 预留能力，当前环境未启用：DSH 没有屏幕采集入口，
                # ingest_screen() 从未被调用，dialogue_turns.saw_index 恒为 NULL。
                # 这里显式写 active=False，避免 entry_count/capacity 被误读成
                # 「屏幕关联机制正在工作」。接入屏幕采集后把 active 改成 True 即可。
                "active": False,
                "unused_reason": "no screen-capture source in DSH; ingest_screen() is never called",
                "entry_count": len(self.saw_buffer.get_all()),
                "capacity": self.config.saw_buffer_capacity,
            },
            "learned_store": self.learned_store.health(),
            "importance_llm": {
                "configured": self._importance_key_configured(),
                "note": "distill/importance LLM 评分需要 IMPORTANCE_API_KEY（~/.config/memory-skill/.env），未配置时主动学习 distill 不可用，记忆核心不受影响",
            },
            "config": {
                "namespace": self.config.namespace,
                "db_path": self.config.db_path,
            },
        }

    def _importance_key_configured(self) -> bool:
        """True when IMPORTANCE_API_KEY is available (env or the two .env locations)."""
        try:
            from pathlib import Path
            from dotenv import load_dotenv
            load_dotenv(Path.home() / ".config" / "memory-skill" / ".env", override=False)
            load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
        except Exception:
            pass
        return bool(os.environ.get("IMPORTANCE_API_KEY", ""))

    def consolidate(self) -> int:
        from memory_skill.observation import ObservationConsolidator
        return ObservationConsolidator(
            self.dialogue_store, self.learned_store,
        ).run()

    def clean(self) -> dict:
        from memory_skill.cleaner import MemoryCleaner
        return MemoryCleaner(
            self.learned_store, self.embedder, self.config,
        ).run()

    def reflect(self):
        from memory_skill.reflect import reflect as _reflect
        return _reflect(self.dialogue_store, self.learned_store)

    def _link_fact_mentions(self, turn) -> None:
        """主卡/副卡判定 + 自动 mention + 实体入栈（方案 C 的指代与去噪层）。

        规则：
          1. 代词消解（窗口内），锚定实体优先成为主卡；
          2. 候选 = 别名精确命中 ∪ 语义相似（>= 0.55）；
          3. 主卡一张；副卡最多两张，且需 >= 0.70，只记 mention 不改值；
          4. 提到过的实体入栈，供后续代词消解使用。
        """
        from memory_skill.entity_stack import infer_type

        text = (getattr(turn, "content", "") or "").strip()
        if not text:
            return

        # 1) 代词消解
        anchor = None
        try:
            anchor = self.entity_stack.resolve(text)
        except Exception:
            anchor = None

        # 2) 候选打分
        scored: dict[str, float] = {}
        try:
            for eid in self.alias_index.hits(text):
                if eid.startswith("fact:"):
                    scored[eid] = 0.95
        except Exception:
            pass
        try:
            vec = self.embedder.embed(text)
            res = self.learned_store._collection.query(
                query_embeddings=[vec], n_results=3
            )
            for eid, dist in zip(
                (res.get("ids") or [[]])[0], (res.get("distances") or [[]])[0]
            ):
                eid = str(eid)
                if eid.startswith("fact:"):
                    scored[eid] = max(scored.get(eid, 0.0), 1.0 - float(dist))
        except Exception:
            pass

        if not scored:
            return

        # 3) 主卡
        primary = None
        if anchor and anchor["entity_id"] in scored:
            primary = anchor["entity_id"]
        else:
            # 同分时取"别名在句子中先出现"的那个（通常就是主语）
            pos: dict[str, int] = {e: 10 ** 6 for e in scored}
            try:
                for alias, eid in self.alias_index.list_all():
                    if eid in pos:
                        p = text.find(alias)
                        if 0 <= p < pos[eid]:
                            pos[eid] = p
            except Exception:
                pass
            best = min(scored, key=lambda e: (pos.get(e, 10 ** 6), -scored[e]))
            if scored[best] >= 0.55:
                primary = best
        if not primary:
            return

        role = getattr(turn, "role", "user") or "user"
        turn_id = getattr(turn, "id", None)
        turn_ts = getattr(turn, "timestamp", None)
        self.facts.append_mention(primary, text, turn_id, role,
                                  source_said_at=turn_ts)

        # 4) 副卡
        others = sorted(
            ((e, s) for e, s in scored.items() if e != primary and s >= 0.70),
            key=lambda x: -x[1],
        )[:2]
        for eid, _score in others:
            self.facts.append_mention(
                eid, text, turn_id, role, secondary=True,
                source_said_at=turn_ts,
            )

        # 5) 实体入栈
        for eid in [primary] + [e for e, _ in others]:
            try:
                fact = self.facts.get(eid)
                if not fact:
                    continue
                self.entity_stack.push(
                    eid, fact["current_value"],
                    infer_type(fact["key"]), turn_id,
                )
            except Exception:
                pass

    def ingest_fact(self, category: str, key: str, value: str,
                    aliases: list | None = None, quote: str = "",
                    turn_id: str | None = None, speaker: str = "user",
                    source_said_at=None, scope: str = "web") -> dict:
        """建立/更新一张事实卡，自动生成双语别名并追加证据。"""
        return self.facts.upsert(
            category=category, key=key, value=value,
            aliases=aliases, quote=quote, turn_id=turn_id, speaker=speaker,
            source_said_at=source_said_at, scope=scope,
        )

    def fact_timeline(self, key_or_id: str, limit: int = 50,
                      scope: str = "web") -> dict:
        """按 key 或 fact_id 返回事实、历史值与证据时间线。"""
        fact = self.facts.get(key_or_id) or self.facts.find_by_key(
            key_or_id, scope=scope
        )
        if not fact:
            return {"status": "not_found", "query": key_or_id}
        return {
            "status": "ok",
            "fact": fact,
            "history": self.facts.history(fact["id"]),
            "timeline": self.facts.timeline(fact["id"], limit),
        }

    def prune_dialogue(self, ttl_days=None, max_turns=None,
                       dry_run: bool = False) -> dict:
        """清理过期对话（L3）。清理前自动生成区间摘要（L4）。"""
        return self.retention.prune(
            ttl_days=ttl_days, max_turns=max_turns, dry_run=dry_run
        )

    def episodes(self, limit: int = 20) -> list:
        """列出已生成的区间摘要（L4）。"""
        return self.retention.list_episodes(limit)

    def ensure_embedder_loaded(self) -> str:
        emb = self.embedder
        emb._load_attempted = False
        emb._ensure_model_loaded()
        return "onnx" if emb._session else "fallback"

    def auto_context(self, messages: list) -> list:
        """Inject memory context before the last user message."""
        if not messages:
            return messages

        try:
            ctx = self.weave(
                user_message=(
                    messages[-1]["content"]
                    if messages and messages[-1].get("role") == "user"
                    else ""
                ),
                scene_summary="ongoing conversation",
            )
            if not ctx.is_empty:
                block = ctx.to_prompt_block()
                if block and messages and messages[-1].get("role") == "user":
                    messages = list(messages)
                    messages[-1] = dict(messages[-1])
                    messages[-1]["content"] = f"{block}\n\n{messages[-1]['content']}"
        except Exception as exc:
            _logger.warning("auto_context failed: %s", exc)

        return messages

    def auto_ingest(self, user_msg: str, assistant_msg: str) -> None:
        """Persist both sides of a conversation turn via the ingest pipeline.

        Uses ``enrich=False`` (no LLM stages) — this is the high-frequency
        path (transparent proxy). Truncation follows the unified head+tail
        policy (see ADR-0001).
        """
        from memory_skill.contracts import DialogueTurn, clip_auto_ingest

        now = datetime.now(UTC)
        for role, content in [("user", user_msg), ("assistant", assistant_msg)]:
            if not content:
                continue
            try:
                turn = DialogueTurn(
                    id=f"auto_{now:%Y%m%d_%H%M%S}_{hash(content) & 0xFFFF:04x}",
                    role=role,
                    content=clip_auto_ingest(content),
                    timestamp=now,
                )
                self.ingest(turn, enrich=False)
            except Exception as exc:
                _logger.warning("auto_ingest failed for %s: %s", role, exc)

    @property
    def gaps(self) -> list:
        return self.ingestor.gaps

    def ingest_skill(self, title: str, content: str, source_urls: list[str] | None = None) -> object:
        from memory_skill.memory_extract import ingest_skill_ex
        return ingest_skill_ex(self, title, content, source_urls=source_urls)

    def ingest_pers(self, trait: str) -> object:
        from memory_skill.memory_extract import ingest_pers as _ip
        return _ip(self, trait)

    def ingest_pref(self, key: str, value: str) -> object:
        from memory_skill.memory_extract import ingest_pref as _ipr
        return _ipr(self, key, value)

    def count_turns(self) -> int:
        return self.dialogue_store.count()

    def get_turn(self, turn_id: str):
        return self.dialogue_store.get_by_id(turn_id)

    def get_recent_turns(self, n: int = 10):
        return self.dialogue_store.get_recent(n)

    def expand(self, message: str, limit: int = 5) -> str:
        """Expand a memory reference into its full context.

        Searches for matching titles, then returns the matched entry
        plus temporally nearby entries (same time window).
        """
        if not message:
            return ""
        results = self.retriever.retrieve(message, limit=2)
        if not results.entries:
            return ""
        match = results.entries[0]
        m = getattr(match, "metadata", {}) or {}
        title = m.get("title", match.content[:40])

        from datetime import timedelta
        ts = getattr(match, "created_at", None) or getattr(match, "updated_at", None)
        if ts:
            window_start = ts - timedelta(hours=2)
            window_end = ts + timedelta(hours=2)
            nearby = self.dialogue_store.time_range(window_start, window_end)
            if nearby:
                lines = [f"[扩展记忆 — {title}]"]
                for t in nearby[:limit]:
                    lines.append(f"  · {t.content[:100]}")
                return "\n".join(lines)
        return f"[扩展记忆 — {title}]\n  · {match.content[:200]}"

    def _ns_for(self, partner: str | None = None) -> str:
        return self.config.agent_name


# ── Factory ─────────────────────────────────────────────────────────────


def _build_system(config: MemorySkillConfig) -> MemorySystem:
    """Create a fully assembled MemorySystem from config."""
    from memory_skill.dialogue_store import DialogueStore
    from memory_skill.embedder import Embedder
    from memory_skill.ingestor import Ingestor
    from memory_skill.learned_store import LearnedStore
    from memory_skill.pending_store import PendingStore
    from memory_skill.retriever import Retriever
    from memory_skill.saw_buffer import SawRingBuffer

    embedder = Embedder(config)
    saw_buffer = SawRingBuffer(capacity=config.saw_buffer_capacity)
    dialogue_store = DialogueStore(config)
    learned_store = LearnedStore(config, embedder,
                                 chroma_path=config.db_path + "_chroma")

    from memory_skill.alias_index import AliasIndex
    from memory_skill.entity_stack import EntityStack
    from memory_skill.fact_store import FactStore

    alias_index = AliasIndex(config.db_path)
    facts = FactStore(config.db_path, alias_index=alias_index,
                      learned_store=learned_store)
    entity_stack = EntityStack()
    from memory_skill.retention import RetentionManager

    retention = RetentionManager(
        config.db_path,
        dialogue_store=dialogue_store,
        learned_store=learned_store,
        alias_index=alias_index,
        config=config,
    )
    tree = _build_tree(config) if config.tree_enabled else None
    retriever = Retriever(config=config, dialogue_store=dialogue_store,
                          learned_store=learned_store,
                          episode_source=retention)
    learning_queue = _build_learning_queue(config) if config.tree_enabled else None
    pending_store = PendingStore(config.db_path) if config.tree_enabled else None
    ingestor = Ingestor(config=config, saw_buffer=saw_buffer,
                        dialogue_store=dialogue_store,
                        learned_store=learned_store, embedder=embedder,
                        tree=tree, learning_queue=learning_queue)

    ms = object.__new__(MemorySystem)
    ms.__dict__.update(
        config=config, embedder=embedder, saw_buffer=saw_buffer,
        dialogue_store=dialogue_store, learned_store=learned_store,
        tree=tree, ingestor=ingestor, retriever=retriever,
        learning_queue=learning_queue, pending_store=pending_store,
        facts=facts, alias_index=alias_index, entity_stack=entity_stack,
        retention=retention,
        protocol=ProtocolState(),
        _composed_at=datetime.now(UTC),
    )
    return ms


def _tree_creds() -> tuple[str, str, str]:
    return (
        os.getenv("IMPORTANCE_API_BASE", "https://api.deepseek.com/v1"),
        os.getenv("IMPORTANCE_API_KEY", ""),
        os.getenv("IMPORTANCE_MODEL", "deepseek-v4-flash"),
    )


def _build_tree(config: MemorySkillConfig):
    from memory_skill.tree import TreeManager
    api_base, api_key, model = _tree_creds()
    return TreeManager(db_path=config.db_path,
                       api_base=api_base, api_key=api_key, model=model)


def _build_learning_queue(config: MemorySkillConfig):
    from memory_skill.learning_queue import LearningQueue
    return LearningQueue(db_path=config.db_path)


create = _build_system
