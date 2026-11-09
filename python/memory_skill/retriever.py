"""
Memory Skill — Retriever (3-signal fusion + RRF)

Combines three retrieval signals — semantic (vector), BM25 (FTS5), and
temporal (recency decay) — into a single ranked result set via Reciprocal
Rank Fusion (RRF k=60).

Algorithm:
  1. Semantic:  ``learned_store.search(query, top_k=50)``
  2. BM25:      ``dialogue_store.search(query, limit=50)``
   3. Temporal:  ``weight × exp(-λ × age_hours)`` on all candidates
  4. RRF k=60:  weighted combination across all 3 ranked lists
  5. Top-N:     take the highest-scoring entries → MemoryEnvelope

Weights (per Task 4 calibration):
  semantic = 1.5,  BM25 = 1.5,  temporal = 0.5

Usage::

    from memory_skill.contracts import MemorySkillConfig
    from memory_skill.dialogue_store import DialogueStore
    from memory_skill.learned_store import LearnedStore
    from memory_skill.retriever import Retriever

    cfg = MemorySkillConfig()
    ds  = DialogueStore(cfg)
    ls  = LearnedStore(cfg, emb)
    ret = Retriever(cfg, ds, ls)

    envelope = ret.retrieve("What does the user prefer?")
    for entry in envelope.entries:
        print(entry.content)
"""

from __future__ import annotations

import functools
import logging
import math
import re
from collections import OrderedDict
from datetime import datetime
from typing import TYPE_CHECKING, Any

from memory_skill.contracts import (
    DialogueStoreProtocol,
    LearnedStoreProtocol,
    utcnow,
)

if TYPE_CHECKING:
    from memory_skill.contracts import (
        DialogueTurn,
        MemoryEntry,
        MemoryEnvelope,
        MemorySkillConfig,
    )


_logger = logging.getLogger(__name__)

# Broad-search fetch size before RRF fusion
_BROAD_TOP_K: int = 50

# ── Retrieval quality floors (diagnosis 2026-10-24) ───────────────────────
# Measured noise baseline for bge-large-zh on unrelated Chinese queries is
# ~0.32, so anything below 0.30 is noise and must not enter the candidate set.
_MIN_SEMANTIC_SCORE: float = 0.30
# Fact cards are exempt from bigram matching (their content is a structured
# key plus aliases), so they need a higher vector floor of their own to stop
# unrelated technical queries from surfacing them.
_FACT_MIN_SCORE: float = 0.45
# A fact card this relevant is a direct answer and must outrank dialogue
# passages. RRF fuses ranks only, so a fact card matching one leg perfectly
# used to lose to a long passage that matched three legs.
_FACT_PIN_MIN_SCORE: float = 0.35
_MAX_FACT_PINS: int = 3
# A fact card states something durable; an event records what happened. Letting
# facts take every slot crowds out the episode a question is actually about
# (fish-memory's factsMax idea). Facts may occupy at most this many slots.
_MAX_FACT_SLOTS: int = 2
# Skill cards are long procedural documents; their vector similarity to a
# short question is diluted, so the dedicated skill leg uses a lower floor.
_SKILL_MIN_SCORE: float = 0.40
# Weight of the skill leg in RRF fusion (between semantic 1.0 and alias 3.5).
_SKILL_LEG_WEIGHT: float = 2.0
# Long passages are semantic black holes: their mean-pooled embedding drifts
# toward every technical topic, and they win BM25 by sheer token count.
_LONG_CONTENT_CHARS: int = 800
_LONG_CONTENT_FACTOR: float = 0.7
# A near-exact vector match is evidence in itself. Rank-only fusion otherwise
# lets a high-weight fact card outrank the literal answer ("大橘最近不吃饭"
# losing to cat_name, which merely shares the alias).
_EXACT_SEM: float = 0.85
_EXACT_SEM_BONUS: float = 0.12
# Jaccard overlap of segmented content words above which two candidates are
# treated as the same memory and only the higher-ranked one is kept.
_RETRIEVAL_DUP_RATIO: float = 0.85
# Fact cards are pinned only while they keep pace with the best dialogue hit:
# a passage that clearly beats the card is a more precise answer.
_FACT_PIN_TOLERANCE: float = 0.05

# ── Lexical corroboration (noise control, 2026-10-24) ────────────────────
# bge-large-zh clusters Chinese technical text tightly: an unrelated technical
# query still lands around 0.41-0.42, above any usable similarity floor. Below
# _LOW_CONF_SEM a hit must also share literal Chinese text with the query.
_LOW_CONF_SEM: float = 0.45

_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_WORD_RE = re.compile(r"[a-zA-Z]{4,}")

# Function words that must never count as lexical evidence.
_STOP_CJK_BIGRAMS = frozenset({
    "什么", "怎么", "哪个", "哪些", "如何", "这个", "那个", "我们", "你们",
    "他们", "一个", "可以", "没有", "就是", "这是", "那是", "么样", "怎样",
    "什么", "为什", "现在", "时候", "一下", "还是",
})

# Content words that carry no topical evidence: question scaffolding and
# vocabulary every technical passage in this corpus contains. Keeping them
# would let "Python 装饰器" match any note that merely mentions Python.
_STOP_WORDS_CN = frozenset({
    "什么", "怎么", "哪个", "哪些", "如何", "现在", "可以", "没有", "就是",
    "这个", "那个", "我们", "你们", "他们", "一个", "一下", "还是", "已经",
    "但是", "因为", "所以", "如果", "然后", "就是", "不是", "什么", "怎样",
    "东西", "内容", "问题", "方法", "方式", "时候", "地方",
    # generic technical vocabulary
    "python", "dsh", "js", "json", "api", "http", "github", "cli",
    "记忆", "插件", "模型", "系统", "功能", "测试", "安装", "配置", "数据",
    "文件", "工具", "项目", "版本", "使用", "支持", "需要", "帮我",
})


@functools.lru_cache(maxsize=2048)
def _content_words(text: str) -> frozenset[str]:
    """Segmented content words of *text*, stop words removed.

    Uses jieba (already a hard dependency of this backend) and caches per
    string: retrieval re-segments the same passage once per query otherwise.
    Returns an empty set when segmentation is unavailable, which callers treat
    as "no evidence available" and fail open.
    """
    try:
        import jieba
    except Exception:
        return frozenset()
    words: set[str] = set()
    for word in jieba.lcut((text or "").lower()):
        word = word.strip()
        if len(word) < 2 or word in _STOP_WORDS_CN:
            continue
        if not any(ch.isalnum() for ch in word):
            continue
        words.add(word)
    return frozenset(words)


# Substantive interrogatives: a query carrying one of these is asking for
# something even when all its content words are stop words ("你叫什么").
_QUESTION_MARKERS = ("什么", "谁", "哪", "多少", "怎么", "为什么", "如何")

# Below this length a query needs real content or an interrogative to justify
# a full retrieval pass ("嗯"/"好的"/"在吗" should not scan the store).
_MIN_INFORMATIVE_CHARS = 8


def _query_is_informative(query: str) -> bool:
    """False when the query carries no topic worth searching for.

    Mirrors fish-memory's queryGate idea: filler turns ("嗯", "好的", "在吗")
    are extremely common in chat, and running the full three-leg retrieval on
    them wastes time and always surfaces *something*, which then gets injected
    as if it were relevant.
    """
    q = (query or "").strip()
    if not q:
        return False
    if len(q) >= _MIN_INFORMATIVE_CHARS:
        return True
    if _content_words(q):
        return True
    return any(marker in q for marker in _QUESTION_MARKERS)


def _has_cjk(text: str) -> bool:
    """True when the text contains any Chinese character."""
    return bool(_CJK_RE.search(text))


def _cjk_bigrams(text: str) -> set[str]:
    """Character bigrams of every CJK run, minus stop bigrams."""
    grams: set[str] = set()
    for seg in _CJK_RE.findall(text):
        for i in range(len(seg) - 1):
            gram = seg[i:i + 2]
            if gram not in _STOP_CJK_BIGRAMS:
                grams.add(gram)
    return grams


def _entry_scope(entry) -> str:
    """记忆条目的分区（web / platform）。旧条目没有该字段 → web。"""
    return str((getattr(entry, "metadata", None) or {}).get("scope", "web"))


def _entry_epoch(entry) -> float | None:
    """条目的创建时间（epoch 秒）；缺失或不可解析返回 None。"""
    from datetime import UTC, datetime

    ts = getattr(entry, "created_at", None)
    if ts is None:
        return None
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            return None
    try:
        if getattr(ts, "tzinfo", None) is None:
            ts = ts.replace(tzinfo=UTC)
        return float(ts.timestamp())
    except Exception:
        return None


def _is_dialogue_entry(entry) -> bool:
    """是不是"某一次对话"的原始条目。

    时间过滤只作用于这类：事实卡 / 技能 / 结论 / 偏好这些都是长期知识，
    问"上周的对话"时不该把它们排除掉。判断只看 category——``dialogue:``
    前缀的条目也可能是转正后的卡（``dialogue:conclusion_…``），不能按前缀。
    """
    cat = str(getattr(entry, "category", "") or "")
    return cat in ("dialogue", "default")


def _time_bounds(since_days=None, date_from=None, date_to=None):
    """把 since_days / date_from / date_to 归一成 [lo, hi) epoch 秒。

    ``date_from`` / ``date_to`` 接受 ``YYYY-MM-DD`` 或完整 ISO 时间，按**本地
    时区**解释（用户说"10-30"指本地那一天）；``date_to`` 含当天。
    """
    from datetime import UTC, datetime, timedelta

    lo = hi = None
    now = datetime.now(UTC)
    if since_days:
        try:
            lo = (now - timedelta(days=float(since_days))).timestamp()
        except Exception:
            pass
    if date_from:
        try:
            d = datetime.fromisoformat(str(date_from).strip())
            if d.tzinfo is None:
                d = d.astimezone()  # naive → 本地时区
            lo = d.timestamp()
        except Exception:
            pass
    if date_to:
        try:
            d = datetime.fromisoformat(str(date_to).strip())
            if d.tzinfo is None:
                d = d.astimezone()
            hi = (d + timedelta(days=1)).timestamp()
        except Exception:
            pass
    return lo, hi


def _lexically_corroborated(query: str, content: str) -> bool:
    """True when query and content share ENOUGH meaningful content words.

    Character bigrams are too permissive: a long passage contains most common
    two-character sequences ("原理", "排序"), and CJK text has no word
    boundaries so cross-word bigrams inflate the count. Segment first, drop
    stop words and generic technical vocabulary, then require either two
    shared content words or one for a very short query.

    Pure-Latin or unparseable queries are exempt (fail open): the bigram
    machinery cannot speak for them.
    """
    q_words = _content_words(query)
    if not q_words:
        return True
    shared = q_words & _content_words(content)
    if len(shared) >= 2:
        return True
    return len(shared) >= 1 and len(q_words) <= 2


# ═══════════════════════════════════════════════════════════════════════════════
# Retriever
# ═══════════════════════════════════════════════════════════════════════════════


class Retriever:
    """Three-signal memory retriever with Reciprocal Rank Fusion (RRF k=60).

    Combines semantic (ChromaDB), BM25 (SQLite FTS5), and temporal (recency
    decay) signals into a single relevance-ranked result set.

    Parameters
    ----------
    config:
        ``MemorySkillConfig`` — drives ``rrf_k`` and ``temporal_decay_lambda``.
    dialogue_store:
        ``DialogueStore`` for FTS5 BM25 full-text search.
    learned_store:
        ``LearnedStore`` for vector-based semantic search.
    """

    def __init__(
        self,
        config: MemorySkillConfig,
        dialogue_store: DialogueStoreProtocol,
        learned_store: LearnedStoreProtocol,
        episode_source: object | None = None,
    ) -> None:
        self._config = config
        self._dialogue_store = dialogue_store
        self._learned_store = learned_store

        self._rrf_k: int = config.rrf_k
        self._temporal_decay_lambda: float = config.temporal_decay_lambda
        # L4 摘要腿：episode_summaries 是"模糊记忆"层，权重必须低于主腿，
        # 只在主检索没有高置信命中时补位（软遗忘表达的数据来源）。
        self._episode_source = episode_source
        self._episode_weight: float = getattr(config, "rrf_episode_weight", 0.8)

        # RRF fusion weights — config-driven so Evolution Loop can adjust
        self._semantic_weight: float = config.rrf_semantic_weight
        self._bm25_weight: float = config.rrf_bm25_weight
        self._temporal_weight: float = config.rrf_temporal_weight
        self._alias_weight: float = getattr(config, "rrf_alias_weight", 3.5)
        self._alias_index = None  # lazily built

    def _alias_ix(self):
        """Lazily build the alias index against the session database."""
        if self._alias_index is None:
            from memory_skill.alias_index import AliasIndex
            self._alias_index = AliasIndex(self._config.db_path)
        return self._alias_index

    def _visible_scopes(self, session_id: str) -> set[str]:
        """四域可见性：world 永远可见 + 会话自身 + 已绑定域。"""
        try:
            import sqlite3
            conn = sqlite3.connect(self._config.db_path, timeout=5)
            try:
                rows = conn.execute(
                    "SELECT scope FROM scope_bindings WHERE session_id=?",
                    (session_id,)).fetchall()
            finally:
                conn.close()
        except Exception:
            rows = []
        out = {session_id or "web", "world"}
        out.update(r[0] for r in rows if r and r[0])
        return out

    # ── Public API ────────────────────────────────────────────────────────

    def _paired_answer(self, question_entry):
        """找某条 user 问句同一轮的下一条 assistant 回复（配对答案）。"""
        try:
            meta = getattr(question_entry, "metadata", None) or {}
            turn_id = meta.get("turn_id") or str(question_entry.id).split(":", 1)[-1]
            conn = self._dialogue_store._conn
            row = conn.execute(
                "SELECT timestamp FROM dialogue_turns WHERE id=?", (turn_id,)
            ).fetchone()
            if not row:
                return None
            nxt = conn.execute(
                "SELECT id, role, content, timestamp FROM dialogue_turns "
                "WHERE role='assistant' AND timestamp >= ? "
                "ORDER BY timestamp LIMIT 1",
                (row[0],),
            ).fetchone()
            if not nxt:
                return None
            from datetime import UTC, datetime

            from memory_skill.contracts import DialogueTurn

            turn = DialogueTurn(
                id=nxt[0], role="assistant", content=nxt[2],
                timestamp=datetime.fromtimestamp(nxt[3], UTC), saw_index=0,
            )
            entry = self._turn_to_entry(turn)
            try:
                entry.semantic_score = 0.0
            except Exception:
                pass
            return entry
        except Exception:
            return None

    def retrieve(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        scope: str = "web",
        since_days: int | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> MemoryEnvelope:
        """Retrieve memories by fusing 3 signals via RRF.

        Parameters
        ----------
        query:
            Natural language search query.
        limit:
            Maximum number of entries to return in the envelope.
        filters:
            Optional ChromaDB metadata filters (passed to ``learned_store``
            only; ``dialogue_store`` does not support metadata filtering).

        Returns
        -------
        MemoryEnvelope
            Envelope with entries sorted by combined RRF score (descending).
            If no results, returns an empty envelope.
        """
        from memory_skill.contracts import MemoryEnvelope

        ts_lo, ts_hi = _time_bounds(since_days, date_from, date_to)

        def _in_time_window(entry) -> bool:
            """对话条目按时间窗过滤；长期知识（事实卡/技能）不受影响。"""
            if ts_lo is None and ts_hi is None:
                return True
            if not _is_dialogue_entry(entry):
                return True
            t = _entry_epoch(entry)
            if t is None:
                return True
            if ts_lo is not None and t < ts_lo:
                return False
            if ts_hi is not None and t >= ts_hi:
                return False
            return True

        if not query:
            return MemoryEnvelope(
                type="recall",
                entries=[],
                truncated=False,
                total_candidates=0,
                timestamp=utcnow(),
            )

        # 查询有效性：没有任何实义词（字母/汉字/数字）时直接返回空。
        # 纯符号查询的向量是噪声，会与某条记忆偶然相似并"召回"它。
        if not re.search(r"[A-Za-z0-9\u4e00-\u9fff]", query):
            return MemoryEnvelope(
                type="recall",
                entries=[],
                truncated=False,
                total_candidates=0,
                timestamp=utcnow(),
            )

        # Filler turns must not trigger a retrieval pass: every scan returns
        # something, and that something then looks like a memory hit.
        if not _query_is_informative(query):
            return MemoryEnvelope(
                type="recall",
                entries=[],
                truncated=False,
                total_candidates=0,
                timestamp=utcnow(),
                diagnostics=[{"gate": "low_information_query", "query": query[:60]}],
            )

        # ── 1. Semantic (ChromaDB vector search) ──────────────────────────
        semantic_entries: list[MemoryEntry] = self._learned_store.search(
            query,
            limit=_BROAD_TOP_K,
            filters=filters,
        )

        # Alias exact hits are resolved BEFORE filtering: a fact card whose
        # alias appears verbatim in the query is authoritative even when its
        # vector score is mediocre ("大橘最近不吃饭" must keep cat_name).
        alias_ids: list[str] = []
        try:
            alias_ids = self._alias_ix().hits(
                query, scope=None if scope == "all" else scope
            )
        except Exception:
            alias_ids = []
        alias_ids = sorted(alias_ids, key=lambda x: 0 if x.startswith("fact:") else 1)
        _alias_set = set(alias_ids)

        # Drop hits below the measured unrelated-query baseline. Dialogue hits
        # additionally need lexical corroboration: similarity alone cannot
        # separate "React vs Vue" from a passage about Chinese reasoning.
        semantic_entries = [
            e for e in semantic_entries
            if (
                # Fact cards: structured content, so a higher vector floor
                # replaces the bigram requirement — unless an alias matched.
                (
                    e.id.startswith("fact:")
                    and not e.id.startswith(
                        ("fact:lesson:", "fact:topic:", "fact:context:")
                    )
                    and (
                        (e.semantic_score or 0.0) >= _FACT_MIN_SCORE
                        or e.id in _alias_set
                    )
                )
                # Dialogue: vector floor plus literal text overlap.
                or (
                    not e.id.startswith("fact:")
                    and (e.semantic_score or 0.0) >= _MIN_SEMANTIC_SCORE
                    and _lexically_corroborated(query, e.content)
                )
            )
        ]

        # 记忆分区：四域可见性（world 永远可见 + 会话自身 + 绑定域）。
        if scope != "all":
            _visible = self._visible_scopes(scope)
            semantic_entries = [
                e for e in semantic_entries if _entry_scope(e) in _visible
            ]

        # 时间窗：只裁对话条目（事实卡/技能是长期知识，不受时间影响）
        if ts_lo is not None or ts_hi is not None:
            semantic_entries = [
                e for e in semantic_entries if _in_time_window(e)
            ]

        # ── 1.5 Skill leg (procedural memory) ─────────────────────────────
        # 技能是长文档，短问句的向量相似度被稀释，单独用更低的门槛召回：
        # "记忆服务起不来怎么排查"必须能找到排障技能。只在"无分类过滤"或
        # "明确查 skill"时启用，避免 skill 混进 mission/pref 等分类查询
        # （曾把 mission 槽位渲染成技能标题）。
        skill_entries: list[MemoryEntry] = []
        _scoped_cat = (filters or {}).get("category") if isinstance(filters, dict) else None
        if _scoped_cat is None or _scoped_cat == "skill":
            try:
                skill_entries = [
                    e for e in self._learned_store.search(
                        query, limit=5, filters={"category": "skill"})
                    if (e.semantic_score or 0.0) >= _SKILL_MIN_SCORE
                ]
            except Exception:
                skill_entries = []
            if scope != "all":
                skill_entries = [
                    e for e in skill_entries if _entry_scope(e) == scope
                ]

        # ── 2. BM25 (SQLite FTS5 full-text search) ────────────────────────
        # Use truncated query: FTS5 ANDs all tokens, long queries never match.
        # 40 chars captures 1-2 key terms while keeping match requirements loose.
        bm25_query = query[:120] if len(query) > 120 else query
        try:
            dialogue_turns: list[DialogueTurn] = self._dialogue_store.search(
                bm25_query,
                limit=_BROAD_TOP_K,
                scope=None if scope == "all" else scope,
            )
        except Exception:
            dialogue_turns = []

        # Convert DialogueTurn → MemoryEntry for unified ranking
        bm25_entries: list[MemoryEntry] = [
            self._turn_to_entry(t) for t in dialogue_turns
        ]

        # Apply category filter to BM25 entries too. filters only reach the
        # semantic (ChromaDB) leg; dialogue turns are tagged "dialogue" and
        # would otherwise leak into pref/pers/skill results regardless of
        # the requested category.
        if filters and filters.get("category"):
            wanted = filters["category"]
            if isinstance(wanted, dict) and wanted.get("$ne"):
                # Exclusion filter (e.g. {"$ne": "default"}): BM25 leg only
                # carries raw dialogue turns, which never carry the excluded
                # category — keep them all.
                pass
            elif wanted == "default":
                bm25_entries = [
                    e for e in bm25_entries
                    if e.category == "dialogue" or e.category == "default"
                ]
            else:
                bm25_entries = [
                    e for e in bm25_entries if e.category == wanted
                ]

        # FTS5 tokenizes CJK per character, so one shared character ("聊")
        # can pull an unrelated turn in. Demand a real bigram overlap too.
        bm25_entries = [
            e for e in bm25_entries
            if _lexically_corroborated(query, e.content)
        ]

        if ts_lo is not None or ts_hi is not None:
            bm25_entries = [e for e in bm25_entries if _in_time_window(e)]

        # ── 2.5 L4 episode leg (模糊记忆) ──────────────────────────────────
        # 旧对话被 prune 后只留下摘要；摘要腿权重低，只在主检索弱命中时补位，
        # 为"软遗忘表达"提供数据来源。
        episode_entries: list[MemoryEntry] = []
        try:
            episode_entries = self._episode_candidates(query, limit=3)
        except Exception:
            episode_entries = []

        # ── 3. Build candidate set (union, dedup by id) ───────────────────
        now = utcnow()
        candidates: dict[str, MemoryEntry] = OrderedDict()

        for e in semantic_entries:
            if e.id not in candidates:
                candidates[e.id] = e
        for e in bm25_entries:
            if e.id not in candidates:
                candidates[e.id] = e
        for e in skill_entries:
            if e.id not in candidates:
                candidates[e.id] = e
        for e in episode_entries:
            if e.id not in candidates:
                candidates[e.id] = e

        if not candidates:
            return MemoryEnvelope(
                type="recall",
                entries=[],
                truncated=False,
                total_candidates=0,
                timestamp=now,
            )

        candidate_list: list[MemoryEntry] = list(candidates.values())

        # ── 3.5 Alias exact-hit leg (cross-lingual) ───────────────────────
        # alias_ids was resolved before filtering; here we pull any alias
        # targets missing from the candidate set, then rebuild candidate_list
        # so the pulled-in entries actually take part in RRF scoring (they
        # used to be added to `candidates` after the list was frozen, so an
        # alias-hit fact card that missed the semantic leg was silently
        # dropped and could never be ranked).
        if alias_ids:
            missing = [i for i in alias_ids if i not in candidates]
            if missing:
                try:
                    for entry in self._learned_store.get_entries(missing):
                        candidates.setdefault(entry.id, entry)
                except Exception:
                    pass
            candidate_list = list(candidates.values())

        # ── 4. Build ranked lists for RRF ─────────────────────────────────

        # Semantic rank map: entry_id → 0-based rank in semantic results
        semantic_rank: dict[str, int] = {}
        for rank, e in enumerate(semantic_entries):
            if e.id not in semantic_rank:
                semantic_rank[e.id] = rank

        # BM25 rank map: entry_id → 0-based rank in BM25 results
        bm25_rank: dict[str, int] = {}
        for rank, e in enumerate(bm25_entries):
            if e.id not in bm25_rank:
                bm25_rank[e.id] = rank

        # Alias rank map: exact alias hits always rank first
        alias_rank: dict[str, int] = {
            eid: rank for rank, eid in enumerate(alias_ids)
        }

        # Temporal ranking: sort all candidates by time_decay (descending)
        temporal_scores: list[tuple[str, float]] = [
            (e.id, self._time_decay(e, now)) for e in candidate_list
        ]
        temporal_scores.sort(key=lambda x: x[1], reverse=True)
        temporal_rank: dict[str, int] = {
            eid: rank for rank, (eid, _) in enumerate(temporal_scores)
        }

        # Skill rank map: the dedicated skill leg is its own RRF signal.
        # Skills are long documents with diluted vector scores and their
        # similarity range overlaps unrelated dialogue, so they get a leg
        # weight rather than a hard pin (a pin would steal precise fact
        # answers such as the memory-db-path card).
        skill_rank: dict[str, int] = {
            e.id: rank for rank, e in enumerate(skill_entries)
        }

        # L4 episode rank map（低权重腿）
        episode_rank: dict[str, int] = {
            e.id: rank for rank, e in enumerate(episode_entries)
        }

        # Best vector score among dialogue passages — a passage that clearly
        # beats a fact card answers the question more precisely, so the card
        # must not be force-ranked above it.
        _best_dialogue_sem = max(
            (e.semantic_score or 0.0 for e in candidate_list
             if not e.id.startswith("fact:")),
            default=0.0,
        )

        # ── 5. Compute weighted RRF scores ────────────────────────────────
        k = float(self._rrf_k)

        scored: list[tuple[MemoryEntry, float]] = []
        for entry in candidate_list:
            eid = entry.id
            score = 0.0

            if eid in semantic_rank:
                score += self._semantic_weight / (k + semantic_rank[eid])
            if eid in bm25_rank:
                score += self._bm25_weight / (k + bm25_rank[eid])
            if eid in temporal_rank:
                score += self._temporal_weight / (k + temporal_rank[eid])
            if eid in skill_rank:
                score += _SKILL_LEG_WEIGHT / (k + skill_rank[eid])
            if eid in episode_rank:
                score += self._episode_weight / (k + episode_rank[eid])
            if eid in alias_rank:
                score += self._alias_weight / (k + alias_rank[eid])
                if eid.startswith("fact:") and (
                    (entry.semantic_score or 0.0)
                    >= _best_dialogue_sem - _FACT_PIN_TOLERANCE
                ):
                    # 事实卡才是检索终点；同一别名也挂在对话原文上时，
                    # 让事实卡稳定排在前面（0.25 远高于各 RRF 项的量级）。
                    # 但若某条对话的向量分明显更高，那条才是精确答案。
                    score += 0.25

            # ── Per-entry evolution weight boost ──────────────────────────
            score *= (0.5 + entry.weight)

            # Long passages win BM25 by token count alone; damp them so a
            # precise short fact is not outranked by a wall of text.
            if len(entry.content) > _LONG_CONTENT_CHARS:
                score *= _LONG_CONTENT_FACTOR

            # Applied after the weight multiplier so it cannot be amplified:
            # a literal match should win on evidence, not on entry weight.
            if (entry.semantic_score or 0.0) >= _EXACT_SEM:
                score += _EXACT_SEM_BONUS

            scored.append((entry, score))

        # Sort by RRF score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # ── 5.5 Pin strongly-relevant fact cards ──────────────────────────
        # RRF fuses ranks only, so a fact card that matches the semantic leg
        # perfectly but appears on no other leg loses to a long passage that
        # appears on three. Pin the clear winners to the front — but never
        # over a passage the vector leg scored clearly higher, which means
        # that passage answers the question more precisely.
        pins = [
            (entry, score) for entry, score in scored
            if entry.id.startswith("fact:")
            and (
                # 别名精确/覆盖命中 = 权威答案，无条件置顶。这类卡常常不在
                # 语义腿结果里（semantic_score=None），旧条件会让它输给
                # 融合了多条腿的对话片段。
                entry.id in alias_rank
                or (
                    (entry.semantic_score or 0.0) >= _FACT_PIN_MIN_SCORE
                    and (entry.semantic_score or 0.0)
                    >= _best_dialogue_sem - _FACT_PIN_TOLERANCE
                )
            )
        ][:_MAX_FACT_PINS]
        if pins:
            pinned_ids = {entry.id for entry, _ in pins}
            rest = [(entry, score) for entry, score in scored if entry.id not in pinned_ids]
            scored = pins + rest

        # ── 5.55 问句优先级 + 配对答案 ─────────────────────────────────────
        # 问句是不可变的"历史快照"，事实卡是可更新的"当前状态"。
        # 查询本身是问句时：问句类对话降权并排到末尾；命中的问句把同一轮
        # 的 assistant 回复（配对答案）带出来，排在问句之前、事实卡之后。
        try:
            from memory_skill.classification_helpers import (
                _looks_like_question as _lq,
            )
        except Exception:  # pragma: no cover
            def _lq(text: str) -> bool:
                return (text or "").rstrip().endswith(("?", "？", "吗", "呢"))

        def _entry_is_question(_e) -> bool:
            _meta = getattr(_e, "metadata", None) or {}
            if str(_meta.get("role", "user")) != "user":
                return False
            return bool(_lq(getattr(_e, "content", "")))

        if _lq(query):
            _answer_ids: set[str] = set()
            _pair_scores: dict[str, float] = {}
            for _entry, _score in list(scored):
                if not _entry_is_question(_entry):
                    continue
                _ans = self._paired_answer(_entry)
                if _ans is None or _ans.id in _answer_ids:
                    continue
                if any(e.id == _ans.id for e, _ in scored):
                    _answer_ids.add(_ans.id)
                    _pair_scores[_ans.id] = max(
                        _pair_scores.get(_ans.id, 0.0), _score * 1.1
                    )
                    continue
                _answer_ids.add(_ans.id)
                _pair_scores[_ans.id] = _score * 1.1
                scored.append((_ans, _score * 1.1))

            _q_ids = {e.id for e, _ in scored if _entry_is_question(e)}

            def _qkey(_item):
                _e, _s = _item
                if _e.id.startswith("fact:"):
                    return (0, -_s)
                if _e.id in _answer_ids:
                    return (1, -_s)
                if _e.id in _q_ids:
                    return (3, -_s * 0.6)
                return (2, -_s)

            scored = sorted(scored, key=_qkey)

        # ── 5.8 Retrieval dedup ───────────────────────────────────────────
        # Near-identical entries (the same thing said twice, or one turn stored
        # both alone and merged) each eat a slate slot. fish-memory dedups at
        # 0.80; the threshold here is stricter because scores already order
        # them, so only genuinely duplicated text is dropped.
        _deduped: list[tuple[MemoryEntry, float]] = []
        _seen_word_sets: list[frozenset[str]] = []
        for _entry, _score in scored:
            _words = _content_words(_entry.content)
            _dup_of = None
            if len(_words) >= 3:
                for _prev in _seen_word_sets:
                    _union = len(_words | _prev)
                    if _union and (len(_words & _prev) / _union) >= _RETRIEVAL_DUP_RATIO:
                        _dup_of = _prev
                        break
            if _dup_of is not None:
                continue
            _deduped.append((_entry, _score))
            if _words:
                _seen_word_sets.append(_words)
        scored = _deduped

        # ── 5.6 Cap the slate by evidence strength ────────────────────────
        # RRF orders by rank, not by how much evidence exists, so a weak query
        # still fills the whole limit. fish-memory caps similarly (C>=0.6 →
        # maxItems, ...), but softened here: it only ever shrinks the slate,
        # never grows it, so it cannot introduce entries that filters removed.
        _best_sem_top = max(
            ((e.semantic_score or 0.0) for e, _ in scored[:3]),
            default=0.0,
        )
        if _best_sem_top >= 0.50:
            _slate = limit
        elif _best_sem_top >= 0.40:
            _slate = min(4, limit)
        elif _best_sem_top >= 0.30:
            _slate = min(3, limit)
        else:
            _slate = min(2, limit)
        effective_limit = max(1, min(limit, _slate))

        # ── 5.7 Two queues: facts vs events ───────────────────────────────
        # Fill the slate while capping fact cards, so one durable statement
        # cannot crowd out the episode being asked about.
        _picked: list[tuple[MemoryEntry, float]] = []
        _facts_in_slate = 0
        for _entry, _score in scored:
            if _entry.id.startswith("fact:"):
                if _facts_in_slate >= _MAX_FACT_SLOTS:
                    continue
                _facts_in_slate += 1
            _picked.append((_entry, _score))
            if len(_picked) >= effective_limit:
                break

        # ── 5.75 L4 摘要补位 ──────────────────────────────────────────────
        # 主检索没有高置信命中（向量分 < 0.45）时，补一条摘要级"模糊记忆"，
        # 为软遗忘表达提供内容；只追加到末尾，不挤占主结果。
        if episode_entries and _best_sem_top < 0.45:
            _picked_ids_now = {e.id for e, _ in _picked}
            for _ep in episode_entries:
                if _ep.id in _picked_ids_now:
                    continue
                _picked.append((_ep, 0.0))
                break

        total_candidates = len(scored)
        top_n = _picked
        _picked_ids = {e.id for e, _ in top_n}
        truncated = total_candidates > len(top_n)

        # ── 6. Build envelope ─────────────────────────────────────────────
        # Diagnosis rows for the top candidates only: enough to explain any
        # ranking question without shipping hundreds of records through MCP.
        diagnostics = [
            {
                "rank": rank + 1,
                "id": entry.id,
                "semantic_score": entry.semantic_score,
                "weight": entry.weight,
                "chars": len(entry.content),
                "ranks": {
                    "semantic": semantic_rank.get(entry.id),
                    "bm25": bm25_rank.get(entry.id),
                    "temporal": temporal_rank.get(entry.id),
                    "alias": alias_rank.get(entry.id),
                },
                "fused_score": round(score, 6),
                "selected": entry.id in _picked_ids,
                "omitted": None if entry.id in _picked_ids else "slate_cap",
            }
            for rank, (entry, score) in enumerate(scored[:20])
        ]
        return MemoryEnvelope(
            type="recall",
            entries=[entry for entry, _ in top_n],
            truncated=truncated,
            total_candidates=total_candidates,
            timestamp=now,
            diagnostics=diagnostics,
        )

    def best_semantic_match(
        self,
        query: str,
        limit: int = 5,
    ) -> tuple[MemoryEntry | None, float]:
        """Return the entry with the highest semantic score for the query.

        Answerability checks use the semantic leg directly because RRF-fused
        ordering is recency-biased — the most relevant entry is often not the
        first envelope entry (KNOWN-ISSUES #1).

        Returns
        -------
        ``(entry, semantic_score)`` — ``(None, 0.0)`` when nothing matched.
        """
        if not query:
            return None, 0.0

        entries: list[MemoryEntry] = self._learned_store.search(
            query,
            limit=limit,
        )
        if not entries:
            return None, 0.0

        best = max(entries, key=lambda e: (e.semantic_score or 0.0, len(e.content)))
        return best, best.semantic_score or 0.0

    # ── Temporal scoring ──────────────────────────────────────────────────

    def _time_decay(self, entry: MemoryEntry, now: datetime) -> float:
        """注意力权重 + 情绪加权后的时间衰减。

        默认路径与旧行为完全一致（``weight × exp(-λ × hours)``）。
        条目 metadata 带 ``attention_stability``（天，再巩固/注入强化后写入）
        时改用 ``R = exp(-Δt / S)``；``arousal``（0-1）按
        ``S_eff = S × (1 + 0.5a)`` 调制——情绪唤醒越高衰减越慢（情绪加权）。
        """
        age_seconds = (now - entry.updated_at).total_seconds()
        age_hours = max(age_seconds / 3600.0, 0.0)

        meta = getattr(entry, "metadata", None) or {}
        try:
            stability_days = float(meta.get("attention_stability") or 0.0)
        except Exception:
            stability_days = 0.0

        if stability_days > 0:
            lam = 1.0 / (stability_days * 24.0)
        else:
            lam = self._temporal_decay_lambda
            if lam <= 0:
                return entry.weight

        try:
            arousal = float(meta.get("arousal") or 0.0)
        except Exception:
            arousal = 0.0
        if arousal > 0:
            lam = lam / (1.0 + 0.5 * min(1.0, arousal))

        return entry.weight * math.exp(-lam * age_hours)

    # ── Attention reinforcement (注入路径专用) ─────────────────────────────

    def reinforce_injected(self, entry_ids: list[str], now: float | None = None) -> dict[str, float]:
        """被实际注入的记忆做一次再巩固（注意力权重）。"""
        store = self._learned_store
        fn = getattr(store, "reinforce_many", None)
        if fn is None:
            return {}
        try:
            return fn(entry_ids, now=now)
        except Exception:
            return {}

    # ── L4 episode leg ─────────────────────────────────────────────────────

    def _episode_candidates(self, query: str, limit: int = 3) -> list[MemoryEntry]:
        """L4 摘要腿：旧对话被 prune 后的"模糊记忆"，低权重补位。"""
        from datetime import UTC

        from memory_skill.contracts import MemoryEntry

        src = self._episode_source
        if src is None or not query:
            return []
        try:
            episodes = src.list_episodes(60)
        except Exception:
            return []
        if not episodes:
            return []

        qb = _cjk_bigrams(query)
        qw = _content_words(query)
        scored: list[tuple[float, dict]] = []
        for ep in episodes:
            summary = str(ep.get("summary") or "")
            if not summary:
                continue
            sb = _cjk_bigrams(summary)
            sw = _content_words(summary)
            overlap = (len(qb & sb) / len(qb)) if qb else 0.0
            word_overlap = (len(qw & sw) / len(qw)) if qw else 0.0
            score = max(overlap, word_overlap)
            if score >= 0.15:
                scored.append((score, ep))
        scored.sort(key=lambda x: x[0], reverse=True)

        now = utcnow()
        out: list[MemoryEntry] = []
        for _score, ep in scored[:limit]:
            try:
                end = float(ep.get("period_end") or 0)
                ts = datetime.fromtimestamp(end, UTC)
            except Exception:
                ts = now
            out.append(MemoryEntry(
                id=f"episode:{ep.get('id')}",
                content=str(ep.get("summary") or ""),
                created_at=ts,
                updated_at=ts,
                weight=0.6,
                category="episode",
                tags=[],
                metadata={
                    "source": "episode",
                    "episode_id": str(ep.get("id")),
                    "period_end": ep.get("period_end"),
                },
            ))
        return out

    # ── DialogueTurn → MemoryEntry conversion ─────────────────────────────

    @staticmethod
    def _turn_to_entry(turn: DialogueTurn) -> MemoryEntry:
        """Convert a ``DialogueTurn`` into a ``MemoryEntry`` for unified ranking.

        The entry id mirrors the learned-store format (``dialogue:<turn.id>``)
        so feedback/boost_weight lookups by id resolve to the same chroma
        entry.  A divergent prefix (e.g. ``dialogue_``) would make boosts
        silently no-op.
        """
        from memory_skill.contracts import MemoryEntry

        return MemoryEntry(
            id=f"dialogue:{turn.id}",
            content=turn.content,
            created_at=turn.timestamp,
            updated_at=turn.timestamp,
            weight=0.5,
            category="dialogue",
            tags=[],
            metadata={"role": turn.role, "source": "dialogue"},
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════



