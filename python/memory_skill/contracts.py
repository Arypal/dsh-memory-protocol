"""
Memory Skill — contracts, data models, protocols, and exceptions.

This module defines the public interface for the Memory Skill subsystem:
- Immutable dataclasses for all memory-related data structures
- Protocols (interfaces) for the MemorySkill and EventBus
- Exception hierarchy for all error types
- Configuration dataclass with sensible defaults
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

# ─── Data Models ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MemoryEntry:
    """A single memory entry with metadata for ranking, categorization, and evolution.

    Fields:
        id: Unique identifier for the entry.
        content: The memory text content.
        created_at: When this memory was first created.
        updated_at: When this memory was last modified.
        weight: Importance/ranking weight (0.0–1.0).
        category: Namespace or category for isolation (e.g. 'default', 'work').
        tags: Human-readable tags for filtering.
        metadata: Extensible key-value store for arbitrary metadata.
        is_system: When True, this entry cannot be modified by the agent.
            Evolution and consolidation set this for their outputs.
        semantic_score: Cosine similarity (0.0–1.0) of this entry to the query
            that produced it, or ``None`` when not computed (e.g. entries
            matched only via the BM25 leg).  Used by ``CapabilityRegistry``
            to judge answerability — see KNOWN-ISSUES #1.
    """
    id: str
    content: str
    created_at: datetime
    updated_at: datetime
    weight: float
    category: str
    tags: list[str]
    metadata: dict[str, Any]
    is_system: bool = False
    semantic_score: float | None = None


@dataclass(frozen=True)
class MemoryEnvelope:
    """A batch of memory entries delivered via EventBus, with truncation metadata.

    Fields:
        type: Envelope type label (e.g. 'recall', 'ingest', 'evolve').
        entries: The memory entries included in this batch.
        truncated: Whether the result set was truncated (more candidates existed).
        total_candidates: Total number of candidate entries before truncation.
        timestamp: When this envelope was created.
    """
    type: str
    entries: list[MemoryEntry]
    truncated: bool
    total_candidates: int
    timestamp: datetime
    # Per-candidate scoring breakdown for diagnosis. Never injected into the
    # prompt and empty by default: each row records the four RRF rank
    # positions, the raw semantic score, the fused score, and why the entry
    # was left out (``limit``/``below_threshold``/...). This is what makes
    # "why didn't it remember X" answerable without re-running probes.
    diagnostics: list[dict] | None = None


@dataclass(frozen=True)
class IngestReceipt:
    """The result of persisting a turn through the ingest pipeline (see ADR-0001).

    Replaces the empty MemoryEnvelope that ``ingest_dialogue`` used to return —
    the write chain now reports what actually happened, so callers and tests
    can observe dedup, weight, and per-stage enrichment status.

    Fields:
        entry_id: The id of the entry stored in the learned store
            (``"dialogue:{turn.id}"`` when newly inserted, or the merged
            duplicate's id when deduped).
        deduped: True when the turn merged into an existing semantically
            similar entry (weight +0.05) instead of inserting a new one.
        weight: The entry's weight after the write (0.5 on new insert, or
            the bumped duplicate weight when deduped).
        staged: Per-stage status for enrichable stages. Keys: ``"title"``,
            ``"extracted"``, ``"gap"``. Each value is a dict with at least
            ``ok: bool``; on skip/failure it also carries ``reason``.
        timestamp: When the write occurred.
    """
    entry_id: str
    deduped: bool
    weight: float
    staged: dict[str, dict]
    timestamp: datetime


@dataclass(frozen=True)
class SawEntry:
    """A single frame in the Saw ring buffer (short-term screen observation buffer).

    Fields:
        heartbeat_index: Monotonically increasing index used as primary key
            for O(1) ring-buffer reads.
        content: The textual description of the screen observation.
        timestamp: When this observation was captured.
    """
    heartbeat_index: int
    content: str
    timestamp: datetime


@dataclass(frozen=True)
class DialogueTurn:
    """A single turn in a dialogue, linked to the Saw index at the time it occurred.

    Fields:
        id: Unique identifier for this turn.
        role: Speaker role (e.g. 'user', 'assistant', 'system').
        content: The text content of the turn.
        timestamp: When this turn occurred.
        saw_index: Optional heartbeat_index of the Saw frame active
            when this turn happened (for cross-modal correlation).
        partner: Optional partner identity — who the speaker is
            talking *to*.  Together with ``MemorySkillConfig.agent_name``
            this forms the object-tagged namespace
            (e.g. ``"agent_a/user"``).
    """
    id: str
    role: str
    content: str
    timestamp: datetime
    saw_index: int | None = None
    partner: str | None = None


# ─── Exception Hierarchy ───────────────────────────────────────────────────────


class MemorySkillError(Exception):
    """Base exception for all Memory Skill errors."""

class StoreCorruptionError(MemorySkillError):
    """Raised when persistent storage is corrupted or in an inconsistent state."""


class SearchStoreCorruptionError(StoreCorruptionError):
    """Raised when vector search fails due to store corruption."""


class ModelLoadError(MemorySkillError):
    """Raised when an ML model (e.g., ONNX embedder) fails to load."""


# ─── Helpers ────────────────────────────────────────────────────────────────────


def utcnow() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(UTC)


# ── Ingest truncation policy (ADR-0001) ────────────────────────────────────────
# Head+tail truncation for auto-ingest: preserve opening context and closing
# conclusion, drop the middle — embedding only sees ≤512 tokens anyway
# (bge-large-en-v1.5), so retrieval quality is unaffected. Every ingest entry
# point applies this policy so proxy and MCP write paths behave identically.
_MAX_AUTO_INGEST_CHARS: int = 800
_AUTO_INGEST_HEAD: int = 490
_AUTO_INGEST_TAIL: int = 300
_AUTO_INGEST_ELLIPSIS: str = "\n…[中段省略]…\n"


def clip_auto_ingest(content: str) -> str:
    """Truncate long content head+tail for ingest (see ADR-0001)."""
    if len(content) <= _MAX_AUTO_INGEST_CHARS:
        return content
    head = content[:_AUTO_INGEST_HEAD]
    tail = content[-_AUTO_INGEST_TAIL:]
    if len(head) + len(_AUTO_INGEST_ELLIPSIS) + len(tail) <= _MAX_AUTO_INGEST_CHARS:
        return head + _AUTO_INGEST_ELLIPSIS + tail
    return head + tail


# ─── Configuration ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MemorySkillConfig:
    """Configuration for the MemorySkill system with sensible defaults.

    All fields have defaults so ``MemorySkillConfig()`` produces a valid config.
    """

    # ── Namespace / tenant isolation ──────────────────────────────────────
    namespace: str = "default"
    agent_name: str = ""  # When set, auto-namespace = f"{agent_name}/{partner}"

    # ── Envelope limits ───────────────────────────────────────────────────
    max_entries_per_envelope: int = 10
    max_envelope_tokens: int = 4096

    # ── Saw ring buffer (short-term screen observations) ──────────────────
    saw_buffer_capacity: int = 1000

    # ── Embedding ─────────────────────────────────────────────────────────
    embedding_dim: int = 1024
    model_path: str = os.getenv(
        "MEMORY_MODEL_PATH",
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "bge-large-en-v1.5",
        ),
    )

    # ── Retrieval ─────────────────────────────────────────────────────────
    similarity_top_k: int = 10
    rrf_k: int = 60
    temporal_decay_lambda: float = 0.01  # λ in: effective = weight × exp(-λ × hours_since_updated)

    # ── RRF fusion weights (configurable for Evolution Loop V5) ──────────
    rrf_semantic_weight: float = 0.5     # V8: BM25 is primary signal (MemPalace pattern)
    rrf_bm25_weight: float = 2.5        # V8: BM25 boosted for Chinese ver batim content
    rrf_temporal_weight: float = 0.5
    # 别名精确命中：跨语种（Daju → 大橘）且不依赖向量，因此权重最高
    rrf_alias_weight: float = 3.5

    # ── Storage ───────────────────────────────────────────────────────────
    db_path: str = "memory.db"
    max_learned_entries: int = 100_000

    # ── Retention（L3 / L4）────────────────────────────────────────────────
    dialogue_ttl_days: int = 90      # 0 = 不按时间淘汰
    dialogue_max_turns: int = 2000   # 0 = 不按条数淘汰

    # ── Importance gate ───────────────────────────────────────────────────
    importance_gate: str = os.getenv("MEMORY_IMPORTANCE_GATE", "rule")
    # "rule" = rule-based ImportanceScorer (default)
    # "llm"  = LLMImportanceGate (DeepSeek V4 Flash via .env)

    # ── Tree-based memory organisation ────────────────────────────────────
    tree_enabled: bool = True
    # When True, every memory is also placed into a navigable tree.
    # User memories branch from 偏好/回忆与目标.
    # Assistant memories branch from 人格/任务与技能.
    # LLM classification runs only on dedup miss (no gate).


# ═══════════════════════════════════════════════════════════════════════════════
# Protocols — structural subtyping for DI / testability
# ═══════════════════════════════════════════════════════════════════════════════

from typing import Protocol


class EmbedderProtocol(Protocol):
    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class DialogueStoreProtocol(Protocol):
    def insert(self, turn) -> None: ...
    def insert_batch(self, turns: list) -> None: ...
    def get_by_id(self, turn_id: str): ...
    def get_recent(self, n: int = 5): ...
    def count(self) -> int: ...
    def search(self, query: str, limit: int = 10): ...
    def time_range(self, start: str, end: str, limit: int = 100): ...


class LearnedStoreProtocol(Protocol):
    def insert(self, entry) -> None: ...
    def search(self, query: str, limit: int = 10, filters=None): ...
    def get_weight(self, entry_id: str) -> float | None: ...
    def set_weight(self, entry_id: str, weight: float) -> None: ...
    def boost_weight(self, entry_id: str, delta: float = 0.05, cap: float = 0.95) -> float: ...
    def find_duplicate(self, embedding: list[float], threshold: float) -> dict | None: ...
    def health(self) -> dict: ...


class SawBufferProtocol(Protocol):
    def put(self, entry) -> None: ...
    def get_all(self) -> list: ...
    def get_at_offset(self, heartbeat_index: int): ...


class TreeManagerProtocol(Protocol):
    def classify(self, content: str) -> dict: ...
    def add_node(self, content: str, memory_id: str, root: str, branch: str) -> str: ...
    def get_context(self, query: str) -> str: ...
    def navigate(self, query: str) -> str: ...
    def branch_counts(self, branch_id: str) -> tuple[int, int]: ...
    def branch_avg_weight(self, branch_id: str) -> float: ...
    def branch_last_updated(self, branch_id: str) -> str | None: ...
