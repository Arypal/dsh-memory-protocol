"""
Memory Skill — LearnedStore (ChromaDB-backed vector storage)

Provides the ``LearnedStore`` class for persistent vector-based memory
storage using ChromaDB as the backend (chosen over LanceDB — spike
showed 7× faster search).

Key design:
- ChromaDB PersistentClient with SQLite for metadata
- HNSW index with cosine distance
- Metadata stored as flat dict (complex types JSON-serialized)
- Max-entries eviction: oldest entries evicted when count exceeds limit
- Corruption fallback: search gracefully returns empty on storage errors
- EUA-style update: upsert replaces embeddings + metadata atomically

Usage::

    from memory_skill.contracts import MemorySkillConfig
    from memory_skill.embedder import Embedder
    from memory_skill.learned_store import LearnedStore

    cfg = MemorySkillConfig()
    emb = Embedder(cfg)
    store = LearnedStore(cfg, emb)
    store.insert(MemoryEntry(...))
    results = store.search("query", limit=10)
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from memory_skill import secure

if TYPE_CHECKING:
    from memory_skill.contracts import MemoryEntry, MemorySkillConfig
    from memory_skill.embedder import Embedder

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# LearnedStore
# ═══════════════════════════════════════════════════════════════════════════════


class LearnedStore:
    """ChromaDB-backed vector store for learned agent memories.

    Parameters
    ----------
    config:
        ``MemorySkillConfig`` instance — drives db_path, namespace,
        and max_learned_entries.
    embedder:
        ``Embedder`` instance for generating text embeddings.
    chroma_path:
        Optional override for the ChromaDB persistence directory.
        If not set, ``config.db_path`` is used.
    """

    _COLLECTION_PREFIX = "learned"

    def __init__(
        self,
        config: MemorySkillConfig,
        embedder: Embedder,
        chroma_path: str | None = None,
    ) -> None:
        import chromadb

        self._config: MemorySkillConfig = config
        self._embedder: Embedder = embedder

        persist_dir = chroma_path if chroma_path is not None else config.db_path
        os.makedirs(persist_dir, exist_ok=True)

        # ChromaDB settings: use cosine distance (L2-normalized vectors)
        self._client = chromadb.PersistentClient(path=persist_dir)

        collection_name = f"{self._COLLECTION_PREFIX}_{config.namespace.replace('/', '_').replace(':', '_')}"
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Public API ────────────────────────────────────────────────────────

    def insert(self, entry: MemoryEntry) -> None:
        """Insert a memory entry into the vector store.

        The entry's content is embedded, and both the vector and metadata
        are persisted.  After insertion, the max-entries constraint is enforced.

        Parameters
        ----------
        entry:
            The ``MemoryEntry`` to store.
        """
        vector = self._embedder.embed(entry.content)
        metadata = self._entry_to_chroma_meta(entry)

        self._collection.add(
            ids=[entry.id],
            embeddings=[vector],
            documents=[secure.encrypt(entry.content)],
            metadatas=[metadata],
        )

        self._enforce_max_entries()

    def insert_batch(
        self,
        entries: list[MemoryEntry],
        embeddings: list[list[float]],
    ) -> None:
        """Insert multiple memory entries in a single ChromaDB batch call.

        This uses ChromaDB's native ``collection.add()`` with parallel list
        arguments (ids, embeddings, documents, metadatas), which is 10-50×
        faster than per-entry ``add()`` calls.

        After insertion, the max-entries constraint is enforced once.

        Parameters
        ----------
        entries:
            List of ``MemoryEntry`` objects to store.
        embeddings:
            Pre-computed embedding vectors, one per entry (same order).
            Each vector must have length ``config.embedding_dim``.
        """
        if not entries:
            return

        ids = [e.id for e in entries]
        documents = [secure.encrypt(e.content) for e in entries]
        metadatas = [self._entry_to_chroma_meta(e) for e in entries]

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        self._enforce_max_entries()

    def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryEntry]:
        """Search for memories most similar to the query.

        Supports optional ChromaDB ``where`` metadata filters for hybrid
        (semantic + scalar) search.

        On storage corruption, the error is caught internally and an
        empty list is returned (graceful degradation).

        Parameters
        ----------
        query:
            Natural language query string.
        limit:
            Maximum number of results to return.
        filters:
            ChromaDB metadata filter dict (e.g. ``{"category": "work"}``
            or ``{"weight": {"$gt": 0.5}}``).

        Returns
        -------
        list[MemoryEntry]
            Matching entries ordered by cosine similarity (closest first).
        """
        try:
            return self._search_impl(query, limit, filters)
        except Exception as exc:

            logger.error(
                "LearnedStore search failed (corruption suspected): %s",
                exc,
            )
            # Graceful degradation — caller can continue with empty results.
            return []

    def update(self, entry_id: str, content: str, **kwargs) -> None:
        """Update a memory entry's content (EUA-style efficient update).

        The content is re-embedded and the old vector + document are
        replaced atomically via ChromaDB upsert.  The ``updated_at``
        timestamp is refreshed.

        Parameters
        ----------
        entry_id:
            The ID of the entry to update.
        content:
            The new content string.
        **kwargs:
            Additional metadata to merge into the entry's metadata dict.
            Use ``metadata={...}`` to pass a dict of metadata key-values.

        Raises
        ------
        KeyError
            If the entry_id does not exist in the collection.
        """
        # Check existence
        existing = self._collection.get(ids=[entry_id])
        if not existing["ids"]:
            raise KeyError(
                f"Entry '{entry_id}' not found in LearnedStore"
            )

        old_meta = existing["metadatas"][0] if existing["metadatas"] else {}

        vector = self._embedder.embed(content)
        new_meta = {
            **old_meta,
            **kwargs.get("metadata", {}),
            "updated_at": time.time(),
        }

        self._collection.upsert(
            ids=[entry_id],
            embeddings=[vector],
            documents=[secure.encrypt(content)],
            metadatas=[new_meta],
        )

    def get_entries(self, ids: list[str]) -> list["MemoryEntry"]:
        """Fetch entries by id (used by the alias leg, which crosses languages)."""
        from datetime import UTC, datetime

        from memory_skill.contracts import MemoryEntry

        if not ids:
            return []
        try:
            res = self._collection.get(ids=ids, include=["documents", "metadatas"])
        except Exception:
            return []

        out: list[MemoryEntry] = []
        now = datetime.now(UTC)
        docs = res.get("documents") or []
        metas = res.get("metadatas") or []
        for i, eid in enumerate(res.get("ids") or []):
            doc = (docs[i] if i < len(docs) else "") or ""
            doc = secure.decrypt_or_placeholder(doc)[0]
            meta = (metas[i] if i < len(metas) else None) or {}
            ext_meta = dict(meta)
            raw_ext = meta.get("ext_metadata_json")
            if isinstance(raw_ext, str):
                try:
                    parsed = json.loads(raw_ext)
                    if isinstance(parsed, dict):
                        ext_meta.update(parsed)
                except Exception:
                    pass
            out.append(
                MemoryEntry(
                    id=eid,
                    content=doc,
                    created_at=now,
                    updated_at=now,
                    weight=float(meta.get("weight", 0.5)),
                    category=str(meta.get("category") or "default"),
                    tags=[],
                    metadata=ext_meta,
                )
            )
        return out

    def delete(self, entry_id: str) -> None:
        """Delete a memory entry from the store.

        Idempotent — deleting a non-existent entry does not raise.

        Parameters
        ----------
        entry_id:
            The ID of the entry to delete.
        """
        self._collection.delete(ids=[entry_id])

    def find_duplicate(
        self,
        embedding: list[float],
        threshold: float = 0.85,
    ) -> dict | None:
        """Search for a near-duplicate memory by embedding cosine distance.

        Finds the single nearest neighbor in the collection and returns it
        if the cosine distance is *below* ``1 - threshold`` (i.e. similarity
        above ``threshold``), indicating a match.

        The collection uses ``hnsw:space=cosine``, so chroma's ``distance``
        is cosine distance (``1 - similarity``): identical text has distance
        ``0.0``, maximally different text approaches ``1.0``.  A duplicate
        is therefore a *small* distance — the historical ``distance >
        threshold`` comparison inverted this and silently disabled dedup
        (see KNOWN-ISSUES #9: thousands of duplicate fragments).

        Parameters
        ----------
        embedding:
            The embedding vector to compare against stored entries.
        threshold:
            Cosine similarity threshold (default 0.85).  Only results with
            similarity ``>= threshold`` (distance ``<= 1 - threshold``)
            are considered duplicates.

        Returns
        -------
        dict | None
            ``{entry_id, content, weight, distance}`` if a match is found,
            ``None`` otherwise.
        """
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=1,
        )

        if not results.get("ids") or not results["ids"][0]:
            return None

        entry_id = results["ids"][0][0]
        distance = (
            results["distances"][0][0] if results.get("distances") else 0.0
        )
        content = (
            results["documents"][0][0] if results.get("documents") else ""
        )
        meta = results["metadatas"][0][0] if results.get("metadatas") else {}
        meta = meta or {}  # chroma may return None for an entry's metadata
        weight = float(meta.get("weight", 0.5))

        # Cosine space: distance = 1 - similarity.  A duplicate is a small
        # distance, so match when distance <= (1 - threshold).
        if distance <= (1.0 - threshold):
            return {
                "entry_id": entry_id,
                "content": content,
                "weight": weight,
                "distance": distance,
                "category": meta.get("category"),
            }

        return None

    def get_weight(self, entry_id: str) -> float | None:
        """Get the current weight for a memory entry.

        Public API for ``EvolutionLoop`` — avoids direct ``_collection``
        access from outside the store.

        Parameters
        ----------
        entry_id:
            The memory entry ID to look up.

        Returns
        -------
        float | None
            The current weight, or ``None`` if the entry is not found
            or on storage error.
        """
        try:
            result = self._collection.get(
                ids=[entry_id],
                include=["metadatas"],
            )
        except Exception:
            logger.warning(
                "LearnedStore.get_weight: failed for %s", entry_id
            )
            return None

        if (
            result
            and result.get("metadatas")
            and result["metadatas"][0]
        ):
            return float(result["metadatas"][0].get("weight", 0.5))

        return None

    def get_all_weights(self) -> dict[str, float]:
        """Return a dict of {entry_id: weight} for all stored entries.

        Returns
        -------
        dict[str, float]
            Map of entry_id to current weight.  Returns empty dict on
            storage error.
        """
        try:
            result = self._collection.get(include=["metadatas"])
        except Exception:
            logger.warning("LearnedStore.get_all_weights: collection.get failed")
            return {}

        weights: dict[str, float] = {}
        for eid, meta in zip(result.get("ids", []), result.get("metadatas", [])):
            weights[eid] = float(meta.get("weight", 0.5)) if meta else 0.5
        return weights

    def list_by_category(self, category: str, limit: int = 10,
                         sort_by: str = "created_at",
                         descending: bool = True) -> list[MemoryEntry]:
        """Return entries of a given category, newest first by default.

        Queries ChromaDB metadata directly (NOT semantic search), so the
        result is complete and accurate for the category.  ``limit=0``
        returns all matching entries (use sparingly).

        Parameters
        ----------
        category:
            The metadata ``category`` value to filter on (e.g. ``"conclusion"``).
        limit:
            Maximum number of entries to return. ``0`` means no limit.
        sort_by:
            Metadata field to sort on: ``"created_at"`` or ``"updated_at"``.
        descending:
            ``True`` = newest first.

        Returns
        -------
        list[MemoryEntry]
            Matching entries (empty on storage error).
        """
        try:
            result = self._collection.get(
                where={"category": category},
                include=["documents", "metadatas"],
            )
        except Exception:
            logger.warning("LearnedStore.list_by_category: get failed for %r", category)
            return []

        entries: list[MemoryEntry] = []
        ids = result.get("ids", [])
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])
        for i, eid in enumerate(ids):
            doc = documents[i] if i < len(documents) else None
            meta = metadatas[i] if i < len(metadatas) else {}
            entries.append(self._chroma_meta_to_entry(eid, doc, meta or {}))

        key = "updated_at" if sort_by == "updated_at" else "created_at"
        entries.sort(key=lambda e: getattr(e, key).timestamp() if getattr(e, key) else 0.0,
                     reverse=descending)
        return entries if limit <= 0 else entries[:limit]

    def set_weight(self, entry_id: str, weight: float) -> None:
        """Update the weight metadata for a memory entry.

        Public API for ``EvolutionLoop`` — avoids direct ``_collection``
        access from outside the store.  Idempotent: does nothing if the
        entry does not exist.

        Parameters
        ----------
        entry_id:
            The memory entry ID to update.
        weight:
            The new weight value (clamped by the caller).
        """
        try:
            existing = self._collection.get(
                ids=[entry_id],
                include=["metadatas"],
            )
        except Exception:
            logger.warning(
                "LearnedStore.set_weight: failed for %s", entry_id
            )
            return

        if not existing or not existing.get("ids"):
            return

        meta = dict(existing["metadatas"][0]) if existing["metadatas"][0] else {}
        meta["weight"] = weight
        self._collection.update(ids=[entry_id], metadatas=[meta])

    def set_title(self, entry_id: str, title: str) -> None:
        try:
            existing = self._collection.get(ids=[entry_id], include=["metadatas"])
        except Exception:
            return
        if not existing or not existing.get("ids"):
            return
        meta = dict(existing["metadatas"][0]) if existing["metadatas"][0] else {}
        meta["title"] = title
        self._collection.update(ids=[entry_id], metadatas=[meta])

    def boost_weight(self, entry_id: str, delta: float = 0.05, cap: float = 0.95) -> float:
        """Increase an entry's weight by *delta*, capped at *cap*.

        Returns the new weight, or 0.5 if the entry doesn't exist.
        """
        current = self.get_weight(entry_id)
        if current is None:
            return 0.5
        new = min(current + delta, cap)
        self.set_weight(entry_id, new)
        return new

    # ── 注意力权重（Attention Weight）──────────────────────────────────────
    # 被"实际注入"的记忆会被强化：稳定度 S 按"召回时越接近遗忘、增益越大"
    # 增长（再巩固），写入 metadata 的 attention_stability / last_access_at。
    # 检索只读，强化只发生在注入路径，因此不影响检索回归。

    def reinforce(self, entry_id: str, now: float | None = None,
                  alpha: float = 0.3, cap_days: float = 90.0) -> float | None:
        """强化一条记忆的注意力权重，返回新的稳定度 S（天）。"""
        import math

        now = now if now is not None else time.time()
        try:
            existing = self._collection.get(ids=[entry_id], include=["metadatas"])
        except Exception:
            return None
        if not existing.get("ids"):
            return None
        meta = dict((existing.get("metadatas") or [{}])[0] or {})
        ext: dict = {}
        raw_ext = meta.get("ext_metadata_json")
        if isinstance(raw_ext, str):
            try:
                parsed = json.loads(raw_ext)
                if isinstance(parsed, dict):
                    ext = parsed
            except Exception:
                ext = {}
        try:
            age_h = max(0.0, (now - float(meta.get("updated_at") or now)) / 3600.0)
        except Exception:
            age_h = 0.0
        try:
            s = float(ext.get("attention_stability") or meta.get("attention_stability") or 4.1667)
        except Exception:
            s = 4.1667
        try:
            arousal = float(ext.get("arousal") or meta.get("arousal") or 0.0)
        except Exception:
            arousal = 0.0
        s_eff = s * (1.0 + 0.5 * min(1.0, arousal))
        r = math.exp(-age_h / (s_eff * 24.0))
        new_s = min(cap_days, s * (1.0 + alpha * (1.0 - r)))
        ext["attention_stability"] = round(new_s, 4)
        ext["last_access_at"] = now
        meta["ext_metadata_json"] = json.dumps(ext)
        # 被想起的记忆在时间腿上重新变"新"（MemoryBank 的 t 归零）
        meta["updated_at"] = now
        try:
            meta["weight"] = min(float(meta.get("weight", 0.5)) + 0.03, 0.95)
        except Exception:
            pass
        try:
            self._collection.update(ids=[entry_id], metadatas=[meta])
        except Exception:
            return None
        return new_s

    def reinforce_many(self, entry_ids: list[str], now: float | None = None,
                       alpha: float = 0.3, cap_days: float = 90.0) -> dict[str, float]:
        """批量强化（同一轮注入的多条），返回 id → 新稳定度。"""
        out: dict[str, float] = {}
        for eid in entry_ids or []:
            if not isinstance(eid, str):
                continue
            try:
                s = self.reinforce(eid, now=now, alpha=alpha, cap_days=cap_days)
            except Exception:
                continue
            if s is not None:
                out[eid] = s
        return out

    def health(self) -> dict[str, Any]:
        """Return a health/diagnostics summary of the vector store.

        Returns
        -------
        dict
            Keys: ``status``, ``collection``, ``entry_count``,
            ``backend``, ``embedding_dim``.
        """
        try:
            count = self._collection.count()
            status = "healthy"
        except Exception as exc:
            count = -1
            status = f"degraded: {exc}"

        return {
            "status": status,
            "collection": f"{self._COLLECTION_PREFIX}_{self._config.namespace}",
            "entry_count": count,
            "backend": "chromadb",
            "embedding_dim": self._config.embedding_dim,
        }

    # ── Private helpers ───────────────────────────────────────────────────

    def _search_impl(
        self,
        query: str,
        limit: int,
        filters: dict[str, Any] | None,
    ) -> list[MemoryEntry]:
        """Core search logic (no corruption handling — wrapped by search())."""
        vector = self._embedder.embed(query) if query else [0.0] * self._config.embedding_dim

        kwargs: dict[str, Any] = dict(
            query_embeddings=[vector],
            n_results=limit,
        )
        if filters:
            kwargs["where"] = self._normalise_where(filters)

        results = self._collection.query(**kwargs)
        return self._parse_query_results(results)

    @staticmethod
    def _normalise_where(filters: dict[str, Any]) -> dict[str, Any]:
        """Normalise a filter dict into a ChromaDB-compatible ``where``.

        ChromaDB rejects a top-level dict with more than one operator key
        (e.g. ``{"weight": ..., "category": ...}``) — multiple conditions
        must be wrapped in ``$and``.
        """
        if len(filters) <= 1:
            return filters
        return {"$and": [{k: v} for k, v in filters.items()]}

    def _enforce_max_entries(self) -> None:
        """Evict oldest entries if collection exceeds config.max_learned_entries."""
        max_entries = self._config.max_learned_entries
        if max_entries <= 0:
            return

        current = self._collection.count()
        if current <= max_entries:
            return

        # Fetch all entries with their metadata timestamps
        all_data = self._collection.get(
            include=["metadatas"],
        )

        # Pair (id, created_at) and sort by oldest first
        pairs: list[tuple[str, float]] = []
        for eid, meta in zip(all_data["ids"], all_data["metadatas"]):
            created_at = meta.get("created_at", 0.0) if meta else 0.0
            pairs.append((eid, created_at))

        pairs.sort(key=lambda x: x[1])  # oldest first

        to_evict = current - max_entries
        evict_ids = [eid for eid, _ in pairs[:to_evict]]

        if evict_ids:
            self._collection.delete(ids=evict_ids)
            logger.debug(
                "Evicted %d oldest entries (count=%d → %d)",
                len(evict_ids),
                current,
                current - len(evict_ids),
            )

    def _parse_query_results(
        self,
        results: dict[str, Any],
    ) -> list[MemoryEntry]:
        """Convert ChromaDB query results into ``MemoryEntry`` objects.

        Attaches ``semantic_score`` (cosine similarity to the query, derived
        from the ChromaDB cosine distance) so downstream consumers such as
        ``CapabilityRegistry.can_answer`` can judge genuine relevance instead
        of raw result counts.
        """
        entries: list[MemoryEntry] = []

        ids_list = results.get("ids")
        if not ids_list or not ids_list[0]:
            return entries

        documents = results.get("documents")
        metadatas = results.get("metadatas")
        distances = results.get("distances")

        for i, eid in enumerate(ids_list[0]):
            doc = documents[0][i] if documents and documents[0] else None
            meta = metadatas[0][i] if metadatas and metadatas[0] else {}

            semantic_score: float | None = None
            if distances and distances[0]:
                # ChromaDB cosine distance ∈ [0, 2]; similarity = 1 - distance.
                semantic_score = round(max(0.0, 1.0 - float(distances[0][i])), 4)

            entries.append(
                self._chroma_meta_to_entry(eid, doc, meta, semantic_score)
            )

        return entries

    # ── Metadata conversion ───────────────────────────────────────────────

    @staticmethod
    def _entry_to_chroma_meta(entry: MemoryEntry) -> dict[str, Any]:
        """Convert a ``MemoryEntry`` to a ChromaDB-safe metadata dict."""
        return {
            "created_at": entry.created_at.timestamp(),
            "updated_at": entry.updated_at.timestamp(),
            "weight": entry.weight,
            "category": entry.category,
            "tags_json": json.dumps(entry.tags),
            "ext_metadata_json": json.dumps(entry.metadata),
            # 记忆分区：顶级字段便于 where 过滤；旧条目没有该字段，视为 web。
            "scope": (entry.metadata or {}).get("scope", "web"),
        }

    @staticmethod
    def _chroma_meta_to_entry(
        entry_id: str,
        document: str | None,
        meta: dict[str, Any],
        semantic_score: float | None = None,
    ) -> MemoryEntry:
        """Reconstruct a ``MemoryEntry`` from ChromaDB metadata."""
        # Late import to avoid circular dependency at module level
        from memory_skill.contracts import MemoryEntry

        def _safe_json_load(raw: Any, default: Any) -> Any:
            """Parse JSON string, falling back to default on failure."""
            if isinstance(raw, str):
                try:
                    return json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    return default
            return default

        created_at = datetime.fromtimestamp(
            meta.get("created_at", 0.0), tz=UTC
        )
        updated_at = datetime.fromtimestamp(
            meta.get("updated_at", 0.0), tz=UTC
        )

        ext_meta = _safe_json_load(meta.get("ext_metadata_json"), {})
        if "title" in meta:
            ext_meta["title"] = meta["title"]
        return MemoryEntry(
            id=entry_id,
            content=secure.decrypt_or_placeholder(document or "")[0],
            created_at=created_at,
            updated_at=updated_at,
            weight=float(meta.get("weight", 0.0)),
            category=str(meta.get("category", "default")),
            tags=_safe_json_load(meta.get("tags_json"), []),
            metadata=ext_meta,
            semantic_score=semantic_score,
        )
