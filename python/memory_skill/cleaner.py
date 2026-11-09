"""
Memory Skill — Memory Cleaner (Phase 6.3).

Periodic background cleanup that prevents memory bloat without losing
critical knowledge.

Operations (safe, soft-delete only)::

    1. Merge near-duplicate entries (cosine similarity > 0.95).
    2. Mark stale entries (30+ days unaccessed AND low weight).
    3. NEVER touch entries with weight > 0.7 or category="preference".

Usage::

    from memory_skill.cleaner import MemoryCleaner

    cleaner = MemoryCleaner(skill)
    report = cleaner.run()
    # → {"merged": 3, "stale_marked": 12, "protected": 5}
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from memory_skill.contracts import utcnow


logger = logging.getLogger(__name__)

# ── Thresholds ─────────────────────────────────────────────────────────────────

# Cosine similarity above which two entries are considered duplicates.
_MERGE_SIMILARITY_THRESHOLD: float = 0.95

# Days without access after which an entry is considered stale.
# Weighted: high-weight entries get longer TTL.
_STALE_DAYS_BASE: int = 14
_STALE_DAYS_MAX: int = 90

def _stale_cutoff(weight: float) -> datetime:
    """Compute staleness cutoff — higher weight → longer TTL."""
    days = _STALE_DAYS_BASE + (_STALE_DAYS_MAX - _STALE_DAYS_BASE) * weight
    return utcnow() - timedelta(days=int(days))

# Weight above which entries are immune to cleanup.
_PROTECTED_WEIGHT: float = 0.7

# Categories that are always protected.
_PROTECTED_CATEGORIES: frozenset[str] = frozenset({"preference", "observation"})


# ═══════════════════════════════════════════════════════════════════════════════
# MemoryCleaner
# ═══════════════════════════════════════════════════════════════════════════════


class MemoryCleaner:
    """Safe memory cleanup — never hard-deletes, only soft-deletes.

    Parameters
    ----------
    skill:
        The ``MemorySkill`` instance providing access to stores.
    """

    def __init__(self, learned_store, embedder, config) -> None:
        self._store = learned_store
        self._embedder = embedder
        self._config = config

    def run(self) -> dict[str, int]:
        """Run one cleanup cycle.

        Returns a report dict with counts for each operation.
        """
        report: dict[str, int] = {
            "merged": 0,
            "stale_marked": 0,
            "protected": 0,
        }

        store = self._store

        # ── 1. Get all entries ─────────────────────────────────────────
        try:
            entries = store.search("", limit=1000)
        except Exception as e:
            logger.debug("Cleaner phase failed: %s", e)
            return report

        if not entries:
            return report

        # ── 2. Classify: protected vs. candidates ──────────────────────
        protected: set[str] = set()
        candidates: list = []

        for e in entries:
            if (
                e.weight >= _PROTECTED_WEIGHT
                or e.category in _PROTECTED_CATEGORIES
                or any("preference" in t.lower() for t in e.tags)
            ):
                protected.add(e.id)
            else:
                candidates.append(e)

        report["protected"] = len(protected)

        # ── 3. Merge near-duplicates ───────────────────────────────────
        merged_ids = self._merge_duplicates(store, candidates)
        report["merged"] = len(merged_ids)

        # ── 4. Mark stale entries (weighted: high weight → longer TTL) ──
        stale_marked = 0
        for e in candidates:
            if e.id in merged_ids or e.id in protected:
                continue
            cutoff = _stale_cutoff(e.weight)
            if e.updated_at < cutoff and e.weight < _PROTECTED_WEIGHT:
                try:
                    store.update(
                        e.id,
                        e.content,
                        metadata={
                            **(e.metadata or {}),
                            "stale": True,
                            "stale_since": utcnow().isoformat(),
                        },
                    )
                    stale_marked += 1
                except Exception as e:
                    logger.debug("Cleaner phase failed: %s", e)
                    logger.debug("Failed to mark stale entry %s", e.id)

        report["stale_marked"] = stale_marked

        return report

    # ── Duplicate detection ────────────────────────────────────────────────

    def _merge_duplicates(self, store, candidates: list) -> set[str]:
        """Find and merge near-duplicate entries.

        Uses the embedder to compare cosine similarity between pairs of
        entries.  When duplicates are found, the older entry is soft-deleted
        by adding a ``merged_into`` metadata tag.
        """
        merged: set[str] = set()

        # Build embedding cache (batched for performance)
        emb_cache: dict[str, list[float]] = {}
        if not candidates:
            return merged

        # Cap pairwise comparison — O(n²) gets expensive above 200
        _MAX_CANDIDATES: int = 200
        if len(candidates) > _MAX_CANDIDATES:
            # Keep most recently updated entries
            candidates.sort(key=lambda e: e.updated_at, reverse=True)
            candidates = candidates[:_MAX_CANDIDATES]
            logger.debug("Cleaner: capped candidates to %d", _MAX_CANDIDATES)

        try:
            texts = [e.content for e in candidates]
            vecs = self._embedder.embed_batch(texts)
            for e, vec in zip(candidates, vecs):
                emb_cache[e.id] = vec
        except Exception as e:
            logger.debug("Cleaner embed failed: %s", e)
            return merged

        # Compare pairs within same category
        by_category: dict[str, list] = {}
        for e in candidates:
            by_category.setdefault(e.category, []).append(e)

        for cat, group in by_category.items():
            for i in range(len(group)):
                if group[i].id in merged:
                    continue
                for j in range(i + 1, len(group)):
                    if group[j].id in merged:
                        continue
                    sim = _cosine_similarity(
                        emb_cache.get(group[i].id, []),
                        emb_cache.get(group[j].id, []),
                    )
                    if sim >= _MERGE_SIMILARITY_THRESHOLD:
                        # Keep the newer entry, soft-delete the older
                        newer, older = (
                            (group[i], group[j])
                            if group[i].updated_at >= group[j].updated_at
                            else (group[j], group[i])
                        )
                        try:
                            store.update(
                                older.id,
                                older.content,
                                metadata={
                                    **(older.metadata or {}),
                                    "merged_into": newer.id,
                                    "merged_at": utcnow().isoformat(),
                                },
                            )
                            merged.add(older.id)
                            logger.debug(
                                "Merged duplicate: %s → %s (sim=%.3f)",
                                older.id, newer.id, sim,
                            )
                        except Exception as e:
                            logger.debug("Cleaner phase failed: %s", e)
                            pass

        return merged


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)
