"""Capability Registry — maps tree branches to answerable domains.

Each tree branch is treated as a *capability domain*.  The registry
answers one question: "can the system answer this query?"

Uses the existing retriever + tree to estimate answerability without
calling an LLM.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime

logger = logging.getLogger("memory_skill.capability")


# ── Answerability thresholds (calibrated 2026-08-02 on bge-large-en-v1.5) ──
# Real-store calibration (KNOWN-ISSUES #1): the English BGE model is noisy on
# Chinese — unrelated hits reach cosine 0.62-0.78, overlapping the 0.74-0.89
# range of genuinely-relevant hits.  A pure semantic threshold therefore
# cannot separate "knows" from "coincidental hit".  We pair the semantic
# score with query↔content token overlap; see the KNOWN-ISSUES entry for the
# full calibration table.

# Semantic score alone is sufficient above this (overwhelmingly strong match).
_SEM_STRONG = 0.85
# Semantic score + corroborating token overlap is sufficient above this.
_SEM_CORROBORATED = 0.72
# Uncorroborated hits are damped: confidence reflects "weak evidence", so
# gap detection treats them as gaps instead of answered.
_UNCORROBORATED_DAMP = 0.5
# Max semantic candidates considered by can_answer.
_CAN_ANSWER_TOP_K = 5

# CJK bigrams that carry no topical signal — pure question/function particles.
# Without this filter a garbage query like "量子场论 是什么" would be
# "corroborated" by the shared bigram "什么".
_CN_STOP_BIGRAMS = frozenset({
    "什么", "怎么", "如何", "哪个", "哪些", "为何", "多少", "多久",
    "是不是", "有没有", "可不可", "是否", "能否", "可否",
    "我们", "你们", "他们", "大家", "自己", "本人",
    "这个", "那个", "一个", "一种", "一下", "一些", "一点", "这么",
    "因为", "所以", "但是", "然后", "如果", "就是", "还是", "或者",
    "知道", "觉得", "想要", "应该", "可以", "可能", "需要",
    "今天", "明天", "昨天", "时候", "问题", "东西",
})


def _query_tokens(text: str) -> frozenset[str]:
    """Extract distinctive tokens: ASCII words + non-stop CJK bigrams."""
    toks: set[str] = set()
    for word in re.findall(r"[A-Za-z][A-Za-z0-9_]*", text):
        if len(word) >= 2:
            toks.add(word.lower())
    for run in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        for i in range(len(run) - 1):
            bigram = run[i:i + 2]
            if bigram not in _CN_STOP_BIGRAMS:
                toks.add(bigram)
    return frozenset(toks)


def _token_overlap(query: str, content: str) -> bool:
    """True when query and content share at least one distinctive token."""
    q_tokens = _query_tokens(query)
    if not q_tokens:
        return False
    return bool(q_tokens & _query_tokens(content))


def token_overlap(query: str, content: str) -> bool:
    """Public entry point for the distinctive-token corroboration check.

    Modules that need the KNOWN-ISSUES #1 double-corroboration (weaver's
    historic hint, skill registry's verdict) call this instead of importing
    the underscore-private helper at runtime. Semantics identical to
    ``_token_overlap``; kept separate so the private name can stay for
    internal callers while the seam is public.
    """
    return _token_overlap(query, content)


@dataclass(frozen=True)
class Capability:
    """A capability domain — one tree branch."""
    branch_id: str
    label: str
    root: str
    entry_count: int
    date_count: int
    avg_weight: float
    last_updated: datetime | None


class CapabilityRegistry:
    """Answers whether the system can handle a given query.

    Parameters
    ----------
    tree: TreeManager
        The memory tree (branch labels, counts).
    retriever: Retriever
        The hybrid retriever for scoring how well we know a topic.
    """

    def __init__(self, tree, retriever):
        self._tree = tree
        self._retriever = retriever

    def list_capabilities(self) -> list[Capability]:
        """Return all capability domains with health stats."""
        from memory_skill.tree import _BASE_BRANCHES, _BRANCH_LABEL_MAP

        caps: list[Capability] = []
        branch_ids = [b["id"] for b in _BASE_BRANCHES]

        for bid in branch_ids:
            label = _BRANCH_LABEL_MAP.get(bid, bid)
            root = bid.split("_")[0]
            count, date_count = self._branch_counts(bid)
            avg_w = self._branch_avg_weight(bid)
            last_up = self._branch_last_updated(bid)

            caps.append(Capability(
                branch_id=bid,
                label=label,
                root=root,
                entry_count=count,
                date_count=date_count,
                avg_weight=avg_w,
                last_updated=last_up,
            ))

        return caps

    def classify_query(self, query: str) -> dict[str, str]:
        """Return ``{"root": ..., "branch": ...}`` for a query using TreeManager."""
        try:
            return self._tree.classify(query)
        except Exception:
            return {"root": "user", "branch": "mem"}

    def can_answer(self, query: str) -> tuple[bool, float]:
        """Return ``(can_answer, confidence)`` for a query.

        Confidence ranges from 0 (no knowledge) to 1 (highly confident).

        Relevance comes from the top semantic hit's cosine score, corroborated
        by query↔content token overlap.  Uncorroborated hits are damped so
        gap detection fires on coincidental matches (KNOWN-ISSUES #1).
        """
        if not query or not query.strip():
            return False, 0.0

        top, sem = self._retriever.best_semantic_match(query, limit=_CAN_ANSWER_TOP_K)

        if top is None or sem <= 0.0:
            return False, 0.0

        if sem >= _SEM_STRONG:
            return True, round(sem, 3)
        if sem >= _SEM_CORROBORATED and _token_overlap(query, top.content):
            return True, round(sem, 3)

        return False, round(sem * _UNCORROBORATED_DAMP, 3)

    def get_confidence(self, branch: str) -> float:
        """Overall health of a capability domain (0-1)."""
        count, _ = self._branch_counts(branch)
        avg_w = self._branch_avg_weight(branch)
        if count == 0:
            return 0.0
        # Scale: more entries + higher avg weight → higher confidence
        entry_score = min(count / 20.0, 1.0)
        return round(entry_score * avg_w, 3)

    # ── Helpers ─────────────────────────────────────────────────────────

    def _branch_counts(self, branch_id: str) -> tuple[int, int]:
        try:
            return self._tree.branch_counts(branch_id)
        except Exception:
            return 0, 0

    def _branch_avg_weight(self, branch_id: str) -> float:
        try:
            return self._tree.branch_avg_weight(branch_id)
        except Exception:
            return 0.5

    def _branch_last_updated(self, branch_id: str) -> datetime | None:
        try:
            ts = self._tree.branch_last_updated(branch_id)
            return datetime.fromisoformat(ts) if ts else None
        except Exception:
            return None
