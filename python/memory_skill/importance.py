"""
Memory Skill — Importance Scorer (Phase 6.1).

Rule-based filter that prevents trivial messages from bloating long-term
storage.  Messages with low importance scores are stored only in the
SawRingBuffer (short-term, volatile) and skipped by LearnedStore and
DialogueStore.

Algorithm::

    score = _base_score(content)
    score = _boost_keywords(content, score)
    score = _penalise_trivial(content, score)
    → clamp to [0.0, 1.0]

Default threshold: score < 0.3 → short-term only.

Usage::

    from memory_skill.importance import ImportanceScorer

    scorer = ImportanceScorer(threshold=0.3)
    importance, persist = scorer.evaluate("ok")
    # → (0.05, False) — too trivial, skip persistence
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ── Threshold ──────────────────────────────────────────────────────────────────

_DEFAULT_THRESHOLD: float = 0.3
_DEFAULT_IMPORTANCE: float = 0.5

# ── Signal patterns ────────────────────────────────────────────────────────────

# Words/phrases indicating the content is important (cumulative boost).
_HIGH_SIGNAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"记住", re.IGNORECASE),
    re.compile(r"重要", re.IGNORECASE),
    re.compile(r"别忘了", re.IGNORECASE),
    re.compile(r"切记", re.IGNORECASE),
    re.compile(r"下次|以后|以后要|之后要", re.IGNORECASE),
    re.compile(r"面试|deadline|截止|紧急", re.IGNORECASE),
]

# Words indicating the user is stating a preference or fact.
_PREFERENCE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"我(喜欢|讨厌|偏好|习惯|经常|从来不|总是|一般)"),
    re.compile(r"我的.*是"),
]

_SINGLE_BOOST: float = 0.1  # per high-signal match (capped at 2 matches)

# ── Trivial detection ──────────────────────────────────────────────────────────

_TRIVIAL_MAX_CHARS: int = 4  # messages ≤ 4 chars are suspicious

_TRIVIAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^(ok|okay|yeah|yes|no|nope|sure|right|fine|got it|thanks|thx|ty)$", re.IGNORECASE),
    re.compile(r"^(em+|呵呵|哈哈|。。+)$", re.IGNORECASE),
]


# ═══════════════════════════════════════════════════════════════════════════════
# ImportanceScorer
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ImportanceScorer:
    """Rule-based importance evaluator.

    Parameters
    ----------
    threshold:
        Messages with score below this value are considered too trivial
        for long-term storage (SawRingBuffer only).  Default: 0.3.
    """

    threshold: float = _DEFAULT_THRESHOLD

    def evaluate(self, content: str) -> tuple[float, bool]:
        """Evaluate content importance.

        Returns
        -------
        (score, persist)
            *score* is in [0.0, 1.0].
            *persist* is ``True`` when score >= threshold.
        """
        if not content or not content.strip():
            return (0.0, False)

        text = content.strip()
        score: float = _DEFAULT_IMPORTANCE

        # ── Boost for high-signal keywords ──────────────────────────────
        high_hits = 0
        for pat in _HIGH_SIGNAL_PATTERNS:
            if pat.search(text):
                high_hits += 1
        score += min(high_hits, 2) * _SINGLE_BOOST

        # ── Boost for preference statements ─────────────────────────────
        pref_hits = sum(1 for p in _PREFERENCE_PATTERNS if p.search(text))
        score += pref_hits * 0.15

        # ── Penalise: very short or low-diversity messages ──────────────
        text = content.strip()

        # Short messages: score inversely proportional to length
        if len(text) <= _TRIVIAL_MAX_CHARS:
            score -= 0.3 + 0.05 * (_TRIVIAL_MAX_CHARS - len(text))

        # Low character diversity: mostly repeated chars or single type
        unique_ratio = len(set(text)) / max(len(text), 1)
        if unique_ratio < 0.4 and len(text) <= 10:
            score -= 0.3

        # Pattern match for known trivial forms (interjections, laughs)
        if any(p.fullmatch(text) for p in _TRIVIAL_PATTERNS):
            score = 0.05

        # ── Clamp ───────────────────────────────────────────────────────
        score = max(0.0, min(1.0, score))

        return (score, score >= self.threshold)
