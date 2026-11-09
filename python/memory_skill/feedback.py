"""
Memory Skill — Feedback auto-detection (V5).

Automatically detects positive/negative/neutral outcome signals from
conversation results without requiring the agent to explicitly label
every retrieval.  Uses keyword-overlap heuristics (rule path) between
the final response and search result contents.

Usage::

    from memory_skill.feedback import auto_detect_outcome

    outcome = auto_detect_outcome(
        query="What does the user prefer?",
        search_results=[
            {"id": "mem_001", "content": "User prefers dark mode"},
            {"id": "mem_002", "content": "User uses Python 3.13"},
        ],
        final_response="Based on previous conversations, the user prefers dark mode.",
    )
    # → "positive"
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_VALID_OUTCOMES: frozenset[str] = frozenset({"positive", "negative", "neutral"})

# Rule-path thresholds
_OVERLAP_POSITIVE_THRESHOLD: float = 0.3
_OVERLAP_NEGATIVE_THRESHOLD: float = 0.0

# ── Public API ────────────────────────────────────────────────────────────────


def auto_detect_outcome(
    query: str,
    search_results: list[dict[str, Any]],
    final_response: str,
) -> str:
    """Auto-detect whether a memory retrieval was helpful.

    Uses keyword-overlap heuristics to classify the outcome.

    Parameters
    ----------
    query:
        The original user query that triggered the search.
    search_results:
        List of search result dicts, each with ``"id"`` and ``"content"`` keys.
    final_response:
        The agent's final response that used (or ignored) the search results.

    Returns
    -------
    str
        One of ``"positive"``, ``"negative"``, or ``"neutral"``.
    """
    return _rule_detect_outcome(search_results, final_response)


# ── Rule path ─────────────────────────────────────────────────────────────────


def _rule_detect_outcome(
    search_results: list[dict[str, Any]],
    final_response: str,
) -> str:
    """Detect outcome via keyword overlap heuristics.

    Algorithm:
    1. Tokenize ``final_response`` into a set of lowercase words.
    2. For each search result, compute overlap ratio:
       ``|resp_words ∩ content_words| / |content_words|``
    3. If max overlap > 30% → ``"positive"``
    4. If max overlap = 0% AND final_response non-empty → ``"negative"``
    5. Otherwise → ``"neutral"``
    """

    # ── Edge cases ──────────────────────────────────────────────────────
    if not final_response or not search_results:
        return "neutral"

    resp_words = set(_tokenize(final_response))
    if not resp_words:
        return "neutral"

    # ── Compute max overlap across all search results ────────────────────
    max_overlap: float = 0.0
    any_content: bool = False

    for result in search_results:
        content = str(result.get("content", ""))
        if not content:
            continue
        any_content = True

        content_words = set(_tokenize(content))
        if not content_words:
            continue

        intersection = len(resp_words & content_words)
        overlap = intersection / len(content_words)
        if overlap > max_overlap:
            max_overlap = overlap

    if not any_content:
        return "neutral"

    # ── Classify ────────────────────────────────────────────────────────
    if max_overlap > _OVERLAP_POSITIVE_THRESHOLD:
        return "positive"
    elif max_overlap <= _OVERLAP_NEGATIVE_THRESHOLD:
        return "negative"
    else:
        return "neutral"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer → lowercase words, ignoring stop words and short tokens."""
    _stop_words: frozenset[str] = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "in", "on", "at", "to", "for", "of", "with", "from", "by",
        "and", "or", "but", "not", "no", "if", "so", "as", "it",
        "this", "that", "these", "those", "has", "have", "had",
        "do", "does", "did", "will", "would", "can", "could",
        "i", "you", "he", "she", "we", "they", "me", "him", "her",
        "us", "them", "my", "your", "his", "our", "their", "its",
    })
    return [
        w.lower().strip(".,!?;:\"'()[]{}")
        for w in text.split()
        if len(w) > 1 and w.lower() not in _stop_words
    ]
