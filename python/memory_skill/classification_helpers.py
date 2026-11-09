"""Domain heuristics extracted from ingestor.py.

These are pure logic functions — question detection and screen noise
filtering.  They live separately from the storage layer so the
Ingestor class can focus on persistence.
"""

from __future__ import annotations

import re

import numpy as np

_QUESTION_KW = ("?", "吗", "呢", "什么", "怎么", "如何", "哪", "谁", "多少", "在哪")


def _looks_like_question(content: str) -> bool:
    s = content.strip()
    return "?" in s or s.endswith(("吗", "呢")) or any(kw in s for kw in _QUESTION_KW[3:])


class _ScreenNoiseFilter:
    _ERROR_PATTERN = re.compile(
        r"Error|Exception|FATAL|Traceback|TypeError|Connection refused",
        re.IGNORECASE,
    )

    def __init__(self, threshold: float = 0.85, ngram: int = 3):
        self._threshold = threshold
        self._ngram = ngram

    def is_error_frame(self, text: str) -> bool:
        return bool(self._ERROR_PATTERN.search(text))

    def should_keep(self, current: str, last: str) -> bool:
        if self._ERROR_PATTERN.search(current):
            return True
        if current == last:
            return False
        if not current and last:
            return True
        if current and not last:
            return True
        if not current and not last:
            return False
        return self._cosine(current, last) < self._threshold

    def _cosine(self, a: str, b: str) -> float:
        na = self._ngrams(a)
        nb = self._ngrams(b)
        if not na or not nb:
            return 0.0
        vocab = sorted(set(na) | set(nb))
        idx = {ng: i for i, ng in enumerate(vocab)}
        va = np.zeros(len(vocab), dtype=np.float64)
        vb = np.zeros(len(vocab), dtype=np.float64)
        for ng in na:
            va[idx[ng]] += 1.0
        for ng in nb:
            vb[idx[ng]] += 1.0
        dot = float(np.dot(va, vb))
        na_norm, nb_norm = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
        return dot / (na_norm * nb_norm) if na_norm and nb_norm else 0.0

    def _ngrams(self, text: str) -> list[str]:
        t = text.lower()
        n = self._ngram
        return [t[i:i + n] for i in range(len(t) - n + 1)] if len(t) >= n else []
