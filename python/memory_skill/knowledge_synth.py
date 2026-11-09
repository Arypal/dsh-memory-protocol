"""Knowledge Synthesis — multi-source cross-validation + LLM refinement.

Takes chunks crawled from multiple URLs, extracts facts, cross-validates
them across sources, and produces structured ``SynthesizedEntry`` objects
ready for memory ingestion.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

logger = logging.getLogger("memory_skill.synth")

_MAX_TOKENS = 1024
_TEMPERATURE = 0.1
_TIMEOUT = 30

# ── Prompt ────────────────────────────────────────────────────────────────────

_SYNTH_PROMPT = """\
Topic: "{topic}"

Analyze the following sources and produce a structured knowledge summary.

{source_blocks}

Return a JSON array of fact objects. Each object must have these keys:
- "fact": the knowledge statement (concise, 1-3 sentences, use Chinese)
- "source_urls": list of source IDs that support this fact
- "confidence": number 0.0-1.0 (1.0 = all sources agree, 0.5 = single source only)
- "conflict_with": list of fact indices this contradicts, or [] if none

Rules:
- Drop duplicate facts — keep only the most complete version
- Facts confirmed by ≥2 sources → confidence ≥ 0.8
- Facts from single source → confidence 0.5
- Conflicting facts → mark conflict_with and keep both
- Irrelevant or trivial content → omit completely
- At most 10 facts total
- Return JSON only. No markdown, no explanations.
"""


# ── Data ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SynthesizedEntry:
    content: str
    source_urls: list[str]
    confidence: float
    verified: bool
    has_conflict: bool
    conflict_with: list[str] = field(default_factory=list)
    branch_hint: str = ""
    synthesized_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class SynthResult:
    entries: list[SynthesizedEntry]
    total_facts: int
    verified_count: int
    conflict_count: int
    overall_confidence: float


# ── Engine ────────────────────────────────────────────────────────────────────


_MARKDOWN_PROMPT = """\
Topic: "{topic}"

Synthesize the following sources into a well-structured Markdown document.

{source_blocks}

Requirements:
- Title: # topic name as a heading
- Organize content with ## subsections
- Use bullet lists for key facts
- Keep code samples in ``` blocks
- Include a ## 来源 section listing source URLs
- Be concise: focus on actionable knowledge, skip fluff
- Use Chinese for explanations, keep code in English
- Return Markdown only, no JSON wrapper
"""


class KnowledgeSynth:
    """Synthesize multi-source crawled content into structured knowledge."""

    def __init__(self, api_base: str, api_key: str, model: str):
        self._api_base = api_base
        self._api_key = api_key
        self._model = model

    def synthesize(
        self,
        topic: str,
        chunks_by_url: dict[str, list],
    ) -> SynthResult:
        """Synthesize crawled chunks from multiple URLs.

        Parameters
        ----------
        topic:
            The topic being learned.
        chunks_by_url:
            Mapping of ``url → list[CrawledChunk]`` from the web crawler.

        Returns
        -------
        SynthResult
            Structured facts with cross-validation markers.
        """
        if not chunks_by_url:
            return SynthResult(
                entries=[], total_facts=0, verified_count=0,
                conflict_count=0, overall_confidence=0.0,
            )

        # Build source blocks
        blocks: list[str] = []
        for i, (url, chunks) in enumerate(chunks_by_url.items()):
            combined = "\n\n".join(c.text for c in chunks)
            truncated = combined[:3000]  # keep prompt manageable
            blocks.append(f"[Source {i} — {url}]\n{truncated}")

        prompt = _SYNTH_PROMPT.format(
            topic=topic,
            source_blocks="\n\n---\n\n".join(blocks),
        )

        try:
            raw_facts = self._call_llm(prompt)
        except Exception as exc:
            logger.warning("Synth LLM failed: %s", exc)
            return SynthResult(
                entries=[], total_facts=0, verified_count=0,
                conflict_count=0, overall_confidence=0.0,
            )

        entries: list[SynthesizedEntry] = []
        for fact in raw_facts:
            e = SynthesizedEntry(
                content=fact.get("fact", ""),
                source_urls=fact.get("source_urls", []),
                confidence=float(fact.get("confidence", 0.5)),
                verified=float(fact.get("confidence", 0.5)) >= 0.8,
                has_conflict=bool(fact.get("conflict_with", [])),
                conflict_with=[str(x) for x in fact.get("conflict_with", [])],
            )
            entries.append(e)

        verified = sum(1 for e in entries if e.verified)
        conflicts = sum(1 for e in entries if e.has_conflict)
        overall = (
            sum(e.confidence for e in entries) / len(entries)
            if entries else 0.0
        )

        return SynthResult(
            entries=entries,
            total_facts=len(entries),
            verified_count=verified,
            conflict_count=conflicts,
            overall_confidence=round(overall, 3),
        )

    def synthesize_markdown(self, topic: str,
                            chunks_by_url: dict[str, list]) -> str:
        """Synthesize multi-source content into a markdown document."""
        if not chunks_by_url:
            return ""
        blocks = []
        for i, (url, chunks) in enumerate(chunks_by_url.items()):
            combined = "\n\n".join(c.text for c in chunks)
            blocks.append(f"[Source {i} — {url}]\n{combined[:3000]}")
        prompt = _MARKDOWN_PROMPT.format(
            topic=topic,
            source_blocks="\n\n---\n\n".join(blocks),
        )
        try:
            from memory_skill._llm_utils import call_llm
            md = call_llm(
                self._api_base, self._api_key, self._model,
                prompt, max_tokens=2048, temperature=0.1,
            )
            return md or ""
        except Exception:
            return ""

    # ── LLM ──────────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> list[dict]:
        from memory_skill._llm_utils import call_llm, parse_json_response
        raw = call_llm(
            self._api_base, self._api_key, self._model,
            prompt, _MAX_TOKENS, _TEMPERATURE,
        )
        return _parse_json(raw) if raw else []

    @staticmethod
    def _parse_json(raw: str) -> list[dict]:
        from memory_skill._llm_utils import parse_json_response
        parsed = parse_json_response(raw)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and "facts" in parsed:
            return parsed["facts"]
        return []
        return []
