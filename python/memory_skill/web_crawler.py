"""
Memory Skill — Web Crawler
===========================

Fetches URLs via curl, extracts clean text, chunks long pages, and
ingests into MemorySkill as web-sourced knowledge entries.

Requires ``curl`` on the system PATH. No Python package dependencies.

Usage::

    from memory_skill.web_crawler import WebCrawler
    crawler = WebCrawler()
    chunks = crawler.crawl("https://example.com/article")
    crawler.ingest(skill, chunks)
"""

from __future__ import annotations

import html as _html_lib
import logging
import re
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime

logger = logging.getLogger("memory_skill.crawler")

_CHUNK_MAX_CHARS = 2000
_FETCH_TIMEOUT = 30
_TEXT_CONTENT_TYPES = frozenset({
    "text/html", "text/plain", "text/markdown",
    "application/xhtml+xml", "application/xml",
})


@dataclass(frozen=True)
class CrawledChunk:
    """A chunk of web content ready for ingestion.

    Attributes
    ----------
    source_url: The URL this chunk was extracted from.
    text: Cleaned plain-text content (≤2000 chars).
    index: 0-based chunk index within the page.
    title: Page <title> if found, else empty string.
    crawled_at: UTC timestamp of the crawl.
    """
    source_url: str
    text: str
    index: int
    title: str = ""
    crawled_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class WebCrawler:
    """Fetches web pages, extracts clean text, and chunks for memory ingestion."""

    def __init__(self, timeout: int = _FETCH_TIMEOUT):
        self._timeout = timeout

    # ── Public API ───────────────────────────────────────────────────────

    def crawl(self, url: str) -> list[CrawledChunk]:
        """Fetch *url*, extract clean text, and return chunks.

        Returns an empty list on any fetch or parse error.
        """
        try:
            html_text, content_type = self._fetch(url)
        except Exception as exc:
            logger.warning("Fetch failed for %s: %s", url, exc)
            return []

        if not self._is_text(content_type):
            logger.info("Skipping non-text content: %s (%s)", url, content_type)
            return []

        try:
            title, body = self._extract(html_text)
        except Exception as exc:
            logger.warning("Parse failed for %s: %s", url, exc)
            return []

        return self._chunk(title, body, url)

    def ingest(
        self,
        skill,          # MemorySkill (lazy import)
        chunks: list[CrawledChunk],
        partner: str | None = None,
    ) -> int:
        """Ingest crawled chunks into *skill* as DialogueTurn entries.

        Returns the number of chunks ingested.
        """
        from memory_skill.contracts import DialogueTurn

        count = 0
        for chunk in chunks:
            try:
                turn = DialogueTurn(
                    id=f"web_{hash(chunk.source_url) & 0xFFFF:04x}_{chunk.index:03d}",
                    role="system",
                    content=f"[来源: {chunk.source_url}]\n{chunk.text}",
                    timestamp=chunk.crawled_at,
                    partner=partner,
                )
                skill.ingest(turn)
                count += 1
            except Exception:
                logger.exception("Ingest failed for chunk %d of %s", chunk.index, chunk.source_url)

        return count

    # ── Internals ────────────────────────────────────────────────────────

    def _fetch(self, url: str) -> tuple[str, str]:
        """Return (response_body, content_type) using curl."""
        try:
            result = subprocess.run(
                [
                    "curl", "-sS", "-L",
                    "--max-time", str(self._timeout),
                    "-H", (
                        "User-Agent: MemorySkill-Crawler/0.5 "
                        "(knowledge bot; low-rate)"
                    ),
                    "-H", "Accept: text/html, text/plain, application/xhtml+xml",
                    "-w", "\n%{content_type}",
                    url,
                ],
                capture_output=True,
                text=False,
                timeout=self._timeout + 5,
            )
        except subprocess.TimeoutExpired:
            raise Exception(f"Fetch timed out after {self._timeout}s") from None

        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace").strip()
            raise Exception(stderr or f"curl exited {result.returncode}")

        output = result.stdout
        if b"\n" in output:
            body, ct_line = output.rsplit(b"\n", 1)
            content_type = ct_line.decode(errors="replace").strip()
        else:
            body = output
            content_type = "text/html"

        ct = content_type.split(";")[0].strip().lower() or "text/html"
        return body.decode(errors="replace"), ct

    @staticmethod
    def _is_text(content_type: str) -> bool:
        return content_type.lower() in _TEXT_CONTENT_TYPES

    # ── Text extraction ──────────────────────────────────────────────────

    _RE_SCRIPT_STYLE = re.compile(
        r"<(script|style|noscript|iframe|svg|nav|header|footer|aside)[^>]*>.*?</\1>",
        flags=re.DOTALL | re.IGNORECASE,
    )
    _RE_TAG = re.compile(r"<[^>]+>")
    _RE_WHITESPACE = re.compile(r"[ \t]+")
    _RE_BLANK_LINES = re.compile(r"\n{3,}")

    def _extract(self, html_text: str) -> tuple[str, str]:
        """Return (title, body_text) from raw HTML."""
        # Extract <title>
        title = ""
        m = re.search(r"<title[^>]*>(.*?)</title>", html_text, re.DOTALL | re.IGNORECASE)
        if m:
            title = self._clean_text(m.group(1))

        # Strip scripts, styles, and HTML tags
        text = self._RE_SCRIPT_STYLE.sub("", html_text)
        text = self._RE_TAG.sub("\n", text)
        text = _html_lib.unescape(text)

        # Clean whitespace
        text = self._RE_WHITESPACE.sub(" ", text)
        text = self._RE_BLANK_LINES.sub("\n\n", text)

        # Remove very short lines (likely navigation / boilerplate)
        lines = [line.strip() for line in text.split("\n")]
        lines = [line for line in lines if len(line) > 10 or line == ""]
        body = "\n".join(lines).strip()

        return title, body

    @staticmethod
    def _clean_text(text: str) -> str:
        """Strip tags and entities from a short piece of HTML text."""
        text = re.sub(r"<[^>]+>", "", text)
        return _html_lib.unescape(text).strip()

    # ── Chunking ─────────────────────────────────────────────────────────

    def _chunk(self, title: str, body: str, url: str) -> list[CrawledChunk]:
        """Split *body* into ≤2000-char chunks at paragraph boundaries."""
        if not body:
            return []

        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
        chunks: list[CrawledChunk] = []
        buf: list[str] = []
        buf_len = 0

        def _flush() -> None:
            nonlocal buf, buf_len
            if buf:
                text = "\n\n".join(buf)
                chunks.append(CrawledChunk(
                    source_url=url,
                    text=text,
                    index=len(chunks),
                    title=title,
                ))
                buf = []
                buf_len = 0

        for para in paragraphs:
            if buf_len + len(para) + 2 > _CHUNK_MAX_CHARS and buf:
                _flush()
            buf.append(para)
            buf_len += len(para) + 2  # +2 for "\n\n" separator

            # Single paragraph that exceeds limit → split mid-sentence
            while buf_len > _CHUNK_MAX_CHARS:
                # Move to overflow buffer
                overflow = []
                while buf and buf_len > _CHUNK_MAX_CHARS:
                    last = buf.pop()
                    buf_len -= len(last) + 2
                    overflow.insert(0, last)
                _flush()
                buf = overflow
                buf_len = sum(len(p) + 2 for p in buf)

        _flush()

        # Prepend title to first chunk if present
        if title and chunks:
            chunks[0] = CrawledChunk(
                source_url=url,
                text=f"[{title}]\n\n{chunks[0].text}",
                index=0,
                title=title,
            )

        return chunks
